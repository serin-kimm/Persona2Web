import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import pickle
import gc
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from openai import OpenAI
from AgentOccam.llms.qwen import call_gpt as call_qwen
from pathlib import Path
from AgentOccam.llms.gemini import call_gemini
from AgentOccam.llms.llama import call_llama

@dataclass
class MemoryItem:
    """Memory item data class"""
    timestamp: str
    type: str
    object: str
    website: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "object": self.object,
            "website": self.website
        }

class MemoryBankRetriever:
    """Memory bank retriever class"""

    def __init__(
        self,
        model_name: str = "dunzhang/stella_en_1.5B_v5",
        memory_bank_path: str = None,
        llm_model: str = "gpt-5",
        cache_dir: str = "./cache",
        force_rebuild: bool = False
    ):
        """
        Args:
            model_name: Embedding model name
            memory_bank_path: Memory bank JSON file path (required)
            llm_model: LLM model name for answer generation
            cache_dir: Index cache storage directory
            force_rebuild: If True, ignore cache and rebuild index from scratch
        """
        if not memory_bank_path:
            raise ValueError("memory_bank_path is required")
            
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_bank_path = memory_bank_path
        
        print(f"Loading embedding model: {model_name}")
        self.model = None
        load_errors: list[str] = []
        
        # Set environment variables to avoid meta tensor issues
        original_device_map = os.environ.get('TRANSFORMERS_NO_ADVISORY_WARNINGS', None)
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
        
        # Optional device override via env: EMBEDDING_DEVICE=cpu|cuda
        preferred_device = os.environ.get('EMBEDDING_DEVICE', None)
        
        # First attempt with specified or default device
        try:
            if preferred_device:
                print(f"Attempting to load model on device: {preferred_device}")
                self.model = SentenceTransformer(
                    model_name, 
                    device=preferred_device,
                    model_kwargs={'torch_dtype': torch.float32, 'low_cpu_mem_usage': True}
                )
            else:
                # Try CUDA first if available, otherwise CPU
                if torch.cuda.is_available():
                    print("CUDA available, attempting GPU load...")
                    self.model = SentenceTransformer(
                        model_name, 
                        device='cuda',
                        model_kwargs={'torch_dtype': torch.float32, 'low_cpu_mem_usage': True}
                    )
                else:
                    print("CUDA not available, using CPU...")
                    self.model = SentenceTransformer(
                        model_name, 
                        device='cpu',
                        model_kwargs={'torch_dtype': torch.float32, 'low_cpu_mem_usage': True}
                    )
        except Exception as e_default:
            load_errors.append(str(e_default))
            print(f"First attempt failed: {e_default}")
            
            # Fallback to CPU with more conservative settings
            try:
                print("Retrying on CPU with conservative settings...")
                self.model = SentenceTransformer(
                    model_name, 
                    device='cpu',
                    model_kwargs={
                        'torch_dtype': torch.float32, 
                        'low_cpu_mem_usage': True,
                        'device_map': None,  # Disable automatic device mapping
                        'trust_remote_code': False
                    }
                )
            except Exception as e_cpu:
                load_errors.append(str(e_cpu))
                print(f"CPU fallback failed: {e_cpu}")
                
                # Final attempt with minimal configuration
                try:
                    print("Final attempt with minimal configuration...")
                    # Clear any cached models that might be causing issues
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.model = SentenceTransformer(
                        model_name,
                        device='cpu',
                        cache_folder=None  # Don't use cache to avoid corrupted files
                    )
                except Exception as e_final:
                    load_errors.append(str(e_final))
                    raise RuntimeError(
                        "Failed to initialize embedding model. Errors: " + " | ".join(load_errors)
                    )
        
        # Restore original environment variable
        if original_device_map is not None:
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = original_device_map
        elif 'TRANSFORMERS_NO_ADVISORY_WARNINGS' in os.environ:
            del os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize LLM client (OpenAI, Gemini, Qwen)
        self.llm_model = llm_model
        model_lower = (llm_model or "").lower()
        if "gemini" in model_lower:
            self.llm_provider = "gemini"
            self.llm_client = None  # call_gemini handles auth via GEMINI_API_KEY
            if not os.environ.get("GEMINI_API_KEY"):
                print("Warning: GEMINI_API_KEY not found; Gemini calls may fail.")
        elif "qwen" in model_lower:
            self.llm_provider = "qwen"
            self.llm_client = None  # DashScope handled inside call_qwen via DASHSCOPE_API_KEY
        elif "llama" in model_lower:
            self.llm_provider = "llama"
            self.llm_client = None
        else:
            self.llm_provider = "openai"
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if openai_api_key:
                self.llm_client = OpenAI(api_key=openai_api_key)
            else:
                self.llm_client = None
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize memory storage
        self.memory_items: List[MemoryItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        
        # ì¸ë±ìŠ¤ ì´ˆê¸°í™” (ìºì‹œ ì‚¬ìš© ë˜ëŠ” ìƒˆë¡œ êµ¬ì¶•)
        self._initialize_index(force_rebuild)
    
    def _get_cache_paths(self) -> Dict[str, Path]:
        """Generate cache file paths"""
        person_name = Path(self.memory_bank_path).stem
        
        # Cache name: {person_name}_{embedding_dim}
        cache_name = f"{person_name}_{self.embedding_dim}"
        
        return {
            'index': self.cache_dir / f"{cache_name}_index.faiss",
            'memories': self.cache_dir / f"{cache_name}_memories.pkl"
        }
    
    def _cache_exists(self) -> bool:
        """Check if cache files exist"""
        cache_paths = self._get_cache_paths()
        exists = all(path.exists() for path in cache_paths.values())
        
        if exists:
            print(f"Cache found for {Path(self.memory_bank_path).stem}")
        else:
            print("Cache not found")
        
        return exists
    
    def _save_to_cache(self):
        """Save current index and memories to cache"""
        cache_paths = self._get_cache_paths()

        try:
            # Save FAISS index
            print(f"Saving FAISS index to {cache_paths['index']}")
            faiss.write_index(self.index, str(cache_paths['index']))

            # Save memory items and embeddings
            print(f"Saving memories to {cache_paths['memories']}")
            with open(cache_paths['memories'], 'wb') as f:
                pickle.dump({
                    'memory_items': self.memory_items,
                    'embeddings': self.embeddings
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Successfully saved cache for {len(self.memory_items)} items")
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            # Delete partially created files on cache save failure
            for path in cache_paths.values():
                if path.exists():
                    path.unlink()
    
    def _load_from_cache(self) -> bool:
        """Load index and memories from cache"""
        cache_paths = self._get_cache_paths()
        
        try:
            # Load FAISS index
            print(f"Loading FAISS index from {cache_paths['index']}")
            self.index = faiss.read_index(str(cache_paths['index']))
            
            # Load memory items and embeddings
            print(f"Loading memories from {cache_paths['memories']}")
            with open(cache_paths['memories'], 'rb') as f:
                data = pickle.load(f)
                self.memory_items = data['memory_items']
                self.embeddings = data['embeddings']
            
            print(f"Successfully loaded {len(self.memory_items)} items from cache")
            return True
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def _initialize_index(self, force_rebuild: bool = False):
        """Initialize index - use cache or build from scratch"""
        if not force_rebuild and self._cache_exists():
            # Try loading from cache
            if self._load_from_cache():
                return
            else:
                print("Failed to load from cache, rebuilding index...")
        
        # Build from scratch
        self.load_memory_bank(self.memory_bank_path)
        
        # Save to cache
        self._save_to_cache()
    
    def load_memory_bank(self, memory_bank_path: str):
        """Load and index memory bank"""
        print(f"Building new index from {memory_bank_path}")
        
        with open(memory_bank_path, 'r', encoding='utf-8') as f:
            raw_memorybank = json.load(f)

        memory_data = raw_memorybank.get('memory_bank', [])

        # Process memory items
        self.memory_items = []
        for item in memory_data:
            memory_item = MemoryItem(
                timestamp=item['timestamp'],
                type=item['type'],
                object=item['object'],
                website=item.get('website', '')
            )
            self.memory_items.append(memory_item)
        
        # Generate batch embeddings (combine type + object)
        combined_texts = [f"{item.type}: {item.object}" for item in self.memory_items]
        print(f"Generating embeddings for {len(combined_texts)} memory items...")
        
        self.embeddings = self.model.encode(
            combined_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16
        )
        
        # Create and add FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Successfully indexed {len(self.memory_items)} memory items")
    
    def clear_cache(self):
        """Delete cache files"""
        cache_paths = self._get_cache_paths()
        
        for name, path in cache_paths.items():
            if path.exists():
                path.unlink()
                print(f"Deleted {name} cache: {path}")
        
        print("Cache cleared")
    
    def parse_memory_query(self, query_string: str) -> str:
        """Parse memory_access query"""
        match = re.search(r'memory_access\s*\[(.*?)\]', query_string, re.DOTALL)
        if match:
            return match.group(1).strip()
        return query_string.strip()
    
    def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for relevant memories for the query"""
        # Ensure index is initialized even if constructor initialization didn't complete
        if self.index is None or self.memory_items is None or len(self.memory_items) == 0:
            try:
                self._initialize_index(force_rebuild=False)
            except Exception as e:
                raise ValueError(f"Index not initialized. Load memory bank first. Details: {e}")
        if self.index is None or self.memory_items is None or len(self.memory_items) == 0:
            raise ValueError("Index not initialized. Load memory bank first.")
        
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.memory_items)))
        
        results = []
        retrieved_indices = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= threshold:
                memory_item = self.memory_items[idx]
                results.append({
                    **memory_item.to_dict(),
                    "similarity_score": float(dist),
                    "relevance_rank": len(results) + 1
                })
                retrieved_indices.append(int(idx))

        if retrieved_indices:
            print(f"Retrieved memory index: {', '.join(map(str, retrieved_indices))}")
        else:
            print("No memories retrieved above threshold")

        return results, retrieved_indices
    
    def generate_answer(self, query: str, memories: List[Dict[str, Any]]) -> str:
        """Generate answer to query based on retrieved memories"""
        if not memories:
            return "No relevant memories found to generate an answer."
        
        memory_context = self._format_memories_for_prompt(memories)

        prompt = f"""Based on the following user memories, provide a helpful and specific answer to the query.

    Query: {query}

    Relevant User Memories:
    {memory_context}

    CRITICAL INSTRUCTIONS:
    1. Answer ONLY based on the information found in the provided memories
    2. Do NOT create or invent any new facts that are not explicitly mentioned in the memories
    3. Be SPECIFIC and DETAILED in your response:
    - If asked about preferred brands, provide exact brand names found in the memories
    - If asked about favorite food types, provide specific food names, not generic categories
    - If asked about locations, provide specific venue/store names
    - If asked about timing/schedule, provide specific times or patterns shown in memories
    4. If the memories don't contain enough information to fully answer the query, explicitly state what information is available and what is missing
    5. Use direct evidence from the memories to support your answer

    ANSWER REQUIREMENTS:
    - The answer must directly address what the user is asking about
    - Include specific names, brands, products, locations, or times as found in the memories
    - If multiple options exist in the memories, mention all of them with their frequency or context
    - Never generalize when specific information is available
    
    Answer:"""
#     - IMPORTANT: Summarize your answer in one sentence at the end.

        try:
            model_lower = (self.llm_model or "").lower()
            # Gemini provider branch
            if getattr(self, "llm_provider", "").lower() == "gemini" or "gemini" in model_lower:
                return call_gemini(
                    prompt=prompt,
                    model_id=self.llm_model,
                    system_prompt="You are a precise assistant that analyzes user memories to provide factual, specific answers.",
                    temperature=0.7,
                )

            # Qwen provider branch
            if getattr(self, "llm_provider", "").lower() == "qwen" or "qwen" in model_lower:
                return call_qwen(
                    prompt=prompt,
                    model_id=self.llm_model,
                    system_prompt="You are a precise assistant that analyzes user memories to provide factual, specific answers.",
                    temperature=0.7,
                )

            # Llama provider branch
            if getattr(self, "llm_provider", "").lower() == "llama" or "llama" in model_lower:
                return call_llama(
                    prompt=prompt,
                    model_id=self.llm_model,
                    system_prompt="You are a precise assistant that analyzes user memories to provide factual, specific answers.",
                    temperature=0.7,
                )

            # OpenAI provider branch
            if not self.llm_client:
                return "LLM client not initialized. Cannot generate answer."

            if "o3" in model_lower:
                # Normalize common aliases to Responses API model ids
                normalized_model = (
                    "o3-mini-high" if "gpt-o3-mini-high" in model_lower else
                    "o3-mini" if "gpt-o3-mini" in model_lower else
                    ("o3" if model_lower.startswith("gpt-o3") or model_lower == "o3" else self.llm_model)
                )
                input_text = (
                    "System: You are a precise assistant that analyzes user memories to provide factual, specific answers.\n\n"
                    f"User: {prompt}"
                )
                resp = self.llm_client.responses.create(
                    model=normalized_model,
                    input=input_text
                )
                # Best-effort extraction
                try:
                    return resp.output_text.strip()
                except Exception:
                    try:
                        output = getattr(resp, "output", None)
                        if isinstance(output, list) and output:
                            content = getattr(output[0], "content", None)
                            if isinstance(content, list) and content:
                                first = content[0]
                                text = getattr(first, "text", None)
                                if text:
                                    return text.strip()
                                if isinstance(first, dict) and "text" in first:
                                    return str(first["text"]).strip()
                    except Exception:
                        pass
                    return str(resp)
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that analyzes user memories to provide factual, specific answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories as text suitable for LLM prompts"""
        formatted_memories = []
        
        for i, memory in enumerate(memories, 1):
            formatted_memory = (
                f"{i}. [{memory['timestamp']}] "
                f"Type: {memory['type']}, "
                f"Action: {memory['object']}"
            )
            if memory.get('website'):
                formatted_memory += f", Website: {memory['website']}"
            formatted_memory += f" (Relevance: {memory['similarity_score']:.2f})"
            
            formatted_memories.append(formatted_memory)
        
        return "\n".join(formatted_memories)
    
    def process_memory_request(self, memory_request: str, top_k: int = 10, generate_answer: bool = True) -> Dict[str, Any]:
        """Process memory_access request and generate answer"""
        query = self.parse_memory_query(memory_request)
        print(f"Processing query: {query}")
        
        retrieved_memories, retrieved_indices = self.retrieve(query, top_k=top_k)  # â† ë³€ê²½!
        
        results = {
            "query": query,
            "memories": retrieved_memories,
            "retrieved_indices": retrieved_indices,  # â† ì¸ë±ìŠ¤ ì¶”ê°€!
            "total_retrieved": len(retrieved_memories),
            "search_timestamp": datetime.now().isoformat()
        }
        
        if generate_answer and retrieved_memories:
            print("Generating answer based on retrieved memories...")
            results['generated_answer'] = self.generate_answer(query, retrieved_memories)
        else:
            results['generated_answer'] = None
        
        if retrieved_memories:
            results['summary'] = f"Found {len(retrieved_memories)} relevant memories and generated answer."
        else:
            results['summary'] = "No relevant memories found for the query."
        
        return results


# Usage example
def main():
    # Initialize index
    retriever = MemoryBankRetriever(
        model_name="dunzhang/stella_en_1.5B_v5",
        memory_bank_path="AgentOccam/memory_bank/Alex_Garcia.json",
        cache_dir="./cache",
        force_rebuild=True  # Trueë¡œ ì„¤ì •í•˜ë©´ ìºì‹œ ë¬´ì‹œí•˜ê³  ìž¬êµ¬ì¶•
    )
    
    # Check cache path
    cache_paths = retriever._get_cache_paths()
    print(f"\nCache files:")
    for name, path in cache_paths.items():
        print(f"  {name}: {path}")
    
    # Search test
    queries = [
        "memory_access [any restaurant that is **located where the user frequently visited**]",
    ]
    
    for memory_request in queries:
        results = retriever.process_memory_request(memory_request, top_k=10)
        
        print("\n" + "="*50)
        print(f"Query: {results['query']}")
        print(f"Found: {results['total_retrieved']} memories")

        print("\nRetrieved Memories:")
        for i, mem in enumerate(results['memories'], 1):
            print(f"  {i}. [{mem['timestamp']}] {mem['object'][:50]}...")

        if results['generated_answer']:
            print(f"\nGenerated Answer:\n{results['generated_answer']}")

if __name__ == "__main__":
    main()