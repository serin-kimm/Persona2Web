"""
Memory Retriever: Embedding-based retrieval for user memory banks.

Uses sentence-transformers for encoding and FAISS for fast similarity search.
Supports caching of embeddings and FAISS index for faster subsequent loads.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)


class MemoryItem:
	"""A single memory item from the JSON memory bank."""

	def __init__(self, data: dict[str, Any], index: int):
		self.data = data
		self.index = index
		# Extract common fields - support multiple formats
		self.timestamp = data.get('timestamp', '')
		self.type = data.get('type', data.get('category', 'unknown'))
		self.object = data.get('object', data.get('value', data.get('content', data.get('text', ''))))
		self.website = data.get('website', '')
		self.key = data.get('key', '')
		self.metadata = data.get('metadata', {})

	@property
	def category(self) -> str:
		"""Alias for type - used for compatibility."""
		return self.type

	@property
	def content(self) -> str:
		"""Alias for object - used for compatibility."""
		return self.object

	def to_dict(self) -> dict[str, Any]:
		return {
			**self.data,
			'index': self.index,
		}

	def to_searchable_text(self) -> str:
		"""Convert to text for embedding."""
		parts = []
		if self.category:
			parts.append(f"[{self.category}]")
		if self.key:
			parts.append(f"{self.key}:")
		parts.append(self.content)
		return " ".join(parts)


class MemoryRetriever:
	"""
	Embedding-based memory retriever using sentence-transformers and FAISS.

	Loads memories from a JSON file and builds a vector index for fast retrieval.
	Supports caching of embeddings and index to avoid rebuilding on every load.
	"""

	# Default cache directory relative to this file's location
	DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "cache"
	# Process-wide model cache: (model_name, device) -> SentenceTransformer
	_MODEL_CACHE: dict[tuple[str, str], Any] = {}
	_MODEL_DIM_CACHE: dict[tuple[str, str], int] = {}
	
	def __init__(
		self,
		memory_bank_path: str | Path | None = None,
		embedding_model: str = 'dunzhang/stella_en_1.5B_v5',
		llm: 'BaseChatModel | None' = None,
		llm_model: str | None = None,
		cache_dir: str | Path | None = None,
		force_rebuild: bool = False,
	):
		"""
		Initialize the MemoryRetriever.

		Args:
			memory_bank_path: Path to the JSON file containing memories
			embedding_model: Name of the sentence-transformers model to use
			llm: LLM client for answer generation (browser-use BaseChatModel)
			llm_model: Model name for the LLM (if different from llm.model)
			cache_dir: Directory to store cached embeddings and index
			force_rebuild: If True, ignore cache and rebuild index from scratch
		"""
		self.memory_bank_path = Path(memory_bank_path) if memory_bank_path else None
		self.embedding_model_name = embedding_model
		self.llm = llm
		self.llm_model = llm_model or (getattr(llm, 'model', None) if llm else None)
		self.cache_dir: Path = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		self.force_rebuild = force_rebuild

		self.model = None  # SentenceTransformer model
		self.index = None  # FAISS index
		self.memory_items: list[MemoryItem] = []
		self.embeddings: np.ndarray | None = None
		self.embedding_dim: int | None = None

		self.logger = logging.getLogger(f'{__name__}.MemoryRetriever')

		if self.memory_bank_path and self.memory_bank_path.exists():
			self._initialize_index(force_rebuild=force_rebuild)

	def _get_cache_paths(self) -> dict[str, Path]:
		"""Generate cache file paths based on memory bank name and embedding dimension."""
		if not self.memory_bank_path:
			return {}

		person_name = self.memory_bank_path.stem
		dim_str = str(self.embedding_dim) if self.embedding_dim else 'unknown'
		cache_name = f"{person_name}_{dim_str}"

		base: Path = Path(self.cache_dir)
		return {
			'index': base / f'{cache_name}_index.faiss',
			'memories': base / f'{cache_name}_memories.pkl',
		}

	def _cache_exists(self) -> bool:
		"""Check if cache files exist."""
		cache_paths = self._get_cache_paths()
		if not cache_paths or not self.memory_bank_path:
			return False

		exists = all(path.exists() for path in cache_paths.values())
		if exists:
			self.logger.info(f'Cache found for {self.memory_bank_path.stem}')
		else:
			self.logger.debug('Cache not found')
		return exists

	def _save_to_cache(self) -> None:
		"""Save current index and memories to cache."""
		cache_paths = self._get_cache_paths()
		if not cache_paths or self.index is None:
			return

		try:
			import faiss  # type: ignore[import-not-found]

			# Save FAISS index
			self.logger.info(f'Saving FAISS index to {cache_paths["index"]}')
			faiss.write_index(self.index, str(cache_paths['index']))

			# Save memory items and embeddings
			self.logger.info(f'Saving memories to {cache_paths["memories"]}')
			with open(cache_paths['memories'], 'wb') as f:
				pickle.dump({
					'memory_items': self.memory_items,
					'embeddings': self.embeddings,
					'embedding_dim': self.embedding_dim,
				}, f, protocol=pickle.HIGHEST_PROTOCOL)

			self.logger.info(f'Successfully saved cache for {len(self.memory_items)} items')

		except Exception as e:
			self.logger.error(f'Error saving cache: {e}')
			# Delete partial cache files on error
			for path in cache_paths.values():
				if path.exists():
					path.unlink()

	def _load_from_cache(self) -> bool:
		"""Load index and memories from cache."""
		cache_paths = self._get_cache_paths()
		if not cache_paths:
			return False

		try:
			import faiss  # type: ignore[import-not-found]

			# Load FAISS index
			self.logger.info(f'Loading FAISS index from {cache_paths["index"]}')
			self.index = faiss.read_index(str(cache_paths['index']))

			# Load memory items and embeddings
			# Use custom unpickler to handle module path changes (e.g., old 'retriever' -> new 'browser_use.personalization.retriever')
			self.logger.info(f'Loading memories from {cache_paths["memories"]}')

			class ModuleRemappingUnpickler(pickle.Unpickler):
				"""Custom unpickler that remaps old module paths to new ones."""
				def find_class(self, module: str, name: str):
					# Remap 'retriever' -> 'browser_use.personalization.retriever'
					if module == 'retriever':
						module = 'browser_use.personalization.retriever'
					return super().find_class(module, name)

			with open(cache_paths['memories'], 'rb') as f:
				data = ModuleRemappingUnpickler(f).load()
				self.memory_items = data['memory_items']
				self.embeddings = data['embeddings']
				self.embedding_dim = data.get('embedding_dim')

			self.logger.info(f'Successfully loaded {len(self.memory_items)} items from cache')
			return True

		except Exception as e:
			self.logger.error(f'Error loading cache: {e}')
			return False

	def _infer_embedding_dim_from_cache(self) -> int | None:
		"""Infer embedding dimension from existing cache files.

		Cache files are named like: {person_name}_{dim}_index.faiss
		This allows us to find the cache without loading the model first.
		"""
		if not self.memory_bank_path:
			return None

		person_name = self.memory_bank_path.stem
		cache_dir = Path(self.cache_dir)

		if not cache_dir.exists():
			return None

		# Look for existing cache files matching pattern: {person_name}_{dim}_index.faiss
		import re
		pattern = re.compile(rf'^{re.escape(person_name)}_(\d+)_index\.faiss$')

		for f in cache_dir.iterdir():
			match = pattern.match(f.name)
			if match:
				dim = int(match.group(1))
				self.logger.debug(f'Inferred embedding_dim={dim} from cache file: {f.name}')
				return dim

		return None

	def _load_embedding_model(self) -> None:
		"""Load the embedding model for encoding queries."""
		if self.model is not None:
			return

		from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

		self.logger.info(f'Loading embedding model: {self.embedding_model_name}')
		# Device selection with env override
		device_env = os.environ.get('BROWSER_USE_EMBEDDING_DEVICE')
		if device_env:
			device = device_env.strip().lower()
		else:
			# Auto-detect device: use GPU if available, else CPU
			import torch
			if torch.cuda.is_available():
				device = 'cuda'
			# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
			# 	device = 'mps'  # Apple Silicon
			else:
				device = 'cpu'
			
		self.logger.info(f'Using device: {device}')
		cache_key = (self.embedding_model_name, device)
		cached_model = MemoryRetriever._MODEL_CACHE.get(cache_key)
		if cached_model is not None:
			self.model = cached_model
			self.embedding_dim = MemoryRetriever._MODEL_DIM_CACHE.get(cache_key, self.embedding_dim)
		else:
			self.logger.info(f'Initializing SentenceTransformer (this may take a while)...')
			try:
				self.model = SentenceTransformer(
					self.embedding_model_name,
					device=device,
					trust_remote_code=True,
					model_kwargs={
						'low_cpu_mem_usage': True,
					},
				)
				self.logger.info(f'Embedding model loaded successfully')
			except Exception as e:
				self.logger.error(f'Failed to load embedding model: {e}')
				raise
			dim = self.model.get_sentence_embedding_dimension()
			if self.embedding_dim is None:
				self.embedding_dim = dim
			MemoryRetriever._MODEL_CACHE[cache_key] = self.model
			if dim is not None:
				MemoryRetriever._MODEL_DIM_CACHE[cache_key] = dim

	def _initialize_index(self, force_rebuild: bool = False) -> None:
		"""Initialize the FAISS index - use cache or build new."""
		try:
			import faiss  # type: ignore[import-not-found]
		except ImportError as e:
			self.logger.error(
				f'Missing dependencies for MemoryRetriever: {e}. '
				'Install with: pip install sentence-transformers faiss-cpu'
			)
			raise

		# Infer embedding_dim from cache files before checking cache existence
		if self.embedding_dim is None:
			inferred_dim = self._infer_embedding_dim_from_cache()
			if inferred_dim is not None:
				self.embedding_dim = inferred_dim

		# Try to load from cache first (before any model load).
		require_cache = os.environ.get('BROWSER_USE_MEMORY_REQUIRE_CACHE', '').strip().lower() in ('1', 'true', 'yes')
		if not force_rebuild:
			if self._cache_exists():
				if self._load_from_cache():
					# Cache loaded, but we still need the model for encoding queries
					return
				else:
					if require_cache:
						raise RuntimeError('Cache present but failed to load and rebuild is disabled (BROWSER_USE_MEMORY_REQUIRE_CACHE).')
					self.logger.warning('Failed to load from cache, rebuilding index...')
			elif require_cache:
				# No cache available and rebuild is disabled
				raise RuntimeError('Cache not found and rebuild is disabled (BROWSER_USE_MEMORY_REQUIRE_CACHE).')

		# Load model for building index
		self._load_embedding_model()

		# Build new index
		self._load_memory_bank()

		if not self.memory_items:
			self.logger.warning('No memory items found in memory bank')
			return

		# Generate embeddings
		self.logger.info(f'Generating embeddings for {len(self.memory_items)} memories...')
		texts = [item.to_searchable_text() for item in self.memory_items]
		assert self.model is not None, 'Embedding model not loaded'
		self.embeddings = self.model.encode(
			texts,
			normalize_embeddings=True,
			show_progress_bar=False,
			batch_size=32,
		).astype('float32')

		# Build FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
		self.index = faiss.IndexFlatIP(self.embedding_dim)
		self.index.add(self.embeddings)  # type: ignore[arg-type]

		self.logger.info(f'FAISS index built with {self.index.ntotal} vectors')

		# Save to cache
		self._save_to_cache()

	def _load_memory_bank(self) -> None:
		"""Load memory items from the JSON file."""
		if not self.memory_bank_path or not self.memory_bank_path.exists():
			self.logger.warning(f'Memory bank file not found: {self.memory_bank_path}')
			return

		self.logger.info(f'Loading memory bank from {self.memory_bank_path}')

		with open(self.memory_bank_path, 'r', encoding='utf-8') as f:
			data = json.load(f)

		# Handle multiple JSON formats
		if isinstance(data, list):
			items = data
		elif isinstance(data, dict):
			# Try various common keys
			items = (
				data.get('memory_bank') or
				data.get('entries') or
				data.get('memories') or
				data.get('items') or
				[]
			)
		else:
			items = []

		self.memory_items = [MemoryItem(item, idx) for idx, item in enumerate(items)]
		self.logger.info(f'Loaded {len(self.memory_items)} memory items')

	def load_memory_bank(self, path: str | Path) -> None:
		"""Load a new memory bank from the given path."""
		self.memory_bank_path = Path(path)
		self._initialize_index(force_rebuild=self.force_rebuild)

	def clear_cache(self) -> None:
		"""Delete cache files for current memory bank."""
		cache_paths = self._get_cache_paths()
		for name, path in cache_paths.items():
			if path.exists():
				path.unlink()
				self.logger.info(f'Deleted {name} cache: {path}')
		self.logger.info('Cache cleared')

	def retrieve(
		self,
		query: str,
		top_k: int = 10,
		threshold: float = 0.5,
	) -> tuple[list[dict[str, Any]], list[int]]:
		"""
		Retrieve relevant memories for a query using embedding similarity.

		Args:
			query: The search query
			top_k: Maximum number of results to return
			threshold: Minimum similarity score (0-1) to include

		Returns:
			Tuple of (list of memory dicts with scores, list of indices)
		"""
		if self.index is None or not self.memory_items:
			try:
				self._initialize_index(force_rebuild=False)
			except Exception as e:
				raise ValueError(f'Index not initialized. Load memory bank first. Details: {e}')

		if self.index is None or not self.memory_items:
			raise ValueError('Index not initialized. Load memory bank first.')

		# Load model if not already loaded (e.g., when loaded from cache)
		if self.model is None:
			self._load_embedding_model()

		if self.model is None:
			raise ValueError('Failed to load embedding model.')

		# Encode query
		query_embedding = self.model.encode(
			[query],
			normalize_embeddings=True
		).astype('float32')

		# Search
		k = min(top_k, len(self.memory_items))
		distances, indices = self.index.search(query_embedding, k)  # type: ignore[arg-type]

		results = []
		retrieved_indices = []

		for dist, idx in zip(distances[0], indices[0]):
			if idx < 0:  # FAISS returns -1 for empty results
				continue
			if dist >= threshold:
				memory_item = self.memory_items[idx]
				results.append({
					**memory_item.to_dict(),
					'similarity_score': float(dist),
					'relevance_rank': len(results) + 1,
				})
				retrieved_indices.append(int(idx))

		if retrieved_indices:
			self.logger.debug(f"Retrieved memory indices: {', '.join(map(str, retrieved_indices))}")
		else:
			self.logger.debug('No memories retrieved above threshold')

		return results, retrieved_indices

	def _format_memories_for_prompt(self, memories: list[dict[str, Any]]) -> str:
		"""Format retrieved memories for the LLM prompt."""
		lines = []
		for i, mem in enumerate(memories, 1):
			timestamp = mem.get('timestamp', '')
			mem_type = mem.get('type', 'unknown')
			obj = mem.get('object', '')
			website = mem.get('website', '')
			score = mem.get('similarity_score', 0)

			line = f"{i}. [{timestamp}] [{mem_type}] {obj} (website: {website}, relevance: {score:.2f})"
			lines.append(line)

		return "\n".join(lines)

	async def generate_answer(self, query: str, memories: list[dict[str, Any]]) -> str:
		"""
		Generate an answer based on retrieved memories using the LLM.

		Args:
			query: The original query
			memories: List of retrieved memory dicts

		Returns:
			Generated answer string
		"""
		if not memories:
			return "No relevant memories found to generate an answer."

		if not self.llm:
			return self._format_memories_for_prompt(memories)

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

		try:
			from browser_use.llm.messages import SystemMessage, UserMessage

			messages = [
				SystemMessage(content='You are a precise assistant that analyzes user memories to provide factual, specific answers.'),
				UserMessage(content=prompt),
			]

			response = await self.llm.ainvoke(messages)
			return response.completion.strip() if hasattr(response, 'completion') else str(response)

		except Exception as e:
			self.logger.error(f'Error generating answer: {e}')
			return f"Error generating answer: {str(e)}"

	def unload_model(self) -> None:
		"""
		Unload the embedding model from memory to free up RAM.
		Call this after pre-execution processing, before browser starts.
		"""
		import gc

		cache_key = None
		for key in list(MemoryRetriever._MODEL_CACHE.keys()):
			if key[0] == self.embedding_model_name:
				cache_key = key
				break

		if cache_key and cache_key in MemoryRetriever._MODEL_CACHE:
			del MemoryRetriever._MODEL_CACHE[cache_key]
			self.logger.info(f'Removed model from cache: {cache_key}')

		if self.model is not None:
			del self.model
			self.model = None
			self.logger.info('Unloaded embedding model from instance')

		gc.collect()

		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				self.logger.info('Cleared CUDA cache')
		except ImportError:
			pass

		self.logger.info('Model unloaded and memory freed')

	@classmethod
	def clear_all_models(cls) -> None:
		"""Clear all cached models across all instances."""
		import gc

		model_count = len(cls._MODEL_CACHE)
		cls._MODEL_CACHE.clear()
		cls._MODEL_DIM_CACHE.clear()
		gc.collect()

		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except ImportError:
			pass

		logger.info(f'Cleared {model_count} models from global cache')

	def retrieve_and_format(
		self,
		query: str,
		top_k: int = 10,
		threshold: float = 0.5,
	) -> str:
		"""
		Retrieve memories and format them as context text.

		Args:
			query: The search query
			top_k: Maximum number of results
			threshold: Minimum similarity score

		Returns:
			Formatted context string
		"""
		memories, _ = self.retrieve(query, top_k, threshold)

		if not memories:
			return ""

		lines = ["<user_context>"]
		lines.append("The following user-specific information was retrieved from their memory bank:")
		lines.append("")

		for mem in memories:
			mem_type = mem.get('type', mem.get('category', 'unknown'))
			key = mem.get('key', '')
			content = mem.get('object', mem.get('value', mem.get('content', mem.get('text', ''))))

			if key:
				lines.append(f"- [{mem_type}] {key}: {content}")
			else:
				lines.append(f"- [{mem_type}] {content}")

		lines.append("</user_context>")
		return "\n".join(lines)
