# access_scheme.py
import json
from typing import Dict, Any, Optional
from AgentOccam.AgentOccam_personalized import AgentOccam
from AgentOccam.retriever import MemoryBankRetriever
from AgentOccam.llms.gpt import call_gpt as call_openai
from AgentOccam.llms.qwen import call_gpt as call_qwen
from AgentOccam.llms.gemini import call_gemini
from AgentOccam.llms.llama import call_llama

class MemoryEnhancedAgent:
    def __init__(self, config, prompt_dict, memory_bank_config):
        """
        Memory-enhanced agent

        Args:
            config: Overall configuration
            prompt_dict: Prompt dictionary
            memory_bank_config: Memory bank configuration
        """
        self.config = config
        self.prompt_dict = prompt_dict
        self.memory_bank_config = memory_bank_config

        # Set memory mode (default: on-demand)
        self.memory_mode = getattr(config.agent, 'memory_mode', 'on-demand')

        # AgentOccam instance
        self.agent_occam = None

        # Memory retriever for pre-execution mode
        self.memory_retriever = None
        # Common: memory event log storage (used by both modes)
        self.memory_events = []
        if self.memory_mode == 'pre-execution' and memory_bank_config:
            self.init_memory_retriever()
    
    def init_memory_retriever(self):
        """Initialize memory retriever (for pre-execution mode)"""
        try:
            self.memory_retriever = MemoryBankRetriever(
                model_name=self.memory_bank_config.get('model_name'),
                memory_bank_path=self.memory_bank_config.get('memory_bank_path'),
                llm_model=self.memory_bank_config.get('llm_model', 'gpt-4-turbo')
            )
            print(f"Memory retriever initialized for pre-execution mode")
        except Exception as e:
            print(f"Failed to initialize memory retriever: {e}")
            self.memory_retriever = None
    
    def act(self, objective: str, env):
        """
        Main execution method (same interface as AgentOccam)
        """
        print(f"Memory mode: {self.memory_mode}")
        print(f"Received objective: {objective[:200]}..." if len(objective) > 200 else f"Received objective: {objective}")
        
        if self.memory_mode == 'pre-execution':
            print("\n" + "="*60)
            print("=== PRE-EXECUTION MEMORY PROCESSING ===")
            print("="*60)

            # Pre-execution: Refine query with memory in advance
            enhanced_objective = self.preprocess_with_memory(objective)
            
            print(f"\n=== OBJECTIVE TRANSFORMATION ===")
            print(f"Original: {objective}")
            print(f"Enhanced: {enhanced_objective}")
            print("="*60 + "\n")

            # Execute AgentOccam with enhanced query (memory functionality disabled)
            print("=== STARTING AGENT EXECUTION WITH ENHANCED OBJECTIVE ===\n")
            self.agent_occam = AgentOccam(
                config=self.config,
                prompt_dict=self.prompt_dict,
                memory_bank_config=None  # Memory functionality OFF
            )

            self.agent_occam.disable_memory_check = True

            return self.agent_occam.act(enhanced_objective, env)

        else:  # 'on-demand' (default)
            # On-demand: Use original AgentOccam as is (access memory during execution if needed)
            self.agent_occam = AgentOccam(
                config=self.config,
                prompt_dict=self.prompt_dict,
                memory_bank_config=self.memory_bank_config  # Memory functionality ON
            )
            return self.agent_occam.act(objective, env)
    
    def preprocess_with_memory(self, objective: str) -> str:
        """
        Pre-execution mode: Query preprocessing using memory
        """
        if not self.memory_retriever:
            print("Memory retriever not available, using original objective")
            return objective
        
        try:
            # 1. Analyze memory type needed for task
            print("\n=== Analyzing required memory ===")
            memory_queries = self.analyze_required_memory(objective)
            print(f"Memory queries found: {memory_queries}")

            if not memory_queries:
                print("No memory needed for this task")
                return objective

            # 2. Retrieve memories
            print("\n=== Retrieving memories ===")
            retrieved_memories = self.retrieve_memories(memory_queries)
            print(f"Retrieved memories: {retrieved_memories}")

            # 3. Rewrite query with memory information
            print("\n=== Rewriting objective with memory ===")
            enhanced_objective = self.rewrite_objective_with_memory(
                objective, retrieved_memories
            )
            
            return enhanced_objective
            
        except Exception as e:
            print(f"Error in pre-execution memory processing: {e}")
            import traceback
            traceback.print_exc()
            return objective
    
    def analyze_required_memory(self, objective: str) -> list:
        """Analyze memory type needed for task - can generate up to 5 queries"""
        prompt = f"""
        Think carefully and step by step.
        Given this web navigation task: "{objective}"
        
        Analyze what personal information could help make this task more concrete and achievable:
        1. Identify different types of user-specific information that would resolve ambiguities
        2. Consider various aspects: preferences, history, locations, timing, brands, etc.
        3. Generate separate, specific memory queries for each distinct information type needed
        4. Maximum 5 queries allowed - prioritize the most important ones

        If no personal information is needed, return "NONE".
        
        Format your response as:
        NEEDS_MEMORY: [YES/NO]
        REASON: [One sentence explaining why memory is/isn't needed]
        QUERIES:
        - [First specific query about user preferences/history]
        - [Second specific query about different aspect, if needed]
        - [Third specific query about another aspect, if needed]
        - [Fourth specific query about another aspect, if needed]
        - [Fifth specific query about another aspect, if needed]
        
        Alternative format (if only one query needed):
        QUERY: [single specific information needed]
        
        Examples:
        - For shopping: "user's preferred clothing brands", "user's size preferences", "user's budget range", "user's favorite stores", "user's shopping history"
        - For travel: "user's preferred airlines", "user's seating preferences", "user's frequent destinations", "user's accommodation preferences", "user's travel budget"
        - For dining: "user's favorite cuisines", "user's dietary restrictions", "user's preferred restaurants", "user's food allergies", "user's dining budget"
        """
        
        # Use the same model as the agent actor for consistency
        actor_model = getattr(self.config.agent.actor, 'model', 'gpt-4-turbo')
        model_lower = (actor_model or "").lower().strip()
        if 'qwen' in model_lower:
            response = call_qwen(prompt, model_id=actor_model, temperature=0.95)
        elif 'gemini' in model_lower:
            response = call_gemini(prompt=prompt, model_id=actor_model, temperature=0.95)
        elif 'llama' in model_lower:
            response = call_llama(prompt=prompt, model_id=actor_model, temperature=0.95)
        else:
            response = call_openai(prompt, model_id=actor_model, temperature=0.95)
        print("=== LLM Response ===")
        print(response)
        print("=== End LLM Response ===")
        
        # Parse response (robust handling)
        import re
        # Whether to use Qwen thinking special parsing is only used in parsing stage
        is_qwen_thinking = model_lower == 'qwen3-next-80b-a3b-thinking'
        
        # 1) Parse NEEDS_MEMORY flag explicitly
        needs_match = re.search(r"NEEDS_MEMORY\s*:\s*(YES|NO)", response, re.IGNORECASE)
        needs_memory = needs_match.group(1).upper() if needs_match else None
        print(f"NEEDS_MEMORY parsed: {needs_memory}")
        
        # 2) Check if QUERIES section is explicitly NONE (Qwen thinking judges based on last section)
        if is_qwen_thinking:
            lines = response.splitlines()
            import re as _re
            header_pat = _re.compile(r"^\s*QUERIES\s*:\s*(.*)$", _re.IGNORECASE)
            last_idx = None
            last_header_inline = None
            for i, line in enumerate(lines):
                m = header_pat.match(line)
                if m:
                    last_idx = i
                    last_header_inline = (m.group(1) or "").strip()
            if last_idx is not None and last_header_inline and last_header_inline.upper() in {"NONE", "N/A"}:
                print("Found explicit NONE in last QUERIES section")
                return []
        else:
            if re.search(r"^\s*QUERIES?\s*:\s*(NONE|N/A)\s*$", response, re.IGNORECASE | re.MULTILINE):
                print("Found explicit NONE in QUERIES section")
                return []
        
        # If NEEDS_MEMORY is explicitly NO, exit immediately
        if needs_memory == 'NO':
            print("NEEDS_MEMORY is NO, returning empty list")
            return []
        
        # 3) Collect queries
        queries = []
        lines = response.splitlines()
        if is_qwen_thinking:
            # Qwen thinking: parse only last QUERIES section
            import re as _re
            header_pat = _re.compile(r"^\s*QUERIES\s*:\s*(.*)$", _re.IGNORECASE)
            section_start = None
            for i, line in enumerate(lines):
                if header_pat.match(line):
                    section_start = i
            if section_start is not None:
                # Until next section header (capital+colon) appears, but 'QUERY:' is treated as content
                section_end = len(lines)
                for j in range(section_start + 1, len(lines)):
                    if _re.match(r"^\s*[A-Z_]+\s*:", lines[j]) and not _re.match(r"^\s*QUERY\s*:", lines[j], _re.IGNORECASE):
                        section_end = j
                        break
                block = lines[section_start + 1:section_end]
                for line in block:
                    b = _re.match(r"^\s*-\s*(.+)$", line)
                    if b:
                        q = b.group(1).strip()
                        if q and not q.lower().startswith(("example", "e.g.", "for ")):
                            queries.append(q)
                            print(f"Found bullet query (last QUERIES): {q}")
                        continue
                    s = _re.match(r"^\s*QUERY\s*:\s*(.+)$", line, _re.IGNORECASE)
                    if s:
                        q = s.group(1).strip()
                        if q and q not in ['NONE', 'N/A']:
                            queries.append(q)
                            print(f"Found single query (last QUERIES): {q}")
                        continue
            else:
                # If last QUERIES section not found, fallback to existing general logic
                in_queries_section = False
                for i, line in enumerate(lines):
                    if re.match(r"^\s*QUERIES\s*:\s*$", line, re.IGNORECASE):
                        in_queries_section = True
                        print("Found QUERIES section")
                        continue
                    if re.match(r"^\s*[A-Z_]+\s*:", line) and not re.match(r"^\s*QUERIES\s*:", line, re.IGNORECASE):
                        in_queries_section = False
                    bullet = re.match(r"^\s*-\s*(.+)$", line)
                    if bullet:
                        query = bullet.group(1).strip()
                        if query and not query.lower().startswith(('example', 'e.g.', 'for ')):
                            queries.append(query)
                            print(f"Found bullet query: {query}")
                        continue
                    single = re.match(r"^\s*QUERY\s*:\s*(.+)$", line, re.IGNORECASE)
                    if single:
                        query = single.group(1).strip()
                        if query and query not in ['NONE', 'N/A']:
                            queries.append(query)
                            print(f"Found single query: {query}")
                        continue
        else:
            # Existing general logic (when model is not Qwen)
            in_queries_section = False
            for i, line in enumerate(lines):
                if re.match(r"^\s*QUERIES\s*:\s*$", line, re.IGNORECASE):
                    in_queries_section = True
                    print("Found QUERIES section")
                    continue
                if re.match(r"^\s*[A-Z_]+\s*:", line) and not re.match(r"^\s*QUERIES\s*:", line, re.IGNORECASE):
                    in_queries_section = False
                bullet = re.match(r"^\s*-\s*(.+)$", line)
                if bullet:
                    query = bullet.group(1).strip()
                    if query and not query.lower().startswith(('example', 'e.g.', 'for ')):
                        queries.append(query)
                        print(f"Found bullet query: {query}")
                    continue
                single = re.match(r"^\s*QUERY\s*:\s*(.+)$", line, re.IGNORECASE)
                if single:
                    query = single.group(1).strip()
                    if query and query not in ['NONE', 'N/A']:
                        queries.append(query)
                        print(f"Found single query: {query}")
                    continue
        
        # 4) Cleanup: remove empty items and duplicates
        queries = [q for q in (q.strip() for q in queries) if q]
        queries = list(dict.fromkeys(queries))
        
        # 5) Limit to maximum 5
        if len(queries) > 5:
            print(f"Warning: {len(queries)} queries found, limiting to top 5")
            queries = queries[:5]
        
        print(f"Final queries after cleanup: {queries}")
        print(f"Generated {len(queries)} memory queries")
        return queries
    
    def retrieve_memories(self, queries: list) -> dict:
        """Memory search"""
        all_memories = {}
        
        for query in queries:
            print(f"Retrieving: {query}")
            results = self.memory_retriever.process_memory_request(
                f"memory_access [{query}]",
                top_k=20,
                generate_answer=True
            )
            
            if results.get('generated_answer'):
                all_memories[query] = results['generated_answer']
                print(f"Found relevant memory for: {query}")
                # Save event (pre-execution)
                self.memory_events.append({
                    'query': query,
                    'retrieved_indices': list(results.get('retrieved_indices', [])),
                    'answer': results.get('generated_answer')
                })
        
        return all_memories
    
    def rewrite_objective_with_memory(self, objective: str, memories: dict) -> str:
        """Refine query with memory information"""
        if not memories:
            return objective
        
        memory_context = "\n".join([f"- {q}: {a}" for q, a in memories.items()])
        
        prompt = f"""
        Original task: {objective}
        
        User-specific information:
        {memory_context}
        
        Rewrite the task by replacing vague personal references with the specific information provided.
        Keep the task structure the same, just make it more concrete.
        
        Enhanced task:
        """
        
        # Use the same model as the agent actor for consistency
        actor_model = getattr(self.config.agent.actor, 'model', 'gpt-4-turbo')
        model_lower = (actor_model or "").lower()
        is_qwen_thinking = model_lower.strip() == 'qwen3-next-80b-a3b-thinking'
        # Force output format only for Qwen thinking model to separate reasoning trace
        if is_qwen_thinking:
            prompt += """
            
            At the very end of your response, output the following section:
            enhanced query:
            [the entire rewritten task content, possibly multiple lines]
            Only include the enhanced content after the header above. Do not include any content after this section.
            """
        if is_qwen_thinking:
            raw_response = call_qwen(prompt, model_id=actor_model, temperature=0.7)
            # Extract only enhanced query block (can be multiple lines) from Qwen response
            import re
            text = (raw_response or "").strip()
            lines = text.splitlines()
            enhanced_block = None
            # Extract entire block after last appearing enhanced query: header
            header_pattern = re.compile(r"^\s*enhanced\s*query\s*:\s*(.*)$", re.IGNORECASE)
            for idx in range(len(lines) - 1, -1, -1):
                m = header_pattern.match(lines[idx])
                if m:
                    first_content = m.group(1).rstrip()
                    following = lines[idx + 1:]
                    candidate = "\n".join([c for c in ([first_content] + following) if c is not None])
                    enhanced_block = candidate.strip()
                    break
            def strip_fences(s: str) -> str:
                if not s:
                    return s
                parts = s.splitlines()
                if parts and parts[0].lstrip().startswith("```"):
                    if len(parts) >= 2 and parts[-1].lstrip().startswith("```"):
                        parts = parts[1:-1]
                    else:
                        parts = parts[1:]
                return "\n".join(parts).strip()
            if enhanced_block is None:
                # Fallback 1: entire block after "Enhanced task:" header
                task_header = re.compile(r"^\s*Enhanced\s*task\s*:\s*(.*)$", re.IGNORECASE)
                for idx in range(len(lines) - 1, -1, -1):
                    m = task_header.match(lines[idx])
                    if m:
                        first_content = m.group(1).rstrip()
                        following = lines[idx + 1:]
                        candidate = "\n".join([c for c in ([first_content] + following) if c is not None])
                        enhanced_block = candidate.strip()
                        break
            enhanced = strip_fences(enhanced_block) if enhanced_block is not None else text
        elif "qwen" in model_lower:
            # Other Qwen models like Qwen instruct use as is without special parsing
            enhanced = call_qwen(prompt, model_id=actor_model, temperature=0.7)
        elif "gemini" in model_lower:
            # Gemini family is called with Gemini provider
            enhanced = call_gemini(prompt=prompt, model_id=actor_model, temperature=0.7)
        elif "llama" in model_lower:
            # Gemini family is called with Gemini provider
            enhanced = call_llama(prompt=prompt, model_id=actor_model, temperature=0.7)
        else:
            enhanced = call_openai(prompt, model_id=actor_model, temperature=0.7)
        return enhanced.strip()
    
    def get_trajectory(self):
        """Return trajectory"""
        if self.agent_occam:
            return self.agent_occam.get_trajectory()
        return []
    
    def get_memory_indices(self):
        """Return memory indices"""
        # pre-execution: return indices collected in this class
        if self.memory_mode == 'pre-execution':
            indices = []
            for event in self.memory_events:
                indices.extend(event.get('retrieved_indices', []))
            return list(set(indices))
        
        # on-demand: collected from internal AgentOccam
        if self.agent_occam and hasattr(self.agent_occam, 'actor'):
            return list(set(getattr(self.agent_occam.actor, 'accessed_memory_indices', [])))
        return []

    def get_memory_details(self):
        """Return memory detail events (common format for both modes)
        Each item: query, retrieved_indices, answer
        """
        # pre-execution: return events collected in this class
        if self.memory_mode == 'pre-execution':
            return list(self.memory_events)
        
        # on-demand: collected from internal AgentOccam
        if self.agent_occam and hasattr(self.agent_occam, 'get_memory_details'):
            return self.agent_occam.get_memory_details()
        
        # fallback: empty list
        return []