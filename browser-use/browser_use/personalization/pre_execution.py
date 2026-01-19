"""
PreExecutionService: Pre-execution memory processing for browser-use agents.

Flow:
1. Analyze task BEFORE execution to identify needed user information
2. Generate up to 5 specific memory queries
3. Retrieve relevant memories for each query
4. Rewrite the task with concrete user information (enhanced task)
5. Agent executes with enhanced task, no further memory access

This is an alternative to on-demand memory access during execution.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from browser_use.personalization.prompts import (
	PRE_EXECUTION_ANALYZE_SYSTEM_PROMPT,
	PRE_EXECUTION_ANALYZE_USER_PROMPT,
	PRE_EXECUTION_REWRITE_SYSTEM_PROMPT,
	PRE_EXECUTION_REWRITE_USER_PROMPT,
)

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel
	from browser_use.personalization.retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class PreExecutionService:
	"""Service for pre-execution memory processing."""

	def __init__(
		self,
		memory_bank_path: str | Path,
		llm: 'BaseChatModel',
		retriever: 'MemoryRetriever | None' = None,
		embedding_model: str = 'dunzhang/stella_en_1.5B_v5',
		retrieval_top_k: int = 20,
		retrieval_threshold: float = 0.5,
	):
		"""
		Initialize the PreExecutionService.

		Args:
			memory_bank_path: Path to the JSON file containing user memories
			llm: The LLM to use for analysis, query generation, and answer generation
			retriever: Optional pre-configured MemoryRetriever instance
			embedding_model: Name of the sentence-transformers model for embeddings
			retrieval_top_k: Maximum number of memories to retrieve per query
			retrieval_threshold: Minimum similarity score for retrieval
		"""
		self.memory_bank_path = Path(memory_bank_path)
		self.llm = llm
		self.retrieval_top_k = retrieval_top_k
		self.retrieval_threshold = retrieval_threshold
		self.logger = logging.getLogger(f'{__name__}.PreExecutionService')

		# Memory events log for trajectory recording
		self.memory_events: list[dict[str, Any]] = []

		# Initialize or use provided retriever
		if retriever is not None:
			self.retriever = retriever
		else:
			from browser_use.personalization.retriever import MemoryRetriever
			self.retriever = MemoryRetriever(
				memory_bank_path=self.memory_bank_path,
				embedding_model=embedding_model,
				llm=llm,
			)

	async def preprocess_task(self, task: str, original_task: str | None = None) -> tuple[str, list[dict[str, Any]]]:
		"""
		Preprocess task by analyzing and retrieving needed user information.

		Args:
			task: Full task with system_instruction for analysis (used for memory query generation)
			original_task: Original task without system_instruction (used for rewriting).
			              If None, uses task for both.

		Returns:
			Tuple of (enhanced_task, memory_events)
			- enhanced_task: Task with vague references replaced by concrete info (no system_instruction)
			- memory_events: List of memory retrieval events for trajectory logging
		"""
		self.memory_events = []  # Reset for this task
		# If original_task not provided, use task for both analysis and rewriting
		task_for_rewrite = original_task if original_task is not None else task

		try:
			# Step 1: Analyze task to identify needed memory queries (max 5)
			# Uses full task (with system_instruction) for better context
			self.logger.info('Analyzing task for required memory queries...')
			queries = await self._analyze_required_memory(task)

			if not queries:
				self.logger.info('No memory queries needed for this task')
				return task_for_rewrite, []

			self.logger.info(f'Generated {len(queries)} memory queries')

			# Step 2: Retrieve memories for each query
			self.logger.info('Retrieving memories for each query...')
			memories = await self._retrieve_memories(queries)

			if not memories:
				self.logger.warning('No relevant memories found')
				return task_for_rewrite, self.memory_events

			# Step 3: Rewrite original task (without system_instruction) with retrieved memory
			self.logger.info('Rewriting task with retrieved memory...')
			enhanced_task = await self._rewrite_task_with_memory(task_for_rewrite, memories)

			self.logger.info('Pre-execution processing complete')
			return enhanced_task, self.memory_events

		except Exception as e:
			self.logger.error(f'Error in pre-execution processing: {e}')
			import traceback
			traceback.print_exc()
			return task_for_rewrite, self.memory_events

	async def _analyze_required_memory(self, task: str) -> list[str]:
		"""
		Analyze task to identify what user-specific information is needed.

		Args:
			task: The task description

		Returns:
			List of specific memory queries (max 5)
		"""
		from browser_use.llm.messages import SystemMessage, UserMessage

		user_prompt = PRE_EXECUTION_ANALYZE_USER_PROMPT.format(task=task)

		messages = [
			SystemMessage(content=PRE_EXECUTION_ANALYZE_SYSTEM_PROMPT),
			UserMessage(content=user_prompt),
		]

		try:
			response = await self.llm.ainvoke(messages)
			response_text = response.completion if hasattr(response, 'completion') else str(response)
			self.logger.debug(f'Analysis response: {response_text}')

			return self._parse_queries_from_response(response_text)

		except Exception as e:
			self.logger.error(f'Failed to analyze required memory: {e}')
			return []

	def _parse_queries_from_response(self, response: str) -> list[str]:
		"""Parse memory queries from LLM response."""
		queries: list[str] = []
		lines = response.splitlines()

		# Check if NEEDS_MEMORY is NO
		needs_match = re.search(r"NEEDS_MEMORY\s*:\s*(YES|NO)", response, re.IGNORECASE)
		if needs_match and needs_match.group(1).upper() == 'NO':
			self.logger.debug('NEEDS_MEMORY is NO, returning empty list')
			return []

		# Check for explicit NONE in QUERIES section
		if re.search(r"^\s*QUERIES?\s*:\s*(NONE|N/A)\s*$", response, re.IGNORECASE | re.MULTILINE):
			self.logger.debug('Found explicit NONE in QUERIES section')
			return []

		# Parse queries from bullet points or QUERY: format
		in_queries_section = False
		for line in lines:
			# Check for QUERIES section header
			if re.match(r"^\s*QUERIES\s*:\s*$", line, re.IGNORECASE):
				in_queries_section = True
				continue

			# Check for new section (ends queries section)
			if re.match(r"^\s*[A-Z_]+\s*:", line) and not re.match(r"^\s*QUER", line, re.IGNORECASE):
				in_queries_section = False

			# Parse bullet points
			bullet = re.match(r"^\s*[-•*]\s*(.+)$", line)
			if bullet:
				query = bullet.group(1).strip()
				# Skip example lines
				if query and not query.lower().startswith(('example', 'e.g.', 'for ')):
					queries.append(query)
					self.logger.debug(f'Found bullet query: {query}')
				continue

			# Parse single QUERY: format
			single = re.match(r"^\s*QUERY\s*:\s*(.+)$", line, re.IGNORECASE)
			if single:
				query = single.group(1).strip()
				if query and query.upper() not in ['NONE', 'N/A']:
					queries.append(query)
					self.logger.debug(f'Found single query: {query}')

		# Cleanup: remove duplicates, empty strings, limit to 5
		queries = [q for q in (q.strip() for q in queries) if q]
		queries = list(dict.fromkeys(queries))  # Remove duplicates preserving order

		if len(queries) > 5:
			self.logger.warning(f'{len(queries)} queries found, limiting to 5')
			queries = queries[:5]

		self.logger.info(f'Parsed {len(queries)} memory queries')
		return queries

	async def _retrieve_memories(self, queries: list[str]) -> dict[str, str]:
		"""
		Retrieve memories for each query.

		Args:
			queries: List of memory queries

		Returns:
			Dict mapping query to generated answer
		"""
		all_memories: dict[str, str] = {}

		for query in queries:
			self.logger.debug(f'Retrieving memories for: {query}')

			try:
				# Retrieve relevant memories
				memories, retrieved_indices = self.retriever.retrieve(
					query=query,
					top_k=self.retrieval_top_k,
					threshold=self.retrieval_threshold,
				)

				if not memories:
					self.logger.debug(f'No memories found for query: {query}')
					continue

				# Generate answer from memories
				generated_answer = await self.retriever.generate_answer(
					query=query,
					memories=memories,
				)

				all_memories[query] = generated_answer
				self.logger.info(f'Retrieved {len(memories)} memories for: {query}')

				# Record event for trajectory
				self.memory_events.append({
					'query': query,
					'retrieved_indices': retrieved_indices,
					'answer': generated_answer,
				})

			except Exception as e:
				self.logger.error(f'Error retrieving memories for query "{query}": {e}')
				continue

		return all_memories

	async def _rewrite_task_with_memory(self, task: str, memories: dict[str, str]) -> str:
		"""
		Rewrite task by replacing vague references with concrete information.

		Args:
			task: The original task
			memories: Dict mapping query to answer

		Returns:
			Enhanced task with concrete user information
		"""
		if not memories:
			return task

		from browser_use.llm.messages import SystemMessage, UserMessage

		# Format memory context
		memory_context = "\n".join([
			f"- {query}: {answer}"
			for query, answer in memories.items()
		])

		user_prompt = PRE_EXECUTION_REWRITE_USER_PROMPT.format(
			task=task,
			memory_context=memory_context,
		)

		messages = [
			SystemMessage(content=PRE_EXECUTION_REWRITE_SYSTEM_PROMPT),
			UserMessage(content=user_prompt),
		]

		try:
			response = await self.llm.ainvoke(messages)
			enhanced_task = response.completion if hasattr(response, 'completion') else str(response)
			return enhanced_task.strip()

		except Exception as e:
			self.logger.error(f'Failed to rewrite task: {e}')
			return task

	def get_memory_events(self) -> list[dict[str, Any]]:
		"""Get the memory events log for trajectory recording."""
		return self.memory_events

	def get_retrieved_indices(self) -> list[int]:
		"""Get all retrieved memory indices across all queries."""
		indices: list[int] = []
		for event in self.memory_events:
			indices.extend(event.get('retrieved_indices', []))
		return list(set(indices))

	def cleanup(self) -> None:
		"""
		Free memory by unloading the embedding model.
		Call this after preprocess_task() and before starting the browser.
		"""
		if self.retriever is not None:
			self.retriever.unload_model()
			self.logger.info('PreExecutionService cleanup complete')
