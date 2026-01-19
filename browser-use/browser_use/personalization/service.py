"""
PersonalizationService: Handles user context retrieval for ambiguous tasks.

Flow (combined mode - single LLM call):
1. Check if memory is needed AND generate query in one call
2. Retrieve relevant memories using embedding-based search (MemoryRetriever)
3. Generate answer from retrieved memories (LLM)
4. Return the context to be included in the agent's prompt

Legacy Flow (two LLM calls):
1. Check if the current step needs user context (LLM-based ambiguity detection)
2. If needed, generate a retrieval query (LLM)
3. Retrieve relevant memories using embedding-based search (MemoryRetriever)
4. Generate answer from retrieved memories (LLM)
5. Return the context to be included in the agent's prompt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from browser_use.personalization.views import (
	AmbiguityCheckResult,
	MemoryCheckResult,
	PersonalizationContext,
	RetrievalQuery,
)
from browser_use.personalization.prompts import (
	AMBIGUITY_CHECK_SYSTEM_PROMPT,
	AMBIGUITY_CHECK_USER_PROMPT,
	MEMORY_CHECK_SYSTEM_PROMPT,
	MEMORY_CHECK_USER_PROMPT,
	RETRIEVAL_QUERY_SYSTEM_PROMPT,
	RETRIEVAL_QUERY_USER_PROMPT,
)

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel
	from browser_use.personalization.retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class PersonalizationService:
	"""Service for handling user-specific context and ambiguity resolution."""

	def __init__(
		self,
		memory_bank_path: str | Path,
		llm: 'BaseChatModel',
		retriever: 'MemoryRetriever | None' = None,
		embedding_model: str = 'dunzhang/stella_en_1.5B_v5',
		enabled: bool = True,
		retrieval_top_k: int = 20,
		retrieval_threshold: float = 0.5,
		use_combined_prompt: bool = True,
	):
		"""
		Initialize the PersonalizationService.

		Args:
			memory_bank_path: Path to the JSON file containing user memories
			llm: The LLM to use for ambiguity detection, query generation, and answer generation
			retriever: Optional pre-configured MemoryRetriever instance
			embedding_model: Name of the sentence-transformers model for embeddings
			enabled: Whether personalization is enabled
			retrieval_top_k: Maximum number of memories to retrieve (default: 20)
			retrieval_threshold: Minimum similarity score for retrieval (default: 0.5)
			use_combined_prompt: Use combined prompt (single LLM call) instead of separate calls
		"""
		self.memory_bank_path = Path(memory_bank_path)
		self.llm = llm
		self.enabled = enabled
		self.retrieval_top_k = retrieval_top_k
		self.retrieval_threshold = retrieval_threshold
		self.use_combined_prompt = use_combined_prompt
		self.logger = logging.getLogger(f'{__name__}.PersonalizationService')

		# Cache for previously retrieved memories (full context for Agent's action decisions)
		# Each entry: {'query': str, 'answer': str}
		self.memory_context_cache: list[dict[str, str]] = []

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

		# Note: Model pre-loading moved to first retrieve() call to avoid
		# blocking issues during Agent initialization in asyncio context

	async def process_step(
		self,
		task: str,
		current_url: str,
		page_title: str,
		page_content: str = '',
		step: int = 0,
		interaction_history: str = '',
		next_action_description: str | None = None,
	) -> PersonalizationContext:
		"""
		Process a step to determine if user context is needed and retrieve it.

		Args:
			task: The user's task/request
			current_url: Current browser URL
			page_title: Current page title
			page_content: Current page content (for combined prompt)
			step: Current step number
			interaction_history: History of previous interactions
			next_action_description: Description of the next planned action (from previous step)

		Returns:
			PersonalizationContext with retrieved user context if needed
		"""
		if not self.enabled:
			return PersonalizationContext(was_ambiguous=False)

		# Check if retriever is initialized
		if self.retriever.index is None or not self.retriever.memory_items:
			self.logger.warning('Memory retriever not initialized, skipping personalization')
			return PersonalizationContext(was_ambiguous=False)

		if self.use_combined_prompt:
			return await self._process_step_combined(
				task=task,
				url=current_url,
				title=page_title,
				page_content=page_content,
				step=step,
				interaction_history=interaction_history,
				next_action_description=next_action_description,
			)
		else:
			return await self._process_step_legacy(
				task=task,
				url=current_url,
				title=page_title,
			)

	async def _process_step_combined(
		self,
		task: str,
		url: str,
		title: str,
		page_content: str,
		step: int,
		interaction_history: str,
		next_action_description: str | None = None,
	) -> PersonalizationContext:
		"""Process step using combined prompt (single LLM call for check + query)."""
		# Step 1: Check memory need AND generate query in one call
		memory_check = await self._check_memory_need(
			task=task,
			url=url,
			title=title,
			page_content=page_content,
			step=step,
			interaction_history=interaction_history,
			next_action_description=next_action_description,
		)

		if not memory_check.need_memory:
			self.logger.debug(f'No memory needed: {memory_check.reason}')
			return PersonalizationContext(was_ambiguous=False)

		if not memory_check.query_detail:
			self.logger.warning('Memory needed but no query generated')
			return PersonalizationContext(was_ambiguous=True)

		self.logger.info(f'Memory needed: {memory_check.reason}')
		self.logger.info(f'Query: {memory_check.query_detail}')

		# Step 2: Retrieve relevant memories using embedding-based search
		try:
			memories, retrieved_indices = self.retriever.retrieve(
				query=memory_check.query_detail,
				top_k=self.retrieval_top_k,
				threshold=self.retrieval_threshold,
			)
		except Exception as e:
			self.logger.error(f'Failed to retrieve memories: {e}')
			return PersonalizationContext(
				was_ambiguous=True,
				retrieval_query=memory_check.query_detail,
			)

		if not memories:
			self.logger.warning('No relevant memories found for the query')
			return PersonalizationContext(
				was_ambiguous=True,
				retrieval_query=memory_check.query_detail,
			)

		self.logger.info(f'Retrieved {len(memories)} memories')

		# Step 3: Generate answer from retrieved memories (LLM call)
		generated_answer = await self.retriever.generate_answer(
			query=memory_check.query_detail,
			memories=memories,
		)

		# Add to cache for future reference (query + answer pair)
		self.memory_context_cache.append({
			'query': memory_check.query_detail,
			'answer': generated_answer,
		})

		# Step 4: Format as context
		context_text = self._format_context(
			query=memory_check.query_detail,
			memories=memories,
			generated_answer=generated_answer,
		)

		return PersonalizationContext(
			was_ambiguous=True,
			retrieved_context=context_text,
			retrieval_query=memory_check.query_detail,
			source_entries=[str(idx) for idx in retrieved_indices],
			current_observation=memory_check.current_observation,
			memory_reason=memory_check.reason,
		)

	async def _process_step_legacy(
		self,
		task: str,
		url: str,
		title: str,
	) -> PersonalizationContext:
		"""Process step using legacy prompts (two separate LLM calls)."""
		# Step 1: Check if user context is needed (LLM call)
		ambiguity_result = await self._check_ambiguity(
			task=task,
			url=url,
			title=title,
			next_action="Not yet determined",
		)

		if not ambiguity_result.needs_user_context:
			self.logger.debug(f'No user context needed: {ambiguity_result.reasoning}')
			return PersonalizationContext(was_ambiguous=False)

		self.logger.info(f'User context needed: {ambiguity_result.reasoning}')
		self.logger.debug(f'Ambiguous elements: {ambiguity_result.ambiguous_elements}')

		# Step 2: Generate retrieval query (LLM call)
		retrieval_query = await self._generate_retrieval_query(
			task=task,
			ambiguous_elements=ambiguity_result.ambiguous_elements,
			suggested_categories=ambiguity_result.suggested_categories,
		)

		# Step 3: Retrieve relevant memories using embedding-based search
		try:
			memories, retrieved_indices = self.retriever.retrieve(
				query=retrieval_query.query,
				top_k=self.retrieval_top_k,
				threshold=self.retrieval_threshold,
			)
		except Exception as e:
			self.logger.error(f'Failed to retrieve memories: {e}')
			return PersonalizationContext(
				was_ambiguous=True,
				retrieval_query=retrieval_query.query,
			)

		if not memories:
			self.logger.warning('No relevant memories found for the query')
			return PersonalizationContext(
				was_ambiguous=True,
				retrieval_query=retrieval_query.query,
			)

		self.logger.info(f'Retrieved {len(memories)} memories for query: {retrieval_query.query}')

		# Step 4: Generate answer from retrieved memories (LLM call)
		generated_answer = await self.retriever.generate_answer(
			query=retrieval_query.query,
			memories=memories,
		)

		# Add to cache for future reference (query + answer pair)
		self.memory_context_cache.append({
			'query': retrieval_query.query,
			'answer': generated_answer,
		})

		# Step 5: Format as context
		context_text = self._format_context(
			query=retrieval_query.query,
			memories=memories,
			generated_answer=generated_answer,
		)

		return PersonalizationContext(
			was_ambiguous=True,
			retrieved_context=context_text,
			retrieval_query=retrieval_query.query,
			source_entries=[str(idx) for idx in retrieved_indices],
		)

	def _generate_cached_info(self) -> str:
		"""Generate summarized cached memories for memory-need detection prompt.

		For memory check LLM, we provide a condensed view:
		- Query: full text (to know what was asked)
		- Answer: first 200 + last 200 chars (to get key info without overwhelming context)

		The full content remains in memory_context_cache for actual action generation.

		Returns:
			Summarized cached query/answer pairs string for the memory check prompt
		"""
		if not self.memory_context_cache:
			return "None"

		cached_items = []
		for entry in self.memory_context_cache:
			query = entry.get('query', '')
			answer = entry.get('answer', '')

			# Truncate answer: first 200 + ... + last 200 chars
			if len(answer) > 450:  # 200 + 50 buffer + 200
				truncated_answer = f"{answer[:200]} ... {answer[-200:]}"
			else:
				truncated_answer = answer

			cached_items.append(f"Q: {query}\nA: {truncated_answer}")

		return "\n\n".join(cached_items)

	async def _check_memory_need(
		self,
		task: str,
		url: str,
		title: str,
		page_content: str,
		step: int,
		interaction_history: str,
		next_action_description: str | None = None,
	) -> MemoryCheckResult:
		"""Check if memory is needed and generate query in a single LLM call."""
		from browser_use.llm.messages import SystemMessage, UserMessage

		# Generate truncated cached info for memory-need detection
		cached_info = self._generate_cached_info()

		user_prompt = MEMORY_CHECK_USER_PROMPT.format(
			task=task,
			url=url,
			title=title,
			page_content=page_content[:2000] if page_content else "No content available",
			step=step,
			next_action=next_action_description or "Not yet determined",
			interaction_history=interaction_history or "No history yet",
			cached_memory=cached_info,
		)

		messages = [
			SystemMessage(content=MEMORY_CHECK_SYSTEM_PROMPT),
			UserMessage(content=user_prompt),
		]

		try:
			response = await self.llm.ainvoke(messages, output_format=MemoryCheckResult)
			return response.completion  # type: ignore
		except Exception as e:
			self.logger.error(f'Failed to check memory need: {e}')
			return MemoryCheckResult(
				current_observation="Error during observation",
				need_memory=False,
				query_detail=None,
				reason=f"Error during memory check: {e}",
			)

	async def _check_ambiguity(
		self,
		task: str,
		url: str,
		title: str,
		next_action: str,
	) -> AmbiguityCheckResult:
		"""Check if the current step requires user context."""
		from browser_use.llm.messages import SystemMessage, UserMessage

		# Get available categories from memory items
		categories = list(set(
			item.category for item in self.retriever.memory_items
		)) if self.retriever.memory_items else []

		user_prompt = AMBIGUITY_CHECK_USER_PROMPT.format(
			task=task,
			url=url,
			title=title,
			next_action=next_action,
			memory_categories=", ".join(categories) if categories else "No categories available",
		)

		messages = [
			SystemMessage(content=AMBIGUITY_CHECK_SYSTEM_PROMPT),
			UserMessage(content=user_prompt),
		]

		try:
			response = await self.llm.ainvoke(messages, output_format=AmbiguityCheckResult)
			return response.completion  # type: ignore
		except Exception as e:
			self.logger.error(f'Failed to check ambiguity: {e}')
			return AmbiguityCheckResult(
				needs_user_context=False,
				reasoning=f"Error during ambiguity check: {e}",
			)

	async def _generate_retrieval_query(
		self,
		task: str,
		ambiguous_elements: list[str],
		suggested_categories: list[str],
	) -> RetrievalQuery:
		"""Generate a retrieval query for the memory bank."""
		from browser_use.llm.messages import SystemMessage, UserMessage

		user_prompt = RETRIEVAL_QUERY_USER_PROMPT.format(
			task=task,
			ambiguous_elements=", ".join(ambiguous_elements) if ambiguous_elements else "None specified",
			suggested_categories=", ".join(suggested_categories) if suggested_categories else "All categories",
		)

		messages = [
			SystemMessage(content=RETRIEVAL_QUERY_SYSTEM_PROMPT),
			UserMessage(content=user_prompt),
		]

		try:
			response = await self.llm.ainvoke(messages, output_format=RetrievalQuery)
			result = response.completion  # type: ignore
			if not hasattr(result, 'max_results') or result.max_results is None:
				result.max_results = self.retrieval_top_k
			return result
		except Exception as e:
			self.logger.error(f'Failed to generate retrieval query: {e}')
			return RetrievalQuery(
				query=" ".join(ambiguous_elements) if ambiguous_elements else task,
				categories=suggested_categories,
				max_results=self.retrieval_top_k,
			)

	def _format_context(
		self,
		query: str,
		memories: list[dict[str, Any]],
		generated_answer: str,
	) -> str:
		"""Format retrieved memories and generated answer as context text."""
		lines = ["<user_context>"]
		lines.append(f"Query: {query}")
		lines.append("")
		lines.append("Retrieved user memories:")

		for mem in memories:
			timestamp = mem.get('timestamp', '')
			mem_type = mem.get('type', 'unknown')
			obj = mem.get('object', '')
			website = mem.get('website', '')
			score = mem.get('similarity_score', 0)

			lines.append(f"- [{timestamp}] [{mem_type}] {obj} (website: {website}, relevance: {score:.2f})")

		lines.append("")
		lines.append("Summary:")
		lines.append(generated_answer)
		lines.append("</user_context>")

		return "\n".join(lines)

	def get_memory_summary(self) -> str:
		"""Get a summary of the memory bank for debugging."""
		if not self.retriever.memory_items:
			return "Memory bank is empty or not loaded."

		types: dict[str, int] = {}
		for item in self.retriever.memory_items:
			mem_type = item.type
			types[mem_type] = types.get(mem_type, 0) + 1

		summary_lines = [f"Memory Bank ({self.memory_bank_path}):"]
		for mem_type, count in types.items():
			summary_lines.append(f"  - {mem_type}: {count} entries")
		summary_lines.append(f"Total: {len(self.retriever.memory_items)} entries")

		return "\n".join(summary_lines)

	def get_cached_context(self) -> str | None:
		"""Get the full cached memory context for Agent's action decisions.

		Returns:
			Formatted user_context string with all cached query/answer pairs, or None if empty
		"""
		if not self.memory_context_cache:
			return None

		lines = ["<user_context>"]
		lines.append("Previously retrieved user information:")
		lines.append("")

		for entry in self.memory_context_cache:
			query = entry.get('query', '')
			answer = entry.get('answer', '')
			lines.append(f"Q: {query}")
			lines.append(f"A: {answer}")
			lines.append("")

		lines.append("</user_context>")
		return "\n".join(lines)
