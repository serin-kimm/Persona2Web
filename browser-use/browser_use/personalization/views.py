"""
Pydantic models for the personalization module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AmbiguityCheckResult(BaseModel):
	"""Result of checking if user context is needed for the current step."""

	model_config = ConfigDict(extra='forbid')

	needs_user_context: bool = Field(
		description='Whether user-specific context is needed to proceed'
	)
	reasoning: str = Field(
		description='Explanation of why user context is or is not needed'
	)
	ambiguous_elements: list[str] = Field(
		default_factory=list,
		description='List of ambiguous elements that need clarification'
	)
	suggested_categories: list[str] = Field(
		default_factory=list,
		description='Suggested memory categories to search (e.g., preferences, personal_info)'
	)


class MemoryCheckResult(BaseModel):
	"""Combined result for memory need detection and query generation (single LLM call)."""

	model_config = ConfigDict(extra='forbid')

	current_observation: str = Field(
		description='Description of current page state relevant to the task'
	)
	need_memory: bool = Field(
		description='Whether user memory is needed for this action'
	)
	query_detail: str | None = Field(
		default=None,
		description='Specific memory query if need_memory is True, otherwise N/A'
	)
	reason: str = Field(
		description='Reason why memory is needed or why cached info is sufficient'
	)


class RetrievalQuery(BaseModel):
	"""Query for retrieving relevant memories from the memory bank."""

	model_config = ConfigDict(extra='forbid')

	query: str = Field(description='The retrieval query text')
	categories: list[str] = Field(
		default_factory=list,
		description='Categories to search in (empty means all categories)'
	)
	max_results: int = Field(default=5, description='Maximum number of results to return')


class PersonalizationContext(BaseModel):
	"""Context derived from personalization for the current step."""

	model_config = ConfigDict(extra='forbid')

	was_ambiguous: bool = Field(default=False)
	retrieved_context: str | None = Field(
		default=None,
		description='Retrieved user context formatted as text'
	)
	retrieval_query: str | None = Field(
		default=None,
		description='Query used for retrieval'
	)
	source_entries: list[str] = Field(
		default_factory=list,
		description='Indices of memory entries used'
	)
	current_observation: str | None = Field(
		default=None,
		description='LLM observation of current page state (from MemoryCheckResult)'
	)
	memory_reason: str | None = Field(
		default=None,
		description='LLM reasoning for why memory was needed (from MemoryCheckResult)'
	)
