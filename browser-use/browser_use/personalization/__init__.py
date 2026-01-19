"""
Personalization module for Browser-Use.

This module provides user-specific memory bank and ambiguity resolution
to handle vague tasks by retrieving relevant user context.

Flow:
1. Ambiguity Detection (LLM) - Check if user context is needed
2. Query Generation (LLM) - Generate retrieval query
3. Memory Retrieval (Embedding) - Search memory bank using FAISS
4. Answer Generation (LLM) - Summarize retrieved memories
"""

from browser_use.personalization.views import (
	AmbiguityCheckResult,
	MemoryCheckResult,
	RetrievalQuery,
	PersonalizationContext,
)
from browser_use.personalization.service import PersonalizationService
from browser_use.personalization.retriever import MemoryRetriever, MemoryItem

__all__ = [
	'AmbiguityCheckResult',
	'MemoryCheckResult',
	'RetrievalQuery',
	'PersonalizationContext',
	'PersonalizationService',
	'MemoryRetriever',
	'MemoryItem',
]
