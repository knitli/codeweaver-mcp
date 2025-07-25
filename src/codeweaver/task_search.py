# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Task tool integration for comprehensive and uncertain-scope searches.

Provides enhanced search methods that delegate to the Task tool when:
- Search scope is uncertain or very broad
- Query complexity requires multi-step analysis
- Comprehensive cross-file pattern matching is needed
"""

import logging
import typing

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from codeweaver._types.base_enum import BaseEnum

logger = logging.getLogger(__name__)


class SearchComplexity(BaseEnum):
    """Categorize search complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    UNCERTAIN = "uncertain"


@dataclass
class SearchAssessment:
    """Assessment of search query complexity and scope."""

    complexity: SearchComplexity
    confidence: float
    estimated_scope: str
    should_use_task: bool
    reasoning: str


class TaskSearchCoordinator:
    """Coordinates search operations with Task tool delegation for comprehensive searches."""

    # Thresholds for Task tool delegation
    COMPLEXITY_THRESHOLD = 0.7
    UNCERTAINTY_THRESHOLD = 0.6
    FILE_COUNT_THRESHOLD = 100

    # Keywords that indicate broad/uncertain scope
    BROAD_KEYWORDS: ClassVar[set[str]] = {
        "all",
        "every",
        "any",
        "anywhere",
        "everywhere",
        "entire",
        "whole",
        "throughout",
        "comprehensive",
        "exhaustive",
        "complete",
        "full",
    }

    # Keywords that indicate complex patterns
    COMPLEX_KEYWORDS: ClassVar[set[str]] = {
        "pattern",
        "similar",
        "like",
        "related",
        "associated",
        "connected",
        "dependent",
        "reference",
        "usage",
        "implementation",
        "example",
    }

    def __init__(self, *, task_tool_available: bool = True):
        """Initialize the task search coordinator.

        Args:
            task_tool_available: Whether Task tool delegation is available for complex searches
        """
        self.task_tool_available = task_tool_available

    def assess_search_complexity(
        self,
        query: str,
        file_filter: str | None = None,
        language_filter: str | None = None,
        estimated_files: int = 0,
    ) -> SearchAssessment:
        """Assess whether a search query should use Task tool delegation."""
        query_indicators = self._analyze_query_indicators(query)
        filter_specificity = self._analyze_filter_specificity(file_filter, language_filter)
        complexity_score = self._calculate_complexity_score(
            query_indicators, filter_specificity, estimated_files, len(query)
        )

        complexity = self._determine_complexity_level(complexity_score)
        confidence = 1.0 - (complexity_score * 0.3)
        estimated_scope = self._determine_search_scope(filter_specificity)
        should_use_task = self._should_use_task_tool(
            complexity_score, confidence, query_indicators.has_broad_scope, estimated_files
        )
        reasoning = self._build_reasoning(query_indicators, filter_specificity, estimated_files)

        return SearchAssessment(
            complexity=complexity,
            confidence=confidence,
            estimated_scope=estimated_scope,
            should_use_task=should_use_task,
            reasoning=reasoning,
        )

    def _analyze_query_indicators(self, query: str) -> tuple:
        """Analyze query for complexity indicators."""

        class QueryIndicators(typing.NamedTuple):
            """Indicators of query complexity."""
            has_broad_scope: bool
            has_complex_pattern: bool

        query_lower = query.lower()
        words = set(query_lower.split())

        has_broad_scope = bool(words & self.BROAD_KEYWORDS)
        has_complex_pattern = bool(words & self.COMPLEX_KEYWORDS)

        return QueryIndicators(has_broad_scope, has_complex_pattern)

    def _analyze_filter_specificity(
        self, file_filter: str | None, language_filter: str | None
    ) -> tuple:
        """Analyze filter specificity."""

        class FilterSpecificity(typing.NamedTuple):
            """Specificity of search filters."""
            has_specific_filters: bool
            is_path_specific: bool

        has_specific_filters = bool(file_filter and language_filter)
        is_path_specific = bool(file_filter and "/" in file_filter)

        return FilterSpecificity(has_specific_filters, is_path_specific)

    def _calculate_complexity_score(
        self, query_indicators, filter_specificity, estimated_files: int, query_length: int
    ) -> float:
        """Calculate complexity score based on various factors."""
        complexity_score = 0.0

        if query_indicators.has_broad_scope:
            complexity_score += 0.4
        if query_indicators.has_complex_pattern:
            complexity_score += 0.3
        if not filter_specificity.has_specific_filters:
            complexity_score += 0.2
        if estimated_files > self.FILE_COUNT_THRESHOLD:
            complexity_score += 0.3
        if query_length < 10:  # Very short queries are often too vague
            complexity_score += 0.2

        return complexity_score

    def _determine_complexity_level(self, complexity_score: float) -> SearchComplexity:
        """Determine complexity level from score."""
        if complexity_score >= 0.8:
            return SearchComplexity.COMPLEX
        if complexity_score >= 0.5:
            return SearchComplexity.MODERATE
        return SearchComplexity.SIMPLE

    def _determine_search_scope(self, filter_specificity) -> str:
        """Determine estimated search scope."""
        if filter_specificity.is_path_specific:
            return f"specific path: {filter_specificity.file_filter}"
        if filter_specificity.has_specific_filters:
            return "filtered search"
        return "entire codebase"

    def _should_use_task_tool(
        self,
        complexity_score: float,
        confidence: float,
        *,
        has_broad_scope: bool,
        estimated_files: int,
    ) -> bool:
        """Determine if Task tool should be used."""
        return self.task_tool_available and (
            complexity_score >= self.COMPLEXITY_THRESHOLD
            or confidence < self.UNCERTAINTY_THRESHOLD
            or (has_broad_scope and estimated_files > 50)
        )

    def _build_reasoning(self, query_indicators, filter_specificity, estimated_files: int) -> str:
        """Build reasoning string for the assessment."""
        reasoning_parts = []

        if query_indicators.has_broad_scope:
            reasoning_parts.append("broad scope indicators detected")
        if query_indicators.has_complex_pattern:
            reasoning_parts.append("complex pattern matching required")
        if not filter_specificity.has_specific_filters:
            reasoning_parts.append("no specific filters provided")
        if estimated_files > self.FILE_COUNT_THRESHOLD:
            reasoning_parts.append(f"large file count ({estimated_files})")

        return "; ".join(reasoning_parts) or "straightforward search"

    def create_task_prompt_for_semantic_search(
        self,
        query: str,
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None,
        chunk_type_filter: str | None = None,
        *,
        rerank: bool = True,
    ) -> str:
        """Create a comprehensive Task tool prompt for semantic search."""
        prompt_parts = [
            f"Perform a comprehensive semantic code search for: '{query}'",
            "",
            "Search Requirements:",
            f"- Find up to {limit} most relevant code chunks",
            "- Use natural language understanding to match intent, not just keywords",
        ]

        if file_filter:
            prompt_parts.append(f"- Focus on files matching path: {file_filter}")
        if language_filter:
            prompt_parts.append(f"- Only search {language_filter} files")
        if chunk_type_filter:
            prompt_parts.append(f"- Prioritize {chunk_type_filter} code structures")

        prompt_parts.extend([
            "",
            "Search Strategy:",
            "1. First, understand the conceptual meaning of the query",
            "2. Identify related terms, synonyms, and implementation patterns",
            "3. Search broadly to ensure comprehensive coverage",
            "4. Consider both direct matches and conceptually related code",
        ])

        if rerank:
            prompt_parts.append("5. Rank results by relevance to the original query intent")

        prompt_parts.extend([
            "",
            "Return results in this format:",
            "- File path and location (line numbers)",
            "- Relevant code snippet",
            "- Brief explanation of why it matches the query",
            "- Confidence score (high/medium/low)",
        ])

        return "\n".join(prompt_parts)

    def create_task_prompt_for_structural_search(
        self, pattern: str, language: str, root_path: str, limit: int = 20
    ) -> str:
        """Create a comprehensive Task tool prompt for structural search."""
        return f"""Perform a comprehensive structural code search using this ast-grep pattern:

Pattern: {pattern}
Language: {language}
Root Path: {root_path}
Maximum Results: {limit}

Search Requirements:
1. Find ALL instances of this structural pattern across the codebase
2. Use ast-grep or equivalent tree-sitter parsing for accurate matching
3. Search recursively through all {language} files
4. Include variations and similar patterns that might be relevant

Pattern Interpretation:
- $_ matches any single node
- $$_ matches any sequence of nodes
- Named captures like $VAR should be noted in results

Search Strategy:
1. Parse the pattern to understand the code structure being sought
2. Identify all {language} files in the codebase
3. Parse each file's AST and match against the pattern
4. Consider related patterns that might be of interest
5. Group results by similarity or usage pattern

Return Format:
- File path with line and column numbers
- Matched code snippet with syntax highlighting
- Any captured variables from the pattern
- Context about how the code is used
- Suggestions for similar patterns found

Special Considerations:
- Handle large files efficiently
- Report if pattern syntax seems incorrect
- Suggest pattern improvements if applicable"""

    def create_exploratory_search_prompt(self, topic: str, context: str | None = None) -> str:
        """Create a Task prompt for exploratory/learning searches."""
        return f"""Help me understand '{topic}' in this codebase through comprehensive search and analysis.

Context: {context or "General exploration"}

Exploration Goals:
1. Find the main implementation(s) of {topic}
2. Identify usage patterns and examples
3. Discover related functionality and dependencies
4. Understand the architectural decisions around {topic}

Search Approach:
1. Start with semantic searches for core concepts
2. Use structural patterns to find implementations
3. Trace dependencies and relationships
4. Identify configuration and setup code
5. Find tests and documentation

Provide:
- Overview of how {topic} is implemented
- Key files and their roles
- Important code snippets with explanations
- Architectural insights and patterns used
- Suggestions for further exploration

Focus on providing a learning path through the code."""

    def should_delegate_to_task(
        self, assessment: SearchAssessment, *, user_preference: bool | None = None
    ) -> tuple[bool, str]:
        """Determine if search should be delegated to Task tool."""
        # User preference overrides
        if user_preference is not None:
            reason = "user preference" if user_preference else "user disabled delegation"
            return user_preference, reason

        # Check Task tool availability
        if not self.task_tool_available:
            return False, "Task tool not available"

        # Use assessment recommendation
        if assessment.should_use_task:
            return True, assessment.reasoning

        return False, "search is straightforward enough for direct execution"


def enhance_search_with_task_delegation(
    original_search_func: Callable,
    task_coordinator: TaskSearchCoordinator,
    task_tool_func: Callable | None = None,
) -> Callable:
    """Decorator to enhance search functions with Task tool delegation."""

    async def enhanced_search(*args: Any, **kwargs: Any) -> Any:
        # Extract search parameters
        query = args[0] if args else kwargs.get("query", "")
        file_filter = kwargs.get("file_filter")
        language_filter = kwargs.get("language_filter")

        # Assess complexity
        assessment = task_coordinator.assess_search_complexity(
            query=query, file_filter=file_filter, language_filter=language_filter
        )

        # Check if we should delegate
        should_delegate, reason = task_coordinator.should_delegate_to_task(assessment)

        if should_delegate and task_tool_func:
            logger.info("Delegating search to Task tool: %s", reason)

            # Create appropriate prompt based on search type
            if "pattern" in kwargs:  # Structural search
                prompt = task_coordinator.create_task_prompt_for_structural_search(
                    pattern=kwargs["pattern"],
                    language=kwargs["language"],
                    root_path=kwargs["root_path"],
                    limit=kwargs.get("limit", 20),
                )
            else:  # Semantic search
                prompt = task_coordinator.create_task_prompt_for_semantic_search(
                    query=query,
                    limit=kwargs.get("limit", 10),
                    file_filter=file_filter,
                    language_filter=language_filter,
                    chunk_type_filter=kwargs.get("chunk_type_filter"),
                    rerank=kwargs.get("rerank", True),
                )

            # Execute via Task tool
            return await task_tool_func(prompt)

        # Otherwise, use original search
        logger.info("Using direct search: %s", reason)
        return await original_search_func(*args, **kwargs)

    return enhanced_search
