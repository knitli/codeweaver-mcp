# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Simple search strategy for SEARCH intents."""

import logging

from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.cw_types import (
    Complexity,
    IntentResult,
    IntentStrategy,
    IntentType,
    ParsedIntent,
    ServiceType,
)
from codeweaver.cw_types.services.config import ServiceConfig


class SimpleSearchStrategy(BaseServiceProvider, IntentStrategy):
    """
    Simple search strategy for SEARCH intents.

    This strategy handles straightforward search requests by routing them
    to the existing search_code_handler. It's optimized for:
    - SEARCH intent type
    - SIMPLE to MODERATE complexity
    - High confidence queries

    The strategy leverages existing CodeWeaver tools while providing
    the intent layer abstraction for natural language queries.
    """

    def __init__(self, services_manager):
        """Initialize simple search strategy with service dependencies."""
        config = ServiceConfig(provider="simple_search_strategy")
        super().__init__(ServiceType.INTENT, config)
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.name = "simple_search_strategy"
        self.supported_intent_types = [IntentType.SEARCH]
        self.supported_complexity = [Complexity.SIMPLE, Complexity.MODERATE]

    async def _initialize_provider(self) -> None:
        """Initialize strategy with service dependencies."""

    async def _shutdown_provider(self) -> None:
        """Shutdown strategy resources."""

    async def _check_health(self) -> bool:
        """Check strategy health."""
        try:
            from codeweaver.server import search_code_handler  # noqa: F401
        except ImportError:
            self.logger.warning("search_code_handler not available")
            return False
        else:
            return True

    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """
        Check if strategy can handle the intent.

        Args:
            parsed_intent: The parsed intent to evaluate

        Returns:
            Confidence score (0.0-1.0) for handling this intent
        """
        score = 0.0
        if parsed_intent.intent_type == IntentType.SEARCH:
            score += 0.8
        else:
            return 0.0
        match parsed_intent.complexity:
            case Complexity.SIMPLE | Complexity.MODERATE:
                score += 0.15
            case Complexity.COMPLEX:
                score += 0.05
        if parsed_intent.confidence >= 0.7:
            score += 0.05
        self.logger.debug(
            "SimpleSearchStrategy can_handle score: %.2f for %s intent",
            score,
            parsed_intent.intent_type.value,
        )
        return min(1.0, score)

    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """
        Execute search using existing search_code_handler.

        Args:
            parsed_intent: The parsed intent to execute
            context: Service context with dependencies

        Returns:
            Result of the search execution
        """
        try:
            self.logger.info("Executing simple search for: %s", parsed_intent.primary_target)
            search_params = self._transform_intent_to_search(parsed_intent)
            search_result = await self._execute_search(search_params, context)
            intent_result = self._transform_search_to_intent_result(search_result, parsed_intent)
            self.logger.info("Simple search completed successfully")
        except Exception as e:
            self.logger.exception("Simple search execution failed")
            return IntentResult(
                success=False,
                data=None,
                metadata={
                    "strategy": self.name,
                    "intent_type": parsed_intent.intent_type.value,
                    "error_type": type(e).__name__,
                },
                error_message=f"Search execution failed: {e}",
                suggestions=[
                    "Try rephrasing your search query",
                    "Use more specific search terms",
                    "Check if the codebase is properly indexed",
                ],
                strategy_used=self.name,
            )
        else:
            return intent_result

    def _transform_intent_to_search(self, parsed_intent: ParsedIntent) -> dict[str, Any]:
        """Transform parsed intent to search_code_handler parameters."""
        search_params = {"query": parsed_intent.primary_target, "max_results": 20}
        if parsed_intent.filters:
            if "language" in parsed_intent.filters:
                search_params["language"] = parsed_intent.filters["language"]
            if "include_tests" in parsed_intent.filters:
                search_params["include_tests"] = parsed_intent.filters["include_tests"]
            if "recent_only" in parsed_intent.filters:
                search_params["recent_only"] = parsed_intent.filters["recent_only"]
        if parsed_intent.scope.value == "file":
            search_params["max_results"] = 10
        elif parsed_intent.scope.value == "system":
            search_params["max_results"] = 50
        if parsed_intent.complexity == Complexity.SIMPLE:
            search_params["max_results"] = min(search_params["max_results"], 15)
        self.logger.debug("Transformed intent to search params: %s", search_params)
        return search_params

    async def _execute_search(
        self, search_params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute search using existing search_code_handler."""
        try:
            from codeweaver.server import search_code_handler
        except ImportError:
            self.logger.warning("search_code_handler not available, using mock result")
            return {
                "results": [],
                "total_results": 0,
                "message": "Search handler not available - mock result",
                "query": search_params["query"],
            }
        else:
            return await search_code_handler(
                query=search_params["query"],
                max_results=search_params.get("max_results", 20),
                language=search_params.get("language"),
                include_tests=search_params.get("include_tests", True),
            )

    def _transform_search_to_intent_result(
        self, search_result: dict[str, Any], parsed_intent: ParsedIntent
    ) -> IntentResult:
        """Transform search result back to intent result format."""
        results = search_result.get("results", [])
        total_results = search_result.get("total_results", len(results))
        success = total_results > 0 or search_result.get("success", True)
        metadata = {
            "strategy": self.name,
            "intent_type": parsed_intent.intent_type.value,
            "original_query": parsed_intent.primary_target,
            "search_query": search_result.get("query", parsed_intent.primary_target),
            "total_results": total_results,
            "confidence": parsed_intent.confidence,
            "scope": parsed_intent.scope.value,
            "complexity": parsed_intent.complexity.value,
            "background_indexing_active": True,
        }
        suggestions = None
        if not success or total_results == 0:
            suggestions = [
                f"Try broader search terms instead of '{parsed_intent.primary_target}'",
                "Check if the relevant files are in the indexed codebase",
                "Use different keywords or synonyms",
                "Verify the correct spelling of technical terms",
            ]
        elif total_results > 30:
            suggestions = [
                "Refine your search with more specific terms",
                "Add language or file type filters",
                "Use more precise technical terminology",
            ]
        return IntentResult(
            success=success,
            data={
                "results": results,
                "summary": f"Found {total_results} results for '{parsed_intent.primary_target}'",
                "search_details": search_result,
            },
            metadata=metadata,
            error_message=None if success else search_result.get("error"),
            suggestions=suggestions,
            strategy_used=self.name,
        )
