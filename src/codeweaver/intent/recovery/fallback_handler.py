"""Error handling and recovery system using existing exception hierarchy."""


import contextlib
import logging

from dataclasses import dataclass
from enum import Enum
from typing import Any

from codeweaver.types import (
    IntentError,
    IntentParsingError,
    IntentResult,
    IntentType,
    ParsedIntent,
    ServiceError,
    ServiceIntegrationError,
    StrategyExecutionError,
    StrategySelectionError,
)


class FallbackType(Enum):
    """Types of fallback strategies available."""

    STRATEGY_FALLBACK = "strategy_fallback"
    "Try alternative strategy."
    PARSER_FALLBACK = "parser_fallback"
    "Fall back to simpler parsing."
    SERVICE_DEGRADATION = "service_degradation"
    "Degrade to reduced functionality."
    TOOL_FALLBACK = "tool_fallback"
    "Route to appropriate original tool."
    GENERIC_RECOVERY = "generic_recovery"
    "Generic error recovery."


@dataclass
class FallbackResult:
    """Result of a fallback attempt."""

    success: bool
    "Whether the fallback succeeded."
    result: IntentResult | None
    "Result from fallback execution if successful."
    fallback_type: FallbackType
    "Type of fallback that was attempted."
    error_message: str | None = None
    "Error message if fallback failed."
    suggestions: list[str] | None = None
    "Suggested next actions."


@dataclass
class FallbackChain:
    """Defines a chain of fallback strategies to try in order."""

    handlers: list[tuple[type[Exception], FallbackType]]
    "List of (exception_type, fallback_type) pairs to try in order."
    max_attempts: int = 3
    "Maximum number of fallback attempts."
    enable_generic_fallback: bool = True
    "Whether to try generic fallback as last resort."


class IntentErrorHandler:
    """
    Error handling using existing exception hierarchy.

    This handler provides intelligent error recovery for intent processing
    failures by implementing a chain of fallback strategies:

    1. Strategy Fallback: Try alternative strategies
    2. Parser Fallback: Fall back to simpler parsing approaches
    3. Service Degradation: Operate with reduced functionality
    4. Tool Fallback: Route to appropriate original tools
    5. Generic Recovery: Provide helpful error messages and guidance

    The handler integrates with existing CodeWeaver error patterns and
    leverages the services manager for fallback operations.
    """

    def __init__(self, services_manager):
        """Initialize error handler with services manager."""
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.fallback_chain = self._build_default_fallback_chain()
        self._error_stats = {
            "total_errors": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "fallback_types_used": {},
        }

    def _build_default_fallback_chain(self) -> FallbackChain:
        """Build the default fallback chain from specification."""
        return FallbackChain(
            handlers=[
                (StrategyExecutionError, FallbackType.STRATEGY_FALLBACK),
                (StrategySelectionError, FallbackType.STRATEGY_FALLBACK),
                (ServiceIntegrationError, FallbackType.SERVICE_DEGRADATION),
                (IntentParsingError, FallbackType.PARSER_FALLBACK),
                (ServiceError, FallbackType.SERVICE_DEGRADATION),
                (IntentError, FallbackType.TOOL_FALLBACK),
            ],
            max_attempts=3,
            enable_generic_fallback=True,
        )

    async def handle_error(
        self,
        error: Exception,
        context: dict[str, Any],
        parsed_intent: ParsedIntent | None = None,
        original_intent_text: str | None = None,
    ) -> IntentResult:
        """
        Handle errors with fallback chain.

        Args:
            error: The exception that occurred
            context: Service context for recovery attempts
            parsed_intent: The parsed intent that failed (if available)
            original_intent_text: Original intent text (if available)

        Returns:
            Result from successful fallback or final error result
        """
        self._error_stats["total_errors"] += 1
        self.logger.info("Handling error: %s - %s", type(error).__name__, str(error))
        for attempt, (exception_type, fallback_type) in enumerate(self.fallback_chain.handlers):
            if attempt >= self.fallback_chain.max_attempts:
                break
            if isinstance(error, exception_type):
                self.logger.debug(
                    "Attempting %s fallback for %s", fallback_type.value, type(error).__name__
                )
                fallback_result = await self._execute_fallback(
                    fallback_type, error, context, parsed_intent, original_intent_text
                )
                if fallback_result.success:
                    self._error_stats["successful_fallbacks"] += 1
                    self._update_fallback_stats(fallback_type, success=True)
                    self.logger.info("Fallback successful: %s", fallback_type.value)
                    return fallback_result.result
                self.logger.warning(
                    "Fallback failed: %s - %s", fallback_type.value, fallback_result.error_message
                )
        if self.fallback_chain.enable_generic_fallback:
            self.logger.debug("Attempting generic recovery fallback")
            fallback_result = await self._execute_fallback(
                FallbackType.GENERIC_RECOVERY, error, context, parsed_intent, original_intent_text
            )
            if fallback_result.success:
                self._error_stats["successful_fallbacks"] += 1
                self._update_fallback_stats(FallbackType.GENERIC_RECOVERY, success=True)
                return fallback_result.result
        self._error_stats["failed_fallbacks"] += 1
        self.logger.error("All fallback attempts failed for error: %s", error)
        return self._create_final_error_result(error, parsed_intent, original_intent_text)

    async def _execute_fallback(
        self,
        fallback_type: FallbackType,
        error: Exception,
        context: dict[str, Any],
        parsed_intent: ParsedIntent | None,
        original_intent_text: str | None,
    ) -> FallbackResult:
        """Execute a specific type of fallback."""
        try:
            if fallback_type == FallbackType.STRATEGY_FALLBACK:
                return await self._strategy_fallback(error, context, parsed_intent)
            if fallback_type == FallbackType.PARSER_FALLBACK:
                return await self._parser_fallback(error, context, original_intent_text)
            if fallback_type == FallbackType.SERVICE_DEGRADATION:
                return await self._service_degradation_fallback(error, context, parsed_intent)
            if fallback_type == FallbackType.TOOL_FALLBACK:
                return await self._tool_fallback(error, context, parsed_intent)
            if fallback_type == FallbackType.GENERIC_RECOVERY:
                return await self._generic_recovery_fallback(error, context, parsed_intent)
        except Exception as fallback_error:
            self.logger.exception("Fallback execution failed")
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=fallback_type,
                error_message=f"Fallback execution failed: {fallback_error}",
            )
        else:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=fallback_type,
                error_message=f"Unknown fallback type: {fallback_type}",
            )

    async def _strategy_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent | None
    ) -> FallbackResult:
        """Try alternative strategy for failed strategy execution."""
        if not parsed_intent:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.STRATEGY_FALLBACK,
                error_message="No parsed intent available for strategy fallback",
            )
        try:
            from codeweaver.intent.strategies.adaptive import AdaptiveStrategy

            adaptive_strategy = AdaptiveStrategy(self.services_manager)
            can_handle_score = await adaptive_strategy.can_handle(parsed_intent)
            if can_handle_score > 0.0:
                result = await adaptive_strategy.execute(parsed_intent, context)
                result.metadata["fallback_used"] = True
                result.metadata["original_error"] = str(error)
                result.metadata["fallback_type"] = "strategy"
                return FallbackResult(
                    success=result.success,
                    result=result,
                    fallback_type=FallbackType.STRATEGY_FALLBACK,
                )
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.STRATEGY_FALLBACK,
                error_message=f"Strategy fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.STRATEGY_FALLBACK,
                error_message="Adaptive strategy cannot handle intent",
            )

    async def _parser_fallback(
        self, error: Exception, context: dict[str, Any], original_intent_text: str | None
    ) -> FallbackResult:
        """Fall back to simpler parsing approach."""
        if not original_intent_text:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.PARSER_FALLBACK,
                error_message="No original intent text available for parser fallback",
            )
        try:
            from codeweaver.intent.parsing.pattern_matcher import PatternBasedParser

            fallback_parser = PatternBasedParser()
            parsed_intent = await fallback_parser.parse(original_intent_text)
            from codeweaver.intent.strategies.adaptive import AdaptiveStrategy

            adaptive_strategy = AdaptiveStrategy(self.services_manager)
            result = await adaptive_strategy.execute(parsed_intent, context)
            result.metadata["fallback_used"] = True
            result.metadata["original_error"] = str(error)
            result.metadata["fallback_type"] = "parser"
            result.metadata["fallback_parser"] = "pattern_based"
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.PARSER_FALLBACK,
                error_message=f"Parser fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=result.success, result=result, fallback_type=FallbackType.PARSER_FALLBACK
            )

    async def _service_degradation_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent | None
    ) -> FallbackResult:
        """Degrade to reduced functionality."""
        try:
            if parsed_intent:
                intent_type = parsed_intent.intent_type.value
                target = parsed_intent.primary_target
            else:
                intent_type = "unknown"
                target = "unknown"
            degraded_result = IntentResult(
                success=True,
                data={
                    "message": f"Service operating in degraded mode for {intent_type} intent",
                    "target": target,
                    "functionality": "reduced",
                    "note": "Some services are unavailable - operating with basic functionality",
                    "original_error": str(error),
                },
                metadata={
                    "degraded_mode": True,
                    "fallback_used": True,
                    "fallback_type": "service_degradation",
                    "available_services": self._get_available_services(),
                },
                suggestions=[
                    "Try again later when services are restored",
                    "Use simpler queries while in degraded mode",
                    "Check service status for more information",
                ],
                strategy_used="service_degradation_fallback",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.SERVICE_DEGRADATION,
                error_message=f"Service degradation fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=True, result=degraded_result, fallback_type=FallbackType.SERVICE_DEGRADATION
            )

    async def _tool_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent | None
    ) -> FallbackResult:
        """Route to appropriate original tool."""
        try:
            if not parsed_intent:
                return await self._generic_search_fallback(error, context)
            if parsed_intent.intent_type == IntentType.SEARCH:
                return await self._search_tool_fallback(error, context, parsed_intent)
            if parsed_intent.intent_type in [IntentType.UNDERSTAND, IntentType.ANALYZE]:
                return await self._analysis_tool_fallback(error, context, parsed_intent)
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.TOOL_FALLBACK,
                error_message=f"Tool fallback failed: {e}",
            )
        else:
            return await self._generic_search_fallback(error, context)

    async def _search_tool_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent
    ) -> FallbackResult:
        """Fallback to search_code_handler."""
        try:
            from codeweaver.server import search_code_handler

            search_result = await search_code_handler(
                query=parsed_intent.primary_target,
                max_results=20,
                language=parsed_intent.filters.get("language"),
                include_tests=True,
            )
            tool_result = IntentResult(
                success=True,
                data={
                    "results": search_result.get("results", []),
                    "total_results": search_result.get("total_results", 0),
                    "fallback_note": "Routed to original search tool",
                    "original_error": str(error),
                },
                metadata={
                    "tool_fallback": True,
                    "fallback_used": True,
                    "fallback_type": "search_tool",
                    "tool_used": "search_code_handler",
                },
                suggestions=[
                    "Results from original search tool",
                    "Intent layer will be restored when service is available",
                ],
                strategy_used="search_tool_fallback",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.TOOL_FALLBACK,
                error_message=f"Search tool fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=True, result=tool_result, fallback_type=FallbackType.TOOL_FALLBACK
            )

    async def _analysis_tool_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent
    ) -> FallbackResult:
        """Fallback to analysis tools."""
        try:
            results = {"search": None, "ast": None}
            try:
                from codeweaver.server import search_code_handler

                search_result = await search_code_handler(
                    query=parsed_intent.primary_target,
                    max_results=30,
                    language=parsed_intent.filters.get("language"),
                    include_tests=True,
                )
                results["search"] = search_result
            except Exception as e:
                self.logger.debug("Search fallback failed: %s", e)
            try:
                from codeweaver.server import ast_grep_search_handler

                pattern = (
                    "function $NAME() { $$$ }"
                    if "function" in parsed_intent.primary_target.lower()
                    else "$NAME"
                )
                ast_result = await ast_grep_search_handler(
                    pattern=pattern,
                    language=parsed_intent.filters.get("language", "python"),
                    max_results=20,
                )
                results["ast"] = ast_result
            except Exception as e:
                self.logger.debug("AST fallback failed: %s", e)
            success = any(r is not None for r in results.values())
            tool_result = IntentResult(
                success=success,
                data={
                    "analysis_results": results,
                    "fallback_note": "Analysis using original tools",
                    "original_error": str(error),
                    "summary": self._create_analysis_summary(results, parsed_intent),
                },
                metadata={
                    "tool_fallback": True,
                    "fallback_used": True,
                    "fallback_type": "analysis_tools",
                    "tools_attempted": list(results.keys()),
                    "successful_tools": [k for k, v in results.items() if v is not None],
                },
                suggestions=[
                    "Analysis from original tools",
                    "Intent layer will provide richer analysis when restored",
                ],
                strategy_used="analysis_tool_fallback",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.TOOL_FALLBACK,
                error_message=f"Analysis tool fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=success, result=tool_result, fallback_type=FallbackType.TOOL_FALLBACK
            )

    async def _generic_search_fallback(
        self, error: Exception, context: dict[str, Any]
    ) -> FallbackResult:
        """Generic search fallback when no specific intent is available."""
        try:
            from codeweaver.server import search_code_handler

            search_query = self._extract_search_terms(error, context)
            search_result = await search_code_handler(
                query=search_query, max_results=15, include_tests=True
            )
            generic_result = IntentResult(
                success=True,
                data={
                    "results": search_result.get("results", []),
                    "total_results": search_result.get("total_results", 0),
                    "fallback_note": "Generic search fallback",
                    "search_query": search_query,
                    "original_error": str(error),
                },
                metadata={
                    "generic_fallback": True,
                    "fallback_used": True,
                    "fallback_type": "generic_search",
                },
                suggestions=[
                    "Generic search results",
                    "Try rephrasing your request when services are restored",
                ],
                strategy_used="generic_search_fallback",
            )
        except Exception as e:
            return FallbackResult(
                success=False,
                result=None,
                fallback_type=FallbackType.TOOL_FALLBACK,
                error_message=f"Generic search fallback failed: {e}",
            )
        else:
            return FallbackResult(
                success=True, result=generic_result, fallback_type=FallbackType.TOOL_FALLBACK
            )

    async def _generic_recovery_fallback(
        self, error: Exception, context: dict[str, Any], parsed_intent: ParsedIntent | None
    ) -> FallbackResult:
        """Generic error recovery with helpful guidance."""
        error_type = type(error).__name__
        suggestions = self._generate_error_suggestions(error, parsed_intent)
        recovery_result = IntentResult(
            success=False,
            data={
                "error_type": error_type,
                "error_message": str(error),
                "recovery_attempted": True,
                "guidance": self._generate_error_guidance(error, parsed_intent),
            },
            metadata={
                "generic_recovery": True,
                "fallback_used": True,
                "fallback_type": "generic_recovery",
                "error_category": self._categorize_error(error),
            },
            error_message=f"Intent processing failed: {error}",
            suggestions=suggestions,
            strategy_used="generic_recovery_fallback",
        )
        return FallbackResult(
            success=True,
            result=recovery_result,
            fallback_type=FallbackType.GENERIC_RECOVERY,
            suggestions=suggestions,
        )

    def _create_final_error_result(
        self, error: Exception, parsed_intent: ParsedIntent | None, original_intent_text: str | None
    ) -> IntentResult:
        """Create final error result when all fallbacks fail."""
        error_type = type(error).__name__
        return IntentResult(
            success=False,
            data=None,
            metadata={
                "all_fallbacks_failed": True,
                "error_type": error_type,
                "fallback_attempts": len(self.fallback_chain.handlers),
                "original_intent": original_intent_text,
            },
            error_message=f"All recovery attempts failed. Original error: {error}",
            suggestions=[
                "The intent processing system is experiencing issues",
                "Try using the original CodeWeaver tools directly",
                "Contact support if the issue persists",
                "Check system status and logs for more information",
            ],
            strategy_used="final_error_fallback",
        )

    def _extract_search_terms(self, error: Exception, context: dict[str, Any]) -> str:
        """Extract searchable terms from error and context."""
        str(error).lower()
        import re

        if quoted_terms := re.findall(r"'([^']+)'", str(error)):
            return quoted_terms[0]
        technical_terms = re.findall(r"\b\w*[A-Z]\w*\b", str(error))
        return technical_terms[0] if technical_terms else "function"

    def _create_analysis_summary(self, results: dict[str, Any], parsed_intent: ParsedIntent) -> str:
        """Create analysis summary from fallback results."""
        summaries = []
        if results.get("search"):
            search_count = results["search"].get("total_results", 0)
            summaries.append(f"Found {search_count} search results")
        if results.get("ast"):
            ast_count = results["ast"].get("total_matches", 0)
            summaries.append(f"Found {ast_count} structural matches")
        if not summaries:
            summaries.append("Limited analysis available")
        return f"Analysis of '{parsed_intent.primary_target}': " + ", ".join(summaries)

    def _generate_error_suggestions(
        self, error: Exception, parsed_intent: ParsedIntent | None
    ) -> list[str]:
        """Generate helpful suggestions based on error type."""
        if isinstance(error, IntentParsingError):
            return [
                "Try rephrasing your request using simpler language",
                "Use keywords like 'find', 'search', 'understand', or 'analyze'",
                "Be more specific about what you're looking for",
                "Check spelling and terminology",
            ]
        if isinstance(error, StrategyExecutionError):
            return [
                "Try a simpler version of your request",
                "Break down complex requests into smaller parts",
                "Ensure the relevant code is indexed",
                "Try using original tools directly",
            ]
        if isinstance(error, ServiceIntegrationError):
            return [
                "Some services may be temporarily unavailable",
                "Try again in a few moments",
                "Use basic search functionality",
                "Contact support if issues persist",
            ]
        return [
            "Try rephrasing your request",
            "Use more specific search terms",
            "Check if the relevant content exists in the codebase",
            "Try the original CodeWeaver tools directly",
        ]

    def _generate_error_guidance(self, error: Exception, parsed_intent: ParsedIntent | None) -> str:
        """Generate contextual guidance for the error."""
        error_type = type(error).__name__
        if isinstance(error, IntentParsingError):
            return "The system had trouble understanding your request. Try using clearer language or standard technical terms."
        if isinstance(error, StrategyExecutionError):
            return "The system understood your request but couldn't execute it properly. Try simplifying your request or breaking it into smaller parts."
        if isinstance(error, ServiceIntegrationError):
            return "Some internal services are experiencing issues. The system is operating in reduced functionality mode."
        return f"An unexpected error occurred ({error_type}). The system attempted multiple recovery strategies but was unable to process your request."

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for metadata."""
        if isinstance(error, IntentParsingError):
            return "parsing"
        if isinstance(error, StrategyExecutionError):
            return "execution"
        if isinstance(error, ServiceIntegrationError):
            return "service"
        return "intent" if isinstance(error, IntentError) else "unknown"

    def _get_available_services(self) -> list[str]:
        """Get list of currently available services."""
        available = []
        with contextlib.suppress(ImportError):
            from codeweaver.server import search_code_handler  # noqa: F401

            available.append("search")
        with contextlib.suppress(ImportError):
            from codeweaver.server import ast_grep_search_handler  # noqa: F401

            available.append("ast_analysis")
        return available

    def _update_fallback_stats(self, fallback_type: FallbackType, *, success: bool) -> None:
        """Update fallback statistics."""
        if fallback_type.value not in self._error_stats["fallback_types_used"]:
            self._error_stats["fallback_types_used"][fallback_type.value] = {
                "attempts": 0,
                "successful": 0,
            }
        stats = self._error_stats["fallback_types_used"][fallback_type.value]
        stats["attempts"] += 1
        if success:
            stats["successful"] += 1

    def get_error_stats(self) -> dict[str, Any]:
        """Get comprehensive error handling statistics."""
        return {
            "error_stats": self._error_stats.copy(),
            "fallback_chain_length": len(self.fallback_chain.handlers),
            "success_rate": self._error_stats["successful_fallbacks"]
            / max(1, self._error_stats["total_errors"]),
        }
