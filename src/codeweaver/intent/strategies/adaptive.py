# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: no-complex-if-expressions
"""Adaptive fallback strategy for all intent types."""

import logging

from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import (
    Complexity,
    IntentResult,
    IntentStrategy,
    IntentType,
    ParsedIntent,
    ServiceType,
)
from codeweaver.types.services.config import ServiceConfig


class AdaptiveStrategy(BaseServiceProvider, IntentStrategy):
    """
    Adaptive fallback strategy for all intent types.

    This strategy serves as the universal fallback that can handle any
    intent type by dynamically adapting its approach. It uses a
    combination of heuristics and delegation to other strategies
    or direct tool invocation.

    Key capabilities:
    - Handles all intent types (SEARCH, UNDERSTAND, ANALYZE)
    - Adapts to all complexity levels (SIMPLE, MODERATE, COMPLEX)
    - Fallback behavior when specialized strategies are unavailable
    - Dynamic strategy delegation and tool orchestration
    - Intelligent error recovery and graceful degradation

    This strategy prioritizes getting a result over optimal performance,
    making it ideal as a final fallback when no specialized strategy
    can handle the intent.
    """

    def __init__(self, services_manager):
        """Initialize adaptive strategy."""
        config = ServiceConfig(provider="adaptive_strategy")
        super().__init__(ServiceType.INTENT, config)
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.name = "adaptive_strategy"
        self.supported_intent_types = [IntentType.SEARCH, IntentType.UNDERSTAND, IntentType.ANALYZE]
        self.supported_complexity = [Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX]

    async def _initialize_provider(self) -> None:
        """Initialize adaptive strategy."""

    async def _shutdown_provider(self) -> None:
        """Shutdown adaptive strategy."""

    async def _check_health(self) -> bool:
        """Check adaptive strategy health."""
        try:
            from codeweaver.server import search_code_handler  # noqa: F401
        except ImportError:
            self.logger.warning("Basic handlers not available")
            return False
        else:
            return True

    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """
        Check if adaptive strategy can handle the intent.

        The adaptive strategy can always handle any intent, but returns
        a low score to ensure it's only selected as a fallback.

        Args:
            parsed_intent: The parsed intent to evaluate

        Returns:
            Confidence score (0.0-1.0) for handling this intent
        """
        score = 0.1
        if parsed_intent.complexity == Complexity.COMPLEX:
            score += 0.05
        if parsed_intent.confidence < 0.5:
            score += 0.03
        self.logger.debug(
            "AdaptiveStrategy can_handle score: %.2f for %s intent",
            score,
            parsed_intent.intent_type.value,
        )
        return score

    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """
        Execute adaptive strategy with dynamic approach selection.

        Args:
            parsed_intent: The parsed intent to execute
            context: Service context with dependencies

        Returns:
            Result of the adaptive execution
        """
        try:
            self.logger.info(
                "Executing adaptive strategy for: %s (%s)",
                parsed_intent.primary_target,
                parsed_intent.intent_type.value,
            )
            if parsed_intent.intent_type == IntentType.SEARCH:
                result = await self._execute_adaptive_search(parsed_intent, context)
            elif parsed_intent.intent_type == IntentType.UNDERSTAND:
                result = await self._execute_adaptive_understanding(parsed_intent, context)
            elif parsed_intent.intent_type == IntentType.ANALYZE:
                result = await self._execute_adaptive_analysis(parsed_intent, context)
            else:
                self.logger.warning(
                    "Unknown intent type %s, treating as search", parsed_intent.intent_type.value
                )
                result = await self._execute_adaptive_search(parsed_intent, context)
            result.metadata.update({
                "strategy": self.name,
                "adaptation_type": f"adaptive_{parsed_intent.intent_type.value.lower()}",
                "fallback_used": True,
                "complexity_handled": parsed_intent.complexity.value,
            })
            self.logger.info("Adaptive strategy completed successfully")
        except Exception as e:
            self.logger.exception("Adaptive strategy execution failed")
            return IntentResult(
                success=False,
                data=None,
                metadata={
                    "strategy": self.name,
                    "intent_type": parsed_intent.intent_type.value,
                    "error_type": type(e).__name__,
                    "fallback_used": True,
                },
                error_message=f"Adaptive strategy failed: {e}",
                suggestions=[
                    "Try breaking down your request into simpler parts",
                    "Use more specific terminology",
                    "Check if the relevant content is available in the codebase",
                    "Consider using a different phrasing for your request",
                ],
                strategy_used=self.name,
            )
        else:
            return result

    async def _execute_adaptive_search(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Execute adaptive search approach."""
        try:
            self.logger.debug("Executing adaptive search")
            from codeweaver.server import search_code_handler

            max_results = self._get_adaptive_max_results(parsed_intent.complexity)
            search_result = await search_code_handler(
                query=parsed_intent.primary_target,
                max_results=max_results,
                language=parsed_intent.filters.get("language"),
                include_tests=parsed_intent.complexity != Complexity.SIMPLE,
            )
            results = search_result.get("results", [])
            total_results = search_result.get("total_results", len(results))
        except ImportError:
            return IntentResult(
                success=False,
                data=None,
                metadata={},
                error_message="Search handler not available",
                suggestions=["Check if CodeWeaver server is properly configured"],
                strategy_used=self.name,
            )
        else:
            return IntentResult(
                success=total_results > 0,
                data={
                    "results": results,
                    "summary": f"Adaptive search found {total_results} results",
                    "search_details": search_result,
                    "adaptation_notes": "Direct search handler delegation",
                },
                metadata={
                    "total_results": total_results,
                    "max_results_used": max_results,
                    "original_query": parsed_intent.primary_target,
                },
                suggestions=self._generate_search_suggestions(total_results, parsed_intent),
                strategy_used=self.name,
            )

    async def _execute_adaptive_understanding(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Execute adaptive understanding approach."""
        try:
            self.logger.debug("Executing adaptive understanding")
            search_result = await self._execute_adaptive_search(parsed_intent, context)
            if search_result.success:
                understanding_analysis = self._generate_understanding_analysis(
                    search_result.data, parsed_intent
                )
                return IntentResult(
                    success=True,
                    data={
                        **search_result.data,
                        "understanding_analysis": understanding_analysis,
                        "approach": "search_plus_analysis",
                    },
                    metadata={**search_result.metadata, "analysis_depth": "basic_understanding"},
                    suggestions=[
                        "Review the found code to understand its structure",
                        "Look at the relationships between different components",
                        "Consider the broader architectural context",
                    ],
                    strategy_used=self.name,
                )
        except Exception as e:
            self.logger.exception("Adaptive understanding failed")
            return IntentResult(
                success=False,
                data=None,
                metadata={},
                error_message=f"Understanding execution failed: {e}",
                strategy_used=self.name,
            )
        else:
            return IntentResult(
                success=False,
                data={
                    "understanding_guidance": f"Could not find specific code for '{parsed_intent.primary_target}'"
                },
                metadata={},
                error_message="No code found to understand",
                suggestions=[
                    "Check if the term exists in the codebase",
                    "Try broader or more specific search terms",
                    "Ensure the relevant files are indexed",
                ],
                strategy_used=self.name,
            )

    async def _execute_adaptive_analysis(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Execute adaptive analysis approach."""
        try:
            self.logger.debug("Executing adaptive analysis")
            search_result = await self._execute_adaptive_search(parsed_intent, context)
            analysis_results = {"search": search_result.data if search_result.success else None}
            try:
                ast_result = await self._try_ast_analysis(parsed_intent, context)
                analysis_results["ast_analysis"] = ast_result
            except Exception as e:
                self.logger.debug("AST analysis not available: %s", e)
                analysis_results["ast_analysis"] = None
            comprehensive_analysis = self._generate_comprehensive_analysis(
                analysis_results, parsed_intent
            )
            success = any(result is not None for result in analysis_results.values())
        except Exception as e:
            self.logger.exception("Adaptive analysis failed")
            return IntentResult(
                success=False,
                data=None,
                metadata={},
                error_message=f"Analysis execution failed: {e}",
                strategy_used=self.name,
            )
        else:
            return IntentResult(
                success=success,
                data={
                    "analysis": comprehensive_analysis,
                    "analysis_results": analysis_results,
                    "approach": "multi_tool_adaptive",
                },
                metadata={
                    "tools_used": [k for k, v in analysis_results.items() if v is not None],
                    "analysis_depth": "comprehensive_adaptive",
                },
                suggestions=[
                    "Review the analysis findings carefully",
                    "Consider the architectural implications",
                    "Look for potential improvements or issues",
                ]
                if success
                else [
                    "No analyzable code found for this target",
                    "Try a different search term or approach",
                    "Ensure the relevant code is in the indexed codebase",
                ],
                strategy_used=self.name,
            )

    def _get_adaptive_max_results(self, complexity: Complexity) -> int:
        """Get adaptive max results based on complexity."""
        if complexity == Complexity.SIMPLE:
            return 10
        return 25 if complexity == Complexity.MODERATE else 50

    def _generate_search_suggestions(
        self, total_results: int, parsed_intent: ParsedIntent
    ) -> list[str]:
        """Generate adaptive search suggestions."""
        if total_results == 0:
            return [
                f"No results found for '{parsed_intent.primary_target}'",
                "Try using different keywords or synonyms",
                "Check spelling and terminology",
                "Consider broadening your search terms",
            ]
        if total_results > 30:
            return [
                f"Many results found ({total_results}) - consider narrowing your search",
                "Add more specific terms to refine results",
                "Use technical terminology for precision",
            ]
        return [
            f"Found {total_results} results - review for relevance",
            "Results are filtered by adaptive strategy parameters",
        ]

    def _generate_understanding_analysis(
        self, search_data: dict[str, Any], parsed_intent: ParsedIntent
    ) -> dict[str, Any]:
        """Generate understanding-specific analysis."""
        results = search_data.get("results", [])
        return {
            "target": parsed_intent.primary_target,
            "approach": "adaptive_understanding",
            "findings": [
                f"Found {len(results)} code references",
                "Analysis based on search results and adaptive heuristics",
            ],
            "understanding_notes": [
                "This is a basic understanding generated by adaptive strategy",
                "For deeper analysis, consider using specialized tools",
            ],
            "next_steps": [
                "Examine the found code files for implementation details",
                "Look for related functions or classes",
                "Consider the broader system context",
            ],
        }

    def _generate_comprehensive_analysis(
        self, analysis_results: dict[str, Any], parsed_intent: ParsedIntent
    ) -> dict[str, Any]:
        """Generate comprehensive analysis from multiple results."""
        search_data = analysis_results.get("search")
        ast_data = analysis_results.get("ast_analysis")
        findings = []
        recommendations = []
        if search_data and search_data.get("results"):
            result_count = len(search_data["results"])
            findings.append(f"Found {result_count} code references through semantic search")
            if result_count > 10:
                recommendations.append("Multiple implementations found - review for consistency")
        if ast_data and ast_data.get("success"):
            findings.append("Structural analysis completed with AST patterns")
            recommendations.append("Review structural patterns for architectural insights")
        if not findings:
            findings.append(f"Limited analysis available for '{parsed_intent.primary_target}'")
            recommendations.extend([
                "Consider using more specific search terms",
                "Ensure relevant code is in the indexed codebase",
            ])
        return {
            "target": parsed_intent.primary_target,
            "approach": "adaptive_comprehensive",
            "findings": findings,
            "recommendations": recommendations,
            "analysis_depth": "adaptive_multi_tool",
            "tools_attempted": list(analysis_results.keys()),
            "successful_tools": [k for k, v in analysis_results.items() if v is not None],
        }

    async def _try_ast_analysis(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Try AST analysis if available."""
        try:
            from codeweaver.server import ast_grep_search_handler

            target = parsed_intent.primary_target.lower()
            patterns = []
            if "function" in target or "method" in target:
                patterns.extend(("function $NAME() { $$$ }", "def $NAME($$): $$$"))
            elif "class" in target:
                patterns.extend(("class $NAME { $$$ }", "class $NAME: $$$"))
            else:
                patterns.extend([
                    "function $NAME() { $$$ }",
                    "class $NAME { $$$ }",
                    "def $NAME($$): $$$",
                ])
            if patterns:
                result = await ast_grep_search_handler(
                    pattern=patterns[0],
                    language=parsed_intent.filters.get("language", "python"),
                    max_results=10,
                )
                return {
                    "success": True,
                    "pattern_used": patterns[0],
                    "matches": result.get("matches", []),
                    "total_matches": result.get("total_matches", 0),
                }
        except Exception as e:
            self.logger.debug("AST analysis failed: %s", e)
            return {"success": False, "error": str(e)}
        return {"success": False, "error": "No patterns generated"}
