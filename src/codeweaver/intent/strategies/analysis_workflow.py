# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Multi-step analysis strategy for UNDERSTAND and ANALYZE intents."""

import logging

from typing import Any

from codeweaver.cw_types import (
    Complexity,
    IntentResult,
    IntentStrategy,
    IntentType,
    ParsedIntent,
    ServiceType,
)
from codeweaver.cw_types.services.config import ServiceConfig
from codeweaver.services.providers.base_provider import BaseServiceProvider


class AnalysisWorkflowStrategy(BaseServiceProvider, IntentStrategy):
    """
    Multi-step analysis strategy for UNDERSTAND and ANALYZE intents.

    This strategy handles complex analysis requests by executing a
    multi-step workflow:
    1. Search for relevant code using search_code_handler
    2. Perform structural analysis with ast_grep_search_handler
    3. Generate comprehensive analysis and insights

    Optimized for:
    - UNDERSTAND and ANALYZE intent types
    - MODERATE to COMPLEX complexity
    - PROJECT to SYSTEM scope

    Leverages multiple existing CodeWeaver tools to provide rich,
    contextual analysis beyond simple search results.
    """

    def __init__(self, services_manager):
        """Initialize analysis workflow strategy."""
        config = ServiceConfig(provider="analysis_workflow_strategy")
        super().__init__(ServiceType.INTENT, config)
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.name = "analysis_workflow_strategy"
        self.supported_intent_types = [IntentType.UNDERSTAND, IntentType.ANALYZE]
        self.supported_complexity = [Complexity.MODERATE, Complexity.COMPLEX]

    async def _initialize_provider(self) -> None:
        """Initialize strategy with service dependencies."""

    async def _shutdown_provider(self) -> None:
        """Shutdown strategy resources."""

    async def _check_health(self) -> bool:
        """Check strategy health."""
        try:
            # TODO: Implement health checks for required handlers
            pass
        except ImportError:
            self.logger.warning("Required handlers not available")
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
        if parsed_intent.intent_type in [IntentType.UNDERSTAND, IntentType.ANALYZE]:
            score += 0.8
        else:
            return 0.0
        if parsed_intent.complexity in [Complexity.MODERATE, Complexity.COMPLEX]:
            score += 0.15
        elif parsed_intent.complexity == Complexity.SIMPLE:
            score += 0.05
        if parsed_intent.scope.value in ["project", "system"]:
            score += 0.05
        self.logger.debug(
            "AnalysisWorkflowStrategy can_handle score: %.2f for %s intent",
            score,
            parsed_intent.intent_type.value,
        )
        return min(1.0, score)

    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """
        Execute multi-step analysis workflow.

        Args:
            parsed_intent: The parsed intent to execute
            context: Service context with dependencies

        Returns:
            Result of the analysis workflow
        """
        try:
            self.logger.info(
                "Executing analysis workflow for: %s (%s)",
                parsed_intent.primary_target,
                parsed_intent.intent_type.value,
            )
            search_result = await self._execute_search_step(parsed_intent, context)
            workflow_results = {"search": search_result}
            if search_result["success"]:
                ast_result = await self._execute_ast_analysis(parsed_intent, context)
                workflow_results["ast_analysis"] = ast_result
            else:
                workflow_results["ast_analysis"] = {"skipped": "No search results found"}
            analysis = await self._generate_analysis(workflow_results, parsed_intent, context)
            intent_result = IntentResult(
                success=True,
                data={
                    "analysis": analysis,
                    "workflow_results": workflow_results,
                    "summary": analysis.get("summary", "Analysis completed"),
                },
                metadata={
                    "strategy": self.name,
                    "intent_type": parsed_intent.intent_type.value,
                    "steps_completed": list(workflow_results.keys()),
                    "confidence": parsed_intent.confidence,
                    "workflow_type": "multi_step_analysis",
                },
                strategy_used=self.name,
            )
            self.logger.info("Analysis workflow completed successfully")
        except Exception as e:
            self.logger.exception("Analysis workflow execution failed")
            return IntentResult(
                success=False,
                data=None,
                metadata={
                    "strategy": self.name,
                    "intent_type": parsed_intent.intent_type.value,
                    "error_type": type(e).__name__,
                },
                error_message=f"Analysis workflow failed: {e}",
                suggestions=[
                    "Try breaking down your analysis request into smaller parts",
                    "Use more specific terminology in your request",
                    "Ensure the relevant code is in the indexed codebase",
                ],
                strategy_used=self.name,
            )
        else:
            return intent_result

    async def _execute_search_step(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute search step of the workflow."""
        try:
            self.logger.debug("Executing search step for: %s", parsed_intent.primary_target)
            from codeweaver.server import search_code_handler

            search_result = await search_code_handler(
                query=parsed_intent.primary_target,
                max_results=30,
                language=parsed_intent.filters.get("language"),
                include_tests=True,
            )
        except ImportError:
            self.logger.warning("search_code_handler not available")
            return {
                "success": False,
                "error": "Search handler not available",
                "results": [],
                "total_results": 0,
            }
        except Exception as e:
            self.logger.exception("Search step failed")
            return {"success": False, "error": str(e), "results": [], "total_results": 0}
        else:
            return {
                "success": True,
                "results": search_result.get("results", []),
                "total_results": search_result.get("total_results", 0),
                "query": parsed_intent.primary_target,
            }

    async def _execute_ast_analysis(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute AST analysis step of the workflow."""
        try:
            self.logger.debug("Executing AST analysis step")
            from codeweaver.server import ast_grep_search_handler

            patterns = self._generate_ast_patterns(parsed_intent)
            ast_results = []
            for pattern in patterns:
                try:
                    result = await ast_grep_search_handler(
                        pattern=pattern,
                        language=parsed_intent.filters.get("language", "python"),
                        max_results=20,
                    )
                    if result.get("matches"):
                        ast_results.append({
                            "pattern": pattern,
                            "matches": result["matches"],
                            "total_matches": result.get("total_matches", 0),
                        })
                except Exception as e:
                    self.logger.warning("AST pattern failed: %s - %s", pattern, e)
                    continue
        except ImportError:
            self.logger.warning("ast_grep_search_handler not available")
            return {"success": False, "error": "AST analysis handler not available"}
        except Exception as e:
            self.logger.exception("AST analysis step failed")
            return {"success": False, "error": str(e)}
        else:
            return {
                "success": True,
                "patterns_analyzed": len(patterns),
                "successful_patterns": len(ast_results),
                "results": ast_results,
            }

    def _generate_ast_patterns(self, parsed_intent: ParsedIntent) -> list[str]:
        """Generate AST patterns based on the intent."""
        target = parsed_intent.primary_target.lower()
        patterns = []
        if "function" in target or "method" in target:
            patterns.extend([
                "function $NAME() { $$$ }",
                "def $NAME($$): $$$",
                "function $NAME($$$) { $$$ }",
            ])
        if "class" in target:
            patterns.extend(["class $NAME { $$$ }", "class $NAME: $$$", "interface $NAME { $$$ }"])
        if "variable" in target or "constant" in target:
            patterns.extend(["let $NAME = $$$", "const $NAME = $$$", "$NAME = $$$"])
        if "import" in target or "dependency" in target:
            patterns.extend(["import $$ from $$$", "from $$ import $$$", "require($$$)"])
        if "api" in target or "endpoint" in target:
            patterns.extend(["@app.route($$$)", "app.get($$$)", "app.post($$$)"])
        if "database" in target or "db" in target:
            patterns.extend(["$$.query($$$)", "$$.execute($$$)", "SELECT $$$ FROM $$$"])
        if not patterns:
            patterns = ["function $NAME() { $$$ }", "class $NAME { $$$ }", "def $NAME($$): $$$"]
        return patterns

    async def _generate_analysis(
        self, workflow_results: dict[str, Any], parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive analysis from workflow results."""
        analysis = {
            "intent_type": parsed_intent.intent_type.value,
            "target": parsed_intent.primary_target,
            "summary": "",
            "findings": [],
            "recommendations": [],
            "technical_details": {},
        }
        search_results = workflow_results.get("search", {})
        if search_results.get("success"):
            total_results = search_results.get("total_results", 0)
            analysis["findings"].append(
                f"Found {total_results} code references related to '{parsed_intent.primary_target}'"
            )
            if total_results > 10:
                analysis["findings"].append("Multiple implementations or usages detected")
                analysis["recommendations"].append(
                    "Consider reviewing all implementations for consistency"
                )
        ast_results = workflow_results.get("ast_analysis", {})
        if ast_results.get("success"):
            successful_patterns = ast_results.get("successful_patterns", 0)
            if successful_patterns > 0:
                analysis["findings"].append(
                    f"Structural analysis identified {successful_patterns} code patterns"
                )
                analysis["technical_details"]["ast_patterns"] = ast_results.get("results", [])
        if parsed_intent.intent_type == IntentType.UNDERSTAND:
            analysis["summary"] = f"Architecture analysis of '{parsed_intent.primary_target}'"
            analysis["recommendations"].extend([
                "Review the code structure and dependencies",
                "Understand the data flow and interactions",
                "Document key architectural decisions",
            ])
        elif parsed_intent.intent_type == IntentType.ANALYZE:
            analysis["summary"] = f"Code analysis of '{parsed_intent.primary_target}'"
            analysis["recommendations"].extend([
                "Look for potential improvements or issues",
                "Check for code quality and best practices",
                "Consider performance and security implications",
            ])
        analysis["technical_details"]["workflow_metadata"] = {
            "steps_executed": list(workflow_results.keys()),
            "complexity": parsed_intent.complexity.value,
            "scope": parsed_intent.scope.value,
            "confidence": parsed_intent.confidence,
        }
        return analysis
