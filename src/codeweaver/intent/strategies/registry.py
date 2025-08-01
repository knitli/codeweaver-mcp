# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Strategy registry integrated with ExtensibilityManager."""

import contextlib
import logging
import operator
import time

from typing import Any

from codeweaver.cw_types import (
    IntentStrategy,
    ParsedIntent,
    ServiceIntegrationError,
    StrategySelectionError,
)


class StrategyPerformanceTracker:
    """Tracks performance metrics for strategy selection."""

    def __init__(self):
        """Initialize performance tracker."""
        self.strategy_stats = {}
        self.selection_history = []
        self.logger = logging.getLogger(f"{__name__}.PerformanceTracker")

    def get_score(self, strategy_name: str) -> float:
        """Get performance score for a strategy (0.0-1.0)."""
        if strategy_name not in self.strategy_stats:
            return 0.5
        stats = self.strategy_stats[strategy_name]
        success_rate = stats["successes"] / max(1, stats["total_executions"])
        avg_time = stats["total_time"] / max(1, stats["total_executions"])
        time_score = max(0.0, min(1.0, (10.0 - avg_time) / 10.0))
        return success_rate * 0.7 + time_score * 0.3

    async def record_selection(self, strategy_name: str, parsed_intent: ParsedIntent) -> None:
        """Record strategy selection for learning."""
        self.selection_history.append({
            "strategy": strategy_name,
            "intent_type": parsed_intent.intent_type.value,
            "complexity": parsed_intent.complexity.value,
            "timestamp": time.time(),
        })
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

    def record_execution_result(
        self, strategy_name: str, *, success: bool, execution_time: float
    ) -> None:
        """Record strategy execution result."""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                "total_executions": 0,
                "successes": 0,
                "total_time": 0.0,
                "last_used": 0.0,
            }
        stats = self.strategy_stats[strategy_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        stats["last_used"] = time.time()
        if success:
            stats["successes"] += 1

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get comprehensive strategy statistics."""
        return {
            "total_strategies": len(self.strategy_stats),
            "total_selections": len(self.selection_history),
            "strategy_performance": {
                name: {
                    **stats,
                    "success_rate": stats["successes"] / max(1, stats["total_executions"]),
                    "avg_execution_time": stats["total_time"] / max(1, stats["total_executions"]),
                    "performance_score": self.get_score(name),
                }
                for name, stats in self.strategy_stats.items()
            },
        }


class StrategyRegistry:
    """
    Strategy registry integrated with existing ExtensibilityManager.

    This registry manages the discovery, registration, and selection of
    intent execution strategies. It integrates with CodeWeaver's existing
    ExtensibilityManager to leverage the established plugin architecture.

    Key features:
    - Uses ExtensibilityManager for component discovery
    - Performance tracking for strategy selection optimization
    - Service dependency injection
    - Health monitoring integration
    """

    def __init__(self, services_manager):
        """Initialize strategy registry with services manager."""
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.extensibility_manager = None
        self.performance_tracker = StrategyPerformanceTracker()
        self._strategy_cache = {}
        self._registry_initialized = False

    async def initialize(self) -> None:
        """Initialize the strategy registry with ExtensibilityManager."""
        if self._registry_initialized:
            return
        try:
            self.logger.info("Initializing strategy registry with ExtensibilityManager")
            self.extensibility_manager = await self._get_extensibility_manager()
            await self._register_core_strategies()
            self._registry_initialized = True
            self.logger.info("Strategy registry initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize strategy registry")
            raise ServiceIntegrationError(f"Strategy registry initialization failed: {e}") from e

    def _raise_strategy_error(self, message: str) -> None:
        """Raise a StrategySelectionError with the given message."""
        raise StrategySelectionError(message)

    async def _register_core_strategies(self) -> None:
        """Register core strategies through ExtensibilityManager."""
        if not self.extensibility_manager:
            self.logger.warning("ExtensibilityManager not available - using fallback registration")
            return
        try:
            from codeweaver.intent.strategies.adaptive import AdaptiveStrategy
            from codeweaver.intent.strategies.analysis_workflow import AnalysisWorkflowStrategy
            from codeweaver.intent.strategies.simple_search import SimpleSearchStrategy

            self.extensibility_manager.register_component(
                "simple_search_strategy",
                SimpleSearchStrategy,
                component_type="intent_strategy",
                metadata={
                    "intent_types": ["SEARCH"],
                    "complexity_levels": ["SIMPLE", "MODERATE"],
                    "priority": 0.8,
                },
            )
            self.extensibility_manager.register_component(
                "analysis_workflow_strategy",
                AnalysisWorkflowStrategy,
                component_type="intent_strategy",
                metadata={
                    "intent_types": ["UNDERSTAND", "ANALYZE"],
                    "complexity_levels": ["MODERATE", "COMPLEX"],
                    "priority": 0.9,
                },
            )
            self.extensibility_manager.register_component(
                "adaptive_strategy",
                AdaptiveStrategy,
                component_type="intent_strategy",
                metadata={
                    "intent_types": ["SEARCH", "UNDERSTAND", "ANALYZE"],
                    "complexity_levels": ["SIMPLE", "MODERATE", "COMPLEX", "ADAPTIVE"],
                    "priority": 0.1,
                },
            )
            self.logger.info("Core strategies registered successfully")
        except ImportError as e:
            self.logger.warning("Some strategy classes not yet implemented: %s", e)
        except Exception as e:
            self.logger.exception("Failed to register core strategies")
            raise ServiceIntegrationError(f"Core strategy registration failed: {e}") from e

    async def select_strategy(self, parsed_intent: ParsedIntent) -> IntentStrategy:
        """
        Select strategy using existing extensibility patterns.

        Args:
            parsed_intent: The parsed intent to find a strategy for

        Returns:
            Selected strategy instance

        Raises:
            StrategySelectionError: If no suitable strategy is found
        """
        if not self._registry_initialized:
            await self.initialize()
        try:
            self.logger.debug("Selecting strategy for intent: %s", parsed_intent.intent_type.value)
            if self.extensibility_manager:
                available_strategies = self.extensibility_manager.discover_components(
                    component_type="intent_strategy"
                )
            else:
                available_strategies = await self._get_fallback_strategies()
            candidates = []
            for strategy_info in available_strategies:
                try:
                    strategy = await self._create_strategy_instance(strategy_info)
                    can_handle_score = await strategy.can_handle(parsed_intent)
                    if can_handle_score > 0.1:
                        performance_score = self.performance_tracker.get_score(
                            strategy_info.get("name", "unknown")
                        )
                        final_score = can_handle_score * 0.7 + performance_score * 0.3
                        candidates.append((
                            final_score,
                            strategy_info.get("name", "unknown"),
                            strategy,
                        ))
                        self.logger.debug(
                            "Strategy %s: capability=%.2f, performance=%.2f, final=%.2f",
                            strategy_info.get("name", "unknown"),
                            can_handle_score,
                            performance_score,
                            final_score,
                        )
                except Exception as e:
                    self.logger.warning(
                        "Failed to evaluate strategy %s: %s",
                        strategy_info.get("name", "unknown"),
                        e,
                    )
            if not candidates:
                fallback_strategy = await self._get_fallback_adaptive_strategy()
                if fallback_strategy:
                    self.logger.info("Using fallback adaptive strategy")
                    return fallback_strategy
                self._raise_strategy_error(
                    "No suitable strategy found for intent: %s", parsed_intent.intent_type.value
                )
            candidates.sort(key=operator.itemgetter(0), reverse=True)
            selected_score, selected_name, selected_strategy = candidates[0]
            await self.performance_tracker.record_selection(selected_name, parsed_intent)
            self.logger.info("Selected strategy: %s (score: %.2f)", selected_name, selected_score)
        except Exception as e:
            self.logger.exception("Strategy selection failed")
            raise StrategySelectionError(f"Failed to select strategy: {e}") from e
        else:
            return selected_strategy

    async def _create_strategy_instance(self, strategy_info: dict) -> IntentStrategy:
        """Create strategy instance with dependency injection."""
        if strategy_class := strategy_info.get("component_class"):
            return strategy_class(self.services_manager)
        raise StrategySelectionError("Strategy class not found in info")

    async def _get_fallback_strategies(self) -> list[dict]:
        """Get fallback strategy list when ExtensibilityManager is not available."""
        fallback_strategies = []
        with contextlib.suppress(ImportError):
            from codeweaver.intent.strategies.adaptive import AdaptiveStrategy

            fallback_strategies.append({
                "name": "adaptive_strategy",
                "component_class": AdaptiveStrategy,
                "metadata": {"priority": 0.1},
            })
        return fallback_strategies

    async def _get_fallback_adaptive_strategy(self) -> IntentStrategy | None:
        """Get adaptive strategy as final fallback."""
        try:
            from codeweaver.intent.strategies.adaptive import AdaptiveStrategy
        except ImportError:
            return None
        else:
            return AdaptiveStrategy(self.services_manager)

    async def _get_extensibility_manager(self):
        """Get ExtensibilityManager through services."""
        try:
            if hasattr(self.services_manager, "get_extensibility_manager"):
                return await self.services_manager.get_extensibility_manager()
            from codeweaver.config import CodeWeaverConfig
            from codeweaver.factories.extensibility_manager import ExtensibilityManager

            config = CodeWeaverConfig()
            extensibility_manager = ExtensibilityManager(config)
            await extensibility_manager.initialize()
        except Exception as e:
            self.logger.warning("Failed to get ExtensibilityManager: %s", e)
            return None
        else:
            return extensibility_manager

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "initialized": self._registry_initialized,
            "cached_strategies": len(self._strategy_cache),
            "extensibility_manager_available": bool(self.extensibility_manager),
            **self.performance_tracker.get_strategy_stats(),
        }
