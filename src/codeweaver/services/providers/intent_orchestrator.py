# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent orchestrator service provider."""

import logging
import time

from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import (
    CacheService,
    HealthStatus,
    IntentParsingError,
    IntentResult,
    IntentServiceConfig,
    IntentType,
    ParsedIntent,
    ServiceHealth,
    ServiceIntegrationError,
    ServiceType,
    StrategyExecutionError,
    StrategySelectionError,
)


class IntentOrchestrator(BaseServiceProvider):
    """
    Service-compliant orchestrator for intent processing.

    This service coordinates the entire intent processing pipeline:
    1. Parse natural language intents into structured format
    2. Select appropriate strategy based on intent type and complexity
    3. Execute the strategy with proper service context
    4. Handle errors and provide fallback mechanisms
    5. Cache results for improved performance

    The orchestrator integrates with existing CodeWeaver services:
    - Uses CacheService for result caching
    - Leverages ServicesManager for dependency injection
    - Follows BaseServiceProvider patterns for health monitoring
    """

    def __init__(self, config: IntentServiceConfig, logger: logging.Logger | None = None):
        """Initialize the intent orchestrator with configuration."""
        super().__init__(ServiceType.INTENT, config, logger)
        self.parser = None
        self.strategy_registry = None
        self.cache_service: CacheService | None = None
        self._intent_config = config
        self._intent_stats = {
            "total_processed": 0,
            "successful_intents": 0,
            "failed_intents": 0,
            "cached_hits": 0,
            "avg_processing_time": 0.0,
        }

    async def _initialize_provider(self) -> None:
        """Initialize with existing service dependencies."""
        self._logger.info("Initializing intent orchestrator with service dependencies")
        try:
            from codeweaver.intent.parsing.factory import IntentParserFactory

            parser_config = {
                "type": "pattern",
                "use_nlp_fallback": self._intent_config.use_nlp_fallback,
                "pattern_matching": self._intent_config.pattern_matching,
            }
            self.parser = IntentParserFactory.create(parser_config)
            self.strategy_registry = None
            self.cache_service = await self._get_cache_service()
            self._logger.info("Intent orchestrator initialized successfully")
        except Exception as e:
            self._logger.exception("Failed to initialize intent orchestrator")
            raise ServiceIntegrationError(f"Intent orchestrator initialization failed: {e}") from e

    async def _shutdown_provider(self) -> None:
        """Shutdown provider-specific resources."""
        self._logger.info("Shutting down intent orchestrator")
        self._logger.info(
            "Intent orchestrator statistics: %d total, %d successful, %d failed, %d cached",
            self._intent_stats["total_processed"],
            self._intent_stats["successful_intents"],
            self._intent_stats["failed_intents"],
            self._intent_stats["cached_hits"],
        )
        self.parser = None
        self.strategy_registry = None
        self.cache_service = None

    async def _check_health(self) -> bool:
        """Perform provider-specific health check."""
        try:
            if not self.parser:
                self._logger.warning("Intent parser not available")
                return False
            test_intent = "test search intent"
            parsed = await self.parser.parse(test_intent)
            if not isinstance(parsed, ParsedIntent):
                self._logger.warning("Parser returned invalid result type")
                return False
            if self.cache_service:
                cache_health = await self.cache_service.health_check()
                if cache_health.status == HealthStatus.UNHEALTHY:
                    self._logger.warning("Cache service is unhealthy")
        except Exception as e:
            self._logger.warning("Health check failed: %s", e)
            return False
        else:
            return True

    def _raise_intent_error(self, error: Exception, msg: str | None = None) -> None:
        raise error(msg) if msg else error

    async def process_intent(self, intent_text: str, context: dict[str, Any]) -> IntentResult:
        """
        Process intent with full service integration.

        Args:
            intent_text: Natural language intent from user
            context: Service context with dependencies and metadata

        Returns:
            Result of intent processing with data and metadata
        """
        start_time = time.time()
        self._intent_stats["total_processed"] += 1
        try:
            self._logger.debug("Processing intent: %s", intent_text[:100])
            cache_key = self._generate_cache_key(intent_text)
            if self.cache_service:
                cached_result = await self.cache_service.get(cache_key)
                if cached_result:
                    self._intent_stats["cached_hits"] += 1
                    self._logger.debug("Cache hit for intent")
                    return cached_result
            parsed_intent = await self._parse_intent(intent_text)
            if not self._validate_parsed_intent(parsed_intent):
                self._raise_intent_error(
                    IntentParsingError, "Invalid parsed intent: %s", parsed_intent
                )
            result = await self._execute_strategy(parsed_intent, context)
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.metadata["intent_orchestrator"] = "v1.0.0"
            result.metadata["cache_used"] = bool(self.cache_service)
            if self.cache_service and result.success:
                await self.cache_service.set(cache_key, result, ttl=self._intent_config.cache_ttl)
            if result.success:
                self._intent_stats["successful_intents"] += 1
            else:
                self._intent_stats["failed_intents"] += 1
            total_successful = self._intent_stats["successful_intents"]
            if total_successful > 0:
                current_avg = self._intent_stats["avg_processing_time"]
                self._intent_stats["avg_processing_time"] = (
                    current_avg * (total_successful - 1) + execution_time
                ) / total_successful
        except Exception as e:
            execution_time = time.time() - start_time
            self._intent_stats["failed_intents"] += 1
            self._logger.exception("Intent processing failed: %s", e)
            return await self._execute_fallback(intent_text, context, e, execution_time)
        else:
            return result

    async def _parse_intent(self, intent_text: str) -> ParsedIntent:
        """Parse intent text into structured format."""
        if not self.parser:
            raise ServiceIntegrationError("Parser not available")
        try:
            parsed = await self.parser.parse(intent_text)
            if parsed.intent_type == "INDEX":
                self._logger.warning(
                    "Parser attempted to return INDEX intent - converting to SEARCH"
                )
                parsed.intent_type = IntentType.SEARCH
                parsed.metadata["index_converted"] = True
                parsed.metadata["indexing_note"] = "Indexing handled automatically in background"
        except Exception as e:
            raise IntentParsingError(f"Failed to parse intent '{intent_text}': {e}") from e
        else:
            return parsed

    async def _execute_strategy(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Execute strategy for the parsed intent."""
        try:
            if not self.strategy_registry:
                return await self._basic_fallback_execution(parsed_intent, context)
            strategy = await self.strategy_registry.select_strategy(parsed_intent)
            if not strategy:
                self._raise_intent_error(
                    StrategySelectionError,
                    "No strategy available for intent: %s",
                    parsed_intent.intent_type,
                )
            result = await strategy.execute(parsed_intent, context)
            result.strategy_used = getattr(strategy, "name", "unknown")
        except Exception as e:
            raise StrategyExecutionError(f"Strategy execution failed: {e}") from e
        else:
            return result

    async def _basic_fallback_execution(
        self, parsed_intent: ParsedIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Basic fallback execution until full strategy system is implemented."""
        return IntentResult(
            success=True,
            data={
                "message": f"Intent recognized: {parsed_intent.intent_type.value}",
                "target": parsed_intent.primary_target,
                "scope": parsed_intent.scope.value,
                "note": "Basic fallback - full strategy system pending",
            },
            metadata={
                "intent_type": parsed_intent.intent_type.value,
                "confidence": parsed_intent.confidence,
                "fallback_used": True,
                "background_indexing_note": "Indexing happens automatically in background",
            },
            strategy_used="basic_fallback",
        )

    async def _execute_fallback(
        self, intent_text: str, context: dict[str, Any], error: Exception, execution_time: float
    ) -> IntentResult:
        """Handle errors with fallback logic."""
        error_type = type(error).__name__
        self.record_operation(success=False, error=str(error))
        if isinstance(error, IntentParsingError):
            suggestions = [
                "Try rephrasing your request",
                "Use keywords like 'find', 'search', 'understand', or 'analyze'",
                "Specify what you're looking for (e.g., 'find authentication functions')",
            ]
        elif isinstance(error, StrategyExecutionError):
            suggestions = [
                "Try a simpler search query",
                "Check if the codebase is indexed",
                "Verify your search terms are correct",
            ]
        else:
            suggestions = [
                "Try again with a different query",
                "Contact support if the issue persists",
            ]
        return IntentResult(
            success=False,
            data=None,
            metadata={
                "error_type": error_type,
                "fallback_used": True,
                "execution_time": execution_time,
                "background_indexing_active": True,
            },
            error_message=f"Intent processing failed: {error}",
            suggestions=suggestions,
            execution_time=execution_time,
            strategy_used="error_fallback",
        )

    def _validate_parsed_intent(self, parsed_intent: ParsedIntent) -> bool:
        """Validate parsed intent structure."""
        try:
            if not isinstance(parsed_intent.intent_type, IntentType):
                return False
            if not parsed_intent.primary_target or not isinstance(
                parsed_intent.primary_target, str
            ):
                return False
            if not 0.0 <= parsed_intent.confidence <= 1.0:
                return False
        except Exception:
            return False
        else:
            return True

    def _generate_cache_key(self, intent_text: str) -> str:
        """Generate cache key for intent text."""
        import hashlib

        return f"intent:{hashlib.md5(intent_text.encode()).hexdigest()}"  # noqa: S324

    async def _get_cache_service(self) -> CacheService | None:
        """Get cache service through dependency injection."""
        return None

    async def health_check(self) -> ServiceHealth:
        """Enhanced health check with intent-specific metrics."""
        base_health = await super().health_check()
        base_health.metadata = {
            "total_processed": self._intent_stats["total_processed"],
            "success_rate": self._intent_stats["successful_intents"]
            / max(1, self._intent_stats["total_processed"]),
            "cache_hit_rate": self._intent_stats["cached_hits"]
            / max(1, self._intent_stats["total_processed"]),
            "avg_processing_time": self._intent_stats["avg_processing_time"],
            "parser_available": bool(self.parser),
            "strategy_registry_available": bool(self.strategy_registry),
            "cache_service_available": bool(self.cache_service),
        }
        return base_health
