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
    MetricsService,
    MonitoringService,
    ParsedIntent,
    ServiceHealth,
    ServiceIntegrationError,
    ServiceType,
    StrategyExecutionError,
    StrategySelectionError,
    TelemetryService,
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
        self.metrics_service: MetricsService | None = None
        self.monitoring_service: MonitoringService | None = None
        self.telemetry_service: TelemetryService | None = None
        self._intent_config = config
        self._intent_stats = {
            "total_processed": 0,
            "successful_intents": 0,
            "failed_intents": 0,
            "cached_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "min_processing_time": float("inf"),
            "max_processing_time": 0.0,
            "parsing_failures": 0,
            "strategy_failures": 0,
            "concurrent_requests": 0,
        }
        # Performance optimization settings
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = config.circuit_breaker_threshold
        self._circuit_breaker_reset_time = config.circuit_breaker_reset_time
        self._last_circuit_breaker_failure = 0

        # Performance thresholds
        self._performance_excellent_threshold = config.performance_excellent_threshold
        self._performance_good_threshold = config.performance_good_threshold
        self._performance_acceptable_threshold = config.performance_acceptable_threshold

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

            # Initialize monitoring and performance services
            self.cache_service = await self._get_cache_service()
            self.metrics_service = await self._get_metrics_service()
            self.monitoring_service = await self._get_monitoring_service()
            self.telemetry_service = await self._get_telemetry_service()

            # Register this service with monitoring if available
            if self.monitoring_service:
                await self.monitoring_service.add_service(ServiceType.INTENT, self)

            self._logger.info(
                "Intent orchestrator initialized successfully with monitoring services"
            )
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

        # Unregister from monitoring service
        if self.monitoring_service:
            try:
                await self.monitoring_service.remove_service(ServiceType.INTENT)
            except Exception as e:
                self._logger.warning("Failed to unregister from monitoring service: %s", e)

        # Final telemetry flush
        if self.telemetry_service:
            try:
                await self.telemetry_service.flush()
            except Exception as e:
                self._logger.warning("Failed to flush telemetry: %s", e)

        self.parser = None
        self.strategy_registry = None
        self.cache_service = None
        self.metrics_service = None
        self.monitoring_service = None
        self.telemetry_service = None

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
        Process intent with full service integration and performance monitoring.

        Args:
            intent_text: Natural language intent from user
            context: Service context with dependencies and metadata

        Returns:
            Result of intent processing with data and metadata
        """
        # Setup processing context
        start_time, operation_id = self._setup_processing_context()

        # Circuit breaker check
        if await self._is_circuit_breaker_open():
            self._finalize_processing_context()
            return await self._circuit_breaker_fallback(intent_text)

        try:
            self._logger.debug("Processing intent: %s", intent_text[:100])

            # Record start metrics
            await self._record_start_metrics()

            # Check cache first
            cached_result = await self._check_cache_for_intent(intent_text, start_time)
            if cached_result:
                self._finalize_processing_context()
                return cached_result

            # Core intent processing
            parsed_intent, result = await self._process_core_intent(intent_text, context)
            execution_time = time.time() - start_time

            # Update performance statistics
            self._update_performance_stats(execution_time, success=result.success)

            # Enhance result metadata
            self._enhance_result_metadata(result, execution_time, operation_id)

            # Cache successful results
            await self._store_cache_result(intent_text, result, parsed_intent)

            # Record success metrics and telemetry
            await self._record_success_metrics(parsed_intent, result, execution_time, operation_id)

            # Reset circuit breaker on success
            if result.success:
                self._circuit_breaker_failures = 0

        except Exception as e:
            execution_time = time.time() - start_time
            return await self._handle_processing_error(intent_text, context, e, execution_time)
        else:
            return result

        finally:
            self._finalize_processing_context()

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
        from datetime import UTC, datetime

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
            executed_at=datetime.now(UTC),
            execution_time=0.0,  # Will be updated by caller
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
        from datetime import UTC, datetime

        return IntentResult(
            success=False,
            data=None,
            metadata={
                "error_type": error_type,
                "fallback_used": True,
                "execution_time": execution_time,
                "background_indexing_active": True,
            },
            executed_at=datetime.now(UTC),
            execution_time=execution_time,
            error_message=f"Intent processing failed: {error}",
            suggestions=suggestions,
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

    async def _check_cache_for_intent(self, intent_text: str, start_time: float) -> IntentResult | None:
        """Check cache for existing intent result and record metrics."""
        if not self.cache_service:
            return None

        cache_key = self._generate_cache_key(intent_text)
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            self._intent_stats["cached_hits"] += 1
            execution_time = time.time() - start_time

            # Record cache hit metrics
            if self.metrics_service:
                await self.metrics_service.record_timing(
                    "intent.processing.cache_hit",
                    execution_time * 1000,
                    tags={"cached": "true"},
                )
                await self.metrics_service.increment_counter("intent.cache.hits")

            # Track telemetry for cache hit
            if self.telemetry_service:
                await self.telemetry_service.track_performance(
                    operation="intent_processing",
                    duration=execution_time,
                    metadata={"cache_hit": True, "intent_type": "cached"},
                )

            self._logger.debug("Cache hit for intent")
            return cached_result
        self._intent_stats["cache_misses"] += 1
        if self.metrics_service:
            await self.metrics_service.increment_counter("intent.cache.misses")
        return None

    async def _store_cache_result(self, intent_text: str, result: IntentResult, parsed_intent: ParsedIntent) -> None:
        """Store successful result in cache."""
        if self.cache_service and result.success:
            cache_key = self._generate_cache_key(intent_text)
            await self.cache_service.set(
                cache_key,
                result,
                ttl=self._intent_config.cache_ttl,
                tags=["intent", parsed_intent.intent_type.value],
            )

    async def _get_cache_service(self) -> CacheService | None:
        """Get cache service through dependency injection."""
        # TODO: Implement proper service manager integration
        return None

    async def _get_metrics_service(self) -> MetricsService | None:
        """Get metrics service through dependency injection."""
        # TODO: Implement proper service manager integration
        return None

    async def _get_monitoring_service(self) -> MonitoringService | None:
        """Get monitoring service through dependency injection."""
        # TODO: Implement proper service manager integration
        return None

    async def _get_telemetry_service(self) -> TelemetryService | None:
        """Get telemetry service through dependency injection."""
        # TODO: Implement proper service manager integration
        return None

    # Performance optimization methods

    async def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open due to too many failures."""
        if self._circuit_breaker_failures < self._circuit_breaker_threshold:
            return False

        # Check if enough time has passed to attempt reset
        if time.time() - self._last_circuit_breaker_failure > self._circuit_breaker_reset_time:
            self._circuit_breaker_failures = 0
            self._logger.info("Circuit breaker reset - attempting to process requests")
            return False

        return True

    async def _circuit_breaker_fallback(self, intent_text: str) -> IntentResult:
        """Handle requests when circuit breaker is open."""
        self._logger.warning("Circuit breaker is open - using fallback response")
        from datetime import UTC, datetime

        return IntentResult(
            success=False,
            data=None,
            metadata={
                "circuit_breaker": "open",
                "failure_count": self._circuit_breaker_failures,
                "retry_after": self._circuit_breaker_reset_time,
            },
            executed_at=datetime.now(UTC),
            execution_time=0.0,
            error_message="Service temporarily unavailable due to repeated failures",
            suggestions=[
                "Wait a moment and try again",
                "Try a simpler query",
                "Check system status",
            ],
            strategy_used="circuit_breaker_fallback",
        )

    def _update_performance_stats(self, execution_time: float, *, success: bool) -> None:
        """Update internal performance statistics."""
        if success:
            self._intent_stats["successful_intents"] += 1
        else:
            self._intent_stats["failed_intents"] += 1

        # Update timing statistics
        self._intent_stats["min_processing_time"] = min(
            self._intent_stats["min_processing_time"], execution_time
        )
        self._intent_stats["max_processing_time"] = max(
            self._intent_stats["max_processing_time"], execution_time
        )

        # Update rolling average
        total_successful = self._intent_stats["successful_intents"]
        if total_successful > 0:
            current_avg = self._intent_stats["avg_processing_time"]
            self._intent_stats["avg_processing_time"] = (
                current_avg * (total_successful - 1) + execution_time
            ) / total_successful

    def _get_performance_tier(self, execution_time: float) -> str:
        """Classify performance tier based on configurable execution time thresholds."""
        if execution_time < self._performance_excellent_threshold:
            return "excellent"
        if execution_time < self._performance_good_threshold:
            return "good"
        if execution_time < self._performance_acceptable_threshold:
            return "acceptable"
        return "slow"

    async def _record_processing_metrics(
        self,
        parsed_intent: ParsedIntent,
        result: IntentResult,
        execution_time: float,
        operation_id: str,
    ) -> None:
        """Record comprehensive processing metrics."""
        if not self.metrics_service:
            return

        try:
            # Basic timing metrics
            await self.metrics_service.record_timing(
                "intent.processing.duration",
                execution_time * 1000,
                tags={
                    "intent_type": parsed_intent.intent_type.value,
                    "success": str(result.success).lower(),
                    "strategy": result.strategy_used or "unknown",
                },
            )

            # Success/failure counters
            metric_name = (
                "intent.requests.successful" if result.success else "intent.requests.failed"
            )
            await self.metrics_service.increment_counter(
                metric_name,
                tags={
                    "intent_type": parsed_intent.intent_type.value,
                    "complexity": parsed_intent.complexity.value,
                },
            )

            # Performance tier metrics
            await self.metrics_service.increment_counter(
                "intent.performance.tier",
                tags={
                    "tier": self._get_performance_tier(execution_time),
                    "intent_type": parsed_intent.intent_type.value,
                },
            )

            # Record concurrent requests gauge
            await self.metrics_service.record_metric(
                "intent.concurrent_requests", float(self._intent_stats["concurrent_requests"])
            )

        except Exception as e:
            self._logger.warning("Failed to record processing metrics: %s", e)

    async def _track_telemetry(
        self, parsed_intent: ParsedIntent, result: IntentResult, execution_time: float
    ) -> None:
        """Track telemetry for intent processing."""
        if not self.telemetry_service:
            return

        try:
            # Track performance
            await self.telemetry_service.track_performance(
                operation="intent_processing",
                duration=execution_time,
                metadata={
                    "intent_type": parsed_intent.intent_type.value,
                    "complexity": parsed_intent.complexity.value,
                    "confidence": parsed_intent.confidence,
                    "success": result.success,
                    "strategy_used": result.strategy_used,
                    "cache_hit": result.metadata.get("cache_hit", False),
                },
            )

            # Track specific intent event
            await self.telemetry_service.track_event(
                event_name="intent_processed",
                properties={
                    "intent_type": parsed_intent.intent_type.value,
                    "execution_time_ms": execution_time * 1000,
                    "success": result.success,
                    "complexity": parsed_intent.complexity.value,
                    "confidence_score": parsed_intent.confidence,
                    "strategy": result.strategy_used,
                    "performance_tier": self._get_performance_tier(execution_time),
                },
            )

        except Exception as e:
            self._logger.warning("Failed to track telemetry: %s", e)

    async def _record_start_metrics(self) -> None:
        """Record metrics at the start of intent processing."""
        if self.metrics_service:
            await self.metrics_service.increment_counter(
                "intent.requests.total", tags={"operation": "process_intent"}
            )

    async def _record_success_metrics(
        self,
        parsed_intent: ParsedIntent,
        result: IntentResult,
        execution_time: float,
        operation_id: str
    ) -> None:
        """Record all success-related metrics and telemetry."""
        # Record processing metrics
        await self._record_processing_metrics(parsed_intent, result, execution_time, operation_id)

        # Track telemetry
        await self._track_telemetry(parsed_intent, result, execution_time)

    async def _record_failure_metrics(self, error: Exception, execution_time: float) -> None:
        """Record failure metrics and telemetry."""
        if self.metrics_service:
            await self.metrics_service.increment_counter(
                "intent.requests.failed", tags={"error_type": type(error).__name__}
            )
            await self.metrics_service.record_timing(
                "intent.processing.failed", execution_time * 1000
            )

        # Track error telemetry
        if self.telemetry_service:
            await self.telemetry_service.track_error(
                error_type=type(error).__name__,
                error_category="intent_processing",
                operation="process_intent",
                error_message=str(error),
            )

    def _setup_processing_context(self) -> tuple[float, str]:
        """Setup processing context and return start time and operation ID."""
        start_time = time.time()
        operation_id = f"intent_{int(start_time * 1000)}"
        self._intent_stats["total_processed"] += 1
        self._intent_stats["concurrent_requests"] += 1
        return start_time, operation_id

    def _finalize_processing_context(self) -> None:
        """Clean up processing context."""
        self._intent_stats["concurrent_requests"] -= 1

    async def _process_core_intent(self, intent_text: str, context: dict[str, Any]) -> tuple[ParsedIntent, IntentResult]:
        """Parse intent and execute strategy (core processing logic)."""
        # Parse intent
        parsed_intent = await self._parse_intent(intent_text)

        if not self._validate_parsed_intent(parsed_intent):
            self._raise_intent_error(
                IntentParsingError, f"Invalid parsed intent: {parsed_intent}"
            )

        # Execute strategy
        result = await self._execute_strategy(parsed_intent, context)
        return parsed_intent, result

    async def _handle_processing_error(
        self,
        intent_text: str,
        context: dict[str, Any],
        error: Exception,
        execution_time: float
    ) -> IntentResult:
        """Handle processing errors with comprehensive error tracking and fallback."""
        self._intent_stats["failed_intents"] += 1
        self._circuit_breaker_failures += 1
        self._last_circuit_breaker_failure = time.time()

        # Record failure metrics
        await self._record_failure_metrics(error, execution_time)

        self._logger.warning("Intent processing failed.")
        return await self._execute_fallback(intent_text, context, error, execution_time)

    def _enhance_result_metadata(self, result: IntentResult, execution_time: float, operation_id: str) -> None:
        """Enhance result with metadata and performance information."""
        result.execution_time = execution_time
        result.metadata.update({
            "intent_orchestrator": "v1.1.0",
            "cache_used": bool(self.cache_service),
            "monitoring_enabled": bool(self.metrics_service),
            "operation_id": operation_id,
            "performance_tier": self._get_performance_tier(execution_time),
        })

    async def get_capabilities(self) -> dict[str, Any]:
        """Get enhanced intent orchestrator capabilities."""
        total_processed = self._intent_stats["total_processed"]
        success_rate = self._intent_stats["successful_intents"] / max(1, total_processed)
        cache_hit_rate = self._intent_stats["cached_hits"] / max(1, total_processed)

        return {
            "intent_types": ["SEARCH", "UNDERSTAND", "ANALYZE"],
            "complexity_levels": ["SIMPLE", "MODERATE", "COMPLEX"],
            "parser_type": "pattern" if self.parser else None,
            "services": {
                "cache_enabled": bool(self.cache_service),
                "metrics_enabled": bool(self.metrics_service),
                "monitoring_enabled": bool(self.monitoring_service),
                "telemetry_enabled": bool(self.telemetry_service),
            },
            "strategy_registry_enabled": bool(self.strategy_registry),
            "performance": {
                "circuit_breaker_enabled": True,
                "circuit_breaker_threshold": self._circuit_breaker_threshold,
                "current_failures": self._circuit_breaker_failures,
                "performance_tiers": ["excellent", "good", "acceptable", "slow"],
            },
            "processing_stats": {
                "total_processed": total_processed,
                "successful_intents": self._intent_stats["successful_intents"],
                "failed_intents": self._intent_stats["failed_intents"],
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "avg_processing_time": self._intent_stats["avg_processing_time"],
                "min_processing_time": self._intent_stats["min_processing_time"]
                if total_processed > 0
                else 0,
                "max_processing_time": self._intent_stats["max_processing_time"],
                "concurrent_requests": self._intent_stats["concurrent_requests"],
            },
        }

    async def health_check(self) -> ServiceHealth:
        """Enhanced health check with comprehensive intent monitoring data."""
        # For now, use the base health check and log additional metrics
        # TODO: Consider extending ServiceHealth to include metadata field
        base_health = await super().health_check()

        # Log comprehensive metrics for monitoring
        total_processed = self._intent_stats["total_processed"]
        success_rate = self._intent_stats["successful_intents"] / max(1, total_processed)
        cache_hit_rate = self._intent_stats["cached_hits"] / max(1, total_processed)

        self._logger.debug(
            "Intent orchestrator health metrics: processed=%d, success_rate=%.2f, "
            "cache_hit_rate=%.2f, avg_time=%.3f, circuit_breaker_failures=%d",
            total_processed,
            success_rate,
            cache_hit_rate,
            self._intent_stats["avg_processing_time"],
            self._circuit_breaker_failures,
        )

        return base_health
