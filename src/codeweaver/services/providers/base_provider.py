"""Base service provider implementation."""

import asyncio
import contextlib
import logging
import time

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from codeweaver.types import (
    HealthStatus,
    ProviderStatus,
    ServiceCapabilities,
    ServiceConfig,
    ServiceHealth,
    ServiceInitializationError,
    ServiceProvider,
    ServiceStopError,
    ServiceType,
)
from codeweaver.utils.decorators import require_implementation


class BaseServiceProvider(ServiceProvider, ABC):
    """Base implementation for all service providers."""

    @require_implementation("_initialize_provider", "_shutdown_provider")
    def __init__(
        self, service_type: ServiceType, config: ServiceConfig, logger: logging.Logger | None = None
    ):
        """Initialize the service provider with type and configuration."""
        self._service_type = service_type
        self._config = config
        self._logger = logger or logging.getLogger(f"codeweaver.services.{self.name}")
        self._status = ProviderStatus.REGISTERED
        self._initialized = False
        self._started_at: datetime | None = None
        self._health_stats = {
            "errors": 0,
            "successful_operations": 0,
            "last_error": None,
            "last_response_time": 0.0,
        }
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return self._config.provider

    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"

    @property
    def service_type(self) -> ServiceType:
        """Type of service this provider implements."""
        return self._service_type

    @property
    def config(self) -> ServiceConfig:
        """Provider configuration."""
        return self._config

    @property
    def status(self) -> ProviderStatus:
        """Current provider status."""
        return self._status

    @property
    def capabilities(self) -> ServiceCapabilities:
        """Provider capabilities."""
        return ServiceCapabilities(
            supports_streaming=False,
            supports_batch=True,
            supports_async=True,
            max_concurrency=self._config.metadata.get("max_concurrency", 10),
            memory_usage="medium",
            performance_profile="standard",
        )

    async def start(self) -> None:
        """Alias for `initialize`; starts the service provider."""
        await self.initialize()

    async def stop(self) -> None:
        """Alias for `shutdown`; stops the service provider."""
        await self.shutdown()

    async def initialize(self) -> None:
        """Initialize the service provider."""
        if self._initialized:
            self._logger.warning("Provider %s already initialized", self.name)
            return
        try:
            self._status = ProviderStatus.INITIALIZING
            self._logger.info("Initializing service provider: %s", self.name)
            await self._validate_config()
            await self._initialize_provider()
            if self._config.health_check_interval > 0:
                await self._start_health_monitoring()
            self._initialized = True
            self._started_at = datetime.now(UTC)
            self._status = ProviderStatus.READY
            self._logger.info("Service provider initialized successfully: %s", self.name)
        except Exception as e:
            self._status = ProviderStatus.ERROR
            error_msg = f"Failed to initialize provider {self.name}: {e}"
            self._logger.exception(error_msg)
            raise ServiceInitializationError(error_msg) from e

    async def shutdown(self) -> None:
        """Shutdown the service provider gracefully."""
        if not self._initialized:
            return
        try:
            self._logger.info("Shutting down service provider: %s", self.name)
            self._shutdown_event.set()
            if self._health_check_task and (not self._health_check_task.done()):
                self._health_check_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._health_check_task
            await self._shutdown_provider()
            self._initialized = False
            self._status = ProviderStatus.REGISTERED
            self._logger.info("Service provider shut down successfully: %s", self.name)
        except Exception as e:
            error_msg = f"Failed to shutdown provider {self.name}: {e}"
            self._logger.exception(error_msg)
            raise ServiceStopError(error_msg) from e

    async def health_check(self) -> ServiceHealth:
        """Check if the service is healthy."""
        start_time = time.time()
        try:
            is_healthy = await self._check_health()
            response_time = time.time() - start_time
            self._health_stats["last_response_time"] = response_time
            if is_healthy:
                self._health_stats["successful_operations"] += 1
            else:
                self._health_stats["errors"] += 1
            success_rate = self._calculate_success_rate()
            status = self._determine_health_status(
                is_healthy=is_healthy, success_rate=success_rate, response_time=response_time
            )
            uptime = 0.0
            if self._started_at:
                uptime = (datetime.now(UTC) - self._started_at).total_seconds()
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Health check failed: {e}"
            self._health_stats["errors"] += 1
            self._health_stats["last_error"] = error_msg
            self._health_stats["last_response_time"] = response_time
            self._logger.warning("Health check failed for %s: %s", self.name, error_msg)
            return ServiceHealth(
                service_type=self._service_type,
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(UTC),
                response_time=response_time,
                error_count=self._health_stats["errors"],
                success_rate=self._calculate_success_rate(),
                last_error=error_msg,
                uptime=0.0,
            )
        else:
            return ServiceHealth(
                service_type=self._service_type,
                status=status,
                last_check=datetime.now(UTC),
                response_time=response_time,
                error_count=self._health_stats["errors"],
                success_rate=success_rate,
                last_error=self._health_stats.get("last_error"),
                uptime=uptime,
                memory_usage=0,
            )

    def record_operation(self, *, success: bool, error: str | None = None) -> None:
        """Record an operation result for health tracking."""
        if success:
            self._health_stats["successful_operations"] += 1
        else:
            self._health_stats["errors"] += 1
            if error:
                self._health_stats["last_error"] = error

    @abstractmethod
    async def _initialize_provider(self) -> None:
        """Initialize provider-specific resources."""

    @abstractmethod
    async def _shutdown_provider(self) -> None:
        """Shutdown provider-specific resources."""

    @abstractmethod
    async def _check_health(self) -> bool:
        """Perform provider-specific health check."""

    async def _validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._config.provider:
            raise ServiceInitializationError("Provider name is required")
        if self._config.timeout <= 0:
            raise ServiceInitializationError("Timeout must be positive")
        if self._config.max_retries < 0:
            raise ServiceInitializationError("Max retries cannot be negative")

    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._health_check_task:
            return
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._config.health_check_interval
                )
                break
            except TimeoutError:
                try:
                    await self.health_check()
                except Exception as e:
                    self._logger.warning("Health monitoring failed for %s: %s", self.name, e)

    def _calculate_success_rate(self) -> float:
        """Calculate operation success rate."""
        total_ops = self._health_stats["successful_operations"] + self._health_stats["errors"]
        if total_ops == 0:
            return 1.0
        return self._health_stats["successful_operations"] / total_ops

    def _determine_health_status(
        self, *, is_healthy: bool, success_rate: float, response_time: float
    ) -> HealthStatus:
        """Determine health status based on metrics."""
        if not is_healthy:
            return HealthStatus.UNHEALTHY
        if success_rate < 0.5:
            return HealthStatus.UNHEALTHY
        if success_rate < 0.8 or response_time > self._config.timeout * 0.8:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
