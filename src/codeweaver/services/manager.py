# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Services manager for coordinating all service providers."""

import asyncio
import contextlib
import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from fastmcp import FastMCP

from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.services.providers.chunking import ChunkingService
from codeweaver.services.providers.file_filtering import FilteringService
from codeweaver.types import (
    CacheService,
    ErrorHandlingService,
    HealthStatus,
    LoggingService,
    MetricsService,
    MonitoringService,
    RateLimitingService,
    ServiceConfig,
    ServiceCreationError,
    ServiceInitializationError,
    ServiceNotFoundError,
    ServiceProvider,
    ServicesConfig,
    ServicesHealthReport,
    ServiceType,
    TelemetryService,
    TimingService,
    ValidationService,
)


class ServicesManager:
    """Manager for coordinating all service providers."""

    def __init__(
        self,
        config: ServicesConfig,
        logger: logging.Logger | None = None,
        fastmcp_server: "FastMCP | None" = None,  # NEW: FastMCP server reference
    ):
        """Initialize the services manager with configuration, logger, and FastMCP server."""
        self._config = config
        self._logger = logger or logging.getLogger("codeweaver.services.manager")
        self._fastmcp_server = fastmcp_server  # NEW: Store FastMCP server reference

        # Service registry
        self._registry = ServiceRegistry(self._logger)

        # Service instances
        self._services: dict[ServiceType, ServiceProvider] = {}

        # NEW: Middleware service tracking
        self._middleware_services: dict[ServiceType, ServiceProvider] = {}
        self._middleware_registration_order: list[ServiceType] = []

        # State management
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._health_monitor_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the services manager and all configured services."""
        if self._initialized:
            self._logger.warning("Services manager already initialized")
            return

        try:
            self._logger.info("Initializing services manager")

            # Register built-in providers
            await self._register_builtin_providers()

            # Create and initialize core services
            await self._create_core_services()

            # Create optional services if enabled
            await self._create_optional_services()

            # NEW: Create and register middleware services
            await self._create_middleware_services()

            # Start health monitoring if enabled
            if self._config.health_check_enabled:
                await self._start_health_monitoring()

            self._initialized = True
            self._logger.info("Services manager initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize services manager: {e}"
            self._logger.exception(error_msg)
            raise ServiceInitializationError(error_msg) from e

    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        if not self._initialized:
            return

        try:
            self._logger.info("Shutting down services manager")

            # Signal shutdown
            self._shutdown_event.set()

            # Stop health monitoring
            if self._health_monitor_task and not self._health_monitor_task.done():
                self._health_monitor_task.cancel()
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(self._health_monitor_task, timeout=5.0)

            # Shutdown all services
            await self._registry.shutdown_all()

            self._services.clear()
            self._initialized = False

            self._logger.info("Services manager shut down successfully")

        except Exception:
            self._logger.exception("Error during services manager shutdown")

    def get_chunking_service(self) -> ChunkingService:
        """Get the chunking service."""
        if service := self._services.get(ServiceType.CHUNKING):
            return service
        raise ServiceNotFoundError(ServiceType.CHUNKING)

    def get_filtering_service(self) -> FilteringService:
        """Get the filtering service."""
        if service := self._services.get(ServiceType.FILTERING):
            return service
        raise ServiceNotFoundError(ServiceType.FILTERING)

    def get_validation_service(self) -> ValidationService | None:
        """Get the validation service if available."""
        service = self._services.get(ServiceType.VALIDATION)
        return service or None

    def get_cache_service(self) -> CacheService | None:
        """Get the cache service if available."""
        service = self._services.get(ServiceType.CACHE)
        return service or None

    def get_monitoring_service(self) -> MonitoringService | None:
        """Get the monitoring service if available."""
        service = self._services.get(ServiceType.MONITORING)
        return service or None

    def get_metrics_service(self) -> MetricsService | None:
        """Get the metrics service if available."""
        service = self._services.get(ServiceType.METRICS)
        return service or None

    def get_telemetry_service(self) -> "TelemetryService | None":
        """Get the telemetry service if available."""
        service = self._services.get(ServiceType.TELEMETRY)
        return service or None

    def get_service(self, service_type: ServiceType) -> ServiceProvider | None:
        """Get any service by type."""
        return self._services.get(service_type)

    def list_active_services(self) -> dict[ServiceType, ServiceProvider]:
        """List all active services."""
        return self._services.copy()

    def start_all_services(self) -> None:
        """Start all services that support starting."""
        for service in self._services.values():
            if hasattr(service, "start"):
                try:
                    service.start()
                    self._logger.info("Started service: %s", service.service_type.value)
                except Exception:
                    self._logger.exceptions("Failed to start service %s", service.service_type.value)

    def stop_all_services(self) -> None:
        """Stop all services that support stopping."""
        for service in self._services.values():
            if hasattr(service, "stop"):
                try:
                    service.stop()
                    self._logger.info("Stopped service: %s", service.service_type.value)
                except Exception:
                    self._logger.exception("Failed to stop service %s", service.service_type.value)

    def get_logging_service(self) -> LoggingService | None:
        """Get the logging middleware service."""
        return self._middleware_services.get(ServiceType.LOGGING)

    def get_timing_service(self) -> TimingService | None:
        """Get the timing middleware service."""
        return self._middleware_services.get(ServiceType.TIMING)

    def get_error_handling_service(self) -> ErrorHandlingService | None:
        """Get the error handling middleware service."""
        return self._middleware_services.get(ServiceType.ERROR_HANDLING)

    def get_rate_limiting_service(self) -> RateLimitingService | None:
        """Get the rate limiting middleware service."""
        return self._middleware_services.get(ServiceType.RATE_LIMITING)

    def get_middleware_service(self, service_type: ServiceType) -> ServiceProvider | None:
        """Get a middleware service by type."""
        return self._middleware_services.get(service_type)

    def list_middleware_services(self) -> dict[ServiceType, ServiceProvider]:
        """Get all middleware services."""
        return self._middleware_services.copy()

    def create_service_context(self) -> dict[str, Any]:
        """Create a service context for provider operations."""
        return {
            "services_manager": self,
            "chunking_service": self.get_chunking_service() if ServiceType.CHUNKING in self._services else None,
            "filtering_service": self.get_filtering_service() if ServiceType.FILTERING in self._services else None,
            "telemetry_service": self.get_telemetry_service(),
            "validation_service": self.get_validation_service(),
            "cache_service": self.get_cache_service(),
            "monitoring_service": self.get_monitoring_service(),
            "metrics_service": self.get_metrics_service(),
        }

    async def get_service_health(self) -> dict[str, Any]:
        """Get service health information."""
        health_report = await self.get_health_report()
        return {
            "overall_status": health_report.overall_status.value,
            "services": {
                service_type.value: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "error_count": health.error_count,
                    "success_rate": health.success_rate,
                }
                for service_type, health in health_report.services.items()
            },
            "check_time": health_report.check_time.isoformat(),
        }

    async def get_health_report(self) -> ServicesHealthReport:
        """Get comprehensive health report for all services."""
        services_health = {}
        overall_status = HealthStatus.HEALTHY

        for service_type, service in self._services.items():
            try:
                health = await service.health_check()
                services_health[service_type] = health

                # Determine overall status
                if health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    health.status == HealthStatus.DEGRADED
                    and overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                self._logger.warning("Health check failed for %s: %s", service_type.value, e)
                # Create unhealthy status for failed check
                from codeweaver.types import ServiceHealth

                services_health[service_type] = ServiceHealth(
                    service_type=service_type,
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.now(UTC),
                    response_time=0.0,
                    error_count=1,
                    success_rate=0.0,
                    last_error=str(e),
                    uptime=0.0,
                )
                overall_status = HealthStatus.UNHEALTHY

        # Get additional metrics
        registry_health = await self._registry.health_check()
        metrics = {
            "total_services": len(self._services),
            "registry_health": registry_health,
            "config_version": str(hash(str(self._config))),
        }

        return ServicesHealthReport(
            overall_status=overall_status, services=services_health, check_time=UTC, metrics=metrics
        )

    async def restart_service(self, service_type: ServiceType) -> None:
        """Restart a specific service."""
        if service_type not in self._services:
            raise ServiceNotFoundError(service_type)

        self._logger.info("Restarting service: %s", service_type.value)

        # Get service config
        config = self._get_service_config(service_type)

        # Shutdown existing service
        await self._registry.destroy_service(service_type)
        del self._services[service_type]

        # Recreate service
        service = await self._registry.create_service(service_type, config)
        self._services[service_type] = service

        self._logger.info("Service restarted successfully: %s", service_type.value)

    async def _register_builtin_providers(self) -> None:
        """Register all built-in service providers including middleware providers."""
        try:
            # Register core service providers
            self._registry.register_provider(
                ServiceType.CHUNKING, "fastmcp_chunking", ChunkingService
            )
            self._registry.register_provider(
                ServiceType.CHUNKING, "ast_grep_chunking", ChunkingService
            )
            self._registry.register_provider(
                ServiceType.FILTERING, "fastmcp_filtering", FilteringService
            )
            
            # Register provider aliases for backward compatibility
            self._registry.register_provider(
                ServiceType.FILTERING, "gitignore_filtering", FilteringService
            )
            self._registry.register_provider(
                ServiceType.CHUNKING, "basic_chunking", ChunkingService
            )

            # NEW: Register middleware service providers
            from codeweaver.services.providers.middleware import (
                FastMCPErrorHandlingProvider,
                FastMCPLoggingProvider,
                FastMCPRateLimitingProvider,
                FastMCPTimingProvider,
            )

            # Register middleware providers
            self._registry.register_provider(
                ServiceType.LOGGING, "fastmcp_logging", FastMCPLoggingProvider
            )
            self._registry.register_provider(
                ServiceType.TIMING, "fastmcp_timing", FastMCPTimingProvider
            )
            self._registry.register_provider(
                ServiceType.ERROR_HANDLING, "fastmcp_error_handling", FastMCPErrorHandlingProvider
            )
            self._registry.register_provider(
                ServiceType.RATE_LIMITING, "fastmcp_rate_limiting", FastMCPRateLimitingProvider
            )

            self._logger.info("Middleware service providers registered")

            # Register telemetry service provider
            from codeweaver.services.providers.telemetry import PostHogTelemetryProvider

            self._registry.register_provider(
                ServiceType.TELEMETRY, "posthog_telemetry", PostHogTelemetryProvider
            )

            self._logger.info("Telemetry service provider registered")

            # TODO: Register other built-in providers when available

            self._logger.info("Built-in service providers registered")

        except Exception as e:
            error_msg = f"Failed to register builtin providers: {e}"
            self._logger.exception(error_msg)
            raise ServiceInitializationError(error_msg) from e

    async def _create_core_services(self) -> None:
        """Create and initialize core services."""
        core_services = ServiceType.get_core_services()

        for service_type in core_services:
            config = self._get_service_config(service_type)
            if config.enabled:
                try:
                    service = await self._registry.create_service(service_type, config)
                    self._services[service_type] = service
                    self._logger.info("Core service created: %s", service_type.value)

                except Exception as e:
                    error_msg = f"Failed to create core service {service_type.value}: {e}"
                    self._logger.exception(error_msg)
                    raise ServiceCreationError(service_type, str(e)) from e
            else:
                self._logger.warning("Core service disabled: %s", service_type.value)

    async def _create_optional_services(self) -> None:
        """Create optional services if enabled."""
        optional_services = ServiceType.get_optional_services()

        for service_type in optional_services:
            config = self._get_service_config(service_type)
            if config.enabled:
                try:
                    # Create telemetry service if it's available
                    if service_type == ServiceType.TELEMETRY:
                        service = await self._registry.create_service(service_type, config)
                        self._services[service_type] = service
                        self._logger.info("Optional service created: %s", service_type.value)
                    else:
                        # TODO: Implement other optional service providers
                        self._logger.info(
                            "Optional service %s enabled but not implemented yet", service_type.value
                        )

                except Exception as e:
                    # Optional services failures are not fatal
                    self._logger.warning(
                        "Failed to create optional service %s: %s", service_type.value, e
                    )

    async def _create_middleware_services(self) -> None:
        """Create and register middleware services with FastMCP server."""
        if not self._fastmcp_server:
            self._logger.warning(
                "No FastMCP server provided, skipping middleware service registration"
            )
            return

        try:
            self._logger.info("Creating middleware services")

            # Use configured initialization order
            middleware_order = self._config.middleware_initialization_order or [
                "error_handling",
                "rate_limiting",
                "logging",
                "timing",
            ]

            for service_name in middleware_order:
                service_type = ServiceType(service_name)

                # Get service configuration
                service_config = getattr(self._config, service_name, None)
                if not service_config or not service_config.enabled:
                    self._logger.debug("Middleware service %s disabled, skipping", service_name)
                    continue

                # Create middleware service
                middleware_service = await self._create_middleware_service(
                    service_type, service_config
                )
                if middleware_service:
                    self._middleware_services[service_type] = middleware_service
                    self._middleware_registration_order.append(service_type)

                    # Register middleware with FastMCP server if auto-registration is enabled
                    if self._config.middleware_auto_registration:
                        await self._register_middleware_with_server(
                            service_type, middleware_service
                        )

            self._logger.info("Middleware services created and registered")

        except Exception as e:
            error_msg = f"Failed to create middleware services: {e}"
            self._logger.exception(error_msg)
            raise ServiceInitializationError(error_msg) from e

    async def _create_middleware_service(
        self, service_type: ServiceType, config: "ServiceConfig"
    ) -> ServiceProvider | None:
        """Create a single middleware service."""
        try:
            # Create service using registry
            service = await self._registry.create_service(service_type, config)

            # Initialize the service
            await service.initialize()

        except Exception as e:
            self._logger.exception("Failed to create middleware service %s")

            if getattr(config, 'fail_fast', True):
                raise ServiceCreationError(service_type, f"Failed to initialize {service_type.value} service") from e
            return None
        else:
            self._logger.info("Created middleware service: %s", service_type.value)
            return service

    async def _register_middleware_with_server(
        self, service_type: ServiceType, service: ServiceProvider
    ) -> None:
        """Register middleware service with FastMCP server."""
        try:
            if hasattr(service, "get_middleware_instance"):
                middleware_instance = service.get_middleware_instance()
                self._fastmcp_server.add_middleware(middleware_instance)
                self._logger.info(
                    "Registered %s middleware with FastMCP server", service_type.value
                )
            else:
                self._logger.warning(
                    "Service %s does not provide middleware instance", service_type.value
                )
        except Exception:
            self._logger.exception("Failed to register %s middleware with server")

            raise

    def _get_service_config(self, service_type: ServiceType) -> Any:
        """Get configuration for a specific service type."""
        config_map = {
            # Core services
            ServiceType.CHUNKING: self._config.chunking,
            ServiceType.FILTERING: self._config.filtering,
            # Middleware services
            ServiceType.LOGGING: self._config.logging,
            ServiceType.TIMING: self._config.timing,
            ServiceType.ERROR_HANDLING: self._config.error_handling,
            ServiceType.RATE_LIMITING: self._config.rate_limiting,
            # Optional services
            ServiceType.VALIDATION: self._config.validation,
            ServiceType.CACHE: self._config.cache,
            ServiceType.MONITORING: self._config.monitoring,
            ServiceType.METRICS: self._config.metrics,
            ServiceType.TELEMETRY: self._config.telemetry,
        }

        return config_map.get(service_type)

    async def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._health_monitor_task:
            return

        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._logger.info("Health monitoring started")

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for monitoring interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60.0,  # Default monitoring interval
                )
                break  # Shutdown requested

            except TimeoutError:
                # Time for health check
                try:
                    health_report = await self.get_health_report()

                    # Log health status
                    if health_report.overall_status == HealthStatus.UNHEALTHY:
                        self._logger.warning("System health: UNHEALTHY")
                    elif health_report.overall_status == HealthStatus.DEGRADED:
                        self._logger.warning("System health: DEGRADED")
                    else:
                        self._logger.debug("System health: HEALTHY")

                    # Handle auto-recovery if enabled
                    if self._config.auto_recovery:
                        await self._handle_auto_recovery(health_report)

                except Exception as e:
                    self._logger.warning("Health monitoring failed: %s", e)

    async def _handle_auto_recovery(self, health_report: ServicesHealthReport) -> None:
        """Handle automatic recovery for unhealthy services."""
        for service_type, health in health_report.services.items():
            if health.status == HealthStatus.UNHEALTHY:
                try:
                    self._logger.info(
                        "Attempting auto-recovery for service: %s", service_type.value
                    )
                    await self.restart_service(service_type)
                    self._logger.info(
                        "Auto-recovery successful for service: %s", service_type.value
                    )

                except Exception:
                    self._logger.exception("Auto-recovery failed for service %s")
