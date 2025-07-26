# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Services manager for coordinating all service providers."""

import asyncio
import contextlib
import logging

from datetime import datetime
from typing import Any

from codeweaver._types.config import ServiceType
from codeweaver._types.service_config import ServicesConfig
from codeweaver._types.service_data import HealthStatus, ServicesHealthReport
from codeweaver._types.service_exceptions import (
    ServiceCreationError,
    ServiceInitializationError,
    ServiceNotFoundError,
)
from codeweaver._types.services import (
    CacheService,
    MetricsService,
    MonitoringService,
    ServiceProvider,
    ValidationService,
)
from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.services.providers.chunking import ChunkingService
from codeweaver.services.providers.file_filtering import FilteringService


class ServicesManager:
    """Manager for coordinating all service providers."""

    def __init__(self, config: ServicesConfig, logger: logging.Logger | None = None):
        """Initialize the services manager with configuration and logger."""
        self._config = config
        self._logger = logger or logging.getLogger("codeweaver.services.manager")

        # Service registry
        self._registry = ServiceRegistry(self._logger)

        # Service instances
        self._services: dict[ServiceType, ServiceProvider] = {}

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

    def get_service(self, service_type: ServiceType) -> ServiceProvider | None:
        """Get any service by type."""
        return self._services.get(service_type)

    def list_active_services(self) -> dict[ServiceType, ServiceProvider]:
        """List all active services."""
        return self._services.copy()

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
                from codeweaver._types.service_data import ServiceHealth

                services_health[service_type] = ServiceHealth(
                    service_type=service_type,
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.now(datetime.timezone.utc),
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
            overall_status=overall_status,
            services=services_health,
            check_time=datetime.now(datetime.timezone.utc),
            metrics=metrics,
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
        """Register all built-in service providers."""
        # Register chunking providers
        self._registry.register_provider(ServiceType.CHUNKING, "fastmcp_chunking", ChunkingService)

        # Register filtering providers
        self._registry.register_provider(
            ServiceType.FILTERING, "fastmcp_filtering", FilteringService
        )

        # TODO: Register other built-in providers when available

        self._logger.info("Built-in service providers registered")

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
                    # TODO: Implement optional service providers
                    self._logger.info(
                        "Optional service %s enabled but not implemented yet", service_type.value
                    )

                except Exception as e:
                    # Optional services failures are not fatal
                    self._logger.warning(
                        "Failed to create optional service %s: %s", service_type.value, e
                    )

    def _get_service_config(self, service_type: ServiceType) -> Any:
        """Get configuration for a specific service type."""
        config_map = {
            ServiceType.CHUNKING: self._config.chunking,
            ServiceType.FILTERING: self._config.filtering,
            ServiceType.VALIDATION: self._config.validation,
            ServiceType.CACHE: self._config.cache,
            ServiceType.MONITORING: self._config.monitoring,
            ServiceType.METRICS: self._config.metrics,
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
                    self._logger.exception(
                        "Auto-recovery failed for service %s", service_type.value
                    )
