# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service registry for managing service providers."""

import logging

from datetime import datetime
from typing import Any

from codeweaver._types.config import ServiceType
from codeweaver._types.service_config import ServiceConfig
from codeweaver._types.service_data import (
    ProviderStatus,
    ServiceCapabilities,
    ServiceInstanceInfo,
    ServiceProviderInfo,
    ServiceRegistryHealth,
)
from codeweaver._types.service_exceptions import (
    DuplicateProviderError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    ServiceCreationError,
    ServiceNotFoundError,
)
from codeweaver._types.services import ServiceProvider
from codeweaver.services.providers.base_provider import BaseServiceProvider


class ServiceRegistry:
    """Registry for managing service providers and instances."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the service registry."""
        self._logger = logger or logging.getLogger("codeweaver.factories.service_registry")

        # Registry storage
        self._provider_classes: dict[ServiceType, dict[str, type[ServiceProvider]]] = {}
        self._provider_info: dict[ServiceType, dict[str, ServiceProviderInfo]] = {}
        self._service_instances: dict[ServiceType, ServiceProvider | None] = {}
        self._instance_info: dict[ServiceType, ServiceInstanceInfo | None] = {}

        # Registry state
        self._initialized = False
        self._created_at = datetime.now(datetime.timezone.utc)

    def register_provider(
        self,
        service_type: ServiceType,
        provider_name: str,
        provider_class: type[ServiceProvider],
        capabilities: ServiceCapabilities | None = None,
        configuration_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a service provider class."""

        def _raise_error(error_type: str, message: str, service_type: str) -> None:
            """Helper function to raise formatted errors."""
            match error_type:
                case "DuplicateProviderError":
                    raise DuplicateProviderError(service_type, provider_name)
                case "ProviderNotFoundError":
                    raise ProviderNotFoundError(service_type, provider_name)

        try:
            # Initialize service type registry if needed
            if service_type not in self._provider_classes:
                self._provider_classes[service_type] = {}
                self._provider_info[service_type] = {}

            # Check for duplicates
            if provider_name in self._provider_classes[service_type]:
                _raise_error(
                    "DuplicateProviderError",
                    f"Provider {provider_name} already registered for service {service_type.value}",
                    service_type.value,
                )

            # Validate provider class
            if not issubclass(provider_class, BaseServiceProvider):
                _raise_error(
                    "ProviderNotFoundError",
                    f"Provider class {provider_class.__name__} is not a valid ServiceProvider subclass",
                    service_type.value,
                )

            # Register provider class
            self._provider_classes[service_type][provider_name] = provider_class

            # Create provider info
            provider_info = ServiceProviderInfo(
                name=provider_name,
                version=getattr(provider_class, "VERSION", "1.0.0"),
                capabilities=capabilities or ServiceCapabilities(),
                configuration_schema=configuration_schema or {},
                status=ProviderStatus.REGISTERED,
                created_at=datetime.now(datetime.timezone.utc),
                last_modified=datetime.now(datetime.timezone.utc),
            )

            self._provider_info[service_type][provider_name] = provider_info

            self._logger.info(
                "Registered provider %s for service %s", provider_name, service_type.value
            )

        except Exception as e:
            error_msg = f"Failed to register provider {provider_name} for {service_type.value}"
            self._logger.exception(error_msg)
            raise ProviderRegistrationError(service_type, provider_name, str(e)) from e

    def unregister_provider(self, service_type: ServiceType, provider_name: str) -> None:
        """Unregister a service provider."""
        if service_type not in self._provider_classes:
            raise ServiceNotFoundError(service_type)

        if provider_name not in self._provider_classes[service_type]:
            raise ProviderNotFoundError(service_type, provider_name)

        # Remove active instance if using this provider
        if (
            service_type in self._service_instances
            and self._service_instances[service_type] is not None
            and self._service_instances[service_type].name == provider_name
        ):
            self._service_instances[service_type] = None
            self._instance_info[service_type] = None

        # Remove from registry
        del self._provider_classes[service_type][provider_name]
        del self._provider_info[service_type][provider_name]

        self._logger.info(
            "Unregistered provider %s for service %s", provider_name, service_type.value
        )

    async def create_service(
        self, service_type: ServiceType, config: ServiceConfig
    ) -> ServiceProvider:
        """Create a service instance using the configured provider."""
        try:
            # Find provider class
            provider_class = self._get_provider_class(service_type, config.provider)

            # Create service instance
            service_instance = provider_class(service_type, config)

            # Initialize the service
            await service_instance.initialize()

            # Store instance
            self._service_instances[service_type] = service_instance

            # Create instance info
            instance_info = ServiceInstanceInfo(
                service_type=service_type,
                provider_name=config.provider,
                instance_id=f"{service_type.value}_{config.provider}_{id(service_instance)}",
                status="ready",
                created_at=datetime.now(datetime.timezone.utc),
                last_health_check=None,
                config_hash=str(hash(str(config))),
            )

            self._instance_info[service_type] = instance_info

            self._logger.info(
                "Created service %s using provider %s", service_type.value, config.provider
            )

        except Exception as e:
            error_msg = (
                f"Failed to create service {service_type.value} with provider {config.provider}"
            )
            self._logger.exception(error_msg)
            raise ServiceCreationError(service_type, str(e)) from e
        else:
            return service_instance

    async def get_service(self, service_type: ServiceType) -> ServiceProvider | None:
        """Get an existing service instance."""
        return self._service_instances.get(service_type)

    async def destroy_service(self, service_type: ServiceType) -> None:
        """Destroy a service instance."""
        if service := self._service_instances.get(service_type):
            try:
                await service.shutdown()
            except Exception as e:
                self._logger.warning("Error shutting down service %s: %s", service_type.value, e)

            self._service_instances[service_type] = None
            self._instance_info[service_type] = None

            self._logger.info("Destroyed service %s", service_type.value)

    def list_providers(self, service_type: ServiceType = None) -> dict[ServiceType, list[str]]:
        """List all registered providers."""
        if service_type:
            return {service_type: list(self._provider_classes.get(service_type, {}).keys())}

        return {
            stype: list(providers.keys()) for stype, providers in self._provider_classes.items()
        }

    def get_provider_info(
        self, service_type: ServiceType, provider_name: str
    ) -> ServiceProviderInfo:
        """Get information about a specific provider."""
        if service_type not in self._provider_info:
            raise ServiceNotFoundError(service_type)

        if provider_name not in self._provider_info[service_type]:
            raise ProviderNotFoundError(service_type, provider_name)

        return self._provider_info[service_type][provider_name]

    def get_instance_info(self, service_type: ServiceType) -> ServiceInstanceInfo | None:
        """Get information about a service instance."""
        return self._instance_info.get(service_type)

    def list_active_services(self) -> dict[ServiceType, ServiceInstanceInfo]:
        """List all active service instances."""
        return {stype: info for stype, info in self._instance_info.items() if info is not None}

    async def health_check(self) -> ServiceRegistryHealth:
        """Check health of the service registry."""
        total_services = len(self._service_instances)
        healthy_services = 0
        issues = []

        # Check each service
        for service_type, service in self._service_instances.items():
            if service is None:
                continue

            try:
                health = await service.health_check()
                if health.status.value in ["healthy", "degraded"]:
                    healthy_services += 1
                else:
                    issues.append(f"Service {service_type.value} is unhealthy: {health.last_error}")

                # Update instance info
                if self._instance_info.get(service_type):
                    self._instance_info[service_type].last_health_check = datetime.now(
                        datetime.timezone.utc
                    )

            except Exception as e:
                issues.append(f"Health check failed for {service_type.value}: {e}")

        total_providers = sum(len(providers) for providers in self._provider_classes.values())

        return ServiceRegistryHealth(
            total_services=total_services,
            healthy_services=healthy_services,
            total_providers=total_providers,
            last_check=datetime.now(datetime.timezone.utc),
            issues=issues,
        )

    async def shutdown_all(self) -> None:
        """Shutdown all active services."""
        for service_type in list(self._service_instances.keys()):
            await self.destroy_service(service_type)

        self._logger.info("All services shut down")

    def _get_provider_class(
        self, service_type: ServiceType, provider_name: str
    ) -> type[ServiceProvider]:
        """Get provider class for service creation."""
        if service_type not in self._provider_classes:
            raise ServiceNotFoundError(service_type)

        if provider_name not in self._provider_classes[service_type]:
            raise ProviderNotFoundError(service_type, provider_name)

        return self._provider_classes[service_type][provider_name]
