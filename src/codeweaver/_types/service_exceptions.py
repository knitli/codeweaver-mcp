# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service-specific exceptions for CodeWeaver."""

from pathlib import Path

from codeweaver._types.config import ServiceType
from codeweaver._types.exceptions import CodeWeaverError


class ServiceError(CodeWeaverError):
    """Base exception for service-related errors."""


class ServiceNotFoundError(ServiceError):
    """Exception raised when a service is not found."""

    def __init__(self, service_type: ServiceType):
        super().__init__(f"Service not found: {service_type.value}")
        self.service_type = service_type


class ServiceCreationError(ServiceError):
    """Exception raised when service creation fails."""

    def __init__(self, service_type: ServiceType, reason: str):
        super().__init__(f"Failed to create service {service_type.value}: {reason}")
        self.service_type = service_type
        self.reason = reason


class ServiceConfigurationError(ServiceError):
    """Exception raised for service configuration errors."""

    def __init__(self, service_type: ServiceType, config_error: str):
        super().__init__(f"Configuration error for {service_type.value}: {config_error}")
        self.service_type = service_type
        self.config_error = config_error


class ServiceInitializationError(ServiceError):
    """Exception raised when service initialization fails."""

    def __init__(self, message: str):
        super().__init__(f"Service initialization failed: {message}")


class ServiceStartError(ServiceError):
    """Exception raised when service startup fails."""

    def __init__(self, message: str):
        super().__init__(f"Service startup failed: {message}")


class ServiceStopError(ServiceError):
    """Exception raised when service shutdown fails."""

    def __init__(self, message: str):
        super().__init__(f"Service shutdown failed: {message}")


class ServiceRestartError(ServiceError):
    """Exception raised when service restart fails."""

    def __init__(self, service_type: ServiceType, reason: str):
        super().__init__(f"Failed to restart service {service_type.value}: {reason}")
        self.service_type = service_type
        self.reason = reason


class ServiceNotReadyError(ServiceError):
    """Exception raised when service is not ready for operations."""

    def __init__(self, service_type: ServiceType, current_state: str):
        super().__init__(f"Service {service_type.value} not ready (current state: {current_state})")
        self.service_type = service_type
        self.current_state = current_state


class ProviderRegistrationError(ServiceError):
    """Exception raised when provider registration fails."""

    def __init__(self, service_type: ServiceType, provider_name: str, reason: str):
        super().__init__(
            f"Failed to register provider {provider_name} for {service_type.value}: {reason}"
        )
        self.service_type = service_type
        self.provider_name = provider_name
        self.reason = reason


class DuplicateProviderError(ServiceError):
    """Exception raised when attempting to register a duplicate provider."""

    def __init__(self, service_type: ServiceType, provider_name: str):
        super().__init__(f"Provider {provider_name} already registered for {service_type.value}")
        self.service_type = service_type
        self.provider_name = provider_name


class ProviderNotFoundError(ServiceError):
    """Exception raised when a provider is not found."""

    def __init__(self, service_type: ServiceType, provider_name: str):
        super().__init__(f"Provider {provider_name} not found for {service_type.value}")
        self.service_type = service_type
        self.provider_name = provider_name


class DependencyInjectionError(ServiceError):
    """Exception raised when dependency injection fails."""

    def __init__(self, message: str):
        super().__init__(f"Dependency injection failed: {message}")


class DependencyResolutionError(ServiceError):
    """Exception raised when dependency resolution fails."""

    def __init__(self, message: str):
        super().__init__(f"Dependency resolution failed: {message}")


class ChunkingError(ServiceError):
    """Exception raised for chunking-related errors."""

    def __init__(self, file_path: Path, reason: str):
        super().__init__(f"Chunking failed for {file_path}: {reason}")
        self.file_path = file_path
        self.reason = reason


class UnsupportedLanguageError(ChunkingError):
    """Exception raised for unsupported programming languages."""

    def __init__(self, file_path: Path, language: str):
        super().__init__(file_path, f"Unsupported language: {language}")
        self.language = language


class FilteringError(ServiceError):
    """Exception raised for filtering-related errors."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"Filtering failed for {path}: {reason}")
        self.path = path
        self.reason = reason


class AccessDeniedError(FilteringError):
    """Exception raised when access to a path is denied."""

    def __init__(self, path: Path, reason: str):
        super().__init__(path, f"Access denied: {reason}")


class DirectoryNotFoundError(FilteringError):
    """Exception raised when a directory is not found."""

    def __init__(self, path: Path):
        super().__init__(path, "Directory not found")


class ValidationError(ServiceError):
    """Exception raised for validation errors."""

    def __init__(self, item_id: str, rule_id: str, message: str):
        super().__init__(f"Validation failed for {item_id} (rule {rule_id}): {message}")
        self.item_id = item_id
        self.rule_id = rule_id


class CacheError(ServiceError):
    """Exception raised for cache-related errors."""

    def __init__(self, operation: str, key: str, reason: str):
        super().__init__(f"Cache {operation} failed for key '{key}': {reason}")
        self.operation = operation
        self.key = key
        self.reason = reason


class ReconfigurationError(ServiceError):
    """Exception raised when service reconfiguration fails."""

    def __init__(self, service_type: ServiceType, reason: str):
        super().__init__(f"Failed to reconfigure service {service_type.value}: {reason}")
        self.service_type = service_type
        self.reason = reason
