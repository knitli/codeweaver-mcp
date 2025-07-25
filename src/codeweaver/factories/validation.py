# sourcery skip: no-complex-if-expressions
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Validation utilities for the factory system.

Provides comprehensive validation for factory configurations, component compatibility,
and system health checks.
"""

import asyncio
import logging

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import auto
from typing import Any

from codeweaver._types.base_enum import BaseEnum
from codeweaver.config import BackendConfig, CodeWeaverConfig
from codeweaver.factories.plugin_discovery import PluginDiscovery
from codeweaver.factories.unified_factory import UnifiedFactory


logger = logging.getLogger(__name__)


class ValidationLevel(BaseEnum):
    """Validation levels for different scenarios."""

    MINIMAL = auto()  # Basic validation only
    STANDARD = auto()  # Standard validation (default)
    COMPREHENSIVE = auto()  # Full validation including performance tests


class CompatibilityLevel(BaseEnum):
    """Compatibility levels between components."""

    INCOMPATIBLE = auto()  # Components cannot work together
    PARTIAL = auto()  # Limited functionality
    COMPATIBLE = auto()  # Full compatibility
    OPTIMAL = auto()  # Optimal pairing


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    component: str
    check_name: str
    message: str
    severity: str = "info"  # info, warning, error, critical
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompatibilityResult:
    """Result of a compatibility check between components."""

    level: CompatibilityLevel
    component_a: str
    component_b: str
    message: str
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    overall_health: str  # healthy, degraded, critical
    validation_results: list[ValidationResult] = field(default_factory=list)
    compatibility_results: list[CompatibilityResult] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=str)


class FactoryValidator:
    """
    Comprehensive validation system for factories and components.

    Validates configurations, checks component compatibility, and provides
    system health assessments.
    """

    def __init__(
        self,
        factory: UnifiedFactory | None = None,
        level: ValidationLevel = ValidationLevel.STANDARD,
    ):
        """Initialize the factory validator.

        Args:
            factory: Unified factory instance to validate
            level: Validation level to use
        """
        self.factory = factory
        self.level = level
        self._validation_checks: list[Callable] = []
        self._compatibility_checks: list[Callable] = []

        # Register default validation checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default validation checks."""
        # Configuration validation
        self._validation_checks.extend([
            self._validate_backend_config,
            self._validate_embedding_config,
            self._validate_rate_limiting,
            self._validate_paths,
        ])

        # Component validation
        if self.level >= ValidationLevel.STANDARD:
            self._validation_checks.extend([
                self._validate_backend_connectivity,
                self._validate_CW_EMBEDDING_PROVIDER,
                self._validate_plugin_system,
            ])

        # Performance validation
        if self.level == ValidationLevel.COMPREHENSIVE:
            self._validation_checks.extend([
                self._validate_performance_thresholds,
                self._validate_resource_usage,
            ])

        # Compatibility checks
        self._compatibility_checks.extend([
            self._check_backend_provider_compatibility,
            self._check_embedding_dimension_compatibility,
            self._check_api_key_requirements,
        ])

    async def validate_configuration(self, config: CodeWeaverConfig) -> list[ValidationResult]:
        """Validate a CodeWeaver configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation results
        """
        results = []

        # Run all validation checks
        for check in self._validation_checks:
            if asyncio.iscoroutinefunction(check):
                result = await check(config)
            else:
                result = check(config)

            if result:
                results.append(result)

        return results

    def _validate_backend_config(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate backend configuration."""
        if not hasattr(config, "backend") or not config.backend:
            return ValidationResult(
                passed=False,
                component="backend",
                check_name="backend_config",
                message="Missing backend configuration",
                severity="error",
            )

        try:
            # Validate backend config structure
            if isinstance(config.backend, dict):
                BackendConfig(**config.backend)
            elif not isinstance(config.backend, BackendConfig):
                return ValidationResult(
                    passed=False,
                    component="backend",
                    check_name="backend_config_type",
                    message="Invalid backend configuration type",
                    severity="error",
                )

            # Check if provider is supported
            if self.factory:
                supported = self.factory.backends.list_supported_providers()
                if config.backend.provider not in supported:
                    return ValidationResult(
                        passed=False,
                        component="backend",
                        check_name="backend_provider",
                        message=f"Unsupported backend provider: {config.backend.provider}",
                        severity="error",
                        details={"supported_providers": list(supported.keys())},
                    )

            return ValidationResult(
                passed=True,
                component="backend",
                check_name="backend_config",
                message="Backend configuration valid",
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                component="backend",
                check_name="backend_config",
                message=f"Backend configuration error: {e}",
                severity="error",
            )

    def _validate_embedding_config(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate embedding configuration."""
        if not hasattr(config, "embedding") or not config.embedding:
            return ValidationResult(
                passed=False,
                component="embedding",
                check_name="embedding_config",
                message="Missing embedding configuration",
                severity="error",
            )

        # Check API key if required
        if not config.embedding.api_key and config.embedding.provider != "sentence-transformers":
            return ValidationResult(
                passed=False,
                component="embedding",
                check_name="api_key",
                message=f"API key required for {config.embedding.provider}",
                severity="error",
            )

        return ValidationResult(
            passed=True,
            component="embedding",
            check_name="embedding_config",
            message="Embedding configuration valid",
        )

    def _validate_rate_limiting(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate rate limiting configuration."""
        if not hasattr(config, "rate_limiting"):
            return ValidationResult(
                passed=True,
                component="rate_limiting",
                check_name="rate_limiting_config",
                message="Using default rate limiting configuration",
                severity="info",
            )

        # Validate rate limit values
        if config.rate_limiting.requests_per_minute <= 0:
            return ValidationResult(
                passed=False,
                component="rate_limiting",
                check_name="requests_per_minute",
                message="Invalid requests_per_minute value",
                severity="warning",
                details={"value": config.rate_limiting.requests_per_minute},
            )

        return ValidationResult(
            passed=True,
            component="rate_limiting",
            check_name="rate_limiting_config",
            message="Rate limiting configuration valid",
        )

    def _validate_paths(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate file paths and directories."""
        # This is a placeholder - extend based on actual path requirements
        return None

    async def _validate_backend_connectivity(
        self, config: CodeWeaverConfig
    ) -> ValidationResult | None:
        """Validate backend connectivity."""
        if not self.factory:
            return None

        try:
            # Create backend instance
            backend = self.factory.backends.create_backend(config.backend)

            # Try to get collection info (basic connectivity test)
            if hasattr(backend, "get_collection_info"):
                await backend.get_collection_info("test_collection")

            return ValidationResult(
                passed=True,
                component="backend",
                check_name="connectivity",
                message="Backend connectivity verified",
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                component="backend",
                check_name="connectivity",
                message=f"Backend connectivity failed: {e}",
                severity="error",
            )

    async def _validate_CW_EMBEDDING_PROVIDER(
        self, config: CodeWeaverConfig
    ) -> ValidationResult | None:
        """Validate embedding provider functionality."""
        if not self.factory:
            return None

        try:
            # Create provider instance
            provider = self.factory.providers.create_CW_EMBEDDING_PROVIDER(config.embedding)

            # Test basic functionality
            test_text = "This is a validation test"
            embedding = await provider.embed_query(test_text)

            if not embedding or len(embedding) != provider.dimension:
                return ValidationResult(
                    passed=False,
                    component="embedding",
                    check_name="functionality",
                    message="Embedding dimension mismatch",
                    severity="error",
                    details={
                        "expected": provider.dimension,
                        "actual": len(embedding) if embedding else 0,
                    },
                )

            return ValidationResult(
                passed=True,
                component="embedding",
                check_name="functionality",
                message="Embedding provider functional",
                details={"dimension": provider.dimension},
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                component="embedding",
                check_name="functionality",
                message=f"Embedding provider error: {e}",
                severity="error",
            )

    async def _validate_plugin_system(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate plugin discovery system."""
        try:
            discovery = PluginDiscovery(auto_load=False)
            await discovery.discover_plugins()

            plugin_info = discovery.get_plugin_info()

            return ValidationResult(
                passed=True,
                component="plugins",
                check_name="discovery",
                message="Plugin system operational",
                details={
                    "directories_scanned": len(plugin_info["plugin_directories"]),
                    "backends_found": len(plugin_info["backend_plugins"]),
                    "providers_found": len(plugin_info["provider_plugins"]),
                    "sources_found": len(plugin_info["source_plugins"]),
                },
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                component="plugins",
                check_name="discovery",
                message=f"Plugin system error: {e}",
                severity="warning",
            )

    async def _validate_performance_thresholds(
        self, config: CodeWeaverConfig
    ) -> ValidationResult | None:
        """Validate performance thresholds."""
        # This would include actual performance tests
        return None

    async def _validate_resource_usage(self, config: CodeWeaverConfig) -> ValidationResult | None:
        """Validate resource usage."""
        # This would check memory, CPU, etc.
        return None

    async def check_compatibility(self, config: CodeWeaverConfig) -> list[CompatibilityResult]:
        """Check compatibility between configured components.

        Args:
            config: Configuration to check

        Returns:
            List of compatibility results
        """
        results = []

        for check in self._compatibility_checks:
            result = await check(config) if asyncio.iscoroutinefunction(check) else check(config)
            if result:
                results.append(result)

        return results

    def _check_backend_provider_compatibility(
        self, config: CodeWeaverConfig
    ) -> CompatibilityResult | None:
        """Check compatibility between backend and embedding provider."""
        if not config.backend or not config.embedding:
            return None

        # Known optimal pairings
        optimal_pairs = {
            ("qdrant", "voyage-ai"),
            ("pinecone", "openai"),
            ("weaviate", "cohere"),
            ("chroma", "sentence-transformers"),
        }

        backend = config.backend.provider
        provider = config.embedding.provider

        if (backend, provider) in optimal_pairs:
            return CompatibilityResult(
                level=CompatibilityLevel.OPTIMAL,
                component_a=f"backend:{backend}",
                component_b=f"embedding:{provider}",
                message="Optimal pairing for performance",
            )

        # All combinations should work
        return CompatibilityResult(
            level=CompatibilityLevel.COMPATIBLE,
            component_a=f"backend:{backend}",
            component_b=f"embedding:{provider}",
            message="Components are compatible",
        )

    def _check_embedding_dimension_compatibility(
        self, config: CodeWeaverConfig
    ) -> CompatibilityResult | None:
        """Check embedding dimension compatibility."""
        if not config.embedding:
            return None

        if dimension := config.embedding.dimension:
            # Check dimension limits for backends
            return (
                CompatibilityResult(
                    level=CompatibilityLevel.PARTIAL,
                    component_a="embedding",
                    component_b="backend",
                    message=f"Large embedding dimension ({dimension}) may impact performance",
                    recommendations=[
                        "Consider using dimensionality reduction",
                        "Ensure backend supports large dimensions",
                    ],
                )
                if dimension > 4096
                else None
            )
        return CompatibilityResult(
            level=CompatibilityLevel.COMPATIBLE,
            component_a="embedding",
            component_b="backend",
            message="Using default embedding dimension",
            recommendations=["Consider specifying dimension explicitly"],
        )

    def _check_api_key_requirements(self, config: CodeWeaverConfig) -> CompatibilityResult | None:
        """Check API key requirements."""
        missing_keys = []

        if (
            config.embedding
            and config.embedding.provider != "sentence-transformers"
            and not config.embedding.api_key
        ):
            missing_keys.append("embedding")

        if (
            hasattr(config.backend, "api_key")
            and not config.backend.api_key
            and config.backend.provider in ["pinecone", "weaviate"]
        ):
            missing_keys.append("backend")

        if missing_keys:
            return CompatibilityResult(
                level=CompatibilityLevel.INCOMPATIBLE,
                component_a="configuration",
                component_b="providers",
                message=f"Missing required API keys: {', '.join(missing_keys)}",
                recommendations=["Add required API keys to configuration"],
            )

        return None

    async def generate_health_report(self, config: CodeWeaverConfig) -> SystemHealthReport:
        """Generate a comprehensive system health report.

        Args:
            config: Configuration to assess

        Returns:
            Complete system health report
        """
        # Run validation checks
        validation_results = await self.validate_configuration(config)

        # Run compatibility checks
        compatibility_results = await self.check_compatibility(config)

        # Determine overall health
        error_count = sum(r.severity == "error" for r in validation_results)
        warning_count = sum(r.severity == "warning" for r in validation_results)

        if error_count > 0:
            overall_health = "critical"
        elif warning_count > 0:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        # Generate recommendations
        recommendations = []
        if error_count > 0:
            recommendations.append(f"Fix {error_count} critical errors before deployment")
        if warning_count > 0:
            recommendations.append(f"Address {warning_count} warnings for optimal performance")

        # Add specific recommendations from compatibility checks
        for compat in compatibility_results:
            recommendations.extend(compat.recommendations)

        import datetime

        return SystemHealthReport(
            overall_health=overall_health,
            validation_results=validation_results,
            compatibility_results=compatibility_results,
            recommendations=recommendations,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
