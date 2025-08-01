# sourcery skip: do-not-use-staticmethod
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Component initialization patterns and lifecycle management.

Implements comprehensive initialization pipeline with stages, validation,
and component lifecycle tracking as specified in the architecture.
"""

import asyncio
import logging
import time

from abc import abstractmethod
from typing import Any, Protocol

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeweaver.factories.error_handling import ErrorHandler
from codeweaver.types import (
    BaseComponentConfig,
    ComponentLifecycle,
    ComponentState,
    InitializationContext,
    InitializationResult,
)


logger = logging.getLogger(__name__)


@dataclass
class FactoryInitializationResult:
    """Result of factory initialization."""

    success: bool
    total_duration_ms: float
    stage_results: list[InitializationResult]
    factory_state: ComponentState
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InitializationStage(Protocol):
    """Protocol for initialization stages."""

    @abstractmethod
    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Execute this initialization stage."""
        ...

    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        ...


class ConfigurationValidationStage:
    """Validate factory configuration."""

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "configuration_validation"

    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        return []

    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Validate configuration completeness and consistency."""
        errors = []
        warnings = []

        try:
            # Basic configuration validation would go here
            # For now, just check if config exists
            if not context.config:
                errors.append("Missing configuration")

            # Additional validation can be added based on specific config type

        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")

        return InitializationResult(
            success=not errors,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
        )


class RegistryInitializationStage:
    """Initialize component registries."""

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "registry_initialization"

    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        return ["configuration_validation"]

    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Initialize and populate component registries."""
        errors = []
        warnings = []

        try:
            # Initialize built-in components would go here
            # For now, just validate registries exist
            errors.extend(
                f"Registry '{name}' is not initialized"
                for name, registry in context.registries.items()
                if registry is None
            )
        except Exception as e:
            errors.append(f"Registry initialization failed: {e}")

        # Validate registry consistency
        registry_stats = {}
        for name, registry in context.registries.items():
            if hasattr(registry, "list_available_components"):
                available_components = registry.list_available_components()
                registry_stats[name] = len(available_components)

                if len(available_components) == 0:
                    warnings.append(f"No components available in {name} registry")

        return InitializationResult(
            success=not errors,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
            metadata={"registry_stats": registry_stats},
        )


class PluginDiscoveryStage:
    """Discover and register plugins."""

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "plugin_discovery"

    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        return ["registry_initialization"]

    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Discover and register available plugins."""
        errors = []
        warnings = []

        if not context.plugin_manager:
            return InitializationResult(
                success=True,
                stage_name=self.get_stage_name(),
                duration_ms=0,
                warnings=["Plugin system disabled"],
            )

        try:
            # Plugin discovery logic would go here
            # For now, just validate plugin manager exists
            if hasattr(context.plugin_manager, "discover_plugins"):
                # Simulate plugin discovery
                plugin_stats = {"discovered": 0, "registered": 0, "failed": 0}
            else:
                errors.append("Plugin manager doesn't support discovery")
                plugin_stats = {"error": "Invalid plugin manager"}

        except Exception as e:
            errors.append(f"Plugin discovery failed: {e}")
            plugin_stats = {"error": str(e)}

        return InitializationResult(
            success=not errors,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
            metadata={"plugin_stats": plugin_stats},
        )


class ComponentValidationStage:
    """Validate component availability and configuration."""

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "component_validation"

    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        return ["plugin_discovery"]

    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Validate all components are properly configured."""
        errors = []
        warnings = []

        try:
            # Component validation logic would go here
            validation_count = 0
            for registry in context.registries.values():
                if hasattr(registry, "list_available_components"):
                    components = registry.list_available_components()
                    validation_count += len(components)

                    for component_name in components:
                        if hasattr(registry, "validate_component"):
                            result = registry.validate_component(component_name)
                            if hasattr(result, "is_valid") and not result.is_valid:
                                warnings.append(f"Component {component_name} validation failed")

            metadata = {"components_validated": validation_count}

        except Exception as e:
            errors.append(f"Component validation failed: {e}")
            metadata = {"error": str(e)}

        return InitializationResult(
            success=not errors,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


class FactoryInitializer:
    """Factory initialization orchestrator."""

    def __init__(self):
        """Initialize the factory initializer with default stages."""
        self._stages: list[InitializationStage] = [
            ConfigurationValidationStage(),
            RegistryInitializationStage(),
            PluginDiscoveryStage(),
            ComponentValidationStage(),
        ]
        self._error_handler = ErrorHandler()

    async def initialize_factory(self, factory: Any, config: Any) -> FactoryInitializationResult:
        """Execute complete factory initialization."""
        context = InitializationContext(
            factory=factory,
            config=config,
            registries=getattr(factory, "_registries", {}),
            plugin_manager=getattr(factory, "_plugin_manager", None),
            dependency_resolver=getattr(factory, "_dependency_resolver", None),
        )

        stage_results = []
        overall_start = time.time()

        for stage in self._stages:
            stage_start = time.time()

            try:
                # Check dependencies
                await self._validate_stage_dependencies(stage, stage_results)

                # Execute stage
                result = await stage.execute(context)
                result.duration_ms = (time.time() - stage_start) * 1000

                stage_results.append(result)
                context.stage_results[result.stage_name] = result

                if not result.success:
                    logger.error(
                        "Initialization stage '%s' failed: %s", result.stage_name, result.errors
                    )
                    break

                logger.info(
                    "Initialization stage '%s' completed in %.2fms",
                    result.stage_name,
                    result.duration_ms,
                )

            except Exception as e:
                logger.exception("Initialization stage '%s' crashed")

                stage_results.append(
                    InitializationResult(
                        success=False,
                        stage_name=stage.get_stage_name(),
                        duration_ms=(time.time() - stage_start) * 1000,
                        errors=[f"Stage crashed: {e}"],
                    )
                )
                break

        total_duration = (time.time() - overall_start) * 1000
        overall_success = all(result.success for result in stage_results)

        return FactoryInitializationResult(
            success=overall_success,
            total_duration_ms=total_duration,
            stage_results=stage_results,
            factory_state=ComponentState.INITIALIZED if overall_success else ComponentState.ERROR,
        )

    async def _validate_stage_dependencies(
        self, stage: InitializationStage, completed_stages: list[InitializationResult]
    ) -> None:
        """Validate that stage dependencies have been completed."""
        completed_stage_names = {result.stage_name for result in completed_stages if result.success}

        for dependency in stage.get_dependencies():
            if dependency not in completed_stage_names:
                raise ValueError(
                    f"Stage '{stage.get_stage_name()}' depends on '{dependency}' which hasn't completed successfully"
                )


class ComponentInitializer:
    """Universal component initialization pattern."""

    @staticmethod
    async def initialize_component(
        component: Any, config: BaseComponentConfig, lifecycle: ComponentLifecycle
    ) -> InitializationResult:
        """Initialize a component following standard pattern."""
        errors = []
        warnings = []

        try:
            lifecycle.transition_to(ComponentState.INITIALIZING)

            # Pre-initialization validation
            if hasattr(component, "validate_config") and not component.validate_config(config):
                errors.append("Component configuration validation failed")

            # Component initialization
            if hasattr(component, "initialize"):
                if asyncio.iscoroutinefunction(component.initialize):
                    await component.initialize()
                else:
                    component.initialize()
            elif hasattr(component, "__aenter__"):
                await component.__aenter__()

            # Post-initialization validation
            if hasattr(component, "health_check"):
                if asyncio.iscoroutinefunction(component.health_check):
                    health_ok = await component.health_check()
                else:
                    health_ok = component.health_check()
                if not health_ok:
                    warnings.append("Component health check failed")

            if not errors:
                lifecycle.transition_to(ComponentState.INITIALIZED)
            else:
                lifecycle.transition_to(ComponentState.ERROR)

        except Exception as e:
            errors.append(f"Component initialization failed: {e}")
            lifecycle.transition_to(ComponentState.ERROR, e)

        return InitializationResult(
            success=not errors,
            stage_name=f"component_init_{config.provider}",
            duration_ms=0,
            errors=errors,
            warnings=warnings,
        )
