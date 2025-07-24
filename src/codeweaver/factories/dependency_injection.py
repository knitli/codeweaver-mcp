# sourcery skip: avoid-single-character-names-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Dependency injection container for CodeWeaver extensibility system.

Manages component lifecycles, dependency resolution, and circular dependency detection
with support for singleton, transient, and scoped instances.
"""

import asyncio
import logging

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar
from weakref import WeakValueDictionary


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifecycle(Enum):
    """Component lifecycle types."""

    TRANSIENT = auto()  # New instance on every request
    SINGLETON = auto()  # Single instance for entire application
    SCOPED = auto()  # Single instance per scope (e.g., request)


@dataclass
class DependencyRegistration:
    """Registration details for a dependency."""

    component_type: str
    component_name: str
    factory: Callable[..., Any]
    lifecycle: Lifecycle = Lifecycle.TRANSIENT
    dependencies: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionContext:
    """Context for dependency resolution to detect circular dependencies."""

    resolving: set[tuple[str, str]] = field(default_factory=set)
    resolved: dict[tuple[str, str], Any] = field(default_factory=dict)
    scope_id: str | None = None


class DependencyContainer:
    """
    Dependency injection container for managing component lifecycles.

    Supports automatic dependency resolution, circular dependency detection,
    and flexible lifecycle management for extensible components.
    """

    def __init__(self, *, singleton_backends: bool = True, singleton_providers: bool = True):
        """Initialize the dependency container.

        Args:
            singleton_backends: Use singleton lifecycle for backends by default
            singleton_providers: Use singleton lifecycle for providers by default
        """
        self._registrations: dict[tuple[str, str], DependencyRegistration] = {}
        self._singletons: dict[tuple[str, str], Any] = {}
        self._scoped_instances: defaultdict[str, dict[tuple[str, str], Any]] = defaultdict(dict)
        self._weak_refs: WeakValueDictionary = WeakValueDictionary()

        # Default lifecycle settings
        self._default_lifecycles = {
            "backend": Lifecycle.SINGLETON if singleton_backends else Lifecycle.TRANSIENT,
            "embedding": Lifecycle.SINGLETON if singleton_providers else Lifecycle.TRANSIENT,
            "reranking": Lifecycle.SINGLETON if singleton_providers else Lifecycle.TRANSIENT,
            "source": Lifecycle.TRANSIENT,
        }

        # Resource cleanup handlers
        self._cleanup_handlers: list[tuple[Any, Callable]] = []

    def register(
        self,
        component_type: str,
        component_name: str,
        factory: Callable[..., T],
        lifecycle: Lifecycle | None = None,
        dependencies: dict[str, str] | None = None,
        **metadata: Any,
    ) -> None:
        """Register a component with the container.

        Args:
            component_type: Type of component (backend, embedding, etc.)
            component_name: Name of the specific component
            factory: Factory function to create instances
            lifecycle: Component lifecycle (uses default if not specified)
            dependencies: Map of dependency names to component keys
            **metadata: Additional metadata about the component
        """
        key = (component_type, component_name)

        if key in self._registrations:
            logger.warning(
                "Overwriting existing registration for %s:%s", component_type, component_name
            )

        # Use default lifecycle if not specified
        if lifecycle is None:
            lifecycle = self._default_lifecycles.get(component_type, Lifecycle.TRANSIENT)

        registration = DependencyRegistration(
            component_type=component_type,
            component_name=component_name,
            factory=factory,
            lifecycle=lifecycle,
            dependencies=dependencies or {},
            metadata=metadata,
        )

        self._registrations[key] = registration
        logger.debug(
            "Registered %s:%s with lifecycle %s", component_type, component_name, lifecycle.name
        )

    def resolve(
        self,
        component_type: str,
        component_name: str,
        scope_id: str | None = None,
        **override_params: Any,
    ) -> Any:
        """Resolve a component instance.

        Args:
            component_type: Type of component to resolve
            component_name: Name of the specific component
            scope_id: Optional scope identifier for scoped instances
            **override_params: Parameters to override in factory call

        Returns:
            Resolved component instance

        Raises:
            ValueError: If component not registered or circular dependency detected
        """
        context = ResolutionContext(scope_id=scope_id)
        return self._resolve_with_context(component_type, component_name, context, override_params)

    def _resolve_with_context(
        self,
        component_type: str,
        component_name: str,
        context: ResolutionContext,
        override_params: dict[str, Any],
    ) -> Any:
        """Internal resolution with circular dependency detection."""
        key = (component_type, component_name)

        # Check if already resolved in this context
        if key in context.resolved:
            return context.resolved[key]

        # Check for circular dependency
        if key in context.resolving:
            chain = " -> ".join(f"{t}:{n}" for t, n in context.resolving)
            raise ValueError(
                f"Circular dependency detected: {chain} -> {component_type}:{component_name}"
            )

        # Get registration
        registration = self._registrations.get(key)
        if not registration:
            raise ValueError(f"Component not registered: {component_type}:{component_name}")

        # Check lifecycle caches
        if registration.lifecycle == Lifecycle.SINGLETON and key in self._singletons:
            return self._singletons[key]

        if registration.lifecycle == Lifecycle.SCOPED and context.scope_id:
            scoped = self._scoped_instances[context.scope_id].get(key)
            if scoped is not None:
                return scoped

        # Mark as resolving
        context.resolving.add(key)

        try:
            # Resolve dependencies
            resolved_deps = {}
            for dep_name, dep_key in registration.dependencies.items():
                dep_type, dep_component = dep_key.split(":", 1)
                resolved_deps[dep_name] = self._resolve_with_context(
                    dep_type, dep_component, context, {}
                )

            # Merge resolved dependencies with override parameters
            factory_params = {**resolved_deps, **override_params}

            # Create instance
            instance = registration.factory(**factory_params)

            # Cache based on lifecycle
            if registration.lifecycle == Lifecycle.SINGLETON:
                self._singletons[key] = instance
                self._register_cleanup(instance)

            elif registration.lifecycle == Lifecycle.SCOPED and context.scope_id:
                self._scoped_instances[context.scope_id][key] = instance

            # Store in context
            context.resolved[key] = instance

            return instance

        finally:
            context.resolving.remove(key)

    def resolve_dependencies(self, component_type: str, component_name: str) -> dict[str, Any]:
        """Resolve all dependencies for a component without creating it.

        Args:
            component_type: Type of component
            component_name: Name of the specific component

        Returns:
            Dictionary of resolved dependencies
        """
        key = (component_type, component_name)
        registration = self._registrations.get(key)

        if not registration:
            return {}

        context = ResolutionContext()
        resolved_deps = {}

        for dep_name, dep_key in registration.dependencies.items():
            dep_type, dep_component = dep_key.split(":", 1)
            resolved_deps[dep_name] = self._resolve_with_context(
                dep_type, dep_component, context, {}
            )

        return resolved_deps

    def create_scope(self, scope_id: str) -> None:
        """Create a new dependency scope.

        Args:
            scope_id: Unique identifier for the scope
        """
        if scope_id in self._scoped_instances:
            logger.warning("Scope already exists: %s", scope_id)
        else:
            self._scoped_instances[scope_id] = {}
            logger.debug("Created scope: %s", scope_id)

    def dispose_scope(self, scope_id: str) -> None:
        """Dispose of a dependency scope and its instances.

        Args:
            scope_id: Identifier of the scope to dispose
        """
        if scope_id in self._scoped_instances:
            # Cleanup scoped instances
            for instance in self._scoped_instances[scope_id].values():
                self._cleanup_instance(instance)

            del self._scoped_instances[scope_id]
            logger.debug("Disposed scope: %s", scope_id)

    def _register_cleanup(self, instance: Any) -> None:
        """Register cleanup handler for an instance."""
        cleanup_methods = ["close", "cleanup", "dispose", "shutdown"]

        for method_name in cleanup_methods:
            if hasattr(instance, method_name):
                method = getattr(instance, method_name)
                if callable(method):
                    self._cleanup_handlers.append((instance, method))
                    logger.debug(
                        "Registered cleanup handler %s for %s", method_name, type(instance).__name__
                    )
                    break

    def _cleanup_instance(self, instance: Any) -> None:
        """Cleanup a single instance."""
        for inst, handler in self._cleanup_handlers:
            if inst is instance:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        # Handle async cleanup in sync context
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            task = (asyncio.create_task(handler()))
                            tasks = {task}
                            task.add_done_callback(tasks.discard)
                        else:
                            loop.run_until_complete(handler())
                    else:
                        handler()
                except Exception as e:
                    logger.warning("Error during cleanup of %s: %s", type(instance).__name__, e)

    def get_registrations(self) -> dict[tuple[str, str], DependencyRegistration]:
        """Get all registered components.

        Returns:
            Dictionary of all registrations
        """
        return self._registrations.copy()

    def is_registered(self, component_type: str, component_name: str) -> bool:
        """Check if a component is registered.

        Args:
            component_type: Type of component
            component_name: Name of the specific component

        Returns:
            True if registered, False otherwise
        """
        return (component_type, component_name) in self._registrations

    def clear_singletons(self) -> None:
        """Clear all singleton instances (useful for testing)."""
        # Cleanup existing singletons
        for instance in self._singletons.values():
            self._cleanup_instance(instance)

        self._singletons.clear()
        logger.debug("Cleared all singleton instances")

    async def cleanup(self) -> None:
        """Cleanup all managed resources."""
        logger.info("Starting dependency container cleanup")

        # Cleanup all scoped instances
        for scope_id in list(self._scoped_instances.keys()):
            self.dispose_scope(scope_id)

        # Cleanup singletons
        self.clear_singletons()

        # Clear registrations
        self._registrations.clear()
        self._cleanup_handlers.clear()

        logger.info("Dependency container cleanup complete")


class ServiceLocator:
    """Simple service locator pattern for global container access."""

    _container: DependencyContainer | None = None

    @classmethod
    def set_container(cls, container: DependencyContainer) -> None:
        """Set the global container instance."""
        cls._container = container

    @classmethod
    def get_container(cls) -> DependencyContainer:
        """Get the global container instance."""
        if cls._container is None:
            raise RuntimeError("Service container not initialized")
        return cls._container

    @classmethod
    def resolve(cls, component_type: str, component_name: str, **kwargs: Any) -> Any:
        """Resolve a component from the global container."""
        return cls.get_container().resolve(component_type, component_name, **kwargs)
