<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Factories: Initialization and Error Handling Patterns

**Document Version**: 1.0  
**Date**: 2025-01-25  
**Status**: Implementation Guide

## Executive Summary

This document defines comprehensive initialization and error handling patterns for the CodeWeaver factories system, ensuring robust, predictable, and graceful behavior across all components, plugins, and factory operations. The patterns follow fail-fast principles while providing clear recovery paths and detailed error information.

---

## 🚀 Initialization Patterns

### 1. Component Lifecycle States

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable
import time

class ComponentState(Enum):
    """Component lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing" 
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"

@dataclass
class ComponentLifecycle:
    """Component lifecycle tracking."""
    component_name: str
    component_type: ComponentType
    state: ComponentState = ComponentState.UNINITIALIZED
    created_at: float = field(default_factory=time.time)
    initialized_at: float | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    error_at: float | None = None
    last_error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def transition_to(self, new_state: ComponentState, error: Exception | None = None) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        current_time = time.time()
        
        if new_state == ComponentState.INITIALIZED:
            self.initialized_at = current_time
        elif new_state == ComponentState.RUNNING:
            self.started_at = current_time
        elif new_state == ComponentState.STOPPED:
            self.stopped_at = current_time
        elif new_state == ComponentState.ERROR:
            self.error_at = current_time
            self.last_error = error
        
        logger.debug(
            "Component %s (%s) transitioned from %s to %s",
            self.component_name,
            self.component_type.value,
            old_state.value,
            new_state.value
        )
```

### 2. Factory Initialization Pipeline

```python
from typing import Protocol
from abc import abstractmethod

class InitializationStage(Protocol):
    """Protocol for initialization stages."""
    
    @abstractmethod
    async def execute(self, context: "InitializationContext") -> "InitializationResult":
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

@dataclass
class InitializationContext:
    """Context for initialization process."""
    factory: "CodeWeaverFactory"
    config: CodeWeaverConfig
    registries: dict[str, ComponentRegistry]
    plugin_manager: "PluginManager | None"
    dependency_resolver: "DependencyResolver | None"
    metadata: dict[str, Any] = field(default_factory=dict)
    stage_results: dict[str, Any] = field(default_factory=dict)

@dataclass
class InitializationResult:
    """Result of an initialization stage."""
    success: bool
    stage_name: str
    duration_ms: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class FactoryInitializer:
    """Factory initialization orchestrator."""
    
    def __init__(self):
        self._stages: list[InitializationStage] = [
            ConfigurationValidationStage(),
            RegistryInitializationStage(),
            PluginDiscoveryStage(),
            ComponentValidationStage(),
            DependencyResolutionStage(),
            HealthCheckStage()
        ]
    
    async def initialize_factory(
        self,
        factory: "CodeWeaverFactory",
        config: CodeWeaverConfig
    ) -> "FactoryInitializationResult":
        """Execute complete factory initialization."""
        
        context = InitializationContext(
            factory=factory,
            config=config,
            registries={
                "backend": factory._backend_registry,
                "provider": factory._provider_registry,
                "source": factory._source_registry
            },
            plugin_manager=factory._plugin_manager,
            dependency_resolver=factory._dependency_resolver
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
                        "Initialization stage '%s' failed: %s",
                        result.stage_name,
                        result.errors
                    )
                    break
                    
                logger.info(
                    "Initialization stage '%s' completed in %.2fms",
                    result.stage_name,
                    result.duration_ms
                )
                
            except Exception as e:
                logger.exception("Initialization stage '%s' crashed", stage.get_stage_name())
                stage_results.append(InitializationResult(
                    success=False,
                    stage_name=stage.get_stage_name(),
                    duration_ms=(time.time() - stage_start) * 1000,
                    errors=[f"Stage crashed: {e}"]
                ))
                break
        
        total_duration = (time.time() - overall_start) * 1000
        overall_success = all(result.success for result in stage_results)
        
        return FactoryInitializationResult(
            success=overall_success,
            total_duration_ms=total_duration,
            stage_results=stage_results,
            factory_state=ComponentState.INITIALIZED if overall_success else ComponentState.ERROR
        )

@dataclass
class FactoryInitializationResult:
    """Result of factory initialization."""
    success: bool
    total_duration_ms: float
    stage_results: list[InitializationResult]
    factory_state: ComponentState
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

### 3. Initialization Stages Implementation

```python
class ConfigurationValidationStage(InitializationStage):
    """Validate factory configuration."""
    
    def get_stage_name(self) -> str:
        return "configuration_validation"
    
    def get_dependencies(self) -> list[str]:
        return []
    
    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Validate configuration completeness and consistency."""
        errors = []
        warnings = []
        
        # Validate configuration structure
        try:
            # Use Pydantic validation
            validated_config = CodeWeaverConfig.model_validate(context.config.model_dump())
            context.config = validated_config
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        
        # Validate component consistency
        if context.config.backend.provider == context.config.providers.provider:
            warnings.append("Backend and provider have same name - ensure this is intentional")
        
        # Validate resource requirements
        await self._validate_resource_requirements(context, errors, warnings)
        
        return InitializationResult(
            success=len(errors) == 0,
            stage_name=self.get_stage_name(),
            duration_ms=0,  # Will be set by caller
            errors=errors,
            warnings=warnings
        )

class RegistryInitializationStage(InitializationStage):
    """Initialize component registries."""
    
    def get_stage_name(self) -> str:
        return "registry_initialization"
    
    def get_dependencies(self) -> list[str]:
        return ["configuration_validation"]
    
    async def execute(self, context: InitializationContext) -> InitializationResult:
        """Initialize and populate component registries."""
        errors = []
        warnings = []
        
        # Initialize built-in components
        try:
            await self._initialize_backend_registry(context.registries["backend"])
            await self._initialize_provider_registry(context.registries["provider"])
            await self._initialize_source_registry(context.registries["source"])
        except Exception as e:
            errors.append(f"Registry initialization failed: {e}")
        
        # Validate registry consistency
        registry_stats = {}
        for name, registry in context.registries.items():
            available_components = registry.list_available_components()
            registry_stats[name] = len(available_components)
            
            if len(available_components) == 0:
                warnings.append(f"No components available in {name} registry")
        
        return InitializationResult(
            success=len(errors) == 0,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
            metadata={"registry_stats": registry_stats}
        )

class PluginDiscoveryStage(InitializationStage):
    """Discover and register plugins."""
    
    def get_stage_name(self) -> str:
        return "plugin_discovery"
    
    def get_dependencies(self) -> list[str]:
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
                warnings=["Plugin system disabled"]
            )
        
        try:
            # Discover plugins
            discovered_plugins = context.plugin_manager.discover_plugins()
            
            # Register discovered plugins
            registration_results = context.plugin_manager.register_discovered_plugins()
            
            # Analyze results
            successful_registrations = [
                name for name, result in registration_results.items() 
                if result.success
            ]
            failed_registrations = [
                (name, result.errors) for name, result in registration_results.items()
                if not result.success
            ]
            
            if failed_registrations:
                for name, plugin_errors in failed_registrations:
                    warnings.append(f"Plugin '{name}' registration failed: {plugin_errors}")
            
            plugin_stats = {
                "discovered": sum(len(plugins) for plugins in discovered_plugins.values()),
                "registered": len(successful_registrations),
                "failed": len(failed_registrations)
            }
            
        except Exception as e:
            errors.append(f"Plugin discovery failed: {e}")
            plugin_stats = {"error": str(e)}
        
        return InitializationResult(
            success=len(errors) == 0,
            stage_name=self.get_stage_name(),
            duration_ms=0,
            errors=errors,
            warnings=warnings,
            metadata={"plugin_stats": plugin_stats}
        )
```

### 4. Component Initialization Pattern

```python
class ComponentInitializer:
    """Universal component initialization pattern."""
    
    @staticmethod
    async def initialize_component(
        component: Any,
        config: BaseComponentConfig,
        lifecycle: ComponentLifecycle
    ) -> InitializationResult:
        """Initialize a component following standard pattern."""
        
        errors = []
        warnings = []
        
        try:
            lifecycle.transition_to(ComponentState.INITIALIZING)
            
            # Pre-initialization validation
            pre_validation = await ComponentInitializer._pre_initialization_validation(
                component, config
            )
            if not pre_validation.success:
                errors.extend(pre_validation.errors)
                warnings.extend(pre_validation.warnings)
            
            # Component initialization
            if hasattr(component, 'initialize'):
                await component.initialize()
            elif hasattr(component, '__aenter__'):
                await component.__aenter__()
            
            # Post-initialization validation
            post_validation = await ComponentInitializer._post_initialization_validation(
                component, config
            )
            if not post_validation.success:
                errors.extend(post_validation.errors)
                warnings.extend(post_validation.warnings)
            
            # Health check
            health_result = await ComponentInitializer._component_health_check(component)
            if not health_result.success:
                errors.extend(health_result.errors)
                warnings.extend(health_result.warnings)
            
            if len(errors) == 0:
                lifecycle.transition_to(ComponentState.INITIALIZED)
            else:
                lifecycle.transition_to(ComponentState.ERROR)
            
        except Exception as e:
            errors.append(f"Component initialization failed: {e}")
            lifecycle.transition_to(ComponentState.ERROR, e)
        
        return InitializationResult(
            success=len(errors) == 0,
            stage_name=f"component_init_{config.provider}",
            duration_ms=0,
            errors=errors,
            warnings=warnings
        )
```

---

## 🚨 Error Handling Patterns

### 1. Error Classification System

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import traceback

class ErrorSeverity(Enum):
    """Error severity levels."""
    TRACE = "trace"          # Debugging information
    DEBUG = "debug"          # Debug-level issues
    INFO = "info"            # Informational messages
    WARNING = "warning"      # Warnings that don't prevent operation
    ERROR = "error"          # Errors that prevent operation
    CRITICAL = "critical"    # Critical errors requiring immediate attention
    FATAL = "fatal"          # Fatal errors causing system shutdown

class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"      # Configuration-related errors
    VALIDATION = "validation"           # Validation failures
    COMPONENT = "component"             # Component-specific errors
    PLUGIN = "plugin"                   # Plugin-related errors
    NETWORK = "network"                 # Network connectivity errors
    RESOURCE = "resource"               # Resource availability errors
    SECURITY = "security"               # Security violations
    SYSTEM = "system"                   # System-level errors
    USER = "user"                       # User input errors

@dataclass
class ErrorContext:
    """Detailed error context information."""
    component_type: ComponentType | None = None
    component_name: str | None = None
    operation: str | None = None
    config_section: str | None = None
    plugin_name: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class FactoryError:
    """Comprehensive error information."""
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: ErrorContext
    exception: Exception | None = None
    timestamp: float = field(default_factory=time.time)
    traceback_str: str | None = None
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recovery_suggestions: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-init processing."""
        if self.exception and not self.traceback_str:
            self.traceback_str = ''.join(traceback.format_exception(
                type(self.exception), self.exception, self.exception.__traceback__
            ))
```

### 2. Error Handler System

```python
class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self._error_handlers: dict[ErrorCategory, list[Callable]] = defaultdict(list)
        self._error_log: list[FactoryError] = []
        self._recovery_strategies: dict[ErrorCategory, list[Callable]] = defaultdict(list)
    
    def register_error_handler(
        self, 
        category: ErrorCategory,
        handler: Callable[[FactoryError], None]
    ) -> None:
        """Register an error handler for a specific category."""
        self._error_handlers[category].append(handler)
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[FactoryError], bool]
    ) -> None:
        """Register a recovery strategy for a specific error category."""
        self._recovery_strategies[category].append(strategy)
    
    def handle_error(self, error: FactoryError) -> bool:
        """Handle an error using registered handlers and recovery strategies."""
        
        # Log the error
        self._error_log.append(error)
        logger.log(
            self._severity_to_log_level(error.severity),
            "Factory error [%s/%s]: %s",
            error.category.value,
            error.severity.value,
            error.message,
            extra={"error_context": error.context.__dict__}
        )
        
        # Execute category-specific handlers
        for handler in self._error_handlers[error.category]:
            try:
                handler(error)
            except Exception as e:
                logger.exception("Error handler failed: %s", e)
        
        # Attempt recovery
        recovery_successful = False
        for strategy in self._recovery_strategies[error.category]:
            try:
                if strategy(error):
                    recovery_successful = True
                    logger.info("Recovery successful for error %s", error.error_id)
                    break
            except Exception as e:
                logger.exception("Recovery strategy failed: %s", e)
        
        return recovery_successful
    
    def create_error(
        self,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: str,
        context: ErrorContext | None = None,
        exception: Exception | None = None
    ) -> FactoryError:
        """Create a standardized error object."""
        
        return FactoryError(
            severity=severity,
            category=category,
            message=message,
            context=context or ErrorContext(),
            exception=exception,
            recovery_suggestions=self._generate_recovery_suggestions(category, exception)
        )
    
    def _generate_recovery_suggestions(
        self,
        category: ErrorCategory,
        exception: Exception | None
    ) -> list[str]:
        """Generate recovery suggestions based on error category."""
        
        suggestions_map = {
            ErrorCategory.CONFIGURATION: [
                "Check configuration file syntax and completeness",
                "Verify all required environment variables are set",
                "Ensure configuration values are within valid ranges"
            ],
            ErrorCategory.COMPONENT: [
                "Verify component dependencies are installed",
                "Check component configuration parameters",
                "Ensure required services are running"
            ],
            ErrorCategory.PLUGIN: [
                "Verify plugin compatibility with current version",
                "Check plugin installation and dependencies",
                "Validate plugin configuration"
            ],
            ErrorCategory.NETWORK: [
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Check firewall and proxy settings"
            ],
            ErrorCategory.RESOURCE: [
                "Check available system resources (memory, disk, CPU)",
                "Verify database/storage accessibility",
                "Check service quotas and limits"
            ]
        }
        
        return suggestions_map.get(category, ["Contact support for assistance"])
```

### 3. Graceful Degradation Patterns

```python
class GracefulDegradationManager:
    """Manages graceful degradation when components fail."""
    
    def __init__(self):
        self._fallback_strategies: dict[ComponentType, list[FallbackStrategy]] = {
            ComponentType.BACKEND: [
                InMemoryBackendFallback(),
                ReadOnlyModeFallback()
            ],
            ComponentType.PROVIDER: [
                CachedEmbeddingsFallback(),
                LocalEmbeddingsFallback()
            ],
            ComponentType.SOURCE: [
                CachedContentFallback(),
                SingleSourceFallback()
            ]
        }
    
    async def handle_component_failure(
        self,
        component_type: ComponentType,
        original_config: BaseComponentConfig,
        error: FactoryError
    ) -> ComponentFallbackResult:
        """Handle component failure with graceful degradation."""
        
        fallback_strategies = self._fallback_strategies.get(component_type, [])
        
        for strategy in fallback_strategies:
            try:
                if await strategy.can_handle(original_config, error):
                    fallback_component = await strategy.create_fallback(original_config)
                    
                    return ComponentFallbackResult(
                        success=True,
                        fallback_component=fallback_component,
                        strategy_name=strategy.get_name(),
                        limitations=strategy.get_limitations(),
                        warnings=[
                            f"Using fallback strategy: {strategy.get_name()}",
                            f"Limitations: {', '.join(strategy.get_limitations())}"
                        ]
                    )
            except Exception as e:
                logger.warning("Fallback strategy '%s' failed: %s", strategy.get_name(), e)
        
        return ComponentFallbackResult(
            success=False,
            errors=[f"No suitable fallback strategy for {component_type.value}"]
        )

@dataclass
class ComponentFallbackResult:
    """Result of component fallback attempt."""
    success: bool
    fallback_component: Any | None = None
    strategy_name: str | None = None
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

class FallbackStrategy(Protocol):
    """Protocol for fallback strategies."""
    
    @abstractmethod
    async def can_handle(
        self, 
        original_config: BaseComponentConfig, 
        error: FactoryError
    ) -> bool:
        """Check if this strategy can handle the failure."""
        ...
    
    @abstractmethod
    async def create_fallback(self, original_config: BaseComponentConfig) -> Any:
        """Create a fallback component."""
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        ...
    
    @abstractmethod
    def get_limitations(self) -> list[str]:
        """Get list of limitations for this fallback."""
        ...
```

### 4. Retry and Circuit Breaker Patterns

```python
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_ms: float = 100
    max_delay_ms: float = 5000
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: tuple[type[Exception], ...] = (Exception,)

class RetryHandler:
    """Retry handler with exponential backoff and jitter."""
    
    @staticmethod
    async def retry_async(
        operation: Callable[[], Awaitable[T]],
        config: RetryConfig,
        operation_name: str = "operation"
    ) -> T:
        """Retry an async operation with exponential backoff."""
        
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                result = await operation()
                if attempt > 0:
                    logger.info(
                        "%s succeeded on attempt %d/%d",
                        operation_name,
                        attempt + 1,
                        config.max_attempts
                    )
                return result
                
            except config.retry_on_exceptions as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    logger.error(
                        "%s failed after %d attempts: %s",
                        operation_name,
                        config.max_attempts,
                        e
                    )
                    break
                
                # Calculate delay
                delay_ms = config.base_delay_ms
                if config.exponential_backoff:
                    delay_ms = min(
                        config.base_delay_ms * (2 ** attempt),
                        config.max_delay_ms
                    )
                
                if config.jitter:
                    delay_ms *= (0.5 + random.random() * 0.5)
                
                logger.warning(
                    "%s failed on attempt %d/%d: %s (retrying in %.1fms)",
                    operation_name,
                    attempt + 1,
                    config.max_attempts,
                    e,
                    delay_ms
                )
                
                await asyncio.sleep(delay_ms / 1000)
        
        raise last_exception

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout_ms: float = 30000
    success_threshold: int = 3
    timeout_ms: float = 5000

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for component operations."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "circuit"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self._lock = asyncio.Lock()
    
    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation through circuit breaker."""
        
        async with self._lock:
            # Check if circuit should transition states
            await self._check_state_transition()
            
            # Handle open circuit
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
        
        # Execute operation
        try:
            start_time = time.time()
            
            # Apply timeout
            result = await asyncio.wait_for(
                operation(),
                timeout=self.config.timeout_ms / 1000
            )
            
            # Record success
            async with self._lock:
                await self._record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            async with self._lock:
                await self._record_failure()
            raise
    
    async def _check_state_transition(self) -> None:
        """Check if circuit breaker should transition states."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (current_time - self.last_failure_time) * 1000 >= self.config.recovery_timeout_ms:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker '%s' transitioning to half-open", self.name)
    
    async def _record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker '%s' recovered (closed)", self.name)
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker '%s' opened due to failures", self.name)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker '%s' reopened during recovery", self.name)

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass
```

### 5. Health Monitoring and Diagnostics

```python
@dataclass
class HealthStatus:
    """Component health status."""
    is_healthy: bool
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    response_time_ms: float | None = None

class HealthChecker:
    """Component health monitoring."""
    
    def __init__(self):
        self._health_checks: dict[str, Callable[[], Awaitable[HealthStatus]]] = {}
        self._health_cache: dict[str, HealthStatus] = {}
        self._cache_ttl_ms = 5000  # 5 second cache
    
    def register_health_check(
        self,
        component_name: str,
        health_check: Callable[[], Awaitable[HealthStatus]]
    ) -> None:
        """Register a health check for a component."""
        self._health_checks[component_name] = health_check
    
    async def check_component_health(self, component_name: str) -> HealthStatus:
        """Check health of a specific component."""
        
        # Check cache first
        if component_name in self._health_cache:
            cached_status = self._health_cache[component_name]
            if (time.time() - cached_status.last_check) * 1000 < self._cache_ttl_ms:
                return cached_status
        
        # Perform health check
        if component_name not in self._health_checks:
            return HealthStatus(
                is_healthy=False,
                status=f"No health check registered for {component_name}"
            )
        
        try:
            start_time = time.time()
            status = await self._health_checks[component_name]()
            status.response_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self._health_cache[component_name] = status
            return status
            
        except Exception as e:
            error_status = HealthStatus(
                is_healthy=False,
                status=f"Health check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
            self._health_cache[component_name] = error_status
            return error_status
    
    async def check_all_components_health(self) -> dict[str, HealthStatus]:
        """Check health of all registered components."""
        
        health_results = {}
        
        # Create tasks for parallel health checks
        tasks = {
            name: self.check_component_health(name)
            for name in self._health_checks
        }
        
        # Execute health checks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=10.0  # 10 second timeout for all checks
            )
            
            for name, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    health_results[name] = HealthStatus(
                        is_healthy=False,
                        status=f"Health check exception: {result}"
                    )
                else:
                    health_results[name] = result
                    
        except asyncio.TimeoutError:
            for name in tasks:
                if name not in health_results:
                    health_results[name] = HealthStatus(
                        is_healthy=False,
                        status="Health check timeout"
                    )
        
        return health_results
```

---

## 📊 Usage Examples

### Factory Initialization with Error Handling

```python
async def initialize_factory_with_error_handling():
    """Example of factory initialization with comprehensive error handling."""
    
    error_handler = ErrorHandler()
    
    # Register error handlers
    error_handler.register_error_handler(
        ErrorCategory.CONFIGURATION,
        lambda error: logger.error("Config error: %s", error.message)
    )
    
    # Register recovery strategies
    error_handler.register_recovery_strategy(
        ErrorCategory.COMPONENT,
        lambda error: attempt_component_recovery(error)
    )
    
    try:
        factory = CodeWeaverFactory(
            config=CodeWeaverConfig.from_file("config.yaml"),
            enable_plugins=True
        )
        
        # Initialize with monitoring
        initializer = FactoryInitializer()
        result = await initializer.initialize_factory(factory, factory._config)
        
        if not result.success:
            logger.error("Factory initialization failed")
            for stage_result in result.stage_results:
                if not stage_result.success:
                    for error_msg in stage_result.errors:
                        error = error_handler.create_error(
                            ErrorSeverity.ERROR,
                            ErrorCategory.SYSTEM,
                            error_msg,
                            ErrorContext(operation=stage_result.stage_name)
                        )
                        error_handler.handle_error(error)
        
        return factory, result.success
        
    except Exception as e:
        error = error_handler.create_error(
            ErrorSeverity.FATAL,
            ErrorCategory.SYSTEM,
            f"Factory creation failed: {e}",
            exception=e
        )
        error_handler.handle_error(error)
        raise

### Component Creation with Retry

```python
async def create_component_with_retry():
    """Example of component creation with retry logic."""
    
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay_ms=1000,
        exponential_backoff=True
    )
    
    async def create_backend():
        return factory.create_backend(BackendConfig(
            provider="qdrant",
            url="http://localhost:6333"
        ))
    
    try:
        backend = await RetryHandler.retry_async(
            create_backend,
            retry_config,
            "backend_creation"
        )
        return backend
        
    except Exception as e:
        logger.error("Backend creation failed after retries: %s", e)
        
        # Attempt graceful degradation
        degradation_manager = GracefulDegradationManager()
        fallback_result = await degradation_manager.handle_component_failure(
            ComponentType.BACKEND,
            BackendConfig(provider="qdrant"),
            FactoryError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.COMPONENT,
                message=str(e),
                context=ErrorContext(component_type=ComponentType.BACKEND)
            )
        )
        
        if fallback_result.success:
            logger.warning("Using fallback backend: %s", fallback_result.strategy_name)
            return fallback_result.fallback_component
        
        raise
```

This comprehensive initialization and error handling framework ensures robust, predictable behavior across all factory operations while providing clear recovery paths and detailed diagnostic information.