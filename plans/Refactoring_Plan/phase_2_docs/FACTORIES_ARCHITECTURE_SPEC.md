<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Factories Module Architecture Specification

**Document Version**: 1.0  
**Date**: 2025-01-25  
**Status**: Design Specification  

## Executive Summary

This specification defines the complete redesign of CodeWeaver's factories module to create a unified, extensible architecture that brings together backends, providers, and sources with consistent patterns, plugin support, and capability-driven design. The design eliminates legacy compatibility and establishes the providers module's A+ architecture as the template for all components.

---

## 🎯 Design Principles

### Core Philosophy
- **Capability-Driven Architecture**: All components expose and query capabilities through unified patterns
- **Plugin Extensibility**: Seamless integration of custom backends, providers, and sources
- **Zero Legacy Code**: Complete elimination of legacy compatibility layers
- **Providers-Pattern Template**: All modules follow the proven A+ providers architecture
- **Type Safety**: Enum-based systems with comprehensive Pydantic v2 validation
- **Configuration-First**: Declarative configuration with validation and serialization

### Architectural Tenets
1. **Unified Patterns**: Identical capability queries, registration APIs, and validation frameworks
2. **Composition Over Inheritance**: Favor composition for factory creation and management
3. **Fail Fast**: Early validation and clear error messages
4. **Performance by Design**: Efficient initialization, caching, and resource management
5. **Extensibility by Default**: Plugin system supporting user-defined components

---

## 🏗️ Architecture Overview

### Component Hierarchy

```yaml
CodeWeaverFactory (Orchestrator)
├── ComponentRegistry (Universal Registration)
│   ├── BackendRegistry
│   ├── ProviderRegistry (Already A+)
│   └── SourceRegistry
├── CapabilityManager (Unified Capability Queries)
├── ConfigurationValidator (Cross-Component Validation)
├── PluginManager (Plugin Discovery & Loading)
└── DependencyResolver (Dependency Injection)
```

### Factory Responsibilities

| Factory Component | Primary Responsibility | Secondary Functions |
|-------------------|----------------------|-------------------|
| **CodeWeaverFactory** | Orchestration and lifecycle | Configuration validation, component creation |
| **ComponentRegistry** | Component registration | Capability tracking, availability checks |
| **CapabilityManager** | Capability queries | Compatibility validation, feature detection |
| **ConfigurationValidator** | Configuration validation | Cross-component dependency validation |
| **PluginManager** | Plugin discovery | Custom component registration, validation |
| **DependencyResolver** | Dependency injection | Resource sharing, lifecycle management |

---

## 🔧 Core Components Design

### 1. Universal Component Registry

#### ComponentRegistry Protocol
```python
@runtime_checkable
class ComponentRegistry(Protocol):
    """Universal protocol for component registries."""
    
    def register_component(
        self,
        name: str,
        component_class: type,
        capabilities: BaseCapabilities,
        component_info: BaseComponentInfo,
        *,
        validate: bool = True
    ) -> RegistrationResult
    
    def get_component_info(self, name: str) -> BaseComponentInfo
    def get_capabilities(self, name: str) -> BaseCapabilities
    def list_available_components(self) -> dict[str, BaseComponentInfo]
    def validate_component(self, name: str) -> ValidationResult
```

#### Unified Registration Pattern
```python
# All modules follow identical registration patterns
@dataclass
class ComponentRegistration(Generic[T]):
    """Universal component registration."""
    component_class: type[T]
    capabilities: BaseCapabilities
    component_info: BaseComponentInfo
    is_available: bool = True
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 2. Capability-Driven Architecture

#### Base Capability System
```python
class BaseCapability(Enum):
    """Base capability interface for all component types."""
    pass

# Component-specific capabilities inherit from base
class BackendCapability(BaseCapability):
    VECTOR_SEARCH = "vector_search"
    HYBRID_SEARCH = "hybrid_search"
    METADATA_FILTERING = "metadata_filtering"
    # ... additional capabilities

class ProviderCapability(BaseCapability):  # Already exists and excellent
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    # ... existing capabilities

class SourceCapability(BaseCapability):
    CONTENT_DISCOVERY = "content_discovery"
    CONTENT_READING = "content_reading"
    # ... additional capabilities
```

#### Unified Capability Query Pattern
```python
class CapabilityManager:
    """Unified capability queries across all components."""
    
    def get_component_capabilities(
        self, 
        component_type: ComponentType, 
        component_name: str
    ) -> BaseCapabilities:
        """Universal capability query pattern."""
        registry = self._get_registry(component_type)
        return registry.get_capabilities(component_name)
    
    def supports_capability(
        self,
        component_type: ComponentType,
        component_name: str,
        capability: BaseCapability
    ) -> bool:
        """Check if component supports specific capability."""
        capabilities = self.get_component_capabilities(component_type, component_name)
        return capability in capabilities.supported_capabilities
```

### 3. Configuration System

#### Unified Configuration Model
```python
class CodeWeaverConfig(BaseModel):
    """Master configuration for all components."""
    model_config = ConfigDict(
        extra="forbid",  # Strict validation for main config
        validate_assignment=True,
        frozen=False
    )
    
    # Component configurations
    backend: BackendConfig
    providers: ProviderConfig
    sources: list[SourceConfig] = Field(default_factory=list)
    
    # Factory settings
    factory: FactoryConfig = Field(default_factory=FactoryConfig)
    
    # Plugin settings
    plugins: PluginConfig = Field(default_factory=PluginConfig)
```

#### Component Configuration Templates
```python
class BaseComponentConfig(BaseModel):
    """Base configuration template for all components."""
    model_config = ConfigDict(
        extra="allow",  # Allow component-specific extensions
        validate_assignment=True
    )
    
    # Universal settings
    component_type: str
    enabled: bool = True
    name: str
    provider: str
    
    # Plugin support
    custom_class: str | None = None
    custom_module: str | None = None
```

### 4. Plugin System Architecture

#### Plugin Discovery and Registration
```python
@dataclass
class PluginInfo:
    """Universal plugin information."""
    name: str
    component_type: ComponentType
    plugin_class: type
    capabilities: BaseCapabilities
    metadata: dict[str, Any]
    entry_point: str | None = None

class PluginManager:
    """Universal plugin management."""
    
    def discover_plugins(
        self, 
        plugin_directories: list[str] | None = None
    ) -> dict[ComponentType, list[PluginInfo]]:
        """Discover all available plugins."""
        
    def register_plugin(self, plugin_info: PluginInfo) -> RegistrationResult:
        """Register a discovered plugin."""
        
    def validate_plugin(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin compatibility."""
```

#### Plugin Interface Protocol
```python
@runtime_checkable
class PluginInterface(Protocol):
    """Universal plugin interface."""
    
    @classmethod
    def get_plugin_info(cls) -> PluginInfo:
        """Get plugin metadata and capabilities."""
        
    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        """Get plugin capabilities."""
        
    @classmethod
    def validate_config(cls, config: BaseComponentConfig) -> ValidationResult:
        """Validate plugin configuration."""
```

---

## 🚀 Factory Implementation Design

### 1. CodeWeaver Factory (Main Orchestrator)

```python
class CodeWeaverFactory:
    """
    Main factory orchestrator for all CodeWeaver components.
    
    Provides unified creation, configuration, and lifecycle management
    for backends, providers, and sources with plugin extensibility.
    """
    
    def __init__(
        self,
        config: CodeWeaverConfig | None = None,
        enable_plugins: bool = True,
        enable_dependency_injection: bool = True
    ):
        self._config = config or CodeWeaverConfig()
        self._registries = self._initialize_registries()
        self._capability_manager = CapabilityManager(self._registries)
        self._plugin_manager = PluginManager() if enable_plugins else None
        self._dependency_resolver = DependencyResolver() if enable_dependency_injection else None
        
        # Initialize plugin discovery
        if self._plugin_manager:
            self._discover_and_register_plugins()
    
    def create_server(self, config: CodeWeaverConfig | None = None) -> CodeEmbeddingsServer:
        """Create a complete CodeWeaver server instance."""
        effective_config = config or self._config
        
        # Validate configuration
        validation_result = self.validate_configuration(effective_config)
        if not validation_result.is_valid:
            raise ConfigurationError(validation_result.errors)
        
        # Create components
        backend = self.create_backend(effective_config.backend)
        provider = self.create_provider(effective_config.providers)
        sources = [self.create_source(source_config) for source_config in effective_config.sources]
        
        return CodeEmbeddingsServer(
            backend=backend,
            provider=provider,
            sources=sources
        )
    
    def create_backend(self, config: BackendConfig) -> VectorBackend:
        """Create vector backend with unified factory pattern."""
        return self._create_component("backend", config)
    
    def create_provider(self, config: ProviderConfig) -> EmbeddingProvider:
        """Create embedding provider with unified factory pattern."""
        return self._create_component("provider", config)
    
    def create_source(self, config: SourceConfig) -> DataSource:
        """Create data source with unified factory pattern."""
        return self._create_component("source", config)
```

### 2. Component Creation Pattern

```python
def _create_component(
    self, 
    component_type: str, 
    config: BaseComponentConfig
) -> Any:
    """Unified component creation pattern."""
    
    # Get registry and validate component exists
    registry = self._registries[component_type]
    if not registry.has_component(config.provider):
        raise ComponentNotFoundError(f"{component_type} '{config.provider}' not found")
    
    # Check capabilities and validate configuration
    capabilities = registry.get_capabilities(config.provider)
    validation_result = self._validate_component_config(config, capabilities)
    if not validation_result.is_valid:
        raise ConfigurationError(validation_result.errors)
    
    # Resolve dependencies
    dependencies = {}
    if self._dependency_resolver:
        dependencies = self._dependency_resolver.resolve_dependencies(
            component_type, config.provider
        )
    
    # Create component instance
    component_class = registry.get_component_class(config.provider)
    return component_class(config=config, **dependencies)
```

### 3. Registry Implementation Pattern

```python
class UnifiedBackendRegistry(ComponentRegistry):
    """Backend registry following providers A+ pattern."""
    
    _components: ClassVar[dict[str, ComponentRegistration[VectorBackend]]] = {}
    _initialized: ClassVar[bool] = False
    
    @classmethod
    def register_component(
        cls,
        name: str,
        component_class: type[VectorBackend],
        capabilities: BackendCapabilities,
        component_info: BackendInfo,
        *,
        validate: bool = True
    ) -> RegistrationResult:
        """Register backend following providers pattern."""
        
        # Validate component if requested
        if validate:
            validation_result = cls._validate_backend_class(component_class)
            if not validation_result.is_valid:
                return RegistrationResult(success=False, errors=validation_result.errors)
        
        # Check availability
        is_available, unavailable_reason = cls._check_component_availability(component_class)
        
        # Create registration
        registration = ComponentRegistration(
            component_class=component_class,
            capabilities=capabilities,
            component_info=component_info,
            is_available=is_available,
            unavailable_reason=unavailable_reason
        )
        
        cls._components[name] = registration
        return RegistrationResult(success=True)
```

---

## 🔌 Plugin System Design

### Plugin Discovery Mechanism

```python
class PluginDiscovery:
    """Advanced plugin discovery with multiple source support."""
    
    def __init__(self):
        self._discovery_sources = [
            EntryPointDiscovery(),      # setuptools entry points
            DirectoryDiscovery(),       # file system scanning
            ModuleDiscovery(),          # Python module scanning
            RegistryDiscovery()         # package registry scanning
        ]
    
    def discover_all_plugins(
        self, 
        component_types: list[ComponentType] | None = None,
        plugin_directories: list[str] | None = None
    ) -> dict[ComponentType, list[PluginInfo]]:
        """Comprehensive plugin discovery."""
        
        discovered_plugins = {}
        for source in self._discovery_sources:
            try:
                source_plugins = source.discover_plugins(component_types, plugin_directories)
                discovered_plugins = self._merge_plugin_discoveries(discovered_plugins, source_plugins)
            except Exception as e:
                logger.warning("Plugin discovery source %s failed: %s", source.__class__.__name__, e)
        
        return discovered_plugins
```

### Plugin Validation Framework

```python
class PluginValidator:
    """Comprehensive plugin validation."""
    
    def validate_plugin(self, plugin_info: PluginInfo) -> ValidationResult:
        """Multi-stage plugin validation."""
        
        validation_stages = [
            self._validate_plugin_interface,
            self._validate_plugin_capabilities,
            self._validate_plugin_dependencies,
            self._validate_plugin_security,
            self._validate_plugin_configuration
        ]
        
        for stage in validation_stages:
            result = stage(plugin_info)
            if not result.is_valid:
                return result
        
        return ValidationResult(is_valid=True)
```

---

## 🔄 Configuration and Lifecycle Management

### Configuration Validation Pipeline

```python
class ConfigurationValidator:
    """Multi-stage configuration validation."""
    
    def validate_configuration(self, config: CodeWeaverConfig) -> ValidationResult:
        """Comprehensive configuration validation."""
        
        validation_pipeline = [
            self._validate_syntax,           # Basic syntax and structure
            self._validate_component_availability,  # Component existence
            self._validate_capabilities,     # Capability compatibility
            self._validate_dependencies,     # Inter-component dependencies
            self._validate_resource_requirements,  # Resource constraints
            self._validate_security_requirements   # Security policies
        ]
        
        for validator in validation_pipeline:
            result = validator(config)
            if not result.is_valid:
                return result
        
        return ValidationResult(is_valid=True)
```

### Lifecycle Management

```python
class ComponentLifecycleManager:
    """Unified component lifecycle management."""
    
    async def initialize_component(
        self, 
        component: Any, 
        config: BaseComponentConfig
    ) -> InitializationResult:
        """Initialize component with proper lifecycle management."""
        
        try:
            # Pre-initialization validation
            await self._validate_pre_initialization(component, config)
            
            # Initialize component
            if hasattr(component, 'initialize'):
                await component.initialize()
            
            # Post-initialization validation
            await self._validate_post_initialization(component, config)
            
            return InitializationResult(success=True)
            
        except Exception as e:
            logger.exception("Component initialization failed")
            return InitializationResult(success=False, error=str(e))
    
    async def shutdown_component(self, component: Any) -> ShutdownResult:
        """Graceful component shutdown."""
        try:
            if hasattr(component, 'shutdown'):
                await component.shutdown()
            return ShutdownResult(success=True)
        except Exception as e:
            logger.exception("Component shutdown failed")
            return ShutdownResult(success=False, error=str(e))
```

---

## 📊 Performance and Monitoring

### Factory Performance Design

```python
class FactoryPerformanceManager:
    """Performance monitoring and optimization."""
    
    def __init__(self):
        self._metrics_collector = MetricsCollector()
        self._component_cache = ComponentCache()
        self._resource_monitor = ResourceMonitor()
    
    def optimize_component_creation(
        self, 
        component_type: str, 
        config: BaseComponentConfig
    ) -> OptimizationResult:
        """Optimize component creation process."""
        
        # Check cache first
        if self._component_cache.has_component(component_type, config):
            return OptimizationResult(use_cache=True)
        
        # Analyze resource requirements
        resource_analysis = self._resource_monitor.analyze_requirements(component_type, config)
        
        # Determine optimization strategy
        return OptimizationResult(
            use_cache=False,
            resource_allocation=resource_analysis.recommended_allocation,
            initialization_strategy=resource_analysis.optimal_strategy
        )
```

### Monitoring and Observability

```python
class FactoryObservability:
    """Comprehensive factory monitoring."""
    
    def track_component_creation(
        self,
        component_type: str,
        component_name: str,
        creation_time: float,
        success: bool
    ) -> None:
        """Track component creation metrics."""
        
    def track_plugin_registration(
        self,
        plugin_info: PluginInfo,
        registration_time: float,
        success: bool
    ) -> None:
        """Track plugin registration metrics."""
        
    def get_factory_health(self) -> FactoryHealthReport:
        """Generate comprehensive factory health report."""
```

---

## 🛡️ Security and Validation

### Security Framework

```python
class FactorySecurity:
    """Security validation for factory operations."""
    
    def validate_plugin_security(self, plugin_info: PluginInfo) -> SecurityValidationResult:
        """Validate plugin security requirements."""
        
        security_checks = [
            self._check_code_signing,
            self._check_dependency_security,
            self._check_permission_requirements,
            self._check_data_access_patterns
        ]
        
        for check in security_checks:
            result = check(plugin_info)
            if not result.is_secure:
                return result
        
        return SecurityValidationResult(is_secure=True)
```

---

## 🎯 Migration and Integration Strategy

### Migration from Current Implementation

1. **Phase 1: Core Factory Redesign**
   - Implement new `CodeWeaverFactory` with unified patterns
   - Create universal component registries
   - Establish capability-driven architecture

2. **Phase 2: Plugin System Integration**
   - Implement plugin discovery and validation
   - Create plugin interface protocols
   - Add security validation framework

3. **Phase 3: Legacy Code Elimination**
   - Remove all legacy compatibility adapters
   - Eliminate dual initialization systems
   - Clean up hardcoded capability definitions

4. **Phase 4: Performance Optimization**
   - Add component caching
   - Implement resource monitoring
   - Optimize initialization patterns

### Backward Compatibility Strategy

**NO BACKWARD COMPATIBILITY** - This is a new tool that hasn't been released. The factory system should establish the correct patterns from the start rather than preserving suboptimal legacy APIs.

---

## 📈 Success Metrics

### Architecture Quality Metrics
- **Unified Patterns**: All modules follow identical capability query patterns
- **Plugin Support**: Seamless registration and validation of custom components
- **Type Safety**: Zero TypedDict conflicts, comprehensive enum usage
- **Configuration Validation**: Comprehensive Pydantic v2 validation throughout
- **Performance**: Sub-100ms component creation, efficient resource usage

### Extensibility Metrics
- **Plugin Discovery**: Support for multiple discovery mechanisms
- **Custom Components**: Easy integration of user-defined backends, providers, sources
- **Validation Framework**: Comprehensive validation with clear error messages
- **Documentation**: Complete plugin development guides and examples

---

## 🔮 Future Extensions

### Planned Enhancements
1. **Advanced Plugin Marketplace**: Plugin discovery from remote registries
2. **Configuration Templates**: Pre-built configurations for common use cases
3. **Auto-scaling Components**: Dynamic component scaling based on usage
4. **Advanced Monitoring**: Real-time performance and health monitoring
5. **Configuration Migration**: Automated migration between configuration versions

### Extension Points
- Custom validation rules for specialized environments
- Alternative discovery mechanisms for enterprise environments
- Custom dependency injection frameworks
- Specialized component lifecycle hooks

---

This specification establishes a comprehensive, extensible factory architecture that eliminates legacy code while providing a foundation for CodeWeaver's continued evolution and user extensibility.