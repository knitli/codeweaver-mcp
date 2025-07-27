# ServicesManager Integration Specification

## Overview

This specification defines how to integrate FastMCP middleware services into the existing `ServicesManager` architecture, including middleware registration, lifecycle management, and coordination with the FastMCP server.

## Current ServicesManager Architecture

The current `ServicesManager` handles:
- Service provider registration and lifecycle
- Health monitoring and auto-recovery
- Core services (chunking, filtering) and optional services (validation, cache, monitoring, metrics)
- Service bridge integration with FastMCP tools

## Required Changes

### 1. ServicesManager Constructor Enhancement

**File**: `src/codeweaver/services/manager.py`

Add FastMCP server dependency and middleware management:

```python
class ServicesManager:
    """Manager for coordinating all service providers."""

    def __init__(
        self, 
        config: ServicesConfig, 
        logger: logging.Logger | None = None,
        fastmcp_server: FastMCP | None = None  # NEW: FastMCP server reference
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
```

### 2. Middleware Service Registration

Add middleware provider registration to the `_register_builtin_providers` method:

```python
async def _register_builtin_providers(self) -> None:
    """Register built-in service providers including middleware providers."""
    try:
        # Existing provider registrations...
        
        # NEW: Register middleware service providers
        from codeweaver.services.providers.middleware import (
            FastMCPLoggingProvider,
            FastMCPTimingProvider, 
            FastMCPErrorHandlingProvider,
            FastMCPRateLimitingProvider,
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
        
    except Exception as e:
        error_msg = f"Failed to register builtin providers: {e}"
        self._logger.exception(error_msg)
        raise ServiceInitializationError(error_msg) from e
```

### 3. Middleware Service Creation and Registration

Add new method for creating middleware services:

```python
async def _create_middleware_services(self) -> None:
    """Create and register middleware services with FastMCP server."""
    if not self._fastmcp_server:
        self._logger.warning("No FastMCP server provided, skipping middleware service registration")
        return
        
    try:
        self._logger.info("Creating middleware services")
        
        # Use configured initialization order
        middleware_order = self._config.middleware_initialization_order or [
            "error_handling", "rate_limiting", "logging", "timing"
        ]
        
        for service_name in middleware_order:
            service_type = ServiceType(service_name)
            
            # Get service configuration
            service_config = getattr(self._config, service_name, None)
            if not service_config or not service_config.enabled:
                self._logger.debug("Middleware service %s disabled, skipping", service_name)
                continue
            
            # Create middleware service
            middleware_service = await self._create_middleware_service(service_type, service_config)
            if middleware_service:
                self._middleware_services[service_type] = middleware_service
                self._middleware_registration_order.append(service_type)
                
                # Register middleware with FastMCP server if auto-registration is enabled
                if self._config.middleware_auto_registration:
                    await self._register_middleware_with_server(service_type, middleware_service)
        
        self._logger.info("Middleware services created and registered")
        
    except Exception as e:
        error_msg = f"Failed to create middleware services: {e}"
        self._logger.exception(error_msg)
        raise ServiceInitializationError(error_msg) from e

async def _create_middleware_service(
    self, service_type: ServiceType, config: ServiceConfig
) -> ServiceProvider | None:
    """Create a single middleware service."""
    try:
        # Create service using registry
        service = await self._registry.create_service(service_type, config)
        
        # Initialize the service
        await service.initialize()
        
        self._logger.info("Created middleware service: %s", service_type.value)
        return service
        
    except Exception as e:
        self._logger.exception("Failed to create middleware service %s: %s", service_type.value, e)
        if config.fail_fast:
            raise ServiceCreationError(f"Failed to create {service_type.value} service") from e
        return None

async def _register_middleware_with_server(
    self, service_type: ServiceType, service: ServiceProvider
) -> None:
    """Register middleware service with FastMCP server."""
    try:
        if hasattr(service, 'get_middleware_instance'):
            middleware_instance = service.get_middleware_instance()
            self._fastmcp_server.add_middleware(middleware_instance)
            self._logger.info("Registered %s middleware with FastMCP server", service_type.value)
        else:
            self._logger.warning(
                "Service %s does not provide middleware instance", service_type.value
            )
    except Exception as e:
        self._logger.exception(
            "Failed to register %s middleware with server: %s", service_type.value, e
        )
        raise
```

### 4. Enhanced Initialization Method

Update the main initialization method to include middleware services:

```python
async def initialize(self) -> None:
    """Initialize the services manager and all configured services."""
    if self._initialized:
        self._logger.warning("Services manager already initialized")
        return

    try:
        self._logger.info("Initializing services manager")

        # Register built-in providers (now includes middleware providers)
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
```

### 5. Middleware Service Access Methods

Add methods to access middleware services:

```python
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
```

### 6. Enhanced Health Monitoring

Update health monitoring to include middleware services:

```python
async def get_health_report(self) -> ServicesHealthReport:
    """Get comprehensive health report for all services."""
    try:
        service_health = {}
        
        # Check core and optional services
        for service_type, service in self._services.items():
            try:
                health = await service.health_check()
                service_health[service_type.value] = health
            except Exception as e:
                self._logger.exception("Health check failed for service %s", service_type.value)
                service_health[service_type.value] = ServiceHealth(
                    service_type=service_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    last_check=datetime.utcnow(),
                )
        
        # NEW: Check middleware services
        middleware_health = {}
        for service_type, service in self._middleware_services.items():
            try:
                health = await service.health_check()
                middleware_health[service_type.value] = health
            except Exception as e:
                self._logger.exception("Health check failed for middleware service %s", service_type.value)
                middleware_health[service_type.value] = ServiceHealth(
                    service_type=service_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    last_check=datetime.utcnow(),
                )
        
        # Determine overall status
        all_health = {**service_health, **middleware_health}
        overall_status = HealthStatus.HEALTHY
        
        for health in all_health.values():
            if health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif health.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return ServicesHealthReport(
            overall_status=overall_status,
            service_health=service_health,
            middleware_health=middleware_health,  # NEW: Include middleware health
            timestamp=datetime.utcnow(),
            manager_initialized=self._initialized,
        )
        
    except Exception as e:
        self._logger.exception("Failed to generate health report")
        return ServicesHealthReport(
            overall_status=HealthStatus.UNHEALTHY,
            service_health={},
            middleware_health={},
            timestamp=datetime.utcnow(),
            manager_initialized=self._initialized,
            error=str(e),
        )
```

### 7. Enhanced Shutdown Method

Update shutdown to properly handle middleware services:

```python
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

        # Shutdown middleware services first (reverse order)
        middleware_shutdown_order = list(reversed(self._middleware_registration_order))
        for service_type in middleware_shutdown_order:
            if service := self._middleware_services.get(service_type):
                try:
                    await service.shutdown()
                    self._logger.debug("Shutdown middleware service: %s", service_type.value)
                except Exception as e:
                    self._logger.exception("Error shutting down middleware service %s", service_type.value)

        # Shutdown regular services
        for service_type, service in self._services.items():
            try:
                await service.shutdown()
                self._logger.debug("Shutdown service: %s", service_type.value)
            except Exception as e:
                self._logger.exception("Error shutting down service %s", service_type.value)

        # Clear service caches
        self._services.clear()
        self._middleware_services.clear()
        self._middleware_registration_order.clear()
        self._initialized = False

        self._logger.info("Services manager shutdown complete")

    except Exception as e:
        self._logger.exception("Error during services manager shutdown")
        raise
```

## ServiceBridge Enhancement

**File**: `src/codeweaver/services/middleware_bridge.py`

Update ServiceBridge to include middleware services:

```python
async def _inject_services(self, context: MiddlewareContext) -> None:
    """Inject appropriate services into the context."""
    tool_name = context.message.name

    # Define service injection mappings
    service_mappings = {
        "index_codebase": {
            "chunking_service": ServiceType.CHUNKING,
            "filtering_service": ServiceType.FILTERING,
            "logging_service": ServiceType.LOGGING,  # NEW
            "timing_service": ServiceType.TIMING,    # NEW
        },
        "search_code": {
            "filtering_service": ServiceType.FILTERING,
            "timing_service": ServiceType.TIMING,    # NEW
        },
        "ast_grep_search": {
            "filtering_service": ServiceType.FILTERING,
            "timing_service": ServiceType.TIMING,    # NEW
        },
    }

    # Inject services for this tool
    if tool_name in service_mappings:
        for context_key, service_type in service_mappings[tool_name].items():
            try:
                # Check both regular services and middleware services
                service = (
                    self._services_manager.get_service(service_type) or
                    self._services_manager.get_middleware_service(service_type)
                )
                
                if service:
                    context.fastmcp_context.set_state_value(context_key, service)
                    self._logger.debug("Injected %s service into context", service_type.value)
                else:
                    self._logger.warning("Service %s not available", service_type.value)

            except Exception as e:
                self._logger.warning("Failed to inject %s service: %s", service_type.value, e)
```

## Updated Data Structures

**File**: `src/codeweaver/_types/service_data.py`

Update health report structure to include middleware services:

```python
class ServicesHealthReport(BaseModel):
    """Comprehensive health report for all services."""
    
    model_config = ConfigDict(extra="allow")
    
    overall_status: HealthStatus
    service_health: dict[str, ServiceHealth]
    middleware_health: dict[str, ServiceHealth] = Field(default_factory=dict)  # NEW
    timestamp: datetime
    manager_initialized: bool
    error: str | None = None
```

## Server Integration

**File**: `src/codeweaver/server.py`

Update server to pass FastMCP instance to ServicesManager:

```python
class CodeWeaverServer:
    def __init__(
        self,
        config: "CodeWeaverConfig | None" = None,
        extensibility_config: ExtensibilityConfig | None = None,
    ):
        # ... existing initialization ...
        
        # Create FastMCP server instance
        self.mcp = FastMCP("CodeWeaver")
        
        # NEW: Pass FastMCP server to services manager
        # Note: This will be done during initialization to avoid circular dependencies

    async def initialize(self) -> None:
        """Initialize server with FastMCP middleware and plugin system."""
        if self._initialized:
            logger.warning("Server already initialized")
            return

        logger.info("Initializing clean CodeWeaver server")

        # Initialize plugin system first
        await self.extensibility_manager.initialize()

        # NEW: Create services manager with FastMCP server reference
        services_config = self.config.services  # Assuming services config is added to main config
        self.services_manager = ServicesManager(
            config=services_config,
            fastmcp_server=self.mcp  # Pass FastMCP server reference
        )
        await self.services_manager.initialize()

        # Setup domain-specific middleware (chunking, filtering)
        await self._setup_domain_middleware()

        # Get plugin system components
        await self._initialize_components()

        # Register MCP tools
        self._register_tools()

        self._initialized = True
        logger.info("Clean CodeWeaver server initialization complete")

    async def _setup_domain_middleware(self) -> None:
        """Setup domain-specific middleware (chunking, filtering)."""
        # NOTE: FastMCP builtin middleware now handled by ServicesManager
        # Only setup domain-specific middleware here
        
        logger.info("Setting up domain-specific middleware")

        # Chunking middleware (using existing chunking config)
        chunking_config = {
            "max_chunk_size": self.config.chunking.max_chunk_size,
            "min_chunk_size": self.config.chunking.min_chunk_size,
            "ast_grep_enabled": True,
        }
        chunking_middleware = ChunkingMiddleware(chunking_config)
        self.mcp.add_middleware(chunking_middleware)
        logger.info("Added chunking middleware")

        # File filtering middleware (using existing indexing config)
        filtering_config = {
            "use_gitignore": self.config.indexing.use_gitignore,
            "max_file_size": f"{self.config.chunking.max_file_size_mb}MB",
            "excluded_dirs": self.config.indexing.additional_ignore_patterns,
            "included_extensions": None,
        }
        filtering_middleware = FileFilteringMiddleware(filtering_config)
        self.mcp.add_middleware(filtering_middleware)
        logger.info("Added file filtering middleware")

        logger.info("Domain-specific middleware setup complete")
```

## Configuration Integration

Add services configuration to main CodeWeaver config:

```python
class CodeWeaverConfig(BaseSettings):
    """Main configuration for CodeWeaver."""
    
    # ... existing configuration sections ...
    
    # NEW: Services configuration
    services: ServicesConfig = Field(default_factory=ServicesConfig)
```

## Benefits of This Integration

1. **Unified Management**: All services managed through single ServicesManager
2. **Configurable Middleware**: FastMCP middleware fully configurable via TOML
3. **Health Monitoring**: Middleware services included in health monitoring
4. **Service Access**: Middleware functionality accessible to other services and tools
5. **Proper Lifecycle**: Middleware services follow standard service lifecycle
6. **Dependency Injection**: Middleware services available via service bridge
7. **Observability**: Rich metrics and status information for middleware

## Implementation Notes

1. **Initialization Order**: Middleware services must be created after FastMCP server but before tool registration
2. **Error Handling**: Robust error handling to prevent middleware failures from breaking initialization
3. **Backward Compatibility**: Existing middleware behavior preserved
4. **Testing**: Comprehensive testing for service integration and middleware registration
5. **Documentation**: Update documentation to reflect new configuration options and service access patterns