# FastMCP Middleware Services Integration Specification

## Overview

This specification defines the integration of FastMCP's builtin middleware (`rate_limiting`, `timing`, `logging`, `error_handling`) with CodeWeaver's services layer architecture. The goal is to make these middleware components configurable through the existing configuration system while maintaining their performance benefits and adding internal service accessibility.

## Current State

FastMCP middleware is currently instantiated directly in `server.py` with hardcoded configurations:
- Error handling middleware: hardcoded settings
- Rate limiting middleware: fixed 1.0 requests/second 
- Logging middleware: uses server config but limited configurability
- Timing middleware: basic configuration

## Target Architecture

### 1. Service Type Extensions

**File**: `src/codeweaver/_types/config.py`

Add new middleware service types to the `ServiceType` enum:

```python
class ServiceType(BaseEnum):
    # Existing services
    CHUNKING = "chunking"
    FILTERING = "filtering" 
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
    METRICS = "metrics"
    
    # NEW: Middleware services
    LOGGING = "logging"
    TIMING = "timing"
    ERROR_HANDLING = "error_handling"
    RATE_LIMITING = "rate_limiting"

    @classmethod
    def get_core_services(cls) -> list["ServiceType"]:
        """Get core services required for basic operation."""
        return [cls.CHUNKING, cls.FILTERING]

    @classmethod
    def get_optional_services(cls) -> list["ServiceType"]:
        """Get optional services for enhanced functionality."""
        return [cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS]
        
    @classmethod
    def get_middleware_services(cls) -> list["ServiceType"]:
        """Get middleware services that bridge FastMCP and services layer."""
        return [cls.LOGGING, cls.TIMING, cls.ERROR_HANDLING, cls.RATE_LIMITING]
```

### 2. Middleware Service Configurations

**File**: `src/codeweaver/_types/service_config.py`

Add middleware-specific configuration classes:

```python
class LoggingServiceConfig(ServiceConfig):
    """Configuration for logging middleware service."""
    
    provider: str = "fastmcp_logging"
    
    # FastMCP LoggingMiddleware parameters
    log_level: Annotated[str, Field(description="Log level for middleware")] = "INFO"
    include_payloads: Annotated[bool, Field(description="Include request/response payloads")] = False
    max_payload_length: Annotated[int, Field(gt=0, description="Max payload chars to log")] = 1000
    methods: Annotated[list[str] | None, Field(description="Methods to log (None=all)")] = None
    
    # Enhanced service features
    structured_logging: Annotated[bool, Field(description="Use structured logging format")] = False
    log_performance_metrics: Annotated[bool, Field(description="Log performance data")] = True
    log_to_service_bridge: Annotated[bool, Field(description="Make logs accessible via service")] = True

class TimingServiceConfig(ServiceConfig):
    """Configuration for timing middleware service."""
    
    provider: str = "fastmcp_timing"
    
    # FastMCP TimingMiddleware parameters  
    log_level: Annotated[str, Field(description="Log level for timing info")] = "INFO"
    
    # Enhanced service features
    track_performance_metrics: Annotated[bool, Field(description="Track detailed metrics")] = True
    expose_metrics_endpoint: Annotated[bool, Field(description="Expose metrics via service")] = True
    metric_aggregation_window: Annotated[int, Field(gt=0, description="Metric window in seconds")] = 300

class ErrorHandlingServiceConfig(ServiceConfig):
    """Configuration for error handling middleware service."""
    
    provider: str = "fastmcp_error_handling"
    
    # FastMCP ErrorHandlingMiddleware parameters
    include_traceback: Annotated[bool, Field(description="Include traceback in errors")] = False
    transform_errors: Annotated[bool, Field(description="Transform to MCP error format")] = True
    
    # Enhanced service features
    error_aggregation: Annotated[bool, Field(description="Aggregate error statistics")] = True
    error_notification_enabled: Annotated[bool, Field(description="Enable error notifications")] = False
    max_error_history: Annotated[int, Field(gt=0, description="Max errors to track")] = 100

class RateLimitingServiceConfig(ServiceConfig):
    """Configuration for rate limiting middleware service."""
    
    provider: str = "fastmcp_rate_limiting"
    
    # FastMCP RateLimitingMiddleware parameters
    max_requests_per_second: Annotated[float, Field(gt=0, description="Max requests per second")] = 1.0
    burst_capacity: Annotated[int, Field(gt=0, description="Burst request capacity")] = 10
    global_limit: Annotated[bool, Field(description="Apply limit globally vs per-client")] = True
    
    # Enhanced service features
    expose_rate_limit_status: Annotated[bool, Field(description="Expose rate limit status")] = True
    rate_limit_metrics: Annotated[bool, Field(description="Track rate limiting metrics")] = True
    custom_rate_limit_keys: Annotated[list[str], Field(description="Custom rate limit groupings")] = Field(default_factory=list)
```

### 3. Service Protocol Extensions

**File**: `src/codeweaver/_types/services.py`

Add middleware service protocols:

```python
class LoggingService(ServiceProvider):
    """Protocol for logging middleware service."""
    
    async def log_request(self, request_data: dict, context: dict | None = None) -> None:
        """Log a request with optional context."""
        ...
        
    async def log_response(self, response_data: dict, context: dict | None = None) -> None:
        """Log a response with optional context.""" 
        ...
        
    async def get_log_metrics(self) -> dict[str, Any]:
        """Get logging metrics and statistics."""
        ...
        
    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance for server registration."""
        ...

class TimingService(ServiceProvider):
    """Protocol for timing middleware service."""
    
    async def start_timing(self, operation_id: str) -> None:
        """Start timing an operation."""
        ...
        
    async def end_timing(self, operation_id: str) -> dict[str, float]:
        """End timing and return metrics."""
        ...
        
    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get aggregated performance metrics."""
        ...
        
    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance for server registration."""
        ...

class ErrorHandlingService(ServiceProvider):
    """Protocol for error handling middleware service."""
    
    async def handle_error(self, error: Exception, context: dict | None = None) -> dict[str, Any]:
        """Handle an error and return processed error info."""
        ...
        
    async def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics and trends."""
        ...
        
    async def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent error history."""
        ...
        
    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance for server registration."""
        ...

class RateLimitingService(ServiceProvider):
    """Protocol for rate limiting middleware service."""
    
    async def check_rate_limit(self, client_id: str, operation: str | None = None) -> bool:
        """Check if request is within rate limits."""
        ...
        
    async def get_rate_limit_status(self, client_id: str | None = None) -> dict[str, Any]:
        """Get current rate limiting status."""
        ...
        
    async def get_rate_limit_metrics(self) -> dict[str, Any]:
        """Get rate limiting metrics."""
        ...
        
    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance for server registration."""
        ...
```

### 4. ServicesConfig Extension

**File**: `src/codeweaver/_types/service_config.py`

Update the main services configuration:

```python
class ServicesConfig(BaseModel):
    """Configuration for all service providers."""
    
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    
    # Existing service configs
    chunking: ChunkingServiceConfig = Field(default_factory=ChunkingServiceConfig)
    filtering: FilteringServiceConfig = Field(default_factory=FilteringServiceConfig)
    validation: ValidationServiceConfig | None = None
    cache: CacheServiceConfig | None = None
    monitoring: MonitoringServiceConfig | None = None
    metrics: MetricsServiceConfig | None = None
    
    # NEW: Middleware service configs
    logging: LoggingServiceConfig = Field(default_factory=LoggingServiceConfig)
    timing: TimingServiceConfig = Field(default_factory=TimingServiceConfig)
    error_handling: ErrorHandlingServiceConfig = Field(default_factory=ErrorHandlingServiceConfig)
    rate_limiting: RateLimitingServiceConfig = Field(default_factory=RateLimitingServiceConfig)
    
    # Global service settings
    health_check_enabled: bool = True
    health_check_interval: float = 60.0
    auto_recovery_enabled: bool = True
    service_timeout: float = 30.0
    max_concurrent_services: int = 10
    
    # NEW: Middleware-specific settings
    middleware_auto_registration: Annotated[bool, Field(description="Auto-register middleware with FastMCP")] = True
    middleware_initialization_order: Annotated[list[str], Field(description="Order to initialize middleware services")] = Field(
        default_factory=lambda: ["error_handling", "rate_limiting", "logging", "timing"]
    )
```

## Configuration Integration

### TOML Configuration Support

**File**: Configuration files (`.codeweaver.toml`, etc.)

Add middleware service configuration sections:

```toml
[services.logging]
enabled = true
provider = "fastmcp_logging"
log_level = "INFO"
include_payloads = false
max_payload_length = 1000
structured_logging = true
log_performance_metrics = true
log_to_service_bridge = true

[services.timing]
enabled = true
provider = "fastmcp_timing"
log_level = "INFO"
track_performance_metrics = true
expose_metrics_endpoint = true
metric_aggregation_window = 300

[services.error_handling]
enabled = true
provider = "fastmcp_error_handling"
include_traceback = false
transform_errors = true
error_aggregation = true
error_notification_enabled = false
max_error_history = 100

[services.rate_limiting]
enabled = true
provider = "fastmcp_rate_limiting"
max_requests_per_second = 2.0
burst_capacity = 15
global_limit = true
expose_rate_limit_status = true
rate_limit_metrics = true
```

## Benefits

1. **Unified Configuration**: All middleware settings configurable via TOML
2. **Service Accessibility**: Middleware functionality accessible to other services
3. **Enhanced Observability**: Rich metrics and status information available
4. **Testability**: Services can be mocked and tested independently
5. **Extensibility**: Easy to add custom middleware behaviors
6. **Consistency**: All cross-cutting concerns managed through services layer

## Implementation Notes

- **Backward Compatibility**: Existing middleware behavior preserved
- **Performance**: No performance degradation from service wrapping
- **Dependencies**: Requires FastMCP server reference in ServicesManager
- **Testing**: Comprehensive test coverage for service integration
- **Documentation**: Update user documentation with new configuration options

## Next Steps

1. Implement middleware service provider classes
2. Update ServicesManager for middleware integration
3. Modify server.py to use service-managed middleware
4. Create migration path for existing configurations
5. Add comprehensive testing for middleware services