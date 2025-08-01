<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Server Configuration Changes Specification

## Overview

This specification defines the changes needed to integrate middleware services with the existing configuration system, including TOML configuration updates, environment variable support, and configuration validation.

## Current Configuration Architecture

CodeWeaver uses:
- Hierarchical TOML configuration files (`.codeweaver.toml`, etc.)
- Pydantic-based configuration models with validation
- Environment variable overrides
- Multiple configuration locations (workspace, repository, user)

## Required Configuration Changes

### 1. Main Configuration Integration

**File**: `src/codeweaver/config.py`

Add services configuration to the main CodeWeaver configuration:

```python
# Add to imports
from codeweaver.cw_types import ServicesConfig

class CodeWeaverConfig(BaseSettings):
    """Main configuration for CodeWeaver."""

    model_config = SettingsConfigDict(
        env_prefix="CW_",
        env_nested_delimiter="__",
        toml_file=None,  # Set dynamically
        case_sensitive=False,
        extra="allow",
        validate_assignment=True,
    )

    # Existing configuration sections
    embedding: EmbeddingProviderConfig = Field(default_factory=EmbeddingProviderConfig)
    reranking: RerankingProviderConfig = Field(default_factory=RerankingProviderConfig)
    backend: BackendConfigExtended = Field(default_factory=BackendConfigExtended)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # NEW: Services configuration
    services: ServicesConfig = Field(
        default_factory=ServicesConfig,
        description="Configuration for all service providers including middleware"
    )

    @field_validator("services")
    @classmethod
    def validate_services_config(cls, v: ServicesConfig) -> ServicesConfig:
        """Validate services configuration."""
        # Validate middleware initialization order
        valid_middleware = {"logging", "timing", "error_handling", "rate_limiting"}
        for service_name in v.middleware_initialization_order:
            if service_name not in valid_middleware:
                raise ValueError(f"Invalid middleware service name: {service_name}")

        # Validate that core services are enabled
        if not v.chunking.enabled:
            logger.warning("Chunking service is disabled - this may affect functionality")
        if not v.filtering.enabled:
            logger.warning("Filtering service is disabled - this may affect functionality")

        return v
```

### 2. Environment Variable Support

Add environment variable mapping for middleware services:

```python
# Environment variable patterns for middleware services:
# CW_SERVICES__LOGGING__ENABLED=true
# CW_SERVICES__LOGGING__LOG_LEVEL=DEBUG
# CW_SERVICES__TIMING__TRACK_PERFORMANCE_METRICS=true
# CW_SERVICES__ERROR_HANDLING__INCLUDE_TRACEBACK=false
# CW_SERVICES__RATE_LIMITING__MAX_REQUESTS_PER_SECOND=2.0
```

### 3. TOML Configuration Schema

**Example TOML Configuration**: `.codeweaver.toml`

```toml
# CodeWeaver Configuration File

[embedding]
provider = "voyage"
model = "voyage-code-2"

[backend]
provider = "qdrant"
url = "http://localhost:6333"
collection_name = "codeweaver-default"

[chunking]
max_chunk_size = 1500
min_chunk_size = 50
max_file_size_mb = 1

[indexing]
use_gitignore = true
additional_ignore_patterns = ["build", "dist", ".next"]

[server]
log_level = "INFO"
enable_request_logging = false

# NEW: Services Configuration
[services]
health_check_enabled = true
health_check_interval = 60.0
auto_recovery_enabled = true
middleware_auto_registration = true
middleware_initialization_order = ["error_handling", "rate_limiting", "logging", "timing"]

# Core Services
[services.chunking]
enabled = true
provider = "fastmcp_chunking"
max_chunk_size = 1500
min_chunk_size = 50
ast_grep_enabled = true
respect_code_structure = true
preserve_comments = true

[services.filtering]
enabled = true
provider = "fastmcp_filtering"
max_file_size = 1048576  # 1MB in bytes
use_gitignore = true
parallel_scanning = true
max_concurrent_scans = 10

# Middleware Services
[services.logging]
enabled = true
provider = "fastmcp_logging"
log_level = "INFO"
include_payloads = false
max_payload_length = 1000
structured_logging = false
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
max_requests_per_second = 1.0
burst_capacity = 10
global_limit = true
expose_rate_limit_status = true
rate_limit_metrics = true

# Optional Services
[services.validation]
enabled = false
provider = "default_validation"

[services.cache]
enabled = false
provider = "memory_cache"

[services.monitoring]
enabled = false
provider = "prometheus_monitoring"

[services.metrics]
enabled = false
provider = "statsd_metrics"
```

### 4. Configuration Migration

**File**: `src/codeweaver/config.py`

Add configuration migration support for existing installations:

```python
class ConfigMigration:
    """Handle migration of existing configurations to include services."""

    @staticmethod
    def migrate_server_config_to_services(config: dict) -> dict:
        """Migrate existing server-level middleware config to services config."""
        migrated = config.copy()

        # Create services section if it doesn't exist
        if "services" not in migrated:
            migrated["services"] = {}

        services = migrated["services"]

        # Migrate existing server logging settings to services.logging
        if "server" in migrated:
            server_config = migrated["server"]

            if "logging" not in services:
                services["logging"] = {}

            # Map server log_level to services.logging.log_level
            if "log_level" in server_config:
                services["logging"]["log_level"] = server_config["log_level"]

            # Map enable_request_logging to include_payloads
            if "enable_request_logging" in server_config:
                services["logging"]["include_payloads"] = server_config["enable_request_logging"]

        # Migrate chunking config to services.chunking
        if "chunking" in migrated:
            chunking_config = migrated["chunking"]

            if "chunking" not in services:
                services["chunking"] = {}

            # Copy chunking settings
            for key, value in chunking_config.items():
                services["chunking"][key] = value

            # Ensure provider is set
            if "provider" not in services["chunking"]:
                services["chunking"]["provider"] = "fastmcp_chunking"

        # Migrate indexing config to services.filtering
        if "indexing" in migrated:
            indexing_config = migrated["indexing"]

            if "filtering" not in services:
                services["filtering"] = {}

            # Map indexing settings to filtering
            mapping = {
                "use_gitignore": "use_gitignore",
                "additional_ignore_patterns": "ignore_directories",
            }

            for old_key, new_key in mapping.items():
                if old_key in indexing_config:
                    services["filtering"][new_key] = indexing_config[old_key]

            # Ensure provider is set
            if "provider" not in services["filtering"]:
                services["filtering"]["provider"] = "fastmcp_filtering"

        return migrated

def get_config(config_path: Path | None = None) -> CodeWeaverConfig:
    """Get configuration with migration support."""

    # Find and load configuration
    config_data = {}

    if config_path:
        config_data = _load_toml_config(config_path)
    else:
        # Search for configuration files in standard locations
        for search_path in _get_config_search_paths():
            if search_path.exists():
                config_data = _load_toml_config(search_path)
                break

    # Apply configuration migration
    config_data = ConfigMigration.migrate_server_config_to_services(config_data)

    # Create and validate configuration
    try:
        # Set the TOML file for pydantic-settings
        config = CodeWeaverConfig(_env_file=None, **config_data)
        return config
    except ValidationError as e:
        logger.error("Configuration validation failed: %s", e)
        # Return default configuration with warning
        logger.warning("Using default configuration due to validation errors")
        return CodeWeaverConfig()
```

### 5. Configuration Validation

Add comprehensive validation for services configuration:

```python
class ServicesConfigValidator:
    """Validator for services configuration."""

    @staticmethod
    def validate_middleware_dependencies(config: ServicesConfig) -> list[str]:
        """Validate middleware service dependencies and return warnings."""
        warnings = []

        # Check if logging is disabled but other services depend on it
        if not config.logging.enabled:
            if config.timing.enabled and config.timing.log_level:
                warnings.append("Timing service logging may not work with logging service disabled")

        # Check rate limiting configuration
        if config.rate_limiting.enabled:
            if config.rate_limiting.max_requests_per_second <= 0:
                warnings.append("Rate limiting max_requests_per_second must be positive")

            if config.rate_limiting.burst_capacity <= 0:
                warnings.append("Rate limiting burst_capacity must be positive")

        # Check timing configuration
        if config.timing.enabled:
            if config.timing.metric_aggregation_window <= 0:
                warnings.append("Timing metric_aggregation_window must be positive")

        # Check error handling configuration
        if config.error_handling.enabled:
            if config.error_handling.max_error_history <= 0:
                warnings.append("Error handling max_error_history must be positive")

        return warnings

    @staticmethod
    def validate_service_provider_availability(config: ServicesConfig) -> list[str]:
        """Validate that required service providers are available."""
        errors = []

        # Check core services
        if config.chunking.enabled and config.chunking.provider == "fastmcp_chunking":
            # Validate that FastMCP chunking is available
            pass  # Implementation depends on provider registry

        if config.filtering.enabled and config.filtering.provider == "fastmcp_filtering":
            # Validate that FastMCP filtering is available
            pass  # Implementation depends on provider registry

        return errors

# Add to CodeWeaverConfig class
@model_validator(mode="after")
def validate_complete_config(self) -> "CodeWeaverConfig":
    """Validate the complete configuration."""

    # Validate services configuration
    warnings = ServicesConfigValidator.validate_middleware_dependencies(self.services)
    for warning in warnings:
        logger.warning("Services configuration warning: %s", warning)

    errors = ServicesConfigValidator.validate_service_provider_availability(self.services)
    if errors:
        raise ValueError(f"Services configuration errors: {'; '.join(errors)}")

    return self
```

### 6. Configuration Documentation

**File**: `docs/configuration.md` (conceptual - would be created)

Add comprehensive documentation for services configuration:

```markdown
# Services Configuration

CodeWeaver uses a hierarchical services architecture that includes middleware services. All services can be configured through TOML configuration files or environment variables.

## Core Services

### Chunking Service
Controls how code is segmented for embedding and search.

```toml
[services.chunking]
enabled = true
provider = "fastmcp_chunking"
max_chunk_size = 1500
min_chunk_size = 50
ast_grep_enabled = true
```

### Filtering Service
Controls file discovery and filtering during indexing.

```toml
[services.filtering]
enabled = true
provider = "fastmcp_filtering"
use_gitignore = true
max_file_size = 1048576
```

## Middleware Services

### Logging Service
Provides request/response logging and internal service logging.

```toml
[services.logging]
enabled = true
log_level = "INFO"
include_payloads = false
structured_logging = true
```

### Timing Service
Provides performance monitoring and timing metrics.

```toml
[services.timing]
enabled = true
track_performance_metrics = true
metric_aggregation_window = 300
```

### Error Handling Service
Provides error aggregation and notification capabilities.

```toml
[services.error_handling]
enabled = true
error_aggregation = true
max_error_history = 100
```

### Rate Limiting Service
Provides request rate limiting and metrics.

```toml
[services.rate_limiting]
enabled = true
max_requests_per_second = 1.0
burst_capacity = 10
```

## Environment Variables

All configuration can be overridden with environment variables using the pattern:
`CW_SERVICES__{SERVICE}__{SETTING}=value`

Examples:
- `CW_SERVICES__LOGGING__LOG_LEVEL=DEBUG`
- `CW_SERVICES__RATE_LIMITING__MAX_REQUESTS_PER_SECOND=2.0`
- `CW_SERVICES__TIMING__TRACK_PERFORMANCE_METRICS=true`
```

### 7. Configuration Testing

**File**: `tests/unit/test_services_config.py` (conceptual)

```python
"""Tests for services configuration."""

import pytest
from pydantic import ValidationError

from codeweaver.cw_types import ServicesConfig, LoggingServiceConfig
from codeweaver.config import CodeWeaverConfig, ConfigMigration


class TestServicesConfig:
    """Test services configuration validation."""

    def test_default_services_config(self):
        """Test default services configuration is valid."""
        config = ServicesConfig()
        assert config.chunking.enabled
        assert config.filtering.enabled
        assert config.logging.enabled
        assert config.timing.enabled

    def test_middleware_initialization_order_validation(self):
        """Test middleware initialization order validation."""
        config = ServicesConfig(
            middleware_initialization_order=["invalid_service"]
        )
        # Should raise validation error
        with pytest.raises(ValidationError):
            CodeWeaverConfig(services=config)

    def test_logging_service_config_validation(self):
        """Test logging service configuration validation."""
        # Valid config
        config = LoggingServiceConfig(
            log_level="DEBUG",
            max_payload_length=500
        )
        assert config.log_level == "DEBUG"

        # Invalid log level should be handled gracefully
        config = LoggingServiceConfig(log_level="INVALID")
        # Validation should handle this appropriately

class TestConfigMigration:
    """Test configuration migration functionality."""

    def test_migrate_server_logging_to_services(self):
        """Test migration of server logging config to services."""
        old_config = {
            "server": {
                "log_level": "DEBUG",
                "enable_request_logging": True
            }
        }

        migrated = ConfigMigration.migrate_server_config_to_services(old_config)

        assert "services" in migrated
        assert "logging" in migrated["services"]
        assert migrated["services"]["logging"]["log_level"] == "DEBUG"
        assert migrated["services"]["logging"]["include_payloads"] is True

    def test_migrate_chunking_config_to_services(self):
        """Test migration of chunking config to services."""
        old_config = {
            "chunking": {
                "max_chunk_size": 2000,
                "min_chunk_size": 100
            }
        }

        migrated = ConfigMigration.migrate_server_config_to_services(old_config)

        assert "services" in migrated
        assert "chunking" in migrated["services"]
        assert migrated["services"]["chunking"]["max_chunk_size"] == 2000
        assert migrated["services"]["chunking"]["provider"] == "fastmcp_chunking"
```

## Implementation Priority

### Phase 1: Core Integration
1. Add `ServicesConfig` to main configuration
2. Implement basic TOML configuration support
3. Add environment variable mapping
4. Implement configuration migration

### Phase 2: Enhanced Features
1. Add comprehensive validation
2. Implement configuration documentation
3. Add configuration testing
4. Create configuration examples

### Phase 3: Advanced Features
1. Dynamic configuration reloading
2. Configuration validation CLI tools
3. Configuration schema generation
4. Advanced migration scenarios

## Benefits

1. **Unified Configuration**: All services configured through single system
2. **Environment Flexibility**: Full environment variable support
3. **Migration Support**: Smooth upgrade path from existing configurations
4. **Validation**: Comprehensive validation with helpful error messages
5. **Documentation**: Clear documentation for all configuration options
6. **Testing**: Thorough testing of configuration scenarios

## Backward Compatibility

- Existing configurations continue to work with automatic migration
- Environment variables maintain existing patterns where possible
- Default values preserve existing behavior
- Migration warnings help users understand changes
