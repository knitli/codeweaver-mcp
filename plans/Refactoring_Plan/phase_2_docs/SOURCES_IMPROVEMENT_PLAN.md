<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Sources Module Improvement Plan

## Overview
Based on analysis of the sources modules and comparison with the backend improvement plan, I've identified significant improvements needed to address hardcoded attributes, type system issues, backwards compatibility code, and manual serialization.

## Phase 1: Type System Consolidation

### 1.1 Create Centralized Enums
**File**: `src/codeweaver/_types/source_enums.py`

Replace all string literals with proper enums:

```python
class SourceProvider(Enum):
    """All supported source providers."""
    FILESYSTEM = "filesystem"
    GIT = "git"
    DATABASE = "database"
    API = "api"
    WEB = "web"

class ContentType(Enum):
    """Content types for discovered items."""
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    API = "api"
    GIT = "git"

class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"

class APIType(Enum):
    """Supported API types."""
    REST = "rest"
    GRAPHQL = "graphql"
    OPENAPI = "openapi"
    SWAGGER = "swagger"

class AuthType(Enum):
    """Authentication types."""
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
```

### 1.2 Create SourceCapabilities Model
**File**: `src/codeweaver/_types/source_capabilities.py`

```python
from pydantic import BaseModel, Field
from typing import Annotated

class SourceCapabilities(BaseModel):
    """Centralized source capability definitions."""
    supports_content_discovery: bool = Field(True, description="Can discover content items")
    supports_content_reading: bool = Field(True, description="Can read content from items")
    supports_change_watching: bool = Field(False, description="Can watch for content changes")
    supports_incremental_sync: bool = Field(False, description="Supports incremental updates")
    supports_version_history: bool = Field(False, description="Provides version/commit history")
    supports_metadata_extraction: bool = Field(False, description="Rich metadata extraction")
    supports_real_time_updates: bool = Field(False, description="Real-time change notifications")
    supports_batch_processing: bool = Field(False, description="Efficient batch operations")
    supports_content_deduplication: bool = Field(False, description="Built-in deduplication")
    supports_rate_limiting: bool = Field(False, description="Built-in rate limiting")
    supports_authentication: bool = Field(False, description="Supports authentication")
    supports_pagination: bool = Field(False, description="Supports paginated discovery")

    # Performance characteristics
    max_content_size_mb: Annotated[int | None, Field(None, ge=1, description="Maximum supported content size")]
    max_concurrent_requests: Annotated[int, Field(10, ge=1, le=100, description="Maximum concurrent requests")]
    default_batch_size: Annotated[int, Field(8, ge=1, le=1000, description="Default batch processing size")]

    # Dependencies
    required_dependencies: list[str] = Field(default_factory=list, description="Required Python packages")
    optional_dependencies: list[str] = Field(default_factory=list, description="Optional Python packages")
```

### 1.3 Provider Registry Enhancement
**File**: `src/codeweaver/_types/source_providers.py`

Following backend pattern:

```python
from dataclasses import dataclass
from typing import TypeAlias
from codeweaver.sources.base import DataSource

@dataclass
class SourceProviderInfo:
    """Information about a source provider."""
    source_class: type[DataSource]
    capabilities: SourceCapabilities
    provider_type: SourceProvider
    display_name: str
    description: str
    implemented: bool = True

# Registry with full capability information
SOURCE_PROVIDERS: dict[SourceProvider, SourceProviderInfo] = {
    SourceProvider.FILESYSTEM: SourceProviderInfo(
        source_class=FileSystemSource,
        capabilities=SourceCapabilities(
            supports_content_discovery=True,
            supports_content_reading=True,
            supports_change_watching=True,
            supports_batch_processing=True,
            supports_content_deduplication=True,
            max_content_size_mb=100,
            max_concurrent_requests=8,
        ),
        provider_type=SourceProvider.FILESYSTEM,
        display_name="File System",
        description="Local file system with gitignore support",
        implemented=True,
    ),
    # ... other providers
}
```

## Phase 2: Provider Attribute Consolidation Strategy

### 2.1 Eliminate Hardcoded Capabilities
**Current Problem**: Each source implementation hardcodes its capabilities in [`get_capabilities()`](src/codeweaver/sources/filesystem.py:224)

**Solution**: Move to declarative capability definition:

```python
class FileSystemSource(AbstractDataSource):
    # Define capabilities as class attribute
    CAPABILITIES = SourceCapabilities(
        supports_content_discovery=True,
        supports_content_reading=True,
        supports_change_watching=True,
        supports_batch_processing=True,
        supports_content_deduplication=True,
        max_content_size_mb=100,
        max_concurrent_requests=8,
        required_dependencies=[],
        optional_dependencies=["watchdog"]
    )

    def get_capabilities(self) -> SourceCapabilities:
        """Get capabilities - no hardcoding needed."""
        return self.CAPABILITIES
```

### 2.2 Capability-Driven Feature Detection
Replace scattered capability checks with centralized logic:

```python
class SourceCapabilityChecker:
    """Centralized capability checking."""

    @staticmethod
    def supports_feature(source: DataSource, capability: SourceCapability) -> bool:
        """Check if source supports a specific capability."""
        caps = source.get_capabilities()
        return getattr(caps, f"supports_{capability.value}", False)

    @staticmethod
    def get_max_concurrent_requests(source: DataSource) -> int:
        """Get max concurrent requests for source."""
        return source.get_capabilities().max_concurrent_requests
```

### 2.3 Custom Adapter Support Enhancement
**Current Issue**: No clear path for custom adapters in registry system

**Solution**: Enhanced registration with validation:

```python
class SourceRegistry:
    def register_custom_source(
        self,
        provider_type: str,
        source_class: type[DataSource],
        capabilities: SourceCapabilities,
        display_name: str,
        description: str
    ) -> None:
        """Register a custom source with full validation."""
        # Validate source implements required methods
        self._validate_source_implementation(source_class)

        # Create custom provider enum value
        custom_provider = SourceProvider(provider_type)  # Allow custom values

        # Register with full metadata
        self._providers[custom_provider] = SourceProviderInfo(
            source_class=source_class,
            capabilities=capabilities,
            provider_type=custom_provider,
            display_name=display_name,
            description=description,
            implemented=True
        )
```

## Phase 3: Pydantic V2 Migration Approach

### 3.1 Convert Core Data Structures
**Target Files**: [`base.py`](src/codeweaver/sources/base.py), [`config.py`](src/codeweaver/sources/config.py)

**Current Problems**:
- [`ContentItem`](src/codeweaver/sources/base.py:46) uses manual `__init__` and dict conversion
- [`DataSourcesConfig`](src/codeweaver/sources/config.py:23) uses `@dataclass`
- All TypedDict configurations should be Pydantic models

**Migration Plan**:

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from datetime import datetime

class ContentItem(BaseModel):
    """Universal content item with Pydantic validation."""
    model_config = ConfigDict(
        extra="allow",  # Allow source-specific metadata
        validate_assignment=True,
        use_enum_values=True
    )

    path: Annotated[str, Field(description="Universal identifier for content")]
    content_type: Annotated[ContentType, Field(description="Type of content")]
    metadata: Annotated[dict[str, Any], Field(default_factory=dict, description="Source-specific metadata")]
    last_modified: Annotated[datetime | None, Field(None, description="Last modification timestamp")]
    size: Annotated[int | None, Field(None, ge=0, description="Content size in bytes")]
    language: Annotated[str | None, Field(None, description="Detected programming language")]
    source_id: Annotated[str | None, Field(None, description="Source identifier")]
    version: Annotated[str | None, Field(None, description="Version identifier")]
    checksum: Annotated[str | None, Field(None, description="Content checksum")]

    @computed_field
    @property
    def id(self) -> str:
        """Generate unique identifier."""
        import hashlib
        parts = [self.path, self.content_type.value]
        if self.source_id:
            parts.append(self.source_id)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
```

### 3.2 Convert Configuration Classes
Replace [`DataSourcesConfig`](src/codeweaver/sources/config.py:23) dataclass:

```python
class SourceConfig(BaseModel):
    """Base configuration for all sources."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: Annotated[bool, Field(True, description="Whether source is enabled")]
    priority: Annotated[int, Field(1, ge=1, le=100, description="Source priority")]
    source_id: Annotated[str | None, Field(None, description="Unique source identifier")]

    # Content filtering
    include_patterns: Annotated[list[str], Field(default_factory=list)]
    exclude_patterns: Annotated[list[str], Field(default_factory=list)]
    max_file_size_mb: Annotated[int, Field(1, ge=1, le=1000)]

    # Performance settings
    batch_size: Annotated[int, Field(8, ge=1, le=1000)]
    max_concurrent_requests: Annotated[int, Field(10, ge=1, le=100)]
    request_timeout_seconds: Annotated[int, Field(30, ge=1, le=300)]

class FileSystemSourceConfig(SourceConfig):
    """File system specific configuration."""
    root_path: Annotated[str, Field(description="Root directory path")]
    use_gitignore: Annotated[bool, Field(True, description="Respect .gitignore files")]
    follow_symlinks: Annotated[bool, Field(False, description="Follow symbolic links")]
    additional_ignore_patterns: Annotated[list[str], Field(default_factory=list)]

    @field_validator('root_path')
    @classmethod
    def validate_root_path(cls, v: str) -> str:
        from pathlib import Path
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Root path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Root path is not a directory: {v}")
        return str(path.resolve())
```

### 3.3 Configuration Serialization Support
Add comprehensive serialization with tomlkit integration:

```python
class DataSourcesConfig(BaseModel):
    """Master configuration for all data sources."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: bool = True
    default_source_type: SourceProvider = SourceProvider.FILESYSTEM
    max_concurrent_sources: Annotated[int, Field(5, ge=1, le=20)]

    sources: list[dict[str, Any]] = Field(default_factory=list)

    def add_source(self, provider: SourceProvider, config: SourceConfig) -> None:
        """Add a source configuration with validation."""
        self.sources.append({
            "type": provider.value,
            "enabled": config.enabled,
            "priority": config.priority,
            "config": config.model_dump(exclude_unset=True)
        })

    def to_toml(self) -> str:
        """Export to TOML with preserved formatting."""
        import tomlkit
        data = self.model_dump(exclude_unset=True)
        return tomlkit.dumps(data)

    @classmethod
    def from_toml(cls, toml_str: str) -> "DataSourcesConfig":
        """Load from TOML with validation."""
        import tomlkit
        data = tomlkit.parse(toml_str)
        return cls.model_validate(data)
```

## Phase 4: Remove Backwards Compatibility Code

### 4.1 Code to Remove - Legacy Migration Functions

**Files to Clean**:

1. **[`config.py:92-127`](src/codeweaver/sources/config.py:92)** - `migrate_from_legacy_config()` method
2. **[`integration.py:223-327`](src/codeweaver/sources/integration.py:223)** - `BackwardCompatibilityAdapter` class
3. **[`integration.py:355-411`](src/codeweaver/sources/integration.py:355)** - `create_backward_compatible_server_integration()` function
4. **[`filesystem.py:56`](src/codeweaver/sources/filesystem.py:56)** - `patterns` field for backward compatibility

**Specific Deletions**:

```python
# DELETE: Legacy migration method
def migrate_from_legacy_config(self, legacy_config: Any) -> None:
    """This entire method should be removed"""

# DELETE: Backward compatibility adapter
class BackwardCompatibilityAdapter:
    """This entire class should be removed"""

# DELETE: Server integration compatibility
def create_backward_compatible_server_integration(server_class: type) -> type:
    """This entire function should be removed"""

# DELETE: Legacy pattern support
class FileSystemSourceConfig(TypedDict, total=False):
    patterns: list[str]  # Remove this field entirely
```

### 4.2 Simplify Factory and Registry
**File**: [`factory.py`](src/codeweaver/sources/factory.py)

**Remove**:
- Lines 120-158: `create_multiple_sources()` method complexity
- Lines 206-235: `_is_source_implemented()` and placeholder detection
- All migration helper logic

**Simplify to**:
```python
class SourceFactory:
    def create_source(self, provider: SourceProvider, config: SourceConfig) -> DataSource:
        """Create source - no legacy support."""
        if provider not in SOURCE_PROVIDERS:
            available = list(SOURCE_PROVIDERS.keys())
            raise ValueError(f"Unsupported provider: {provider}. Available: {available}")

        provider_info = SOURCE_PROVIDERS[provider]
        return provider_info.source_class(config=config)
```

### 4.3 Clean Environment Variable Support
**Remove all legacy environment variable handling**:

```python
# DELETE: These fallback patterns
if not config.get("root_path"):
    config["root_path"] = os.getenv("LEGACY_ROOT_PATH", ".")  # Remove

# KEEP ONLY: Clean new-style variables
SOURCE_PROVIDER = os.getenv("SOURCE_PROVIDER", "filesystem")
SOURCE_ROOT_PATH = os.getenv("SOURCE_ROOT_PATH", ".")
```

## Phase 5: Config System Centralization Approach

### 5.1 Unified Config Module Structure
**Target**: Centralize all configuration handling following backend improvement pattern

**New File Structure**:
```
src/codeweaver/config/
├── __init__.py           # Public API
├── schema.py            # All Pydantic models
├── loader.py            # TOML loading with tomlkit
├── validation.py        # Configuration validation
└── defaults.py          # Default configurations
```

### 5.2 Centralized Configuration Schema
**File**: `src/codeweaver/config/schema.py`

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated

class SourcesConfig(BaseModel):
    """Sources section of main config."""
    model_config = ConfigDict(
        extra="allow",  # Allow future extensions
        validate_assignment=True,
        use_enum_values=True
    )

    enabled: bool = Field(True, description="Enable sources system")
    default_provider: SourceProvider = Field(SourceProvider.FILESYSTEM)
    max_concurrent_sources: Annotated[int, Field(5, ge=1, le=20)]
    enable_content_deduplication: bool = Field(True)
    content_cache_ttl_hours: Annotated[int, Field(24, ge=1, le=168)]

    # Source configurations
    sources: list[SourceConfig] = Field(default_factory=list)

class CodeWeaverConfig(BaseModel):
    """Master configuration model."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        use_enum_values=True
    )

    # All subsystem configs
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    backends: BackendConfig = Field(default_factory=BackendConfig)  # From existing
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)  # From existing
    server: ServerConfig = Field(default_factory=ServerConfig)
```

### 5.3 TOML Handling with Preservation
**File**: `src/codeweaver/config/loader.py`

```python
import tomlkit
from pathlib import Path
from typing import Any

class ConfigLoader:
    """Configuration loader with comment preservation."""

    @staticmethod
    def load_config(config_path: Path | str) -> CodeWeaverConfig:
        """Load config from TOML with validation."""
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.exists():
            return CodeWeaverConfig()  # Use defaults

        with config_path.open('r', encoding='utf-8') as f:
            toml_data = tomlkit.load(f)

        return CodeWeaverConfig.model_validate(toml_data)

    @staticmethod
    def save_config(config: CodeWeaverConfig, config_path: Path | str) -> None:
        """Save config to TOML preserving comments."""
        if isinstance(config_path, str):
            config_path = Path(config_path)

        # Load existing TOML to preserve comments
        existing_toml = tomlkit.document()
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as f:
                existing_toml = tomlkit.load(f)

        # Update with new values while preserving structure
        config_dict = config.model_dump(exclude_unset=True)
        ConfigLoader._merge_preserve_comments(existing_toml, config_dict)

        with config_path.open('w', encoding='utf-8') as f:
            tomlkit.dump(existing_toml, f)

    @staticmethod
    def _merge_preserve_comments(existing: tomlkit.Document, new_data: dict[str, Any]) -> None:
        """Merge new data while preserving existing comments and structure."""
        for key, value in new_data.items():
            if isinstance(value, dict) and key in existing:
                ConfigLoader._merge_preserve_comments(existing[key], value)
            else:
                existing[key] = value
```

### 5.4 Hierarchical Config Loading
**File**: `src/codeweaver/config/__init__.py`

```python
from pathlib import Path
import os

class ConfigManager:
    """Manages hierarchical configuration loading."""

    DEFAULT_CONFIG_PATHS = [
        Path.cwd() / "codeweaver.toml",
        Path.home() / ".codeweaver" / "config.toml",
        Path("/etc/codeweaver/config.toml")  # System-wide
    ]

    @classmethod
    def load_configuration(cls) -> CodeWeaverConfig:
        """Load configuration with hierarchy: defaults -> file -> env -> runtime."""
        # 1. Start with defaults
        config = CodeWeaverConfig()

        # 2. Load from config file (first found)
        for config_path in cls.DEFAULT_CONFIG_PATHS:
            if config_path.exists():
                file_config = ConfigLoader.load_config(config_path)
                config = cls._merge_configs(config, file_config)
                break

        # 3. Override with environment variables
        config = cls._apply_env_overrides(config)

        return config

    @staticmethod
    def _apply_env_overrides(config: CodeWeaverConfig) -> CodeWeaverConfig:
        """Apply environment variable overrides."""
        env_overrides = {}

        # Sources overrides
        if provider := os.getenv("SOURCE_PROVIDER"):
            env_overrides["sources"] = {"default_provider": provider}

        if max_sources := os.getenv("SOURCE_MAX_CONCURRENT"):
            env_overrides.setdefault("sources", {})["max_concurrent_sources"] = int(max_sources)

        if env_overrides:
            override_config = CodeWeaverConfig.model_validate(env_overrides)
            config = cls._merge_configs(config, override_config)

        return config
```

## Phase 6: Interface Flexibility Improvements

### 6.1 Simplify and Consolidate Interfaces
**Current Problem**: Both `DataSource` protocol and `AbstractDataSource` class create redundancy

**Solution**: Single, focused protocol with optional base class

```python
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod

@runtime_checkable
class DataSource(Protocol):
    """Streamlined data source protocol - minimal and focused."""

    def get_capabilities(self) -> SourceCapabilities:
        """Get source capabilities - now returns rich object."""
        ...

    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover content from this source."""
        ...

    async def read_content(self, item: ContentItem) -> str:
        """Read content from an item."""
        ...

    # Optional methods with default implementations
    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate source configuration."""
        return True  # Default implementation

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get enhanced metadata."""
        return {"source_type": getattr(self, 'source_type', 'unknown')}

# Optional base class for convenience
class BaseDataSource(ABC):
    """Optional base class with sensible defaults."""

    def __init__(self, source_type: SourceProvider, source_id: str | None = None):
        self.source_type = source_type
        self.source_id = source_id or f"{source_type.value}_{id(self)}"
        self._watchers: list[SourceWatcher] = []

    @abstractmethod
    def get_capabilities(self) -> SourceCapabilities:
        """Subclasses must define capabilities."""
        pass

    @abstractmethod
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Subclasses must implement discovery."""
        pass

    @abstractmethod
    async def read_content(self, item: ContentItem) -> str:
        """Subclasses must implement reading."""
        pass

    # Default implementations for optional methods
    async def validate_source(self, config: SourceConfig) -> bool:
        """Default validation - override for custom logic."""
        return config.enabled

    async def cleanup(self) -> None:
        """Clean up resources."""
        for watcher in self._watchers:
            await watcher.stop()
        self._watchers.clear()
```

### 6.2 Enhanced Extension Points
**Add capability-driven extensibility**:

```python
class ExtensibleSourceCapabilities(SourceCapabilities):
    """Extended capabilities that support custom features."""

    # Custom capability registry
    custom_capabilities: dict[str, Any] = Field(default_factory=dict)

    def add_custom_capability(self, name: str, value: Any, description: str = "") -> None:
        """Add a custom capability."""
        self.custom_capabilities[name] = {
            "value": value,
            "description": description,
            "added_at": datetime.now(UTC).isoformat()
        }

    def supports_custom(self, capability_name: str) -> bool:
        """Check if custom capability is supported."""
        return capability_name in self.custom_capabilities

class ExtensibleContentItem(ContentItem):
    """Content item with extensible metadata."""

    # Plugin-specific metadata namespace
    plugin_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def add_plugin_metadata(self, plugin_name: str, metadata: dict[str, Any]) -> None:
        """Add plugin-specific metadata."""
        self.plugin_metadata[plugin_name] = metadata

    def get_plugin_metadata(self, plugin_name: str) -> dict[str, Any]:
        """Get plugin-specific metadata."""
        return self.plugin_metadata.get(plugin_name, {})
```

### 6.3 Comprehensive Interface Documentation
**Enhanced docstrings with examples**:

```python
@runtime_checkable
class DataSource(Protocol):
    """Universal data source protocol for content discovery and processing.

    This protocol defines the interface that all data sources must implement
    to provide universal content discovery, reading, and change watching.

    Examples:
        Basic implementation:

        ```python
        class MyCustomSource:
            def get_capabilities(self) -> SourceCapabilities:
                return SourceCapabilities(
                    supports_content_discovery=True,
                    supports_content_reading=True,
                    max_concurrent_requests=5
                )

            async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
                # Discover content from your source
                return [ContentItem(path="example", content_type=ContentType.FILE)]

            async def read_content(self, item: ContentItem) -> str:
                # Read content from item
                return "content text"
        ```

        Registration:

        ```python
        from codeweaver.sources import get_source_registry

        registry = get_source_registry()
        registry.register_custom_source(
            provider_type="my_custom",
            source_class=MyCustomSource,
            capabilities=MyCustomSource().get_capabilities(),
            display_name="My Custom Source",
            description="Custom source for special content"
        )
        ```

    Interface Design Principles:
        - Minimal required methods for maximum flexibility
        - Rich capability system for feature discovery
        - Async-first design for performance
        - Extensible metadata for plugin systems
        - Clear error handling patterns
    """

    def get_capabilities(self) -> SourceCapabilities:
        """Get the capabilities supported by this data source.

        Returns rich capability information including performance characteristics,
        supported features, and dependency requirements.

        Returns:
            SourceCapabilities object with detailed capability information

        Examples:
            ```python
            def get_capabilities(self) -> SourceCapabilities:
                return SourceCapabilities(
                    supports_content_discovery=True,
                    supports_content_reading=True,
                    supports_change_watching=True,
                    max_content_size_mb=100,
                    max_concurrent_requests=10,
                    required_dependencies=["requests"],
                    optional_dependencies=["asyncio"]
                )
            ```
        """
        ...
```

## Phase 7: Custom Adapter Support Enhancements

### 7.1 Enhanced Registry with Validation
**Current Issue**: [`SourceRegistry`](src/codeweaver/sources/base.py:420) lacks comprehensive custom adapter support

**Solution**: Robust custom adapter registration with validation

```python
class EnhancedSourceRegistry:
    """Enhanced registry with comprehensive custom adapter support."""

    def __init__(self):
        self._providers: dict[SourceProvider, SourceProviderInfo] = {}
        self._custom_providers: dict[str, SourceProviderInfo] = {}
        self._validation_rules: list[Callable[[type[DataSource]], bool]] = []

    def register_custom_source(
        self,
        provider_name: str,
        source_class: type[DataSource],
        capabilities: SourceCapabilities,
        display_name: str,
        description: str,
        *,
        validate_implementation: bool = True
    ) -> None:
        """Register a custom source with comprehensive validation.

        Args:
            provider_name: Unique identifier for the custom provider
            source_class: Implementation class
            capabilities: Declared capabilities
            display_name: Human-readable name
            description: Detailed description
            validate_implementation: Whether to validate the implementation

        Raises:
            ValueError: If provider name conflicts or validation fails
            TypeError: If source_class doesn't implement DataSource protocol
        """
        # Check for name conflicts
        if provider_name in [p.value for p in SourceProvider]:
            raise ValueError(f"Provider name '{provider_name}' conflicts with built-in provider")

        if provider_name in self._custom_providers:
            raise ValueError(f"Custom provider '{provider_name}' already registered")

        # Validate implementation
        if validate_implementation:
            self._validate_source_implementation(source_class, capabilities)

        # Create provider info
        provider_info = SourceProviderInfo(
            source_class=source_class,
            capabilities=capabilities,
            provider_type=provider_name,  # Custom string instead of enum
            display_name=display_name,
            description=description,
            implemented=True
        )

        self._custom_providers[provider_name] = provider_info
        logger.info("Registered custom source provider: %s", provider_name)

    def _validate_source_implementation(
        self,
        source_class: type[DataSource],
        declared_capabilities: SourceCapabilities
    ) -> None:
        """Validate that source implementation matches declared capabilities."""

        # Check protocol compliance
        if not isinstance(source_class, type) or not hasattr(source_class, '__mro__'):
            raise TypeError("source_class must be a class")

        # Check required methods exist
        required_methods = ['get_capabilities', 'discover_content', 'read_content']
        for method in required_methods:
            if not hasattr(source_class, method):
                raise TypeError(f"Source class missing required method: {method}")

        # Validate capabilities match implementation
        try:
            # Create temporary instance for validation
            temp_instance = source_class()
            actual_capabilities = temp_instance.get_capabilities()

            # Check capability consistency
            if actual_capabilities != declared_capabilities:
                logger.warning(
                    "Declared capabilities don't match implementation for %s",
                    source_class.__name__
                )

        except Exception as e:
            raise ValueError(f"Failed to validate source implementation: {e}") from e

        # Run custom validation rules
        for rule in self._validation_rules:
            if not rule(source_class):
                raise ValueError("Custom validation rule failed")

    def add_validation_rule(self, rule: Callable[[type[DataSource]], bool]) -> None:
        """Add a custom validation rule for source registration."""
        self._validation_rules.append(rule)

    def get_all_providers(self) -> dict[str, SourceProviderInfo]:
        """Get all providers (built-in and custom)."""
        all_providers = {}

        # Add built-in providers
        for provider in self._providers:
            all_providers[provider.value] = self._providers[provider]

        # Add custom providers
        all_providers.update(self._custom_providers)

        return all_providers
```

### 7.2 Custom Source Development Kit
**File**: `src/codeweaver/sources/sdk.py`

```python
class SourceSDK:
    """Software Development Kit for creating custom sources."""

    @staticmethod
    def create_source_template(
        provider_name: str,
        base_capabilities: set[SourceCapability] | None = None
    ) -> str:
        """Generate a source implementation template."""

        capabilities = base_capabilities or {SourceCapability.CONTENT_DISCOVERY, SourceCapability.CONTENT_READING}

        template = f'''
"""
Custom source implementation for {provider_name}.

Generated by CodeWeaver Source SDK.
"""

from codeweaver.sources.base import BaseDataSource, ContentItem, SourceCapabilities
from codeweaver.sources.types import SourceProvider, ContentType
from typing import Any

class {provider_name.title()}Source(BaseDataSource):
    """Custom source implementation for {provider_name}."""

    # Define capabilities as class attribute
    CAPABILITIES = SourceCapabilities(
        supports_content_discovery={SourceCapability.CONTENT_DISCOVERY in capabilities},
        supports_content_reading={SourceCapability.CONTENT_READING in capabilities},
        supports_change_watching={SourceCapability.CHANGE_WATCHING in capabilities},
        # Add more capabilities as needed
        max_concurrent_requests=10,
        required_dependencies=[],  # Add required packages
        optional_dependencies=[],  # Add optional packages
    )

    def __init__(self, source_id: str | None = None):
        super().__init__(SourceProvider("{provider_name}"), source_id)

    def get_capabilities(self) -> SourceCapabilities:
        return self.CAPABILITIES

    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Implement content discovery logic."""
        if not config.enabled:
            return []

        # TODO: Implement your content discovery logic here
        # Example:
        discovered_items = []

        # Your discovery logic here
        # For each discovered item:
        # item = ContentItem(
        #     path="path/to/content",
        #     content_type=ContentType.FILE,  # or appropriate type
        #     source_id=self.source_id,
        #     # ... other fields
        # )
        # discovered_items.append(item)

        return discovered_items

    async def read_content(self, item: ContentItem) -> str:
        """Implement content reading logic."""
        # TODO: Implement your content reading logic here
        # Return the text content of the item
        return "Content text here"

    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate source configuration."""
        # TODO: Add custom validation logic
        # Check connectivity, permissions, etc.
        return await super().validate_source(config)

# Registration example:
# def register_source():
#     from codeweaver.sources import get_source_registry
#
#     registry = get_source_registry()
#     registry.register_custom_source(
#         provider_name="{provider_name}",
#         source_class={provider_name.title()}Source,
#         capabilities={provider_name.title()}Source.CAPABILITIES,
#         display_name="{provider_name.title()} Source",
#         description="Custom source for {provider_name}"
#     )
'''
        return template

    @staticmethod
    def validate_source_before_registration(source_class: type[DataSource]) -> list[str]:
        """Validate a source implementation and return issues."""
        issues = []

        # Check required methods
        required_methods = ['get_capabilities', 'discover_content', 'read_content']
        for method in required_methods:
            if not hasattr(source_class, method):
                issues.append(f"Missing required method: {method}")

        # Check capabilities consistency
        try:
            instance = source_class()
            capabilities = instance.get_capabilities()

            if not isinstance(capabilities, SourceCapabilities):
                issues.append("get_capabilities() must return SourceCapabilities instance")

        except Exception as e:
            issues.append(f"Failed to instantiate source: {e}")

        return issues
```

### 7.3 Plugin-Style Registration
**File**: `src/codeweaver/sources/plugins.py`

```python
class SourcePluginManager:
    """Manages source plugins with automatic discovery."""

    def __init__(self):
        self.registry = get_source_registry()
        self._loaded_plugins: set[str] = set()

    def discover_plugins(self, plugin_dirs: list[Path] | None = None) -> None:
        """Automatically discover and load source plugins."""
        plugin_dirs = plugin_dirs or [
            Path.cwd() / "codeweaver_plugins",
            Path.home() / ".codeweaver" / "plugins",
        ]

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists():
                continue

            for plugin_file in plugin_dir.glob("*_source.py"):
                self._load_plugin_file(plugin_file)

    def _load_plugin_file(self, plugin_file: Path) -> None:
        """Load a single plugin file."""
        try:
            import importlib.util
            import sys

            module_name = plugin_file.stem
            if module_name in self._loaded_plugins:
                return

            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Look for registration function
                if hasattr(module, 'register_source'):
                    module.register_source()
                    self._loaded_plugins.add(module_name)
                    logger.info("Loaded source plugin: %s", module_name)

        except Exception:
            logger.exception("Failed to load plugin: %s", plugin_file)
```

# Sources Module Improvement Plan - FINAL DOCUMENT

## Executive Summary

Based on comprehensive analysis of the sources modules in `src/codeweaver/sources/*`, I've identified significant improvements needed to address your key observations: attribute consolidation, type system modernization, interface flexibility, removal of backwards compatibility code, and adoption of Pydantic v2. This plan provides a detailed roadmap consistent with the backend improvements approach.

## Implementation Phases & Timeline

### **Phase 1: Type System Consolidation (Week 1)**

#### 1.1 Create Centralized Enums
**Files**: `src/codeweaver/_types/source_enums.py`

**Changes**:
- Replace string literals with proper enums (`SourceProvider`, `ContentType`, `DatabaseType`, `APIType`, `AuthType`)
- Follow backend pattern for enum-based type definitions
- Consolidate all type attributes with their enums

#### 1.2 Source Capabilities Model
**Files**: `src/codeweaver/_types/source_capabilities.py`

**Changes**:
- Create Pydantic model for centralized capability definitions
- Include performance characteristics and dependency requirements
- Replace hardcoded capability sets across implementations

#### 1.3 Provider Registry Enhancement
**Files**: `src/codeweaver/_types/source_providers.py`

**Changes**:
- Create `SourceProviderInfo` dataclass following backend pattern
- Registry with full capability information and validation
- Support for custom provider registration

### **Phase 2: Pydantic V2 Migration (Week 2)**

#### 2.1 Convert Core Data Structures
**Target Files**: [`base.py`](src/codeweaver/sources/base.py), [`config.py`](src/codeweaver/sources/config.py)

**Changes**:
- Convert [`ContentItem`](src/codeweaver/sources/base.py:46) from manual class to Pydantic model
- Replace [`DataSourcesConfig`](src/codeweaver/sources/config.py:23) dataclass with Pydantic model
- Convert all TypedDict configurations to Pydantic models with validation

#### 2.2 Configuration Serialization
**New Files**: `src/codeweaver/config/loader.py`

**Changes**:
- Implement TOML handling with `tomlkit` for comment preservation
- Add model serialization methods (`model_dump()`, `model_dump_json()`, `model_dump_toml()`)
- Support partial updates with `exclude_unset`

### **Phase 3: Remove Backwards Compatibility (Week 3)**

#### 3.1 Delete Legacy Code
**Files to Clean**:
- **DELETE**: [`config.py:92-127`](src/codeweaver/sources/config.py:92) - `migrate_from_legacy_config()`
- **DELETE**: [`integration.py:223-327`](src/codeweaver/sources/integration.py:223) - `BackwardCompatibilityAdapter`
- **DELETE**: [`integration.py:355-411`](src/codeweaver/sources/integration.py:355) - Server integration compatibility
- **DELETE**: [`filesystem.py:56`](src/codeweaver/sources/filesystem.py:56) - Legacy `patterns` field

#### 3.2 Simplify Factory System
**Files**: [`factory.py`](src/codeweaver/sources/factory.py)

**Changes**:
- Remove migration helper functions and legacy support
- Streamline source creation to use only new-style configuration
- Clean environment variable handling (remove fallbacks)

### **Phase 4: Centralize Configuration (Week 4)**

#### 4.1 Unified Config Structure
**New Files**:
- `src/codeweaver/config/schema.py` - All Pydantic models
- `src/codeweaver/config/loader.py` - TOML loading with tomlkit
- `src/codeweaver/config/validation.py` - Configuration validation

#### 4.2 Hierarchical Config Loading
**Implementation**:
- Support config loading hierarchy: defaults → file → environment → runtime
- Centralized config management following backend improvement pattern
- TOML comment and formatting preservation

## File Changes Summary

### **New Files**
```
src/codeweaver/_types/source_enums.py
src/codeweaver/_types/source_capabilities.py
src/codeweaver/_types/source_providers.py
src/codeweaver/config/schema.py
src/codeweaver/config/loader.py
src/codeweaver/config/validation.py
src/codeweaver/sources/sdk.py
src/codeweaver/sources/plugins.py
```

### **Modified Files**
```
src/codeweaver/sources/base.py (protocol simplification)
src/codeweaver/sources/factory.py (remove legacy code)
src/codeweaver/sources/config.py (remove migration code)
src/codeweaver/sources/filesystem.py (use new types)
src/codeweaver/sources/__init__.py (update exports)
```

### **Deleted Files**
```
- Integration compatibility shims
- Legacy migration helpers
- Backwards compatibility adapters
```

## Key Improvements Delivered

### ✅ **1. Attributes for Providers - SOLVED**
- **Before**: Hardcoded capabilities scattered across implementations
- **After**: Centralized `SourceCapabilities` model with single source of truth
- **Custom Adapters**: Enhanced registry with validation and plugin support

### ✅ **2. Types - MODERNIZED**
- **Before**: String literals and TypedDict usage
- **After**: Proper enums with type attributes (`SourceProvider.capabilities`)
- **Consistency**: Type attributes consolidated with their enums

### ✅ **3. Interface Flexibility - ENHANCED**
- **Before**: Multiple redundant interfaces
- **After**: Single focused `DataSource` protocol with optional base class
- **Extensibility**: Plugin system and custom capability support

### ✅ **4. Backwards Compatibility - REMOVED**
- **Before**: Extensive migration and compatibility code
- **After**: Clean, simple APIs without legacy support
- **Impact**: Reduced complexity, easier maintenance

### ✅ **5. Serialization - PYDANTIC V2**
- **Before**: Manual dict conversion and dataclasses
- **After**: Pydantic models with validation and TOML support
- **Config**: Centralized config system with `tomlkit` preservation

## Dependencies to Add

```toml
[project.dependencies]
pydantic = "^2.5.0"
tomlkit = "^0.12.0"
```

## Success Criteria

1. **✅ Single Source of Truth**: All source capabilities defined in one place
2. **✅ Type Safety**: All string literals replaced with enums
3. **✅ Clean API**: No backwards compatibility code
4. **✅ Pydantic Everywhere**: All data models use Pydantic v2
5. **✅ Centralized Config**: One config system with tomlkit
6. **✅ Extensibility**: Easy custom source registration and plugins
7. **✅ Documentation**: All interfaces comprehensively documented
8. **✅ Consistency**: Approach aligned with backend improvements

## Risk Mitigation

1. **Breaking Changes**: All changes are internal to unreleased tool
2. **Performance**: Pydantic v2 provides better performance than manual approaches
3. **Complexity**: Interfaces simplified, complexity reduced
4. **Custom Sources**: Enhanced plugin system makes custom sources easier
5. **Dependencies**: Minimal new dependencies, both are industry standard

This improvement plan addresses all your observations and provides a clear implementation path that modernizes the sources system while maintaining consistency with the backend improvements approach.
