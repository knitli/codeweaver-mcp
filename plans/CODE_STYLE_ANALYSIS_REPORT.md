<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Code Style & Pattern Analysis Report

**Analysis Date:** December 28, 2024
**Scope:** System-wide coding style, patterns, and architectural alignment
**Focus:** Alignment with providers/* "gold standard" patterns

## Executive Summary

After comprehensive analysis of the CodeWeaver codebase focusing on coding style, patterns, and naming conventions, several key findings emerged:

- **Providers module** represents the established "gold standard" with excellent consistency
- **Sources, backends, and factories** modules have significant pattern deviations requiring alignment
- **Two critical anti-patterns identified**: direct middleware dependencies and legacy migration code
- **Services layer** has strong architecture but limited integration across plugin implementations

## Providers Module: "Gold Standard" Patterns

### Established Standards ✅

#### 1. File Structure & Organization
```
providers/
├── __init__.py          # Clean public API with explicit __all__
├── base.py             # Protocol definitions and abstract base classes
├── config.py           # Pydantic configuration models
├── factory.py          # Registry and factory pattern implementation
└── {provider_name}.py  # Individual provider implementations
```

#### 2. Class Naming Convention
- **Providers**: `{ServiceName}Provider` (e.g., `VoyageAIProvider`, `OpenAIProvider`)
- **Configs**: `{ServiceName}Config` (e.g., `VoyageConfig`, `OpenAIConfig`)
- **Base classes**: `{Type}ProviderBase` (e.g., `EmbeddingProviderBase`)
- **Protocols**: `{Type}Provider` (e.g., `EmbeddingProvider`, `RerankProvider`)

#### 3. Import Organization Pattern
```python
# 1. Standard library imports first
import asyncio
import logging
from typing import Any

# 2. Third-party imports
from pydantic import BaseModel, Field

# 3. Internal imports - organized by layer
from codeweaver.types import (  # Types first
    EmbeddingProviderInfo,
    ProviderCapability,
    ProviderType,
)
from codeweaver.providers.base import CombinedProvider  # Base classes
from codeweaver.providers.config import VoyageConfig   # Config classes
```

#### 4. Configuration Pattern
```python
class ProviderConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",              # Allow provider-specific extensions
        validate_assignment=True,   # Validate on attribute assignment
        frozen=False,              # Allow mutation for runtime updates
    )

    # Multi-stage config validation
    def __init__(self, config: VoyageConfig | dict[str, Any]):
        if isinstance(config, dict):
            self._config = VoyageConfig(**config)
        else:
            self._config = config
```

#### 5. Error Handling Standard
```python
async def embed_documents(self, texts: list[str]) -> list[list[float]]:
    try:
        result = self.client.embed(texts=texts, model=self._model)
    except Exception:
        logger.exception("Error generating VoyageAI embeddings")
        raise
    else:
        return result.embeddings
```

#### 6. Property & Method Patterns
```python
@property
def provider_name(self) -> str:
    return ProviderType.VOYAGE_AI.value

@classmethod
def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
    if not VOYAGEAI_AVAILABLE:
        return False, "voyageai package not installed"
    return True, None
```

## Module Alignment Analysis

### Sources Module Analysis

#### Major Deviations ❌

**1. Naming Convention Inconsistencies**
- **Current**: `FileSystemSource`, `APISource` (should be `FileSystemSourceProvider`, `APISourceProvider`)
- **Config naming**: `FileSystemSourceConfig` (should be `FileSystemConfig`)

**2. Missing Patterns**
- **No `check_availability` classmethod** pattern
- **No dependency availability checks** (missing feature flag pattern)
- **Inconsistent property patterns** (uses `get_capabilities()` method instead of `capabilities` property)

**3. Configuration Issues**
- **Config duplication**: `APISourceConfig` duplicates base config fields instead of inheriting
- **Missing multi-stage validation** pattern

#### Specific Examples
```python
# CURRENT (Incorrect)
class FileSystemSource(AbstractDataSource):
    def get_capabilities(self) -> SourceCapabilities:
        return self.CAPABILITIES

# SHOULD BE (Aligned with providers)
class FileSystemSourceProvider(AbstractDataSource):
    @property
    def source_name(self) -> str:
        return SourceType.FILESYSTEM.value

    @property
    def capabilities(self) -> SourceCapabilities:
        return self._capabilities

    @classmethod
    def check_availability(cls, capability: SourceCapability) -> tuple[bool, str | None]:
        return True, None
```

### Backends Module Analysis

#### Moderate Deviations ⚠️

**1. Naming & Configuration**
- **Classes**: `QdrantBackend` (acceptable) but should consider `QdrantProvider` for consistency
- **Config**: Uses generic `BackendConfig` instead of `QdrantConfig`

**2. Missing Provider Patterns**
- **No static info methods**: Missing `get_static_backend_info()` classmethod
- **No availability checking**: Missing `check_availability()` pattern
- **Inconsistent error handling**: Logs inside try blocks instead of else blocks

**3. Configuration Deviations**
```python
# CURRENT (Inconsistent)
model_config = ConfigDict(
    extra="allow",
    str_strip_whitespace=True,     # ← Extra field not in providers
    validate_assignment=True,
    arbitrary_types_allowed=True,  # ← Extra field not in providers
)

# SHOULD BE (Aligned)
model_config = ConfigDict(
    extra="allow",
    validate_assignment=True,
    frozen=False,
)
```

### Factories Module Analysis

#### Significant Deviations ❌

**1. Structural Issues**
- **File proliferation**: 9+ files vs providers' clean 4-file pattern
- **Missing consistent structure**: No clean `base.py`, `config.py` pattern
- **God object pattern**: `CodeWeaverFactory` has too many responsibilities

**2. Error Handling Inconsistencies**
```python
# CURRENT (Inconsistent)
except Exception:
    logger.exception("Plugin registration failed for %s", plugin_info.name)

# SHOULD BE (Aligned with providers)
except Exception as e:
    logger.exception("Plugin registration failed")
    raise ComponentCreationError(f"Plugin registration failed: {e}") from e
```

**3. Missing Patterns**
- **No standardized availability checking**
- **No consistent property patterns**
- **Mixed validation approaches**

## Naming Convention Standardization Recommendations

### 1. Sources Module Rename Plan
```python
# Current → Recommended
FileSystemSource → FileSystemSourceProvider
APISource → APISourceProvider
GitSource → GitSourceProvider
DatabaseSource → DatabaseSourceProvider
WebSource → WebSourceProvider

# Config classes
FileSystemSourceConfig → FileSystemConfig
APISourceConfig → APIConfig
GitSourceConfig → GitConfig
```

### 2. Backends Module Considerations
```python
# Option 1: Keep current (acceptable)
QdrantBackend (keep as-is)
QdrantConfig (rename from BackendConfig)

# Option 2: Full alignment (recommended)
QdrantBackend → QdrantProvider
QdrantHybridBackend → QdrantHybridProvider
```

### 3. Factories Module Restructure
```python
# Current structure → Recommended structure
backend_registry.py → registry.py (consolidated)
source_registry.py → [merged into registry.py]
service_registry.py → [merged into registry.py]
codeweaver_factory.py → factory.py
```

## Import Organization Guidelines

### Standard Pattern (from providers)
```python
"""Module docstring with clear purpose."""

# Standard library imports
import asyncio
import logging
from typing import Any, ClassVar

# Third-party imports (alphabetical)
from pydantic import BaseModel, Field

# Internal imports (organized by layer)
from codeweaver.types import (
    # Type imports first, alphabetical
    ConfigType,
    ProviderInfo,
    ProviderType,
)
from codeweaver.providers.base import BaseProvider  # Base classes second
from codeweaver.providers.config import Config     # Config classes third
```

## Configuration Patterns

### Pydantic Configuration Standard
```python
class StandardConfig(BaseModel):
    """Standard configuration pattern."""

    model_config = ConfigDict(
        extra="allow",              # Allow extensions
        validate_assignment=True,   # Validate on assignment
        frozen=False,              # Allow mutation
    )

    # Annotated fields with descriptions
    api_key: Annotated[
        str | None,
        Field(
            default=None,
            description="API key for authentication",
        ),
    ]
```

### Multi-Stage Validation Pattern
```python
def __init__(self, config: ProviderConfig | dict[str, Any]):
    """Accept both config types."""
    if isinstance(config, dict):
        self._config = ProviderConfig(**config)
    else:
        self._config = config
    self._validate_config()

def _validate_config(self) -> None:
    """Validate configuration."""
    if not self.config.get("required_field"):
        raise ValueError("Required field missing")
```

## Error Handling Standardization

### Standard Pattern
```python
async def operation(self, param: Any) -> Any:
    """Standard error handling pattern."""
    try:
        result = await self._perform_operation(param)
    except SpecificException:
        logger.warning("Expected error: %s", param)
        raise
    except Exception:
        logger.exception("Unexpected error in operation")
        raise
    else:
        logger.debug("Operation completed successfully")
        return result
```

### Exception Chaining Pattern
```python
try:
    result = external_operation()
except ExternalError as e:
    raise InternalError("Operation failed") from e
```

## Property and Method Patterns

### Required Properties
```python
@property
def provider_name(self) -> str:
    """Get the provider name."""
    return ProviderType.SERVICE_NAME.value

@property
def capabilities(self) -> ProviderCapabilities:
    """Get provider capabilities."""
    return self._capabilities
```

### Required Class Methods
```python
@classmethod
def get_static_provider_info(cls) -> ProviderInfo:
    """Get static provider information."""
    registry_entry = get_provider_registry_entry(ProviderType.SERVICE_NAME)
    return ProviderInfo(name=ProviderType.SERVICE_NAME.value, ...)

@classmethod
def check_availability(cls, capability: Capability) -> tuple[bool, str | None]:
    """Check if provider is available."""
    if not DEPENDENCY_AVAILABLE:
        return False, "dependency package not installed"
    return True, None
```

## Type Annotation Standards

### Modern Python Typing
```python
# Use modern union syntax
def process(self, config: Config | dict[str, Any]) -> Result | None:

# Annotated fields with Pydantic
field: Annotated[str, Field(description="Field description")]

# Runtime-checkable protocols
@runtime_checkable
class ProviderProtocol(Protocol):
    def method(self) -> Any: ...
```

## Documentation Standards

### Module Documentation
```python
"""
Module purpose and integration information.

This module provides [specific functionality] for [use case].
Integrates with [other components] through [interface].
"""
```

### Class Documentation
```python
class ProviderClass:
    """
    Provider for [service] with [capabilities].

    Supports [features] and integrates with [systems].
    Requires [dependencies] for operation.
    """
```

### Method Documentation
```python
def method(self, param: Type) -> Type:
    """
    Brief description of method purpose.

    Args:
        param: Description of parameter purpose and constraints.

    Returns:
        Description of return value and format.

    Raises:
        SpecificError: When specific condition occurs.
    """
```

## Priority Alignment Actions

### Immediate (High Priority)
1. **Rename sources classes** to follow `{Name}SourceProvider` pattern
2. **Add missing classmethods** (`check_availability`, `get_static_info`) to sources and backends
3. **Standardize error handling** across all modules to use try/except/else pattern
4. **Implement property patterns** for `provider_name`/`source_name` and `capabilities`

### Short Term (Medium Priority)
1. **Consolidate factories structure** to match providers clean organization
2. **Standardize configuration patterns** with consistent ConfigDict usage
3. **Align import organization** across all modules
4. **Add missing availability checking** patterns

### Long Term (Low Priority)
1. **Consider backend naming alignment** (`QdrantBackend` → `QdrantProvider`)
2. **Enhance documentation consistency** across all modules
3. **Implement comprehensive type annotation standards**
4. **Add automated style checking** for pattern compliance

## Conclusion

The providers module has established excellent patterns that should be the standard for the entire codebase. Sources and factories modules need significant alignment work, while backends module is moderately aligned but needs refinement. Following these patterns will create a more consistent, maintainable, and professional codebase ready for clean launch.
