# Providers Module Improvement Plan

## Overview

Based on comprehensive analysis of the providers modules in `src/codeweaver/providers/*` and alignment with the backend and sources improvement approaches, I've identified critical improvements needed to address hardcoded attributes, broken type systems, interface redundancy, backwards compatibility code, and manual serialization patterns.

## Critical Issues Identified

### 🚨 **CRITICAL: Broken Type System**
- `ProviderCapability` is defined as `TypedDict` but used as an enum throughout the codebase
- Causes runtime failures when accessing `.EMBEDDING`, `.RERANKING` attributes
- Files affected: `voyage.py`, `openai.py`, `cohere.py`, `factory.py`, etc.

### 🔧 **Major Architecture Issues**
- Redundant Protocol vs ABC interfaces
- Hardcoded metadata scattered across 5+ provider classes
- Manual serialization with no type safety
- Complex inheritance in `CombinedProvider`
- Backwards compatibility shims for unreleased tool

## Phase 1: Fix Critical Type System (Week 1)

### 1.1 Create Correct Provider Capability Enum
**File**: `src/codeweaver/_types/provider_enums.py`

```python
from enum import Enum
from typing import Annotated
from pydantic import BaseModel, Field

class ProviderCapability(Enum):
    """Provider capability types."""
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    BATCH_PROCESSING = "batch_processing"
    RATE_LIMITING = "rate_limiting"
    STREAMING = "streaming"
    CUSTOM_DIMENSIONS = "custom_dimensions"

class ProviderType(Enum):
    """Provider types."""
    VOYAGE = "voyage"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    CUSTOM = "custom"

class ModelFamily(Enum):
    """Model families across providers."""
    CODE_EMBEDDING = "code_embedding"
    TEXT_EMBEDDING = "text_embedding"
    RERANKING = "reranking"
    MULTIMODAL = "multimodal"
```

### 1.2 Create Provider Capabilities Model
**File**: `src/codeweaver/_types/provider_capabilities.py`

```python
from pydantic import BaseModel, Field
from typing import Annotated

class ProviderCapabilities(BaseModel):
    """Centralized provider capability definitions."""

    # Core capabilities
    supports_embedding: bool = Field(False, description="Supports document embedding")
    supports_reranking: bool = Field(False, description="Supports document reranking")
    supports_batch_processing: bool = Field(False, description="Efficient batch operations")
    supports_streaming: bool = Field(False, description="Streaming responses")
    supports_rate_limiting: bool = Field(False, description="Built-in rate limiting")

    # Model capabilities
    supports_custom_dimensions: bool = Field(False, description="Supports custom embedding dimensions")
    supports_multiple_models: bool = Field(False, description="Multiple model support")
    supports_model_switching: bool = Field(False, description="Runtime model switching")

    # Performance characteristics
    max_batch_size: Annotated[int | None, Field(None, ge=1, description="Maximum batch size")]
    max_input_length: Annotated[int | None, Field(None, ge=1, description="Maximum input length")]
    max_concurrent_requests: Annotated[int, Field(10, ge=1, le=100, description="Max concurrent requests")]

    # Rate limiting
    requests_per_minute: Annotated[int | None, Field(None, ge=1, description="Request rate limit")]
    tokens_per_minute: Annotated[int | None, Field(None, ge=1, description="Token rate limit")]

    # Dependencies
    requires_api_key: bool = Field(True, description="Requires API key")
    required_dependencies: list[str] = Field(default_factory=list, description="Required packages")
    optional_dependencies: list[str] = Field(default_factory=list, description="Optional packages")
```

### 1.3 Fix Immediate Runtime Failures
**Emergency fix for all provider files using `ProviderCapability.EMBEDDING`**

Replace broken enum access with temporary constants until full migration:
```python
# Replace ProviderCapability.EMBEDDING with
EMBEDDING_CAPABILITY = "embedding"
RERANKING_CAPABILITY = "reranking"
```

## Phase 2: Consolidate Hardcoded Attributes (Week 2)

### 2.1 Provider Registry with Capabilities
**File**: `src/codeweaver/_types/provider_registry.py`

```python
from dataclasses import dataclass
from typing import TypeAlias
from codeweaver.providers.base import EmbeddingProvider, RerankProvider

@dataclass
class ProviderRegistryEntry:
    """Information about a provider."""
    provider_class: type[EmbeddingProvider | RerankProvider]
    capabilities: ProviderCapabilities
    provider_type: ProviderType
    display_name: str
    description: str
    default_embedding_model: str | None = None
    default_reranking_model: str | None = None
    supported_models: dict[ModelFamily, list[str]] = None
    implemented: bool = True

# Registry with full capability information
PROVIDER_REGISTRY: dict[ProviderType, ProviderRegistryEntry] = {
    ProviderType.VOYAGE: ProviderRegistryEntry(
        provider_class=VoyageAIProvider,
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=True,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            max_batch_size=128,
            max_input_length=32000,
            requires_api_key=True,
            requests_per_minute=100,
            tokens_per_minute=1000000,
        ),
        provider_type=ProviderType.VOYAGE,
        display_name="Voyage AI",
        description="Best-in-class code embeddings and reranking",
        default_embedding_model="voyage-code-3",
        default_reranking_model="voyage-rerank-2",
        supported_models={
            ModelFamily.CODE_EMBEDDING: ["voyage-code-3", "voyage-3"],
            ModelFamily.TEXT_EMBEDDING: ["voyage-3-lite", "voyage-large-2"],
            ModelFamily.RERANKING: ["voyage-rerank-2", "voyage-rerank-lite-1"],
        },
    ),
    # ... other providers
}
```

### 2.2 Eliminate Hardcoded Provider Metadata

**Replace scattered ClassVar definitions across all providers with centralized lookup:**

```python
class VoyageAIProvider(CombinedProvider):
    """VoyageAI provider - metadata now centralized."""

    # Define capabilities as class attribute referencing registry
    CAPABILITIES = PROVIDER_REGISTRY[ProviderType.VOYAGE].capabilities

    @property
    def provider_name(self) -> str:
        """Get provider name from registry."""
        return ProviderType.VOYAGE.value

    def get_provider_info(self) -> ProviderInfo:
        """Get info from centralized registry."""
        registry_entry = PROVIDER_REGISTRY[ProviderType.VOYAGE]
        return ProviderInfo(
            name=self.provider_name,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=registry_entry.capabilities,
            # ... other fields from registry
        )
```

### 2.3 Model Name Enums per Provider

```python
class VoyageModels(Enum):
    """VoyageAI supported models."""
    CODE_3 = "voyage-code-3"
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"
    VOYAGE_LARGE_2 = "voyage-large-2"
    VOYAGE_2 = "voyage-2"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.CODE_3: 1024,
            self.VOYAGE_3: 1024,
            self.VOYAGE_3_LITE: 512,
            self.VOYAGE_LARGE_2: 1536,
            self.VOYAGE_2: 1024,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        if "code" in self.value:
            return ModelFamily.CODE_EMBEDDING
        return ModelFamily.TEXT_EMBEDDING

class VoyageRerankModels(Enum):
    """VoyageAI reranking models."""
    RERANK_2 = "voyage-rerank-2"
    RERANK_LITE_1 = "voyage-rerank-lite-1"
```

## Phase 3: Pydantic V2 Migration (Week 3)

### 3.1 Convert Configuration Models
**File**: `src/codeweaver/providers/config.py`

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Annotated

class ProviderConfig(BaseModel):
    """Base configuration for all providers."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: bool = Field(True, description="Whether provider is enabled")
    api_key: Annotated[str | None, Field(None, description="Provider API key")]
    rate_limiter: Annotated[object | None, Field(None, description="Rate limiter instance")]

    # Performance settings
    batch_size: Annotated[int, Field(8, ge=1, le=1000, description="Batch processing size")]
    timeout_seconds: Annotated[float, Field(30.0, ge=0.1, le=300.0, description="Request timeout")]
    max_retries: Annotated[int, Field(3, ge=0, le=10, description="Maximum retry attempts")]

class EmbeddingProviderConfig(ProviderConfig):
    """Configuration for embedding providers."""
    model: Annotated[str, Field(description="Embedding model name")]
    dimension: Annotated[int | None, Field(None, ge=1, le=4096, description="Custom embedding dimension")]

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        # Validate against supported models for the provider
        return v

class RerankingProviderConfig(ProviderConfig):
    """Configuration for reranking providers."""
    model: Annotated[str, Field(description="Reranking model name")]
    top_k: Annotated[int | None, Field(None, ge=1, le=1000, description="Maximum results to return")]

class CombinedProviderConfig(EmbeddingProviderConfig, RerankingProviderConfig):
    """Configuration for providers supporting both capabilities."""
    rerank_model: Annotated[str | None, Field(None, description="Separate reranking model")]
```

### 3.2 Provider Implementation with Pydantic

```python
class VoyageAIProvider(CombinedProvider):
    """VoyageAI provider with Pydantic configuration."""

    def __init__(self, config: CombinedProviderConfig):
        """Initialize with validated Pydantic config."""
        self.config = config
        self._validate_config()

        # Initialize client with validated config
        self.client = voyageai.Client(api_key=self.config.api_key)

    def _validate_config(self) -> None:
        """Validate provider-specific configuration."""
        if not VOYAGEAI_AVAILABLE:
            raise ImportError("VoyageAI library not available")

        # Validate model is supported
        try:
            VoyageModels(self.config.model)  # Will raise ValueError if invalid
        except ValueError:
            supported = [m.value for m in VoyageModels]
            raise ValueError(f"Unsupported model: {self.config.model}. Supported: {supported}")
```

### 3.3 Factory with Pydantic Integration

```python
class ProviderFactory:
    """Factory with Pydantic-based configuration."""

    def create_embedding_provider(
        self, config: EmbeddingProviderConfig
    ) -> EmbeddingProvider:
        """Create provider with validated configuration."""

        provider_type = ProviderType(config.provider.lower())
        registry_entry = PROVIDER_REGISTRY[provider_type]

        if not registry_entry.capabilities.supports_embedding:
            raise ValueError(f"Provider {provider_type.value} doesn't support embedding")

        # No manual dict conversion needed - pass Pydantic model directly
        return registry_entry.provider_class(config)
```

## Phase 4: Remove Backwards Compatibility (Week 4)

### 4.1 Code to Remove - Legacy Configuration

**Files to Clean**:
1. **Delete**: All `LegacyEmbeddingConfig` dataclasses
2. **Delete**: Manual `getattr()` patterns in factory configuration conversion
3. **Delete**: Model name compatibility mappings (e.g., `"voyage-code-3"` → `"text-embedding-3-small"`)
4. **Delete**: Dual `get_provider_info()` and `get_static_provider_info()` methods

**Specific Deletions**:
```python
# DELETE: Legacy configuration support
@dataclass
class LegacyEmbeddingConfig:
    """Delete this entire class"""

# DELETE: Manual dict conversion in factory
provider_config = {
    "api_key": config.api_key,
    "model": config.model,
    # ... manual field mapping
}
provider_config = {k: v for k, v in provider_config.items() if v is not None}

# DELETE: Model compatibility mappings
if self._model == "voyage-code-3":  # Legacy default
    self._model = "text-embedding-3-small"

# DELETE: Redundant static methods
@classmethod
def get_static_provider_info(cls) -> ProviderInfo:
    """Delete - use registry instead"""
```

### 4.2 Simplify Interface Hierarchy

**Current Problems**:
- Both `EmbeddingProvider` (Protocol) and `EmbeddingProviderBase` (ABC)
- Complex `CombinedProvider` inheritance

**Solution**: Single protocol-based approach
```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Unified embedding provider protocol."""

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for query."""
        ...

# Optional base class for convenience
class BaseEmbeddingProvider:
    """Optional base class with common functionality."""

    def __init__(self, config: EmbeddingProviderConfig):
        self.config = config
        self._capabilities = self._get_capabilities_from_registry()

    def get_capabilities(self) -> ProviderCapabilities:
        """Get capabilities from registry."""
        return self._capabilities
```

## Phase 5: Interface Flexibility Improvements (Week 5)

### 5.1 Unified Provider Protocol

```python
@runtime_checkable
class UniversalProvider(Protocol):
    """Unified protocol for all provider types."""

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        ...

    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a capability."""
        ...

class EmbeddingProvider(UniversalProvider, Protocol):
    """Embedding-specific provider."""
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class RerankProvider(UniversalProvider, Protocol):
    """Reranking-specific provider."""
    async def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]: ...
```

### 5.2 Dynamic Capability Detection

```python
class CapabilityDetector:
    """Detect and validate provider capabilities."""

    @staticmethod
    def supports_capability(provider: UniversalProvider, capability: ProviderCapability) -> bool:
        """Check if provider supports capability."""
        capabilities = provider.get_capabilities()

        return {
            ProviderCapability.EMBEDDING: capabilities.supports_embedding,
            ProviderCapability.RERANKING: capabilities.supports_reranking,
            ProviderCapability.BATCH_PROCESSING: capabilities.supports_batch_processing,
            # ... other mappings
        }.get(capability, False)

    @staticmethod
    def get_optimal_batch_size(provider: UniversalProvider) -> int:
        """Get optimal batch size for provider."""
        capabilities = provider.get_capabilities()
        return capabilities.max_batch_size or 8
```

### 5.3 Enhanced Custom Provider Support

```python
class CustomProviderRegistry:
    """Enhanced registry for custom providers."""

    def register_custom_provider(
        self,
        provider_type: str,
        provider_class: type[UniversalProvider],
        capabilities: ProviderCapabilities,
        display_name: str,
        description: str,
        *,
        validate_implementation: bool = True
    ) -> None:
        """Register custom provider with validation."""

        if validate_implementation:
            self._validate_provider_implementation(provider_class, capabilities)

        custom_provider_type = ProviderType(provider_type)  # Allow custom values

        PROVIDER_REGISTRY[custom_provider_type] = ProviderRegistryEntry(
            provider_class=provider_class,
            capabilities=capabilities,
            provider_type=custom_provider_type,
            display_name=display_name,
            description=description,
            implemented=True
        )
```

## Phase 6: Centralized Configuration Integration (Week 6)

### 6.1 Integration with Main Config System

**File**: `src/codeweaver/config/schema.py` (extending existing)

```python
class ProvidersConfig(BaseModel):
    """Providers section of main config."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Global provider settings
    enabled: bool = Field(True, description="Enable providers system")
    default_embedding_provider: ProviderType = Field(ProviderType.VOYAGE)
    default_reranking_provider: ProviderType = Field(ProviderType.VOYAGE)
    enable_automatic_fallback: bool = Field(True, description="Auto-fallback on provider failure")

    # Provider configurations
    providers: dict[ProviderType, ProviderConfig] = Field(default_factory=dict)

    def add_embedding_provider(self, provider_type: ProviderType, config: EmbeddingProviderConfig) -> None:
        """Add embedding provider configuration."""
        self.providers[provider_type] = config

    def get_embedding_config(self, provider_type: ProviderType) -> EmbeddingProviderConfig | None:
        """Get embedding configuration for provider."""
        config = self.providers.get(provider_type)
        if isinstance(config, EmbeddingProviderConfig):
            return config
        return None

# Update main config to include providers
class CodeWeaverConfig(BaseModel):
    """Master configuration - add providers section."""
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    backends: BackendConfig = Field(default_factory=BackendConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
```

### 6.2 TOML Configuration Support

```python
# Example configuration in codeweaver.toml
[providers]
enabled = true
default_embedding_provider = "voyage"
default_reranking_provider = "voyage"

[providers.voyage]
enabled = true
api_key = "${CW_VOYAGE_API_KEY}"
model = "voyage-code-3"
rerank_model = "voyage-rerank-2"
batch_size = 32
timeout_seconds = 30.0

[providers.openai]
enabled = false
api_key = "${CW_OPENAI_API_KEY}"
model = "text-embedding-3-small"
dimension = 1536
```

## Implementation Timeline

### **Week 1: Emergency Type System Fix**
1. Create correct `ProviderCapability` enum
2. Fix runtime failures across all provider files
3. Create basic capabilities model
4. Add comprehensive tests for type system

### **Week 2: Consolidate Hardcoded Attributes**
1. Create provider registry with centralized metadata
2. Create model enums for each provider
3. Eliminate hardcoded ClassVar definitions
4. Update all provider classes to use registry

### **Week 3: Pydantic Migration**
1. Convert all configuration classes to Pydantic models
2. Update provider implementations for Pydantic config
3. Add validation rules and field constraints
4. Implement serialization support

### **Week 4: Remove Backwards Compatibility**
1. Delete all legacy configuration code
2. Remove manual dict conversion patterns
3. Simplify interface hierarchy (Protocol vs ABC)
4. Clean environment variable handling

### **Week 5: Interface Improvements**
1. Create unified provider protocol
2. Implement dynamic capability detection
3. Add enhanced custom provider support
4. Comprehensive interface documentation

### **Week 6: Configuration Integration**
1. Integrate with centralized config system
2. Implement TOML configuration support
3. Add hierarchical config loading
4. Final testing and validation

## File Changes Summary

### **New Files**
```
src/codeweaver/_types/provider_enums.py
src/codeweaver/_types/provider_capabilities.py
src/codeweaver/_types/provider_registry.py
src/codeweaver/providers/config.py
src/codeweaver/providers/detection.py
src/codeweaver/providers/custom.py
```

### **Modified Files**
```
src/codeweaver/providers/base.py (simplify interfaces)
src/codeweaver/providers/factory.py (Pydantic integration)
src/codeweaver/providers/voyage.py (use registry + Pydantic)
src/codeweaver/providers/openai.py (use registry + Pydantic)
src/codeweaver/providers/cohere.py (use registry + Pydantic)
src/codeweaver/providers/huggingface.py (use registry + Pydantic)
src/codeweaver/providers/sentence_transformers.py (use registry + Pydantic)
src/codeweaver/providers/__init__.py (update exports)
```

### **Deleted Files**
```
- Legacy configuration compatibility modules
- Backwards compatibility shims
- Redundant provider info helpers
```

## Key Improvements Delivered

### ✅ **1. Attributes for Providers - SOLVED**
- **Before**: Hardcoded metadata scattered across 5+ provider classes
- **After**: Centralized registry with single source of truth for all provider attributes
- **Custom Adapters**: Enhanced registry with validation and plugin support

### ✅ **2. Types - MODERNIZED**
- **Before**: Broken `ProviderCapability` TypedDict used as enum, string literals everywhere
- **After**: Proper enums for capabilities, provider types, and model names with type safety
- **Consistency**: All type attributes consolidated with their enums (`ProviderType.VOYAGE.capabilities`)

### ✅ **3. Interface Flexibility - ENHANCED**
- **Before**: Redundant Protocol/ABC hierarchy, complex inheritance in CombinedProvider
- **After**: Unified protocol approach with dynamic capability detection
- **Extensibility**: Enhanced custom provider system with validation

### ✅ **4. Backwards Compatibility - REMOVED**
- **Before**: Legacy config classes, manual dict conversion, dual provider info methods
- **After**: Clean, simple APIs without legacy support
- **Impact**: Reduced complexity, eliminated runtime failures

### ✅ **5. Serialization - PYDANTIC V2**
- **Before**: Manual dict conversion, no type safety, `getattr()` patterns
- **After**: Pydantic models with validation, automatic serialization, TOML support
- **Config**: Integrated with centralized config system using tomlkit

## Success Criteria

1. **✅ Runtime Stability**: Fix critical `ProviderCapability` type system failures
2. **✅ Single Source of Truth**: All provider attributes defined in centralized registry
3. **✅ Type Safety**: All string literals replaced with enums, Pydantic validation
4. **✅ Clean API**: No backwards compatibility code, simplified interfaces
5. **✅ Pydantic Everywhere**: All data models use Pydantic v2 with validation
6. **✅ Centralized Config**: Integrated with main config system using tomlkit
7. **✅ Extensibility**: Easy custom provider registration with validation
8. **✅ Consistency**: Approach aligned with backend and sources improvements

## Risk Mitigation

1. **Critical Fixes First**: Address runtime failures in Week 1 before other changes
2. **Type Safety**: Pydantic validation prevents configuration errors
3. **Breaking Changes**: All changes are internal to unreleased tool - no external API impact
4. **Testing**: Comprehensive test suite covers all provider types and configurations
5. **Documentation**: Clear migration path and examples for each improvement

## Dependencies to Add

```toml
[project.dependencies]
pydantic = "^2.5.0"  # Already planned for backends/sources
tomlkit = "^0.12.0"  # Already planned for backends/sources
```

This improvement plan addresses all identified issues with the providers module while maintaining consistency with the backend and sources improvement approaches, resulting in a robust, type-safe, and extensible provider system.
