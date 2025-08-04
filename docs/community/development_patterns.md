<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Development Patterns Guide

**Date:** January 27, 2025
**Author:** CodeWeaver Development Team
**Version:** 1.0

## Overview

This guide establishes the development patterns and coding standards for CodeWeaver based on the "gold standard" patterns from the providers module. All new components should follow these patterns to ensure consistency, maintainability, and proper integration with the services layer.

## Core Design Principles

### 1. **Consistency First**
All components should follow identical patterns for naming, structure, and behavior.

### 2. **Services Layer Integration**
Components should integrate with the services layer while providing fallback functionality.

### 3. **Protocol-Based Design**
Use runtime-checkable protocols for dependency injection and interface definition.

### 4. **Configuration-Driven**
All behavior should be configurable through hierarchical configuration files.

### 5. **Graceful Degradation**
Components should work with reduced functionality when dependencies are unavailable.

## Naming Conventions

### Class Names

```python
# ✅ CORRECT: Follow established patterns
class VoyageAIProvider:          # Embedding/reranking providers
class QdrantBackend:             # Vector database backends
class FileSystemSourceProvider:  # Data source providers
class ChunkingService:           # Service implementations

# ❌ WRONG: Inconsistent naming
class VoyageAI:                  # Missing "Provider" suffix
class QdrantVectorDB:            # Inconsistent naming
class FileSystemSource:          # Missing "Provider" suffix
class ChunkingServiceProvider:   # Redundant "ServiceProvider"
```

### Configuration Classes

```python
# ✅ CORRECT: Clean, descriptive names
class VoyageConfig:              # Provider configuration
class QdrantConfig:              # Backend configuration
class FileSystemConfig:          # Source configuration
class ChunkingServiceConfig:     # Service configuration

# ❌ WRONG: Redundant or unclear names
class VoyageAIProviderConfig:    # Redundant "Provider"
class QdrantBackendConfig:       # Redundant "Backend"
class FileSystemSourceConfig:   # Redundant "Source"
```

### Method and Property Names

```python
# ✅ CORRECT: Consistent property patterns
@property
def provider_name(self) -> str:          # For providers
    return ProviderType.VOYAGE_AI.value

@property
def backend_name(self) -> str:           # For backends
    return BackendType.QDRANT.value

@property
def source_name(self) -> str:            # For sources
    return SourceType.FILESYSTEM.value

# ✅ CORRECT: Consistent method patterns
@classmethod
def check_availability(cls, capability) -> tuple[bool, str | None]:
    """Check if component is available for given capability."""

@classmethod
def get_static_provider_info(cls) -> EmbeddingProviderInfo:
    """Get static information about this provider."""
```

## Required Patterns

### 1. Base Class Inheritance

```python
# ✅ CORRECT: Inherit from appropriate base class
from codeweaver.providers.base import CombinedProvider
from codeweaver.cw_types import EmbeddingProvider, RerankingProvider

class VoyageAIProvider(CombinedProvider):
    """VoyageAI provider supporting both embeddings and reranking."""

    def __init__(self, config: VoyageConfig | dict[str, Any]):
        super().__init__(config)
        # Provider-specific initialization
```

### 2. Required Properties

All components must implement these properties:

```python
class ExampleProvider:
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.EXAMPLE.value

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return self._capabilities

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """Get maximum batch size for processing."""
        return self._max_batch_size
```

### 3. Required Class Methods

```python
class ExampleProvider:
    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if provider is available for the given capability.

        Args:
            capability: The capability to check

        Returns:
            Tuple of (is_available, error_message)
        """
        if not EXAMPLE_LIBRARY_AVAILABLE:
            return False, "example-library package not installed (install with: uv add example-library)"

        if capability in [ProviderCapability.EMBEDDING, ProviderCapability.RERANKING]:
            return True, None

        return False, f"Capability {capability.value} not supported by Example provider"

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static information about this provider.

        Returns:
            Provider information including capabilities and models
        """
        return EmbeddingProviderInfo(
            name=ProviderType.EXAMPLE.value,
            capabilities=cls._get_static_capabilities(),
            supported_models=["example-model-v1", "example-model-v2"],
            default_model="example-model-v1",
            native_dimensions={"example-model-v1": 1024, "example-model-v2": 1536},
            description="Example provider for embeddings and reranking"
        )
```

### 4. Configuration Validation

```python
class ExampleProvider:
    def _validate_config(self) -> None:
        """Validate provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.get("api_key"):
            raise ValueError("Example provider API key is required")

        model = self.config.get("model", self._capabilities.default_model)
        if model not in self._capabilities.supported_models:
            available = ", ".join(self._capabilities.supported_models)
            raise ValueError(f"Unknown Example model: {model}. Available: {available}")

        # Validate other configuration parameters
        max_batch_size = self.config.get("max_batch_size", 128)
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
```

### 5. Services Layer Integration

```python
class ExampleProvider:
    async def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        """Generate embeddings with services layer integration.

        Args:
            texts: List of texts to embed
            context: Service context for rate limiting, caching, etc.

        Returns:
            List of embedding vectors
        """
        if context is None:
            context = {}

        # Rate limiting service integration
        rate_limiter = context.get("rate_limiting_service")
        if rate_limiter:
            await rate_limiter.acquire("example_provider", len(texts))

        # Caching service integration
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = self._generate_cache_key(texts)
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                return cached_result

        try:
            # Generate embeddings
            result = await self._generate_embeddings(texts)

            # Cache result if service available
            if cache_service:
                await cache_service.set(cache_key, result, ttl=3600)

            # Record metrics if service available
            metrics_service = context.get("metrics_service")
            if metrics_service:
                await metrics_service.record_request(
                    provider="example_provider",
                    operation="embed_documents",
                    count=len(texts),
                    success=True
                )

            return result

        except Exception as e:
            # Record failure metrics
            metrics_service = context.get("metrics_service")
            if metrics_service:
                await metrics_service.record_request(
                    provider="example_provider",
                    operation="embed_documents",
                    count=len(texts),
                    success=False,
                    error=str(e)
                )

            logger.exception("Error generating Example embeddings")
            raise
```

## Configuration Patterns

### 1. Base Configuration Classes

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated

class BaseProviderConfig(BaseModel):
    """Base configuration for all providers."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        frozen=False,
    )

    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class ExampleConfig(BaseProviderConfig):
    """Configuration for Example provider."""
    api_key: Annotated[str, Field(description="Example API key")]
    model: Annotated[str, Field(default="example-model-v1", description="Model to use")]
    max_batch_size: Annotated[int, Field(default=128, description="Maximum batch size")]
    dimension: Annotated[int | None, Field(default=None, description="Override embedding dimension")]
```

### 2. Configuration Validation

```python
class ExampleConfig(BaseProviderConfig):
    """Configuration for Example provider with validation."""

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()

    @field_validator("max_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_batch_size must be positive")
        if v > 1000:
            raise ValueError("max_batch_size cannot exceed 1000")
        return v

    @model_validator(mode="after")
    def validate_model_dimension_compatibility(self) -> "ExampleConfig":
        """Validate model and dimension compatibility."""
        if self.dimension is not None:
            valid_dimensions = {
                "example-model-v1": [512, 1024],
                "example-model-v2": [768, 1536],
            }

            if self.model in valid_dimensions:
                if self.dimension not in valid_dimensions[self.model]:
                    valid = ", ".join(map(str, valid_dimensions[self.model]))
                    raise ValueError(
                        f"Dimension {self.dimension} not supported for model {self.model}. "
                        f"Valid dimensions: {valid}"
                    )

        return self
```

## Error Handling Patterns

### 1. Consistent Exception Types

```python
from codeweaver.cw_types import (
    ProviderError,
    ConfigurationError,
    ServiceUnavailableError,
    RateLimitError
)

class ExampleProvider:
    async def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        """Generate embeddings with proper error handling."""
        try:
            # Validate inputs
            if not texts:
                raise ValueError("texts cannot be empty")

            if any(not text.strip() for text in texts):
                raise ValueError("texts cannot contain empty strings")

            # Check rate limits
            if len(texts) > self.max_batch_size:
                raise RateLimitError(
                    f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}"
                )

            # Generate embeddings
            result = await self._call_api(texts)
            return result

        except ValueError as e:
            # Re-raise validation errors as-is
            raise
        except RateLimitError as e:
            # Re-raise rate limit errors as-is
            raise
        except ConnectionError as e:
            # Convert connection errors to service unavailable
            raise ServiceUnavailableError(f"Example API unavailable: {e}") from e
        except Exception as e:
            # Convert unexpected errors to provider errors
            raise ProviderError(f"Example provider error: {e}") from e
```

### 2. Error Recovery Patterns

```python
class ExampleProvider:
    async def embed_documents_with_retry(
        self,
        texts: list[str],
        context: dict | None = None
    ) -> list[list[float]]:
        """Generate embeddings with automatic retry."""
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 1.0)

        for attempt in range(max_retries + 1):
            try:
                return await self.embed_documents(texts, context)
            except ServiceUnavailableError as e:
                if attempt == max_retries:
                    raise

                logger.warning(
                    f"Example API unavailable (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            except RateLimitError as e:
                if attempt == max_retries:
                    raise

                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                await asyncio.sleep(retry_delay * 5)  # Longer delay for rate limits
```

## Testing Patterns

### 1. Component Testing Structure

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from codeweaver.providers.example import ExampleProvider, ExampleConfig

class TestExampleProvider:
    """Test suite for Example provider following standard patterns."""

    @pytest.fixture
    def config(self) -> ExampleConfig:
        """Standard configuration for testing."""
        return ExampleConfig(
            api_key="test-key",
            model="example-model-v1",
            max_batch_size=128
        )

    @pytest.fixture
    def provider(self, config: ExampleConfig) -> ExampleProvider:
        """Provider instance for testing."""
        return ExampleProvider(config)

    @pytest.fixture
    def mock_context(self) -> dict:
        """Mock service context."""
        return {
            "rate_limiting_service": AsyncMock(),
            "caching_service": AsyncMock(),
            "metrics_service": AsyncMock(),
        }

    # Pattern compliance tests
    def test_provider_follows_naming_convention(self, provider: ExampleProvider):
        """Test that provider follows naming conventions."""
        assert provider.__class__.__name__.endswith("Provider")
        assert hasattr(provider, "provider_name")
        assert provider.provider_name == "example"

    def test_provider_has_required_properties(self, provider: ExampleProvider):
        """Test that provider has all required properties."""
        required_properties = [
            "provider_name", "capabilities", "model_name",
            "dimension", "max_batch_size"
        ]
        for prop_name in required_properties:
            assert hasattr(provider, prop_name)
            assert getattr(provider, prop_name) is not None

    def test_provider_has_required_class_methods(self):
        """Test that provider has all required class methods."""
        required_methods = ["check_availability", "get_static_provider_info"]
        for method_name in required_methods:
            assert hasattr(ExampleProvider, method_name)
            assert callable(getattr(ExampleProvider, method_name))

    # Functionality tests
    async def test_embed_documents_with_services(
        self,
        provider: ExampleProvider,
        mock_context: dict
    ):
        """Test embedding generation with services integration."""
        # Mock successful embedding generation
        provider._generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await provider.embed_documents(["test text"], mock_context)

        assert len(result) == 1
        assert len(result[0]) == 3

        # Verify service interactions
        mock_context["rate_limiting_service"].acquire.assert_called_once()
        mock_context["caching_service"].get.assert_called_once()
        mock_context["metrics_service"].record_request.assert_called_once()

    async def test_embed_documents_without_services(self, provider: ExampleProvider):
        """Test embedding generation without services (fallback)."""
        # Mock successful embedding generation
        provider._generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await provider.embed_documents(["test text"], {})

        assert len(result) == 1
        assert len(result[0]) == 3

    async def test_embed_documents_with_cache_hit(
        self,
        provider: ExampleProvider,
        mock_context: dict
    ):
        """Test embedding generation with cache hit."""
        # Mock cache hit
        cached_result = [[0.4, 0.5, 0.6]]
        mock_context["caching_service"].get.return_value = cached_result

        result = await provider.embed_documents(["test text"], mock_context)

        assert result == cached_result
        # Should not call actual embedding generation
        assert not hasattr(provider, '_generate_embeddings') or \
               not provider._generate_embeddings.called

    # Error handling tests
    async def test_embed_documents_handles_api_error(self, provider: ExampleProvider):
        """Test error handling for API failures."""
        provider._generate_embeddings = AsyncMock(
            side_effect=ConnectionError("API unavailable")
        )

        with pytest.raises(ServiceUnavailableError):
            await provider.embed_documents(["test text"], {})

    async def test_embed_documents_validates_input(self, provider: ExampleProvider):
        """Test input validation."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            await provider.embed_documents([], {})

        with pytest.raises(ValueError, match="texts cannot contain empty strings"):
            await provider.embed_documents(["", "valid text"], {})

    # Configuration tests
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = ExampleConfig(
            api_key="valid-key",
            model="example-model-v1",
            max_batch_size=64
        )
        # Should not raise
        provider = ExampleProvider(config)
        assert provider.config["api_key"] == "valid-key"

    def test_config_validation_failure(self):
        """Test configuration validation failures."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            ExampleConfig(api_key="", model="example-model-v1")

        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            ExampleConfig(api_key="valid-key", model="example-model-v1", max_batch_size=0)

    # Availability tests
    def test_check_availability_success(self):
        """Test availability checking for supported capabilities."""
        from codeweaver.cw_types import ProviderCapability

        available, error = ExampleProvider.check_availability(ProviderCapability.EMBEDDING)
        assert available is True
        assert error is None

    def test_check_availability_unsupported(self):
        """Test availability checking for unsupported capabilities."""
        from codeweaver.cw_types import ProviderCapability

        available, error = ExampleProvider.check_availability(ProviderCapability.CLASSIFICATION)
        assert available is False
        assert "not supported" in error.lower()

    # Static info tests
    def test_get_static_provider_info(self):
        """Test static provider information."""
        info = ExampleProvider.get_static_provider_info()

        assert info.name == "example"
        assert len(info.supported_models) > 0
        assert info.default_model in info.supported_models
        assert len(info.native_dimensions) > 0
        assert info.description is not None
```

### 2. Integration Testing Patterns

```python
class TestExampleProviderIntegration:
    """Integration tests for Example provider."""

    @pytest.mark.integration
    async def test_real_api_integration(self):
        """Test integration with real Example API."""
        # Skip if no API key available
        api_key = os.getenv("EXAMPLE_API_KEY")
        if not api_key:
            pytest.skip("EXAMPLE_API_KEY not set")

        config = ExampleConfig(api_key=api_key)
        provider = ExampleProvider(config)

        result = await provider.embed_documents(["Hello, world!"], {})

        assert len(result) == 1
        assert len(result[0]) == provider.dimension
        assert all(isinstance(x, float) for x in result[0])

    @pytest.mark.integration
    async def test_services_integration(self):
        """Test integration with actual services."""
        from codeweaver.services.manager import ServicesManager
        from codeweaver.config import CodeWeaverConfig

        # Load test configuration
        config = CodeWeaverConfig.from_dict({
            "services": {
                "caching": {"enabled": True},
                "rate_limiting": {"enabled": True},
            },
            "providers": {
                "example": {
                    "api_key": os.getenv("EXAMPLE_API_KEY", "test-key"),
                    "model": "example-model-v1"
                }
            }
        })

        # Initialize services
        services_manager = ServicesManager(config.services)
        await services_manager.start_all_services()

        try:
            # Create provider and context
            provider_config = ExampleConfig(**config.providers["example"])
            provider = ExampleProvider(provider_config)
            context = await services_manager.create_service_context()

            # Test with services
            result = await provider.embed_documents(["Test integration"], context)
            assert len(result) == 1

        finally:
            await services_manager.stop_all_services()
```

## Registry Integration Patterns

### 1. Component Registration

```python
# At the end of each component module
from codeweaver.cw_types import register_provider_class, ProviderType

# Register the provider in the registry
register_provider_class(ProviderType.EXAMPLE, ExampleProvider)
```

### 2. Factory Integration

```python
# In factory modules
from codeweaver.providers.example import ExampleProvider

PROVIDER_REGISTRY = {
    "example": ProviderRegistration(
        provider_class=ExampleProvider,
        config_class=ExampleConfig,
        provider_info=ExampleProvider.get_static_provider_info(),
        capabilities=ExampleProvider.get_static_provider_info().capabilities,
    )
}
```

## Documentation Patterns

### 1. Module Documentation

```python
"""
Example provider implementation for embeddings and reranking.

This module provides the ExampleProvider class that implements the unified provider
interface for Example's embedding and reranking APIs. It supports both embedding
generation and document reranking with rate limiting and caching integration.

Example:
    Basic usage:

    ```python
    from codeweaver.providers.example import ExampleProvider, ExampleConfig

    config = ExampleConfig(api_key="your-key", model="example-model-v1")
    provider = ExampleProvider(config)

    embeddings = await provider.embed_documents(["Hello, world!"], context)
    ```

    With services integration:

    ```python
    context = await services_manager.create_service_context()
    embeddings = await provider.embed_documents(["Hello, world!"], context)
    ```
"""
```

### 2. Class Documentation

```python
class ExampleProvider(CombinedProvider):
    """Example provider supporting both embeddings and reranking.

    This provider implements the unified provider interface for Example's APIs,
    supporting both embedding generation and document reranking. It integrates
    with the services layer for rate limiting, caching, and health monitoring.

    Attributes:
        provider_name: The provider identifier ("example")
        capabilities: Provider capabilities (embedding, reranking)
        model_name: Current embedding model name
        dimension: Embedding vector dimension
        max_batch_size: Maximum batch size for API calls

    Example:
        ```python
        config = ExampleConfig(api_key="your-key")
        provider = ExampleProvider(config)

        # Generate embeddings
        embeddings = await provider.embed_documents(["text1", "text2"], context)

        # Rerank documents
        results = await provider.rerank_documents("query", ["doc1", "doc2"], context)
        ```
    """
```

### 3. Method Documentation

```python
async def embed_documents(
    self,
    texts: list[str],
    context: dict | None = None
) -> list[list[float]]:
    """Generate embeddings for the given texts.

    This method generates embedding vectors for the provided texts using the
    configured Example model. It integrates with the services layer for rate
    limiting, caching, and metrics collection.

    Args:
        texts: List of texts to embed. Cannot be empty or contain empty strings.
        context: Optional service context for rate limiting, caching, etc.
            If None, services integration is disabled.

    Returns:
        List of embedding vectors, one per input text. Each vector has
        dimension equal to self.dimension.

    Raises:
        ValueError: If texts is empty or contains empty strings
        RateLimitError: If batch size exceeds maximum or rate limit hit
        ServiceUnavailableError: If Example API is unavailable
        ProviderError: For other API or processing errors

    Example:
        ```python
        embeddings = await provider.embed_documents(
            ["Hello, world!", "How are you?"],
            context
        )
        assert len(embeddings) == 2
        assert len(embeddings[0]) == provider.dimension
        ```
    """
```

## Code Review Checklist

When reviewing new components, ensure they follow these patterns:

### ✅ **Naming and Structure**
- [ ] Class names follow convention (ends with Provider/Backend/Source)
- [ ] Configuration classes follow naming convention
- [ ] Method names are consistent with established patterns
- [ ] File and module names are descriptive and consistent

### ✅ **Required Patterns**
- [ ] Inherits from appropriate base class
- [ ] Implements all required properties (`provider_name`, `capabilities`, etc.)
- [ ] Implements all required class methods (`check_availability`, `get_static_*_info`)
- [ ] Has proper configuration validation (`_validate_config`)

### ✅ **Services Integration**
- [ ] All public methods accept optional `context` parameter
- [ ] Integrates with rate limiting, caching, metrics services
- [ ] Provides fallback behavior when services unavailable
- [ ] Handles service errors gracefully

### ✅ **Error Handling**
- [ ] Uses consistent exception types
- [ ] Provides clear error messages
- [ ] Implements proper error recovery where appropriate
- [ ] Logs errors appropriately

### ✅ **Testing**
- [ ] Has comprehensive test suite
- [ ] Tests both with and without services
- [ ] Tests error conditions and edge cases
- [ ] Includes pattern compliance tests

### ✅ **Documentation**
- [ ] Module has comprehensive docstring
- [ ] Classes have detailed docstrings with examples
- [ ] Methods have complete docstrings with args/returns/raises
- [ ] Examples are provided for common usage patterns

### ✅ **Configuration**
- [ ] Configuration class inherits from appropriate base
- [ ] Has proper field validation
- [ ] Supports environment variable overrides
- [ ] Has sensible defaults

### ✅ **Registry Integration**
- [ ] Component is properly registered
- [ ] Factory integration is complete
- [ ] Static information is accurate
- [ ] Capabilities are correctly defined

## Common Anti-Patterns to Avoid

### ❌ **Direct Dependencies**
```python
# DON'T: Direct middleware imports
from codeweaver.middleware.chunking import ChunkingMiddleware

class BadProvider:
    def process(self, data):
        chunker = ChunkingMiddleware()  # Direct dependency
        return chunker.chunk(data)
```

### ❌ **Missing Fallbacks**
```python
# DON'T: Assume services are always available
class BadProvider:
    async def process(self, data, context):
        service = context["required_service"]  # Will fail if not available
        return await service.process(data)
```

### ❌ **Inconsistent Naming**
```python
# DON'T: Inconsistent class names
class VoyageAI:              # Missing "Provider"
class OpenAIEmbedder:        # Inconsistent suffix
class CohereProviderClass:   # Redundant "Class"
```

### ❌ **Poor Error Handling**
```python
# DON'T: Generic exception handling
class BadProvider:
    async def process(self, data):
        try:
            return await self.api_call(data)
        except Exception:
            return None  # Swallows all errors
```

### ❌ **Missing Configuration Validation**
```python
# DON'T: No configuration validation
class BadProvider:
    def __init__(self, config):
        self.api_key = config.get("api_key")  # No validation
        self.model = config.get("model")      # No validation
```

## Conclusion

Following these development patterns ensures consistency, maintainability, and proper integration across all CodeWeaver components. The key principles are:

1. **Follow Established Patterns** - Use the providers module as the gold standard
2. **Integrate with Services Layer** - Always provide context parameter and fallbacks
3. **Handle Errors Gracefully** - Use consistent exception types and recovery patterns
4. **Test Comprehensively** - Test both with and without services integration
5. **Document Thoroughly** - Provide clear examples and usage guidance

For more information, see:
- [Services Layer Usage Guide](SERVICES_LAYER_GUIDE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Factory System Documentation](FACTORY_SYSTEM.md)
