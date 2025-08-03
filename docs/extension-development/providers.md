# Building Custom Providers

This guide covers building custom embedding and reranking providers for CodeWeaver. Providers handle model inference for semantic search and content ranking.

## 🎯 Overview

CodeWeaver supports three types of providers:

- **Embedding Providers**: Generate dense vector representations of text
- **Reranking Providers**: Re-order search results based on relevance
- **NLP Providers**: Advanced text processing and analysis

## 🏗️ Provider Architecture

### Core Protocols

#### EmbeddingProvider Protocol
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    # Required Properties
    @property
    def provider_name(self) -> str: ...
    @property 
    def model_name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    
    # Optional Properties
    @property
    def max_batch_size(self) -> int | None: ...
    @property
    def max_input_length(self) -> int | None: ...
    
    # Required Methods
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    def get_provider_info(self) -> EmbeddingProviderInfo: ...
    async def health_check(self) -> bool: ...
```

#### RerankProvider Protocol
```python
@runtime_checkable
class RerankProvider(Protocol):
    # Required Properties
    @property
    def provider_name(self) -> str: ...
    @property
    def model_name(self) -> str: ...
    
    # Optional Properties  
    @property
    def max_documents(self) -> int | None: ...
    @property
    def max_query_length(self) -> int | None: ...
    
    # Required Methods
    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[RerankResult]: ...
    def get_provider_info(self) -> EmbeddingProviderInfo: ...
    async def health_check(self) -> bool: ...
```

### Base Classes

CodeWeaver provides abstract base classes to simplify provider development:

#### EmbeddingProviderBase
```python
from codeweaver.providers.base import EmbeddingProviderBase

class MyProvider(EmbeddingProviderBase):
    def _validate_config(self) -> None:
        """Validate configuration during initialization."""
        if not hasattr(self.config, 'api_key'):
            raise ValueError("API key required")
    
    @property
    def provider_name(self) -> str:
        return "my_custom_provider"
    
    # Implement required methods...
```

## 🚀 Implementation Guide

### Step 1: Define Configuration

Create a Pydantic configuration model:

```python
from pydantic import BaseModel, Field
from typing import Annotated

class MyProviderConfig(BaseModel):
    """Configuration for MyProvider."""
    
    api_key: Annotated[str, Field(description="API key for the service")]
    model_name: Annotated[str, Field(default="default-model")]
    base_url: Annotated[str, Field(default="https://api.example.com")]
    timeout: Annotated[int, Field(default=30, ge=1, le=300)]
    max_retries: Annotated[int, Field(default=3, ge=0, le=10)]
```

### Step 2: Implement Provider Class

```python
from codeweaver.providers.base import EmbeddingProviderBase
from codeweaver.cw_types import EmbeddingProviderInfo
import aiohttp
import asyncio

class MyEmbeddingProvider(EmbeddingProviderBase):
    """Custom embedding provider implementation."""
    
    def __init__(self, config: MyProviderConfig):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.api_key:
            raise ValueError("API key is required")
        if not self.config.model_name:
            raise ValueError("Model name is required")
    
    async def _initialize_session(self) -> None:
        """Initialize HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
    
    async def _cleanup_session(self) -> None:
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @property
    def provider_name(self) -> str:
        return "my_custom_provider"
    
    @property
    def model_name(self) -> str:
        return self.config.model_name
    
    @property
    def dimension(self) -> int:
        # Return the embedding dimension for your model
        return 1536  # Example for OpenAI ada-002
    
    @property
    def max_batch_size(self) -> int | None:
        return 100  # Adjust based on your API limits
    
    @property
    def max_input_length(self) -> int | None:
        return 8000  # Adjust based on your model limits
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        await self._initialize_session()
        
        # Process in batches if needed
        batch_size = self.max_batch_size or len(texts)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._embed_batch(batch)
            results.extend(batch_embeddings)
        
        return results
    
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    f"{self.config.base_url}/embeddings",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract embeddings from response
                    embeddings = []
                    for item in data["data"]:
                        embeddings.append(item["embedding"])
                    
                    return embeddings
                    
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get provider information."""
        return EmbeddingProviderInfo(
            provider_name=self.provider_name,
            model_name=self.model_name,
            embedding_dimension=self.dimension,
            max_batch_size=self.max_batch_size,
            max_input_length=self.max_input_length,
            supported_languages=["en", "es", "fr"],  # Adjust as needed
            description="Custom embedding provider implementation"
        )
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            await self._initialize_session()
            # Test with a simple embedding
            await self.embed_query("test")
            return True
        except Exception:
            return False
        finally:
            await self._cleanup_session()
```

### Step 3: Create Plugin Interface

```python
from codeweaver.factories.plugin_protocols import ProviderPlugin
from codeweaver.cw_types import (
    ComponentType, 
    BaseCapabilities, 
    BaseComponentInfo, 
    ValidationResult,
    ProviderCapabilities
)

class MyProviderPlugin(ProviderPlugin):
    """Plugin interface for MyEmbeddingProvider."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "my_custom_provider"
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.PROVIDER
    
    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        return ProviderCapabilities(
            supports_embeddings=True,
            supports_reranking=False,
            max_batch_size=100,
            supported_languages=["en", "es", "fr"],
            embedding_dimensions=[1536]
        )
    
    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        return BaseComponentInfo(
            name="my_custom_provider",
            display_name="My Custom Embedding Provider",
            description="Custom embedding provider for specialized models",
            component_type=ComponentType.PROVIDER,
            version="1.0.0",
            author="Your Name",
            homepage="https://github.com/yourname/my-provider"
        )
    
    @classmethod
    def validate_config(cls, config: MyProviderConfig) -> ValidationResult:
        """Validate provider configuration."""
        errors = []
        warnings = []
        
        if not config.api_key:
            errors.append("API key is required")
        
        if config.timeout < 5:
            warnings.append("Timeout below 5 seconds may cause failures")
        
        if config.max_retries > 5:
            warnings.append("High retry count may cause slow responses")
        
        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def get_dependencies(cls) -> list[str]:
        """Get required dependencies."""
        return ["aiohttp", "pydantic"]
    
    @classmethod
    def get_provider_class(cls) -> type[EmbeddingProvider]:
        return MyEmbeddingProvider
```

### Step 4: Register the Provider

Create an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."codeweaver.providers"]
my_custom_provider = "my_package.provider:MyProviderPlugin"
```

Or register programmatically:

```python
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory

# Create factory instance
factory = CodeWeaverFactory()

# Register provider
factory.register_provider(
    "my_custom_provider",
    MyEmbeddingProvider,
    MyProviderPlugin.get_capabilities(),
    MyProviderPlugin.get_component_info()
)
```

## 🔧 Advanced Features

### Combined Providers

Implement both embedding and reranking in a single provider:

```python
from codeweaver.providers.base import CombinedProvider
from codeweaver.cw_types import RerankResult

class MyComboProvider(CombinedProvider):
    """Provider supporting both embedding and reranking."""
    
    # Implement EmbeddingProvider methods...
    
    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents based on query."""
        # Implementation here
        scores = await self._calculate_relevance_scores(query, documents)
        
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                index=i,
                document=doc,
                relevance_score=score
            ))
        
        # Sort by relevance and apply top_k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if top_k:
            results = results[:top_k]
        
        return results
```

### Local Providers

For providers that don't require API keys:

```python
from codeweaver.providers.base import LocalEmbeddingProvider

class MyLocalProvider(LocalEmbeddingProvider):
    """Local embedding provider using local models."""
    
    def __init__(self, config: MyLocalConfig):
        super().__init__(config)
        self.model = None
    
    async def _load_model(self):
        """Load local model."""
        if not self.model:
            # Load your local model here
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModel.from_pretrained(self.config.model_path)
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        await self._load_model()
        # Use local model for embedding
        # ...
```

### NLP Providers

For advanced text processing:

```python
from codeweaver.providers.base import LocalNLPProvider
from codeweaver.cw_types import IntentType

class MyNLPProvider(LocalNLPProvider):
    """Advanced NLP provider."""
    
    async def classify_intent(self, text: str) -> tuple[IntentType | None, float]:
        """Classify user intent."""
        # Implement intent classification
        intent = self._analyze_intent(text)
        confidence = self._calculate_confidence(text, intent)
        return intent, confidence
    
    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities."""
        # Implement entity extraction
        entities = self._extract_entities(text)
        return entities
```

## 🧪 Testing Your Provider

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch
from my_package.provider import MyEmbeddingProvider, MyProviderConfig

@pytest.fixture
def provider_config():
    return MyProviderConfig(
        api_key="test-key",
        model_name="test-model",
        base_url="https://api.test.com"
    )

@pytest.fixture
def provider(provider_config):
    return MyEmbeddingProvider(provider_config)

class TestMyEmbeddingProvider:
    """Test suite for MyEmbeddingProvider."""
    
    async def test_embed_query(self, provider):
        """Test single query embedding."""
        with patch.object(provider, '_embed_batch') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3]]
            
            result = await provider.embed_query("test query")
            
            assert result == [0.1, 0.2, 0.3]
            mock_embed.assert_called_once_with(["test query"])
    
    async def test_embed_documents_batching(self, provider):
        """Test document batching."""
        texts = ["doc1", "doc2", "doc3"]
        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        with patch.object(provider, '_embed_batch') as mock_embed:
            mock_embed.return_value = expected
            
            result = await provider.embed_documents(texts)
            
            assert result == expected
    
    async def test_health_check(self, provider):
        """Test health check."""
        with patch.object(provider, 'embed_query') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            healthy = await provider.health_check()
            
            assert healthy is True
    
    def test_provider_info(self, provider):
        """Test provider information."""
        info = provider.get_provider_info()
        
        assert info.provider_name == "my_custom_provider"
        assert info.model_name == "test-model"
        assert info.embedding_dimension == 1536
```

### Integration Tests

```python
@pytest.mark.integration
class TestMyProviderIntegration:
    """Integration tests with real API."""
    
    @pytest.fixture
    def real_provider(self):
        config = MyProviderConfig(
            api_key=os.getenv("TEST_API_KEY"),
            model_name="test-model"
        )
        return MyEmbeddingProvider(config)
    
    async def test_real_embedding(self, real_provider):
        """Test with real API (requires API key)."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided")
        
        result = await real_provider.embed_query("test query")
        
        assert isinstance(result, list)
        assert len(result) == 1536  # Expected dimension
        assert all(isinstance(x, float) for x in result)
```

## 📊 Performance Guidelines

### Batch Processing
- Process multiple documents together when possible
- Respect API rate limits and batch size constraints
- Implement exponential backoff for retries

### Memory Management
- Use streaming for large datasets
- Cleanup resources in finally blocks
- Monitor memory usage during processing

### Error Handling
- Implement comprehensive error handling
- Provide meaningful error messages
- Support graceful degradation

### Monitoring
- Track embedding latency and throughput
- Monitor API quota usage
- Log errors and performance metrics

## 🚀 Next Steps

- **[Backend Development →](./backends.md)**: Learn about vector backend development
- **[Testing Framework →](./testing.md)**: Comprehensive testing strategies
- **[Performance Guidelines →](./performance.md)**: Optimization best practices
- **[Protocol Reference →](../reference/protocols.md)**: Complete protocol documentation