<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Testing Framework for Extensions

This guide covers comprehensive testing strategies for CodeWeaver extensions, including unit tests, integration tests, protocol compliance, and performance testing.

## 🎯 Overview

CodeWeaver provides a comprehensive testing framework to ensure extension quality:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and real services
- **Protocol Compliance Tests**: Verify protocol implementation correctness
- **Performance Tests**: Validate performance requirements and benchmarks
- **End-to-End Tests**: Test complete workflows and user scenarios

## 🏗️ Testing Architecture

### Test Organization

```plaintext
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_providers/      # Provider-specific unit tests
│   ├── test_backends/       # Backend-specific unit tests
│   ├── test_sources/        # Source-specific unit tests
│   └── test_services/       # Service-specific unit tests
├── integration/             # Integration tests with real services
│   ├── test_provider_integration.py
│   ├── test_backend_integration.py
│   └── test_service_integration.py
├── protocol/                # Protocol compliance tests
│   ├── test_provider_protocols.py
│   ├── test_backend_protocols.py
│   └── test_plugin_protocols.py
├── performance/             # Performance and benchmark tests
│   ├── test_provider_performance.py
│   ├── test_backend_performance.py
│   └── test_benchmarks.py
└── fixtures/                # Test data and fixtures
    ├── sample_vectors.json
    ├── test_documents.txt
    └── mock_responses.json
```

### Testing Utilities

CodeWeaver provides testing utilities in `src/codeweaver/testing/`:

```python
from codeweaver.testing import (
    MockEmbeddingProvider,
    MockVectorBackend,
    MockDataSource,
    ProtocolTestRunner,
    PerformanceTestRunner,
    create_test_vectors,
    create_test_documents
)
```

## 🚀 Unit Testing

### Provider Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from your_extension.provider import MyEmbeddingProvider, MyProviderConfig
from codeweaver.testing import create_test_vectors

@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return MyProviderConfig(
        api_key="test-api-key",
        model_name="test-model",
        base_url="https://api.test.com",
        timeout=30
    )

@pytest.fixture
def provider(provider_config):
    """Create provider instance for testing."""
    return MyEmbeddingProvider(provider_config)

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API testing."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]
    }
    mock_client.post.return_value.__aenter__.return_value = mock_response
    return mock_client

class TestMyEmbeddingProvider:
    """Test suite for MyEmbeddingProvider."""

    async def test_embed_single_document(self, provider, mock_http_client):
        """Test embedding a single document."""
        provider.client = mock_http_client

        result = await provider.embed_query("test document")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]

        # Verify API call
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["input"] == ["test document"]

    async def test_embed_multiple_documents(self, provider, mock_http_client):
        """Test embedding multiple documents."""
        provider.client = mock_http_client

        texts = ["document 1", "document 2"]
        results = await provider.embed_documents(texts)

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

    async def test_batch_processing(self, provider, mock_http_client):
        """Test batching with large document sets."""
        provider.client = mock_http_client
        provider.config.max_batch_size = 2

        # Create more documents than batch size
        texts = [f"document {i}" for i in range(5)]

        await provider.embed_documents(texts)

        # Should make multiple API calls due to batching
        assert mock_http_client.post.call_count >= 2

    async def test_error_handling(self, provider, mock_http_client):
        """Test error handling and retries."""
        provider.client = mock_http_client

        # Mock API error
        mock_http_client.post.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await provider.embed_query("test")

    async def test_retry_logic(self, provider, mock_http_client):
        """Test retry logic on failures."""
        provider.client = mock_http_client
        provider.config.max_retries = 2

        # First call fails, second succeeds
        mock_response_success = AsyncMock()
        mock_response_success.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        mock_http_client.post.side_effect = [
            Exception("Temporary error"),
            mock_response_success.__aenter__.return_value
        ]

        result = await provider.embed_query("test")
        assert result == [0.1, 0.2, 0.3]
        assert mock_http_client.post.call_count == 2

    def test_provider_info(self, provider):
        """Test provider information."""
        info = provider.get_provider_info()

        assert info.provider_name == "my_custom_provider"
        assert info.model_name == "test-model"
        assert isinstance(info.embedding_dimension, int)
        assert info.embedding_dimension > 0

    async def test_health_check_success(self, provider, mock_http_client):
        """Test successful health check."""
        provider.client = mock_http_client

        with patch.object(provider, 'embed_query') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            healthy = await provider.health_check()
            assert healthy is True

    async def test_health_check_failure(self, provider, mock_http_client):
        """Test failed health check."""
        provider.client = mock_http_client

        with patch.object(provider, 'embed_query') as mock_embed:
            mock_embed.side_effect = Exception("Connection failed")

            healthy = await provider.health_check()
            assert healthy is False

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = MyProviderConfig(
            api_key="test-key",
            model_name="test-model"
        )
        provider = MyEmbeddingProvider(valid_config)
        assert provider.config.api_key == "test-key"

        # Invalid configuration
        with pytest.raises(ValueError, match="API key"):
            invalid_config = MyProviderConfig(
                api_key="",  # Empty API key
                model_name="test-model"
            )
            MyEmbeddingProvider(invalid_config)
```

### Backend Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch
from your_extension.backend import MyVectorBackend, MyBackendConfig
from codeweaver.cw_types import VectorPoint, SearchResult, DistanceMetric
from codeweaver.testing import create_test_vectors

@pytest.fixture
def backend_config():
    return MyBackendConfig(
        host="localhost",
        port=6333,
        api_key="test-key"
    )

@pytest.fixture
def backend(backend_config):
    return MyVectorBackend(backend_config)

@pytest.fixture
def test_vectors():
    return create_test_vectors(count=10, dimension=128)

class TestMyVectorBackend:
    """Test suite for MyVectorBackend."""

    async def test_create_collection(self, backend):
        """Test collection creation."""
        with patch.object(backend, '_initialize_client'):
            with patch.object(backend, 'client') as mock_client:
                mock_response = AsyncMock()
                mock_client.put.return_value.__aenter__.return_value = mock_response

                await backend.create_collection("test_collection", 128)

                mock_client.put.assert_called_once()
                call_url = mock_client.put.call_args[0][0]
                assert "test_collection" in call_url

    async def test_upsert_vectors(self, backend, test_vectors):
        """Test vector upsert."""
        with patch.object(backend, '_initialize_client'):
            with patch.object(backend, 'client') as mock_client:
                mock_response = AsyncMock()
                mock_client.put.return_value.__aenter__.return_value = mock_response

                await backend.upsert_vectors("test_collection", test_vectors)

                mock_client.put.assert_called_once()
                call_data = mock_client.put.call_args[1]["json"]
                assert len(call_data["points"]) == len(test_vectors)

    async def test_search_vectors(self, backend):
        """Test vector search."""
        with patch.object(backend, '_initialize_client'):
            with patch.object(backend, 'client') as mock_client:
                # Mock search response
                mock_response = AsyncMock()
                mock_response.json.return_value = {
                    "result": [
                        {"id": "1", "score": 0.95, "payload": {"text": "test1"}},
                        {"id": "2", "score": 0.85, "payload": {"text": "test2"}}
                    ]
                }
                mock_client.post.return_value.__aenter__.return_value = mock_response

                query_vector = [0.1, 0.2, 0.3]
                results = await backend.search_vectors(
                    "test_collection",
                    query_vector,
                    limit=10
                )

                assert len(results) == 2
                assert isinstance(results[0], SearchResult)
                assert results[0].id == "1"
                assert results[0].score == 0.95

    async def test_batch_upsert(self, backend):
        """Test batch processing for large vector sets."""
        large_vector_set = create_test_vectors(count=250, dimension=128)

        with patch.object(backend, '_initialize_client'):
            with patch.object(backend, 'client') as mock_client:
                mock_response = AsyncMock()
                mock_client.put.return_value.__aenter__.return_value = mock_response

                await backend.upsert_vectors("test_collection", large_vector_set)

                # Should make multiple calls due to batching (batch_size = 100)
                assert mock_client.put.call_count >= 3

    async def test_distance_metric_conversion(self, backend):
        """Test distance metric conversion."""
        cosine_backend = backend._convert_distance_metric(DistanceMetric.COSINE)
        euclidean_backend = backend._convert_distance_metric(DistanceMetric.EUCLIDEAN)
        dot_backend = backend._convert_distance_metric(DistanceMetric.DOT_PRODUCT)

        assert cosine_backend == "Cosine"
        assert euclidean_backend == "Euclid"
        assert dot_backend == "Dot"

    async def test_connection_error_handling(self, backend):
        """Test connection error handling."""
        with patch.object(backend, '_test_connection') as mock_test:
            mock_test.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await backend._initialize_client()
```

## 🔍 Protocol Compliance Testing

### Provider Protocol Tests

```python
import pytest
from codeweaver.testing import ProtocolTestRunner
from codeweaver.providers.base import EmbeddingProvider
from your_extension.provider import MyEmbeddingProvider, MyProviderConfig

class TestProviderProtocolCompliance:
    """Test protocol compliance for providers."""

    @pytest.fixture
    def provider(self):
        config = MyProviderConfig(
            api_key="test-key",
            model_name="test-model"
        )
        return MyEmbeddingProvider(config)

    def test_embedding_provider_protocol(self, provider):
        """Test EmbeddingProvider protocol compliance."""
        runner = ProtocolTestRunner(EmbeddingProvider)

        # Check protocol implementation
        compliance_result = runner.check_protocol_compliance(provider)
        assert compliance_result.is_compliant

        # Check required methods exist
        assert hasattr(provider, 'embed_documents')
        assert hasattr(provider, 'embed_query')
        assert hasattr(provider, 'get_provider_info')
        assert hasattr(provider, 'health_check')

        # Check required properties
        assert hasattr(provider, 'provider_name')
        assert hasattr(provider, 'model_name')
        assert hasattr(provider, 'dimension')

    async def test_method_signatures(self, provider):
        """Test method signatures match protocol."""
        import inspect

        # Test embed_documents signature
        embed_docs_sig = inspect.signature(provider.embed_documents)
        params = list(embed_docs_sig.parameters.keys())
        assert 'texts' in params
        assert embed_docs_sig.return_annotation == list[list[float]]

        # Test embed_query signature
        embed_query_sig = inspect.signature(provider.embed_query)
        params = list(embed_query_sig.parameters.keys())
        assert 'text' in params
        assert embed_query_sig.return_annotation == list[float]

    async def test_property_types(self, provider):
        """Test property return types."""
        assert isinstance(provider.provider_name, str)
        assert isinstance(provider.model_name, str)
        assert isinstance(provider.dimension, int)
        assert provider.dimension > 0

        # Optional properties
        if hasattr(provider, 'max_batch_size') and provider.max_batch_size is not None:
            assert isinstance(provider.max_batch_size, int)
            assert provider.max_batch_size > 0

        if hasattr(provider, 'max_input_length') and provider.max_input_length is not None:
            assert isinstance(provider.max_input_length, int)
            assert provider.max_input_length > 0
```

### Plugin Protocol Tests

```python
from codeweaver.factories.plugin_protocols import ProviderPlugin
from your_extension.provider import MyProviderPlugin

class TestPluginProtocolCompliance:
    """Test plugin protocol compliance."""

    def test_provider_plugin_protocol(self):
        """Test ProviderPlugin protocol compliance."""
        plugin = MyProviderPlugin

        # Check required class methods exist
        assert hasattr(plugin, 'get_plugin_name')
        assert hasattr(plugin, 'get_component_type')
        assert hasattr(plugin, 'get_capabilities')
        assert hasattr(plugin, 'get_component_info')
        assert hasattr(plugin, 'validate_config')
        assert hasattr(plugin, 'get_dependencies')
        assert hasattr(plugin, 'get_provider_class')

        # Test method return types
        assert isinstance(plugin.get_plugin_name(), str)
        assert isinstance(plugin.get_dependencies(), list)

        # Test provider class inheritance
        provider_class = plugin.get_provider_class()
        # Should implement EmbeddingProvider protocol
        from codeweaver.providers.base import EmbeddingProvider
        assert hasattr(provider_class, 'embed_documents')
        assert hasattr(provider_class, 'embed_query')

    def test_configuration_validation(self):
        """Test configuration validation."""
        plugin = MyProviderPlugin

        # Valid configuration
        valid_config = MyProviderConfig(
            api_key="test-key",
            model_name="test-model"
        )
        result = plugin.validate_config(valid_config)
        assert result.is_valid
        assert not result.errors

        # Invalid configuration
        invalid_config = MyProviderConfig(
            api_key="",  # Empty API key
            model_name="test-model"
        )
        result = plugin.validate_config(invalid_config)
        assert not result.is_valid
        assert result.errors
```

## ⚡ Performance Testing

### Provider Performance Tests

```python
import time
import pytest
from codeweaver.testing import PerformanceTestRunner, create_test_documents
from your_extension.provider import MyEmbeddingProvider, MyProviderConfig

class TestProviderPerformance:
    """Test provider performance requirements."""

    @pytest.fixture
    def provider(self):
        config = MyProviderConfig(
            api_key=os.getenv("TEST_API_KEY"),
            model_name="test-model"
        )
        return MyEmbeddingProvider(config)

    @pytest.mark.performance
    async def test_embedding_latency(self, provider):
        """Test embedding latency requirements."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for performance testing")

        test_text = "This is a test document for performance testing."

        # Warmup
        await provider.embed_query(test_text)

        # Measure latency
        start_time = time.time()
        result = await provider.embed_query(test_text)
        end_time = time.time()

        latency = end_time - start_time

        # Assert latency requirements (adjust based on your SLA)
        assert latency < 2.0, f"Embedding latency too high: {latency:.2f}s"
        assert isinstance(result, list)
        assert len(result) == provider.dimension

    @pytest.mark.performance
    async def test_batch_throughput(self, provider):
        """Test batch embedding throughput."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for performance testing")

        # Create test documents
        documents = create_test_documents(count=50, length=100)

        # Measure throughput
        start_time = time.time()
        results = await provider.embed_documents(documents)
        end_time = time.time()

        duration = end_time - start_time
        throughput = len(documents) / duration

        # Assert throughput requirements
        assert throughput > 10, f"Throughput too low: {throughput:.2f} docs/sec"
        assert len(results) == len(documents)

    @pytest.mark.performance
    async def test_memory_usage(self, provider):
        """Test memory usage during embedding."""
        import psutil
        import os

        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for performance testing")

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss

        # Process large batch
        documents = create_test_documents(count=1000, length=500)
        await provider.embed_documents(documents)

        # Measure peak memory
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - baseline_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        max_memory_mb = 500 * 1024 * 1024  # 500MB
        assert memory_increase < max_memory_mb, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
```

### Backend Performance Tests

```python
class TestBackendPerformance:
    """Test backend performance requirements."""

    @pytest.fixture
    def backend(self):
        config = MyBackendConfig(
            host=os.getenv("TEST_BACKEND_HOST", "localhost"),
            port=int(os.getenv("TEST_BACKEND_PORT", "6333"))
        )
        return MyVectorBackend(config)

    @pytest.mark.performance
    async def test_search_latency(self, backend):
        """Test search latency requirements."""
        collection_name = f"perf_test_{int(time.time())}"

        try:
            # Setup test collection with data
            await backend.create_collection(collection_name, 128)

            test_vectors = create_test_vectors(count=1000, dimension=128)
            await backend.upsert_vectors(collection_name, test_vectors)

            # Measure search latency
            query_vector = [0.1] * 128

            # Warmup
            await backend.search_vectors(collection_name, query_vector, limit=10)

            # Measure latency
            start_time = time.time()
            results = await backend.search_vectors(collection_name, query_vector, limit=10)
            end_time = time.time()

            latency = end_time - start_time

            # Assert latency requirements
            assert latency < 0.1, f"Search latency too high: {latency:.3f}s"
            assert len(results) <= 10

        finally:
            await backend.delete_collection(collection_name)

    @pytest.mark.performance
    async def test_upsert_throughput(self, backend):
        """Test vector upsert throughput."""
        collection_name = f"perf_test_{int(time.time())}"

        try:
            await backend.create_collection(collection_name, 128)

            # Test batch upsert performance
            test_vectors = create_test_vectors(count=10000, dimension=128)

            start_time = time.time()
            await backend.upsert_vectors(collection_name, test_vectors)
            end_time = time.time()

            duration = end_time - start_time
            throughput = len(test_vectors) / duration

            # Assert throughput requirements
            assert throughput > 1000, f"Upsert throughput too low: {throughput:.2f} vectors/sec"

        finally:
            await backend.delete_collection(collection_name)
```

## 🔄 Integration Testing

### Full Workflow Integration Tests

```python
@pytest.mark.integration
class TestFullWorkflowIntegration:
    """Test complete workflows with real services."""

    @pytest.fixture
    def provider(self):
        config = MyProviderConfig(
            api_key=os.getenv("TEST_API_KEY"),
            model_name="test-model"
        )
        return MyEmbeddingProvider(config)

    @pytest.fixture
    def backend(self):
        config = MyBackendConfig(
            host=os.getenv("TEST_BACKEND_HOST", "localhost"),
            port=int(os.getenv("TEST_BACKEND_PORT", "6333"))
        )
        return MyVectorBackend(config)

    async def test_end_to_end_workflow(self, provider, backend):
        """Test complete embedding and search workflow."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for integration testing")

        collection_name = f"integration_test_{int(time.time())}"

        try:
            # Step 1: Create collection
            await backend.create_collection(collection_name, provider.dimension)

            # Step 2: Embed documents
            documents = [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning uses algorithms",
                "Databases store structured data"
            ]

            embeddings = await provider.embed_documents(documents)
            assert len(embeddings) == len(documents)

            # Step 3: Store vectors
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vector = VectorPoint(
                    id=str(i),
                    vector=embedding,
                    payload={"text": doc}
                )
                vectors.append(vector)

            await backend.upsert_vectors(collection_name, vectors)

            # Step 4: Search with query
            query = "programming languages"
            query_embedding = await provider.embed_query(query)

            results = await backend.search_vectors(
                collection_name,
                query_embedding,
                limit=2
            )

            # Step 5: Verify results
            assert len(results) == 2
            assert results[0].score > 0.5  # Should be relevant

            # The Python document should be most relevant
            top_result = results[0]
            assert "Python" in top_result.payload["text"]

        finally:
            await backend.delete_collection(collection_name)
```

## 📊 Test Utilities and Fixtures

### Common Test Fixtures

```python
# conftest.py
import pytest
import os
import random
from codeweaver.cw_types import VectorPoint

@pytest.fixture(scope="session")
def test_config():
    """Session-wide test configuration."""
    return {
        "api_key": os.getenv("TEST_API_KEY"),
        "backend_host": os.getenv("TEST_BACKEND_HOST", "localhost"),
        "backend_port": int(os.getenv("TEST_BACKEND_PORT", "6333")),
        "timeout": 30
    }

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Vector databases enable semantic search capabilities",
        "Natural language processing analyzes human language"
    ]

@pytest.fixture
def test_vectors():
    """Create test vectors for backend testing."""
    def _create_vectors(count: int = 10, dimension: int = 128):
        vectors = []
        for i in range(count):
            vector = VectorPoint(
                id=str(i),
                vector=[random.random() for _ in range(dimension)],
                payload={"text": f"test document {i}", "index": i}
            )
            vectors.append(vector)
        return vectors
    return _create_vectors

@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "embedding_response": {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        },
        "search_response": {
            "result": [
                {"id": "1", "score": 0.95, "payload": {"text": "test1"}},
                {"id": "2", "score": 0.85, "payload": {"text": "test2"}}
            ]
        }
    }
```

## 🚀 Running Tests

### Test Commands

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests (requires real services)
uv run pytest tests/integration/ -m integration

# Run performance tests (requires real services)
uv run pytest tests/performance/ -m performance

# Run protocol compliance tests
uv run pytest tests/protocol/

# Run with coverage
uv run pytest --cov=your_extension tests/

# Run specific test file
uv run pytest tests/unit/test_my_provider.py

# Run with verbose output
uv run pytest -v tests/

# Run tests in parallel (if pytest-xdist installed)
uv run pytest -n auto tests/
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Test Extensions

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install uv
        uv sync --group dev

    - name: Run unit tests
      run: uv run pytest tests/unit/ --cov=your_extension

    - name: Run integration tests
      env:
        TEST_BACKEND_HOST: localhost
        TEST_BACKEND_PORT: 6333
      run: uv run pytest tests/integration/ -m integration

    - name: Run protocol compliance tests
      run: uv run pytest tests/protocol/

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🚀 Next Steps

- **[Performance Guidelines :material-arrow-right-circle:](./performance.md)**: Optimization best practices
- **[Protocol Reference :material-arrow-right-circle:](../reference/protocols.md)**: Complete protocol documentation
- **[Examples :material-arrow-right-circle:](../examples/)**: Working extension examples
