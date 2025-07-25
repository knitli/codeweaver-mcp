# CodeWeaver Testing Framework

Comprehensive testing framework for protocol compliance validation, mock implementations, and integration testing of CodeWeaver's extensible architecture.

## Overview

The CodeWeaver testing framework provides:

- **Protocol Compliance Validation**: Ensures all implementations conform to their respective interfaces
- **Mock Implementations**: Realistic mocks for testing without external dependencies
- **Integration Testing**: End-to-end workflow validation
- **Performance Benchmarking**: Performance measurement and regression detection
- **Factory Pattern Validation**: Validates configuration loading and component instantiation

## Quick Start

```python
import asyncio
from codeweaver.testing import (
    run_integration_tests,
    run_performance_benchmarks,
    validate_all_factory_patterns,
)

# Run quick integration test
async def main():
    # Integration testing
    result = await run_integration_tests()
    print(result)

    # Performance benchmarking
    from codeweaver.testing.mocks import MockVectorBackend, MockEmbeddingProvider
    components = {
        "vector_backend": MockVectorBackend(),
        "CW_EMBEDDING_PROVIDER": MockEmbeddingProvider(),
    }
    bench_results = await run_performance_benchmarks(components)

    # Factory validation
    factory_results = await validate_all_factory_patterns()

asyncio.run(main())
```

## Components

### Protocol Compliance Validation

Validates that implementations conform to their protocol interfaces:

```python
from codeweaver.testing import (
    validate_vector_backend_protocol,
    validate_CW_EMBEDDING_PROVIDER_protocol,
)
from codeweaver.testing.mocks import MockVectorBackend, MockEmbeddingProvider

async def test_compliance():
    # Test vector backend compliance
    backend = MockVectorBackend()
    result = await validate_vector_backend_protocol(backend)

    if result.is_compliant:
        print(f"✅ {result}")
    else:
        print(f"❌ {result}")
        for error in result.validation_errors:
            print(f"  - {error}")

    # Test embedding provider compliance
    provider = MockEmbeddingProvider()
    result = await validate_CW_EMBEDDING_PROVIDER_protocol(provider)

    print(result.get_detailed_report())
```

### Mock Implementations

Realistic mock implementations for all protocols:

```python
from codeweaver.testing.mocks import (
    MockVectorBackend,
    MockHybridSearchBackend,
    MockEmbeddingProvider,
    MockRerankProvider,
    MockDataSource,
)

# Vector backend with configurable latency and error rate
backend = MockVectorBackend(latency_ms=10.0, error_rate=0.1)

# Hybrid search backend
hybrid_backend = MockHybridSearchBackend(latency_ms=15.0)

# Embedding provider with custom dimensions
provider = MockEmbeddingProvider(
    provider_name="test_provider",
    model_name="test_model",
    dimension=256,
    latency_ms=50.0
)

# Rerank provider
reranker = MockRerankProvider(latency_ms=30.0)

# Data source
source = MockDataSource(latency_ms=20.0)
```

### Integration Testing

End-to-end workflow validation:

```python
from codeweaver.testing.integration import (
    IntegrationTestSuite,
    TestConfiguration,
    create_test_configuration,
)

# Create custom test configuration
config = create_test_configuration(
    backend_type="mock",
    CW_EMBEDDING_PROVIDER="mock",
    rerank_provider="mock",
    run_performance_tests=False,
    mock_latency_ms=5.0
)

# Run integration tests
suite = IntegrationTestSuite(config)
result = await suite.run_all_tests()

print(result.get_detailed_report())
```

### Performance Benchmarking

Performance measurement and benchmarking:

```python
from codeweaver.testing.benchmarks import (
    BenchmarkSuite,
    run_performance_benchmarks,
    print_benchmark_results,
)

# Create benchmark suite
suite = BenchmarkSuite(
    warmup_iterations=3,
    benchmark_iterations=10,
    timeout_seconds=60,
    measure_resources=True
)

# Benchmark vector backend
backend = MockVectorBackend(latency_ms=1.0)
results = await suite.benchmark_vector_backend(backend)

# Print results
for result in results:
    print(f"{result.benchmark_name}: {result.operations_per_second:.2f} ops/sec")

# Benchmark multiple components
components = {
    "vector_backend": MockVectorBackend(),
    "CW_EMBEDDING_PROVIDER": MockEmbeddingProvider(),
}

all_results = await run_performance_benchmarks(components)
print_benchmark_results(all_results)
```

### Factory Pattern Validation

Validates factory patterns and configuration integration:

```python
from codeweaver.testing.factory_validation import (
    FactoryPatternValidator,
    validate_all_factory_patterns,
    print_factory_validation_results,
)

# Validate all factory patterns
results = await validate_all_factory_patterns()
print_factory_validation_results(results)

# Validate specific factory
from codeweaver.testing.factory_validation import validate_factory_pattern
result = await validate_factory_pattern("backend")

if result.is_valid:
    print(f"✅ Backend factory validation passed")
else:
    print(f"❌ Backend factory validation failed")
    print(result.get_detailed_report())
```

## Testing Scenarios

### Protocol Compliance Testing

```python
from codeweaver.testing.protocol_compliance import ProtocolComplianceValidator

# Create validator with custom settings
validator = ProtocolComplianceValidator(
    strict_mode=True,
    performance_benchmarks=True
)

# Test multiple protocols
implementations = {
    "vector_backend": MockVectorBackend(),
    "CW_EMBEDDING_PROVIDER": MockEmbeddingProvider(),
    "rerank_provider": MockRerankProvider(),
    "data_source": MockDataSource(),
}

results = await validator.validate_all_protocols(implementations)

for protocol_name, result in results.items():
    print(f"{protocol_name}: {result}")
```

### Custom Integration Tests

```python
from codeweaver.testing.integration import IntegrationTestSuite

class CustomIntegrationTests(IntegrationTestSuite):
    async def _test_custom_workflow(self) -> bool:
        """Custom workflow test."""
        try:
            # Custom test logic here
            return True
        except Exception:
            return False

# Use custom test suite
config = create_test_configuration()
suite = CustomIntegrationTests(config)
result = await suite.run_all_tests()
```

### Performance Regression Testing

```python
from codeweaver.testing.benchmarks import BenchmarkSuite

async def test_performance_regression():
    """Test for performance regressions."""
    suite = BenchmarkSuite(benchmark_iterations=5)

    # Baseline measurements
    baseline_backend = MockVectorBackend(latency_ms=1.0)
    baseline_results = await suite.benchmark_vector_backend(baseline_backend)

    # New implementation measurements
    new_backend = MockVectorBackend(latency_ms=2.0)  # Simulated regression
    new_results = await suite.benchmark_vector_backend(new_backend)

    # Compare performance
    for baseline, new in zip(baseline_results, new_results):
        ratio = new.operations_per_second / baseline.operations_per_second
        if ratio < 0.8:  # 20% regression threshold
            print(f"⚠️ Performance regression detected: {ratio:.2f}x slower")
        else:
            print(f"✅ Performance maintained: {ratio:.2f}x")
```

## Configuration

### Test Configuration

```python
from codeweaver.testing.integration import TestConfiguration

config = TestConfiguration(
    # Component selection
    backend_type="mock",  # or "qdrant", "pinecone", etc.
    CW_EMBEDDING_PROVIDER="mock",  # or "voyage", "openai", etc.
    rerank_provider="mock",  # or "voyage", "cohere", etc.
    data_source_type="mock",  # or "filesystem", "git", etc.

    # Test settings
    run_compliance_tests=True,
    run_performance_tests=True,
    run_workflow_tests=True,
    test_timeout_seconds=60,

    # Mock settings
    mock_latency_ms=10.0,
    mock_error_rate=0.0,

    # Test data
    test_documents=[
        "Custom test document 1",
        "Custom test document 2",
    ],
    test_queries=["custom query"],

    # Configuration overrides
    config_overrides={
        "backend": {"batch_size": 50},
        "provider": {"dimension": 512},
    }
)
```

### Benchmark Configuration

```python
from codeweaver.testing.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(
    warmup_iterations=3,      # Warmup before benchmarking
    benchmark_iterations=10,  # Number of benchmark runs
    timeout_seconds=60,       # Timeout per operation
    measure_resources=True    # Measure memory/CPU usage
)

# Custom benchmark scenarios
custom_scenarios = [
    {
        "name": "large_scale",
        "vector_count": 10000,
        "dimension": 1024,
        "operations": ["upsert", "search", "batch_search"]
    }
]

results = await suite.benchmark_vector_backend(backend, custom_scenarios)
```

## Best Practices

### 1. Use Mock Implementations for Unit Tests

```python
# Good: Fast, reliable, no external dependencies
from codeweaver.testing.mocks import MockVectorBackend

async def test_search_functionality():
    backend = MockVectorBackend(latency_ms=1.0)
    # Test your code that uses the backend
```

### 2. Use Integration Tests for End-to-End Validation

```python
# Good: Tests complete workflows
from codeweaver.testing.integration import run_integration_tests

async def test_complete_workflow():
    config = create_test_configuration(
        run_performance_tests=False  # Skip for faster execution
    )
    result = await run_integration_tests(config)
    assert result.success
```

### 3. Use Performance Tests for Regression Detection

```python
# Good: Catches performance regressions
from codeweaver.testing.benchmarks import run_performance_benchmarks

async def test_performance():
    components = {"vector_backend": YourBackend()}
    results = await run_performance_benchmarks(components)

    # Assert performance requirements
    for result_list in results.values():
        for result in result_list:
            assert result.operations_per_second > 100  # Your threshold
```

### 4. Use Protocol Compliance for Interface Validation

```python
# Good: Ensures implementations follow protocols
from codeweaver.testing import validate_vector_backend_protocol

async def test_custom_backend_compliance():
    backend = YourCustomBackend()
    result = await validate_vector_backend_protocol(backend)

    assert result.is_compliant
    assert len(result.validation_errors) == 0
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Timeout Errors**: Increase timeout_seconds in configuration
3. **Memory Issues**: Reduce batch sizes or iteration counts
4. **Mock Registration**: Ensure mock factories are registered

### Debug Mode

Enable debug logging for detailed test information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with debug output
result = await run_integration_tests()
```

### Performance Issues

For slow tests, use reduced configurations:

```python
config = create_test_configuration(
    run_performance_tests=False,
    mock_latency_ms=1.0,
    test_timeout_seconds=30
)
```

## Examples

See the `tests/` directory for comprehensive examples of:

- Protocol compliance testing (`test_protocol_compliance.py`)
- Integration testing (`test_integration.py`)
- Performance benchmarking (`test_benchmarks.py`)
- Factory validation (`test_factory_validation.py`)

## Contributing

When adding new protocols or implementations:

1. Add corresponding mock implementations
2. Add protocol compliance tests
3. Add integration test scenarios
4. Add performance benchmarks
5. Update factory validation if needed

The testing framework should comprehensively cover all new functionality.
