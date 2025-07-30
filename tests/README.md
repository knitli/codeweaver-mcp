<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Test Suite

Comprehensive test suite for CodeWeaver's testing framework validation and all protocol implementations.

## Overview

This test suite validates:

- **Protocol Compliance Framework**: Tests the compliance validation system itself
- **Mock Implementations**: Validates mock implementations conform to protocols
- **Integration Testing Framework**: Tests the integration testing system
- **Performance Benchmarking Framework**: Tests the benchmarking system
- **Factory Pattern Validation**: Tests factory validation utilities

# Testing Guide for CodeWeaver

## Pytest Configuration

The project is configured with comprehensive pytest marks and settings in `pyproject.toml`. Here's how to use them effectively:

## Test Categories

### By Test Type
- `pytest -m unit` - Unit tests (test individual components)
- `pytest -m integration` - Integration tests (test component interactions)
- `pytest -m validation` - Validation tests (ensure system consistency)
- `pytest -m e2e` - End-to-end tests (complete workflows)

### By Feature Area
- `pytest -m config` - Configuration-related tests
- `pytest -m telemetry` - Telemetry and metrics tests
- `pytest -m embeddings` - Embedding functionality tests
- `pytest -m search` - Search functionality tests
- `pytest -m indexing` - Code indexing tests
- `pytest -m mcp` - MCP protocol tests
- `pytest -m services` - Services layer tests

### By Performance
- `pytest -m benchmark` - Performance benchmark tests
- `pytest -m performance` - All performance-related tests
- `pytest -m slow` - Tests that take significant time
- `pytest -m "not slow"` - Skip slow tests for faster feedback

### By Dependencies
- `pytest -m mock_only` - Tests using only mocked dependencies
- `pytest -m network` - Tests requiring network access
- `pytest -m external_api` - Tests calling external APIs
- `pytest -m voyageai` - Tests requiring VoyageAI API
- `pytest -m qdrant` - Tests requiring Qdrant database

## Common Test Commands

### Development Workflow
```bash
# Quick unit tests during development
pytest -m "unit and not slow"

# All unit tests with coverage
pytest -m unit --cov=codeweaver --cov-report=html

# Integration tests without external dependencies
pytest -m "integration and mock_only"

# Configuration tests only
pytest -m config

# All tests except slow benchmarks
pytest -m "not (slow and benchmark)"
```

### CI/CD Pipeline
```bash
# Fast feedback loop
pytest -m "unit and not external_api"

# Full test suite
pytest --cov=codeweaver --cov-fail-under=80

# Performance regression tests
pytest -m "benchmark and not external_api"
```

### Feature Development
```bash
# When working on telemetry
pytest -m telemetry

# When working on search functionality
pytest -m "search or embeddings"

# When working on MCP protocol
pytest -m mcp

# When working on services layer
pytest -m services
```

## Test Organization

### Directory Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for component interactions
- `tests/validation/` - System consistency validation tests

### Naming Conventions
- Test files: `test_*.py` or `*_test.py`
- Test classes: `TestSomething`
- Test methods: `test_something()`

## Coverage Configuration

Coverage is configured to:
- Require 80% minimum coverage (`--cov-fail-under=80`)
- Generate HTML reports in `htmlcov/`
- Generate XML reports for CI tools
- Show missing lines in terminal output

## Performance Testing

Benchmark tests are marked with `@pytest.mark.benchmark` and can be run with:
```bash
# All benchmarks
pytest -m benchmark

# Benchmarks without external APIs (for CI)
pytest -m "benchmark and not external_api"

# Only local performance tests
pytest -m "benchmark and mock_only"
```

## Environment Variables

Tests respect these environment variables:
- `CW_TELEMETRY_ENABLED=false` - Disable telemetry in tests
- `CW_NO_TELEMETRY=true` - Alternative way to disable telemetry
- Tests marked with `@pytest.mark.env_vars` depend on environment variables

## Async Testing

All async tests are automatically handled with `asyncio_mode = "auto"`. Tests marked with `@pytest.mark.async_test` are explicitly async tests.

## Timeout Configuration

- Default timeout: 300 seconds (5 minutes)
- Timeout method: `thread` (more reliable than `signal`)
- Override with `@pytest.mark.timeout(60)` for specific tests

## Flaky Test Handling

Tests that may occasionally fail should be marked with `@pytest.mark.flaky`. Consider using `pytest-flakefinder` to identify flaky tests:

```bash
# Find flaky tests
pytest --flake-finder --flake-runs=10
```

## Example Usage

```bash
# Development cycle: fast feedback
pytest -m "unit and not (external_api or slow)"

# Before commit: comprehensive local testing
pytest -m "not external_api" --cov=codeweaver

# CI pipeline: full suite with coverage
pytest --cov=codeweaver --cov-report=xml --cov-fail-under=80

# Performance testing: benchmarks only
pytest -m "benchmark and mock_only" --tb=short
```

## Debugging Tests

For debugging failing tests:
```bash
# Verbose output with full tracebacks
pytest -v --tb=long

# Stop on first failure
pytest -x

# Debug mode (drop into debugger on failure)
pytest --pdb

# Run specific test with debugging
pytest tests/unit/test_something.py::TestClass::test_method --pdb -s
```


## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/codeweaver/testing --cov-report=html

# Run specific test files
pytest tests/test_protocol_compliance.py -v
pytest tests/test_integration.py -v
pytest tests/test_benchmarks.py -v
pytest tests/test_factory_validation.py -v
```

### Run Tests by Category

```bash
# Protocol compliance tests
pytest tests/test_protocol_compliance.py::TestProtocolComplianceValidator -v

# Integration tests
pytest tests/test_integration.py::TestIntegrationTestSuite -v

# Benchmark tests
pytest tests/test_benchmarks.py::TestBenchmarkSuite -v

# Factory validation tests
pytest tests/test_factory_validation.py::TestFactoryPatternValidator -v
```

### Run Specific Test Cases

```bash
# Test mock implementations
pytest tests/test_protocol_compliance.py::TestVectorBackendCompliance::test_mock_vector_backend_compliance -v

# Test integration workflows
pytest tests/test_integration.py::TestWorkflowTests::test_embedding_workflow -v

# Test performance measurements
pytest tests/test_benchmarks.py::TestBenchmarkSuite::test_benchmark_vector_backend -v

# Test factory patterns
pytest tests/test_factory_validation.py::TestFactoryPatternValidator::test_validate_backend_factory -v
```

## Test Files

### `test_protocol_compliance.py`

Tests the protocol compliance validation framework:

- **TestProtocolComplianceValidator**: Core validator functionality
- **TestVectorBackendCompliance**: VectorBackend protocol validation
- **TestHybridSearchBackendCompliance**: HybridSearchBackend protocol validation
- **TestEmbeddingProviderCompliance**: EmbeddingProvider protocol validation
- **TestRerankProviderCompliance**: RerankProvider protocol validation
- **TestDataSourceCompliance**: DataSource protocol validation
- **TestComplianceResultFormatting**: Result formatting and reporting
- **TestErrorHandling**: Error handling in validation

Key test scenarios:
```python
# Test that mock implementations are compliant
async def test_mock_vector_backend_compliance():
    backend = MockVectorBackend()
    result = await validate_vector_backend_protocol(backend)
    assert result.is_compliant

# Test protocol interface validation
async def test_embedding_provider_properties():
    provider = MockEmbeddingProvider(dimension=256)
    result = await validate_embedding_provider_protocol(provider)
    assert result.is_compliant
    assert provider.dimension == 256
```

### `test_integration.py`

Tests the integration testing framework:

- **TestTestConfiguration**: Test configuration management
- **TestIntegrationTestSuite**: Core integration test functionality
- **TestWorkflowTests**: Individual workflow testing
- **TestIntegrationTestExecution**: Full test execution
- **TestConvenienceFunctions**: Helper functions
- **TestIntegrationTestResult**: Result formatting
- **TestErrorHandling**: Error handling in integration tests

Key test scenarios:
```python
# Test complete integration workflow
async def test_run_all_tests_success():
    config = create_test_configuration(mock_latency_ms=1.0)
    suite = IntegrationTestSuite(config)
    result = await suite.run_all_tests()
    assert result.success is True

# Test individual workflow components
async def test_embedding_workflow():
    config = create_test_configuration()
    suite = IntegrationTestSuite(config)
    await suite._setup_test_environment()
    success = await suite._test_embedding_workflow()
    assert success is True
```

### `test_benchmarks.py`

Tests the performance benchmarking framework:

- **TestBenchmarkResult**: Benchmark result data structure
- **TestBenchmarkSuite**: Core benchmarking functionality
- **TestBenchmarkScenarios**: Custom benchmark scenarios
- **TestConvenienceFunctions**: Helper functions
- **TestPerformanceRegression**: Performance regression detection

Key test scenarios:
```python
# Test benchmark execution
async def test_benchmark_vector_backend():
    suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3)
    backend = MockVectorBackend(latency_ms=1.0)
    results = await suite.benchmark_vector_backend(backend)
    assert len(results) > 0
    for result in results:
        assert result.operations_per_second >= 0

# Test performance regression detection
async def test_performance_scales_with_latency():
    fast_provider = MockEmbeddingProvider(latency_ms=1.0)
    slow_provider = MockEmbeddingProvider(latency_ms=50.0)
    # Fast should have higher ops/sec than slow
```

### `test_factory_validation.py`

Tests the factory pattern validation:

- **TestFactoryValidationResult**: Validation result data structure
- **TestFactoryPatternValidator**: Core factory validation
- **TestConvenienceFunctions**: Helper functions
- **TestErrorHandling**: Error handling in factory validation
- **TestFactoryIntegration**: Integration scenarios

Key test scenarios:
```python
# Test factory validation
async def test_validate_backend_factory():
    validator = FactoryPatternValidator()
    result = await validator.validate_backend_factory()
    assert result.factory_name == "BackendFactory"
    if result.is_valid:
        assert result.created_instances > 0

# Test configuration integration
async def test_validate_configuration_integration():
    validator = FactoryPatternValidator()
    result = await validator.validate_configuration_integration()
    assert isinstance(result, FactoryValidationResult)
```

## Test Configuration

### Environment Variables

Tests can be configured via environment variables:

```bash
# Test timeouts
export TEST_TIMEOUT_SECONDS=120

# Mock settings
export MOCK_LATENCY_MS=5.0
export MOCK_ERROR_RATE=0.1

# Test data
export TEST_ITERATIONS=5
```

### Pytest Configuration

Key pytest settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
]
markers = [
    "asyncio: marks tests as async",
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test that takes a long time."""
    pass

@pytest.mark.integration
async def test_integration_scenario():
    """Integration test."""
    pass
```

Run specific marker categories:

```bash
# Run only fast tests (exclude slow)
pytest -m "not slow" tests/

# Run only integration tests
pytest -m "integration" tests/

# Run only async tests
pytest -m "asyncio" tests/
```

## Test Utilities

### Mock Configurations

```python
# Fast configuration for quick tests
FAST_CONFIG = create_test_configuration(
    run_performance_tests=False,
    mock_latency_ms=0.1,
    test_timeout_seconds=10
)

# Comprehensive configuration for thorough testing
COMPREHENSIVE_CONFIG = create_test_configuration(
    run_compliance_tests=True,
    run_performance_tests=True,
    run_workflow_tests=True,
    mock_latency_ms=5.0,
    test_timeout_seconds=60
)
```

### Test Fixtures

Common pytest fixtures:

```python
@pytest.fixture
async def mock_backend():
    """Create mock vector backend for testing."""
    return MockVectorBackend(latency_ms=1.0)

@pytest.fixture
async def mock_provider():
    """Create mock embedding provider for testing."""
    return MockEmbeddingProvider(dimension=128, latency_ms=1.0)

@pytest.fixture
def test_config():
    """Create test configuration."""
    return create_test_configuration(mock_latency_ms=1.0)
```

### Helper Functions

```python
async def assert_protocol_compliance(implementation, protocol_validator):
    """Assert that implementation is protocol compliant."""
    result = await protocol_validator(implementation)
    assert result.is_compliant, f"Protocol compliance failed: {result.validation_errors}"

async def assert_performance_threshold(component, min_ops_per_second=10.0):
    """Assert that component meets performance threshold."""
    suite = BenchmarkSuite(benchmark_iterations=3)
    results = await suite.benchmark_vector_backend(component)
    avg_ops = sum(r.operations_per_second for r in results) / len(results)
    assert avg_ops >= min_ops_per_second, f"Performance below threshold: {avg_ops}"
```

## Continuous Integration

### GitHub Actions

Example CI configuration:

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev

    - name: Run fast tests
      run: |
        uv run pytest tests/ -m "not slow" --cov=src/codeweaver/testing

    - name: Run integration tests
      run: |
        uv run pytest tests/ -m "integration" --timeout=300

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Coverage

Maintain high test coverage:

```bash
# Generate coverage report
pytest tests/ --cov=src/codeweaver/testing --cov-report=html --cov-report=term

# Coverage requirements
# - Protocol compliance: >95%
# - Mock implementations: >90%
# - Integration framework: >85%
# - Benchmarking framework: >80%
```

## Debugging Tests

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run specific test with debug output
pytest tests/test_integration.py::test_specific_function -v -s
```

### Test Isolation

Ensure tests are isolated:

```python
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup code here
    await cleanup_test_environment()
```

### Performance Debugging

For performance issues:

```python
# Use minimal configurations
config = create_test_configuration(
    run_performance_tests=False,
    mock_latency_ms=0.1,
    test_timeout_seconds=5
)

# Profile specific tests
pytest tests/test_benchmarks.py --profile
```

## Contributing

### Adding New Tests

When adding new functionality:

1. **Add Protocol Tests**: Test protocol compliance
2. **Add Mock Tests**: Test mock implementations
3. **Add Integration Tests**: Test end-to-end workflows
4. **Add Benchmark Tests**: Test performance characteristics
5. **Add Factory Tests**: Test configuration and instantiation

### Test Structure

Follow consistent test structure:

```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_initialization(self):
        """Test feature initialization."""
        pass

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic feature functionality."""
        pass

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test feature error handling."""
        pass

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance(self):
        """Test feature performance."""
        pass
```

### Test Documentation

Document test purpose and scenarios:

```python
def test_complex_scenario(self):
    """Test complex scenario with multiple components.

    This test validates:
    1. Component initialization
    2. Inter-component communication
    3. Error recovery
    4. Performance characteristics

    Expected behavior:
    - All components should initialize successfully
    - Communication should be reliable
    - Errors should be handled gracefully
    - Performance should meet requirements
    """
    pass
```
