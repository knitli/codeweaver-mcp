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
