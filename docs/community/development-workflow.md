<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Development Workflow

This guide covers the complete development workflow for CodeWeaver contributors, from setting up your environment to getting your changes merged.

## 🛠️ Environment Setup

### Prerequisites

**Required Tools:**
- **Python 3.11+** - Modern Python with latest typing features
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager and virtual environment manager
- **Git** - Version control

**Optional but Recommended:**
- **[ast-grep](https://ast-grep.github.io/)** - For structural code search features
- **Docker** - For testing with real vector databases
- **VS Code** or **PyCharm** - IDEs with good Python support

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp

# Install all dependencies (production + development)
uv sync --group dev

# Activate the virtual environment
source .venv/bin/activate

# Verify installation
uv run python -c "import codeweaver; print('✅ CodeWeaver installed successfully')"
```

### Development Dependencies

The project includes several development tool groups:

```bash
# Install all development dependencies
uv sync --group dev

# Or install specific groups
uv sync --group test      # Testing tools (pytest, coverage)
uv sync --group lint      # Linting and formatting (ruff)
uv sync --group docs      # Documentation tools
uv sync --group bench     # Benchmarking tools
```

### IDE Configuration

**VS Code Settings (`.vscode/settings.json`):**
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true
    }
}
```

**PyCharm Configuration:**
- Set Python interpreter to `.venv/bin/python`
- Enable pytest as test runner
- Configure ruff as external tool for linting

## 🏗️ Project Structure

Understanding the codebase structure helps you navigate and contribute effectively:

```
codeweaver-mcp/
├── src/codeweaver/              # Main source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Server entry point
│   ├── server.py                # MCP server implementation
│   ├── config.py                # Configuration management
│   ├── _types/                  # Type definitions and protocols
│   ├── factories/               # Plugin factory system
│   ├── providers/               # Embedding/reranking providers
│   ├── backends/                # Vector database backends
│   ├── sources/                 # Data source integrations
│   ├── services/                # Service layer architecture
│   ├── middleware/              # FastMCP middleware
│   ├── client/                  # Client utilities
│   └── testing/                 # Testing utilities (not tests)
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── validation/              # Architecture validation
├── docs/                        # Documentation
├── examples/                    # Usage examples
└── pyproject.toml              # Project configuration
```

### Key Extension Points

**Where to add new functionality:**

1. **New Providers** → `src/codeweaver/providers/your_provider.py`
2. **New Backends** → `src/codeweaver/backends/your_backend.py`
3. **New Sources** → `src/codeweaver/sources/your_source.py`
4. **New Services** → `src/codeweaver/services/providers/your_service.py`
5. **Type Definitions** → `src/codeweaver/_types/`

## 🔄 Development Cycle

### 1. Planning Your Work

**Before You Start:**
- Check existing [issues](https://github.com/knitli/codeweaver-mcp/issues) and [discussions](https://github.com/knitli/codeweaver-mcp/discussions)
- Review the [architecture documentation](../architecture/index.md)
- Understand the [development patterns](development_patterns.md)

**For New Features:**
```bash
# Create a feature branch
git checkout -b feature/add-openai-provider

# For bug fixes
git checkout -b fix/qdrant-connection-timeout

# For documentation
git checkout -b docs/update-provider-guide
```

### 2. Development Process

**Code Development:**
```bash
# Make your changes
# Follow established patterns from existing code

# Run tests frequently during development
uv run pytest tests/unit/ -v

# Check code style
uv run ruff check
uv run ruff format

# Run specific test file
uv run pytest tests/unit/test_providers.py -v
```

**Testing Your Changes:**
```bash
# Unit tests (fast, isolated)
uv run pytest tests/unit/ -v

# Integration tests (slower, real components)
uv run pytest tests/integration/ -v

# Test specific functionality
uv run pytest -k "test_provider" -v

# Run with coverage
uv run pytest --cov=codeweaver tests/
```

### 3. Quality Assurance

**Automated Checks:**
```bash
# Linting and formatting
uv run ruff check                    # Check for issues
uv run ruff check --fix              # Auto-fix issues
uv run ruff format                   # Format code

# Type checking (if mypy is available)
uv run mypy src/codeweaver/

# Security scanning
uv run bandit -r src/codeweaver/
```

**Manual Testing:**
```bash
# Test the server manually
export CW_EMBEDDING_API_KEY="your-key"
export CW_VECTOR_BACKEND_URL="your-url"
uv run codeweaver

# Test specific functionality
uv run python tests/integration/test_server_functionality.py /path/to/test/code
```

### 4. Documentation

**Update Documentation:**
- Add docstrings to new functions and classes
- Update relevant user guides
- Add examples for new features
- Update configuration documentation

**Documentation Format:**
```python
def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
    """Generate embeddings for the given texts.

    This method creates embedding vectors for the provided texts using the
    configured model. It integrates with the services layer for caching,
    rate limiting, and metrics collection.

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
        ServiceUnavailableError: If the API is unavailable
        ProviderError: For other API or processing errors

    Example:
        ```python
        provider = OpenAIProvider(config)
        embeddings = await provider.embed_documents(
            ["Hello world", "How are you?"],
            context
        )
        print(f"Generated {len(embeddings)} embeddings")
        ```
    """
```

### 5. Commit and Push

**Commit Messages:**
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Good commit messages
git commit -m "feat(providers): add OpenAI provider with GPT-4 embeddings"
git commit -m "fix(backends): handle Qdrant connection timeout gracefully"
git commit -m "docs(api): add configuration examples for new providers"
git commit -m "test(services): improve test coverage for chunking service"

# Commit with detailed description
git commit -m "feat(providers): add OpenAI provider support

- Implement OpenAI embeddings API integration
- Add support for text-embedding-3-small and text-embedding-3-large
- Include rate limiting and error handling
- Add comprehensive test suite
- Update configuration documentation

Closes #123"
```

**Push Your Changes:**
```bash
# Push feature branch
git push origin feature/add-openai-provider

# Or for first push
git push -u origin feature/add-openai-provider
```

## 🧪 Testing Strategy

### Test Types and When to Use Them

**Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1s per test)
- **Scope**: Single class or function
- **Dependencies**: Mocked

```python
# Example unit test
async def test_provider_embed_documents_success(self, mock_provider):
    """Test successful embedding generation."""
    mock_provider._api_call = AsyncMock(return_value={"embeddings": [[0.1, 0.2]]})

    result = await mock_provider.embed_documents(["test"])

    assert len(result) == 1
    assert len(result[0]) == 2
```

**Integration Tests** (`tests/integration/`)
- **Purpose**: Test component interactions
- **Speed**: Moderate (1-10s per test)
- **Scope**: Multiple components working together
- **Dependencies**: Real services when possible

```python
# Example integration test
@pytest.mark.integration
async def test_provider_backend_workflow():
    """Test complete provider to backend workflow."""
    provider = VoyageProvider(config)
    backend = QdrantBackend(config)

    # Generate embeddings
    embeddings = await provider.embed_documents(["test document"])

    # Store in backend
    await backend.store_vectors([
        VectorPoint(id="1", vector=embeddings[0], metadata={"text": "test document"})
    ])

    # Search and verify
    results = await backend.search(embeddings[0], limit=1)
    assert len(results) == 1
    assert results[0].metadata["text"] == "test document"
```

**Validation Tests** (`tests/validation/`)
- **Purpose**: Verify architecture and pattern compliance
- **Speed**: Fast to moderate
- **Scope**: System-wide validation
- **Dependencies**: Static analysis

### Test Organization

**Naming Conventions:**
```python
class TestProviderName:
    """Test suite for ProviderName."""

    def test_method_name_success(self):
        """Test successful method execution."""
        pass

    def test_method_name_error_condition(self):
        """Test method handles error condition properly."""
        pass

    async def test_async_method_success(self):
        """Test successful async method execution."""
        pass
```

**Test Fixtures:**
```python
@pytest.fixture
def provider_config():
    """Standard provider configuration for testing."""
    return ProviderConfig(
        api_key="test-key",
        model="test-model",
        max_batch_size=10
    )

@pytest.fixture
def mock_provider(provider_config):
    """Mock provider instance for testing."""
    provider = Provider(provider_config)
    provider._api_call = AsyncMock()
    return provider

@pytest.fixture
def mock_context():
    """Mock service context for testing."""
    return {
        "rate_limiting_service": AsyncMock(),
        "caching_service": AsyncMock(),
        "metrics_service": AsyncMock(),
    }
```

### Running Tests Efficiently

**During Development:**
```bash
# Run tests for your changes only
uv run pytest tests/unit/test_your_module.py -v

# Run specific test function
uv run pytest tests/unit/test_providers.py::TestVoyageProvider::test_embed_success -v

# Run tests matching pattern
uv run pytest -k "test_provider" -v

# Stop on first failure
uv run pytest -x

# Run in parallel (if pytest-xdist installed)
uv run pytest -n auto
```

**Before Submitting PR:**
```bash
# Full test suite
uv run pytest

# With coverage report
uv run pytest --cov=codeweaver --cov-report=html

# Integration tests (slower)
uv run pytest tests/integration/ -v

# Validation tests
uv run python tests/validation/validate_architecture.py
```

## 🔍 Debugging and Troubleshooting

### Common Development Issues

**Import Errors:**
```bash
# Ensure proper installation
uv sync

# Check Python path
python -c "import sys; print(sys.path)"

# Verify package installation
python -c "import codeweaver; print(codeweaver.__file__)"
```

**Test Failures:**
```bash
# Run single test with verbose output
uv run pytest tests/unit/test_failing.py::test_function -v -s

# Debug with pdb
uv run pytest tests/unit/test_failing.py::test_function --pdb

# Show local variables on failure
uv run pytest tests/unit/test_failing.py::test_function -l
```

**Linting Issues:**
```bash
# Show all linting issues
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Check specific file
uv run ruff check src/codeweaver/providers/your_provider.py
```

### Debugging Strategies

**Use Logging:**
```python
import logging
logger = logging.getLogger(__name__)

async def your_method(self):
    logger.debug("Starting method execution")
    try:
        result = await self.api_call()
        logger.info("API call successful")
        return result
    except Exception as e:
        logger.exception("API call failed")
        raise
```

**Add Type Checking:**
```python
# Use runtime type checking for debugging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Add development-only imports
    pass

# Use assert statements for debugging
assert isinstance(result, list), f"Expected list, got {type(result)}"
assert len(result) > 0, "Result should not be empty"
```

**Profile Performance:**
```python
import time
import cProfile

# Simple timing
start = time.time()
result = await expensive_operation()
print(f"Operation took {time.time() - start:.2f}s")

# Detailed profiling
cProfile.run("your_function()", "profile_output.prof")
```

## 📊 Performance Considerations

### Optimization Guidelines

**For Providers:**
- Batch API calls when possible
- Implement proper rate limiting
- Use async/await for I/O operations
- Cache results when appropriate

**For Backends:**
- Connection pooling for databases
- Batch vector operations
- Efficient query construction
- Proper index configuration

**For Services:**
- Lazy initialization
- Resource cleanup
- Health check optimization
- Metrics collection efficiency

### Benchmarking

**Create Benchmarks:**
```python
import time
import asyncio
from codeweaver.providers.your_provider import YourProvider

async def benchmark_provider():
    """Benchmark provider performance."""
    provider = YourProvider(config)

    # Warm up
    await provider.embed_documents(["warmup"])

    # Benchmark
    texts = ["test text"] * 100
    start = time.time()

    results = await provider.embed_documents(texts)

    duration = time.time() - start
    throughput = len(texts) / duration

    print(f"Processed {len(texts)} texts in {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} texts/second")

if __name__ == "__main__":
    asyncio.run(benchmark_provider())
```

**Run Existing Benchmarks:**
```bash
# Run performance benchmarks
uv run python tests/integration/test_benchmarks.py

# Profile specific operations
uv run python -m cProfile -o profile.prof your_benchmark.py
```

## 🚀 Advanced Workflows

### Working with Multiple Components

**Cross-Component Changes:**
```bash
# Create feature branch
git checkout -b feature/improve-error-handling

# Make changes across providers, backends, services
# Update tests and documentation
# Run comprehensive tests

uv run pytest
uv run python tests/validation/validate_architecture.py
```

**Breaking Changes:**
```bash
# Create feature branch
git checkout -b breaking/update-provider-interface

# Update interface definitions in _types/
# Update all implementations
# Update tests
# Add migration guide
# Update version in pyproject.toml
```

### Working with External Dependencies

**Adding New Dependencies:**
```bash
# Add production dependency
uv add new-package

# Add development dependency
uv add --group dev new-dev-tool

# Add optional dependency
uv add --optional new-optional-package
```

**Testing with Real Services:**
```bash
# Set up test environment variables
export TEST_VOYAGE_API_KEY="your-key"
export TEST_QDRANT_URL="http://localhost:6333"

# Run integration tests
uv run pytest tests/integration/ -m "not slow"

# Run all tests including slow ones
uv run pytest tests/integration/
```

### Release Preparation

**Pre-Release Checklist:**
```bash
# Run full test suite
uv run pytest

# Check code quality
uv run ruff check
uv run ruff format --check

# Validate architecture
uv run python tests/validation/validate_architecture.py

# Update version
# Update CHANGELOG.md
# Update documentation

# Build package
uv build

# Test installation
uv pip install dist/codeweaver-*.whl
```

## 🤝 Collaboration

### Working with Others

**Code Reviews:**
- Be respectful and constructive
- Explain the reasoning behind suggestions
- Ask questions to understand context
- Test reviewers' changes locally when possible

**Pair Programming:**
```bash
# Share work-in-progress
git add .
git commit -m "wip: add provider skeleton"
git push origin feature/branch-name

# Collaborate on same branch
git pull origin feature/branch-name
# Make changes
git push origin feature/branch-name
```

**Resolving Conflicts:**
```bash
# Keep branch updated
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main

# Or use rebase for cleaner history
git rebase main

# Push updated branch
git push --force-with-lease origin feature/your-branch
```

### Communication

**GitHub Issues:**
- Reference issues in commits: `Closes #123`
- Use issue templates
- Provide clear reproduction steps
- Include environment details

**Pull Requests:**
- Use PR templates
- Link to related issues
- Explain the change and motivation
- Include testing instructions

**Discussions:**
- Use for questions and design discussions
- Search existing discussions first
- Provide context and examples
- Be patient for responses

---

## 🎯 Next Steps

Now that you understand the development workflow:

1. **[Review Extension Guidelines](extension-guidelines.md)** - Learn how to build specific types of extensions
2. **[Study Code Review Process](code-review.md)** - Understand our quality standards
3. **[Explore Architecture Documentation](../architecture/index.md)** - Deep dive into system design
4. **[Check Development Patterns](development_patterns.md)** - Learn coding standards

Ready to start coding? Pick an issue from our [good first issue](https://github.com/knitli/codeweaver-mcp/labels/good%20first%20issue) label and follow this workflow!

*Questions about the development process? Ask in [GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions) or [email us](mailto:adam@knit.li).*
