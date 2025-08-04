<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Contributing to CodeWeaver

Thank you for your interest in contributing to CodeWeaver! This guide will help you get started with contributing to our extensible MCP server for semantic code search.

## 🎯 Before You Start

### Read the Contributor License Agreement
By contributing to CodeWeaver, you agree to our [Contributor License Agreement](../../CONTRIBUTORS_LICENSE_AGREEMENT.md). It's developer-friendly and ensures your contributions remain open source while giving you full credit.

**Key Points:**
- ✅ You retain all rights to your work
- ✅ Open source version stays open (MIT/Apache 2.0)
- ✅ You get credit in docs and commit history
- ✅ Dual licensing for maximum compatibility

### Understand Our Values
- **Quality over Speed** - We prefer well-tested, documented code over quick fixes
- **Developer Experience First** - Make it easy for others to use and extend
- **Extensibility by Design** - Follow established patterns and protocols
- **Community Collaboration** - Be respectful, helpful, and inclusive

## 🚀 Getting Started

### 1. Set Up Your Development Environment

The easiest way to get started is with mise:

```bash "Setup with mise"
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp

# If you don't have mise installed:
chmod +x scripts/install-mise.sh && ./scripts/install-mise.sh

# Mise *should* be in your PATH now, but if not, run:
export PATH="$HOME/.local/bin:$PATH"

# Get everything else set up
mise run setup

```

This will:
- Install all of the [repo's tools](https://github.com/knitli/codeweaver-mcp/blob/main/mise.toml#L8-L24), including `uv`, `ruff`, `pytest`, and `hk`
   - Mise will install these in `~/.local/share/mise/installs/` but will only add it to your PATH when you are in the project directory
- Create a virtual environment in `.venv`
- Install all dependencies, including development dependencies
- Add our pre-commit hooks (which run with `hk`)

If you really *like* doing things the hard way, you can also set up the environment manually:

```bash "Manual setup"
# Clone the repository
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp

# Create a virtual environment
uv venv --seed .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
uv sync --all-groups  # this installs everything, there are four development groups: `dev`, `test`, `docs`, and `typing`
                      # you can also install them individually, e.g. `uv sync --group dev`

# We'd still prefer you install hk to run our pre-commit hooks
# You need to install both the `pkl` cli and `hk` package
# we'll use `brew`, for other options see [install pkl-cli](https://pkl-lang.org/main/current/pkl-cli/index.html) and [install hk](https://hk.jdx.dev/getting_started.html#installation)
brew install pkl
brew install hk

# Verify installation
uv run pytest tests/unit/ -v

# We told you it's a lot easier with mise, right? 😄
```

### 2. Understand the Architecture

CodeWeaver uses a plugin-based architecture with four main extension points:

- **Providers** (`src/codeweaver/providers/`) - Embedding and reranking services
- **Backends** (`src/codeweaver/backends/`) - Vector database integrations
- **Sources** (`src/codeweaver/sources/`) - Content discovery and processing
- **Services** (`src/codeweaver/services/`) - Cross-cutting concerns

There's also the [`intent layer`](../intent-layer/index.md), which is the main entry point for AI assistants to interact with CodeWeaver. It provides a natural language interface for code exploration and discovery.

Everything is driven by a factory pattern, where you can register new providers and backends dynamically. This allows for easy *extensibility* without modifying the core codebase; in fact, the main factory manager is `ExtensibilityManager` in `src/codeweaver/factories/extensibility_manager.py`.

### 3. Find Something to Work On

**Good First Issues:**
- Look for issues labeled [`good first issue`](https://github.com/knitli/codeweaver-mcp/labels/good%20first%20issue)
- Documentation improvements and examples
- Test coverage improvements
- Simple bug fixes

**For Experienced Contributors:**
- New provider integrations (OpenAI, Anthropic, etc.)
- Vector database backends (Milvus, LanceDB, etc.)
- Performance optimizations
- Advanced features and functionality

## 📝 Types of Contributions

### Code Contributions

#### New Providers
Add support for new embedding/reranking APIs:

```python
# Example: Adding a new provider
from codeweaver.providers.base import CombinedProvider
from codeweaver.cw_types import EmbeddingProvider, RerankingProvider

class NewProvider(CombinedProvider):
    """New provider supporting embeddings and reranking."""

    @property
    def provider_name(self) -> str:
        return "new_provider"

    async def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        # Implementation here
        pass
```

**Requirements:**
- Follow [development patterns](development_patterns.md)
- Include comprehensive tests
- Add configuration validation
- Document API requirements

#### New Backends
Integrate new vector databases:

```python
# Example: Adding a new backend
from codeweaver.backends.base import VectorBackend
from codeweaver.cw_types import VectorPoint, SearchResult

class NewBackend(VectorBackend):
    """New vector database backend."""

    @property
    def backend_name(self) -> str:
        return "new_backend"

    async def search(self, query_vector: list[float], limit: int = 10) -> list[SearchResult]:
        # Implementation here
        pass
```

**Requirements:**
- Implement all protocol methods
- Handle connection management
- Include health checks
- Add configuration options

#### Bug Fixes
- Include test cases that reproduce the bug
- Fix the root cause, not just symptoms
- Update documentation if behavior changes
- Add regression tests
- If you don't know how to do that, submit an issue with a detailed description and we can help you!

### Documentation Contributions

#### API Documentation
- Improve docstrings and type hints
- Add usage examples
- Document error conditions
- Update configuration references

#### User Guides
- Write/improve tutorials and how-to guides
- Create integration examples
- Improve troubleshooting documentation
- Add migration guides

#### Code Examples
- Provider implementation examples
- Configuration templates
- Integration patterns
- Performance optimization examples

### Testing Contributions

#### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Test error conditions and edge cases
- Follow existing test patterns
- We try to focus our unit tests on important input/output behavior, not implementation details. Getting perfect coverage is not the goal, but rather ensuring that the code behaves as expected.

#### Integration Tests
- We prioritize integration tests for core components; this is what users actually experience, so we want to ensure that the components work together as expected.
- Test component interactions
- Use real services when possible
- Test configuration validation
- Performance benchmarks

#### Testing Infrastructure
- Improve test utilities
- Add test fixtures and helpers
- Enhance CI/CD pipelines
- Create testing documentation

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

**Follow Our Standards:**
- Use [development patterns](development_patterns.md) from existing code
- Write meaningful tests that cover core functionality
- Update documentation
- Follow Python style guidelines (enforced by ruff)

**Note:** We use a pretty strict [ruff](https://docs.astral.sh/ruff/) configuration for linting and formatting, so make sure to run it before committing your changes.
  - It's very opinionated. Most of the time, you can just run `mise run fix` to fix *most* of the issues.
  - The important rules to be aware of are the ones that ruff doesn't fix automatically.
  - We have a script that does a decent job of fixing three of the most common issues that ruff doesn't fix: `./scripts/fix-ruff-patterns.sh`. This will fix most instances of:
    - [`G004` Logging f-string](https://docs.astral.sh/ruff/rules/logging-f-string/). Pass any complex objects for printing to the `extra` parameter, use formatted strings for simple objects. Like:
        ```python
        my_simple_object = "example"

        # Bad
        logger.info(f"Processing {my_simple_object}")

        # Good
        logger.info("Processing %s", my_simple_object)

        # ------

        my_complex_object = {"key": VeryComplexObject()}

        # Bad
        logger.info(f"Processing {my_complex_object}")

        # Good
        logger.info("Processing...", extra=my_complex_object)
        ```
    - [`TRY401` Redundant `exception` in `logging.exception`](https://docs.astral.sh/ruff/rules/verbose-log-message/). Don't include the `exception` object in the log message in an except block. Use `logging.exception` which automatically includes the exception information.
        ```python
        try:
            # Some code that may raise an exception
            pass
        except Exception as e:
            # Bad
            logger.exception("An error occurred: %s", e)

            # Good
            logger.exception("An error occurred")
        ```
    - [`TRY300` Use `else` state for returns in `try` blocks](https://docs.astral.sh/ruff/rules/try-consider-else/). Use the `else` clause for final return statements in `try` blocks to ensure they are only executed if no exceptions were raised (early returns in `try` blocks are OK, but final returns should be in the `else` block).
        ```python
        def bad_example():
            try:
                result = some_error_prone_function()
                return result
            except Exception as e:
                logger.exception("Something went wrong")


        def good_example():
            try:
                result = some_error_prone_function()
            except Exception as e:
                logger.exception("Something went wrong")
            else:  # <-- only executed if no exceptions were raised
                return result

        ```

**Code Quality:**
```bash
# Run linting and formatting
uv run ruff check
uv run ruff format

# Run tests
uv run pytest

# Check type hints
uv run mypy src/codeweaver/
```

### 3. Write Good Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Good commit messages
feat(providers): add OpenAI provider support
fix(backends): handle connection timeout in Qdrant backend
docs(api): improve provider configuration examples
test(services): add integration tests for chunking service

# Bad commit messages
update stuff
fix bug
wip
```

**Format:**
```
type(scope): description

Optional longer description explaining the change
and why it was made.

Closes #123
```

**Types:**
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `test` - Adding or improving tests
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `chore` - Maintenance tasks

### 4. Test Your Changes

**Required Tests:**
```bash
# Unit tests (required)
uv run pytest tests/unit/ -v

# Integration tests (for core changes)
uv run pytest tests/integration/ -v

# Validation tests (for architecture changes)
uv run python tests/validation/validate_architecture.py
```

**Test Coverage:**
- Aim for >90% coverage on new code
- Test both success and error cases
- Include edge cases and boundary conditions

### 5. Submit a Pull Request

**Before Submitting:**
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

**Pull Request Template:**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests for changes
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🧪 Testing Guidelines

### Test Structure
Follow the established test organization:

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_providers.py
│   ├── test_backends.py
│   └── test_services.py
├── integration/            # Component interaction tests
│   ├── test_server_functionality.py
│   └── test_service_integration.py
└── validation/            # Architecture compliance tests
    └── validate_architecture.py
```

### Test Patterns

**Unit Test Example:**
```python
import pytest
from unittest.mock import AsyncMock
from codeweaver.providers.example import ExampleProvider, ExampleConfig

class TestExampleProvider:
    @pytest.fixture
    def config(self):
        return ExampleConfig(api_key="test-key", model="test-model")

    @pytest.fixture
    def provider(self, config):
        return ExampleProvider(config)

    async def test_embed_documents_success(self, provider):
        """Test successful embedding generation."""
        provider._generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await provider.embed_documents(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 3

    async def test_embed_documents_empty_input(self, provider):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            await provider.embed_documents([])
```

**Integration Test Example:**
```python
@pytest.mark.integration
async def test_provider_backend_integration():
    """Test provider and backend working together."""
    # Setup components
    provider = ExampleProvider(config)
    backend = ExampleBackend(config)

    # Test workflow
    embeddings = await provider.embed_documents(["test"])
    await backend.store_vectors([VectorPoint(id="1", vector=embeddings[0])])
    results = await backend.search(embeddings[0], limit=1)

    assert len(results) == 1
    assert results[0].id == "1"
```

### Test Best Practices

1. **Test Naming** - Use descriptive names that explain what's being tested
2. **Test Independence** - Each test should be independent and not rely on others
3. **Mock External Dependencies** - Use mocks for APIs, databases, file systems
4. **Test Error Conditions** - Test both success and failure scenarios
5. **Use Fixtures** - Share common setup code using pytest fixtures

## 🔍 Code Review Process

### What to Expect
1. **Automated Checks** - CI runs tests, linting, and security scans
2. **Maintainer Review** - Core team reviews code quality and design
3. **Community Feedback** - Other contributors may provide suggestions
4. **Iterative Improvement** - Address feedback and update your PR

### Review Criteria
Reviewers will check for:

**Code Quality:**
- [ ] Follows established patterns and conventions
- [ ] Proper error handling and edge cases
- [ ] Clear, readable code with good naming
- [ ] Appropriate abstractions and design

**Testing:**
- [ ] Comprehensive test coverage
- [ ] Both unit and integration tests
- [ ] Tests cover error conditions
- [ ] Tests are maintainable and clear

**Documentation:**
- [ ] Code is well-documented
- [ ] API changes include documentation updates
- [ ] Examples provided for new features
- [ ] Clear commit messages and PR description

**Architecture:**
- [ ] Follows plugin patterns
- [ ] Proper protocol implementation
- [ ] Services integration when appropriate
- [ ] Backward compatibility maintained

### Addressing Review Feedback

**Be Responsive:**
- Respond to review comments promptly
- Ask questions if feedback isn't clear
- Make requested changes or explain why you disagree

**Update Your PR:**
```bash
# Make changes based on feedback
git add .
git commit -m "address review feedback: improve error handling"
git push origin feature/your-feature-name
```

**Handle Conflicts:**
```bash
# Keep your branch up to date
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main

# Resolve conflicts and push
git push --force-with-lease origin feature/your-feature-name
```

## 🎨 Style Guidelines

### Python Code Style
We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check code style
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

**Key Style Points:**
- Line length: 100 characters
- Use type hints for public functions
- Google-style docstrings
- Modern Python features (≥3.11)
- Strong typing with protocols and TypedDict

### Documentation Style
- Use present tense and active voice
- Include code examples for complex features
- Link to related documentation
- Keep language clear and beginner-friendly

### Git Practices
- Use descriptive branch names
- Make atomic commits
- Write clear commit messages
- Keep commit history clean

## 🆘 Getting Help

### Development Questions
- **[GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions)** - General questions
- **[Extension Development Guide](../extension-development/index.md)** - Technical documentation
- **[Development Patterns](development_patterns.md)** - Code patterns and standards

### Technical Support
- **[Troubleshooting Guide](../getting-started/troubleshooting.md)** - Common issues
- **[Email Support](mailto:adam@knit.li)** - Direct access to maintainers
- **Issue Comments** - Ask questions on specific GitHub issues

### Community
- **Be Patient** - Maintainers are volunteers with day jobs
- **Be Respectful** - Follow our code of conduct
- **Help Others** - Answer questions when you can
- **Share Knowledge** - Write tutorials and examples

## 🚫 What We Don't Accept

### Code We Won't Merge
- Code without tests
- Breaking changes without migration path
- Code that doesn't follow established patterns
- Contributions that introduce security vulnerabilities
- Code with unclear licensing or copyright issues

### Pull Requests We'll Close
- Large refactoring without prior discussion
- Changes that break backward compatibility without justification
- Duplicate functionality without clear benefits
- AI-generated code without human review and understanding

## 🎉 Recognition

### Contributor Benefits
- **Credit** in documentation and changelogs
- **Direct access** to maintainers and roadmap discussions
- **Early access** to new features and beta releases
- **Community recognition** for significant contributions

### Hall of Fame
Contributors who make significant impact get featured in:
- Project README
- Release announcements
- Conference talks and presentations
- Community newsletters

---

## Ready to Contribute?

1. **Fork the repository** on GitHub
2. **Set up your development environment** with the steps above
3. **Find an issue** to work on or propose a new feature
4. **Follow the development workflow** and submit a PR
5. **Engage with the community** and help others

Thank you for contributing to CodeWeaver! Your contributions make this project better for everyone. 🚀

*Questions? Start a [discussion](https://github.com/knitli/codeweaver-mcp/discussions) or [email us](mailto:adam@knit.li).*
