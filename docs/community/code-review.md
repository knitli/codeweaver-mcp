<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Code Review Process and Standards

This guide outlines CodeWeaver's code review process, quality standards, and expectations for both contributors and reviewers.

## 🎯 Code Review Philosophy

**Our Core Values:**
- **Quality over Speed** - We prefer thorough reviews over quick merges
- **Learning Together** - Reviews are opportunities for knowledge sharing
- **Constructive Feedback** - Focus on improvement, not criticism
- **Consistency** - Maintain architectural patterns and code standards
- **Security First** - Every change is reviewed for security implications

**Review Goals:**
- Ensure code quality and maintainability
- Verify architectural consistency
- Share knowledge across the team
- Catch bugs and security issues early
- Maintain documentation standards

## 🔄 Review Process

### 1. Pre-Review Checklist (Authors)

Before submitting a pull request:

**Code Quality:**
- [ ] All tests pass locally (`uv run pytest`)
- [ ] Code follows style guidelines (`uv run ruff check`)
- [ ] No linting errors (`uv run ruff format --check`)
- [ ] Type hints are complete
- [ ] Docstrings are comprehensive

**Architecture:**
- [ ] Follows established [development patterns](development_patterns.md)
- [ ] Integrates properly with services layer
- [ ] Uses appropriate protocols and interfaces
- [ ] Maintains backward compatibility

**Testing:**
- [ ] Unit tests cover new functionality
- [ ] Integration tests for component interactions
- [ ] Error conditions are tested
- [ ] Performance impact is acceptable

**Documentation:**
- [ ] Code is self-documenting with clear names
- [ ] Complex logic has explanatory comments
- [ ] API changes include documentation updates
- [ ] Examples provided for new features

### 2. Pull Request Submission

**PR Title Format:**
```plaintext
type(scope): brief description

Examples:
feat(providers): add OpenAI provider support
fix(backends): handle Qdrant connection timeout
docs(api): improve configuration examples
test(services): add integration tests for chunking
```

**PR Description Template:**
```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Implementation Details
- Key changes made
- Architecture decisions
- Trade-offs considered

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance testing (if applicable)

## Breaking Changes
List any breaking changes and migration steps (if applicable).

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests cover new functionality
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced

## Related Issues
Fixes #123
Related to #456
```

### 3. Automated Checks

Before human review, automated systems check:

**Continuous Integration:**
- All tests pass (unit, integration, validation)
- Code coverage meets thresholds (>90% for new code)
- Performance benchmarks within acceptable ranges
- Security scans pass (no high/critical vulnerabilities)

**Code Quality:**
- Linting passes (`ruff check`)
- Formatting is correct (`ruff format --check`)
- Type checking passes (if mypy available)
- Import sorting is correct

**Security:**
- Dependency vulnerability scanning
- Static analysis for security issues
- Secrets detection (no hardcoded keys/passwords)
- Permissions review for new dependencies

### 4. Human Review Process

**Review Assignment:**
- Core maintainers review all changes
- Domain experts review relevant areas
- Community members encouraged to participate
- Minimum 1 approval required, 2 for breaking changes

**Review Timeline:**
- Initial review within 48 hours (business days)
- Follow-up reviews within 24 hours
- Urgent fixes prioritized
- Complex changes may require multiple review rounds

## 📋 Review Standards

### Code Quality Standards

**Readability:**
```python
# ✅ GOOD: Clear, descriptive naming
async def embed_documents_with_retry(
    self,
    texts: list[str],
    max_retries: int = 3,
    context: dict | None = None
) -> list[list[float]]:
    """Generate embeddings with automatic retry on failures."""

# ❌ BAD: Unclear naming and missing documentation
async def embed(self, t, r=3, c=None):
    pass
```

**Error Handling:**
```python
# ✅ GOOD: Specific exception types and context
try:
    embeddings = await self._api_call(texts)
    return embeddings
except ConnectionError as e:
    raise ServiceUnavailableError(f"Provider API unavailable: {e}") from e
except RateLimitError as e:
    raise  # Re-raise rate limit errors as-is
except Exception as e:
    raise ProviderError(f"Unexpected provider error: {e}") from e

# ❌ BAD: Generic exception handling
try:
    return await self._api_call(texts)
except:
    return None
```

**Type Safety:**
```python
# ✅ GOOD: Complete type hints
async def search_vectors(
    self,
    query: list[float],
    limit: int = 10,
    filters: dict[str, Any] | None = None
) -> list[SearchResult]:
    """Search for similar vectors with optional filtering."""

# ❌ BAD: Missing or incomplete type hints
async def search_vectors(self, query, limit=10, filters=None):
    pass
```

### Architecture Standards

**Protocol Compliance:**
```python
# ✅ GOOD: Implements protocol completely
class MyProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "my_provider"

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        # Implementation here
        pass

    async def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        # Implementation here
        pass

# ❌ BAD: Missing required methods
class MyProvider(BaseProvider):
    async def embed_documents(self, texts, context=None):
        pass  # Missing other required methods
```

**Services Integration:**
```python
# ✅ GOOD: Proper services integration with fallbacks
async def process_request(self, data: str, context: dict | None = None) -> ProcessedData:
    if context is None:
        context = {}

    # Try to use caching service
    cache_service = context.get("caching_service")
    if cache_service:
        cached = await cache_service.get(cache_key)
        if cached:
            return cached

    # Process data
    result = await self._process(data)

    # Cache result if service available
    if cache_service:
        await cache_service.set(cache_key, result)

    return result

# ❌ BAD: No services integration
async def process_request(self, data: str) -> ProcessedData:
    return await self._process(data)  # No caching, rate limiting, etc.
```

### Testing Standards

**Test Coverage:**
```python
# ✅ GOOD: Comprehensive test coverage
class TestMyProvider:
    async def test_embed_documents_success(self, provider, mock_context):
        """Test successful embedding generation."""
        # Test implementation
        pass

    async def test_embed_documents_empty_input(self, provider):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            await provider.embed_documents([])

    async def test_embed_documents_api_error(self, provider):
        """Test handling of API errors."""
        # Test error conditions
        pass

    async def test_embed_documents_with_caching(self, provider, mock_context):
        """Test caching integration."""
        # Test services integration
        pass

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test config edge cases
        pass

# ❌ BAD: Minimal test coverage
class TestMyProvider:
    async def test_embed_documents(self, provider):
        result = await provider.embed_documents(["test"])
        assert result is not None
```

**Integration Testing:**
```python
# ✅ GOOD: Real component integration
@pytest.mark.integration
async def test_provider_backend_workflow():
    """Test complete workflow with real components."""
    provider = VoyageProvider(provider_config)
    backend = QdrantBackend(backend_config)

    # Test full workflow
    embeddings = await provider.embed_documents(["test document"])
    await backend.store_vectors([VectorPoint(id="1", vector=embeddings[0])])
    results = await backend.search(embeddings[0], limit=1)

    assert len(results) == 1
    assert results[0].id == "1"

# ❌ BAD: Everything mocked
async def test_workflow():
    mock_provider = Mock()
    mock_backend = Mock()
    # Everything is mocked, no real integration
```

## 🔍 Review Checklist

### For Reviewers

**Architecture Review:**
- [ ] Follows established patterns and conventions
- [ ] Implements required protocols correctly
- [ ] Integrates properly with services layer
- [ ] Maintains backward compatibility
- [ ] Uses appropriate abstractions

**Code Quality Review:**
- [ ] Code is readable and well-structured
- [ ] Names are clear and descriptive
- [ ] Complex logic is well-commented
- [ ] Error handling is comprehensive
- [ ] Performance considerations addressed

**Security Review:**
- [ ] No hardcoded secrets or credentials
- [ ] Input validation is thorough
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies are secure and up-to-date
- [ ] Permissions follow principle of least privilege

**Testing Review:**
- [ ] Test coverage is comprehensive (>90%)
- [ ] Both success and error cases tested
- [ ] Integration tests for component interactions
- [ ] Performance impact is acceptable
- [ ] Tests are maintainable and clear

**Documentation Review:**
- [ ] Public APIs are well-documented
- [ ] Configuration options explained
- [ ] Examples provided for new features
- [ ] Breaking changes clearly documented
- [ ] Migration guides provided (if needed)

### For Authors

**Self-Review:**
- [ ] Read through your own changes carefully
- [ ] Test your changes in different scenarios
- [ ] Consider edge cases and error conditions
- [ ] Verify backward compatibility
- [ ] Check performance impact

**Feedback Response:**
- [ ] Address all reviewer comments
- [ ] Ask for clarification if feedback unclear
- [ ] Update tests based on feedback
- [ ] Keep PR scope focused and manageable
- [ ] Be responsive to review iterations

## 💬 Review Communication

### Giving Feedback

**Constructive Comments:**
```markdown
# ✅ GOOD: Specific, actionable feedback
This method could benefit from input validation. Consider adding a check for empty texts:

```python
if not texts:
    raise ValueError("texts cannot be empty")
```plaintext

# ✅ GOOD: Architectural suggestion
This implementation looks good! For consistency with other providers, consider using the same error handling pattern as VoyageProvider:

[link to relevant code]

# ❌ BAD: Vague or critical
This is wrong. Fix it.

# ❌ BAD: Style preferences without justification
I don't like this approach.
```

**Review Categories:**
- **Must Fix:** Critical issues that block merge
- **Should Fix:** Important improvements that should be addressed
- **Consider:** Suggestions for improvement (optional)
- **Nitpick:** Minor style or preference issues (optional)

**Approval Guidelines:**
- **Approve:** Ready to merge after addressing any "Must Fix" items
- **Request Changes:** Significant issues that require another review
- **Comment:** Provide feedback without blocking merge

### Receiving Feedback

**Best Practices:**
- **Be Open:** Consider feedback objectively
- **Ask Questions:** Clarify anything that's unclear
- **Learn:** Use reviews as learning opportunities
- **Be Patient:** Reviews take time, especially for complex changes
- **Stay Focused:** Keep discussions on-topic and technical

**Response Examples:**
```markdown
# ✅ GOOD: Acknowledge and address
Thanks for the feedback! You're right about the error handling. I've updated it to match the pattern used in other providers.

# ✅ GOOD: Explain reasoning
I chose this approach because it provides better performance for large batches. However, I can see your point about consistency. Let me refactor to use the standard pattern.

# ✅ GOOD: Ask for clarification
Could you clarify what you mean by "more robust error handling"? Are you referring to specific exception types or recovery strategies?

# ❌ BAD: Defensive or dismissive
That's not how I want to do it.

# ❌ BAD: Ignoring feedback
[No response to reviewer comments]
```

## 🚀 Advanced Review Scenarios

### Breaking Changes

**Additional Requirements:**
- Detailed migration documentation
- Backward compatibility layer (when possible)
- Version bump in `pyproject.toml`
- Clear communication plan
- Extra review from core maintainers

**Review Process:**
1. Architectural review by core team
2. Impact assessment on existing users
3. Migration path validation
4. Documentation review
5. Community feedback period (if significant)

### Security-Sensitive Changes

**Additional Checks:**
- Security expert review required
- Threat modeling for new attack vectors
- Dependency vulnerability assessment
- Secrets and credential handling review
- Authorization and authentication review

**Security Review Checklist:**
- [ ] No new attack surfaces introduced
- [ ] Input validation comprehensive
- [ ] Output sanitization appropriate
- [ ] Error handling doesn't leak information
- [ ] Dependencies are secure and minimal

### Performance-Critical Changes

**Additional Requirements:**
- Benchmark results included in PR
- Performance regression tests
- Resource usage analysis
- Scalability considerations
- Memory leak detection

**Performance Review:**
- [ ] No significant performance regressions
- [ ] Benchmarks show expected improvements
- [ ] Memory usage is reasonable
- [ ] No obvious bottlenecks introduced
- [ ] Async patterns used correctly

## 📊 Review Metrics

### Quality Metrics
- **Review Coverage:** % of changes reviewed by humans
- **Review Turnaround:** Time from PR submission to first review
- **Defect Escape Rate:** Issues found in production after review
- **Review Participation:** Number of active reviewers

### Process Metrics
- **PR Size Distribution:** Lines changed per PR
- **Review Iterations:** Average rounds of review per PR
- **Time to Merge:** Total time from PR to merge
- **Review Depth:** Comments per line of code changed

## 🎓 Reviewer Development

### Becoming a Reviewer

**Prerequisites:**
- Understanding of CodeWeaver architecture
- Familiarity with development patterns
- Experience with Python and async programming
- Knowledge of testing best practices

**Learning Path:**
1. Start with documentation reviews
2. Review small bug fixes and improvements
3. Participate in feature reviews as secondary reviewer
4. Learn from experienced reviewers' feedback
5. Ask questions and seek mentorship

### Advanced Review Skills

**Code Analysis:**
- Understand architectural implications
- Spot potential performance issues
- Identify security vulnerabilities
- Recognize code smells and anti-patterns

**Communication:**
- Provide constructive feedback
- Balance thoroughness with efficiency
- Teach through code review
- Handle disagreements professionally

## 🛠️ Review Tools

### GitHub Features
- **Review Requests:** Tag specific reviewers
- **Draft PRs:** Work-in-progress changes
- **Suggestions:** Propose specific code changes
- **Approval Requirements:** Enforce review standards

### IDE Integration
- **VS Code GitHub Extension:** Review PRs in editor
- **GitLens:** Enhanced Git integration
- **Code Review Tools:** Integrated diff viewing

### Automated Tools
- **GitHub Actions:** Automated testing and checks
- **Ruff:** Code formatting and linting
- **Security Scanners:** Vulnerability detection
- **Coverage Reports:** Test coverage analysis

---

## 🎯 Conclusion

Code review is essential to maintaining CodeWeaver's quality and fostering community collaboration. By following these standards and processes, we ensure:

- **High-quality code** that's maintainable and secure
- **Knowledge sharing** across the community
- **Consistent architecture** that scales with the project
- **Learning opportunities** for all contributors

Remember: Every review is an opportunity to learn, teach, and improve our collective codebase.

## 📚 Additional Resources

- **[Development Patterns](development_patterns.md)** - Coding standards and conventions
- **[Extension Guidelines](extension-guidelines.md)** - How to build extensions
- **[Contributing Guide](contributing.md)** - General contribution process
- **[Testing Guide](../extension-development/testing.md)** - Testing best practices

*Questions about the review process? Ask in [GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions) or [email us](mailto:adam@knit.li).*
