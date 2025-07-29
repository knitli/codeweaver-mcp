<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Migration Guide

**Date:** January 27, 2025
**Author:** CodeWeaver Development Team
**Version:** 1.0

## Overview

This guide provides step-by-step instructions for migrating existing CodeWeaver components to align with the standardized patterns and services layer architecture. It addresses the critical anti-patterns identified in the codebase analysis and provides clear migration paths.

## Migration Priorities

### 🚨 **Priority 1: Critical Anti-Patterns (Immediate Action Required)**

1. **Direct Middleware Dependencies** - Remove direct imports of middleware components
2. **Legacy/Migration Code** - Remove unnecessary migration systems
3. **Hard Dependencies** - Replace with service layer integration

### ⚠️ **Priority 2: Pattern Standardization (Short-term)**

1. **Naming Conventions** - Align class and method names with standards
2. **Missing Patterns** - Add required class methods and properties
3. **Configuration Patterns** - Standardize configuration inheritance

### 📈 **Priority 3: Services Layer Integration (Medium-term)**

1. **Context Parameter Addition** - Add context parameters to all plugin methods
2. **Service Integration** - Integrate with caching, rate limiting, monitoring
3. **Fallback Implementation** - Ensure graceful degradation

## Step-by-Step Migration Process

### Phase 1: Remove Direct Middleware Dependencies

#### Problem: Direct Middleware Imports

**Before (Anti-pattern):**
```python
# ❌ WRONG: Direct middleware usage
from codeweaver.middleware.chunking import ChunkingMiddleware
from codeweaver.middleware.filtering import FileFilteringMiddleware

class FileSystemSource:
    async def _chunk_content_fallback(self, content: str, file_path: Path):
        chunker = ChunkingMiddleware()  # Direct dependency!
        return await chunker.chunk_content(content, str(file_path))

    async def _filter_files(self, files: list[Path]):
        filter_middleware = FileFilteringMiddleware()  # Direct dependency!
        return await filter_middleware.filter_files(files)
```

**After (Correct pattern):**
```python
# ✅ CORRECT: Services layer integration
class FileSystemSourceProvider:
    async def _chunk_content(self, content: str, file_path: Path, context: dict) -> list[CodeChunk]:
        """Chunk content using service layer."""
        chunking_service = context.get("chunking_service")
        if chunking_service:
            return await chunking_service.chunk_content(content, str(file_path))
        else:
            return self._simple_chunk_fallback(content, file_path)

    async def _filter_files(self, files: list[Path], context: dict) -> list[Path]:
        """Filter files using service layer."""
        filtering_service = context.get("filtering_service")
        if filtering_service:
            return await filtering_service.filter_files(files)
        else:
            return self._simple_filter_fallback(files)

    def _simple_chunk_fallback(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Simple chunking fallback without middleware dependency."""
        chunk_size = 1500
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            chunks.append(CodeChunk.create_with_hash(
                content=chunk_content,
                start_byte=i,
                end_byte=min(i + chunk_size, len(content)),
                file_path=str(file_path)
            ))
        return chunks

    def _simple_filter_fallback(self, files: list[Path]) -> list[Path]:
        """Simple file filtering without middleware dependency."""
        filtered = []
        for file_path in files:
            if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
                if file_path.stat().st_size <= 1048576:  # 1MB limit
                    filtered.append(file_path)
        return filtered
```

#### Migration Steps:

1. **Identify Direct Imports**
   ```bash
   # Find all direct middleware imports
   grep -r "from codeweaver.middleware" src/
   grep -r "import.*middleware" src/
   ```

2. **Remove Import Statements**
   ```python
   # DELETE these lines
   from codeweaver.middleware.chunking import ChunkingMiddleware
   from codeweaver.middleware.filtering import FileFilteringMiddleware
   ```

3. **Add Context Parameter**
   ```python
   # Update method signatures
   async def process_content(self, content: str, context: dict) -> ProcessedContent:
   #                                           ^^^^^^^^^^^^^^ Add context parameter
   ```

4. **Replace Direct Usage**
   ```python
   # Replace direct instantiation
   # OLD: chunker = ChunkingMiddleware()
   # NEW: chunking_service = context.get("chunking_service")
   ```

5. **Implement Fallback Logic**
   ```python
   # Always provide fallback
   if chunking_service:
       result = await chunking_service.chunk_content(content)
   else:
       result = self._fallback_chunking(content)
   ```

### Phase 2: Standardize Naming Conventions

#### Problem: Inconsistent Class Names

**Before:**
```python
class FileSystemSource:  # Missing "Provider" suffix
class APISource:         # Missing "Provider" suffix
class GitSource:         # Missing "Provider" suffix
```

**After:**
```python
class FileSystemSourceProvider:  # Follows naming convention
class APISourceProvider:         # Follows naming convention
class GitSourceProvider:         # Follows naming convention
```

#### Migration Steps:

1. **Rename Classes**
   ```python
   # Update class definitions
   class FileSystemSource:  # OLD
   class FileSystemSourceProvider:  # NEW
   ```

2. **Update All References**
   ```python
   # Update imports
   from codeweaver.sources.filesystem import FileSystemSourceProvider

   # Update factory registrations
   register_source_class(SourceType.FILESYSTEM, FileSystemSourceProvider)

   # Update type hints
   def create_source(self) -> FileSystemSourceProvider:
   ```

3. **Update Configuration Classes**
   ```python
   # OLD naming
   class FileSystemSourceConfig:

   # NEW naming (remove redundant "Source")
   class FileSystemConfig:
   ```

### Phase 3: Add Missing Patterns

#### Problem: Missing Required Class Methods

**Before (Missing patterns):**
```python
class FileSystemSourceProvider:
    def __init__(self, config: FileSystemConfig):
        self.config = config

    # Missing: check_availability classmethod
    # Missing: get_static_source_info classmethod
    # Missing: properties for capabilities
```

**After (Complete patterns):**
```python
class FileSystemSourceProvider(AbstractDataSource):
    """Filesystem source provider following standard patterns."""

    def __init__(self, config: FileSystemConfig):
        super().__init__(config)
        self._capabilities = self._get_capabilities()

    @property
    def source_name(self) -> str:
        """Get the source name."""
        return SourceType.FILESYSTEM.value

    @property
    def capabilities(self) -> SourceCapabilities:
        """Get source capabilities."""
        return self._capabilities

    @classmethod
    def check_availability(cls, capability: SourceCapability) -> tuple[bool, str | None]:
        """Check if filesystem source is available for the given capability."""
        # Filesystem is always available
        return True, None

    @classmethod
    def get_static_source_info(cls) -> SourceInfo:
        """Get static information about this source."""
        return SourceInfo(
            name=SourceType.FILESYSTEM.value,
            capabilities=cls._get_static_capabilities(),
            description="Local filesystem source provider"
        )

    def _get_capabilities(self) -> SourceCapabilities:
        """Get runtime capabilities."""
        return SourceCapabilities(
            supports_streaming=True,
            supports_filtering=True,
            supports_watching=True
        )

    @classmethod
    def _get_static_capabilities(cls) -> SourceCapabilities:
        """Get static capabilities."""
        return SourceCapabilities(
            supports_streaming=True,
            supports_filtering=True,
            supports_watching=True
        )
```

#### Migration Steps:

1. **Add Required Properties**
   ```python
   @property
   def source_name(self) -> str:
       return SourceType.FILESYSTEM.value

   @property
   def capabilities(self) -> SourceCapabilities:
       return self._capabilities
   ```

2. **Add Required Class Methods**
   ```python
   @classmethod
   def check_availability(cls, capability: SourceCapability) -> tuple[bool, str | None]:
       # Implementation specific to each source
       return True, None

   @classmethod
   def get_static_source_info(cls) -> SourceInfo:
       # Return static information about the source
       return SourceInfo(...)
   ```

3. **Implement Abstract Methods**
   ```python
   # Ensure all abstract methods from base class are implemented
   async def discover_content(self, context: dict) -> AsyncIterator[ContentItem]:
       # Implementation

   async def get_content_item(self, identifier: str, context: dict) -> ContentItem:
       # Implementation
   ```

### Phase 4: Standardize Configuration Patterns

#### Problem: Configuration Duplication

**Before (Duplicated config):**
```python
class FileSystemSourceConfig(BaseModel):
    # Duplicated base fields
    enabled: bool = True
    name: str = "filesystem"

    # Source-specific fields
    root_path: Path
    include_patterns: list[str] = []
    exclude_patterns: list[str] = []

class APISourceConfig(BaseModel):
    # Duplicated base fields again
    enabled: bool = True
    name: str = "api"

    # Source-specific fields
    base_url: str
    api_key: str | None = None
```

**After (Proper inheritance):**
```python
class BaseSourceConfig(BaseModel):
    """Base configuration for all sources."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        frozen=False,
    )

    enabled: bool = True
    name: str
    timeout: int = 30

class FileSystemConfig(BaseSourceConfig):
    """Configuration for filesystem source."""
    name: str = "filesystem"
    root_path: Annotated[Path, Field(description="Root path for scanning")]
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)

class APIConfig(BaseSourceConfig):
    """Configuration for API source."""
    name: str = "api"
    base_url: Annotated[str, Field(description="Base API URL")]
    api_key: Annotated[str | None, Field(default=None, description="API key")]
```

#### Migration Steps:

1. **Create Base Configuration**
   ```python
   class BaseSourceConfig(BaseModel):
       """Base configuration for all sources."""
       # Common fields here
   ```

2. **Update Specific Configurations**
   ```python
   class FileSystemConfig(BaseSourceConfig):
       """Inherit from base and add specific fields."""
       # Only source-specific fields here
   ```

3. **Remove Duplicated Fields**
   ```python
   # Remove fields that are now in base class
   # Keep only source-specific fields
   ```

### Phase 5: Services Layer Integration

#### Problem: No Service Integration

**Before (No services):**
```python
class VoyageAIProvider:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Direct API call without services
        result = self.client.embed(texts=texts, model=self._model)
        return result.embeddings
```

**After (With services integration):**
```python
class VoyageAIProvider:
    async def embed_documents(self, texts: list[str], context: dict) -> list[list[float]]:
        # Check rate limiting service
        rate_limiter = context.get("rate_limiting_service")
        if rate_limiter:
            await rate_limiter.acquire("voyage_ai", len(texts))

        # Check cache service
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = self._generate_cache_key(texts)
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                return cached_result

        try:
            # Generate embeddings
            result = self.client.embed(texts=texts, model=self._model)
            embeddings = result.embeddings

            # Cache result
            if cache_service:
                await cache_service.set(cache_key, embeddings, ttl=3600)

            return embeddings
        except Exception:
            logger.exception("Error generating VoyageAI embeddings")
            raise
```

#### Migration Steps:

1. **Add Context Parameter**
   ```python
   # Update all public methods to accept context
   async def embed_documents(self, texts: list[str], context: dict) -> list[list[float]]:
   ```

2. **Add Service Integration Points**
   ```python
   # Rate limiting
   rate_limiter = context.get("rate_limiting_service")
   if rate_limiter:
       await rate_limiter.acquire("provider_name", request_count)

   # Caching
   cache_service = context.get("caching_service")
   if cache_service:
       cached_result = await cache_service.get(cache_key)
       if cached_result:
           return cached_result

   # Health monitoring
   health_service = context.get("health_service")
   if health_service:
       await health_service.record_request("provider_name", success=True)
   ```

3. **Maintain Backward Compatibility**
   ```python
   async def embed_documents(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
       """Embed documents with optional services integration."""
       if context is None:
           context = {}

       # Service integration code here
   ```

## Common Migration Patterns

### 1. Method Signature Updates

```python
# Before: No context parameter
async def process_data(self, data: Any) -> ProcessedData:

# After: Context parameter added
async def process_data(self, data: Any, context: dict) -> ProcessedData:

# Backward compatible version
async def process_data(self, data: Any, context: dict | None = None) -> ProcessedData:
    if context is None:
        context = {}
    # Implementation
```

### 2. Error Handling Standardization

```python
# Before: Inconsistent error handling
def validate_config(self):
    if not self.api_key:
        raise Exception("API key required")

# After: Consistent error handling
def _validate_config(self) -> None:
    """Validate provider configuration."""
    if not self.config.get("api_key"):
        raise ValueError("API key is required")

    model = self.config.get("model", self._capabilities.default_model)
    if model not in self._capabilities.supported_models:
        available = ", ".join(self._capabilities.supported_models)
        raise ValueError(f"Unknown model: {model}. Available: {available}")
```

### 3. Registry Integration

```python
# Before: Manual registration
PROVIDERS = {
    "voyage": VoyageAIProvider,
    "openai": OpenAIProvider,
}

# After: Automatic registration
from codeweaver.types import register_provider_class, ProviderType

# At end of module
register_provider_class(ProviderType.VOYAGE_AI, VoyageAIProvider)
```

## Testing Migration

### 1. Update Test Structure

```python
# Before: Tests without services
async def test_embedding_generation(self):
    provider = VoyageAIProvider(config)
    result = await provider.embed_documents(["test"])
    assert len(result) == 1

# After: Tests with and without services
async def test_embedding_with_services(self):
    context = {"caching_service": MockCacheService()}
    provider = VoyageAIProvider(config)
    result = await provider.embed_documents(["test"], context)
    assert len(result) == 1

async def test_embedding_without_services(self):
    context = {}  # No services
    provider = VoyageAIProvider(config)
    result = await provider.embed_documents(["test"], context)
    assert len(result) == 1  # Should still work
```

### 2. Add Pattern Validation Tests

```python
def test_provider_follows_naming_convention(self):
    """Test that provider follows naming conventions."""
    assert VoyageAIProvider.__name__.endswith("Provider")
    assert hasattr(VoyageAIProvider, "provider_name")
    assert hasattr(VoyageAIProvider, "check_availability")

def test_provider_has_required_methods(self):
    """Test that provider has all required methods."""
    required_methods = ["check_availability", "get_static_provider_info"]
    for method_name in required_methods:
        assert hasattr(VoyageAIProvider, method_name)
        assert callable(getattr(VoyageAIProvider, method_name))
```

## Migration Checklist

### For Each Component:

- [ ] **Remove Direct Dependencies**
  - [ ] Remove direct middleware imports
  - [ ] Remove direct service instantiation
  - [ ] Add context parameter to methods

- [ ] **Standardize Naming**
  - [ ] Class names end with appropriate suffix (Provider, Backend, etc.)
  - [ ] Configuration classes follow naming convention
  - [ ] Method names follow camelCase/snake_case consistently

- [ ] **Add Required Patterns**
  - [ ] Properties: `provider_name`, `capabilities`, etc.
  - [ ] Class methods: `check_availability`, `get_static_*_info`
  - [ ] Configuration validation: `_validate_config`

- [ ] **Services Integration**
  - [ ] Context parameter in all public methods
  - [ ] Service usage with fallbacks
  - [ ] Error handling for service failures

- [ ] **Testing**
  - [ ] Tests with services
  - [ ] Tests without services (fallback)
  - [ ] Pattern compliance tests

### Validation Commands:

```bash
# Check for direct middleware imports
grep -r "from codeweaver.middleware" src/

# Check for missing Provider suffix
find src/ -name "*.py" -exec grep -l "class.*Source[^P]" {} \;

# Run pattern validation tests
uv run pytest tests/validation/

# Run full test suite
uv run pytest tests/
```

## Troubleshooting Common Issues

### 1. Import Errors After Renaming

```python
# Problem: Old imports still exist
from codeweaver.sources.filesystem import FileSystemSource  # Old name

# Solution: Update all imports
from codeweaver.sources.filesystem import FileSystemSourceProvider  # New name
```

### 2. Missing Context Parameter

```python
# Problem: Method called without context
result = await provider.process_data(data)  # Missing context

# Solution: Always pass context
context = await services_manager.create_service_context()
result = await provider.process_data(data, context)
```

### 3. Service Not Available

```python
# Problem: Assuming service exists
service = context["service_name"]  # Will fail if not available

# Solution: Check availability
service = context.get("service_name")
if service:
    result = await service.process(data)
else:
    result = self._fallback_process(data)
```

## Migration Timeline

### Week 1-2: Critical Anti-Patterns
- Remove direct middleware dependencies
- Add context parameters
- Implement basic fallbacks

### Week 3-4: Pattern Standardization
- Rename classes and methods
- Add missing patterns
- Standardize configurations

### Week 5-6: Services Integration
- Integrate with rate limiting
- Add caching support
- Implement health monitoring

### Week 7: Testing and Validation
- Update test suites
- Add pattern validation
- Performance testing

## Conclusion

Following this migration guide will ensure your CodeWeaver components align with the established patterns and integrate properly with the services layer. The key principles are:

1. **Remove Direct Dependencies** - Use services layer instead
2. **Follow Naming Conventions** - Consistent naming across all components
3. **Implement Required Patterns** - Properties and class methods
4. **Provide Fallbacks** - Graceful degradation when services unavailable
5. **Test Thoroughly** - Both with and without services

For more information, see:
- [Services Layer Usage Guide](SERVICES_LAYER_GUIDE.md)
- [Development Patterns Guide](DEVELOPMENT_PATTERNS.md)
- [Factory System Documentation](FACTORY_SYSTEM.md)
