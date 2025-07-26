# CodeWeaver Factory System Documentation

## Overview

The CodeWeaver Factory System provides a unified, extensible architecture for creating and managing vector backends, embedding providers, and data sources. It supports plugin discovery, dependency injection, and comprehensive lifecycle management while maintaining backward compatibility.

## Architecture

### Core Components

1. **ExtensibilityManager** - Central coordinator for all extensible components
2. **UnifiedFactory** - Coordinated factory system for creating components
3. **DependencyContainer** - Dependency injection with lifecycle management
4. **PluginDiscovery** - Dynamic plugin loading and registration
5. **FactoryValidator** - Comprehensive validation and health checks

### Component Hierarchy

```
ExtensibilityManager
    ├── UnifiedFactory
    │   ├── BackendFactory (Vector Databases)
    │   ├── ProviderFactory (Embeddings/Reranking)
    │   └── SourceFactory (Data Sources)
    ├── DependencyContainer
    │   ├── Singleton Management
    │   ├── Scoped Instances
    │   └── Circular Dependency Detection
    └── PluginDiscovery
        ├── Backend Plugins
        ├── Provider Plugins
        └── Source Plugins
```

## Quick Start

### Basic Usage

```python
from codeweaver.config import get_config
from codeweaver.factories.extensibility_manager import ExtensibilityManager

# Load configuration
config = get_config()

# Create and initialize extensibility manager
manager = ExtensibilityManager(config)
await manager.initialize()

# Get components
backend = await manager.get_backend()
embedder = await manager.get_embedding_provider()
reranker = await manager.get_reranking_provider()

# Use components...

# Cleanup
await manager.shutdown()
```

### Using Context Manager

```python
from codeweaver.factories.integration import create_extensibility_context

async with create_extensibility_context(config) as manager:
    backend = await manager.get_backend()
    # Components are automatically cleaned up on exit
```

## Migration Guide

### Migrating from Direct Instantiation

#### Before (Direct Instantiation)
```python
from qdrant_client import QdrantClient
from codeweaver.embeddings import VoyageAIEmbedder

class CodeEmbeddingsServer:
    def __init__(self, config):
        self.qdrant = QdrantClient(url=config.qdrant.url)
        self.embedder = VoyageAIEmbedder(api_key=config.api_key)
```

#### After (Factory System)
```python
from codeweaver.factories.extensibility_manager import ExtensibilityManager

class CodeEmbeddingsServer:
    def __init__(self, config):
        self.manager = ExtensibilityManager(config)

    async def initialize(self):
        await self.manager.initialize()
        self.backend = await self.manager.get_backend()
        self.embedder = await self.manager.get_embedding_provider()
```

### Using Migration Helper

```python
from codeweaver.factories.integration import ServerMigrationHelper

# Existing server instance
server = CodeEmbeddingsServer(config)

# Migrate to factories
helper = ServerMigrationHelper(server)
await helper.migrate_to_factories()

# Server now uses factory-created components
```

## Configuration

### ExtensibilityConfig Options

```python
from codeweaver.factories.extensibility_manager import ExtensibilityConfig

config = ExtensibilityConfig(
    # Plugin discovery
    enable_plugin_discovery=True,
    plugin_directories=["/path/to/plugins"],
    auto_load_plugins=True,

    # Dependency injection
    enable_dependency_injection=True,
    singleton_backends=True,
    singleton_providers=True,

    # Lifecycle management
    enable_graceful_shutdown=True,
    shutdown_timeout=30.0,

    # Performance
    lazy_initialization=True,
    component_caching=True,

    # Compatibility
    enable_legacy_fallbacks=True,
    migration_mode=False,
)
```

### Backend Configuration

```yaml
backend:
  provider: qdrant  # or pinecone, weaviate, chroma, etc.
  url: http://localhost:6333
  api_key: optional-api-key
  collection_name: code-embeddings
  options:
    timeout: 30
    grpc_port: 6334
```

### Provider Configuration

```yaml
embedding:
  provider: voyage  # or openai, cohere, sentence-transformers
  api_key: your-api-key
  model: voyage-code-3
  dimension: 1536
  batch_size: 128
```

## Plugin Development

### Creating a Backend Plugin

```python
from codeweaver.backends.base import VectorBackend

class MyCustomBackend(VectorBackend):
    """Custom vector database backend."""

    @classmethod
    def get_plugin_info(cls):
        return {
            "type": "backend",
            "name": "my-custom-backend",
            "version": "1.0.0",
            "author": "Your Name",
            "description": "Custom vector database integration",
            "capabilities": {
                "supports_vector_search": True,
                "supports_hybrid_search": True,
                "supports_filtering": True,
                "max_vector_dimension": 4096,
                "supported_distances": ["cosine", "euclidean", "dot"],
            },
            "requirements": ["custom-db-client>=2.0.0"],
        }

    async def create_collection(self, name: str, dimension: int) -> None:
        # Implementation
        pass

    async def search_vectors(self, query_vector, filters=None, limit=10):
        # Implementation
        pass
```

### Creating a Provider Plugin

```python
from codeweaver.providers.base import EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider."""

    @classmethod
    def get_plugin_info(cls):
        return {
            "type": "provider",
            "name": "my-embeddings",
            "version": "1.0.0",
            "capabilities": {
                "supports_embedding": True,
                "supports_reranking": False,
                "embedding_dimension": 768,
                "max_batch_size": 100,
                "requires_api_key": True,
            },
        }

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Implementation
        pass
```

### Plugin Directory Structure

```
~/.codeweaver/plugins/
├── my_backend_plugin/
│   ├── __init__.py
│   ├── backend.py
│   └── requirements.txt
└── my_provider_plugin/
    ├── __init__.py
    ├── provider.py
    └── requirements.txt
```

## Dependency Injection

### Registering Dependencies

```python
from codeweaver.factories.dependency_injection import DependencyContainer, Lifecycle

container = DependencyContainer()

# Register a singleton backend
container.register(
    component_type="backend",
    component_name="qdrant",
    factory=lambda: QdrantBackend(config),
    lifecycle=Lifecycle.SINGLETON,
)

# Register with dependencies
container.register(
    component_type="service",
    component_name="search",
    factory=lambda backend, embedder: SearchService(backend, embedder),
    dependencies={
        "backend": "backend:qdrant",
        "embedder": "embedding:voyage",
    },
)
```

### Resolving Dependencies

```python
# Resolve with automatic dependency injection
search_service = container.resolve("service", "search")

# Resolve with scope (for request-scoped instances)
container.create_scope("request-123")
scoped_service = container.resolve("service", "search", scope_id="request-123")
container.dispose_scope("request-123")
```

## Validation and Health Checks

### Running Validation

```python
from codeweaver.factories.validation import FactoryValidator, ValidationLevel

# Create validator
validator = FactoryValidator(factory, level=ValidationLevel.COMPREHENSIVE)

# Validate configuration
results = await validator.validate_configuration(config)

for result in results:
    if not result.passed:
        print(f"Validation failed: {result.message}")
```

### Generating Health Report

```python
# Generate comprehensive health report
health_report = await validator.generate_health_report(config)

print(f"System Health: {health_report.overall_health}")
print(f"Recommendations: {health_report.recommendations}")
```

### Checking Component Compatibility

```python
# Check compatibility between components
compatibility_results = await validator.check_compatibility(config)

for result in compatibility_results:
    print(f"{result.component_a} + {result.component_b}: {result.level.name}")
```

## Advanced Features

### Lazy Initialization

```python
# Configure for lazy initialization
config = ExtensibilityConfig(lazy_initialization=True)
manager = ExtensibilityManager(config, extensibility_config)

# Components are created only when first accessed
backend = await manager.get_backend()  # Created here
backend2 = await manager.get_backend() # Returns same instance (singleton)
```

### Component Caching

```python
# Enable component caching
config = ExtensibilityConfig(component_caching=True)

# Cached components are reused across operations
```

### Custom Lifecycle Management

```python
# Register cleanup handlers
def cleanup_handler(instance):
    if hasattr(instance, 'close'):
        instance.close()

container._register_cleanup(my_component)
```

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**
   - Check plugin directory permissions
   - Verify plugin implements `get_plugin_info()`
   - Check logs for specific error messages

2. **Circular Dependencies**
   - Review dependency graph
   - Use lazy initialization for circular references
   - Consider refactoring to remove cycles

3. **Performance Issues**
   - Enable component caching
   - Use singleton lifecycle for expensive components
   - Enable lazy initialization

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger("codeweaver.factories").setLevel(logging.DEBUG)

# Get detailed component information
info = manager.get_component_info()
print(json.dumps(info, indent=2))
```

## Best Practices

1. **Always Initialize Asynchronously**
   ```python
   await manager.initialize()  # Don't forget await!
   ```

2. **Use Context Managers**
   ```python
   async with create_extensibility_context(config) as manager:
       # Automatic cleanup on exit
   ```

3. **Validate Before Production**
   ```python
   health = await validator.generate_health_report(config)
   if health.overall_health != "healthy":
       raise ValueError("System not healthy for production")
   ```

4. **Handle Graceful Shutdown**
   ```python
   try:
       # Use components
   finally:
       await manager.shutdown()
   ```

5. **Test Plugin Compatibility**
   ```python
   # Test plugins in isolation before deployment
   plugin = MyBackendPlugin()
   info = plugin.get_plugin_info()
   assert info["type"] == "backend"
   ```

## Performance Considerations

- **Singleton Components**: Use for expensive resources (backends, providers)
- **Lazy Loading**: Defer initialization until needed
- **Component Caching**: Reuse instances across operations
- **Batch Operations**: Leverage provider batch capabilities
- **Connection Pooling**: Backends should implement connection pooling

## Security Considerations

- **API Key Management**: Use environment variables or secure vaults
- **Plugin Validation**: Always validate plugins before loading
- **Dependency Isolation**: Use scoped instances for multi-tenant scenarios
- **Resource Limits**: Configure appropriate rate limiting and timeouts
