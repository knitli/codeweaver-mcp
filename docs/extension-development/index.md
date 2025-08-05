# Extension Development Guide

Welcome to CodeWeaver's extension development documentation. CodeWeaver implements a comprehensive plugin architecture that enables developers to extend its capabilities through custom providers, backends, sources, and services.

## 🎯 Quick Overview

CodeWeaver's extension system supports four main types of plugins:

- **[Providers](./providers.md)**: Embedding and reranking services (Voyage AI, OpenAI, custom)
- **[Backends](./backends.md)**: Vector databases and search engines (Qdrant, Pinecone, custom)
- **[Sources](./sources.md)**: Data source connectors (filesystem, Git, API, custom)
- **[Services](./services.md)**: Middleware services (chunking, filtering, validation, custom)

## 🏗️ Architecture Overview

CodeWeaver uses a **protocol-based plugin architecture** with the following key features:

### Factory Pattern Foundation
- **Unified Component Creation**: Central factory system for all components
- **Dependency Injection**: Automatic dependency resolution and injection
- **Lifecycle Management**: Standardized initialization, health monitoring, and cleanup

### Protocol-Based Interfaces
- **Runtime Checkable**: All protocols use `@runtime_checkable` for type validation
- **Type Safety**: Strong typing with comprehensive type annotations
- **Universal Protocols**: Consistent interfaces across all component types

### Service Layer Architecture
- **Middleware Integration**: FastMCP middleware for cross-cutting concerns
- **Health Monitoring**: Comprehensive health tracking with auto-recovery
- **Configuration Management**: Hierarchical configuration with validation

## 🚀 Getting Started

### Development Environment Setup

1. **Clone and Install Dependencies**
   ```bash
   git clone https://github.com/knitli/codeweaver-mcp.git
   cd codeweaver-mcp
   uv sync --group dev
   ```

2. **Activate Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Verify Installation**
   ```bash
   uv run pytest tests/unit/test_factory_system.py
   ```

### Extension Development Workflow

1. **Choose Extension Type**: Determine which type of extension you need
2. **Implement Protocol**: Follow the protocol interface for your extension type
3. **Create Plugin Class**: Implement the plugin interface for registration
4. **Add Configuration**: Define Pydantic configuration models
5. **Write Tests**: Create comprehensive tests for your extension
6. **Register Plugin**: Register your plugin with the appropriate factory

## 📋 Extension Types Overview

### [Embedding/Reranking Providers](./providers.md)
**Purpose**: Add support for new embedding models and reranking services

**Key Protocols**: `EmbeddingProvider`, `RerankProvider`, `NLPProvider`
**Use Cases**: Custom embedding models, local inference, proprietary APIs
**Examples**: Hugging Face transformers, local BERT models, enterprise APIs

### [Vector Backends](./backends.md)
**Purpose**: Integrate with new vector databases and search engines

**Key Protocols**: `VectorBackend`, `HybridSearchBackend`, `StreamingBackend`
**Use Cases**: New vector databases, custom search logic, hybrid search
**Examples**: Elasticsearch, Redis, custom vector stores

### [Data Sources](./sources.md)
**Purpose**: Connect to new types of data sources and content repositories

**Key Protocols**: `DataSource`, `SourceCapabilities`, `SourceWatcher`
**Use Cases**: Custom repositories, API integrations, specialized file formats
**Examples**: Confluence, Notion, custom CMS, database connectors

### [Services](./services.md)
**Purpose**: Add middleware functionality and processing capabilities

**Key Protocols**: `ServiceProvider`, `ChunkingService`, `FilteringService`
**Use Cases**: Custom chunking algorithms, specialized filtering, validation logic
**Examples**: Custom tokenizers, content validators, format converters

## 🔧 Core Development Concepts

### Protocol Implementation
All extensions implement protocol interfaces that define required methods and properties:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def health_check(self) -> bool: ...
```

### Plugin Registration
Extensions register themselves using the plugin interface system:

```python
from codeweaver.factories.plugin_protocols import ProviderPlugin

class MyProviderPlugin(ProviderPlugin):
    @classmethod
    def get_plugin_name(cls) -> str:
        return "my_custom_provider"

    @classmethod
    def get_provider_class(cls) -> type[EmbeddingProvider]:
        return MyCustomProvider
```

### Configuration Management
All extensions use Pydantic v2 for configuration with validation:

```python
from pydantic import BaseModel, Field
from typing import Annotated

class MyProviderConfig(BaseModel):
    api_key: Annotated[str, Field(description="API key for the service")]
    model_name: Annotated[str, Field(default="default-model")]
    timeout: Annotated[int, Field(default=30, ge=1, le=300)]
```

## 📖 Next Steps

Choose your extension type to get started:

- **[Building Custom Providers :material-arrow-right-circle:](./providers.md)**
- **[Creating Vector Backends :material-arrow-right-circle:](./backends.md)**
- **[Developing Data Sources :material-arrow-right-circle:](./sources.md)**
- **[Implementing Services :material-arrow-right-circle:](./services.md)**

Or dive deeper into the architecture:

- **[Protocol Reference :material-arrow-right-circle:](../reference/protocols.md)**
- **[Testing Framework :material-arrow-right-circle:](./testing.md)**
- **[Performance Guidelines :material-arrow-right-circle:](./performance.md)**

## 💡 Need Help?

- **Examples**: Check the `examples/` directory for working extension examples
- **Tests**: Review `tests/unit/` for protocol compliance examples
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Join discussions for architecture questions and design patterns
