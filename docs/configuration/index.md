<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Configuration Overview

CodeWeaver provides a flexible, hierarchical configuration system designed to be **useful out of the box** while allowing complete customization for advanced use cases.

## Quick Start with Presets

CodeWeaver includes built-in configuration profiles that cover common use cases:

=== "Default (Recommended)"

    **Best for most users** - Optimized performance with VoyageAI and Qdrant:

    ```bash
    # Required environment variables
    export CW_EMBEDDING_API_KEY="your-voyage-api-key"
    export CW_VECTOR_BACKEND_URL="your-qdrant-url"

    # Optional for authentication
    export CW_VECTOR_BACKEND_API_KEY="your-qdrant-api-key"

    # Uses 'recommended' profile automatically
    uv run codeweaver
    ```

=== "Minimal Setup"

    **For testing and development** - Uses local models and in-memory storage:

    ```bash
    # Create minimal configuration file
    echo '[profile]
    name = "minimal"' > .codeweaver.toml

    # No API keys required
    uv run codeweaver
    ```

=== "High Performance"

    **For large codebases** - Optimized for speed and scale:

    ```bash
    # Environment variables
    export CW_EMBEDDING_API_KEY="your-voyage-api-key"
    export CW_VECTOR_BACKEND_URL="your-qdrant-url"
    export CW_ENABLE_HYBRID_SEARCH=true

    # Create performance configuration
    echo '[profile]
    name = "performance"' > .codeweaver.toml

    uv run codeweaver
    ```

## Configuration Hierarchy

CodeWeaver loads configuration from multiple sources in priority order:

1. **Direct parameters** (programmatic configuration)
2. **Environment variables** (`CW_*` prefix)
3. **TOML configuration files** (`.local.codeweaver.toml` :material-arrow-right-circle: `.codeweaver.toml` :material-arrow-right-circle: `~/.config/codeweaver/config.toml`)
4. **`.env` files** (for development)
5. **Secret files** (for secure deployment)

## Core Configuration Sections

| Section | Purpose | Key Components |
|---------|---------|----------------|
| **[Profiles](./profiles.md)** | Pre-configured setups | Built-in presets, custom profiles |
| **[Environment](./environment.md)** | Environment variables | Required keys, optional settings |
| **[Providers](./providers.md)** | AI services | Embedding, reranking, NLP providers |
| **[Backends](./backends.md)** | Vector databases | Qdrant, DocArray, custom backends |
| **[Services](./services.md)** | Service layer | Chunking, filtering, monitoring |
| **[Advanced](./advanced.md)** | Custom configurations | Factory patterns, plugins |
| **[Intent Layer](../intent-layer/configuration.md)** | Intent processing | Custom intent handlers, strategies, patterns |

## Configuration File Example

Here's a complete TOML configuration example:

```toml
# Profile selection (optional - defaults to 'recommended')
[profile]
name = "recommended"

# Vector database backend
[backend]
provider = "qdrant"
url = "https://your-qdrant-cluster.qdrant.io"
api_key = "your-qdrant-api-key"
collection = "codeweaver-embeddings"

# Embedding and reranking providers
[providers.voyage_ai]
api_key = "your-voyage-api-key"
embedding_model = "voyage-code-3"
rerank_model = "voyage-rerank-2"

# Code chunking configuration
[chunking]
max_chunk_size = 1500
min_chunk_size = 50
use_ast_grep = true

# File indexing settings
[indexing]
batch_size = 8
concurrent_files = 10
use_gitignore = true
max_file_size = 1048576  # 1MB

# Service layer configuration
[services]
chunking_provider = "fastmcp_chunking"
filtering_provider = "fastmcp_filtering"
enable_health_monitoring = true

# Rate limiting
[rate_limiting]
enabled = true
requests_per_minute = 60
```

## Next Steps

- **New users**: Start with the [Environment Variables](./environment.md) guide
- **Customization needs**: Explore [Providers](./providers.md) and [Backends](./backends.md)
- **Advanced users**: Check [Services](./services.md) and [Advanced](./advanced.md) configuration
- **Intent processing**: See the [Intent Layer Configuration](../intent-layer/configuration.md)
- **Troubleshooting**: See the [Troubleshooting Guide](../user-guide/troubleshooting.md)
