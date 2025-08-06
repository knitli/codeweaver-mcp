<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Configuration Profiles

CodeWeaver includes built-in configuration profiles that provide optimized setups for common use cases. Profiles allow you to get started quickly while still maintaining full customization control.

## Built-in Profiles

### `codeweaver_default` (Default)

**Best for production use** - The recommended configuration for most users.

**Features:**
- **AST-grep structural chunking** for intelligent code segmentation
- **VoyageAI embedding** (voyage-code-3) - best-in-class for code
- **VoyageAI reranking** (voyage-rerank-2) for result refinement
- **Qdrant vector database** for scalable storage
- **Auto-reindexing** enabled for development workflows
- **Optimized batch processing** (8 files, 10 concurrent)

**Configuration:**
```toml
[profile]
name = "codeweaver_default"
```

**Required Environment Variables:**
```bash
export CW_EMBEDDING_API_KEY="your-voyage-api-key"
export CW_VECTOR_BACKEND_URL="your-qdrant-url"
export CW_VECTOR_BACKEND_API_KEY="your-qdrant-api-key"  # optional
```

**Use Cases:**
- Production deployments
- Large codebases (>10k files)
- Teams requiring high-quality semantic search
- CI/CD integration

---

### `minimal`

**For testing and development** - Zero-configuration setup with local models.

**Features:**
- **Simple text chunking** (1000 character maximum)
- **Sentence Transformers** (all-MiniLM-L6-v2) - runs locally
- **In-memory backend** - no external database required
- **No API keys required** - fully self-contained
- **Fast startup** for quick testing

**Configuration:**
```toml
[profile]
name = "minimal"
```

**No Environment Variables Required** - works out of the box!

**Use Cases:**
- Local development and testing
- CI/CD test suites
- Offline environments
- Learning and experimentation

---

### `performance`

**For large-scale deployments** - Optimized for speed and throughput.

**Features:**
- **Large batch processing** (16 files, 20 concurrent)
- **2000 character chunks** for comprehensive context
- **Hybrid search enabled** (vector + keyword)
- **Sparse vectors** for efficient storage
- **Optimized for large codebases** (>100k files)
- **Enhanced caching** and parallel processing

**Configuration:**
```toml
[profile]
name = "performance"
```

**Required Environment Variables:**
```bash
export CW_EMBEDDING_API_KEY="your-voyage-api-key"
export CW_VECTOR_BACKEND_URL="your-qdrant-url"
export CW_ENABLE_HYBRID_SEARCH=true
```

**Use Cases:**
- Enterprise codebases
- High-throughput environments
- Large-scale indexing operations
- Performance-critical applications

## Using Profiles

### 1. Configuration File Method

Create a `.codeweaver.toml` file in your project root:

```toml
[profile]
name = "minimal"
```

### 2. Environment Variable Method

```bash
export CW_PROFILE_NAME="performance"
```

### 3. Programmatic Method

```python
from codeweaver.config import CodeWeaverConfig

config = CodeWeaverConfig(profile={"name": "minimal"})
```

## Profile Customization

You can override specific settings while using a profile:

```toml
# Start with performance profile
[profile]
name = "performance"

# Override chunking settings
[chunking]
max_chunk_size = 3000  # Larger chunks than default

# Override indexing settings
[indexing]
batch_size = 32  # Even larger batches
```

## Custom Profiles

Create your own profiles by combining settings:

```toml
# Custom profile configuration
[backend]
provider = "qdrant"
url = "https://your-cluster.qdrant.io"

[providers.openai]
api_key = "your-openai-key"
model = "text-embedding-3-small"

[chunking]
max_chunk_size = 2000
use_ast_grep = false  # Use simple text chunking

[indexing]
batch_size = 16
concurrent_files = 20
```

## Profile Selection Priority

CodeWeaver selects profiles in this order:

1. **Explicit configuration** - `[profile] name = "..."`
2. **Environment variable** - `CW_PROFILE_NAME`
3. **Default profile** - `codeweaver_default`

## Profile Comparison

| Feature | codeweaver_default | minimal | performance |
|---------|-------------------|---------|-------------|
| **Chunking** | AST-grep structural | Simple text | Large chunks |
| **Embedding** | VoyageAI (cloud) | SentenceTransformers (local) | VoyageAI (cloud) |
| **Backend** | Qdrant | In-memory | Qdrant |
| **API Keys** | Required | None | Required |
| **Performance** | High quality | Fast startup | High throughput |
| **Use Case** | Production | Testing | Enterprise |

## Troubleshooting Profiles

### Profile Not Found

```plaintext
Error: Profile 'myprofile' not found
```

**Solution:** Use one of the built-in profiles (`codeweaver_default`, `minimal`, `performance`) or create a custom configuration.

### Missing API Keys

```plaintext
Error: CW_EMBEDDING_API_KEY is required for profile 'codeweaver_default'
```

**Solution:** Set the required environment variables or switch to the `minimal` profile for testing.

### Performance Issues

If you're experiencing slow performance:

1. **Try the `performance` profile** for large codebases
2. **Enable hybrid search** with `CW_ENABLE_HYBRID_SEARCH=true`
3. **Increase batch sizes** in your configuration
4. **Check your provider rate limits**

## Next Steps

- **Environment setup**: [Environment Variables](./environment.md)
- **Provider configuration**: [Providers](./providers.md)
- **Advanced customization**: [Advanced Configuration](./advanced.md)
