# Environment Variables

CodeWeaver uses environment variables for secure configuration and deployment flexibility. All settings can be configured via environment variables using the `CW_` prefix.

## Quick Start

### Minimal Setup (Local Testing)

No environment variables required! Just use the minimal profile:

```bash
echo '[profile]
name = "minimal"' > .codeweaver.toml

uv run codeweaver
```

### Production Setup (Recommended)

```bash
# Required: Embedding provider API key
export CW_EMBEDDING_API_KEY="your-voyage-api-key"

# Required: Vector database URL
export CW_VECTOR_BACKEND_URL="https://your-cluster.qdrant.io"

# Optional: Vector database authentication
export CW_VECTOR_BACKEND_API_KEY="your-qdrant-api-key"

# Optional: Custom collection name
export CW_VECTOR_BACKEND_COLLECTION="my-project-embeddings"

# Run CodeWeaver
uv run codeweaver
```

## Core Environment Variables

### Required Variables

#### `CW_EMBEDDING_API_KEY`
**Purpose:** API key for your embedding provider  
**Required for:** VoyageAI, OpenAI, Cohere providers  
**Example:** `export CW_EMBEDDING_API_KEY="voyage-abc123..."`

#### `CW_VECTOR_BACKEND_URL`
**Purpose:** URL for your vector database  
**Required for:** Qdrant, Pinecone, Weaviate backends  
**Example:** `export CW_VECTOR_BACKEND_URL="https://xyz.qdrant.io"`

### Optional Core Variables

#### `CW_VECTOR_BACKEND_API_KEY`
**Purpose:** Authentication for vector database  
**Default:** None (uses public access)  
**Example:** `export CW_VECTOR_BACKEND_API_KEY="qdrant-key-456"`

#### `CW_VECTOR_BACKEND_COLLECTION`
**Purpose:** Collection/index name for embeddings  
**Default:** `"code-embeddings"`  
**Example:** `export CW_VECTOR_BACKEND_COLLECTION="my-project"`

#### `CW_PROFILE_NAME`
**Purpose:** Select configuration profile  
**Default:** `"codeweaver_original"`  
**Options:** `"codeweaver_original"`, `"minimal"`, `"performance"`  
**Example:** `export CW_PROFILE_NAME="performance"`

## Provider-Specific Variables

### VoyageAI Configuration

```bash
# API credentials
export CW_PROVIDERS__VOYAGE_AI__API_KEY="voyage-abc123"

# Model selection
export CW_PROVIDERS__VOYAGE_AI__EMBEDDING_MODEL="voyage-code-3"
export CW_PROVIDERS__VOYAGE_AI__RERANK_MODEL="voyage-rerank-2"

# Performance tuning
export CW_PROVIDERS__VOYAGE_AI__MAX_RETRIES=3
export CW_PROVIDERS__VOYAGE_AI__TIMEOUT=30
```

### OpenAI Configuration

```bash
# API credentials
export CW_PROVIDERS__OPENAI__API_KEY="sk-abc123"
export CW_PROVIDERS__OPENAI__BASE_URL="https://api.openai.com/v1"

# Model selection
export CW_PROVIDERS__OPENAI__EMBEDDING_MODEL="text-embedding-3-small"
export CW_PROVIDERS__OPENAI__DIMENSIONS=1536
```

### Cohere Configuration

```bash
# API credentials
export CW_PROVIDERS__COHERE__API_KEY="cohere-abc123"

# Model selection
export CW_PROVIDERS__COHERE__EMBEDDING_MODEL="embed-english-v3.0"
export CW_PROVIDERS__COHERE__RERANK_MODEL="rerank-english-v3.0"
```

## Backend-Specific Variables

### Qdrant Configuration

```bash
# Connection settings
export CW_VECTOR_BACKEND_PROVIDER="qdrant"
export CW_VECTOR_BACKEND_URL="https://xyz.qdrant.io"
export CW_VECTOR_BACKEND_API_KEY="qdrant-key"

# Collection settings
export CW_VECTOR_BACKEND_COLLECTION="embeddings"
export CW_ENABLE_HYBRID_SEARCH=true

# Performance settings
export CW_VECTOR_BACKEND_TIMEOUT=30
export CW_VECTOR_BACKEND_MAX_RETRIES=3
```

### DocArray Configuration

```bash
# Backend selection
export CW_VECTOR_BACKEND_PROVIDER="docarray"
export CW_DOCARRAY_BACKEND="qdrant"  # or "memory", "weaviate"

# Qdrant via DocArray
export CW_DOCARRAY_QDRANT_URL="https://xyz.qdrant.io"
export CW_DOCARRAY_QDRANT_API_KEY="qdrant-key"
```

## Processing Configuration

### Chunking Settings

```bash
# Chunk size limits
export CW_CHUNKING__MAX_CHUNK_SIZE=1500
export CW_CHUNKING__MIN_CHUNK_SIZE=50

# Chunking method
export CW_CHUNKING__USE_AST_GREP=true  # Structure-aware chunking
export CW_CHUNKING__FALLBACK_ENABLED=true
```

### Indexing Settings

```bash
# File processing
export CW_INDEXING__BATCH_SIZE=8
export CW_INDEXING__CONCURRENT_FILES=10
export CW_INDEXING__MAX_FILE_SIZE=1048576  # 1MB

# File filtering
export CW_INDEXING__USE_GITIGNORE=true
export CW_INDEXING__FOLLOW_SYMLINKS=false
```

## Service Layer Configuration

### Service Providers

```bash
# Core services
export CW_SERVICES__CHUNKING__PROVIDER="fastmcp_chunking"
export CW_SERVICES__FILTERING__PROVIDER="fastmcp_filtering"

# Optional services
export CW_SERVICES__VALIDATION__PROVIDER="fastmcp_validation"
export CW_SERVICES__CACHE__PROVIDER="fastmcp_cache"
```

### Health Monitoring

```bash
# Enable service health monitoring
export CW_SERVICES__ENABLE_HEALTH_MONITORING=true
export CW_SERVICES__HEALTH_CHECK_INTERVAL=60  # seconds

# Auto-recovery settings
export CW_SERVICES__ENABLE_AUTO_RECOVERY=true
export CW_SERVICES__MAX_RECOVERY_ATTEMPTS=3
```

## Server Configuration

### MCP Server Settings

```bash
# Server identification
export CW_SERVER__SERVER_NAME="CodeWeaver"
export CW_SERVER__VERSION="1.0.0"

# Logging configuration
export CW_SERVER__LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export CW_SERVER__LOG_FORMAT="json"  # json, text
```

### Rate Limiting

```bash
# API rate limiting
export CW_RATE_LIMITING__ENABLED=true
export CW_RATE_LIMITING__REQUESTS_PER_MINUTE=60
export CW_RATE_LIMITING__BURST_SIZE=10
```

## Security and Privacy

### Telemetry Settings

```bash
# Disable telemetry (opt-out)
export CW_NO_TELEMETRY=true
# Alternative: export CW_TELEMETRY_ENABLED=false

# Custom telemetry configuration
export CW_POSTHOG_API_KEY="phc-abc123"
export CW_TELEMETRY_ENDPOINT="https://app.posthog.com"
```

### Security Settings

```bash
# API key validation
export CW_SECURITY__VALIDATE_API_KEYS=true
export CW_SECURITY__ENCRYPT_SECRETS=true

# Network security
export CW_SECURITY__VERIFY_SSL=true
export CW_SECURITY__ALLOWED_HOSTS="localhost,127.0.0.1"
```

## Development and Debugging

### Debug Configuration

```bash
# Enable debug mode
export CW_DEBUG=true
export CW_SERVER__LOG_LEVEL="DEBUG"

# Verbose logging
export CW_VERBOSE_LOGGING=true
export CW_LOG_REQUESTS=true
```

### Configuration Loading

```bash
# Explicit configuration file
export CW_CONFIG_FILE="/path/to/config.toml"

# Configuration validation
export CW_VALIDATE_CONFIG=true
export CW_STRICT_VALIDATION=true
```

## Environment Variable Patterns

### Nested Configuration

Use double underscores (`__`) for nested configuration:

```bash
# Equivalent to [providers.voyage_ai] api_key = "..."
export CW_PROVIDERS__VOYAGE_AI__API_KEY="voyage-key"

# Equivalent to [services.chunking] provider = "..."
export CW_SERVICES__CHUNKING__PROVIDER="fastmcp_chunking"
```

### Configuration File Paths

CodeWeaver searches for configuration files in this order:

1. **`CW_CONFIG_FILE`** - Explicit path via environment variable
2. **`.local.codeweaver.toml`** - Workspace-specific (gitignored)
3. **`.codeweaver.toml`** - Project configuration (committed)
4. **`~/.config/codeweaver/config.toml`** - User configuration

### Environment Files

For development, use `.env` files:

```bash
# .env file in project root
CW_EMBEDDING_API_KEY=voyage-abc123
CW_VECTOR_BACKEND_URL=https://xyz.qdrant.io
CW_PROFILE_NAME=minimal
```

## Common Patterns

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install CodeWeaver
RUN pip install codeweaver

# Set environment variables
ENV CW_EMBEDDING_API_KEY=""
ENV CW_VECTOR_BACKEND_URL=""
ENV CW_PROFILE_NAME="performance"

# Create non-root user
RUN useradd -m codeweaver
USER codeweaver

CMD ["codeweaver"]
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: codeweaver-config
data:
  CW_PROFILE_NAME: "performance"
  CW_SERVER__LOG_LEVEL: "INFO"
  CW_ENABLE_HYBRID_SEARCH: "true"
  CW_SERVICES__ENABLE_HEALTH_MONITORING: "true"
```

### CI/CD Integration

```yaml
# GitHub Actions example
env:
  CW_EMBEDDING_API_KEY: ${{ secrets.VOYAGE_API_KEY }}
  CW_VECTOR_BACKEND_URL: ${{ secrets.QDRANT_URL }}
  CW_PROFILE_NAME: "performance"
  CW_NO_TELEMETRY: "true"
```

## Troubleshooting Environment Variables

### Validation Issues

```bash
# Check current configuration
python -c "from codeweaver.config import CodeWeaverConfig; print(CodeWeaverConfig().model_dump())"

# Validate specific settings
export CW_VALIDATE_CONFIG=true
uv run codeweaver
```

### Common Errors

#### Missing API Key
```
Error: CW_EMBEDDING_API_KEY is required
```
**Solution:** Set the API key or use the `minimal` profile

#### Invalid URL Format
```
Error: Invalid URL format for CW_VECTOR_BACKEND_URL
```
**Solution:** Ensure URL includes protocol (`https://`)

#### Configuration Conflicts
```
Warning: Environment variable overrides TOML configuration
```
**Solution:** Check for conflicting environment variables

## Next Steps

- **Provider setup**: [Providers Configuration](./providers.md)
- **Backend setup**: [Backend Configuration](./backends.md)
- **Service configuration**: [Services Configuration](./services.md)
- **Troubleshooting**: [Common Issues](../user-guide/troubleshooting.md)