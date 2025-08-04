# Provider Configuration

Providers are the AI services that power CodeWeaver's semantic search capabilities. This includes embedding models for vector search, reranking models for result refinement, and NLP models for enhanced context.

## Provider Types

### Embedding Providers
Convert code into vector representations for semantic search.

### Reranking Providers  
Refine search results by reordering based on relevance.

### NLP Providers
Extract additional context and features from code.

## Available Providers

### VoyageAI (Recommended)

**Best-in-class performance for code** - Optimized specifically for programming languages.

#### Configuration

=== "Environment Variables"

    ```bash
    # Required
    export CW_EMBEDDING_API_KEY="voyage-abc123..."
    
    # Optional model selection
    export CW_PROVIDERS__VOYAGE_AI__EMBEDDING_MODEL="voyage-code-3"
    export CW_PROVIDERS__VOYAGE_AI__RERANK_MODEL="voyage-rerank-2"
    ```

=== "TOML Configuration"

    ```toml
    [providers.voyage_ai]
    api_key = "voyage-abc123..."
    embedding_model = "voyage-code-3"
    rerank_model = "voyage-rerank-2"
    max_retries = 3
    timeout = 30
    batch_size = 100
    ```

#### Available Models

| Model | Type | Dimensions | Best For |
|-------|------|------------|----------|
| `voyage-code-3` | Embedding | 1024 | Code search (default) |
| `voyage-3` | Embedding | 1024 | General text |
| `voyage-3-lite` | Embedding | 512 | Fast inference |
| `voyage-rerank-2` | Reranking | - | Result refinement |

#### Advanced Settings

```toml
[providers.voyage_ai]
api_key = "voyage-abc123..."

# Performance settings
max_retries = 3
timeout = 30
batch_size = 100
rate_limit = 300  # requests per minute

# Model configuration
embedding_model = "voyage-code-3"
rerank_model = "voyage-rerank-2"
truncate_input = true
```

---

### OpenAI Compatible

Works with OpenAI API and compatible services (Azure OpenAI, Together, etc.).

#### Configuration

=== "Environment Variables"

    ```bash
    # Required
    export CW_PROVIDERS__OPENAI__API_KEY="sk-abc123..."
    
    # Optional
    export CW_PROVIDERS__OPENAI__BASE_URL="https://api.openai.com/v1"
    export CW_PROVIDERS__OPENAI__EMBEDDING_MODEL="text-embedding-3-small"
    ```

=== "TOML Configuration"

    ```toml
    [providers.openai]
    api_key = "sk-abc123..."
    base_url = "https://api.openai.com/v1"
    embedding_model = "text-embedding-3-small"
    dimensions = 1536
    timeout = 30
    ```

#### Available Models

| Model | Dimensions | Cost | Best For |
|-------|------------|------|----------|
| `text-embedding-3-small` | 1536 | Low | General purpose |
| `text-embedding-3-large` | 3072 | Medium | High accuracy |
| `text-embedding-ada-002` | 1536 | Legacy | Compatibility |

#### Azure OpenAI Setup

```toml
[providers.openai]
api_key = "your-azure-key"
base_url = "https://your-resource.openai.azure.com/openai/deployments/your-deployment/embeddings"
api_version = "2023-05-15"
embedding_model = "text-embedding-ada-002"
```

---

### Cohere

High-performance embedding and reranking with multilingual support.

#### Configuration

```toml
[providers.cohere]
api_key = "cohere-abc123..."
embedding_model = "embed-english-v3.0"
rerank_model = "rerank-english-v3.0"
input_type = "search_document"  # or "search_query"
```

#### Available Models

| Model | Type | Languages | Best For |
|-------|------|-----------|----------|
| `embed-english-v3.0` | Embedding | English | High accuracy |
| `embed-multilingual-v3.0` | Embedding | 100+ | International |
| `rerank-english-v3.0` | Reranking | English | Result refinement |
| `rerank-multilingual-v3.0` | Reranking | 100+ | International |

---

### HuggingFace

Use any HuggingFace model locally or via API.

#### Configuration

=== "Local Models"

    ```toml
    [providers.huggingface]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cpu"  # or "cuda"
    batch_size = 32
    normalize_embeddings = true
    ```

=== "HuggingFace API"

    ```toml
    [providers.huggingface]
    api_key = "hf-abc123..."
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    use_api = true
    ```

#### Popular Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `sentence-t5-base` | 220MB | Medium | Good |

---

### Sentence Transformers (Local)

Run embedding models locally without API dependencies.

#### Configuration

```toml
[providers.sentence_transformers]
model_name = "all-MiniLM-L6-v2"
device = "cpu"  # or "cuda", "mps"
cache_folder = "~/.cache/sentence_transformers"
batch_size = 32
```

#### GPU Support

```toml
[providers.sentence_transformers]
model_name = "all-mpnet-base-v2"
device = "cuda"  # Requires CUDA
batch_size = 64  # Larger batches with GPU
```

## NLP Providers

### SpaCy (Default)

Provides enhanced context for search results.

#### Configuration

```toml
[providers.spacy]
model = "en_core_web_sm"  # or "en_core_web_trf"
batch_size = 1000
n_process = 1
```

#### Available Models

| Model | Type | Size | Features |
|-------|------|------|----------|
| `en_core_web_sm` | CNN | 15MB | Basic NLP |
| `en_core_web_md` | CNN | 50MB | Word vectors |
| `en_core_web_lg` | CNN | 750MB | Large vocab |
| `en_core_web_trf` | Transformer | 560MB | Best accuracy |

## Provider Selection Strategy

### Single Provider Setup

Use one provider for simplicity:

```toml
# VoyageAI for everything
[providers.voyage_ai]
api_key = "voyage-key"
embedding_model = "voyage-code-3"
rerank_model = "voyage-rerank-2"
```

### Multi-Provider Setup

Combine providers for optimal performance:

```toml
# VoyageAI for embeddings
[providers.voyage_ai]
api_key = "voyage-key"
embedding_model = "voyage-code-3"

# Cohere for reranking
[providers.cohere]
api_key = "cohere-key"
rerank_model = "rerank-english-v3.0"
```

### Fallback Configuration

Set up fallbacks for reliability:

```toml
# Primary provider
[providers.voyage_ai]
api_key = "voyage-key"
embedding_model = "voyage-code-3"

# Fallback provider
[providers.openai]
api_key = "openai-key"
embedding_model = "text-embedding-3-small"

# Local fallback
[providers.sentence_transformers]
model_name = "all-MiniLM-L6-v2"
```

## Performance Optimization

### Batch Processing

Optimize API usage with batching:

```toml
[providers.voyage_ai]
batch_size = 100  # Process 100 texts at once
max_concurrent_requests = 5  # Parallel requests
```

### Rate Limiting

Respect API limits:

```toml
[providers.voyage_ai]
rate_limit = 300  # requests per minute
burst_size = 10   # allow bursts
backoff_factor = 2  # exponential backoff
```

### Caching

Cache embeddings for faster subsequent runs:

```toml
[providers.voyage_ai]
enable_caching = true
cache_dir = "~/.cache/codeweaver/embeddings"
cache_ttl = 86400  # 24 hours
```

## Provider Comparison

### Performance Comparison

| Provider | Speed | Quality | Cost | Local Option |
|----------|-------|---------|------|--------------|
| **VoyageAI** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | ❌ |
| **OpenAI** | ⭐⭐⭐ | ⭐⭐⭐⭐ | $$ | ❌ |
| **Cohere** | ⭐⭐⭐ | ⭐⭐⭐⭐ | $$ | ❌ |
| **HuggingFace** | ⭐⭐ | ⭐⭐⭐ | $ | ✅ |
| **Sentence Transformers** | ⭐⭐ | ⭐⭐⭐ | Free | ✅ |

### Use Case Recommendations

| Use Case | Recommended Provider | Reasoning |
|----------|---------------------|-----------|
| **Production code search** | VoyageAI | Best accuracy for code |
| **Cost-sensitive** | Sentence Transformers | Free local models |
| **Offline/Private** | HuggingFace Local | No external API calls |
| **Multi-language** | Cohere | Excellent multilingual |
| **Quick testing** | OpenAI | Familiar API |

## Troubleshooting Providers

### API Key Issues

```bash
# Test API key
curl -H "Authorization: Bearer $CW_EMBEDDING_API_KEY" \
     https://api.voyageai.com/v1/embeddings
```

### Model Loading Issues

```python
# Test local model loading
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Performance Issues

1. **Slow embedding**: Increase `batch_size`
2. **Rate limiting**: Decrease `rate_limit` or add delays
3. **Memory issues**: Use smaller models or reduce `batch_size`
4. **Network timeouts**: Increase `timeout` value

### Common Errors

#### Invalid API Key
```
Error: Invalid API key for VoyageAI
```
**Solution:** Verify API key format and permissions

#### Model Not Found
```
Error: Model 'invalid-model' not found
```
**Solution:** Check model name spelling and availability

#### Rate Limit Exceeded
```
Error: Rate limit exceeded (429)
```
**Solution:** Reduce `rate_limit` or implement exponential backoff

## Custom Provider Development

### Provider Interface

Implement the `EmbeddingProvider` protocol:

```python
from codeweaver.cw_types import EmbeddingProvider

class MyCustomProvider(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Implementation here
        pass
        
    async def rerank_results(self, query: str, results: list) -> list:
        # Optional reranking implementation
        pass
```

### Registration

Register your provider with the factory:

```python
from codeweaver.factories import codeweaver_factory

codeweaver_factory.register_embedding_provider(
    "my_provider",
    MyCustomProvider
)
```

### Configuration

Add configuration schema:

```python
from pydantic import BaseModel

class MyProviderConfig(BaseModel):
    api_key: str
    model_name: str = "default-model"
    batch_size: int = 32
```

## Next Steps

- **Backend configuration**: [Backend Configuration](./backends.md)
- **Service configuration**: [Services Configuration](./services.md)
- **Advanced patterns**: [Advanced Configuration](./advanced.md)
- **Performance tuning**: [Performance Guide](../user-guide/performance.md)