<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Overview of all CodeWeaver Component Providers

In the docs, we should probably present each provider with its corresponding configuration options, usage examples, and any specific requirements or limitations. This will help users understand how to integrate each provider into their CodeWeaver setup effectively.

## Backend Providers

### Default (and recommended) Provider: Qdrant
- **Provider**: Qdrant
- **Description**: Qdrant is a high-performance vector database designed for efficient similarity search and retrieval of high-dimensional data. It supports advanced features like filtering, hybrid search, and real-time updates, making it ideal for applications requiring fast and scalable vector search capabilities.
- **Use**: Can use with Qdrant Cloud. Most repositories can safely stay in the free tier. Alternatively, run locally or deployed with docker. (provide examples of how to do this)
- **Configuration**:
  - `BackendConfig` or `BackendConfigExtended`: src/codeweaver/backends/config.py
  - Can be used with only vector search, or with hybrid search (vector + traditional search).

### Alternative: DocArray Backend
- **Provider**: DocArray
- **Description**: The DocArray backend provides an extensible framework, primarily for integrating CodeWeaver with existing DocArray-based applications. The only backend current available with DocArray is qdrant, but more can be added by implementing the `BaseDocArrayAdapter` (or `DocArrayHybridAdapter` if applicable) interface at /src/codeweaver/backends/providers/docarray/adapter.py
  - For ease of offering more backends, we'll probably prioritize implementing DocArray adapters for other vector databases like Weaviate, Milvus, Redis, ElasticSearch, EPSilla, HnSw, and the inmemory (InMemoryExactNNIndex) backend.
- **Use**: Primarily for users who already have DocArray-based applications and want to integrate CodeWeaver with them. Install with `uv pip install "codeweaver[docarray-qdrant]"`.
  - Note: DocArray is not a vector database, but rather a framework for working with vector data. It can be used with various vector databases, including Qdrant, but it does not provide its own storage solution.
  - The DocArray backend is primarily useful for users who want to leverage existing DocArray-based applications or need specific features provided by DocArray.
- **Configuration**:
  - `DocArraySchemaConfig`, `DocArrayBackendConfig`, `QdrantDocArrayConfig`: src/codeweaver/backends/providers/docarray/config.py
  - Can be used with only vector search, or with hybrid search (vector + traditional search) if supported by the DocArray implementation (as in, in DocArray -- something we don't control).
  - CodeWeaver's native Qdrant backend has more features and is faster, so it is recommended to use that unless you have a specific need for DocArray integration.


## Data Source Providers

### Default Provider: Filesystem
- **Provider**: Filesystem
- **Description**: The Filesystem provider allows CodeWeaver to index and search files directly from the local filesystem. It supports various file formats and can be configured to include or exclude specific directories and file types. It respects `.gitignore` and similar ignore files, ensuring that only relevant files are indexed.
- **Use**: Ideal for local development environments where source code and documentation are stored on the filesystem.
- **Configuration**:
  - `FileSystemDataSourceConfig`: src/codeweaver/sources/providers/filesystem.py (config and implementation in same file)
  - Supports indexing files, directories, and can be configured to include or exclude specific patterns.

### Planned Providers

- Filesystem is the only current implementation, but we plan to add more data source providers in the future. These will include:
  - **Git**: For indexing and searching code repositories directly from Git.
  - **Database**: For indexing and searching data from relational or NoSQL databases.
  - **Web**: For indexing and searching web content.
  - **Custom APIs**: For integrating with custom data sources via APIs. (starting with GitHub)

**Note**: Each of the planned providers have scaffolding in place (src/codeweaver/sources/providers/), but implementations aren't complete yet. The Filesystem provider is the only one currently implemented and ready for use.

## Embedding, Rerank, and NLP Providers

### Overview

Configuration note: all providers currently use the same base configuration classes with no custom implementations. All config classes are in `src/codeweaver/providers/config.py`, and all provider implementations are in `src/codeweaver/providers/providers/`. The config classes are: `ProviderConfig` (common to all providers), `EmbeddingProviderConfig`, `RerankingProviderConfig`, `CombinedProviderConfig` (a provider that offers both embedding and reranking) and `NLPProviderConfig`. Each provider can extend these classes as needed.

Codeweaver conceivably supports using multiple embedding, reranking, and NLP providers simultaneously. We haven't tested this much.

### Default Embedding and Reranking Provider: VoyageAI

- **Provider**: VoyageAI
- **Description**: VoyageAI is a high-performance embedding and reranking provider that offers fast and accurate embeddings for various data types, including text and images. It supports multiple models and can be configured to use different models based on the use case. It is best-in-class for embedding and reranking tasks, providing high accuracy and performance.
- **Use**: Recommended for most applications due to its performance and flexibility. It is the default embedding and reranking provider used by CodeWeaver.
  - Requires a VoyageAI API key (for most repositories, you will stay completely free in the free tier).
- **Configuration**:
    - `VoyageConfig`: src/codeweaver/providers/config.py

### Default NLP Provider: SpaCy
- **Provider**: SpaCy
- **Description**: SpaCy is a powerful NLP library that provides various language processing capabilities, including tokenization, named entity recognition, and part-of-speech tagging. It is used to enhance the intent system in CodeWeaver, allowing for more accurate and context-aware search results.
- **Use**: Recommended for any developer who wants better context for their LLM assistants. Notably, while it is very useful and improves results, it's not strictly required for CodeWeaver to function. It's included in the default installation, but you can install CodeWeaver using its feature flags to exclude it if you don't need or want it.
  - Install with `uv pip install "codeweaver"` or `uv pip install "codeweaver[recommended]"`.
  - For mix and match, you can install with `uv pip install "codeweaver[nlp-spacy]"` to only include the SpaCy NLP provider.
  - **For transformers** - install with `uv pip install "codeweaver[nlp-spacy-transformers]"` to use the transformer-based model `en_core_web_trf` instead of the default `en_core_web_sm`. This model provides better accuracy but is slower and requires more resources.
- **Configuration**:
    - `SpaCyProviderConfig`: src/codeweaver/providers/config.py
    - Can be configured to use different language models based on the use case. The default is `en_core_web_sm`, but users can specify a different model if needed. You can enable `use_transformers` to use `en_core_web_trf` instead, which is a transformer-based model that provides better accuracy but is slower and requires more resources.
    - For improved results, users can provide custom patterns for NER and POS tagging, see `src/codeweaver/providers/nlp/spacy.py` for more details.

### Other Embedding, Rerank, Providers

- **Provider**: OpenAI Compatible
- **Description**: Any embedding provider whose API is compatible with the OpenAI API can be used with CodeWeaver. This of course includes OpenAI itself.
- **Use**: Can be used with any OpenAI-compatible API.
  - install with `uv pip install "codeweaver[provider-openai]"`.

- **Provider**: HuggingFace
- **Description**: HuggingFace provides a wide range of pre-trained models for various NLP tasks, including embeddings and reranking. CodeWeaver can use HuggingFace models for these tasks.
- **Use**: Can be used with any HuggingFace model that supports embeddings or reranking.
  - install with `uv pip install "codeweaver[provider-huggingface]"`.

- **Provider**: Cohere
- **Description**: Cohere provides a range of NLP models, including embeddings and reranking models. CodeWeaver can use Cohere models for these tasks.
- **Use**: Can be used with any Cohere model that supports embeddings or reranking.
  - install with `uv pip install "codeweaver[provider-cohere]"`.

- **Provider**: Sentence Transformers
- **Description**: Sentence Transformers is a library for generating sentence embeddings using pre-trained models.
- **Use**: Can be used with any Sentence Transformers model that supports embeddings.
  - install with `uv pip install "codeweaver[provider-sentence-transformers]"`.


## Services

### Overview

Codeweaver provides a range of available services out of the box. These are primarily used for internal functionality, but can also be used by users to extend Codeweaver's capabilities.

### FastMCP Middleware

Codeweaver has native FastMCP middleware implementations, and provides the `middleware_bridge` to support syncing of middleware and dependencies between CodeWeaver's services and the FastMCP middleware. Middleware implementations include:
  - **Chunking**: CodeWeaver's native chunking service implements `fastmcp.MiddleWare`, this provides ast-aware chunking of source code files (using ast-grep-py), and is used to chunk files before indexing them in the vector database.
    src/codeweaver/middleware/chunking.py
  - **Filtering**: CodeWeaver's native filtering service implements `fastmcp.MiddleWare`, this provides smart filtering of files against ignore files and config settings.
    src/codeweaver/middleware/filtering.py
  - **Telemetry**: CodeWeaver's native telemetry service implements `fastmcp.MiddleWare`, this provides telemetry for CodeWeaver's usage.
    src/codeweaver/middleware/telemetry.py

CodeWeaver also proxies FastMCP's built-in middleware (all in `src/codeweaver/services/providers/middleware.py`), including:
    - **logging** - `FastMCPLoggingProvider`
    - **timing** - `FastMCPTimingProvider`
    - **error handling** - `FastMCPErrorHandlingProvider`
    - **rate-limiting** - `FastMCPRateLimitingProvider`


### Service Providers
Codeweaver provides a range of service providers that can be used to extend Codeweaver's capabilities.

- **auto_indexing** - `src/codeweaver/services/providers/auto_indexing.py`
  - Provides auto-indexing capabilities for CodeWeaver, allowing it to automatically index files and directories based on configuration settings.
  - Enabled by default and always runs in the background if the server is running.
- **caching** - `src/codeweaver/services/providers/caching.py`
  - Adds caching capabilities for cached storage of embeddings, rerankings, queries, and other data.
- **chunking** - `src/codeweaver/services/providers/chunking.py`
  - Provides chunking capabilities for source code files, using ast-aware chunking (using ast-grep-py).
  - This is the actual codeweaver service.
- **file_filtering** - `src/codeweaver/services/providers/file_filtering.py`
    - Provides file filtering capabilities, using ignore files and config settings to filter files before indexing them.
    - This is the actual codeweaver service.
- **telemetry** - `src/codeweaver/services/providers/telemetry.py`
  - Provides telemetry capabilities for CodeWeaver's usage, using PostHog for analytics.
  - This is the actual codeweaver service.
- **rate-limiting** - `src/codeweaver/services/providers/rate_limiting.py`
  - This is a different implementation independent of FastMCP's rate-limiting middleware. FastMCP's rate-limiting is primarily for rate-limiting connections to the server, while this service can be applies to any function or method in CodeWeaver.

There are other service providers that are geared towards supporting CodeWeaver's intent layer; we'll document these with the intent layer documentation. But for now, they are:
- **context_intelligence** - `src/codeweaver/services/providers/context_intelligence.py`
  - Provides context intelligence capabilities for CodeWeaver's intent layer.
- **implicit_learning** - `src/codeweaver/services/providers/implicit_learning.py`
  - Provides implicit learning capabilities for CodeWeaver's intent layer, allowing the system to learn from user interactions and improve over time.
- **intent_orchestrator** - `src/codeweaver/services/providers/intent_orchestrator.py`
  - Coordinates the entire intent processing pipeline, including context intelligence, intent recognition, and response generation.
- **zero_shot_optimization** - `src/codeweaver/services/providers/zero_shot_optimization.py`
  - Provides zero-shot optimization capabilities for CodeWeaver's intent layer, allowing the system to optimize its responses without requiring extensive training data.
