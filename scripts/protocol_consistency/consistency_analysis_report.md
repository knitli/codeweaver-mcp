# Protocol Consistency Analysis Report

## Summary

📊 **Total Implementations**: 84
🔧 **Total Protocols**: 8
⚠️  **Total Inconsistencies**: 21

## Package: providers

### Implementations
- **ProviderRegistration** (src/codeweaver/providers/factory.py)
  - Base classes: None
  - Methods: 0
- **ProviderRegistry** (src/codeweaver/providers/factory.py)
  - Base classes: None
  - Methods: 16
- **ProviderFactory** (src/codeweaver/providers/factory.py)
  - Base classes: None
  - Methods: 4
- **CohereProvider** (src/codeweaver/providers/cohere.py)
  - Base classes: CombinedProvider
  - Methods: 15
- **EmbeddingProviderBase** (src/codeweaver/providers/base.py)
  - Base classes: ABC
  - Methods: 10
- **RerankProviderBase** (src/codeweaver/providers/base.py)
  - Base classes: ABC
  - Methods: 8
- **LocalEmbeddingProvider** (src/codeweaver/providers/base.py)
  - Base classes: EmbeddingProviderBase
  - Methods: 1
- **CombinedProvider** (src/codeweaver/providers/base.py)
  - Base classes: EmbeddingProviderBase, RerankProviderBase
  - Methods: 2
- **ProviderConfig** (src/codeweaver/providers/config.py)
  - Base classes: BaseModel
  - Methods: 1
- **EmbeddingProviderConfig** (src/codeweaver/providers/config.py)
  - Base classes: ProviderConfig
  - Methods: 2
- **RerankingProviderConfig** (src/codeweaver/providers/config.py)
  - Base classes: ProviderConfig
  - Methods: 1
- **CombinedProviderConfig** (src/codeweaver/providers/config.py)
  - Base classes: EmbeddingProviderConfig, RerankingProviderConfig
  - Methods: 3
- **VoyageConfig** (src/codeweaver/providers/config.py)
  - Base classes: CombinedProviderConfig
  - Methods: 0
- **OpenAIConfig** (src/codeweaver/providers/config.py)
  - Base classes: EmbeddingProviderConfig
  - Methods: 0
- **OpenAICompatibleConfig** (src/codeweaver/providers/config.py)
  - Base classes: EmbeddingProviderConfig
  - Methods: 0
- **CohereConfig** (src/codeweaver/providers/config.py)
  - Base classes: CombinedProviderConfig
  - Methods: 0
- **HuggingFaceConfig** (src/codeweaver/providers/config.py)
  - Base classes: EmbeddingProviderConfig
  - Methods: 0
- **SentenceTransformersConfig** (src/codeweaver/providers/config.py)
  - Base classes: EmbeddingProviderConfig
  - Methods: 0
- **SentenceTransformersProvider** (src/codeweaver/providers/sentence_transformers.py)
  - Base classes: LocalEmbeddingProvider
  - Methods: 12
- **ValidationResult** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 3
- **CustomProviderRegistration** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 1
- **ProviderImplementationValidator** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 6
- **ProviderCapabilityDetector** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 1
- **EnhancedProviderRegistry** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 10
- **ProviderSDK** (src/codeweaver/providers/custom.py)
  - Base classes: None
  - Methods: 4
- **VoyageAIProvider** (src/codeweaver/providers/voyage.py)
  - Base classes: CombinedProvider
  - Methods: 16
- **HuggingFaceProvider** (src/codeweaver/providers/huggingface.py)
  - Base classes: EmbeddingProviderBase
  - Methods: 16
- **OpenAICompatibleProvider** (src/codeweaver/providers/openai.py)
  - Base classes: EmbeddingProviderBase
  - Methods: 16
- **OpenAIProvider** (src/codeweaver/providers/openai.py)
  - Base classes: OpenAICompatibleProvider
  - Methods: 2

### Protocols
- **EmbeddingProvider** (src/codeweaver/providers/base.py)
  - Methods: 8
- **RerankProvider** (src/codeweaver/providers/base.py)
  - Methods: 6
- **ProviderTemplate** (src/codeweaver/providers/custom.py)
  - Methods: 1

## Package: backends

### Implementations
- **BackendConfig** (src/codeweaver/backends/factory.py)
  - Base classes: BaseModel
  - Methods: 2
- **BackendFactory** (src/codeweaver/backends/factory.py)
  - Base classes: CapabilityQueryMixin
  - Methods: 7
- **BackendConfigExtended** (src/codeweaver/backends/config.py)
  - Base classes: BackendConfig
  - Methods: 3
- **QdrantBackend** (src/codeweaver/backends/qdrant.py)
  - Base classes: None
  - Methods: 13
- **QdrantHybridBackend** (src/codeweaver/backends/qdrant.py)
  - Base classes: QdrantBackend
  - Methods: 7
- **SchemaConfig** (src/codeweaver/backends/docarray/schema.py)
  - Base classes: None
  - Methods: 1
- **DocumentSchemaGenerator** (src/codeweaver/backends/docarray/schema.py)
  - Base classes: None
  - Methods: 5
- **SchemaTemplates** (src/codeweaver/backends/docarray/schema.py)
  - Base classes: None
  - Methods: 3
- **DocArraySchemaConfig** (src/codeweaver/backends/docarray/config.py)
  - Base classes: BaseModel
  - Methods: 1
- **DocArrayBackendConfig** (src/codeweaver/backends/docarray/config.py)
  - Base classes: BackendConfig
  - Methods: 0
- **QdrantDocArrayConfig** (src/codeweaver/backends/docarray/config.py)
  - Base classes: DocArrayBackendConfig
  - Methods: 1
- **PineconeDocArrayConfig** (src/codeweaver/backends/docarray/config.py)
  - Base classes: DocArrayBackendConfig
  - Methods: 1
- **WeaviateDocArrayConfig** (src/codeweaver/backends/docarray/config.py)
  - Base classes: DocArrayBackendConfig
  - Methods: 1
- **DocArrayBackendKind** (src/codeweaver/backends/docarray/config.py)
  - Base classes: BaseEnum
  - Methods: 3
- **DocArrayConfigFactory** (src/codeweaver/backends/docarray/config.py)
  - Base classes: None
  - Methods: 3
- **QdrantDocArrayBackend** (src/codeweaver/backends/docarray/qdrant.py)
  - Base classes: DocArrayHybridAdapter
  - Methods: 6
- **DocArrayAdapterError** (src/codeweaver/backends/docarray/adapter.py)
  - Base classes: Exception
  - Methods: 0
- **VectorConverter** (src/codeweaver/backends/docarray/adapter.py)
  - Base classes: None
  - Methods: 4
- **BaseDocArrayAdapter** (src/codeweaver/backends/docarray/adapter.py)
  - Base classes: VectorBackend, ABC
  - Methods: 11
- **DocArrayHybridAdapter** (src/codeweaver/backends/docarray/adapter.py)
  - Base classes: BaseDocArrayAdapter, HybridSearchBackend
  - Methods: 5

### Protocols
- **VectorBackend** (src/codeweaver/backends/base.py)
  - Methods: 7
- **HybridSearchBackend** (src/codeweaver/backends/base.py)
  - Methods: 3
- **StreamingBackend** (src/codeweaver/backends/base.py)
  - Methods: 2
- **TransactionalBackend** (src/codeweaver/backends/base.py)
  - Methods: 3

## Package: sources

### Implementations
- **SourceFactory** (src/codeweaver/sources/factory.py)
  - Base classes: None
  - Methods: 8
- **WebCrawlerSourceConfig** (src/codeweaver/sources/web.py)
  - Base classes: BaseModel
  - Methods: 0
- **WebCrawlerSourceProvider** (src/codeweaver/sources/web.py)
  - Base classes: AbstractDataSource
  - Methods: 8
- **DatabaseSourceConfig** (src/codeweaver/sources/database.py)
  - Base classes: BaseModel
  - Methods: 0
- **DatabaseSourceProvider** (src/codeweaver/sources/database.py)
  - Base classes: AbstractDataSource
  - Methods: 8
- **SourceWatcher** (src/codeweaver/sources/base.py)
  - Base classes: None
  - Methods: 4
- **SourceConfig** (src/codeweaver/sources/base.py)
  - Base classes: BaseModel
  - Methods: 0
- **AbstractDataSource** (src/codeweaver/sources/base.py)
  - Base classes: ABC
  - Methods: 9
- **SourceRegistry** (src/codeweaver/sources/base.py)
  - Base classes: None
  - Methods: 5
- **DataSourcesConfig** (src/codeweaver/sources/config.py)
  - Base classes: BaseModel
  - Methods: 4
- **DataSourceManager** (src/codeweaver/sources/integration.py)
  - Base classes: None
  - Methods: 9
- **APISourceConfig** (src/codeweaver/sources/api.py)
  - Base classes: BaseModel
  - Methods: 0
- **APISourceProvider** (src/codeweaver/sources/api.py)
  - Base classes: AbstractDataSource
  - Methods: 8
- **GitRepositorySourceConfig** (src/codeweaver/sources/git.py)
  - Base classes: BaseModel
  - Methods: 0
- **GitRepositorySourceProvider** (src/codeweaver/sources/git.py)
  - Base classes: AbstractDataSource
  - Methods: 8
- **FileSystemSourceConfig** (src/codeweaver/sources/filesystem.py)
  - Base classes: SourceConfig
  - Methods: 1
- **FileSystemSourceWatcher** (src/codeweaver/sources/filesystem.py)
  - Base classes: SourceWatcher
  - Methods: 7
- **FileSystemSource** (src/codeweaver/sources/filesystem.py)
  - Base classes: AbstractDataSource
  - Methods: 19

### Protocols
- **DataSource** (src/codeweaver/sources/base.py)
  - Methods: 6

### Common Methods
- __init__

### ⚠️  Inconsistencies
- Method '__init__' in WebCrawlerSourceProvider: args differ: ['source_id'] vs []
- Method '__init__' in DatabaseSourceProvider: args differ: ['source_id'] vs []
- Method '__init__' in SourceWatcher: args differ: ['source_id', 'callback'] vs []
- Method '__init__' in AbstractDataSource: args differ: ['source_type', 'source_id'] vs []
- Method '__init__' in DataSourceManager: args differ: ['sources'] vs []
- Method '__init__' in APISourceProvider: args differ: ['source_id'] vs []
- Method '__init__' in GitRepositorySourceProvider: args differ: ['source_id'] vs []
- Method '__init__' in FileSystemSourceWatcher: args differ: ['source_id', 'callback', 'root_path', 'config'] vs []
- Method '__init__' in FileSystemSource: args differ: ['source_id'] vs []

## Package: services

### Implementations
- **ServicesManager** (src/codeweaver/services/manager.py)
  - Base classes: None
  - Methods: 34
- **ServiceBridge** (src/codeweaver/services/middleware_bridge.py)
  - Base classes: Middleware
  - Methods: 4
- **ServiceCoordinator** (src/codeweaver/services/middleware_bridge.py)
  - Base classes: None
  - Methods: 4
- **FilteringService** (src/codeweaver/services/providers/file_filtering.py)
  - Base classes: BaseServiceProvider, FilteringService
  - Methods: 22
- **PostHogTelemetryProvider** (src/codeweaver/services/providers/telemetry.py)
  - Base classes: BaseServiceProvider, TelemetryService
  - Methods: 20
- **ChunkingService** (src/codeweaver/services/providers/chunking.py)
  - Base classes: BaseServiceProvider, ChunkingService
  - Methods: 18
- **RateLimitConfig** (src/codeweaver/services/providers/rate_limiting.py)
  - Base classes: None
  - Methods: 0
- **TokenBucket** (src/codeweaver/services/providers/rate_limiting.py)
  - Base classes: None
  - Methods: 3
- **RateLimitingService** (src/codeweaver/services/providers/rate_limiting.py)
  - Base classes: BaseServiceProvider
  - Methods: 10
- **CacheConfig** (src/codeweaver/services/providers/caching.py)
  - Base classes: None
  - Methods: 0
- **CacheEntry** (src/codeweaver/services/providers/caching.py)
  - Base classes: None
  - Methods: 3
- **CachingService** (src/codeweaver/services/providers/caching.py)
  - Base classes: BaseServiceProvider
  - Methods: 15
- **FastMCPLoggingProvider** (src/codeweaver/services/providers/middleware.py)
  - Base classes: BaseServiceProvider, LoggingService
  - Methods: 10
- **FastMCPTimingProvider** (src/codeweaver/services/providers/middleware.py)
  - Base classes: BaseServiceProvider, TimingService
  - Methods: 8
- **FastMCPErrorHandlingProvider** (src/codeweaver/services/providers/middleware.py)
  - Base classes: BaseServiceProvider, ErrorHandlingService
  - Methods: 10
- **FastMCPRateLimitingProvider** (src/codeweaver/services/providers/middleware.py)
  - Base classes: BaseServiceProvider, RateLimitingService
  - Methods: 9
- **BaseServiceProvider** (src/codeweaver/services/providers/base_provider.py)
  - Base classes: ServiceProvider, ABC
  - Methods: 21

### Common Methods
- __init__
- _check_health
- _initialize_provider
- _shutdown_provider

### ⚠️  Inconsistencies
- Method '__init__' in ServiceBridge: args differ: ['services_manager'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in ServiceCoordinator: args differ: ['services_manager'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in FilteringService: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']; return type differs: None vs None
- Method '__init__' in PostHogTelemetryProvider: args differ: ['config', 'logger'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in ChunkingService: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in RateLimitingService: args differ: ['config'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in CachingService: args differ: ['config'] vs ['config', 'logger', 'fastmcp_server']
- Method '__init__' in FastMCPLoggingProvider: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']; return type differs: None vs None
- Method '__init__' in FastMCPTimingProvider: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']; return type differs: None vs None
- Method '__init__' in FastMCPErrorHandlingProvider: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']; return type differs: None vs None
- Method '__init__' in FastMCPRateLimitingProvider: args differ: ['service_type', 'config'] vs ['config', 'logger', 'fastmcp_server']; return type differs: None vs None
- Method '__init__' in BaseServiceProvider: args differ: ['service_type', 'config', 'logger'] vs ['config', 'logger', 'fastmcp_server']

## Cross-Package Analysis

### Utility Methods Found Across Packages
- **__init__**: backends, providers, services, sources
- **_ensure_initialized**: providers, sources
- **_register_builtin_providers**: providers, services
- **_validate_config**: providers, services
- **check_availability**: providers, services, sources
- **get_capabilities**: services, sources
- **health_check**: backends, services
- **start**: services, sources
- **stop**: services, sources
- **validate_api_key**: backends, providers

## Recommendations

### 🎯 Priority Issues
1. **Standardize method signatures** for common utility methods
2. **Implement missing protocol methods** in implementations
3. **Use @require_implementation decorator** for mandatory methods

### 🔧 Enforcement Improvements
1. **Protocol validation**: Add runtime checks for protocol compliance
2. **Base class standardization**: Create common base classes for utilities
3. **Type checking**: Enhance static analysis with mypy protocols
4. **Decorator usage**: Apply @require_implementation for abstract methods