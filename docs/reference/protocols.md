# Protocol Reference

This document provides comprehensive reference documentation for all protocols, interfaces, and contracts in the CodeWeaver extension system.

## 🎯 Overview

CodeWeaver uses **runtime-checkable protocols** to define contracts for extensions. All protocols are typed and validated at runtime, ensuring type safety and compatibility.

## 🏗️ Core Architecture Protocols

### Component Interface

All CodeWeaver components implement the base component interface:

```python
from typing import Protocol, runtime_checkable, Any
from codeweaver.cw_types import ValidationResult, BaseComponentConfig

@runtime_checkable
class Component(Protocol):
    """Base protocol for all CodeWeaver components."""
    
    async def initialize(self) -> None:
        """Initialize component resources."""
        ...
    
    async def shutdown(self) -> None:
        """Cleanup component resources."""
        ...
    
    async def health_check(self) -> bool:
        """Check component health status."""
        ...
    
    def validate_config(self, config: BaseComponentConfig) -> ValidationResult:
        """Validate component configuration."""
        ...
```

### Factory Interface

Component factories create and manage component instances:

```python
from typing import Protocol, TypeVar, Generic

T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound=BaseComponentConfig)

@runtime_checkable
class ComponentFactory(Protocol[T, ConfigT]):
    """Protocol for component factory implementations."""
    
    async def create_component(self, config: ConfigT, context: dict[str, Any]) -> T:
        """Create a component instance with the given configuration."""
        ...
    
    def validate_config(self, config: ConfigT) -> ValidationResult:
        """Validate component configuration."""
        ...
    
    def get_component_info(self) -> BaseComponentInfo:
        """Get information about this component factory."""
        ...
```

## 🔌 Provider Protocols

### EmbeddingProvider Protocol

```python
from typing import Protocol, runtime_checkable
from codeweaver.cw_types import EmbeddingProviderInfo

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    # Required Properties
    @property
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        ...
    
    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...
    
    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...
    
    # Optional Properties
    @property
    def max_batch_size(self) -> int | None:
        """Maximum number of documents per batch (None = no limit)."""
        ...
    
    @property
    def max_input_length(self) -> int | None:
        """Maximum input text length in characters (None = no limit)."""
        ...
    
    # Required Methods
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of documents to embed
            
        Returns:
            List of embedding vectors, one per input document
            
        Raises:
            ProviderError: If embedding fails
            ValidationError: If input is invalid
        """
        ...
    
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector for the query
            
        Raises:
            ProviderError: If embedding fails
            ValidationError: If input is invalid
        """
        ...
    
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """
        Get detailed provider information.
        
        Returns:
            Provider metadata and capabilities
        """
        ...
    
    async def health_check(self) -> bool:
        """
        Check provider health and connectivity.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        ...
```

### RerankProvider Protocol

```python
from codeweaver.cw_types import RerankResult

@runtime_checkable
class RerankProvider(Protocol):
    """Protocol for reranking providers."""
    
    # Required Properties
    @property
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        ...
    
    @property
    def model_name(self) -> str:
        """Name of the reranking model."""
        ...
    
    # Optional Properties
    @property
    def max_documents(self) -> int | None:
        """Maximum number of documents to rerank (None = no limit)."""
        ...
    
    @property
    def max_query_length(self) -> int | None:
        """Maximum query length in characters (None = no limit)."""
        ...
    
    # Required Methods
    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[RerankResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of reranked results with relevance scores
            
        Raises:
            ProviderError: If reranking fails
            ValidationError: If input is invalid
        """
        ...
    
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get detailed provider information."""
        ...
    
    async def health_check(self) -> bool:
        """Check provider health and connectivity."""
        ...
```

### NLPProvider Protocol

```python
from codeweaver.cw_types import IntentType

@runtime_checkable
class NLPProvider(Protocol):
    """Protocol for advanced NLP providers."""
    
    # Core NLP Methods
    async def process_text(
        self, 
        text: str, 
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process text with NLP analysis.
        
        Args:
            text: Text to process
            context: Optional processing context
            
        Returns:
            Dictionary with analysis results
        """
        ...
    
    async def classify_intent(self, text: str) -> tuple[IntentType | None, float]:
        """
        Classify user intent from text.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        ...
    
    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entity dictionaries with type, value, and position
        """
        ...
    
    # Optional Embedding Methods
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts (optional)."""
        ...
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for single text (optional)."""
        ...
    
    # Model Management
    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models."""
        ...
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to different model."""
        ...
```

## 🗄️ Backend Protocols

### VectorBackend Protocol

```python
from codeweaver.cw_types import (
    VectorPoint, SearchResult, SearchFilter, 
    CollectionInfo, DistanceMetric
)

@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector database backends."""
    
    # Collection Management
    async def create_collection(
        self, 
        name: str, 
        dimension: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE, 
        **kwargs: Any
    ) -> None:
        """
        Create a new vector collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric for similarity
            **kwargs: Backend-specific parameters
            
        Raises:
            BackendError: If collection creation fails
            ValidationError: If parameters are invalid
        """
        ...
    
    async def list_collections(self) -> list[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
            
        Raises:
            BackendError: If listing fails
        """
        ...
    
    async def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Raises:
            BackendError: If deletion fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
    
    async def get_collection_info(self, name: str) -> CollectionInfo:
        """
        Get collection information.
        
        Args:
            name: Collection name
            
        Returns:
            Collection metadata and statistics
            
        Raises:
            BackendError: If retrieval fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
    
    # Vector Operations
    async def upsert_vectors(
        self, 
        collection_name: str, 
        vectors: list[VectorPoint]
    ) -> None:
        """
        Insert or update vectors in collection.
        
        Args:
            collection_name: Target collection
            vectors: List of vectors to upsert
            
        Raises:
            BackendError: If upsert fails
            CollectionNotFoundError: If collection doesn't exist
            ValidationError: If vectors are invalid
        """
        ...
    
    async def search_vectors(
        self, 
        collection_name: str, 
        query_vector: list[float], 
        limit: int = 10, 
        search_filter: SearchFilter | None = None, 
        score_threshold: float | None = None, 
        **kwargs: Any
    ) -> list[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Maximum number of results
            search_filter: Optional metadata filter
            score_threshold: Minimum similarity score
            **kwargs: Backend-specific search parameters
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            BackendError: If search fails
            CollectionNotFoundError: If collection doesn't exist
            ValidationError: If parameters are invalid
        """
        ...
    
    async def delete_vectors(
        self, 
        collection_name: str, 
        ids: list[str | int]
    ) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Target collection
            ids: List of vector IDs to delete
            
        Raises:
            BackendError: If deletion fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
    
    # Health Check
    async def health_check(self) -> bool:
        """
        Check backend connectivity and health.
        
        Returns:
            True if backend is healthy, False otherwise
        """
        ...
```

### HybridSearchBackend Protocol

```python
from codeweaver.cw_types import HybridStrategy
from typing import Literal

@runtime_checkable
class HybridSearchBackend(VectorBackend, Protocol):
    """Protocol for backends supporting hybrid search."""
    
    # Sparse Index Management
    async def create_sparse_index(
        self, 
        collection_name: str, 
        fields: list[str], 
        index_type: Literal["keyword", "text", "bm25"] = "bm25", 
        **kwargs: Any
    ) -> None:
        """
        Create sparse index for hybrid search.
        
        Args:
            collection_name: Target collection
            fields: Fields to index
            index_type: Type of sparse index
            **kwargs: Backend-specific index parameters
            
        Raises:
            BackendError: If index creation fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
    
    # Hybrid Search
    async def hybrid_search(
        self, 
        collection_name: str, 
        dense_vector: list[float], 
        sparse_query: dict[str, float] | str, 
        limit: int = 10, 
        hybrid_strategy: HybridStrategy = HybridStrategy.RRF, 
        alpha: float = 0.5, 
        search_filter: SearchFilter | None = None, 
        **kwargs: Any
    ) -> list[SearchResult]:
        """
        Perform hybrid dense + sparse search.
        
        Args:
            collection_name: Collection to search
            dense_vector: Dense query vector
            sparse_query: Sparse query (terms with weights or text)
            limit: Maximum number of results
            hybrid_strategy: Fusion strategy for combining results
            alpha: Weighting factor for dense vs sparse (0.0-1.0)
            search_filter: Optional metadata filter
            **kwargs: Backend-specific parameters
            
        Returns:
            List of search results from hybrid search
            
        Raises:
            BackendError: If search fails
            CollectionNotFoundError: If collection doesn't exist
            ValidationError: If parameters are invalid
        """
        ...
    
    # Sparse Vector Updates
    async def update_sparse_vectors(
        self, 
        collection_name: str, 
        vectors: list[VectorPoint]
    ) -> None:
        """
        Update sparse vectors for existing points.
        
        Args:
            collection_name: Target collection
            vectors: Vectors with sparse data to update
            
        Raises:
            BackendError: If update fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
```

### StreamingBackend Protocol

```python
from typing import AsyncIterator

@runtime_checkable
class StreamingBackend(VectorBackend, Protocol):
    """Protocol for backends supporting streaming operations."""
    
    async def stream_upsert_vectors(
        self, 
        collection_name: str, 
        vector_stream: AsyncIterator[list[VectorPoint]], 
        batch_size: int = 100
    ) -> None:
        """
        Stream upsert vectors in batches.
        
        Args:
            collection_name: Target collection
            vector_stream: Async iterator of vector batches
            batch_size: Optimal batch size for processing
            
        Raises:
            BackendError: If streaming upsert fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
    
    async def stream_search_vectors(
        self, 
        collection_name: str, 
        query_stream: AsyncIterator[list[float]], 
        limit: int = 10
    ) -> AsyncIterator[list[SearchResult]]:
        """
        Stream search multiple query vectors.
        
        Args:
            collection_name: Collection to search
            query_stream: Async iterator of query vectors
            limit: Maximum results per query
            
        Yields:
            Lists of search results for each query
            
        Raises:
            BackendError: If streaming search fails
            CollectionNotFoundError: If collection doesn't exist
        """
        ...
```

## 📁 Data Source Protocols

### DataSource Protocol

```python
from codeweaver.cw_types import (
    SourceCapabilities, SourceConfig, ContentItem, SourceWatcher
)
from typing import Callable

@runtime_checkable
class DataSource(Protocol):
    """Protocol for data source implementations."""
    
    # Capability Discovery
    def get_capabilities(self) -> SourceCapabilities:
        """
        Get source capabilities.
        
        Returns:
            Source capabilities and supported operations
        """
        ...
    
    # Content Operations
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """
        Discover content items from source.
        
        Args:
            config: Source configuration
            
        Returns:
            List of discovered content items
            
        Raises:
            SourceError: If discovery fails
            ValidationError: If config is invalid
        """
        ...
    
    async def read_content(self, item: ContentItem) -> str:
        """
        Read content from a content item.
        
        Args:
            item: Content item to read
            
        Returns:
            Content as string
            
        Raises:
            SourceError: If reading fails
            ContentNotFoundError: If content doesn't exist
        """
        ...
    
    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """
        Get metadata for content item.
        
        Args:
            item: Content item
            
        Returns:
            Dictionary of metadata
            
        Raises:
            SourceError: If metadata retrieval fails
        """
        ...
    
    # Change Watching
    async def watch_changes(
        self, 
        config: SourceConfig, 
        callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """
        Watch for content changes.
        
        Args:
            config: Source configuration
            callback: Function to call when changes detected
            
        Returns:
            Watcher instance for managing the watch
            
        Raises:
            SourceError: If watching fails
            NotSupportedError: If watching not supported
        """
        ...
    
    # Validation
    async def validate_source(self, config: SourceConfig) -> bool:
        """
        Validate source configuration and connectivity.
        
        Args:
            config: Source configuration to validate
            
        Returns:
            True if source is valid and accessible
        """
        ...
    
    async def health_check(self) -> bool:
        """
        Check source health and connectivity.
        
        Returns:
            True if source is healthy, False otherwise
        """
        ...
```

### SourceWatcher Protocol

```python
@runtime_checkable
class SourceWatcher(Protocol):
    """Protocol for source change watchers."""
    
    async def start(self) -> None:
        """Start watching for changes."""
        ...
    
    async def stop(self) -> None:
        """Stop watching for changes."""
        ...
    
    @property
    def is_active(self) -> bool:
        """Check if watcher is currently active."""
        ...
    
    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        ...
```

## ⚙️ Service Protocols

### ServiceProvider Protocol

```python
from codeweaver.cw_types import ServiceType, ServiceHealth, ProviderStatus

@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for service providers."""
    
    # Properties
    @property
    def service_type(self) -> ServiceType:
        """Type of service provided."""
        ...
    
    @property
    def status(self) -> ProviderStatus:
        """Current provider status."""
        ...
    
    # Lifecycle Management
    async def initialize(self) -> None:
        """
        Initialize service provider.
        
        Raises:
            ServiceInitializationError: If initialization fails
        """
        ...
    
    async def shutdown(self) -> None:
        """
        Shutdown service provider.
        
        Raises:
            ServiceStopError: If shutdown fails
        """
        ...
    
    # Health Monitoring
    async def health_check(self) -> ServiceHealth:
        """
        Check service health.
        
        Returns:
            Detailed health information
        """
        ...
    
    def record_operation(self, *, success: bool, error: str | None = None) -> None:
        """
        Record operation result for monitoring.
        
        Args:
            success: Whether operation was successful
            error: Error message if operation failed
        """
        ...
```

### ChunkingService Protocol

```python
@runtime_checkable
class ChunkingService(Protocol):
    """Protocol for content chunking services."""
    
    async def chunk_content(
        self, 
        content: str, 
        file_path: str | None = None
    ) -> list[str]:
        """
        Chunk content into segments.
        
        Args:
            content: Content to chunk
            file_path: Optional file path for context
            
        Returns:
            List of content chunks
            
        Raises:
            ChunkingError: If chunking fails
            ValidationError: If content is invalid
        """
        ...
    
    def get_chunk_stats(self, chunks: list[str]) -> dict[str, Any]:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk statistics
        """
        ...
```

### FilteringService Protocol

```python
from pathlib import Path

@runtime_checkable
class FilteringService(Protocol):
    """Protocol for file filtering services."""
    
    async def discover_files(
        self, 
        base_path: Path, 
        patterns: list[str] | None = None
    ) -> list[Path]:
        """
        Discover files based on filtering criteria.
        
        Args:
            base_path: Base directory to search
            patterns: Optional glob patterns to match
            
        Returns:
            List of discovered file paths
            
        Raises:
            FilteringError: If discovery fails
            ValidationError: If parameters are invalid
        """
        ...
    
    def should_include_file(self, file_path: Path) -> bool:
        """
        Check if file should be included.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be included
        """
        ...
    
    def get_file_type(self, file_path: Path) -> str | None:
        """
        Get file type from path.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            File type string or None if unknown
        """
        ...
```

## 🔌 Plugin Protocols

### PluginInterface Protocol

```python
from codeweaver.cw_types import ComponentType, BaseCapabilities, BaseComponentInfo

@runtime_checkable
class PluginInterface(Protocol):
    """Universal interface that all plugins must implement."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """
        Get unique plugin name.
        
        Returns:
            Unique plugin identifier
        """
        ...
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        """
        Get component type provided by plugin.
        
        Returns:
            Component type enum value
        """
        ...
    
    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        """
        Get plugin capabilities.
        
        Returns:
            Capabilities description
        """
        ...
    
    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        """
        Get detailed component information.
        
        Returns:
            Component metadata
        """
        ...
    
    @classmethod
    def validate_config(cls, config: BaseComponentConfig) -> ValidationResult:
        """
        Validate configuration for this plugin.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors/warnings
        """
        ...
    
    @classmethod
    def get_dependencies(cls) -> list[str]:
        """
        Get list of required dependencies.
        
        Returns:
            List of dependency names
        """
        ...
```

### Specialized Plugin Interfaces

#### BackendPlugin

```python
class BackendPlugin(PluginInterface):
    """Plugin interface for vector backends."""
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.BACKEND
    
    @classmethod
    def get_backend_class(cls) -> type[VectorBackend]:
        """
        Get backend implementation class.
        
        Returns:
            Backend class implementing VectorBackend protocol
        """
        ...
```

#### ProviderPlugin

```python
class ProviderPlugin(PluginInterface):
    """Plugin interface for embedding/reranking providers."""
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.PROVIDER
    
    @classmethod
    def get_provider_class(cls) -> type[EmbeddingProvider | RerankProvider]:
        """
        Get provider implementation class.
        
        Returns:
            Provider class implementing appropriate protocol
        """
        ...
```

#### SourcePlugin

```python
class SourcePlugin(PluginInterface):
    """Plugin interface for data sources."""
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.SOURCE
    
    @classmethod
    def get_source_class(cls) -> type[DataSource]:
        """
        Get source implementation class.
        
        Returns:
            Source class implementing DataSource protocol
        """
        ...
```

## 📋 Data Structures

### Core Data Types

#### VectorPoint
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class VectorPoint:
    """Represents a vector with metadata."""
    
    id: str | int
    vector: list[float]
    payload: dict[str, Any] | None = None
    sparse_vector: dict[str, float] | None = None
```

#### SearchResult
```python
@dataclass
class SearchResult:
    """Represents a search result."""
    
    id: str | int
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None
```

#### ContentItem
```python
@dataclass
class ContentItem:
    """Represents a content item from a data source."""
    
    id: str
    path: str
    content_type: str
    size: int | None = None
    modified_time: float | None = None
    metadata: dict[str, Any] | None = None
```

### Configuration Types

#### ValidationResult
```python
@dataclass
class ValidationResult:
    """Result of configuration validation."""
    
    is_valid: bool
    errors: list[str] | None = None
    warnings: list[str] | None = None
    metadata: dict[str, Any] | None = None
```

#### ServiceHealth
```python
@dataclass
class ServiceHealth:
    """Service health information."""
    
    status: ProviderStatus
    last_check: float
    error_count: int = 0
    success_count: int = 0
    last_error: str | None = None
    response_time_ms: float | None = None
```

## 🔍 Enums and Constants

### ComponentType
```python
from enum import Enum

class ComponentType(Enum):
    """Types of components in CodeWeaver."""
    
    PROVIDER = "provider"
    BACKEND = "backend"  
    SOURCE = "source"
    SERVICE = "service"
```

### DistanceMetric
```python
class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
```

### ServiceType  
```python
class ServiceType(Enum):
    """Types of services."""
    
    CHUNKING = "chunking"
    FILTERING = "filtering"
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
```

## 🚀 Usage Examples

### Implementing EmbeddingProvider

```python
class MyEmbeddingProvider:
    def __init__(self, config: MyConfig):
        self.config = config
    
    @property
    def provider_name(self) -> str:
        return "my_provider"
    
    @property
    def model_name(self) -> str:
        return self.config.model
    
    @property
    def dimension(self) -> int:
        return 1536
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Implementation here
        pass
    
    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    def get_provider_info(self) -> EmbeddingProviderInfo:
        return EmbeddingProviderInfo(
            provider_name=self.provider_name,
            model_name=self.model_name,
            embedding_dimension=self.dimension
        )
    
    async def health_check(self) -> bool:
        try:
            await self.embed_query("test")
            return True
        except Exception:
            return False
```

### Implementing VectorBackend

```python
class MyVectorBackend:
    def __init__(self, config: MyConfig):
        self.config = config
        self.client = None
    
    async def create_collection(
        self, 
        name: str, 
        dimension: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE, 
        **kwargs: Any
    ) -> None:
        # Implementation here
        pass
    
    async def upsert_vectors(
        self, 
        collection_name: str, 
        vectors: list[VectorPoint]
    ) -> None:
        # Implementation here
        pass
    
    async def search_vectors(
        self, 
        collection_name: str, 
        query_vector: list[float], 
        limit: int = 10, 
        search_filter: SearchFilter | None = None, 
        score_threshold: float | None = None, 
        **kwargs: Any
    ) -> list[SearchResult]:
        # Implementation here
        pass
    
    async def health_check(self) -> bool:
        # Implementation here
        pass
```

## 🚀 Next Steps

- **[Extension Development →](../extension-development/)**: Start building extensions
- **[Examples →](../examples/)**: See working implementations
- **[Testing Framework →](../extension-development/testing.md)**: Test your protocols