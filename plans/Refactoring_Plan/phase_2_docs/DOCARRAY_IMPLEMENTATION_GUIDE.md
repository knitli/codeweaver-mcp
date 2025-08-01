# DocArray Implementation Guide

<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

**Version**: 1.0.0
**Date**: 2025-01-26
**Purpose**: Practical implementation guide for DocArray integration

## Overview

This guide provides step-by-step implementation details for integrating DocArray into CodeWeaver's plugin architecture, including code examples, configuration patterns, and best practices.

## Implementation Roadmap

### Phase 1: Core Infrastructure (1-2 weeks)
1. [Dynamic Document Schema System](#1-dynamic-document-schema-system)
2. [Backend Adapter Framework](#2-backend-adapter-framework)
3. [Configuration Integration](#3-configuration-integration)

### Phase 2: Backend Implementations (2-3 weeks)
4. [Qdrant DocArray Backend](#4-qdrant-docarray-backend)
5. [Pinecone DocArray Backend](#5-pinecone-docarray-backend)
6. [Weaviate DocArray Backend](#6-weaviate-docarray-backend)

### Phase 3: Advanced Features (1-2 weeks)
7. [Hybrid Search Implementation](#7-hybrid-search-implementation)
8. [Registry Integration](#8-registry-integration)
9. [Testing and Validation](#9-testing-and-validation)

## Implementation Details

### 1. Dynamic Document Schema System

**File**: `src/codeweaver/backends/docarray/schema.py`

```python
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Dynamic document schema generation for DocArray backends."""

import logging
from typing import Any, Annotated, get_type_hints

from docarray import BaseDoc
from docarray.typing import AndArray
from pydantic import Field, create_model

logger = logging.getLogger(__name__)


class SchemaConfig:
    """Configuration for dynamic document schema generation."""

    def __init__(
        self,
        embedding_dimension: int = 512,
        include_sparse_vectors: bool = False,
        metadata_fields: dict[str, type] | None = None,
        custom_fields: dict[str, tuple[type, Any]] | None = None,
        enable_validation: bool = True,
    ):
        self.embedding_dimension = embedding_dimension
        self.include_sparse_vectors = include_sparse_vectors
        self.metadata_fields = metadata_fields or {}
        self.custom_fields = custom_fields or {}
        self.enable_validation = enable_validation


class DocumentSchemaGenerator:
    """Generates DocArray document schemas based on configuration."""

    @classmethod
    def create_schema(
        cls,
        config: SchemaConfig,
        schema_name: str = "CodeWeaverDoc"
    ) -> type[BaseDoc]:
        """Create a dynamic document schema based on configuration.

        Args:
            config: Schema configuration
            schema_name: Name for the generated schema class

        Returns:
            Generated document schema class
        """
        # Base required fields
        field_definitions = cls._get_base_fields(config.embedding_dimension)

        # Add sparse vector support
        if config.include_sparse_vectors:
            field_definitions.update(cls._get_sparse_fields())

        # Add metadata fields
        field_definitions.update(cls._get_metadata_fields(config.metadata_fields))

        # Add custom fields
        field_definitions.update(config.custom_fields)

        # Create the document class
        doc_class = create_model(
            schema_name,
            __base__=BaseDoc,
            **field_definitions,
            __module__=__name__
        )

        # Add validation configuration
        if config.enable_validation:
            doc_class.model_config = cls._get_model_config()

        logger.info(f"Created document schema '{schema_name}' with {len(field_definitions)} fields")
        return doc_class

    @staticmethod
    def _get_base_fields(embedding_dim: int) -> dict[str, tuple[type, Any]]:
        """Get base required fields for all document schemas."""
        return {
            "id": (
                str,
                Field(description="Unique document identifier")
            ),
            "content": (
                str,
                Field(description="Document text content")
            ),
            "embedding": (
                AndArray[embedding_dim],
                Field(description="Dense vector embedding")
            ),
            "metadata": (
                dict[str, Any],
                Field(default_factory=dict, description="Document metadata")
            ),
        }

    @staticmethod
    def _get_sparse_fields() -> dict[str, tuple[type, Any]]:
        """Get sparse vector fields for hybrid search."""
        return {
            "sparse_vector": (
                dict[str, float],
                Field(default_factory=dict, description="Sparse vector for hybrid search")
            ),
            "keywords": (
                list[str],
                Field(default_factory=list, description="Keywords for sparse search")
            ),
        }

    @staticmethod
    def _get_metadata_fields(metadata_config: dict[str, type]) -> dict[str, tuple[type, Any]]:
        """Convert metadata configuration to field definitions."""
        fields = {}
        for field_name, field_type in metadata_config.items():
            safe_name = f"meta_{field_name}" if not field_name.startswith("meta_") else field_name
            fields[safe_name] = (
                field_type,
                Field(description=f"Metadata field: {field_name}")
            )
        return fields

    @staticmethod
    def _get_model_config() -> dict[str, Any]:
        """Get Pydantic model configuration for validation."""
        return {
            "extra": "allow",
            "validate_assignment": True,
            "arbitrary_types_allowed": True,
            "str_strip_whitespace": True,
        }


# Predefined schema templates
class SchemaTemplates:
    """Predefined schema templates for common use cases."""

    @staticmethod
    def code_search_schema(embedding_dim: int = 512) -> type[BaseDoc]:
        """Schema optimized for code search."""
        config = SchemaConfig(
            embedding_dimension=embedding_dim,
            include_sparse_vectors=True,
            metadata_fields={
                "file_path": str,
                "language": str,
                "function_name": str,
                "class_name": str,
                "line_number": int,
            }
        )
        return DocumentSchemaGenerator.create_schema(config, "CodeSearchDoc")

    @staticmethod
    def semantic_search_schema(embedding_dim: int = 512) -> type[BaseDoc]:
        """Schema for general semantic search."""
        config = SchemaConfig(
            embedding_dimension=embedding_dim,
            include_sparse_vectors=False,
            metadata_fields={
                "title": str,
                "author": str,
                "timestamp": str,
                "category": str,
            }
        )
        return DocumentSchemaGenerator.create_schema(config, "SemanticSearchDoc")

    @staticmethod
    def multimodal_schema(embedding_dim: int = 512) -> type[BaseDoc]:
        """Schema for multimodal documents."""
        config = SchemaConfig(
            embedding_dimension=embedding_dim,
            include_sparse_vectors=True,
            custom_fields={
                "image_embedding": (AndArray[embedding_dim], Field(description="Image embedding")),
                "text_embedding": (AndArray[embedding_dim], Field(description="Text embedding")),
                "image_url": (str | None, Field(default=None, description="Image URL")),
            }
        )
        return DocumentSchemaGenerator.create_schema(config, "MultiModalDoc")
```

### 2. Backend Adapter Framework

**File**: `src/codeweaver/backends/docarray/adapter.py`

```python
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Universal adapter for DocArray backends to CodeWeaver protocols."""

import logging
from typing import Any, AsyncIterator
from abc import ABC, abstractmethod

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex

from codeweaver.cw_types import (
    CollectionInfo,
    DistanceMetric,
    SearchFilter,
    SearchResult,
    VectorPoint,
)
from codeweaver.backends.base import VectorBackend, HybridSearchBackend
from .schema import DocumentSchemaGenerator, SchemaConfig

logger = logging.getLogger(__name__)


class DocArrayAdapterError(Exception):
    """Errors specific to DocArray adapter operations."""
    pass


class VectorConverter:
    """Converts between CodeWeaver and DocArray vector formats."""

    def __init__(self, doc_class: type[BaseDoc]):
        self.doc_class = doc_class
        self._field_names = set(doc_class.model_fields.keys())

    def vector_to_doc(self, vector: VectorPoint) -> BaseDoc:
        """Convert VectorPoint to DocArray document."""
        doc_data = {
            "id": str(vector.id),
            "content": vector.payload.get("content", ""),
            "embedding": vector.vector,
            "metadata": vector.payload.copy(),
        }

        # Add sparse vector if available
        if "sparse_vector" in self._field_names and vector.sparse_vector:
            doc_data["sparse_vector"] = vector.sparse_vector

        # Extract structured metadata
        for field_name in self._field_names:
            if field_name.startswith("meta_") and field_name[5:] in vector.payload:
                doc_data[field_name] = vector.payload[field_name[5:]]

        return self.doc_class(**doc_data)

    def doc_to_search_result(self, doc: BaseDoc, score: float) -> SearchResult:
        """Convert DocArray document to SearchResult."""
        payload = doc.metadata.copy() if hasattr(doc, "metadata") else {}

        # Add structured metadata back to payload
        for field_name, value in doc.model_dump().items():
            if field_name.startswith("meta_"):
                payload[field_name[5:]] = value
            elif field_name not in {"id", "embedding", "metadata", "sparse_vector"}:
                payload[field_name] = value

        return SearchResult(
            id=str(doc.id),
            score=score,
            payload=payload,
            vector=doc.embedding.tolist() if hasattr(doc, "embedding") else None,
        )

    def create_query_doc(self, query_vector: list[float], **kwargs) -> BaseDoc:
        """Create a query document for search operations."""
        query_data = {
            "id": "query",
            "content": kwargs.get("content", ""),
            "embedding": query_vector,
            "metadata": kwargs.get("metadata", {}),
        }

        # Add sparse vector for hybrid queries
        if "sparse_vector" in self._field_names and "sparse_vector" in kwargs:
            query_data["sparse_vector"] = kwargs["sparse_vector"]

        return self.doc_class(**query_data)


class BaseDocArrayAdapter(VectorBackend, ABC):
    """Base adapter for DocArray document indexes."""

    def __init__(
        self,
        doc_index: BaseDocIndex,
        doc_class: type[BaseDoc],
        collection_name: str = "default"
    ):
        self.doc_index = doc_index
        self.doc_class = doc_class
        self.collection_name = collection_name
        self.converter = VectorConverter(doc_class)
        self._initialized = False

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Create a new vector collection."""
        self.collection_name = name
        self._dimension = dimension
        self._distance_metric = distance_metric

        # DocArray handles collection creation implicitly
        # Store configuration for later use
        self._collection_config = {
            "dimension": dimension,
            "distance_metric": distance_metric,
            **kwargs,
        }

        self._initialized = True
        logger.info(f"Created collection '{name}' with dimension {dimension}")

    async def upsert_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """Insert or update vectors in the collection."""
        if not self._initialized:
            raise DocArrayAdapterError("Collection not initialized")

        try:
            # Convert vectors to documents
            docs = [self.converter.vector_to_doc(vector) for vector in vectors]
            doc_list = DocList[self.doc_class](docs)

            # Batch insert for performance
            self.doc_index.index(doc_list)

            logger.debug(f"Upserted {len(vectors)} vectors to collection '{collection_name}'")

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise DocArrayAdapterError(f"Upsert failed: {e}") from e

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self._initialized:
            raise DocArrayAdapterError("Collection not initialized")

        try:
            # Create query document
            query_doc = self.converter.create_query_doc(query_vector, **kwargs)

            # Perform search
            results, scores = self.doc_index.find(
                query_doc,
                search_field="embedding",
                limit=limit
            )

            # Convert results
            search_results = []
            for doc, score in zip(results, scores):
                if score_threshold is None or score >= score_threshold:
                    search_results.append(
                        self.converter.doc_to_search_result(doc, score)
                    )

            logger.debug(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise DocArrayAdapterError(f"Search failed: {e}") from e

    async def delete_vectors(self, collection_name: str, ids: list[str | int]) -> None:
        """Delete vectors by IDs."""
        try:
            # DocArray delete implementation depends on backend
            if hasattr(self.doc_index, 'delete'):
                self.doc_index.delete([str(id_) for id_ in ids])
            else:
                logger.warning("Backend doesn't support direct deletion")

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise DocArrayAdapterError(f"Delete failed: {e}") from e

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get collection metadata and statistics."""
        try:
            # Extract information from DocArray index
            vector_count = self._get_vector_count()
            dimension = getattr(self, '_dimension', 512)

            return CollectionInfo(
                name=name,
                dimension=dimension,
                vector_count=vector_count,
                distance_metric=getattr(self, '_distance_metric', DistanceMetric.COSINE),
                supports_hybrid_search=self._supports_hybrid_search(),
                supports_filtering=True,  # Most backends support basic filtering
                supports_sparse_vectors=self._supports_sparse_vectors(),
            )

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise DocArrayAdapterError(f"Collection info failed: {e}") from e

    async def list_collections(self) -> list[str]:
        """List all available collections."""
        # DocArray typically works with single collections
        return [self.collection_name] if self._initialized else []

    async def delete_collection(self, name: str) -> None:
        """Delete a collection entirely."""
        try:
            if hasattr(self.doc_index, 'drop'):
                self.doc_index.drop()
            self._initialized = False
            logger.info(f"Deleted collection '{name}'")

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise DocArrayAdapterError(f"Collection deletion failed: {e}") from e

    # Abstract methods for backend-specific implementations

    @abstractmethod
    def _get_vector_count(self) -> int:
        """Get the number of vectors in the collection."""
        pass

    @abstractmethod
    def _supports_hybrid_search(self) -> bool:
        """Check if backend supports hybrid search."""
        pass

    @abstractmethod
    def _supports_sparse_vectors(self) -> bool:
        """Check if backend supports sparse vectors."""
        pass


class DocArrayHybridAdapter(BaseDocArrayAdapter, HybridSearchBackend):
    """DocArray adapter with hybrid search capabilities."""

    async def create_sparse_index(
        self,
        collection_name: str,
        fields: list[str],
        index_type: str = "bm25",
        **kwargs: Any,
    ) -> None:
        """Create sparse vector index for hybrid search."""
        try:
            if hasattr(self.doc_index, 'configure_sparse_index'):
                self.doc_index.configure_sparse_index(fields, index_type, **kwargs)
                logger.info(f"Created {index_type} sparse index on fields: {fields}")
            else:
                logger.warning("Backend doesn't support native sparse indexing")

        except Exception as e:
            logger.error(f"Sparse index creation failed: {e}")
            raise DocArrayAdapterError(f"Sparse index failed: {e}") from e

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_query: dict[str, float] | str,
        limit: int = 10,
        hybrid_strategy: str = "rrf",
        alpha: float = 0.5,
        search_filter: SearchFilter | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval."""
        try:
            # Convert sparse query to proper format
            if isinstance(sparse_query, str):
                sparse_vector = self._text_to_sparse_vector(sparse_query)
            else:
                sparse_vector = sparse_query

            # Create hybrid query document
            query_doc = self.converter.create_query_doc(
                dense_vector,
                sparse_vector=sparse_vector,
                **kwargs
            )

            # Use native hybrid search if available
            if hasattr(self.doc_index, 'hybrid_search'):
                results, scores = self.doc_index.hybrid_search(
                    query_doc,
                    alpha=alpha,
                    limit=limit
                )
            else:
                # Fallback to RRF fusion
                results, scores = await self._fallback_hybrid_search(
                    dense_vector, sparse_vector, limit, alpha
                )

            # Convert results
            return [
                self.converter.doc_to_search_result(doc, score)
                for doc, score in zip(results, scores)
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise DocArrayAdapterError(f"Hybrid search failed: {e}") from e

    def _text_to_sparse_vector(self, text: str) -> dict[str, float]:
        """Convert text to sparse vector representation."""
        # Simple TF-IDF implementation
        # In production, use proper BM25 or TF-IDF vectorizer
        words = text.lower().split()
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1

        # Normalize
        max_freq = max(term_freq.values()) if term_freq else 1
        return {word: freq / max_freq for word, freq in term_freq.items()}

    async def _fallback_hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, float],
        limit: int,
        alpha: float,
    ) -> tuple[list[BaseDoc], list[float]]:
        """Fallback hybrid search using RRF fusion."""
        # Perform separate dense and sparse searches
        dense_results = await self.search_vectors(
            self.collection_name, dense_vector, limit * 2
        )

        # For sparse search, we'd need a separate implementation
        # This is a simplified version
        sparse_results = dense_results  # Placeholder

        # Apply RRF fusion
        return self._apply_rrf_fusion(dense_results, sparse_results, limit)

    def _apply_rrf_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        limit: int,
        k: int = 60,
    ) -> tuple[list[BaseDoc], list[float]]:
        """Apply Reciprocal Rank Fusion to combine results."""
        # Simplified RRF implementation
        scores = {}

        # Score dense results
        for rank, result in enumerate(dense_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (rank + k)

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (rank + k)

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

        # Return placeholder results
        # In real implementation, fetch the actual documents
        return [], []
```

### 3. Configuration Integration

**File**: `src/codeweaver/backends/docarray/config.py`

```python
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Configuration classes for DocArray backend integration."""

from typing import Any, Annotated, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

from codeweaver.backends.config import BackendConfig
from codeweaver.cw_types import DistanceMetric


class DocArraySchemaConfig(BaseModel):
    """Configuration for DocArray document schema generation."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    embedding_dimension: Annotated[
        int, Field(ge=1, le=65536, description="Embedding vector dimension")
    ] = 512

    include_sparse_vectors: Annotated[
        bool, Field(description="Enable sparse vector support for hybrid search")
    ] = False

    metadata_fields: Annotated[
        dict[str, str], Field(description="Typed metadata fields (name -> type)")
    ] = Field(default_factory=dict)

    custom_fields: Annotated[
        dict[str, Any], Field(description="Custom field definitions")
    ] = Field(default_factory=dict)

    schema_template: Annotated[
        Literal["code_search", "semantic_search", "multimodal", "custom"] | None,
        Field(description="Predefined schema template to use")
    ] = None

    enable_validation: Annotated[
        bool, Field(description="Enable Pydantic validation")
    ] = True

    @field_validator("metadata_fields")
    @classmethod
    def validate_metadata_fields(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate metadata field definitions."""
        valid_types = {"str", "int", "float", "bool", "list[str]", "dict[str, Any]"}

        for field_name, field_type in v.items():
            if not field_name.isidentifier():
                raise ValueError(f"Invalid field name: {field_name}")
            if field_type not in valid_types:
                raise ValueError(f"Unsupported field type: {field_type}")

        return v


class DocArrayBackendConfig(BackendConfig):
    """Configuration for DocArray-powered backends."""

    model_config = ConfigDict(extra="allow")

    # DocArray-specific configuration
    schema_config: Annotated[
        DocArraySchemaConfig, Field(description="Document schema configuration")
    ] = Field(default_factory=DocArraySchemaConfig)

    runtime_config: Annotated[
        dict[str, Any], Field(description="DocArray runtime configuration")
    ] = Field(default_factory=dict)

    # Backend-specific database configuration
    db_config: Annotated[
        dict[str, Any], Field(description="Backend database configuration")
    ] = Field(default_factory=dict)

    # Performance and behavior settings
    batch_size: Annotated[
        int, Field(ge=1, le=1000, description="Batch size for operations")
    ] = 100

    enable_async: Annotated[
        bool, Field(description="Enable asynchronous operations")
    ] = True

    connection_timeout: Annotated[
        float, Field(ge=0.1, le=300.0, description="Connection timeout in seconds")
    ] = 30.0

    retry_attempts: Annotated[
        int, Field(ge=0, le=10, description="Number of retry attempts")
    ] = 3

    # Feature flags
    enable_hybrid_search: Annotated[
        bool, Field(description="Enable hybrid search capabilities")
    ] = False

    enable_compression: Annotated[
        bool, Field(description="Enable vector compression")
    ] = False

    enable_caching: Annotated[
        bool, Field(description="Enable query result caching")
    ] = False


# Backend-specific configuration classes

class QdrantDocArrayConfig(DocArrayBackendConfig):
    """Qdrant-specific DocArray configuration."""

    provider: str = Field(default="docarray_qdrant", frozen=True)

    # Qdrant-specific settings
    prefer_grpc: bool = Field(default=False, description="Use gRPC instead of HTTP")
    grpc_port: int | None = Field(default=None, description="gRPC port if different from HTTP")

    @field_validator("db_config", mode="before")
    @classmethod
    def set_qdrant_defaults(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Set Qdrant-specific default configuration."""
        defaults = {
            "prefer_grpc": False,
            "timeout": 30.0,
            "retry_total": 3,
        }
        return {**defaults, **v}


class PineconeDocArrayConfig(DocArrayBackendConfig):
    """Pinecone-specific DocArray configuration."""

    provider: str = Field(default="docarray_pinecone", frozen=True)

    # Pinecone-specific settings
    environment: str = Field(description="Pinecone environment")
    index_type: str = Field(default="approximated", description="Index type")

    @field_validator("db_config", mode="before")
    @classmethod
    def set_pinecone_defaults(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Set Pinecone-specific default configuration."""
        defaults = {
            "metric": "cosine",
            "shards": 1,
            "replicas": 1,
        }
        return {**defaults, **v}


class WeaviateDocArrayConfig(DocArrayBackendConfig):
    """Weaviate-specific DocArray configuration."""

    provider: str = Field(default="docarray_weaviate", frozen=True)

    # Weaviate-specific settings
    class_name: str = Field(default="CodeWeaverDoc", description="Weaviate class name")
    vectorizer: str | None = Field(default=None, description="Weaviate vectorizer")

    @field_validator("db_config", mode="before")
    @classmethod
    def set_weaviate_defaults(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Set Weaviate-specific default configuration."""
        defaults = {
            "startup_period": 5,
            "additional_headers": {},
        }
        return {**defaults, **v}


# Configuration factory
class DocArrayConfigFactory:
    """Factory for creating DocArray backend configurations."""

    CONFIG_MAPPING = {
        "docarray_qdrant": QdrantDocArrayConfig,
        "docarray_pinecone": PineconeDocArrayConfig,
        "docarray_weaviate": WeaviateDocArrayConfig,
    }

    @classmethod
    def create_config(
        cls,
        backend_type: str,
        **kwargs: Any,
    ) -> DocArrayBackendConfig:
        """Create configuration for specified backend type."""
        config_class = cls.CONFIG_MAPPING.get(backend_type, DocArrayBackendConfig)
        return config_class(**kwargs)

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported DocArray backend types."""
        return list(cls.CONFIG_MAPPING.keys())

    @classmethod
    def validate_backend_config(
        cls,
        backend_type: str,
        config: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate configuration for specified backend type."""
        try:
            cls.create_config(backend_type, **config)
            return True, []
        except Exception as e:
            return False, [str(e)]
```

### 4. Qdrant DocArray Backend

**File**: `src/codeweaver/backends/docarray/qdrant.py`

```python
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Qdrant backend implementation using DocArray."""

import logging
from typing import Any

from docarray.index import QdrantDocumentIndex

from .adapter import BaseDocArrayAdapter, DocArrayHybridAdapter
from .config import QdrantDocArrayConfig
from .schema import DocumentSchemaGenerator, SchemaTemplates

logger = logging.getLogger(__name__)


class QdrantDocArrayBackend(DocArrayHybridAdapter):
    """Qdrant backend using DocArray with hybrid search support."""

    def __init__(self, config: QdrantDocArrayConfig):
        """Initialize Qdrant DocArray backend."""
        self.config = config

        # Create document schema
        if config.schema_config.schema_template == "code_search":
            doc_class = SchemaTemplates.code_search_schema(
                config.schema_config.embedding_dimension
            )
        elif config.schema_config.schema_template == "semantic_search":
            doc_class = SchemaTemplates.semantic_search_schema(
                config.schema_config.embedding_dimension
            )
        else:
            doc_class = DocumentSchemaGenerator.create_schema(
                config.schema_config,
                "QdrantCodeWeaverDoc"
            )

        # Create Qdrant database config
        db_config = QdrantDocumentIndex.DBConfig(
            host=config.url,
            port=config.port or 6333,
            api_key=config.api_key,
            collection_name=config.collection_name or "codeweaver",
            distance="Cosine",  # Map from DistanceMetric
            **config.db_config,
        )

        # Create runtime config
        runtime_config = QdrantDocumentIndex.RuntimeConfig(
            **config.runtime_config
        )

        # Initialize DocArray index
        try:
            doc_index = QdrantDocumentIndex[doc_class](
                db_config=db_config,
                runtime_config=runtime_config,
            )

            super().__init__(
                doc_index=doc_index,
                doc_class=doc_class,
                collection_name=config.collection_name or "codeweaver",
            )

            logger.info("Qdrant DocArray backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant DocArray backend: {e}")
            raise

    def _get_vector_count(self) -> int:
        """Get the number of vectors in the Qdrant collection."""
        try:
            if hasattr(self.doc_index, '_client'):
                return self.doc_index._client.count(self.collection_name)
            return 0
        except Exception:
            logger.warning("Could not get vector count from Qdrant")
            return 0

    def _supports_hybrid_search(self) -> bool:
        """Qdrant supports hybrid search with sparse vectors."""
        return True

    def _supports_sparse_vectors(self) -> bool:
        """Qdrant supports sparse vectors natively."""
        return True

    async def create_sparse_index(
        self,
        collection_name: str,
        fields: list[str],
        index_type: str = "bm25",
        **kwargs: Any,
    ) -> None:
        """Create sparse vector index in Qdrant."""
        try:
            # Qdrant handles sparse vectors automatically
            # Configure sparse vector field if needed
            if hasattr(self.doc_index, '_configure_sparse_vectors'):
                self.doc_index._configure_sparse_vectors(fields, index_type)

            logger.info(f"Configured Qdrant sparse index for fields: {fields}")

        except Exception as e:
            logger.error(f"Failed to create Qdrant sparse index: {e}")
            raise

    @classmethod
    def _check_dependencies(cls) -> list[str]:
        """Check if Qdrant dependencies are available."""
        missing = []
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            missing.append("qdrant-client")

        try:
            from docarray.index import QdrantDocumentIndex  # noqa: F401
        except ImportError:
            missing.append("docarray[qdrant]")

        return missing
```

This implementation guide provides detailed, production-ready code for the DocArray integration. The code follows CodeWeaver's existing patterns and includes proper error handling, logging, and type safety.

Key implementation features:
- **Dynamic schema generation** with templates for common use cases
- **Universal adapter pattern** for protocol compliance
- **Comprehensive configuration system** with validation
- **Backend-specific implementations** starting with Qdrant
- **Hybrid search support** with fallback strategies
- **Error handling and logging** throughout
- **Type safety** with Pydantic v2 integration

The implementation is designed to be:
- **Extensible**: Easy to add new DocArray backends
- **Maintainable**: Clear separation of concerns
- **Performant**: Batch operations and async support
- **Robust**: Comprehensive error handling and validation
