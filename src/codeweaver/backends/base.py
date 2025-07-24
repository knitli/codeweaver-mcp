# sourcery skip: lambdas-should-be-short
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Vector database backend protocols and supporting data structures.

Provides comprehensive abstractions for 15+ vector databases with hybrid search
support, designed for runtime flexibility and extensibility.
"""

import enum
import operator
import re
import types

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


class DistanceMetric(enum.Enum):
    """Supported distance metrics across vector databases."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"  # L2
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"  # L1


class HybridStrategy(enum.Enum):
    """Hybrid search fusion strategies."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    DBSF = "dbsf"  # Distribution-Based Score Fusion
    LINEAR = "linear"  # Linear combination
    CONVEX = "convex"  # Convex combination

    @classmethod
    def from_string(cls, strategy: str) -> "HybridStrategy":
        """
        Create a HybridStrategy from a string.

        Args:
            strategy: Strategy name (e.g., "rrf", "dbsf", "linear", "convex")

        Returns:
            Corresponding HybridStrategy enum member

        Raises:
            ValueError: If strategy is not recognized
        """
        try:
            return cls[strategy.upper().replace("-", "_").replace(" ", "_")]
        except KeyError as e:
            raise ValueError(f"Unsupported hybrid strategy: {strategy}") from e


@dataclass
class VectorPoint:
    """Universal vector point structure compatible with all backends."""

    iden: str | int
    vector: list[float]
    payload: dict[str, Any] | None = None
    sparse_vector: dict[int, float] | None = None  # For hybrid search


@dataclass
class SearchResult:
    """Unified search result format across all vector databases."""

    iden: str | int
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None  # Optional, depends on backend

    # Additional metadata for debugging/optimization
    backend_metadata: dict[str, Any] | None = None


@dataclass
class CollectionInfo:
    """Collection metadata structure for introspection."""

    name: str
    dimension: int
    points_count: int
    distance_metric: DistanceMetric
    indexed: bool

    # Backend-specific capabilities
    supports_sparse_vectors: bool = False
    supports_hybrid_search: bool = False
    supports_filtering: bool = True
    supports_updates: bool = True

    # Storage and performance info
    storage_type: str | None = None  # "memory", "disk", "hybrid"
    index_type: str | None = None  # "hnsw", "ivf", "flat", etc.
    backend_info: dict[str, Any] | None = None


class FilterOperator(enum.Enum):
    """Supported filter operators for vector search."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal
    IN = "in"  # In list
    NIN = "nin"  # Not in list
    CONTAINS = "contains"  # Contains substring
    REGEX = "regex"  # Regular expression match

    @property
    def is_comparison(self) -> bool:
        """Check if operator is a comparison (eq, ne, gt, ge, lt, le)."""
        return self in {self.EQ, self.NE, self.GT, self.GE, self.LT, self.LE}

    @property
    def is_set_operation(self) -> bool:
        """Check if operator is a set operation (in, nin)."""
        return self in {self.IN, self.NIN}

    @property
    def is_text_operation(self) -> bool:
        """Check if operator is a text operation (contains, regex)."""
        return self in {self.CONTAINS, self.REGEX}

    @property
    def operator(self) -> types.FunctionType:
        """Get the Python operator object."""
        return {
            self.EQ: operator.eq,
            self.NE: operator.ne,
            self.GT: operator.gt,
            self.GE: operator.ge,
            self.LT: operator.lt,
            self.LE: operator.le,
            self.IN: lambda a, b: a in b,
            self.NIN: lambda a, b: a not in b,
            self.CONTAINS: lambda a, b: operator.contains(a, b),
            self.REGEX: lambda a, b: bool(re.search(b, a)) if isinstance(a, str) else False,
        }[self]

    @property
    def operator_symbols(self) -> tuple[str, ...]:
        """Get the operator symbols for this filter operator."""
        return {
            self.EQ: ("==",),
            self.NE: ("!=",),
            self.GT: (">",),
            self.GE: (">=",),
            self.LT: ("<",),
            self.LE: ("<=",),
            self.IN: ("in",),
            self.NIN: ("not in", "nin", "notin", "!in"),
            self.CONTAINS: ("contains", "has"),
            self.REGEX: ("regex", "match", "matches", "regexp", "re", "reg", "r", "s/"),
        }[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> "FilterOperator":
        """
        Create a FilterOperator from a symbol.

        Args:
            symbol: Operator symbol (e.g., "==", "!=", "in", "contains")

        Returns:
            Corresponding FilterOperator enum member

        Raises:
            ValueError: If symbol is not recognized
        """
        normalized_symbol = symbol.strip().lower()
        if member := next(
            op for op in cls if normalized_symbol in op.operator_symbols
        ):
            return member
        raise ValueError(f"Unsupported filter operator symbol: {symbol}")


@dataclass
class FilterCondition:
    """Universal filter condition for vector search."""

    field: str
    operator: FilterOperator
    value: Any


@dataclass
class SearchFilter:
    """Composite filter for vector search operations."""

    conditions: list[FilterCondition] | None = None
    must: list["SearchFilter"] | None = None  # AND logic
    should: list["SearchFilter"] | None = None  # OR logic
    must_not: list["SearchFilter"] | None = None  # NOT logic


@runtime_checkable
class VectorBackend(Protocol):
    """
    Core vector database backend protocol.

    Supports 15+ vector databases including:
    - **Cloud Native**: Pinecone, Qdrant Cloud, Weaviate Cloud
    - **Self-Hosted**: Qdrant, Weaviate, Chroma, Milvus, Vespa
    - **Database Extensions**: pgvector (PostgreSQL), vector (SQLite), Redis
    - **In-Memory**: FAISS, Annoy, ScaNN
    - **Multi-Modal**: LanceDB, Marqo
    """

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """
        Create a new vector collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric for similarity computation
            **kwargs: Backend-specific options

        Backend Compatibility:
        - Qdrant: Uses VectorParams with Distance enum conversion
        - Pinecone: Maps to index creation with metric parameter
        - Chroma: Creates collection with metadata
        - Weaviate: Creates class schema with vectorizer config
        - pgvector: Creates table with vector column
        - FAISS: Builds index with specified metric
        """
        ...

    async def upsert_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """
        Insert or update vectors in the collection.

        Args:
            collection_name: Target collection
            vectors: List of vector points to upsert

        Backend Compatibility:
        - Qdrant: Converts to PointStruct format
        - Pinecone: Uses upsert API with (iden, vector, metadata) tuples
        - Chroma: Adds documents with embeddings and metadata
        - Weaviate: Creates objects with vector and properties
        - pgvector: Inserts/updates rows with vector data
        - FAISS: Builds batch insert operations
        """
        ...

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            collection_name: Target collection
            query_vector: Query embedding
            limit: Maximum number of results
            search_filter: Optional filtering conditions
            score_threshold: Minimum similarity score
            **kwargs: Backend-specific search options

        Backend Compatibility:
        - Qdrant: Converts SearchFilter to Filter with FieldCondition
        - Pinecone: Maps to query API with metadata filters
        - Chroma: Uses where clause for filtering
        - Weaviate: Converts to GraphQL where filters
        - pgvector: Builds SQL WHERE clauses
        - FAISS: Applies post-filtering on results
        """
        ...

    async def delete_vectors(self, collection_name: str, ids: list[str | int]) -> None:
        """
        Delete vectors by IDs.

        Args:
            collection_name: Target collection
            ids: Vector IDs to delete

        Backend Compatibility:
        - Qdrant: Uses delete API with point IDs
        - Pinecone: Deletes by vector IDs
        - Chroma: Deletes by document IDs
        - Weaviate: Deletes objects by UUID
        - pgvector: SQL DELETE operations
        - FAISS: Removes from index (rebuild required)
        """
        ...

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """
        Get collection metadata and statistics.

        Args:
            name: Collection name

        Returns:
            Collection information including capabilities

        Backend Compatibility:
        - Qdrant: Gets collection info and config
        - Pinecone: Describes index statistics
        - Chroma: Collection metadata and count
        - Weaviate: Schema and object count
        - pgvector: Table schema and row count
        - FAISS: Index dimensions and total vectors
        """
        ...

    async def list_collections(self) -> list[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        ...

    async def delete_collection(self, name: str) -> None:
        """
        Delete a collection entirely.

        Args:
            name: Collection name to delete
        """
        ...


@runtime_checkable
class HybridSearchBackend(VectorBackend, Protocol):
    """
    Extended protocol for hybrid search capabilities.

    Supports backends with native sparse vector support:
    - **Native Hybrid**: Qdrant (v1.10+), Weaviate, Vespa
    - **Plugin-Based**: Elasticsearch with dense_vector, OpenSearch
    - **Custom Implementation**: Pinecone + BM25, Chroma + sparse index

    For backends without native support, provides fallback mechanisms.
    """

    async def create_sparse_index(
        self,
        collection_name: str,
        fields: list[str],
        index_type: Literal["keyword", "text", "bm25"] = "bm25",
        **kwargs: Any,
    ) -> None:
        """
        Create sparse vector index for hybrid search.

        Args:
            collection_name: Target collection
            fields: Fields to index for sparse search
            index_type: Type of sparse index
            **kwargs: Backend-specific sparse index options

        Backend Compatibility:
        - Qdrant: Creates sparse vector configuration
        - Weaviate: Enables BM25 on text properties
        - Vespa: Creates sparse tensor fields
        - Custom: Builds separate BM25/TF-IDF index
        """
        ...

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_query: dict[str, float] | str,
        limit: int = 10,  # TODO: NEEDS to be configurable
        hybrid_strategy: HybridStrategy = HybridStrategy.RRF,
        alpha: float = 0.5,  # TODO: NEEDS to be configurable
        search_filter: SearchFilter | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        Args:
            collection_name: Target collection
            dense_vector: Dense query embedding
            sparse_query: Sparse query (keywords/BM25 vector)
            limit: Maximum results
            hybrid_strategy: Fusion strategy
            alpha: Dense/sparse balance (0.0-1.0)
            search_filter: Optional filtering
            **kwargs: Backend-specific options

        Returns:
            Fused search results

        Backend Compatibility:
        - Qdrant: Uses Query API with prefetch and fusion
        - Weaviate: Combines vector and BM25 search with hybrid operator
        - Vespa: Uses YQL with multiple ranking expressions
        - Custom: Client-side RRF fusion of separate searches
        """
        ...

    async def update_sparse_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """
        Update sparse vectors for existing points.

        Args:
            collection_name: Target collection
            vectors: Points with sparse_vector data
        """
        ...


@runtime_checkable
class StreamingBackend(Protocol):
    """
    Protocol for backends supporting streaming operations.

    Useful for large-scale indexing and real-time updates.
    """

    async def stream_upsert(
        self,
        collection_name: str,
        vector_stream: Any,  # AsyncIterator[VectorPoint]
        batch_size: int = 100,
    ) -> None:
        """Stream vector upserts for large datasets."""
        ...

    async def stream_search(
        self,
        collection_name: str,
        query_stream: Any,  # AsyncIterator[list[float]]
        limit: int = 10,
    ) -> Any:  # AsyncIterator[list[SearchResult]]
        """Stream search for multiple queries."""
        ...


@runtime_checkable
class TransactionalBackend(Protocol):
    """
    Protocol for backends supporting transactions.

    Ensures ACID properties for complex operations.
    """

    async def begin_transaction(self) -> str:
        """Begin a new transaction, returns transaction ID."""
        ...

    async def commit_transaction(self, transaction_id: str) -> None:
        """Commit the transaction."""
        ...

    async def rollback_transaction(self, transaction_id: str) -> None:
        """Rollback the transaction."""
        ...


class BackendError(Exception):
    """Base exception for backend operations."""

    def __init__(self, message: str, backend_type: str, original_error: Exception | None = None):
        """Initialize backend error with message and type."""
        super().__init__(message)
        self.backend_type = backend_type
        self.original_error = original_error


class BackendConnectionError(BackendError):
    """Backend connection failed."""


class BackendAuthError(BackendError):
    """Backend authentication failed."""


class BackendCollectionNotFoundError(BackendError):
    """Collection does not exist."""


class BackendVectorDimensionMismatchError(BackendError):
    """Vector dimension mismatch."""


class BackendUnsupportedOperationError(BackendError):
    """Operation not supported by backend."""
