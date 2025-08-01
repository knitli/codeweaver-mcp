# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Enums for vector backend providers and related types."""

import operator
import re

from types import FunctionType

from codeweaver.types.base_enum import BaseEnum


class DistanceMetric(BaseEnum):
    """Supported distance metrics across vector databases."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"  # L2
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"  # L1
    OTHER = "other"  # Custom or backend-specific
    NO_METRICS = "none"  # No distance metric


class HybridStrategy(BaseEnum):
    """Hybrid search fusion strategies."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    DBSF = "dbsf"  # Distribution-Based Score Fusion
    LINEAR = "linear"  # Linear combination
    CONVEX = "convex"  # Convex combination
    OTHER = "other"  # Custom or backend-specific
    NO_STRATEGY = "none"  # No hybrid strategy


class FilterOperator(BaseEnum):
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
    def operator(self) -> FunctionType:
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
        if member := next(op for op in cls if normalized_symbol in op.operator_symbols):
            return member
        raise ValueError(f"Unsupported filter operator symbol: {symbol}")

    @classmethod
    def from_string(cls, value: str) -> "FilterOperator":
        """
        Convert a string to the corresponding FilterOperator.
        """
        try:
            resolved_value = super().from_string(value)

        except ValueError:
            return cls.from_symbol(value)

        else:
            if isinstance(resolved_value, cls):
                return resolved_value
            raise ValueError(f"{value} is not a valid {cls.__name__} value") from None


class BackendProvider(BaseEnum):
    """Supported vector database backend providers."""

    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    CHROMA = "chroma"
    FAISS = "faiss"
    PGVECTOR = "pgvector"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"
    MARQO = "marqo"
    VEARCH = "vearch"
    VESPA = "vespa"
    TYPESENSE = "typesense"
    CUSTOM = "custom"

    @property
    def is_cloud_native(self) -> bool:
        """Check if this is a cloud-native provider."""
        return self in {self.QDRANT, self.PINECONE, self.WEAVIATE, self.MARQO}

    @property
    def is_open_source(self) -> bool:
        """Check if this is an open-source provider."""
        return self in {
            self.QDRANT,  # Apache 2.0
            self.WEAVIATE,  # BSD 3-clause
            self.MILVUS,  # Apache 2.0
            self.CHROMA,  # Apache 2.0
            self.FAISS,  # MIT
            self.PGVECTOR,  # MIT-like
            self.OPENSEARCH,  # Apache 2.0
            self.VEARCH,  # Apache 2.0
            self.VESPA,  # Apache 2.0
            self.TYPESENSE,  # GPL-3.0
        }

    @property
    def source_code_available(self) -> bool:
        """Checks if the provider has publicly available source code."""
        return self in {self.REDIS, self.ELASTICSEARCH} or self.is_open_source

    @property
    def supports_hybrid_search(self) -> bool:
        """Check if provider natively supports hybrid search."""
        return self in {self.QDRANT, self.WEAVIATE, self.ELASTICSEARCH, self.OPENSEARCH, self.VESPA}


class SparseIndexType(BaseEnum):
    """Types of sparse indices for hybrid search."""

    KEYWORD = "keyword"  # Traditional keyword/full-text index
    TEXT = "text"  # Text-based sparse index
    BM25 = "bm25"  # BM25 scoring algorithm
    TF_IDF = "tf_idf"  # TF-IDF scoring
    CUSTOM = "custom"  # Custom sparse index implementation

    @property
    def is_text_based(self) -> bool:
        """Check if this is a text-based sparse index."""
        return self in {self.KEYWORD, self.TEXT, self.BM25, self.TF_IDF}


class HybridFusionStrategy(BaseEnum):
    """Strategies for fusing dense and sparse search results."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    DBSF = "dbsf"  # Distribution-Based Score Fusion
    LINEAR = "linear"  # Linear combination (weighted sum)
    CONVEX = "convex"  # Convex combination (normalized weighted sum)
    MAX = "max"  # Maximum score
    MIN = "min"  # Minimum score
    CUSTOM = "custom"  # Custom fusion strategy

    @property
    def requires_weights(self) -> bool:
        """Check if this strategy requires weight parameters."""
        return self in {self.LINEAR, self.CONVEX}

    @property
    def is_rank_based(self) -> bool:
        """Check if this strategy is based on rank rather than scores."""
        return self == self.RRF


class StorageType(BaseEnum):
    """Types of storage backends."""

    DISK = "disk"  # Persistent disk storage
    MEMORY = "memory"  # In-memory storage
    CLOUD = "cloud"  # Cloud object storage (S3, GCS, etc.)
    HYBRID = "hybrid"  # Combination of memory and disk
    NETWORK = "network"  # Network-attached storage

    @property
    def is_persistent(self) -> bool:
        """Check if this storage type persists data."""
        return self in {self.DISK, self.CLOUD, self.HYBRID, self.NETWORK}

    @property
    def is_volatile(self) -> bool:
        """Check if this storage type loses data on restart."""
        return self == self.MEMORY


class IndexType(BaseEnum):
    """Types of vector indices."""

    FLAT = "flat"  # Brute-force search (exact)
    IVF_FLAT = "ivf_flat"  # Inverted file index
    IVF_PQ = "ivf_pq"  # IVF with product quantization
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    LSH = "lsh"  # Locality-Sensitive Hashing
    ANNOY = "annoy"  # Approximate Nearest Neighbors Oh Yeah
    SCANN = "scann"  # Scalable Nearest Neighbors
    DPG = "dpg"  # Disk-based Product Graph
    CUSTOM = "custom"  # Custom index implementation

    @property
    def is_exact(self) -> bool:
        """Check if this index provides exact nearest neighbor search."""
        return self == self.FLAT

    @property
    def is_approximate(self) -> bool:
        """Check if this index provides approximate nearest neighbor search."""
        return not self.is_exact

    @property
    def supports_updates(self) -> bool:
        """Check if this index type supports dynamic updates."""
        return self in {self.FLAT, self.HNSW, self.IVF_FLAT, self.IVF_PQ}

    @property
    def is_graph_based(self) -> bool:
        """Check if this is a graph-based index."""
        return self in {self.HNSW, self.DPG}
