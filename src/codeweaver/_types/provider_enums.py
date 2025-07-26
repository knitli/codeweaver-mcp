# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider enums for embedding and reranking capabilities.

This module defines all enums related to provider capabilities, types, and model families.
Replaces string literals and TypedDict usage with proper type-safe enums.
"""

import operator
import re

from dataclasses import dataclass
from types import FunctionType

from codeweaver._types.base_enum import BaseEnum


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    index: int
    relevance_score: float
    document: str | None = None


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
            self.QDRANT,
            self.WEAVIATE,
            self.MILVUS,
            self.CHROMA,
            self.FAISS,
            self.PGVECTOR,
            self.REDIS,
            self.ELASTICSEARCH,
            self.OPENSEARCH,
            self.VEARCH,
            self.VESPA,
            self.TYPESENSE,
        }

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


class ProviderCapability(BaseEnum):
    """Provider capability types.

    This enum defines the different capabilities that providers can support.
    Replaces the broken TypedDict that was causing runtime failures.
    """

    EMBEDDING = "embedding"  # Fixed naming to follow singular pattern
    RERANKING = "reranking"
    BATCH_PROCESSING = "batch_processing"
    RATE_LIMITING = "rate_limiting"
    STREAMING = "streaming"
    CUSTOM_DIMENSIONS = "custom_dimensions"
    LOCAL_INFERENCE = "local_inference"


class ProviderKind(BaseEnum):
    """Provider kind/role classification.

    Defines what kind of provider this is based on capabilities.
    """

    EMBEDDING = "embedding"  # Embedding-only provider
    RERANKING = "reranking"  # Reranking-only provider
    COMBINED = "combined"  # Both embedding and reranking
    LOCAL = "local"  # Local inference provider


class ProviderType(BaseEnum):
    """Provider types.

    Enum of all supported provider implementations.
    """

    VOYAGE_AI = "voyage-ai"  # Fixed naming for consistency
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"  # Generic OpenAI-compatible provider
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    CUSTOM = "custom"

    # Test/Mock providers
    MOCK_EMBEDDING = "mock_embedding"
    MOCK_RERANK = "mock_rerank"


class ModelFamily(BaseEnum):
    """Model families across providers.

    Categorizes models by their primary use case and capabilities.
    """

    CODE_EMBEDDING = "code_embedding"
    TEXT_EMBEDDING = "text_embedding"
    RERANKING = "reranking"
    MULTIMODAL = "multimodal"


class VoyageModels(BaseEnum):
    """VoyageAI supported models."""

    CODE_3 = "voyage-code-3"
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"
    VOYAGE_LARGE_2 = "voyage-large-2"
    VOYAGE_2 = "voyage-2"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.CODE_3: 1024,
            self.VOYAGE_3: 1024,
            self.VOYAGE_3_LITE: 512,
            self.VOYAGE_LARGE_2: 1536,
            self.VOYAGE_2: 1024,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        if "code" in self.value:
            return ModelFamily.CODE_EMBEDDING
        return ModelFamily.TEXT_EMBEDDING


class VoyageRerankModels(BaseEnum):
    """VoyageAI reranking models."""

    RERANK_2 = "voyage-rerank-2"
    RERANK_LITE_1 = "voyage-rerank-lite-1"

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING


class OpenAIModels(BaseEnum):
    """OpenAI supported models."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.TEXT_EMBEDDING_3_SMALL: 1536,
            self.TEXT_EMBEDDING_3_LARGE: 3072,
            self.TEXT_EMBEDDING_ADA_002: 1536,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereModels(BaseEnum):
    """Cohere supported models."""

    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.EMBED_ENGLISH_V3: 1024,
            self.EMBED_MULTILINGUAL_V3: 1024,
            self.EMBED_ENGLISH_LIGHT_V3: 384,
            self.EMBED_MULTILINGUAL_LIGHT_V3: 384,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereRerankModels(BaseEnum):
    """Cohere reranking models."""

    RERANK_3 = "rerank-3"
    RERANK_MULTILINGUAL_3 = "rerank-multilingual-3"
    RERANK_ENGLISH_3 = "rerank-english-3"

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING
