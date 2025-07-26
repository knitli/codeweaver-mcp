# sourcery skip: avoid-single-character-names-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Mock implementations for all CodeWeaver protocols.

Provides realistic mock implementations that can be used for testing without
external dependencies like API keys, databases, or network services.
"""

import asyncio
import logging
import operator
import random
import time

from collections.abc import Callable
from datetime import datetime
from typing import Any

from codeweaver._types import ProviderCapability, ProviderInfo, RerankResult
from codeweaver.backends.base import (
    CollectionInfo,
    DistanceMetric,
    HybridStrategy,
    SearchFilter,
    SearchResult,
    VectorPoint,
)
from codeweaver.sources.base import (
    AbstractDataSource,
    ContentItem,
    SourceCapability,
    SourceConfig,
    SourceWatcher,
)


logger = logging.getLogger(__name__)


class MockVectorBackend:
    """Mock implementation of VectorBackend protocol for testing."""

    def __init__(
        self,
        latency_ms: float = 10.0,
        error_rate: float = 0.0,
        error_operations: list[str] | None = None,
    ):
        """Initialize mock vector backend.

        Args:
            latency_ms: Simulated latency in milliseconds
            error_rate: Probability of random errors (0.0-1.0)
            error_operations: List of operations to apply error rate to. If None, applies to vector operations only.
                Common operations: ["upsert_vectors", "search_vectors", "delete_vectors"]
                Management operations: ["create_collection", "delete_collection"]
        """
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.error_operations = error_operations or [
            "upsert_vectors",
            "search_vectors",
            "delete_vectors",
        ]
        self.collections: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, dict[str, VectorPoint]] = {}

    async def _simulate_latency(self) -> None:
        """Simulate network/processing latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

    def _maybe_raise_error(self, operation: str) -> None:
        """Randomly raise errors based on error rate."""
        if (
            self.error_rate > 0
            and operation in self.error_operations
            and random.random() < self.error_rate
        ):
            raise RuntimeError(f"Mock error in {operation}")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Create a mock collection."""
        await self._simulate_latency()
        self._maybe_raise_error("create_collection")

        if name in self.collections:
            raise ValueError(f"Collection {name} already exists")

        self.collections[name] = {
            "dimension": dimension,
            "distance_metric": distance_metric,
            "created_at": datetime.now(datetime.timezone.utc),
            "points_count": 0,
            **kwargs,
        }
        self.vectors[name] = {}

        logger.debug("Created mock collection: %s (dim=%d)", name, dimension)

    async def upsert_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """Upsert vectors into mock collection."""
        await self._simulate_latency()
        self._maybe_raise_error("upsert_vectors")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection_vectors = self.vectors[collection_name]

        for vector in vectors:
            # Validate dimension
            expected_dim = self.collections[collection_name]["dimension"]
            if len(vector.vector) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {expected_dim}, got {len(vector.vector)}"
                )

            collection_vectors[str(vector.id)] = vector

        self.collections[collection_name]["points_count"] = len(collection_vectors)
        logger.debug("Upserted %d vectors to collection: %s", len(vectors), collection_name)

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search vectors in mock collection using cosine similarity."""
        await self._simulate_latency()
        self._maybe_raise_error("search_vectors")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection_vectors = self.vectors[collection_name]
        if not collection_vectors:
            return []

        # Calculate similarities
        results = []
        for vector_point in collection_vectors.values():
            # Skip if filters don't match (basic implementation)
            if search_filter and not self._matches_filter(vector_point, search_filter):
                continue

            # Calculate cosine similarity and normalize to [0, 1]
            cosine_sim = self._cosine_similarity(query_vector, vector_point.vector)
            score = (cosine_sim + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]

            # Apply score threshold
            if score_threshold and score < score_threshold:
                continue

            results.append(
                SearchResult(
                    id=vector_point.id,
                    score=score,
                    payload=vector_point.payload,
                    vector=vector_point.vector if kwargs.get("return_vectors") else None,
                    backend_metadata={"backend": "mock"},
                )
            )

        # Sort by score (descending) and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def delete_vectors(self, collection_name: str, ids: list[str | int]) -> None:
        """Delete vectors from mock collection."""
        await self._simulate_latency()
        self._maybe_raise_error("delete_vectors")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection_vectors = self.vectors[collection_name]

        for vector_id in ids:
            collection_vectors.pop(str(vector_id), None)

        self.collections[collection_name]["points_count"] = len(collection_vectors)
        logger.debug("Deleted %d vectors from collection: %s", len(ids), collection_name)

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get mock collection information."""
        await self._simulate_latency()
        self._maybe_raise_error("get_collection_info")

        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")

        collection = self.collections[name]

        return CollectionInfo(
            name=name,
            dimension=collection["dimension"],
            points_count=collection["points_count"],
            distance_metric=collection["distance_metric"],
            indexed=True,
            supports_sparse_vectors=False,
            supports_hybrid_search=False,
            supports_filtering=True,
            supports_updates=True,
            storage_type="memory",
            index_type="custom",
            backend_info={"type": "enum", "created_at": collection["created_at"].isoformat()},
        )

    async def list_collections(self) -> list[str]:
        """List all mock collections."""
        await self._simulate_latency()
        self._maybe_raise_error("list_collections")

        return list(self.collections.keys())

    async def delete_collection(self, name: str) -> None:
        """Delete mock collection."""
        await self._simulate_latency()
        self._maybe_raise_error("delete_collection")

        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")

        del self.collections[name]
        del self.vectors[name]
        logger.debug("Deleted mock collection: %s", name)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(a * a for a in vec2) ** 0.5

        return 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)

    def _matches_filter(self, vector_point: VectorPoint, search_filter: SearchFilter) -> bool:
        """Basic filter matching implementation."""
        # Simplified filter matching for mock purposes
        if not search_filter.conditions:
            return True

        for condition in search_filter.conditions:
            payload = vector_point.payload or {}
            field_value = payload.get(condition.field)

            if (condition.operator == "eq" and field_value != condition.value) or (
                condition.operator == "ne" and field_value == condition.value
            ):
                return False
            # Add more operators as needed

        return True


class MockHybridSearchBackend(MockVectorBackend):
    """Mock implementation of HybridSearchBackend protocol."""

    def __init__(self, latency_ms: float = 15.0, error_rate: float = 0.0):
        """Initialize mock hybrid search backend."""
        super().__init__(latency_ms, error_rate)
        self.sparse_indexes: dict[str, dict[str, Any]] = {}

    async def create_sparse_index(
        self, collection_name: str, fields: list[str], index_type: str = "bm25", **kwargs: Any
    ) -> None:
        """Create mock sparse index."""
        await self._simulate_latency()
        self._maybe_raise_error("create_sparse_index")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        self.sparse_indexes[collection_name] = {
            "fields": fields,
            "index_type": index_type,
            "created_at": datetime.now(datetime.timezone.utc),
            **kwargs,
        }

        # Update collection capabilities
        self.collections[collection_name]["supports_hybrid_search"] = True
        logger.debug("Created sparse index for collection: %s", collection_name)

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_query: dict[str, float] | str,
        limit: int = 10,
        hybrid_strategy: HybridStrategy = HybridStrategy.RRF,
        alpha: float = 0.5,
        search_filter: SearchFilter | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Perform mock hybrid search."""
        await self._simulate_latency()
        self._maybe_raise_error("hybrid_search")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        if collection_name not in self.sparse_indexes:
            raise ValueError(f"No sparse index found for collection {collection_name}")

        # Get dense search results
        dense_results = await self.search_vectors(
            collection_name, dense_vector, limit * 2, search_filter
        )

        # Simulate sparse search results (basic implementation)
        sparse_results = self._mock_sparse_search(collection_name, sparse_query, limit * 2)

        # Combine results using hybrid strategy
        combined_results = self._combine_search_results(
            dense_results, sparse_results, hybrid_strategy, alpha
        )

        return combined_results[:limit]

    async def update_sparse_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """Update sparse vectors in mock collection."""
        await self._simulate_latency()
        self._maybe_raise_error("update_sparse_vectors")

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        # For mock purposes, just store the sparse vector data
        collection_vectors = self.vectors[collection_name]

        for vector in vectors:
            if str(vector.id) in collection_vectors:
                collection_vectors[str(vector.id)].sparse_vector = vector.sparse_vector

        logger.debug(
            "Updated sparse vectors for %d points in collection: %s", len(vectors), collection_name
        )

    def _mock_sparse_search(
        self, collection_name: str, sparse_query: dict[str, float] | str, limit: int
    ) -> list[SearchResult]:
        """Mock sparse search implementation."""
        collection_vectors = self.vectors[collection_name]
        results = []

        for vector_point in collection_vectors.values():
            # Simulate sparse matching with random scores
            score = random.uniform(0.1, 0.9)

            results.append(
                SearchResult(
                    id=vector_point.id,
                    score=score,
                    payload=vector_point.payload,
                    backend_metadata={"search_type": "sparse", "backend": "mock"},
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def _combine_search_results(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        strategy: HybridStrategy,
        alpha: float,
    ) -> list[SearchResult]:
        """Combine dense and sparse search results."""
        if strategy == HybridStrategy.RRF:
            return self._rrf_fusion(dense_results, sparse_results)
        if strategy == HybridStrategy.LINEAR:
            return self._linear_fusion(dense_results, sparse_results, alpha)
        # Default to RRF
        return self._rrf_fusion(dense_results, sparse_results)

    def _rrf_fusion(
        self, dense_results: list[SearchResult], sparse_results: list[SearchResult]
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion."""
        k = 60  # RRF parameter
        scores = {}

        # Add scores from dense results
        for rank, result in enumerate(dense_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)

        # Add scores from sparse results
        for rank, result in enumerate(sparse_results):
            scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)

        # Create combined results
        all_results = {r.id: r for r in dense_results + sparse_results}
        combined = []

        for result_id, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
            if result_id in all_results:
                result = all_results[result_id]
                result.score = score
                result.backend_metadata = {"fusion": "rrf", "backend": "mock"}
                combined.append(result)

        return combined

    def _linear_fusion(
        self, dense_results: list[SearchResult], sparse_results: list[SearchResult], alpha: float
    ) -> list[SearchResult]:
        """Linear combination fusion."""
        scores = {}

        # Normalize and combine scores
        dense_vecs = {r.id: r.score for r in dense_results}
        sparse_vecs = {r.id: r.score for r in sparse_results}

        all_ids = set(dense_vecs.keys()) | set(sparse_vecs.keys())

        for result_id in all_ids:
            dense_score = dense_vecs.get(result_id, 0.0)
            sparse_score = sparse_vecs.get(result_id, 0.0)
            scores[result_id] = alpha * dense_score + (1 - alpha) * sparse_score

        # Create combined results
        all_results = {r.id: r for r in dense_results + sparse_results}
        combined = []

        for result_id, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
            if result_id in all_results:
                result = all_results[result_id]
                result.score = score
                result.backend_metadata = {"fusion": "linear", "alpha": alpha, "backend": "mock"}
                combined.append(result)

        return combined


class MockEmbeddingProvider:
    """Mock implementation of EmbeddingProvider protocol."""

    def __init__(
        self,
        provider_name: str = "mock_embeddings",
        model_name: str = "mock-model-v1",
        dimension: int = 128,
        latency_ms: float = 50.0,
        error_rate: float = 0.0,
    ):
        """Initialize mock embedding provider.

        Args:
            provider_name: Name of the mock provider
            model_name: Name of the mock model
            dimension: Embedding dimension
            latency_ms: Simulated latency in milliseconds
            error_rate: Probability of random errors (0.0-1.0)
        """
        self._provider_name = provider_name
        self._model_name = model_name
        self._dimension = dimension
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self._call_count = 0

    async def _simulate_latency(self) -> None:
        """Simulate API latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

    def _maybe_raise_error(self, operation: str) -> None:
        """Randomly raise errors based on error rate."""
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise RuntimeError(f"Mock error in {operation}")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size."""
        return 100

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length."""
        return 10000

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        await self._simulate_latency()
        self._maybe_raise_error("embed_documents")

        if len(texts) > (self.max_batch_size or len(texts)):
            raise ValueError(f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}")

        embeddings = []
        for i, text in enumerate(texts):
            if len(text) > (self.max_input_length or len(text)):
                raise ValueError(f"Text length {len(text)} exceeds maximum {self.max_input_length}")

            # Generate deterministic but "random" embedding based on text
            embedding = self._generate_mock_embedding(text, i)
            embeddings.append(embedding)

        self._call_count += 1
        logger.debug("Generated %d mock embeddings (call #%d)", len(texts), self._call_count)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for query."""
        await self._simulate_latency()
        self._maybe_raise_error("embed_query")

        if len(text) > (self.max_input_length or len(text)):
            raise ValueError(f"Text length {len(text)} exceeds maximum {self.max_input_length}")

        embedding = self._generate_mock_embedding(text, 0)
        self._call_count += 1
        logger.debug("Generated mock query embedding (call #%d)", self._call_count)
        return embedding

    def get_provider_info(self) -> ProviderInfo:
        """Get mock provider information."""
        return ProviderInfo(
            name=self._provider_name,
            display_name="Mock Embedding Provider",
            description="Mock implementation for testing",
            supported_capabilities=[ProviderCapability.EMBEDDING],
            default_models={"embedding": self._model_name},
            supported_models={"embedding": [self._model_name, "mock-model-v2"]},
            rate_limits={"embed_requests": 1000, "embed_tokens": 100000},
            requires_api_key=False,
            max_batch_size=self.max_batch_size,
            max_input_length=self.max_input_length,
            native_dimensions={self._model_name: self._dimension},
        )

    def _generate_mock_embedding(self, text: str, offset: int = 0) -> list[float]:
        """Generate deterministic mock embedding for text."""
        # Use text hash and offset for deterministic but varied embeddings
        text_hash = hash(text) + offset
        random.seed(text_hash)

        # Generate random embedding
        embedding = [random.gauss(0, 1) for _ in range(self._dimension)]

        # Normalize to unit length
        length = sum(x * x for x in embedding) ** 0.5
        if length > 0:
            embedding = [x / length for x in embedding]

        # Reset random seed
        random.seed()

        return embedding


class MockRerankProvider:
    """Mock implementation of RerankProvider protocol."""

    def __init__(
        self,
        provider_name: str = "mock_reranker",
        model_name: str = "mock-rerank-v1",
        latency_ms: float = 30.0,
        error_rate: float = 0.0,
    ):
        """Initialize mock rerank provider.

        Args:
            provider_name: Name of the mock provider
            model_name: Name of the mock model
            latency_ms: Simulated latency in milliseconds
            error_rate: Probability of random errors (0.0-1.0)
        """
        self._provider_name = provider_name
        self._model_name = model_name
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self._call_count = 0

    async def _simulate_latency(self) -> None:
        """Simulate API latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

    def _maybe_raise_error(self, operation: str) -> None:
        """Randomly raise errors based on error rate."""
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise RuntimeError(f"Mock error in {operation}")

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Get the current reranking model name."""
        return self._model_name

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents."""
        return 50

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length."""
        return 5000

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Perform mock reranking."""
        await self._simulate_latency()
        self._maybe_raise_error("rerank")

        if len(query) > (self.max_query_length or len(query)):
            raise ValueError(f"Query length {len(query)} exceeds maximum {self.max_query_length}")

        if len(documents) > (self.max_documents or len(documents)):
            raise ValueError(
                f"Document count {len(documents)} exceeds maximum {self.max_documents}"
            )

        # Generate mock relevance scores based on simple text similarity
        results = []
        for i, document in enumerate(documents):
            # Simple mock scoring based on common words
            score = self._calculate_mock_relevance(query, document, i)

            results.append(RerankResult(index=i, relevance_score=score, document=document))

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        self._call_count += 1
        logger.debug("Reranked %d documents (call #%d)", len(documents), self._call_count)
        return results

    def get_provider_info(self) -> ProviderInfo:
        """Get mock provider information."""
        return ProviderInfo(
            name=self._provider_name,
            display_name="Mock Reranking Provider",
            description="Mock implementation for testing",
            supported_capabilities=[ProviderCapability.RERANKING],
            default_models={"reranking": self._model_name},
            supported_models={"reranking": [self._model_name, "mock-rerank-v2"]},
            rate_limits={"rerank_requests": 500, "rerank_documents": 10000},
            requires_api_key=False,
            max_batch_size=None,
            max_input_length=self.max_query_length,
        )

    def _calculate_mock_relevance(self, query: str, document: str, index: int) -> float:
        """Calculate mock relevance score."""
        # Simple word overlap scoring with some randomness
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        # Calculate overlap
        overlap = len(query_words & doc_words)
        max_overlap = max(len(query_words), len(doc_words), 1)
        base_score = overlap / max_overlap

        # Add some deterministic randomness based on index
        random.seed(hash(document) + index)
        noise = random.uniform(-0.1, 0.1)
        random.seed()

        # Ensure score is in valid range
        return max(0.0, min(1.0, base_score + noise))


class MockDataSource(AbstractDataSource):
    """Mock implementation of DataSource protocol."""

    def __init__(
        self,
        source_id: str = "mock_source",
        latency_ms: float = 20.0,
        error_rate: float = 0.0,
        content_items: list[ContentItem] | None = None,
    ):
        """Initialize mock data source.

        Args:
            source_id: Identifier for this mock source
            latency_ms: Simulated latency in milliseconds
            error_rate: Probability of random errors (0.0-1.0)
            content_items: Pre-defined content items (will generate if None)
        """
        super().__init__("mock", source_id)
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self._content_items = content_items or self._generate_mock_content()
        self._content_data: dict[str, str] = {}
        self._generate_mock_content_data()

    async def _simulate_latency(self) -> None:
        """Simulate I/O latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

    def _maybe_raise_error(self, operation: str) -> None:
        """Randomly raise errors based on error rate."""
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise RuntimeError(f"Mock error in {operation}")

    def get_capabilities(self) -> set[SourceCapability]:
        """Get capabilities supported by this mock source."""
        return {
            SourceCapability.CONTENT_DISCOVERY,
            SourceCapability.CONTENT_READING,
            SourceCapability.CHANGE_WATCHING,
            SourceCapability.METADATA_EXTRACTION,
            SourceCapability.BATCH_PROCESSING,
        }

    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover mock content items."""
        await self._simulate_latency()
        self._maybe_raise_error("discover_content")

        # Apply basic filtering based on config
        filtered_items = self._apply_content_filters(self._content_items, config)

        logger.debug("Discovered %d content items from mock source", len(filtered_items))
        return filtered_items

    async def read_content(self, item: ContentItem) -> str:
        """Read content from mock item."""
        await self._simulate_latency()
        self._maybe_raise_error("read_content")

        content = self._content_data.get(item.path)
        if content is None:
            raise FileNotFoundError(f"Content item not found: {item.path}")

        logger.debug("Read content for item: %s (%d chars)", item.path, len(content))
        return content

    async def watch_changes(
        self, config: SourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Set up mock change watching."""
        await self._simulate_latency()
        self._maybe_raise_error("watch_changes")

        watcher = SourceWatcher(self.source_id, callback)
        self._watchers.append(watcher)
        # Simulate some changes after a delay
        task = asyncio.create_task(self._simulate_changes(watcher))
        tasks = {task}
        task.add_done_callback(tasks.discard)
        logger.debug("Started change watching for mock source")
        return watcher

    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate mock source configuration."""
        await self._simulate_latency()
        self._maybe_raise_error("validate_source")

        # Mock validation - just check basic config structure
        valid = await super().validate_source(config)

        # Add some mock-specific validation
        if config.get("mock_fail_validation"):
            valid = False

        logger.debug("Mock source validation result: %s", valid)
        return valid

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get mock content metadata."""
        await self._simulate_latency()
        self._maybe_raise_error("get_content_metadata")

        # Get base metadata from parent
        metadata = await super().get_content_metadata(item)

        # Add mock-specific metadata
        metadata.update({
            "mock_score": random.uniform(0.0, 1.0),
            "mock_category": random.choice(["code", "docs", "config", "test"]),
            "mock_complexity": random.choice(["low", "medium", "high"]),
        })

        return metadata

    def _generate_mock_content(self) -> list[ContentItem]:
        """Generate mock content items."""
        items = []

        # Generate some mock files
        file_templates = [
            ("src/main.py", "python", 1024),
            ("src/utils.py", "python", 512),
            ("tests/test_main.py", "python", 768),
            ("README.md", "markdown", 2048),
            ("config.json", "json", 256),
            ("package.json", "json", 384),
            ("src/components/Button.tsx", "typescript", 896),
            ("docs/api.md", "markdown", 1536),
        ]

        for i, (path, language, size) in enumerate(file_templates):
            item = ContentItem(
                path=path,
                content_type="file",
                metadata={"file_type": language, "repository": "mock_repo", "branch": "main"},
                last_modified=datetime.now(datetime.timezone.utc),
                size=size,
                language=language,
                source_id=self.source_id,
                version=f"commit_{i:03d}",
                checksum=f"mock_checksum_{i}",
            )
            items.append(item)

        return items

    def _generate_mock_content_data(self) -> None:
        """Generate mock content data for items."""
        content_templates = {
            "python": '''"""Mock Python module."""

import os
import sys
from typing import List, Dict, Any

def mock_function(param: str) -> Dict[str, Any]:
    """Mock function for testing."""
    return {"result": param, "status": "ok"}

class MockClass:
    """Mock class for testing."""

    def __init__(self, name: str):
        self.name = name

    def process(self, data: List[str]) -> str:
        """Process mock data."""
        return f"Processed {len(data)} items for {self.name}"
''',
            "markdown": """# Mock Documentation

This is a mock markdown document for testing purposes.

## Features

- Feature 1: Does something useful
- Feature 2: Does something else
- Feature 3: Integration with other systems

## Usage

```python
from mock_module import MockClass

instance = MockClass("test")
result = instance.process(["item1", "item2"])
print(result)
```

## API Reference

### MockClass

A class for demonstration purposes.

#### Methods

- `process(data)`: Process the input data
- `configure(options)`: Configure the instance
""",
            "json": """{
  "name": "mock-package",
  "version": "1.0.0",
  "description": "Mock package for testing",
  "main": "index.js",
  "scripts": {
    "test": "jest",
    "build": "webpack",
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "jest": "^29.0.0",
    "webpack": "^5.0.0"
  }
}""",
            "typescript": """import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  variant = 'primary',
  disabled = false
}) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
};

export default Button;
""",
        }

        for item in self._content_items:
            # Get content template based on language
            template = content_templates.get(
                item.language, "# Mock content\n\nThis is mock content."
            )

            # Customize content with file-specific information
            content = template.replace("mock", item.path.split("/")[-1].split(".")[0])
            self._content_data[item.path] = content

    async def _simulate_changes(self, watcher: SourceWatcher) -> None:
        """Simulate content changes for testing."""
        await asyncio.sleep(2.0)  # Wait 2 seconds

        if watcher.is_active:
            # Simulate a content change
            changed_item = self._content_items[0]  # Change first item
            changed_item.last_modified = datetime.now(datetime.timezone.utc)
            changed_item.version = f"updated_{int(time.time())}"

            # Notify the watcher
            await watcher.notify_changes([changed_item])
            logger.debug("Simulated content change for mock source")
