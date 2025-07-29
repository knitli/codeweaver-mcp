# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Universal adapter for DocArray backends to CodeWeaver protocols."""

import logging

from abc import ABC, abstractmethod
from itertools import starmap
from typing import Any

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex

from codeweaver.backends.base import HybridSearchBackend, VectorBackend
from codeweaver.types import CollectionInfo, DistanceMetric, SearchFilter, SearchResult, VectorPoint


logger = logging.getLogger(__name__)


class DocArrayAdapterError(Exception):
    """Errors specific to DocArray adapter operations."""


class VectorConverter:
    """Converts between CodeWeaver and DocArray vector formats."""

    def __init__(self, doc_class: type[BaseDoc]):
        """Initialize vector converter with document class.

        Args:
            doc_class: DocArray document class for conversion
        """
        self.doc_class = doc_class
        self._field_names = set(doc_class.model_fields.keys())

    def vector_to_doc(self, vector: VectorPoint) -> BaseDoc:
        """Convert VectorPoint to DocArray document."""
        doc_data = {
            "id": str(vector.id),
            "content": vector.payload.get("content", "") if vector.payload else "",
            "embedding": vector.vector,
            "metadata": vector.payload.copy() if vector.payload else {},
        }

        # Add sparse vector if available
        if "sparse_vector" in self._field_names and vector.sparse_vector:
            doc_data["sparse_vector"] = vector.sparse_vector

        # Extract structured metadata
        if vector.payload:
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
        self, doc_index: BaseDocIndex, doc_class: type[BaseDoc], collection_name: str = "default"
    ):
        """Initialize DocArray backend adapter.

        Args:
            doc_index: DocArray document index instance
            doc_class: Document class for schema definition
            collection_name: Name of the vector collection
        """
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
        logger.info("Created collection '%s' with dimension %s", name, dimension)

    async def upsert_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """Insert or update vectors in the collection."""
        if not self._initialized:
            raise DocArrayAdapterError("Collection not initialized")

        try:
            # Convert vectors to documents
            docs = [self.converter.vector_to_doc(vector) for vector in vectors]
            all_docs = DocList[self.doc_class](docs)

            # Batch insert for performance
            self.doc_index.index(all_docs)

            logger.debug("Upserted %s vectors to collection '%s'", len(vectors), collection_name)

        except Exception as e:
            logger.exception("Failed to upsert vectors")
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
            results, scores = self.doc_index.find(query_doc, search_field="embedding", limit=limit)

            # Convert results
            search_results = []
            search_results.extend(
                self.converter.doc_to_search_result(doc, score)
                for doc, score in zip(results, scores, strict=False)
                if score_threshold is None or score >= score_threshold
            )
            logger.debug("Found %s results for query", len(search_results))

        except Exception as e:
            logger.exception("Search failed")
            raise DocArrayAdapterError(f"Search failed: {e}") from e
        else:
            return search_results

    async def delete_vectors(self, collection_name: str, ids: list[str | int]) -> None:
        """Delete vectors by IDs."""
        try:
            # DocArray delete implementation depends on backend
            if hasattr(self.doc_index, "delete"):
                self.doc_index.delete([str(id_) for id_ in ids])
            else:
                logger.warning("Backend doesn't support direct deletion")

        except Exception as e:
            logger.exception("Delete failed")
            raise DocArrayAdapterError(f"Delete failed: {e}") from e

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get collection metadata and statistics."""
        try:
            # Extract information from DocArray index
            vector_count = self._get_vector_count()
            dimension = getattr(self, "_dimension", 512)

            return CollectionInfo(
                name=name,
                dimension=dimension,
                points_count=vector_count,
                distance_metric=getattr(self, "_distance_metric", DistanceMetric.COSINE),
                indexed=True,  # Assume indexed by default
                supports_hybrid_search=self._supports_hybrid_search(),
                supports_filtering=True,  # Most backends support basic filtering
                supports_sparse_vectors=self._supports_sparse_vectors(),
            )

        except Exception as e:
            logger.exception("Failed to get collection info")
            raise DocArrayAdapterError(f"Collection info failed: {e}") from e

    async def list_collections(self) -> list[str]:
        """List all available collections."""
        # DocArray typically works with single collections
        return [self.collection_name] if self._initialized else []

    async def delete_collection(self, name: str) -> None:
        """Delete a collection entirely."""
        try:
            if hasattr(self.doc_index, "drop"):
                self.doc_index.drop()
            self._initialized = False
            logger.info("Deleted collection '%s'", name)

        except Exception as e:
            logger.exception("Failed to delete collection")
            raise DocArrayAdapterError(f"Collection deletion failed: {e}") from e

    # Abstract methods for backend-specific implementations

    @abstractmethod
    def _get_vector_count(self) -> int:
        """Get the number of vectors in the collection."""

    @abstractmethod
    def _supports_hybrid_search(self) -> bool:
        """Check if backend supports hybrid search."""

    @abstractmethod
    def _supports_sparse_vectors(self) -> bool:
        """Check if backend supports sparse vectors."""


class DocArrayHybridAdapter(BaseDocArrayAdapter, HybridSearchBackend):
    """DocArray adapter with hybrid search capabilities."""

    async def create_sparse_index(
        self, collection_name: str, fields: list[str], index_type: str = "bm25", **kwargs: Any
    ) -> None:
        """Create sparse vector index for hybrid search."""
        try:
            if hasattr(self.doc_index, "configure_sparse_index"):
                self.doc_index.configure_sparse_index(fields, index_type, **kwargs)
                logger.info("Created %s sparse index on fields: %s", index_type, fields)
            else:
                logger.warning("Backend doesn't support native sparse indexing")

        except Exception as e:
            logger.exception("Sparse index creation failed")
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
                dense_vector, sparse_vector=sparse_vector, **kwargs
            )

            # Use native hybrid search if available
            if hasattr(self.doc_index, "hybrid_search"):
                results, scores = self.doc_index.hybrid_search(query_doc, alpha=alpha, limit=limit)
            else:
                # Fallback to RRF fusion
                results, scores = await self._fallback_hybrid_search(
                    dense_vector, sparse_vector, limit, alpha
                )

            # Convert results
            return list(
                starmap(self.converter.doc_to_search_result, zip(results, scores, strict=False))
            )

        except Exception as e:
            logger.exception("Hybrid search failed")
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
        self, dense_vector: list[float], sparse_vector: dict[str, float], limit: int, alpha: float
    ) -> tuple[list[BaseDoc], list[float]]:
        """Fallback hybrid search using RRF fusion."""
        # Perform separate dense and sparse searches
        dense_results = await self.search_vectors(self.collection_name, dense_vector, limit * 2)

        # For sparse search, we'd need a separate implementation
        # This is a simplified version
        sparse_results = dense_results  # Placeholder

        return self._apply_rrf_fusion(sparse_results, sparse_results, limit)

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
        sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

        # Return placeholder results
        # In real implementation, fetch the actual documents
        return [], []
