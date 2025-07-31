"""
Qdrant vector database backend implementation.

Provides both basic and hybrid search capabilities using Qdrant's
native sparse vector support introduced in v1.10+.
"""

import logging

from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from codeweaver.types import (
    BackendCollectionNotFoundError,
    BackendConnectionError,
    BackendError,
    CollectionInfo,
    DistanceMetric,
    FilterCondition,
    HealthStatus,
    HybridStrategy,
    SearchFilter,
    SearchResult,
    ServiceHealth,
    VectorPoint,
)


logger = logging.getLogger(__name__)


class QdrantBackend:
    """
    Qdrant vector database backend with hybrid search support.

    Supports both basic vector operations and advanced hybrid search
    using Qdrant's native sparse vector capabilities (v1.10+).
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        enable_sparse_vectors: bool = False,
        sparse_on_disk: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Qdrant backend.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            enable_sparse_vectors: Enable sparse vector support for hybrid search
            sparse_on_disk: Store sparse vectors on disk (vs memory)
            **kwargs: Additional Qdrant client options
        """
        try:
            self.client = QdrantClient(url=url, api_key=api_key, **kwargs)
            self.enable_sparse_vectors = enable_sparse_vectors
            self.sparse_on_disk = sparse_on_disk
            self.client.get_collections()
            logger.info("Connected to Qdrant at %s", url)
        except Exception as e:
            raise BackendConnectionError(
                f"Failed to connect to Qdrant at {url}", backend_type="qdrant", original_error=e
            ) from e

    def _convert_distance_metric(self, metric: DistanceMetric) -> Distance:
        """Convert universal distance metric to Qdrant Distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN,
        }
        return mapping[metric]

    def _convert_filter(self, search_filter: SearchFilter) -> Filter:
        """Convert universal SearchFilter to Qdrant Filter."""
        conditions = []
        if search_filter.conditions:
            for condition in search_filter.conditions:
                field_condition = self._convert_filter_condition(condition)
                conditions.append(field_condition)
        must = []
        should = []
        must_not = []
        if search_filter.must:
            must.extend((self._convert_filter(sub_filter) for sub_filter in search_filter.must))
        if search_filter.should:
            should.extend((self._convert_filter(sub_filter) for sub_filter in search_filter.should))
        if search_filter.must_not:
            must_not.extend(
                (self._convert_filter(sub_filter) for sub_filter in search_filter.must_not)
            )
        return Filter(must=conditions + must, should=should or None, must_not=must_not or None)

    def _convert_filter_condition(self, condition: FilterCondition) -> FieldCondition:
        """Convert FilterCondition to Qdrant FieldCondition."""
        if condition.operator == "eq":
            return FieldCondition(key=condition.field, match=MatchValue(value=condition.value))
        if condition.operator == "in":
            return FieldCondition(key=condition.field, match=MatchValue(any=condition.value))
        raise ValueError("Unsupported filter operator: %s", condition.operator)

    def _convert_search_result(self, hit: Any) -> SearchResult:
        """Convert Qdrant search hit to universal SearchResult."""
        return SearchResult(
            id=hit.id,
            score=hit.score,
            payload=hit.payload,
            backend_metadata={"qdrant_version": hit.version if hasattr(hit, "version") else None},
        )

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Create a new collection with optional sparse vector support."""
        try:
            vectors_config = {
                "dense": VectorParams(
                    size=dimension, distance=self._convert_distance_metric(distance_metric)
                )
            }
            sparse_vectors_config = None
            if self.enable_sparse_vectors:
                sparse_vectors_config = {
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=self.sparse_on_disk)
                    )
                }
                logger.info("Enabling sparse vectors for collection %s", name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                **kwargs,
            )
            logger.info("Created Qdrant collection: %s (dimension: %d)", name, dimension)
        except Exception as e:
            raise BackendError(
                f"Failed to create collection {name}", backend_type="qdrant", original_error=e
            ) from e

    async def upsert_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """Insert or update vectors in the collection."""
        try:
            points = []
            for vector_point in vectors:
                point_data = {
                    "id": vector_point.id,
                    "vector": {"dense": vector_point.vector},
                    "payload": vector_point.payload or {},
                }
                if vector_point.sparse_vector and self.enable_sparse_vectors:
                    point_data["vector"]["sparse"] = vector_point.sparse_vector
                points.append(PointStruct(**point_data))
            self.client.upsert(collection_name=collection_name, points=points)
            logger.debug("Upserted %d vectors to collection %s", len(vectors), collection_name)
        except Exception as e:
            raise BackendError(
                f"Failed to upsert vectors to {collection_name}",
                backend_type="qdrant",
                original_error=e,
            ) from e

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search for similar vectors using dense embeddings."""
        try:
            qdrant_filter = self._convert_filter(search_filter) if search_filter else None
            search_params = {
                "collection_name": collection_name,
                "query_vector": ("dense", query_vector),
                "limit": limit,
                "query_filter": qdrant_filter,
                "score_threshold": score_threshold,
                **kwargs,
            }
            results = self.client.search(**search_params)
        except Exception as e:
            raise BackendError(
                f"Failed to search collection {collection_name}",
                backend_type="qdrant",
                original_error=e,
            ) from e
        else:
            return [self._convert_search_result(hit) for hit in results]

    async def delete_vectors(self, collection_name: str, ids: list[str | int]) -> None:
        """Delete vectors by IDs."""
        try:
            self.client.delete(collection_name=collection_name, points_selector=ids)
            logger.debug("Deleted %d vectors from collection %s", len(ids), collection_name)
        except Exception as e:
            raise BackendError(
                f"Failed to delete vectors from {collection_name}",
                backend_type="qdrant",
                original_error=e,
            ) from e

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get collection metadata and capabilities."""
        try:
            collection = self.client.get_collection(name)
            vectors_config = collection.config.params.vectors
            if isinstance(vectors_config, dict):
                dense_config = vectors_config.get("dense")
                dimension = dense_config.size if dense_config else 0
                supports_sparse = "sparse" in vectors_config
            else:
                dimension = vectors_config.size
                supports_sparse = False
        except Exception as e:
            if "not found" in str(e).lower():
                raise BackendCollectionNotFoundError(
                    f"Collection {name} not found", backend_type="qdrant", original_error=e
                ) from e
            raise BackendError(
                f"Failed to get collection info for {name}", backend_type="qdrant", original_error=e
            ) from e
        else:
            return CollectionInfo(
                name=name,
                dimension=dimension,
                points_count=collection.points_count or 0,
                distance_metric=self._convert_qdrant_distance(collection.config.params.vectors),
                indexed=collection.status == "green",
                supports_sparse_vectors=supports_sparse,
                supports_hybrid_search=supports_sparse,
                supports_filtering=True,
                supports_updates=True,
                storage_type="hybrid",
                index_type="hnsw",
                backend_info={
                    "qdrant_version": getattr(collection, "version", None),
                    "optimizer_status": collection.optimizer_status,
                    "payload_schema": collection.config.params.payload_schema,
                },
            )

    def _convert_qdrant_distance(self, vectors_config: Any) -> DistanceMetric:
        """Convert Qdrant Distance to universal DistanceMetric."""
        if isinstance(vectors_config, dict):
            dense_config = vectors_config.get("dense")
            distance = dense_config.distance if dense_config else Distance.COSINE
        else:
            distance = vectors_config.distance
        mapping = {
            Distance.COSINE: DistanceMetric.COSINE,
            Distance.EUCLID: DistanceMetric.EUCLIDEAN,
            Distance.DOT: DistanceMetric.DOT_PRODUCT,
            Distance.MANHATTAN: DistanceMetric.MANHATTAN,
        }
        return mapping.get(distance, DistanceMetric.COSINE)

    async def list_collections(self) -> list[str]:
        """List all available collections."""
        try:
            collections = self.client.get_collections()
        except Exception as e:
            raise BackendError(
                "Failed to list collections", backend_type="qdrant", original_error=e
            ) from e
        else:
            return [collection.name for collection in collections.collections]

    async def delete_collection(self, name: str) -> None:
        """Delete a collection entirely."""
        try:
            self.client.delete_collection(name)
            logger.info("Deleted Qdrant collection: %s", name)
        except Exception as e:
            raise BackendError(
                f"Failed to delete collection {name}", backend_type="qdrant", original_error=e
            ) from e


class QdrantHybridBackend(QdrantBackend):
    """
    Enhanced Qdrant backend with hybrid search capabilities.

    Leverages Qdrant's native sparse vector support for advanced
    hybrid search combining dense embeddings with sparse keyword matching.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the qdrant backend."""
        kwargs["enable_sparse_vectors"] = True
        super().__init__(*args, **kwargs)

    async def create_sparse_index(
        self, collection_name: str, fields: list[str], index_type: str = "bm25", **kwargs: Any
    ) -> None:
        """
        Create sparse vector index for hybrid search.

        Note: In Qdrant, sparse vectors are configured at collection creation time.
        This method validates that sparse vectors are enabled.
        """

        def _raise_backend_error(msg: str, e: BackendError | Any = None) -> None:
            if e and (not isinstance(e, BackendError)):
                raise BackendError(msg, backend_type="qdrant", original_error=e) from e
            if e:
                return
            raise BackendError(msg)

        try:
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info.supports_sparse_vectors:
                _raise_backend_error(
                    f"Collection {collection_name} does not support sparse vectors. Recreate with enable_sparse_vectors=True"
                )
            logger.info("Sparse index already configured for collection %s", collection_name)
        except Exception as e:
            _raise_backend_error(f"Failed to validate sparse index for {collection_name}", e)
            raise

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
        """
        Perform hybrid search using Qdrant's Query API.

        Combines dense vector search with sparse keyword matching
        using server-side fusion for optimal performance.
        """
        try:
            if isinstance(sparse_query, str):
                tokens = sparse_query.lower().split()
                sparse_vector = {hash(token) % 10000: 1.0 for token in tokens}
            else:
                sparse_vector = sparse_query
            prefetch_queries = [Prefetch(query=dense_vector, using="dense", limit=limit * 2)]
            if sparse_vector:
                prefetch_queries.append(
                    Prefetch(query=sparse_vector, using="sparse", limit=limit * 2)
                )
            fusion_mapping = {HybridStrategy.RRF: Fusion.RRF, HybridStrategy.DBSF: Fusion.DBSF}
            fusion = fusion_mapping.get(hybrid_strategy, Fusion.RRF)
            qdrant_filter = self._convert_filter(search_filter) if search_filter else None
            result = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=fusion),
                limit=limit,
                query_filter=qdrant_filter,
                **kwargs,
            )
        except Exception as e:
            raise BackendError(
                f"Failed to perform hybrid search on {collection_name}",
                backend_type="qdrant",
                original_error=e,
            ) from e
        else:
            return [self._convert_search_result(hit) for hit in result.points]

    async def update_sparse_vectors(self, collection_name: str, vectors: list[VectorPoint]) -> None:
        """
        Update sparse vectors for existing points.

        Note: In Qdrant, this is handled by the regular upsert operation
        when sparse_vector is provided in VectorPoint.
        """
        await self.upsert_vectors(collection_name, vectors)

    async def health_check(self) -> bool:
        """Check backend health and connectivity."""
        try:
            health_info = self.client.get_cluster_info()
            if not health_info:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    message="Unable to get cluster info from Qdrant",
                    last_check=datetime.now(UTC),
                )
            collections = self.client.get_collections()
            collection_count = len(collections.collections) if collections else 0
            try:
                telemetry = self.client.get_telemetry()
            except Exception as e:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"Qdrant partially available: {e}",
                    last_check=datetime.now(UTC),
                    metadata={"collection_count": collection_count},
                )
            else:
                return ServiceHealth(
                    status=HealthStatus.HEALTHY,
                    message=f"Qdrant healthy: {collection_count} collections available",
                    last_check=datetime.now(UTC),
                    metadata={
                        "collection_count": collection_count,
                        "cluster_info": str(health_info),
                        "telemetry_available": telemetry is not None,
                    },
                )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant connection failed: {e}",
                last_check=datetime.now(UTC),
            )

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information for monitoring."""
        try:
            return {
                "backend_type": "qdrant",
                "url": getattr(self.client, "_client", {}).get("url", "unknown"),
                "sparse_vectors_enabled": self.enable_sparse_vectors,
                "sparse_on_disk": getattr(self, "sparse_on_disk", False),
            }
        except Exception:
            return {"backend_type": "qdrant", "status": "connection_info_unavailable"}

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for monitoring."""
        try:
            collections = self.client.get_collections()
            metrics = {
                "total_collections": len(collections.collections) if collections else 0,
                "collections": [],
            }
            for collection in collections.collections if collections else []:
                try:
                    collection_info = self.client.get_collection(collection.name)
                    metrics["collections"].append({
                        "name": collection.name,
                        "points_count": collection_info.points_count if collection_info else 0,
                        "vectors_count": collection_info.vectors_count if collection_info else 0,
                        "status": collection_info.status if collection_info else "unknown",
                    })
                except Exception as e:
                    metrics["collections"].append({"name": collection.name, "error": str(e)})
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {e}",
                "total_collections": 0,
                "collections": [],
            }
        else:
            return metrics
