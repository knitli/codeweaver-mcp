# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Qdrant backend implementation using DocArray."""

import logging

from typing import Any

from codeweaver.config import QdrantDocArrayConfig
from codeweaver.schema import DocumentSchemaGenerator, SchemaTemplates
from codeweaver.sources import DocArrayHybridAdapter


logger = logging.getLogger(__name__)


class QdrantDocArrayBackend(DocArrayHybridAdapter):
    """Qdrant backend using DocArray with hybrid search support."""

    def __init__(self, config: QdrantDocArrayConfig):
        """Initialize Qdrant DocArray backend."""
        self.config = config

        if missing_deps := self._check_dependencies():
            raise ImportError(
                f"Missing required dependencies for DocArray Qdrant: {', '.join(missing_deps)}. "
                "Install with: pip install docarray[qdrant] qdrant-client"
            )

        from docarray.index import QdrantDocumentIndex

        # Create document schema
        if config.schema_config.schema_template == "code_search":
            doc_class = SchemaTemplates.code_search_schema(config.schema_config.embedding_dimension)
        elif config.schema_config.schema_template == "semantic_search":
            doc_class = SchemaTemplates.semantic_search_schema(
                config.schema_config.embedding_dimension
            )
        else:
            doc_class = DocumentSchemaGenerator.create_schema(
                config.schema_config, "QdrantCodeWeaverDoc"
            )

        # Create Qdrant database config
        db_config_kwargs = {
            "host": config.url or "localhost",
            "api_key": config.api_key,
            "collection_name": config.collection_name or "codeweaver",
            "distance": "Cosine",  # Map from DistanceMetric
            **config.db_config,
        }

        # Handle port extraction from URL if present
        if config.url and ":" in config.url:
            try:
                host_port = config.url.replace("http://", "").replace("https://", "")
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                    db_config_kwargs["host"] = host
                    db_config_kwargs["port"] = int(port)
            except ValueError:
                logger.warning("Could not parse port from URL, using default")

        db_config = QdrantDocumentIndex.DBConfig(**db_config_kwargs)

        # Create runtime config
        runtime_config = QdrantDocumentIndex.RuntimeConfig(**config.runtime_config)

        # Initialize DocArray index
        try:
            doc_index = QdrantDocumentIndex[doc_class](
                db_config=db_config, runtime_config=runtime_config
            )

            super().__init__(
                doc_index=doc_index,
                doc_class=doc_class,
                collection_name=config.collection_name or "codeweaver",
            )

            logger.info("Qdrant DocArray backend initialized successfully")

        except Exception:
            logger.exception("Failed to initialize Qdrant DocArray backend")
            raise

    def _get_vector_count(self) -> int:
        """Get the number of vectors in the Qdrant collection."""
        try:
            if hasattr(self.doc_index, "_client"):
                collection_info = self.doc_index._client.get_collection(self.collection_name)
                return collection_info.points_count
        except Exception:
            logger.warning("Could not get vector count from Qdrant")
            return 0
        else:
            return 0

    def _supports_hybrid_search(self) -> bool:
        """Qdrant supports hybrid search with sparse vectors."""
        return True

    def _supports_sparse_vectors(self) -> bool:
        """Qdrant supports sparse vectors natively."""
        return True

    async def create_sparse_index(
        self, collection_name: str, fields: list[str], index_type: str = "bm25", **kwargs: Any
    ) -> None:
        """Create sparse vector index in Qdrant."""
        try:
            # Qdrant handles sparse vectors automatically
            # Configure sparse vector field if needed
            if hasattr(self.doc_index, "_configure_sparse_vectors"):
                self.doc_index._configure_sparse_vectors(fields, index_type)

            logger.info("Configured Qdrant sparse index for fields: %s", fields)

        except Exception:
            logger.exception("Failed to create Qdrant sparse index")
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


# Alternative constructor for backwards compatibility
def create_qdrant_docarray_backend(
    url: str | None = None,
    api_key: str | None = None,
    collection_name: str | None = None,
    **kwargs: Any,
) -> QdrantDocArrayBackend:
    """Create a Qdrant DocArray backend with simplified configuration."""
    from codeweaver.cw_types import ProviderKind

    config = QdrantDocArrayConfig(
        provider="docarray_qdrant",
        kind=ProviderKind.COMBINED,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        **kwargs,
    )
    return QdrantDocArrayBackend(config)
