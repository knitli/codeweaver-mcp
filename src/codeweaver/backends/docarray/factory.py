# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Factory for creating DocArray-powered backends."""

import logging

from typing import Any

from codeweaver.backends.factory import BackendFactory

from .config import DocArrayBackendConfig, DocArrayConfigFactory


logger = logging.getLogger(__name__)


def create_docarray_backend(config: DocArrayBackendConfig) -> Any:
    """Create a DocArray backend based on configuration."""
    provider = str(config.provider).lower()

    if provider == "docarray_qdrant":
        from .qdrant import QdrantDocArrayBackend

        return QdrantDocArrayBackend(config)
    if provider == "docarray_pinecone":
        # Future implementation
        raise NotImplementedError("DocArray Pinecone backend not yet implemented")
    if provider == "docarray_weaviate":
        # Future implementation
        raise NotImplementedError("DocArray Weaviate backend not yet implemented")
    raise ValueError(f"Unsupported DocArray backend provider: {provider}")


def register_docarray_backends() -> None:
    """Register DocArray backends with the main factory."""
    try:
        # Register Qdrant DocArray backend
        from .qdrant import QdrantDocArrayBackend

        # Check if dependencies are available
        missing_deps = QdrantDocArrayBackend._check_dependencies()
        if not missing_deps:
            BackendFactory.register_backend(
                "docarray_qdrant",
                lambda **kwargs: create_docarray_backend(
                    DocArrayConfigFactory.create_config("docarray_qdrant", **kwargs)
                ),
                supports_hybrid=True,
            )
            logger.info("Registered DocArray Qdrant backend")
        else:
            logger.warning(
                "DocArray Qdrant backend not available, missing dependencies: %s",
                ", ".join(missing_deps),
            )

    except ImportError as e:
        logger.warning("Failed to register DocArray backends: %s", e)


# Auto-register backends on module import
register_docarray_backends()
