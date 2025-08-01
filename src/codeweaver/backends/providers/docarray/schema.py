# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Dynamic document schema generation for DocArray backends."""

import logging

from typing import Any

from docarray import BaseDoc
from docarray.typing import NdArray
from pydantic import Field, create_model


logger = logging.getLogger(__name__)


class SchemaConfig:
    """Configuration for dynamic document schema generation."""

    def __init__(
        self,
        embedding_dimension: int = 512,
        *,
        include_sparse_vectors: bool = False,
        metadata_fields: dict[str, type] | None = None,
        custom_fields: dict[str, tuple[type, Any]] | None = None,
        enable_validation: bool = True,
    ):
        """Initialize schema configuration.

        Args:
            embedding_dimension: Dimension of vector embeddings
            include_sparse_vectors: Whether to include sparse vector fields
            metadata_fields: Additional typed metadata fields
            custom_fields: Custom field definitions
            enable_validation: Whether to enable Pydantic validation
        """
        self.embedding_dimension = embedding_dimension
        self.include_sparse_vectors = include_sparse_vectors
        self.metadata_fields = metadata_fields or {}
        self.custom_fields = custom_fields or {}
        self.enable_validation = enable_validation


class DocumentSchemaGenerator:
    """Generates DocArray document schemas based on configuration."""

    @classmethod
    def create_schema(
        cls, config: SchemaConfig, schema_name: str = "CodeWeaverDoc"
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
            schema_name, __base__=BaseDoc, **field_definitions, __module__=__name__
        )

        # Add validation configuration
        if config.enable_validation:
            doc_class.model_config = cls._get_model_config()

        logger.info(
            "Created document schema '%s' with %s fields", schema_name, len(field_definitions)
        )
        return doc_class

    @staticmethod
    def _get_base_fields(embedding_dim: int) -> dict[str, tuple[type, Any]]:
        """Get base required fields for all document schemas."""
        return {
            "id": (str, Field(description="Unique document identifier")),
            "content": (str, Field(description="Document text content")),
            "embedding": (NdArray[embedding_dim], Field(description="Dense vector embedding")),
            "metadata": (
                dict[str, Any],
                Field(default_factory=dict, description="Document metadata"),
            ),
        }

    @staticmethod
    def _get_sparse_fields() -> dict[str, tuple[type, Any]]:
        """Get sparse vector fields for hybrid search."""
        return {
            "sparse_vector": (
                dict[str, float],
                Field(default_factory=dict, description="Sparse vector for hybrid search"),
            ),
            "keywords": (
                list[str],
                Field(default_factory=list, description="Keywords for sparse search"),
            ),
        }

    @staticmethod
    def _get_metadata_fields(metadata_config: dict[str, type]) -> dict[str, tuple[type, Any]]:
        """Convert metadata configuration to field definitions."""
        fields = {}
        for field_name, field_type in metadata_config.items():
            safe_name = f"meta_{field_name}" if not field_name.startswith("meta_") else field_name
            fields[safe_name] = (field_type, Field(description=f"Metadata field: {field_name}"))
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
            },
        )
        return DocumentSchemaGenerator.create_schema(config, "CodeSearchDoc")

    @staticmethod
    def semantic_search_schema(embedding_dim: int = 512) -> type[BaseDoc]:
        """Schema for general semantic search."""
        config = SchemaConfig(
            embedding_dimension=embedding_dim,
            include_sparse_vectors=False,
            metadata_fields={"title": str, "author": str, "timestamp": str, "category": str},
        )
        return DocumentSchemaGenerator.create_schema(config, "SemanticSearchDoc")

    @staticmethod
    def multimodal_schema(embedding_dim: int = 512) -> type[BaseDoc]:
        """Schema for multimodal documents."""
        config = SchemaConfig(
            embedding_dimension=embedding_dim,
            include_sparse_vectors=True,
            custom_fields={
                "image_embedding": (NdArray[embedding_dim], Field(description="Image embedding")),
                "text_embedding": (NdArray[embedding_dim], Field(description="Text embedding")),
                "image_url": (str | None, Field(default=None, description="Image URL")),
            },
        )
        return DocumentSchemaGenerator.create_schema(config, "MultiModalDoc")
