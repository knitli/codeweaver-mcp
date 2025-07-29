# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Configuration classes for DocArray backend integration."""

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codeweaver.backends import BackendConfig
from codeweaver.types import BaseEnum


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

    custom_fields: Annotated[dict[str, Any], Field(description="Custom field definitions")] = Field(
        default_factory=dict
    )

    schema_template: Annotated[
        Literal["code_search", "semantic_search", "multimodal", "custom"] | None,
        Field(description="Predefined schema template to use"),
    ] = None

    enable_validation: Annotated[bool, Field(description="Enable Pydantic validation")] = True

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

    # Override to add collection name support
    collection_name: Annotated[
        str | None, Field(default=None, description="Name of the vector collection")
    ] = None

    # DocArray-specific configuration
    schema_config: Annotated[
        DocArraySchemaConfig, Field(description="Document schema configuration")
    ] = Field(default_factory=DocArraySchemaConfig)

    runtime_config: Annotated[
        dict[str, Any], Field(description="DocArray runtime configuration")
    ] = Field(default_factory=dict)

    # Backend-specific database configuration
    db_config: Annotated[dict[str, Any], Field(description="Backend database configuration")] = (
        Field(default_factory=dict)
    )

    # Performance and behavior settings
    batch_size: Annotated[int, Field(ge=1, le=1000, description="Batch size for operations")] = 100

    enable_async: Annotated[bool, Field(description="Enable asynchronous operations")] = True

    connection_timeout: Annotated[
        float, Field(ge=0.1, le=300.0, description="Connection timeout in seconds")
    ] = 30.0

    retry_attempts: Annotated[int, Field(ge=0, le=10, description="Number of retry attempts")] = 3

    # Feature flags
    enable_hybrid_search: Annotated[
        bool, Field(description="Enable hybrid search capabilities")
    ] = False

    enable_compression: Annotated[bool, Field(description="Enable vector compression")] = False

    enable_caching: Annotated[bool, Field(description="Enable query result caching")] = False


# Backend-specific configuration classes


class QdrantDocArrayConfig(DocArrayBackendConfig):
    """
    Qdrant-specific DocArray configuration.

    Note: For better performance and more features, **we recommend you use CodeWeaver's own Qdrant backend provider.** You should only use this if you need to integrate with Qdrant and other DocArray backends.
    """

    provider: str = Field(default="docarray_qdrant", frozen=True)

    # Qdrant-specific settings
    prefer_grpc: bool = Field(default=False, description="Use gRPC instead of HTTP")
    grpc_port: int | None = Field(default=None, description="gRPC port if different from HTTP")

    @field_validator("db_config", mode="before")
    @classmethod
    def set_qdrant_defaults(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Set Qdrant-specific default configuration."""
        defaults = {"prefer_grpc": False, "timeout": 30.0, "retry_total": 3}
        return defaults | v


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
        defaults = {"metric": "cosine", "shards": 1, "replicas": 1}
        return defaults | v


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
        defaults = {"startup_period": 5, "additional_headers": {}}
        return defaults | v


class DocArrayBackendKind(BaseEnum):
    """Enumeration of supported DocArray backend types."""

    QDRANT = "docarray_qdrant"
    PINECONE = "docarray_pinecone"
    WEAVIATE = "docarray_weaviate"

    @property
    def config_class(self) -> type[DocArrayBackendConfig]:
        """Get the configuration class for this backend type."""
        match self:
            case DocArrayBackendKind.QDRANT:
                return QdrantDocArrayConfig
            case DocArrayBackendKind.PINECONE:
                return PineconeDocArrayConfig
            case DocArrayBackendKind.WEAVIATE:
                return WeaviateDocArrayConfig
            case _:
                raise ValueError(f"Unsupported DocArray backend kind: {self.value}")

    @property
    def member_to_class(self) -> tuple[Self, type[DocArrayBackendConfig]]:
        """Convert enum member to (kind, config class) tuple."""
        return (self, self.config_class)

    @classmethod
    def member_to_class_mapping(
        cls,
    ) -> dict[type["DocArrayBackendKind"], type[DocArrayBackendConfig]]:
        """Get mapping of enum members to their configuration classes."""
        return {member: member.config_class for member in cls.members()}


# Configuration factory
class DocArrayConfigFactory:
    """Factory for creating DocArray backend configurations."""

    CONFIG_MAPPING: ClassVar[dict[DocArrayBackendKind, DocArrayBackendConfig]] = (
        DocArrayBackendKind.member_to_class_mapping()
    )

    @classmethod
    def create_config(cls, backend_type: str, **kwargs: Any) -> DocArrayBackendConfig:
        """Create configuration for specified backend type."""
        config_class = cls.CONFIG_MAPPING.get(
            DocArrayBackendKind.from_string(backend_type), DocArrayBackendConfig
        )
        return config_class(**kwargs)

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported DocArray backend types."""
        return list(cls.CONFIG_MAPPING.keys())

    @classmethod
    def validate_backend_config(
        cls, backend_type: str, config: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate configuration for specified backend type."""
        try:
            cls.create_config(backend_type, **config)
        except Exception as e:
            return False, [str(e)]
        else:
            return True, []
