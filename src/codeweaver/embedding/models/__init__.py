# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding/`) from `pydantic_ai`.
# in files that are marked like this one.
#
# SPDX-FileCopyrightText: (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding/`)

"""Entrypoint for CodeWeaver's heavily-pydantic-ai-inspired embedding model system."""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, NotRequired, Required, Self, TypedDict

from pydantic import BaseModel, Field, PositiveInt

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import EmbeddingModelSettings, Provider


type PartialCapabilities = dict[
    Literal[
        "name",
        "provider",
        "output_transformer",
        "version",
        "default_dimension",
        "output_dimensions",
        "default_dtype",
        "output_dtypes",
        "requires_api_key",
        "supports_batching",
        "preferred_metrics",
        "is_normalized",
        "context_window",
        "supports_custom_prompts",
        "custom_document_prompt",
        "custom_query_prompt",
        "supports_context_chunk_embedding",
        "tokenizer",
        "tokenizer_model",
        "input_transformer",
    ],
    Literal[
        "tokenizers", "tiktoken", "dot", "cosine", "euclidean", "manhattan", "hamming", "chebyshev"
    ]
    | str
    | PositiveInt
    | bool
    | Provider
    | None
    | tuple[str, ...]
    | tuple[PositiveInt, ...]
    | Callable[[list[CodeChunk]], Any]
    | Callable[..., Sequence[CodeChunk]],
]


class EmbeddingCapabilities(TypedDict):
    """Describes the capabilities of an embedding model, such as the default dimension."""

    name: Required[str]
    provider: Required[Provider]
    output_transformer: NotRequired[Callable[..., Sequence[CodeChunk]] | None]
    version: NotRequired[str | int | None]
    default_dimension: NotRequired[PositiveInt]
    output_dimensions: NotRequired[tuple[PositiveInt, ...] | None]
    default_dtype: NotRequired[str | None]
    output_dtypes: NotRequired[tuple[str, ...] | None]
    requires_api_key: NotRequired[bool]
    supports_batching: NotRequired[bool]
    supports_custom_prompts: NotRequired[bool]
    custom_document_prompt: NotRequired[str] | None
    custom_query_prompt: NotRequired[str] | None
    is_normalized: NotRequired[bool]
    context_window: NotRequired[PositiveInt]
    supports_context_chunk_embedding: NotRequired[bool]
    tokenizer: NotRequired[Literal["tokenizers", "tiktoken"]]
    tokenizer_model: NotRequired[str]
    preferred_metrics: NotRequired[
        tuple[Literal["dot", "cosine", "euclidean", "manhattan", "hamming", "chebyshev"], ...]
    ]


class EmbeddingModelCapabilities(BaseModel):
    """Describes the capabilities of an embedding model, such as the default dimension."""

    name: Annotated[
        str, Field(min_length=3, description="The name of the model or family of models.")
    ] = ""
    provider: Annotated[
        Provider,
        Field(
            description="The provider of the model. Since available settings vary across providers, each capabilities instance is tied to a provider."
        ),
    ] = Provider._UNSET  # type: ignore
    input_transformer: Callable[[list[CodeChunk]], Any] | None = None
    output_transformer: (
        Callable[..., Sequence[Sequence[float]] | Sequence[Sequence[int]]] | None
    ) = None
    version: Annotated[
        str | int | None,
        Field(
            description="The version of the model, if applicable. Can be a string or an integer. If not specified, defaults to `None`."
        ),
    ] = None
    default_dimension: Annotated[PositiveInt, Field(multiple_of=8)] = 512
    output_dimensions: Annotated[
        tuple[PositiveInt, ...] | None,
        Field(
            multiple_of=8,
            description="Supported output dimensions, if the model and provider support multiple output dimensions. If not specified, defaults to `None`.",
        ),
    ] = None
    default_dtype: Annotated[
        str | None,
        Field(
            description="A string representing the default data type of the model, such as `float`, if the provider/model accepts different data types. If not specified, defaults to `None`."
        ),
    ] = None
    output_dtypes: Annotated[
        tuple[str, ...] | None,
        Field(
            description="A list of accepted values for output data types, if the model/provider allows different output data types. When available, you can use this to reduce the size of the returned vectors, at the cost of some accuracy (depending on which you choose).",
            examples=[
                "VoyageAI: `('float', 'uint8', 'int8', 'binary', 'ubinary')` for the voyage 3-series models."
            ],
        ),
    ] = None
    requires_api_key: bool = False
    supports_batching: bool = False
    is_normalized: bool = False
    context_window: Annotated[PositiveInt, Field(ge=256)] = 512
    supports_context_chunk_embedding: bool = False
    tokenizer: Literal["tokenizers", "tiktoken"] | None = None
    tokenizer_model: Annotated[
        str | None,
        Field(
            min_length=3,
            description="The tokenizer model used by the embedding model. If the tokenizer is `tokenizers`, this should be the full name of the tokenizer or model (if it's listed by its model name), *including the organization*. Like: `voyageai/voyage-code-3`",
        ),
    ] = None
    preferred_metrics: Annotated[
        tuple[Literal["dot", "cosine", "euclidean", "manhattan", "hamming", "chebyshev"], ...],
        Field(
            description="A tuple of preferred metrics for comparing embeddings.",
            examples=[
                "VoyageAI: `('dot',)` for the voyage 3-series models, since they are normalized to length 1."
            ],
        ),
    ] = ("cosine", "dot", "euclidean")
    _version: Annotated[
        str,
        Field(
            init=False,
            pattern=r"^\d{1,2}\.\d{1,3}\.\d{1,3}$",
            description="The version for the capabilities schema.",
        ),
    ] = "1.0.0"

    @classmethod
    def default(cls) -> Self:
        """Create a default instance of the model profile."""
        return cls()

    @property
    def schema_version(self) -> str:
        """Get the schema version of the capabilities."""
        return self._version

    @classmethod
    def validate_settings(cls, settings: EmbeddingModelSettings) -> Self:
        """Validate and create an instance from the provided settings."""
