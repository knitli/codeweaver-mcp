# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding/`)
"""Entrypoint for CodeWeaver's heavily-pydantic-ai-inspired embedding model system."""

from collections.abc import Callable, Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, PositiveInt, SecretStr

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import Provider


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
    output_transformer: Callable[..., Sequence[CodeChunk]] | None = None
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
    api_key: Annotated[
        SecretStr | None, Field(description="The API key for the model, if required.")
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
        # TODO, we need a way to resolve the provider capabilities from the settings.
