"""Bedrock embedding provider."""
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

import io

from typing import Annotated, Literal

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel, to_snake

from codeweaver.embedding.providers import EmbeddingProvider


class BaseBedrockModel(BaseModel):
    """Base model for Bedrock-related Pydantic models."""

    model_config = ConfigDict(
        alias_generator=AliasGenerator(validation_alias=to_snake, serialization_alias=to_camel),
        str_strip_whitespace=True,
        # spellchecker:off
        ser_json_inf_nan="null",
        # spellchecker:on
        serialize_by_alias=True,
    )


class CohereEmbeddingRequestBody(BaseBedrockModel):
    """Request body for Cohere embedding model."""

    input_type: Annotated[
        Literal["search_document", "search_query", "classification", "clustering", "image"],
        Field(description="The type of input to generate embeddings for."),
    ]
    texts: Annotated[
        list[Annotated[str, Field(max_length=2048)]],
        Field(description="The input texts to generate embeddings for.", max_length=96),
    ]
    images: Annotated[
        list[str] | None,
        Field(
            description="The input image (as base64-encoded strings) to generate embeddings for.",
            max_length=1,  # your read that right, only one image at a time... even if you give it as a list...
        ),
    ] = None
    truncate: Annotated[
        Literal["NONE", "START", "END"] | None, Field(description="Truncation strategy.")
    ] = None
    embedding_types: Annotated[
        list[Literal["float", "int8", "uint8", "binary", "ubinary"]] | None,
        Field(
            description="The type of embeddings to generate. You can specify one or more types. Default is float."
        ),
    ] = None


class TitanEmbeddingV2RequestBody(BaseBedrockModel):
    """Request body for Titan Embedding V2. Note that it's one document per request."""

    input_text: Annotated[
        str, Field(description="The input text to generate embeddings for.", max_length=50_000)
    ]
    dimensions: Annotated[
        Literal[1024, 512, 256],
        Field(description="The number of dimensions for the generated embeddings."),
    ] = 1024
    normalize: Annotated[bool, Field(description="Whether to normalize the embeddings.")] = True
    embedding_types: Annotated[
        list[Literal["float", "binary"]] | None,
        Field(
            description="The type of embeddings to generate. You can specify one or both types. I guess that could be useful if you want to keep your options open, especially once data gets stale."
        ),
    ] = None


class BedrockInvokeEmbeddingRequest(BaseBedrockModel):
    """Request for Bedrock embedding."""

    body: Annotated[
        bytes | bytearray | io.BytesIO,
        Field(
            description="The body of the request. A bytes-like object, either bytes or a readable buffer. This is the content to generate embeddings for."
        ),
    ]
    content_type: Annotated[
        Literal["application/json"],
        Field(description="The content type of the body. This must be 'application/json'."),
    ] = "application/json"
    accept: Annotated[
        Literal["application/json"],
        Field(description="The accept header. This must be 'application/json'."),
    ] = "application/json"
    model_id: Annotated[
        str,
        Field(
            description="The model ID to use for generating embeddings. The value for this depends on the model, your account, and other factors. [See the Bedrock docs](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html) for more information. tl;dr use the model ARN if you aren't sure."
        ),
    ]
    trace: Annotated[
        Literal["ENABLED", "DISABLED", "ENABLED_FULL"],
        Field(
            description="The trace level to use for the request. This controls the amount of tracing information returned in the response."
        ),
    ] = "DISABLED"
    guardrail_identifier: Annotated[
        str | None,
        Field(
            description="The guardrail identifier to use for the request. This is used to enforce safety and compliance policies. We'll default to null/None. If you need this, you'll know."
        ),
    ] = None
    guardrail_version: Annotated[
        str | None, Field(description="The guardrail version to use, if using guardrail.")
    ] = None
    performance_config_latency: Annotated[
        Literal["standard", "optimized"],
        Field(
            description="The performance configuration to use for the request. This controls the latency and throughput of the request."
        ),
    ] = "standard"


class TitanEmbeddingV2Response(BaseBedrockModel):
    """Response from Titan Embedding V2."""

    embedding: Annotated[
        list[float], Field(description="The generated embedding as a list of floats.")
    ]
    input_text_token_count: Annotated[
        int, Field(description="The number of tokens in the input text.")
    ]
    embeddings_by_type: Annotated[
        dict[Literal["float", "binary"], list[float] | list[int]],
        Field(description="The generated embeddings by type."),
    ]


class BedrockInvokeEmbeddingResponse(BaseBedrockModel):
    """Response from Bedrock embedding."""

    body: Annotated[
        bytes,
        Field(
            description="The body of the response. This is what AWS calls a `StreamingBody` response -- a bytestream."
        ),
    ]
    content_type: Annotated[
        str,
        Field(
            description="The mimetype of the response body. Most likely 'application/json', but AWS isn't clear on this."
        ),
    ] = "application/json"


class BedrockEmbeddingProvider(EmbeddingProvider[BedrockClient]):
    """Bedrock embedding provider."""

    _client: BedrockClient
