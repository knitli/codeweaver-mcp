# sourcery skip: no-complex-if-expressions
import logging

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, JsonValue, PositiveInt, model_validator

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import Provider
from codeweaver.reranking.models.base import RerankingModelCapabilities
from codeweaver.reranking.providers.base import (
    RerankingProvider,
    RerankingResult,
    StructuredDataInput,
    StructuredDataSequence,
)


logger = logging.getLogger(__name__)

VALID_REGIONS = ["us-west-2", "ap-northeast-1", "ca-central-1", "eu-central-1"]
"""AWS has rerank models available in very few regions. https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""
VALID_REGION_PATTERN = "|".join(VALID_REGIONS)


class BedrockTextQuery(BaseModel):
    """A query for reranking."""

    text_query: Annotated[
        dict[
            Literal["text"],
            Annotated[str, Field(description="The text of the query.", max_length=32_000)],
        ],
        Field(description="The text query.", serialization_alias="textQuery"),
    ]
    # we need to avoid the `type` keyword in python
    kind: Annotated[
        Literal["TEXT"], Field(description="The kind of query.", serialization_alias="type")
    ] = "TEXT"


class BedrockRerankModelConfiguration(BaseModel):
    """Configuration for a Bedrock reranking model."""

    additional_model_request_fields: Annotated[
        None,
        Field(
            description="A json object where each key is a model parameter and the value is the value for that parameter. Currently there's not any values worth setting that can't be set elsewhere.",
            serialization_alias="additionalModelRequestFields",
        ),
    ] = None
    model_arn: Annotated[
        str,
        Field(
            description="The ARN of the model.",
            serialization_alias="modelArn",
            pattern=r"^arn:aws:bedrock:(" + VALID_REGION_PATTERN + r"):\d{12}:.*$",
        ),
    ]


class BedrockRerankConfiguration(BaseModel):
    """Configuration for Bedrock reranking."""

    model_configuration: Annotated[
        BedrockRerankModelConfiguration,
        Field(description="The model configuration.", serialization_alias="modelConfiguration"),
    ]
    number_of_results: Annotated[
        PositiveInt, Field(description="Number of results to return -- this is `top_k`.")
    ] = 40


class RerankConfiguration(BaseModel):
    """Configuration for reranking."""

    bedrock_reranking_configuration: Annotated[
        BedrockRerankConfiguration,
        Field(
            description="Configuration for reranking.",
            serialization_alias="bedrockRerankingConfiguration",
        ),
    ]
    kind: Annotated[
        Literal["BEDROCK_RERANKING_MODEL"],
        Field(description="The kind of configuration.", serialization_alias="type"),
    ] = "BEDROCK_RERANKING_MODEL"


class DocumentSource(BaseModel):
    """A document source for reranking."""

    json_document: Annotated[
        dict[str, JsonValue] | None,
        Field(
            description="A Json document to rerank against. Practically, CodeWeaver will always use this.",
            serialization_alias="jsonDocument",
        ),
    ]
    text_document: Annotated[
        dict[Literal["text"], str] | None,
        Field(description="A text document to rerank against.", serialization_alias="textDocument"),
    ] = None
    kind: Annotated[
        Literal["JSON", "TEXT"],
        Field(description="The kind of document.", serialization_alias="type"),
    ] = "JSON"

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        """Validate that exactly one document type is provided."""
        if (self.json_document and self.text_document) or (
            not self.json_document and not self.text_document
        ):
            raise ValueError("Exactly one of json_document or text_document must be provided.")
        return self


class BedrockInlineDocumentSource(BaseModel):
    """An inline document source for reranking."""

    inline_document_source: Annotated[
        DocumentSource,
        Field(
            description="The inline document source to rerank.",
            serialization_alias="inlineDocumentSource",
        ),
    ]
    kind: Annotated[
        Literal["INLINE"],
        Field(description="The kind of document source.", serialization_alias="type"),
    ] = "INLINE"


class BedrockRerankRequest(BaseModel):
    """Request for Bedrock reranking."""

    queries: Annotated[
        list[BedrockTextQuery], Field(description="List of text queries to rerank against.")
    ]
    reranking_configuration: Annotated[
        RerankConfiguration,
        Field(
            description="Configuration for reranking.", serialization_alias="rerankingConfiguration"
        ),
    ]
    sources: Annotated[
        list[BedrockInlineDocumentSource],
        Field(description="List of document sources to rerank against."),
    ]
    next_token: Annotated[str | None, Field(serialization_alias="nextToken")] = None


class BedrockRerankResultItem(BaseModel):
    """A single reranked result item."""

    document: Annotated[DocumentSource, Field(description="The document that was reranked.")]
    index: Annotated[
        PositiveInt,
        Field(description="The ranking of the document in the results. (Lower is better.)"),
    ]
    relevance_score: Annotated[
        float,
        Field(
            description="The relevance score of the document. Higher values indicate greater relevance.",
            serialization_alias="relevanceScore",
        ),
    ]


class BedrockRerankingResult(BaseModel):
    """Result of a Bedrock reranking request."""

    results: Annotated[
        list[BedrockRerankResultItem], Field(description="List of reranked results.")
    ]
    next_token: Annotated[
        str | None,
        Field(
            description="Token for the next set of results, if any.",
            serialization_alias="nextToken",
        ),
    ] = None


try:
    from boto3 import client as boto3_client  # pyright: ignore[reportUnknownVariableType]

    bedrock_client = boto3_client("bedrock")  # pyright: ignore[reportUnknownVariableType]

except ImportError as e:
    logger.exception("Failed to import boto3")
    raise ImportError("boto3 is not installed. Please install it with `pip install boto3`.") from e


class BedrockRerankingProvider(RerankingProvider[bedrock_client]):
    """Provider for Bedrock reranking."""

    _client: boto3_client = bedrock_client
    _provider = Provider.BEDROCK
    _caps: RerankingModelCapabilities

    _rerank_kwargs: dict[str, Any] | None

    def _initialize(self) -> None:
        self._input_transformer = self.bedrock_reranking_input_transformer
        self._output_transformer = self.bedrock_reranking_output_transformer

    @staticmethod
    def _to_doc_sources(documents: list[DocumentSource]) -> list[BedrockInlineDocumentSource]:
        return [
            BedrockInlineDocumentSource.model_validate([
                {"inline_document_source": doc.model_dump(mode="python"), "type": "INLINE"}
                for doc in documents
            ])
        ]

    def bedrock_reranking_input_transformer(
        self, documents: StructuredDataInput | StructuredDataSequence
    ) -> list[BedrockInlineDocumentSource]:  # this is the sources field of BedrockRerankRequest
        """Transform input documents into the format expected by the Bedrock API."""
        # Transform the input documents into the format expected by the Bedrock API
        if isinstance(documents, list | tuple | set):
            docs = [
                DocumentSource.model_validate(
                    {"json_document": doc.serialize(), "text_document": None}
                    if isinstance(doc, CodeChunk)
                    else {
                        "text_document": {"text": str(doc)},
                        "json_document": None,
                        "kind": "TEXT",
                    }
                )
                for doc in documents
            ]
        else:
            docs = (
                [
                    DocumentSource.model_validate({
                        "json_document": documents.serialize(),
                        "text_document": None,
                    })
                ]
                if isinstance(documents, CodeChunk)
                else [
                    DocumentSource.model_validate({
                        "text_document": {"text": str(documents)},
                        "json_document": None,
                        "kind": "TEXT",
                    })
                ]
            )
        return self._to_doc_sources(docs)

    def bedrock_reranking_output_transformer(
        self, response: BedrockRerankingResult, original_chunks: Sequence[CodeChunk]
    ) -> Sequence[RerankingResult]:
        """Transform the Bedrock API response into the format expected by the reranking provider."""
        results: list[RerankingResult] = []
        for item in response.results:
            chunk = CodeChunk.model_validate_json(item.document.json_document)  # pyright: ignore[reportUnknownArgumentType]
            results.append(
                RerankingResult(
                    original_index=original_chunks.index(chunk),
                    score=item.relevance_score,
                    batch_rank=item.index,
                    chunk=chunk,
                )
            )
        return results
