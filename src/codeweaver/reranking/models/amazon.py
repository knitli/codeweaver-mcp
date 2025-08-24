"""Cohere reranking model capabilities."""

from collections.abc import Sequence

from pydantic import NonNegativeInt

from codeweaver._data_structures import CodeChunk
from codeweaver.reranking.models.base import (
    PartialRerankingCapabilities,
    RerankingModelCapabilities,
)


def amazon_max_input(chunks: Sequence[CodeChunk], query: str) -> tuple[bool, NonNegativeInt]:
    """Determine the maximum input length for the Amazon model."""


def _get_common_capabilities() -> PartialRerankingCapabilities:
    return {
        "name": "amazon.rerank-v1:0",
        "max_input": amazon_max_input,
        "context_window": 4096,
        "supports_custom_prompt": False,
        "tokenizer": "tokenizers",
    }


def get_amazon_reranking_capabilities() -> tuple[RerankingModelCapabilities, ...]:
    """Get the capabilities of the Amazon reranking model."""
    base_capabilities = _get_common_capabilities()
