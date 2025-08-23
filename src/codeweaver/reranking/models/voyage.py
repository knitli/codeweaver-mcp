"""Reranking models for VoyageAI."""

from pydantic import NonNegativeInt

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import Provider
from codeweaver.reranking.models.base import (
    PartialRerankingCapabilities,
    RerankingModelCapabilities,
)


try:
    from voyageai.object.reranking import RerankingObject, RerankingResult

    from codeweaver.tokenizers import get_tokenizer

except ImportError as e:
    raise ImportError(
        "Voyage AI SDK is not installed. Use `pip install voyageai` to install it."
    ) from e


def _handle_too_big(token_list: list[int]) -> list[tuple[int, int]]:
    """Handle the case where a single token exceeds the maximum size."""
    return [(i, size) for i, size in enumerate(token_list) if size > 32_000]


def _handle_too_large(token_list: list[int]) -> tuple[bool, NonNegativeInt]:
    """Determine if the token list fits within the total limit and where to cut."""
    summed: int = 0
    for i, size in enumerate(token_list):
        if summed + size > 600_000:
            return False, i - 1 if i > 0 else 0
        summed += size
    return True, 0


def _voyage_max_limit(chunks: list[CodeChunk], prompt: str) -> tuple[bool, NonNegativeInt]:
    """Check if the number of chunks exceeds the maximum limit."""
    tokenizer = get_tokenizer("tokenizers", "voyageai/voyage-rerank-2.5")
    stringified_chunks = [chunk.serialize() for chunk in chunks]
    sizes = [tokenizer.estimate(chunk) + tokenizer.estimate(prompt) for chunk in stringified_chunks]
    too_large = sum(sizes) > 600_000
    too_many = len(stringified_chunks) > 1000
    too_big = any(size > 32_000 for size in sizes)
    if not too_large and not too_many and not too_big:
        return True, 0
    if too_big and (problem_chunks := _handle_too_big(sizes)):
        raise ValueError(
            f"Some chunks are too big: {problem_chunks}. Voyage AI requires each chunk to be less than 32,000 tokens."
        )
    if too_large and not too_many:
        return _handle_too_large(sizes)
    if too_many:
        # Truncate to the first 1000 chunks and re-evaluate once without recursion.
        truncated_chunks = chunks[:1000]
        truncated_strings = [chunk.serialize() for chunk in truncated_chunks]
        truncated_sizes = [
            tokenizer.estimate(c) + tokenizer.estimate(prompt) for c in truncated_strings
        ]
        # If still too large, determine where to cut; otherwise accept the truncated set.
        if sum(truncated_sizes) > 600_000:
            return _handle_too_large(truncated_sizes)
        return True, 1000
    # If none of the above conditions apply, return a conservative failure.
    return False, 0


def _get_voyage_capabilities() -> PartialRerankingCapabilities:
    return {
        "name": "rerank-2.5",
        "provider": Provider.VOYAGE,
        "max_query": 8_000,
        "max_input": _voyage_max_limit,
        "context_window": 32_000,
        "supports_custom_prompt": True,
        "custom_prompt": "Please re-rank the following options:",
        "tokenizer": "tokenizers",
        "tokenizer_model": "voyageai/voyage-rerank-2.5",
    }


def get_voyage_reranking_capabilities() -> tuple[
    RerankingModelCapabilities, RerankingModelCapabilities
]:
    """Get the capabilities of the Voyage reranking model."""
    base_capabilities = _get_voyage_capabilities()
    lite_capabilities = base_capabilities.copy()
    lite_capabilities["name"] = "voyage-rerank-2.5-lite"
    return RerankingModelCapabilities.model_validate(
        **base_capabilities
    ), RerankingModelCapabilities.model_validate(**lite_capabilities)
