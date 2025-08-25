"""Capabilities for OpenAI embedding models."""

from typing import Literal

from codeweaver._settings import Provider
from codeweaver.embedding.capabilities.base import EmbeddingModelCapabilities, PartialCapabilities


def _get_openai_models(
    provider: Provider,
) -> tuple[
    Literal["text-embedding-3-small", "openai/text-embedding-3-small"],
    Literal["text-embedding-3-large", "openai/text-embedding-3-large"],
]:
    match provider:
        case Provider.OPENAI | Provider.AZURE | Provider.VERCEL:
            return ("text-embedding-3-small", "text-embedding-3-large")
        case Provider.GITHUB:
            return ("openai/text-embedding-3-small", "openai/text-embedding-3-large")


def _get_providers() -> tuple[Provider, ...]:
    return (Provider.OPENAI, Provider.AZURE, Provider.GITHUB, Provider.VERCEL)


def _get_shared_openai_embedding_capabilities() -> PartialCapabilities:
    return {
        "version": 3,
        "preferred_metrics": ("dot", "cosine"),
        "supports_custom_prompts": False,
        "is_normalized": True,
        "tokenizer": "tiktoken",
        "tokenizer_model": "cl100k_base",
        "supports_context_chunk_embedding": False,
        "context_window": 8192,
        "default_dtype": "float",
        "output_dtypes": ("float",),
    }


def get_openai_embedding_capabilities() -> tuple[EmbeddingModelCapabilities, ...]:
    """Get the capabilities for OpenAI embedding models."""
    dimensions = (3072, 2560, 2048, 1536, 1024, 512, 256)
    return tuple(
        EmbeddingModelCapabilities.model_validate({
            **_get_shared_openai_embedding_capabilities(),
            "name": model_name,
            "provider": provider,
            "default_dimension": 1536 if "small" in model_name else 3072,
            "output_dimensions": dimensions if "large" in model_name else dimensions[3:],
        })
        for provider in _get_providers()
        for model_name in _get_openai_models(provider)
    )
