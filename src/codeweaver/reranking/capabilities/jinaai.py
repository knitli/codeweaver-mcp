"""Reranking model capabilities for JinaAI models."""

from collections.abc import Sequence

from codeweaver._settings import Provider
from codeweaver.reranking.capabilities.base import RerankingModelCapabilities


def get_jinaai_reranking_capabilities() -> Sequence[RerankingModelCapabilities]:
    """Get the JinaAI reranking model capabilities."""
    capabilities = {
        "jinaai/jina-reranker-v2-base-multilingual": {
            "provider": Provider.FASTEMBED,
            "max_input": 8192,
            "context_window": 8192,
            "max_query_length": 512,
            "tokenizer": "tokenizers",
            "tokenizer_model": "jinaai/jina-reranker-v2-base-multilingual",
            "supports_custom_prompt": False,
        },
        "jinaai/jina-reranker-m0": {
            "provider": Provider.SENTENCE_TRANSFORMERS,
            "max_input": 10_240,
            "context_window": 10_240,
            "tokenizer": "tokenizers",
            "tokenizer_model": "jinaai/jina-reranker-m0",
            "supports_custom_prompt": False,
        },
    }
    return [
        RerankingModelCapabilities.model_validate({**cap, "name": name})
        for name, cap in capabilities.items()
    ]
