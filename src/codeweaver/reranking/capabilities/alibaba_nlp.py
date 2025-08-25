"""Reranking model capabilities for Alibaba NLP models."""

from collections.abc import Sequence

from codeweaver._settings import Provider
from codeweaver.reranking.capabilities.base import RerankingModelCapabilities


def get_alibaba_reranking_capabilities() -> Sequence[RerankingModelCapabilities]:
    """
    Get the reranking capabilities for Alibaba NLP models.
    """
    return [
        RerankingModelCapabilities.model_validate({
            "name": "Alibaba-NLP/gte-multilingual-reranker-base",
            "tokenizer": "tokenizers",
            "tokenizer_model": "Alibaba-NLP/gte-multilingual-reranker-base",
            "supports_custom_prompt": False,
            "max_input": 8192,
            "context_window": 8192,
            "provider": Provider.SENTENCE_TRANSFORMERS,
        })
    ]
