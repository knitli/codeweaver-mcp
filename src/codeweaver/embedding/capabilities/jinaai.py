"""Settings for JinaAI embedding models."""

from typing import cast

from codeweaver._settings import Provider
from codeweaver.embedding.capabilities.base import (
    EmbeddingModelCapabilities,
    PartialCapabilities,
    default_input_transformer,
    fastembed_all_kwargs,
    huggingface_hub_embed_kwargs,
    huggingface_hub_input_transformer,
    huggingface_hub_output_transformer,
    huggingface_hub_query_kwargs,
)


CAP_MAP = {
    "jina-embeddings-v2-base-code": {
        "providers": {Provider.FASTEMBED, Provider.HUGGINGFACE},
        "default_dimension": 768,
    },
    "jina-embeddings-v3": {"providers": {Provider.FASTEMBED}, "default_dimension": 1024},
    "jina-embeddings-v4": {"providers": {Provider.HUGGINGFACE}, "default_dimension": 2048},
}


def _get_shared_capabilities() -> PartialCapabilities:
    """Get the shared capabilities for all JinaAI embedding models."""
    return {
        "is_normalized": False,
        "context_window": 8_192,
        "tokenizer": "tokenizers",
        "tokenizer_model": "jinaai/",
        "supports_context_chunk_embedding": False,
        "preferred_metrics": ("cosine", "dot"),
    }


def get_jinaai_model_capapabilities() -> tuple[EmbeddingModelCapabilities, ...]:
    """Get capabilities for all JinaAI embedding models."""
    shared_caps = _get_shared_capabilities()
    models: list[EmbeddingModelCapabilities] = []
    for model_name, model_cap in CAP_MAP.items():
        models.extend(
            EmbeddingModelCapabilities.model_validate({
                **shared_caps,
                **model_cap,
                "model_name": model_name,
                "provider": provider,
                "doc_kwargs": huggingface_hub_embed_kwargs()
                if provider == Provider.HUGGINGFACE
                else fastembed_all_kwargs(),
                "query_kwargs": huggingface_hub_query_kwargs()
                if provider == Provider.HUGGINGFACE
                else fastembed_all_kwargs(),
                "tokenizer_model": f"{shared_caps['tokenizer_model']}{next(v for v in ('jina-embeddings-v2', 'jina-embeddings-v3', 'jina-embeddings-v4') if v in model_name)}",
                "version": next(v for v in (2, 3, 4) if str(v) in model_name),
                "output_transformer": huggingface_hub_output_transformer,
                "input_transformer": huggingface_hub_input_transformer
                if provider == Provider.HUGGINGFACE
                else default_input_transformer,
            })
            for provider in cast(list[Provider], model_cap["providers"])
        )
    return tuple(models)
