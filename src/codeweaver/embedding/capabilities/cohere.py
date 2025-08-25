"""Capabilities for Cohere embedding models."""

from types import MappingProxyType

from codeweaver._settings import Provider
from codeweaver.embedding.capabilities.base import EmbeddingModelCapabilities, PartialCapabilities


MODEL_MAP: MappingProxyType[Provider, tuple[str, ...]] = MappingProxyType({
    Provider.COHERE: (
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-multilingual-light-v3.0",
        "embed-v4.0",
    ),
    Provider.BEDROCK: ("cohere.embed-english-v3.0", "cohere.embed-multilingual-v3.0"),
    Provider.GITHUB: ("cohere/Cohere-embed-v3-english", "cohere/Cohere-embed-v3-multilingual"),
    Provider.HEROKU: ("cohere-embed-multilingual",),  # this is v3.0, they just don't say it.
})


def _get_shared_cohere_embedding_capabilities() -> PartialCapabilities:
    return {"model": MODEL_MAP[Provider.COHERE], "provider": Provider.COHERE}


def get_cohere_embedding_capabilities() -> EmbeddingModelCapabilities:
    """Get the capabilities for cohere embedding models."""
