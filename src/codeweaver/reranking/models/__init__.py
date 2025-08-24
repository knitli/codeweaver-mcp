"""Entry point for reranking models."""

from codeweaver.reranking.models.base import RerankingModelCapabilities


def get_model(model: str) -> RerankingModelCapabilities:
    """Get the reranking model class by name."""
    if model.startswith("voyage"):
        from codeweaver.reranking.models.voyage import get_voyage_reranking_capabilities

        return next(model for model in get_voyage_reranking_capabilities() if model.name == model)
    if model.startswith("cohere"):
        from codeweaver.reranking.models.cohere import get_cohere_reranking_capabilities

        return next(model for model in get_cohere_reranking_capabilities() if model.name == model)
    raise ValueError(f"Unknown reranking model: {model}. Maybe check your spelling or the syntax?")
