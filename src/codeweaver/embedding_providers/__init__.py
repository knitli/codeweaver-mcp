
from pydantic_ai.providers infer_provider
from codeweaver.embedding_providers.embedding import EmbeddingModel


# ... existing code ...


def infer_model(model: Model | KnownModelName | str) -> Model:  # noqa: C901
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    if model == "test":
        from .test import TestModel

        return TestModel()

    # --- New embedding inference logic start ---
    # Accepted patterns:
    #   <provider>-embedding:<embedding_model>
    #   <provider>:embedding:<embedding_model>
    # We parse first, then fall through to existing logic for non-embedding kinds.
    raw = model
    provider = None
    model_name = None
    is_embedding = False

    if ":" in raw:
        first, rest = raw.split(":", 1)
        if first.endswith("-embedding"):
            provider = first.removesuffix("-embedding")
            model_name = rest
            is_embedding = True
        elif rest.startswith("embedding:"):
            provider = first
            model_name = rest.split(":", 1)[1]
            is_embedding = True

    if is_embedding and provider and model_name:
        # Build an EmbeddingModel; provider must be known to existing provider inference.
        return EmbeddingModel(model_name, provider=provider)

    # --- Existing logic (unchanged) ---
    try:
        provider, model_name = model.split(":", maxsplit=1)
    except ValueError:
        model_name = model
        # TODO(Marcelo): We should deprecate this way.
        if model_name.startswith(("gpt", "o1", "o3")):
            provider = "openai"
        elif model_name.startswith("claude"):
            provider = "anthropic"
        elif model_name.startswith("gemini"):
            provider = "google-gla"
        else:
            raise UserError(f"Unknown model: {model}")

    if provider == "vertexai":
        provider = "google-vertex"  # pragma: no cover

    if provider == "cohere":
        from .cohere import CohereModel

        return CohereModel(model_name, provider=provider)
    if provider in (
        "openai",
        "deepseek",
        "azure",
        "openrouter",
        "vercel",
        "grok",
        "moonshotai",
        "fireworks",
        "together",
        "heroku",
        "github",
    ):
        from .openai import OpenAIModel

        return OpenAIModel(model_name, provider=provider)
    if provider == "openai-responses":
        from .openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider="openai")
    if provider in ("google-gla", "google-vertex"):
        from .google import GoogleModel

        return GoogleModel(model_name, provider=provider)
    if provider == "groq":
        from .groq import GroqModel

        return GroqModel(model_name, provider=provider)
    if provider == "mistral":
        from .mistral import MistralModel

        return MistralModel(model_name, provider=provider)
    if provider == "anthropic":
        from .anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=provider)
    if provider == "bedrock":
        from .bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name, provider=provider)
    if provider == "huggingface":
        from .huggingface import HuggingFaceModel

        return HuggingFaceModel(model_name, provider=provider)
    raise UserError(f"Unknown model: {model}")  # pragma: no cover
