from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.providers import Provider as PydanticAIProvider
from pydantic_ai.providers import infer_provider

from codeweaver.embedding_profiles import EmbeddingProfile, get_embedding_profile
from codeweaver.exceptions import ConfigurationError


try:
    # Optional; only required for OpenAI provider usage.
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = Any  # type: ignore


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    model_name: str
    provider: str
    usage: dict[str, Any] | None = None
    profile: EmbeddingProfile | None = None


class EmbeddingModel(Model):
    """
    Base embedding model abstraction.

    Mirrors the style of other Models but focuses on vector generation.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: str | Provider[Any] | None = None,
        embedding_profile: EmbeddingProfile | None = None,
    ) -> None:
        if provider is None:
            raise ValueError(
                "provider must be specified for EmbeddingModel (explicit or inferred)."
            )
        if isinstance(provider, str):
            provider_obj = infer_provider(provider)
        else:
            provider_obj = provider
        self._provider: Provider[Any] = provider_obj
        self.model_name = model_name
        self._embedding_profile = embedding_profile or get_embedding_profile(
            provider_obj.name, model_name
        )

    @property
    def provider(self) -> PydanticAIProvider[Any]:
        """Get the provider for this model."""
        return self._provider

    @property
    def embedding_profile(self) -> EmbeddingProfile | None:
        """Get the embedding profile for this model."""
        return self._embedding_profile

    async def aembed(
        self, inputs: Sequence[str] | str, *, user: str | None = None, **kwargs: Any
    ) -> EmbeddingResult:
        """
        Generate embeddings for one or more input strings.

        kwargs are provider-specific passthrough (e.g. dimensions overrides if supported).
        """
        batched: list[str]
        batched = [inputs] if isinstance(inputs, str) else list(inputs)

        # Dispatch on provider
        provider_name = self._provider.name
        if provider_name == "openai":
            return await self._openai_aembed(batched, user=user, **kwargs)
        raise ConfigurationError(
            f"You called the `async embed` method for {self.provider!r}, but you don't seem to have that provider configured."
        )

    async def _openai_aembed(
        self, batched: list[str], *, user: str | None, **kwargs: Any
    ) -> EmbeddingResult:
        client: AsyncOpenAI = self._provider.client  # type: ignore[attr-defined]

        resp = await client.embeddings.create(
            model=self.model_name, input=batched, user=user, **kwargs
        )
        # openai response has .data list with items having .embedding
        vectors = [item.embedding for item in resp.data]
        usage = getattr(resp, "usage", None)
        return EmbeddingResult(
            embeddings=vectors,
            model_name=self.model_name,
            provider=self._provider.name,
            usage=dict(usage) if usage else None,
            profile=self._embedding_profile,
        )

    # (Optional) sync convenience wrapper if project conventions allow:
    # def embed(...): return asyncio.run(self.aembed(...))
