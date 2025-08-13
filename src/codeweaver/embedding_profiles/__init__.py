"""Entry point for embedding profiles."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Any, ClassVar, Literal, LiteralString, TypedDict

from pydantic import ConfigDict, Field, PositiveInt, computed_field
from pydantic.dataclasses import dataclass
from pydantic_ai.providers import Provider as PydanticAIProvider


@dataclass(frozen=True, config=ConfigDict(extra="allow", str_strip_whitespace=True))
class EmbeddingProfile:
    """
    Describes static / analytic metadata for an embedding model.

    EmbeddingProfile contains metadata about an embedding model, such as its
    dimensionality, context window size, and other relevant properties.

    EmbeddingProfile is modeled after `pydantic_ai.profiles.ModelProfile`, but intentionally
    separated to prevent conflating agent request concerns with analytic/vector-space metadata.

    All fields except `PydanticAIProvider` and `model_name` are optional to allow partial info when providers do not
    publish complete specifications. You may define additional settings or metadata in the `extras` field.
    """

    provider: PydanticAIProvider[Any] | LiteralString
    model_name: LiteralString

    dimension: Annotated[
        Literal[128, 256, 384, 512, 768, 1024, 2048, 3084] | None,
        Field(description="The dimensionality of the embedding vectors."),
    ] = None
    max_input_tokens: Annotated[
        PositiveInt | None,
        Field(description="The maximum number of input tokens for the model.", ge=256),
    ] = 1024
    max_batch_size: Annotated[
        PositiveInt | None, Field(description="The maximum batch size for the model.")
    ] = None

    context_window: Annotated[
        PositiveInt | None, Field(description="The model's context window size in tokens.")
    ] = None

    cost_per_cost_unit: Annotated[
        Decimal | None, Field(description="Cost per cost unit, e.g. per 1K tokens.")
    ] = None
    cost_unit: Annotated[
        Literal["1K_tokens", "1M_tokens"],
        Field(
            description="Cost unit used for pricing, typically a value per-number of tokens, e.g. '1K_tokens' or '1M_tokens'."
        ),
    ] = "1K_tokens"

    vectors_normalized: Annotated[
        bool,
        Field(description="Whether vectors are normalized to unit length 1", alias="normalized"),
    ] = False

    distance_metrics: Annotated[
        tuple[LiteralString, ...],
        Field(
            description="Distance metrics known to work well with this model. Will default to ('cosine',) if not specified."
        ),
    ] = ("cosine",)

    # Free-form details or provider-specific extras.
    description: str | None = None

    extras: ClassVar[
        Annotated[
            dict[str, Any] | None,
            Field(
                default_factory=dict,
                description="Extra settings, metadata, provider or model-specific settings.",
            ),
        ]
    ] = {}

    @computed_field
    def optimal_distance_metric(self) -> LiteralString | None:
        """Returns the optimal distance metric for this embedding profile."""
        if not self.distance_metrics:
            return None
        if self.vectors_normalized:
            # vectors normalized to length 1 produce identical results for cosine and dot products.
            # **and** dot-product is much faster to compute.
            return (
                "dot"
                if "dot" in self.distance_metrics
                else "cosine"
                if "cosine" in self.distance_metrics
                else self.distance_metrics[0]
            )
        # Heuristic: prefer cosine if available, otherwise first in list.
        return "cosine" if "cosine" in self.distance_metrics else self.distance_metrics[0]

    def estimate_cost(self, tokens: int) -> Decimal | None:
        """Returns the estimated cost for the given number of tokens."""
        if not self.cost_unit or not self.cost_per_cost_unit or not self.cost_per_1k_tokens:
            return None
        return self.cost_per_1k_tokens * Decimal(tokens) / Decimal(1000)

    @property
    def cost_per_1k_tokens(self) -> Decimal | None:
        """Returns the cost per 1K tokens for this embedding profile."""
        if not self.cost_unit or not self.cost_per_cost_unit:
            return None
        if self.cost_unit == "1K_tokens":
            return self.cost_per_cost_unit
        if self.cost_unit == "1M_tokens":
            return self.cost_per_cost_unit / Decimal(1000)
        return None


# Registry pattern: provider -> function(model_name) -> EmbeddingProfile | None
_REGISTRY: dict[str, Callable[[str], EmbeddingProfile | None]] = {}


def register_embedding_profiles(
    provider: str, resolver: Callable[[str], EmbeddingProfile | None]
) -> None:
    """Register a new embedding profile resolver for a specific provider."""
    _REGISTRY[provider] = resolver


def get_embedding_profile(
    provider: PydanticAIProvider[Any] | LiteralString, model_name: LiteralString
) -> EmbeddingProfile | None:
    """
    Retrieve the embedding profile for a specific model from a provider.
    """
    if (resolver := _REGISTRY.get(provider)) and (prof := resolver(model_name)):
        return prof
    return None


class EmbeddingProfileValues(TypedDict, total=False):
    """
    Represents the values of an embedding profile.

    See `EmbeddingProfile` for more details.

    All values are optional to allow partial information. Fields that aren't provided
    will be the default values as defined in the `EmbeddingProfile` class (mostly `None`).
    """

    provider: LiteralString | None
    dimension: Literal[128, 256, 384, 512, 768, 1024, 2048, 3084] | None
    max_input_tokens: PositiveInt | None
    max_batch_size: PositiveInt | None
    context_window: PositiveInt | None
    cost_per_cost_unit: Decimal | None
    cost_unit: Literal["1K_tokens", "1M_tokens"] | None
    distance_metrics: tuple[LiteralString, ...] | None
    vectors_normalized: bool | None
    description: str | None
    extras: dict[str, Any] | None


ModelName = LiteralString
"""The name of the embedding model, as it would be used in API calls. Like: 'text-embedding-3-small'"""
EmbeddingProfileDict = dict[ModelName, EmbeddingProfileValues]
"""Dictionary for embedding profiles for different models."""
