"""Reranking provider for FastEmbed."""

import logging
import multiprocessing

from collections.abc import Sequence
from typing import Any, ClassVar

from codeweaver._settings import Provider
from codeweaver.reranking.capabilities.base import RerankingModelCapabilities
from codeweaver.reranking.providers.base import RerankingProvider


logger = logging.getLogger(__name__)

try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

except ImportError as e:
    logger.exception("Failed to import TextCrossEncoder from fastembed.rerank.cross_encoder")
    raise ImportError(
        "FastEmbed is not installed. Please install it with `pip install fastembed`."
    ) from e


def fastembed_kwargs(**kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Get all possible kwargs for FastEmbed embedding methods."""
    default_kwargs: dict[str, Any] = {"threads": multiprocessing.cpu_count(), "lazy_load": True}
    if kwargs:
        device_ids: list[int] | None = kwargs.get("device_ids")  # pyright: ignore[reportAssignmentType]
        cuda: bool | None = kwargs.get("cuda")  # pyright: ignore[reportAssignmentType]
        if cuda == False:  # user **explicitly** disabled cuda  # noqa: E712
            return default_kwargs | kwargs
        cuda = bool(cuda)
        from codeweaver._utils import decide_fastembed_runtime

        decision = decide_fastembed_runtime(explicit_cuda=cuda, explicit_device_ids=device_ids)
        if isinstance(decision, tuple) and len(decision) == 2:
            cuda = True
            device_ids = decision[1]
        elif decision == "gpu":
            cuda = True
            device_ids = [0]
        else:
            cuda = False
            device_ids = None
        if cuda:
            kwargs["cuda"] = True  # pyright: ignore[reportArgumentType]
            kwargs["device_ids"] = device_ids  # pyright: ignore[reportArgumentType]
            kwargs["providers"] = ["CUDAExecutionProvider"]  # pyright: ignore[reportArgumentType]
    return default_kwargs | kwargs


class FastEmbedRerankingProvider(RerankingProvider[TextCrossEncoder]):
    """
    FastEmbed implementation of the reranking provider.

    model_name: The name of the FastEmbed model to use.
    """

    _client: TextCrossEncoder
    _provider: Provider = Provider.FASTEMBED
    _caps: RerankingModelCapabilities

    _kwargs: ClassVar[dict[str, Any]] = fastembed_kwargs()

    # default transformers work fine for fastembed]

    def _initialize(self) -> None:
        if "model_name" not in self.kwargs:
            self.kwargs["model_name"] = self._caps.name
        self._client = TextCrossEncoder(**self.kwargs)  # pyright: ignore[reportArgumentType]

    async def _execute_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int = 40,
        **kwargs: dict[str, Any] | None,
    ) -> Any:
        """Execute the reranking process."""
        try:
            # our batch_size needs to be the number of documents because we only get back the scores.
            # If we set it to a lower number, we wouldn't know what documents the scores correspond to without some extra setup.
            response = self.client.rerank(
                query=query, documents=documents, batch_size=len(documents)
            )
        except Exception as e:
            raise RuntimeError(f"Error during reranking with FastEmbed: {e}") from e
        else:
            return response
