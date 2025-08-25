# sourcery skip: no-complex-if-expressions
"""Reranking provider for FastEmbed."""

import asyncio
import logging

from collections.abc import Sequence
from typing import Any, ClassVar

from codeweaver._settings import Provider
from codeweaver.reranking.capabilities.base import RerankingModelCapabilities
from codeweaver.reranking.providers.base import RerankingProvider


logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder

except ImportError as e:
    logger.exception("Failed to import CrossEncoder from sentence_transformers")
    raise ImportError(
        "SentenceTransformers is not installed. Please install it with `pip install sentence-transformers`."
    ) from e


def preprocess_for_qwen(
    query: str,
    documents: Sequence[str],
    instruction: str,
    prefix: str,
    suffix: str,
    model_name: str,
) -> Sequence[tuple[str, str]]:
    """Preprocess the query and documents for Qwen models."""

    def format_doc(doc: str) -> tuple[str, str]:
        return (
            f"{prefix}<Instruct>: {instruction}\n<Query>:\n{query}\n",
            f"<Document>:\n{doc}{suffix}",
        )

    return [format_doc(doc) for doc in documents]


class SentenceTransformersRerankingProvider(RerankingProvider[CrossEncoder]):
    """
    SentenceTransformers implementation of the reranking provider.

    model_name: The name of the SentenceTransformers model to use.
    """

    _client: CrossEncoder
    _provider: Provider = Provider.SENTENCE_TRANSFORMERS
    _caps: RerankingModelCapabilities

    _kwargs: ClassVar[dict[str, Any]] = {"trust_remote_code": True}

    def __init__(
        self,
        capabilities: RerankingModelCapabilities,
        client: CrossEncoder | None = None,
        prompt: str | None = None,
        top_k: int = 40,
        **kwargs: Any,
    ) -> None:
        """Initialize the SentenceTransformersRerankingProvider."""
        self._caps = capabilities
        self._client = client or CrossEncoder(self._caps.name, **self._kwargs)
        self._kwargs = kwargs
        self._prompt = prompt
        self._top_k = top_k
        super().__init__(capabilities, client=self._client, prompt=prompt, top_k=top_k, **kwargs)

    def _initialize(self) -> None:
        """
        Initialize the SentenceTransformersRerankingProvider.
        """
        if "model_name" not in self._kwargs:
            self._kwargs["model_name"] = self._caps.name
        self._client = self.get_client_for_model(self._kwargs["model_name"])
        if "Qwen3" in self._kwargs["model_name"]:
            self._setup_qwen3()

    async def _execute_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int = 40,
        **kwargs: dict[str, Any] | None,
    ) -> Any:
        """Execute the reranking process."""
        preprocessed = (
            preprocess_for_qwen(
                query=query,
                documents=documents,
                instruction=self._caps.custom_prompt,
                prefix=self._query_prefix,
                suffix=self._doc_suffix,
                model_name=self.kwargs["model_name"],
            )
            if "Qwen3" in self._caps.name
            else [(query, doc) for doc in documents]
        )
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._client.predict, preprocessed)
        return scores.tolist()

    def _setup_qwen3(self) -> None:
        """Sets up Qwen3 specific parameters."""
        if "Qwen3" not in self._kwargs["model_name"]:
            return
        from importlib import metadata

        try:
            has_flash_attention = metadata.version("flash_attn")
        except Exception:
            has_flash_attention = None

        if extra := self._caps.extra:
            self._query_prefix = f"{extra.get('prefix', '')}{self._caps.custom_prompt}\n<Query>:\n"
            self._doc_suffix = extra.get("suffix", "")
        self.kwargs["model_kwargs"] = {"torch_dtype": "torch.float16"}
        if has_flash_attention:
            self.kwargs["model_kwargs"]["attention_implementation"] = "flash_attention_2"
