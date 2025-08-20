# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base class for reranking providers."""

from typing import Literal

from codeweaver.reranking.base import RerankingProvider


type KnownRerankModelName = Literal[
    "voyage:voyage-rerank-2.5",
    "voyage:voyage-rerank-2.5-lite",
    "cohere:rerank-v3.5",
    "cohere:rerank-english-v3.0",
    "cohere:rerank-multilingual-v3.0",
    "bedrock:amazon.rerank-v1:0",
    "bedrock:cohere.rerank-v3-5:0",
    "huggingface:BAAI/bge-reranker-v2-m3",
    "huggingface:BAAI/bge-reranker-large",
    "huggingface:Qwen/Qwen3-Reranker-0.6B",
    "huggingface:Qwen/Qwen3-Reranker-4B",
    "huggingface:Qwen/Qwen3-Reranker-8B",
    "huggingface:mixedbread-ai/mxbai-rerank-large-v1",
    "huggingface:mixedbread-ai/mxbai-rerank-large-v2",
    "huggingface:jinaai/jina-reranker-m0",
    "huggingface:Alibaba-NLP/gte-multilingual-reranker-base",
    "huggingface:mixedbread-ai/mxbai-rerank-xsmall-v1",
    "huggingface:mixedbread-ai/mxbai-rerank-base-v1",
]


def get_rerank_model_provider() -> None:  # -> EmbeddingProvider[Any]:
    """Get rerank model provider."""


__all__ = ("KnownRerankModelName", "RerankingProvider")
