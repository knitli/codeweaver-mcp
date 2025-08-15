# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding/`)
"""Entrypoint for CodeWeaver's heavily-pydantic-ai-inspired embedding model system."""

# sourcery skip: avoid-global-variables
from typing import Literal

from codeweaver.embedding.profiles import (
    DEFAULT_PROFILE,
    EmbeddingModelProfile,
    EmbeddingModelProfileSpec,
)
from codeweaver.embedding.providers import EmbeddingProvider, infer_embedding_provider


# placeholders just to keep the imports here withut ruff removing them
default_profile = DEFAULT_PROFILE
embedding_model_profile = EmbeddingModelProfile
embedding_model_profile_spec = EmbeddingModelProfileSpec
embedding_provider = EmbeddingProvider
infer_embedding_provider = infer_embedding_provider


class EmbeddingModelSettings: ...


class RerankModelSettings: ...


type KnownEmbeddingModelName = Literal[
    "azure:embed-v-4-0",  # cohere v4
    "azure:onnx-models/all-minilm-l6-v2-onnx",
    "azure:text-embedding-3-large",
    "azure:text-embedding-3-small",
    "bedrock:amazon.titan-embed-text-v2:0",
    "bedrock:cohere.embed-english-v3.0",
    "bedrock:cohere.embed-multilingual-v3.0",
    "bedrock:twelvelabs.marengo-embed-2-7-v1:0",
    "cohere:embed-english-v3.0",
    "cohere:embed-multilingual-light-v3.0",
    "cohere:embed-multilingual-v3.0",
    "cohere:embed-v4.0",
    "fireworks:WhereIsAI/UAE-Large-V1",
    "fireworks:nomic-ai/nomic-embed-text-v1",
    "fireworks:nomic-ai/nomic-embed-text-v1.5",
    "fireworks:thenlper/gte-base",
    "fireworks:thenlper/gte-large",
    "github:cohere/Cohere-embed-v3-english",
    "github:cohere/Cohere-embed-v3-multilingual",
    "github:openai/text-embedding-3-large",
    "github:openai/text-embedding-3-small",
    "google:gemini-embedding-001",
    "google:gemini-embedding-exp-03-07",
    "heroku:cohere-embed-multilingual",
    "huggingface:Ling-AI-Research/Ling-Embed-Mistral",
    "huggingface:Qwen/Qwen3-Embedding-0.6B",
    "huggingface:Qwen/Qwen3-Embedding-4B",
    "huggingface:Qwen/Qwen3-Embedding-8B",
    "huggingface:Salesforce/SFR-Embedding-Code-400M_R",
    "huggingface:Salesforce/codet5p-110m-embedding",
    "huggingface:Snowflake/snowflake-arctic-embed-1-v2.0",
    "huggingface:jinaai/jina-embeddings-v2-base-code",
    "huggingface:jinaai/jina-embeddings-v4-text-code-GGUF",
    "huggingface:mixedbread-ai/mxbai-embed-large-v1",
    "huggingface:nomic-ai/nomic-embed-code",
    "huggingface:nomic-ai/nomic-embed-code-GGUF",
    "huggingface:nomic-ai/nomic-embed-text-v2-moe",
    "huggingface:nvidia/NV-EmbedCode-7b-v1",
    "mistral:codestral-embed",
    "mistral:mistral-embed",
    "ollama:all-minilm",
    "ollama:mxbai-embed-large",
    "ollama:nomic-embed-text",
    "openai:text-embedding-3-large",
    "openai:text-embedding-3-small",
    "together:Alibaba-NLP/gte-modernbert-base",
    "together:BAAI/bge-base-en-v1.5",
    "together:BAAI/bge-large-en-v1.5",
    "together:intfloat/multilingual-e5-large-instruct",
    "vercel:text-embedding-3-large",
    "vercel:text-embedding-3-small",
    "voyage:voyage-3-large",
    "voyage:voyage-3.5",
    "voyage:voyage-3.5-lite",
    "voyage:voyage-code-3",
    "voyage:voyage-context-3",
]

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
