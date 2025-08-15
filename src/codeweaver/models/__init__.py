# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pydantic models for CodeWeaver."""

# re-export pydantic-ai models for codeweaver

from functools import cache
from typing import Literal

from pydantic_ai.models import (
    DataT,
    DownloadedItem,
    cached_async_http_client,
    download_item,
    infer_model,
    override_allow_model_requests,
)
from pydantic_ai.models import KnownModelName as KnownAgentModelName

from codeweaver.models.core import CodeMatch, FindCodeResponse
from codeweaver.models.intent import IntentResult, QueryIntent


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


@cache
def get_user_agent() -> str:
    """Get the user agent string for CodeWeaver."""
    from codeweaver import __version__

    return f"CodeWeaver/{__version__}"


#  =================================  BEGIN EXAMPLE ======================================
# this is *the* `ModelSettings` object from `pydantic_ai.settings`. We have temporarily copied it here
# So that we can use it as a reference for creating our `EmbeddingModelSettings` and `RerankModelSettings` objects.
# We also re-export this in `codeweaver.settings` as `AgentModelSettings`.

from typing import TypedDict

from httpx import Timeout


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Here we include only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    """

    temperature: float
    """Amount of randomness injected into the response.

    Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
    maximum `temperature` for creative and generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    """

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    You should either alter `temperature` or `top_p`, but not both.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    """

    timeout: float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Mistral
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Groq
    * Anthropic
    """

    seed: int
    """The random seed to use for the model, theoretically allowing for deterministic results.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Mistral
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.
    High settings can lead to odd results, as the model may avoid using punctuation, for example.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI
    * Groq
    """

    stop_sequences: list[str]
    """Sequences that will cause the model to stop generating.

    Supported by:

    * OpenAI
    * Anthropic
    * Bedrock
    * Mistral
    * Groq
    * Cohere
    * Google
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """


#  =================================  END EXAMPLE ======================================


__all__ = [
    "CodeMatch",
    "DataT",
    "DownloadedItem",
    "FindCodeResponse",
    "IntentResult",
    "KnownAgentModelName",
    "QueryIntent",
    "cached_async_http_client",
    "download_item",
    "get_user_agent",
    "infer_model",
    "override_allow_model_requests",
]
