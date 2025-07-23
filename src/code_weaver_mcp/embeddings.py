# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Embedding and reranking functionality using Voyage AI.

Provides semantic embeddings for code chunks and query reranking
using Voyage AI's specialized code models with rate limiting and backoff.
"""

import logging
from typing import Any, Dict, List, Optional

import voyageai

from .rate_limiter import RateLimiter, rate_limited, calculate_embedding_tokens, calculate_rerank_tokens

logger = logging.getLogger(__name__)


class VoyageAIEmbedder:
    """Handles Voyage AI embeddings for code with rate limiting."""
    
    def __init__(self, api_key: str, model: str = "voyage-code-3", rate_limiter: Optional[RateLimiter] = None):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = 1024  # Default dimension for voyage-code-3
        self.rate_limiter = rate_limiter
    
    @rate_limited("voyage_embed_documents", calculate_embedding_tokens)
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents (code chunks) with rate limiting."""
        try:
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="document",
                output_dimension=self.dimension
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @rate_limited("voyage_embed_query", lambda self, text, **kwargs: max(1, len(text) // 4))
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for search query with rate limiting."""
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="query",
                output_dimension=self.dimension
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise


class VoyageAIReranker:
    """Handles Voyage AI reranking with rate limiting."""
    
    def __init__(self, api_key: str, model: str = "voyage-rerank-2", rate_limiter: Optional[RateLimiter] = None):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.rate_limiter = rate_limiter
    
    @rate_limited("voyage_rerank", calculate_rerank_tokens)
    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents for a query with rate limiting."""
        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k
            )
            return result.results
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            raise