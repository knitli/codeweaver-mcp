# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Rate limiting and exponential backoff functionality.

Provides intelligent API rate limiting, exponential backoff strategies,
and cool-off periods for resilient handling of API limits and failures.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from functools import wraps

from .config import RateLimitConfig

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Tracks rate limiting state for a specific operation."""
    requests_made: int = 0
    tokens_used: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = 0
    consecutive_failures: int = 0
    next_allowed_time: float = 0


class RateLimiter:
    """Manages rate limiting and exponential backoff for API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.states: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, operation: str, tokens: int = 1) -> bool:
        """Acquire permission to make an API call with rate limiting."""
        async with self._lock:
            state = self.states.get(operation, RateLimitState())
            self.states[operation] = state
            
            current_time = time.time()
            
            # Check if we're in a backoff period
            if current_time < state.next_allowed_time:
                wait_time = state.next_allowed_time - current_time
                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s for {operation}")
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            # Reset window if needed
            window_duration = self._get_window_duration(operation)
            if current_time - state.window_start >= window_duration:
                state.requests_made = 0
                state.tokens_used = 0
                state.window_start = current_time
            
            # Check rate limits
            if not self._check_rate_limits(operation, state, tokens):
                # Calculate wait time until next window
                wait_time = window_duration - (current_time - state.window_start)
                logger.info(f"Rate limit reached for {operation}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                # Reset for new window
                state.requests_made = 0
                state.tokens_used = 0
                state.window_start = time.time()
            
            # Update counters
            state.requests_made += 1
            state.tokens_used += tokens
            state.last_request = current_time
            
            return True
    
    def _get_window_duration(self, operation: str) -> float:
        """Get the rate limiting window duration for an operation."""
        if operation.startswith("voyage"):
            return 60.0  # 1 minute window for Voyage AI
        elif operation.startswith("qdrant"):
            return 1.0   # 1 second window for Qdrant
        else:
            return 60.0  # Default 1 minute
    
    def _check_rate_limits(self, operation: str, state: RateLimitState, tokens: int) -> bool:
        """Check if the operation is within rate limits."""
        if operation.startswith("voyage"):
            # Check Voyage AI limits
            if state.requests_made >= self.config.voyage_requests_per_minute:
                return False
            if state.tokens_used + tokens > self.config.voyage_tokens_per_minute:
                return False
        elif operation.startswith("qdrant"):
            # Check Qdrant limits
            if state.requests_made >= self.config.qdrant_requests_per_second:
                return False
        
        return True
    
    async def record_success(self, operation: str):
        """Record a successful operation to reset failure count."""
        async with self._lock:
            if operation in self.states:
                self.states[operation].consecutive_failures = 0
                logger.debug(f"Rate limiter: success recorded for {operation}")
    
    async def record_failure(self, operation: str, error: Exception):
        """Record a failure and apply exponential backoff."""
        async with self._lock:
            state = self.states.get(operation, RateLimitState())
            self.states[operation] = state
            
            state.consecutive_failures += 1
            
            # Calculate backoff time using exponential backoff
            backoff_time = min(
                self.config.initial_backoff_seconds * (
                    self.config.backoff_multiplier ** (state.consecutive_failures - 1)
                ),
                self.config.max_backoff_seconds
            )
            
            state.next_allowed_time = time.time() + backoff_time
            
            logger.warning(
                f"Rate limiter: failure #{state.consecutive_failures} for {operation}, "
                f"backing off for {backoff_time:.2f}s. Error: {error}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        current_time = time.time()
        stats = {}
        
        for operation, state in self.states.items():
            window_duration = self._get_window_duration(operation)
            time_in_window = current_time - state.window_start
            
            stats[operation] = {
                "requests_made": state.requests_made,
                "tokens_used": state.tokens_used,
                "consecutive_failures": state.consecutive_failures,
                "time_in_window": time_in_window,
                "window_duration": window_duration,
                "next_allowed_time": state.next_allowed_time,
                "currently_backing_off": current_time < state.next_allowed_time
            }
        
        return stats


def rate_limited(operation: str, tokens_func: Optional[Callable] = None):
    """Decorator for applying rate limiting to async functions.
    
    Args:
        operation: The operation name for rate limiting tracking
        tokens_func: Optional function to calculate token usage from function args
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limiter from the first argument (assumed to be self with a rate_limiter attribute)
            if not args or not hasattr(args[0], 'rate_limiter'):
                logger.warning(f"Rate limiting not available for {operation}")
                return await func(*args, **kwargs)
            
            rate_limiter: RateLimiter = args[0].rate_limiter
            
            # Calculate tokens if function provided
            tokens = 1
            if tokens_func:
                try:
                    tokens = tokens_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error calculating tokens for {operation}: {e}")
            
            # Acquire rate limit permission
            await rate_limiter.acquire(operation, tokens)
            
            # Execute function with retry logic
            max_retries = rate_limiter.config.max_retries
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    await rate_limiter.record_success(operation)
                    return result
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed, record failure and re-raise
                        await rate_limiter.record_failure(operation, e)
                        raise
                    else:
                        # Record failure and retry
                        await rate_limiter.record_failure(operation, e)
                        logger.info(f"Retrying {operation} (attempt {attempt + 2}/{max_retries})")
        
        return wrapper
    return decorator


def calculate_embedding_tokens(embedder, texts: list, **kwargs) -> int:
    """Calculate approximate token usage for embedding requests."""
    # Rough approximation: 4 characters per token
    total_chars = sum(len(text) for text in texts)
    return max(1, total_chars // 4)


def calculate_search_tokens(searcher, query: str, **kwargs) -> int:
    """Calculate approximate token usage for search requests."""
    # Search queries are typically small
    return max(1, len(query) // 4)


def calculate_rerank_tokens(reranker, query: str, documents: list, **kwargs) -> int:
    """Calculate approximate token usage for reranking requests."""
    # Reranking considers query + all documents
    total_chars = len(query) + sum(len(doc) for doc in documents)
    return max(1, total_chars // 4)