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

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from codeweaver.config import RateLimitConfig


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
        """Initialize the rate limiter with configuration settings.

        Args:
            config: Rate limiting configuration including timeouts and backoff settings
        """
        self.config = config
        self.states: dict[str, RateLimitState] = {}
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
                logger.debug("Rate limiter: waiting %.2fs for %s", wait_time, operation)
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
                logger.info("Rate limit reached for %s, waiting %.2fs", operation, wait_time)
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
        if operation.startswith("voyage-ai"):
            return 60.0  # 1 minute window for Voyage AI
        if operation.startswith("openai"):
            return 60.0  # 1 minute window for OpenAI
        return 1.0 if operation.startswith("qdrant") else 60.0

    def _check_rate_limits(self, operation: str, state: RateLimitState, tokens: int) -> bool:
        """Check if the operation is within rate limits."""
        if operation.startswith("voyage-ai"):
            # Check Voyage AI limits
            if state.requests_made >= self.config.voyage_requests_per_minute:
                return False
            if state.tokens_used + tokens > self.config.voyage_tokens_per_minute:
                return False
        elif operation.startswith("openai"):
            # Check OpenAI limits
            if state.requests_made >= self.config.openai_requests_per_minute:
                return False
            if state.tokens_used + tokens > self.config.openai_tokens_per_minute:
                return False
        elif operation.startswith("qdrant"):
            # Check Qdrant limits
            if state.requests_made >= self.config.qdrant_requests_per_second:
                return False

        return True

    async def record_success(self, operation: str) -> None:
        """Record a successful operation to reset failure count."""
        async with self._lock:
            if operation in self.states:
                self.states[operation].consecutive_failures = 0
                logger.debug("Rate limiter: success recorded for %s", operation)

    async def record_failure(self, operation: str, error: Exception) -> None:
        """Record a failure and apply exponential backoff."""
        async with self._lock:
            state = self.states.get(operation, RateLimitState())
            self.states[operation] = state

            state.consecutive_failures += 1

            # Calculate backoff time using exponential backoff
            backoff_time = min(
                self.config.initial_backoff_seconds
                * (self.config.backoff_multiplier ** (state.consecutive_failures - 1)),
                self.config.max_backoff_seconds,
            )

            state.next_allowed_time = time.time() + backoff_time

            logger.warning(
                "Rate limiter: failure #%d for %s, backing off for %.2fs. Error: %s",
                state.consecutive_failures,
                operation,
                backoff_time,
                error,
            )

    def get_stats(self) -> dict[str, Any]:
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
                "currently_backing_off": current_time < state.next_allowed_time,
            }

        return stats


def rate_limited(operation: str, tokens_func: Callable | None = None) -> Callable:
    """Decorator for applying rate limiting to async functions.

    Args:
        operation: The operation name for rate limiting tracking
        tokens_func: Optional function to calculate token usage from function args
    """

    def decorator(func: Callable) -> Callable:
        """Decorator to apply rate limiting and retry logic to async functions."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wraps an async function to apply rate limiting and retry logic.

            This wrapper checks for a rate_limiter attribute, calculates token usage,
            acquires rate limit permission, and retries the function on failure up to
            the configured maximum retries.

            Args:
                *args: Positional arguments for the wrapped function.
                **kwargs: Keyword arguments for the wrapped function.

            Returns:
                The result of the wrapped async function, or None if all retries fail.

            Raises:
                Exception: Re-raises the last exception if all retries are exhausted.
            """
            # Get rate limiter from the first argument (assumed to be self with a rate_limiter attribute)
            if not args or not hasattr(args[0], "rate_limiter"):
                logger.warning("Rate limiting not available for %s", operation)
                return await func(*args, **kwargs)

            rate_limiter: RateLimiter = args[0].rate_limiter

            # Calculate tokens if function provided
            tokens = 1
            if tokens_func:
                try:
                    tokens = tokens_func(*args, **kwargs)
                except Exception as e:
                    logger.warning("Error calculating tokens for %s: %s", operation, e)

            # Acquire rate limit permission
            await rate_limiter.acquire(operation, tokens)

            # Execute function with retry logic
            max_retries = rate_limiter.config.max_retries
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed, record failure and re-raise
                        await rate_limiter.record_failure(operation, e)
                        raise
                    # Record failure and retry
                    await rate_limiter.record_failure(operation, e)
                    logger.info("Retrying %s (attempt %d/%d)", operation, attempt + 2, max_retries)
                else:
                    await rate_limiter.record_success(operation)
                    return result
            return None

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
