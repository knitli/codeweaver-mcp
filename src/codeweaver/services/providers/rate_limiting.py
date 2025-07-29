# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Rate limiting service provider for CodeWeaver.

Provides rate limiting capabilities for API calls and resource usage
with configurable limits, token bucket algorithm, and provider-specific rules.
"""

import asyncio
import logging
import time

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import HealthStatus, ServiceCapabilities, ServiceHealth


logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_capacity: int = 20
    provider_specific_limits: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time needed for tokens to be available."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitingService(BaseServiceProvider):
    """Rate limiting service provider."""

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiting service.

        Args:
            config: Rate limiting configuration
        """
        super().__init__()
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, TokenBucket] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._total_requests = 0
        self._blocked_requests = 0
        self._total_wait_time = 0.0
        logger.info("Initialized rate limiting service")

    async def initialize(self) -> None:
        """Initialize the service."""
        self._create_bucket("default", self.config.requests_per_second, self.config.burst_capacity)
        for provider, limits in self.config.provider_specific_limits.items():
            rps = limits.get("requests_per_second", self.config.requests_per_second)
            burst = limits.get("burst_capacity", self.config.burst_capacity)
            self._create_bucket(provider, rps, int(burst))
        logger.info("Rate limiting service initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._buckets.clear()
        self._locks.clear()
        logger.info("Rate limiting service shutdown")

    def _create_bucket(self, key: str, requests_per_second: float, burst_capacity: int) -> None:
        """Create a token bucket for a specific key."""
        self._buckets[key] = TokenBucket(
            capacity=burst_capacity, tokens=float(burst_capacity), refill_rate=requests_per_second
        )

    async def acquire(self, provider: str, tokens: int = 1) -> None:
        """Acquire tokens for rate limiting.

        Args:
            provider: Provider name (e.g., 'voyage_ai', 'openai')
            tokens: Number of tokens to acquire

        Raises:
            asyncio.TimeoutError: If rate limit exceeded for too long
        """
        bucket_key = provider if provider in self._buckets else "default"
        async with self._locks[bucket_key]:
            bucket = self._buckets[bucket_key]
            self._total_requests += 1
            if bucket.consume(tokens):
                return
            wait_time = bucket.wait_time(tokens)
            if wait_time > 0:
                self._blocked_requests += 1
                self._total_wait_time += wait_time
                logger.debug(
                    "Rate limiting %s: waiting %ss for %s tokens", provider, wait_time, tokens
                )
                await asyncio.sleep(wait_time)
                if not bucket.consume(tokens):
                    logger.warning("Failed to acquire tokens for %s after waiting", provider)
                    raise TimeoutError(f"Rate limit exceeded for {provider}")

    async def check_availability(self, provider: str, tokens: int = 1) -> bool:
        """Check if tokens are available without consuming them.

        Args:
            provider: Provider name
            tokens: Number of tokens to check

        Returns:
            True if tokens are available
        """
        bucket_key = provider if provider in self._buckets else "default"
        bucket = self._buckets[bucket_key]
        bucket._refill()
        return bucket.tokens >= tokens

    async def get_wait_time(self, provider: str, tokens: int = 1) -> float:
        """Get estimated wait time for tokens.

        Args:
            provider: Provider name
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        bucket_key = provider if provider in self._buckets else "default"
        bucket = self._buckets[bucket_key]
        return bucket.wait_time(tokens)

    def get_statistics(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "block_rate": self._blocked_requests / max(1, self._total_requests),
            "total_wait_time": self._total_wait_time,
            "average_wait_time": self._total_wait_time / max(1, self._blocked_requests),
            "active_buckets": list(self._buckets.keys()),
        }

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            bucket_count = len(self._buckets)
            if bucket_count == 0:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY, message="No rate limiting buckets configured"
                )
            depleted_buckets = []
            for key, bucket in self._buckets.items():
                bucket._refill()
                if bucket.tokens < bucket.capacity * 0.1:
                    depleted_buckets.append(key)
            if len(depleted_buckets) > len(self._buckets) * 0.5:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"Many buckets depleted: {depleted_buckets}",
                )
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                message=f"Rate limiting active with {bucket_count} buckets",
            )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY, message=f"Rate limiting health check failed: {e}"
            )

    def get_capabilities(self) -> ServiceCapabilities:
        """Get service capabilities."""
        return ServiceCapabilities(
            supports_streaming=False,
            supports_batching=True,
            max_batch_size=1000,
            supports_async=True,
        )
