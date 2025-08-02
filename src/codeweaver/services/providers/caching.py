# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Caching service provider for CodeWeaver.

Provides caching capabilities for expensive operations like embeddings
with configurable TTL, LRU eviction, and memory management.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import time

from collections import OrderedDict
from typing import Any

from pydantic.dataclasses import dataclass

from codeweaver.cw_types import HealthStatus, ServiceCapabilities, ServiceHealth
from codeweaver.services.providers.base_provider import BaseServiceProvider


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching service."""

    max_size: int = 1000
    default_ttl: int = 3600
    max_memory_mb: int = 100
    cleanup_interval: int = 300


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0

    def __post_init__(self):
        """Initialize access metadata."""
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.created_at + self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class CachingService(BaseServiceProvider):
    """Caching service provider with LRU eviction and TTL support."""

    def __init__(self, config: CacheConfig | None = None):
        """Initialize caching service.

        Args:
            config: Caching configuration
        """
        super().__init__()
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size_bytes = 0
        self._cleanup_task: asyncio.Task | None = None
        logger.info("Initialized caching service")

    async def initialize(self) -> None:
        """Initialize the service."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Caching service initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        async with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
        logger.info("Caching service shutdown")

    def _generate_cache_key(self, key_data: Any) -> str:
        """Generate a cache key from data."""
        if isinstance(key_data, str):
            return key_data
        serialized_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized_json.encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            if isinstance(value, list | tuple):
                if value and isinstance(value[0], int | float):
                    return len(value) * 8
                return len(str(value).encode("utf-8"))
            if isinstance(value, dict):
                return len(json.dumps(value).encode("utf-8"))
        except Exception:
            return 1024
        else:
            return len(str(value).encode("utf-8"))

    async def get(self, key: Any) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key (can be any hashable data)

        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._generate_cache_key(key)
        async with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[cache_key]
                self._total_size_bytes -= entry.size_bytes
                self._misses += 1
                return None
            self._cache.move_to_end(cache_key)
            entry.touch()
            self._hits += 1
            return entry.value

    async def set(self, key: Any, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        cache_key = self._generate_cache_key(key)
        effective_ttl = ttl or self.config.default_ttl
        size_bytes = self._estimate_size(value)
        async with self._lock:
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self._total_size_bytes -= old_entry.size_bytes
                del self._cache[cache_key]
            await self._ensure_capacity(size_bytes)
            entry = CacheEntry(
                value=value, created_at=time.time(), ttl=effective_ttl, size_bytes=size_bytes
            )
            self._cache[cache_key] = entry
            self._total_size_bytes += size_bytes

    async def delete(self, key: Any) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        cache_key = self._generate_cache_key(key)
        async with self._lock:
            if entry := self._cache.pop(cache_key, None):
                self._total_size_bytes -= entry.size_bytes
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0

    async def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        max_size_bytes = self.config.max_memory_mb * 1024 * 1024
        while (
            len(self._cache) >= self.config.max_size
            or self._total_size_bytes + new_size > max_size_bytes
        ) and self._cache:
            _oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._total_size_bytes -= oldest_entry.size_bytes
            self._evictions += 1

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cache cleanup: %s")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = []
            expired_keys.extend(key for key, entry in self._cache.items() if entry.is_expired())
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._total_size_bytes -= entry.size_bytes
            if expired_keys:
                logger.debug("Cleaned up %s expired cache entries", len(expired_keys))

    def get_statistics(self) -> dict[str, Any]:
        """Get caching statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "total_entries": len(self._cache),
            "total_size_bytes": self._total_size_bytes,
            "total_size_mb": self._total_size_bytes / (1024 * 1024),
            "max_size": self.config.max_size,
            "max_memory_mb": self.config.max_memory_mb,
        }

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            stats = self.get_statistics()
            memory_usage_pct = stats["total_size_mb"] / self.config.max_memory_mb * 100
            if memory_usage_pct > 95:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {memory_usage_pct:.1f}%",
                )
            if self._cleanup_task and self._cleanup_task.done():
                return ServiceHealth(status=HealthStatus.DEGRADED, message="Cleanup task stopped")
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY, message=f"Cache health check failed: {e}"
            )
        else:
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                message=f"Cache active: {stats['total_entries']} entries, {stats['hit_rate']:.2f} hit rate",
            )

    def get_capabilities(self) -> ServiceCapabilities:
        """Get service capabilities."""
        return ServiceCapabilities(
            supports_streaming=False,
            supports_batching=True,
            max_batch_size=self.config.max_size,
            supports_async=True,
        )

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for caching operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "caching_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.get_capabilities(),
            "configuration": {
                "max_size": self.config.max_size,
                "max_memory_mb": self.config.max_memory_mb,
                "cleanup_interval": self.config.cleanup_interval,
                "default_ttl": self.config.default_ttl,
            },
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.message if health.status.name != "HEALTHY" else None,
        })

        # Add runtime statistics
        context.update({"statistics": self.get_statistics()})

        return context
