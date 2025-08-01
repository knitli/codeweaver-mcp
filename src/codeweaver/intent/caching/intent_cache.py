# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent result caching using existing cache services."""

import hashlib
import logging
import time

from typing import Any

from codeweaver.cw_types import IntentResult, ParsedIntent


class IntentCacheManager:
    """
    Intent result caching using existing cache services.

    This manager provides intelligent caching for intent processing results
    with features like:
    - Cache key generation based on intent content and context
    - TTL management with different timeouts for different intent types
    - Cache invalidation strategies
    - Performance metrics and hit rate tracking
    - Integration with existing CodeWeaver caching services

    The manager is designed to work seamlessly with any cache service that
    implements the standard get/set/delete interface.
    """

    def __init__(self, cache_service=None):
        """Initialize intent cache manager.

        Args:
            cache_service: Cache service instance with get/set/delete methods.
                          If None, caching will be disabled.
        """
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)
        self._intent_ttl_config = {"SEARCH": 3600, "UNDERSTAND": 7200, "ANALYZE": 1800}
        self._cache_stats = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }
        self._key_prefix = "intent:"
        self._version = "v1"

    async def get_cached_result(
        self,
        intent_text: str,
        parsed_intent: ParsedIntent | None = None,
        context: dict[str, Any] | None = None,
    ) -> IntentResult | None:
        """
        Get cached result for intent if available.

        Args:
            intent_text: Original intent text
            parsed_intent: Parsed intent structure (for better cache keys)
            context: Intent execution context (affects cache key)

        Returns:
            Cached IntentResult if found and valid, None otherwise
        """
        if not self.cache_service:
            return None
        self._cache_stats["requests"] += 1
        try:
            cache_key = self._generate_cache_key(intent_text, parsed_intent, context)
            self.logger.debug("Checking cache for key: %s", cache_key)
            cached_result = await self.cache_service.get(cache_key)
            if cached_result is None:
                self._cache_stats["misses"] += 1
                self.logger.debug("Cache miss for intent: %s", intent_text[:50])
                return None
            if not self._validate_cached_result(cached_result):
                self.logger.warning("Invalid cached result structure, ignoring")
                await self.cache_service.delete(cache_key)
                self._cache_stats["misses"] += 1
                return None
            self._cache_stats["hits"] += 1
            self.logger.debug("Cache hit for intent: %s", intent_text[:50])
            if isinstance(cached_result, IntentResult):
                cached_result.metadata["from_cache"] = True
                cached_result.metadata["cache_key"] = cache_key
        except Exception as e:
            self._cache_stats["errors"] += 1
            self.logger.warning("Cache get failed: %s", e)
            return None
        else:
            return cached_result

    async def cache_result(
        self,
        intent_text: str,
        result: IntentResult,
        parsed_intent: ParsedIntent | None = None,
        context: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """
        Cache intent result.

        Args:
            intent_text: Original intent text
            result: IntentResult to cache
            parsed_intent: Parsed intent structure (for TTL selection)
            context: Intent execution context
            ttl: Custom TTL in seconds (overrides default)

        Returns:
            True if caching succeeded, False otherwise
        """
        if not self.cache_service:
            return False
        if not result.success:
            self.logger.debug("Not caching failed result")
            return False
        try:
            cache_key = self._generate_cache_key(intent_text, parsed_intent, context)
            effective_ttl = ttl or self._get_ttl_for_intent(parsed_intent)
            cacheable_result = self._prepare_result_for_cache(result)
            self.logger.debug(
                "Caching result for key: %s (TTL: %d seconds)", cache_key, effective_ttl
            )
            await self.cache_service.set(cache_key, cacheable_result, ttl=effective_ttl)
            self._cache_stats["sets"] += 1
        except Exception as e:
            self._cache_stats["errors"] += 1
            self.logger.warning("Cache set failed: %s", e)
            return False
        else:
            return True

    async def invalidate_cache(
        self,
        intent_text: str | None = None,
        parsed_intent: ParsedIntent | None = None,
        context: dict[str, Any] | None = None,
        pattern: str | None = None,
    ) -> int:
        """
        Invalidate cached results.

        Args:
            intent_text: Specific intent text to invalidate
            parsed_intent: Parsed intent to invalidate
            context: Context to invalidate
            pattern: Pattern to match for bulk invalidation

        Returns:
            Number of cache entries invalidated
        """
        if not self.cache_service:
            return 0
        invalidated_count = 0
        try:
            if intent_text or parsed_intent:
                cache_key = self._generate_cache_key(intent_text, parsed_intent, context)
                deleted = await self.cache_service.delete(cache_key)
                if deleted:
                    invalidated_count += 1
                    self._cache_stats["deletes"] += 1
                    self.logger.debug("Invalidated cache key: %s", cache_key)
            elif pattern:
                if hasattr(self.cache_service, "delete_pattern"):
                    invalidated_count = await self.cache_service.delete_pattern(pattern)
                    self._cache_stats["deletes"] += invalidated_count
                    self.logger.info(
                        "Invalidated %d cache entries matching pattern: %s",
                        invalidated_count,
                        pattern,
                    )
                else:
                    self.logger.warning("Pattern-based invalidation not supported by cache service")
        except Exception as e:
            self._cache_stats["errors"] += 1
            self.logger.warning("Cache invalidation failed: %s", e)
            return 0
        else:
            return invalidated_count

    async def clear_intent_cache(self) -> bool:
        """
        Clear all intent cache entries.

        Returns:
            True if clearing succeeded, False otherwise
        """
        if not self.cache_service:
            return False
        try:
            if hasattr(self.cache_service, "delete_pattern"):
                deleted_count = await self.cache_service.delete_pattern(f"{self._key_prefix}*")
                self.logger.info("Cleared %d intent cache entries", deleted_count)
                return True
            if hasattr(self.cache_service, "clear"):
                await self.cache_service.clear()
                self.logger.info("Cleared entire cache (intent cache not isolated)")
                return True
            self.logger.warning("Cache clearing not supported by cache service")
        except Exception as e:
            self._cache_stats["errors"] += 1
            self.logger.warning("Cache clearing failed: %s", e)
            return False
        else:
            return False

    def _generate_cache_key(
        self,
        intent_text: str | None,
        parsed_intent: ParsedIntent | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate cache key for intent."""
        key_components = [self._key_prefix, self._version]
        if intent_text:
            normalized_text = intent_text.lower().strip()
            text_hash = hashlib.md5(normalized_text.encode()).hexdigest()[:16]  # noqa: S324
            key_components.append(text_hash)
        if parsed_intent:
            key_components.extend([
                parsed_intent.intent_type.value,
                parsed_intent.complexity.value,
                parsed_intent.scope.value,
            ])
            if parsed_intent.primary_target:
                target_hash = hashlib.md5(parsed_intent.primary_target.encode()).hexdigest()[:8]  # noqa: S324
                key_components.append(target_hash)
        if context:
            context_elements = []
            context_elements.extend(
                f"{key}={context[key]}"
                for key in ["language", "include_tests", "max_results"]
                if key in context
            )
            if context_elements:
                raw_context = "&".join(sorted(context_elements))
                context_hash = hashlib.md5(raw_context.encode()).hexdigest()[:8]  # noqa: S324
                key_components.append(context_hash)
        return ":".join(key_components)

    def _get_ttl_for_intent(self, parsed_intent: ParsedIntent | None) -> int:
        """Get appropriate TTL for intent type."""
        if not parsed_intent:
            return 3600
        intent_type = parsed_intent.intent_type.value.upper()
        return self._intent_ttl_config.get(intent_type, 3600)

    def _prepare_result_for_cache(self, result: IntentResult) -> IntentResult:
        """Prepare result for caching by removing sensitive data."""
        cached_result = IntentResult(
            success=result.success,
            data=result.data,
            metadata=result.metadata.copy() if result.metadata else {},
            error_message=result.error_message,
            suggestions=result.suggestions,
            executed_at=result.executed_at,
            execution_time=result.execution_time,
            strategy_used=result.strategy_used,
        )
        cached_result.metadata.update({"cached_at": time.time(), "cache_version": self._version})
        if "context" in cached_result.metadata:
            safe_keys = ["session_id", "request_id", "timestamp"]
            safe_context = {
                key: cached_result.metadata["context"][key]
                for key in safe_keys
                if key in cached_result.metadata["context"]
            }
            cached_result.metadata["context"] = safe_context
        return cached_result

    def _validate_cached_result(self, cached_result: Any) -> bool:
        """Validate that cached result has the expected structure."""
        try:
            if not isinstance(cached_result, IntentResult):
                return False
            required_fields = ["success", "data", "metadata"]
            for field in required_fields:
                if not hasattr(cached_result, field):
                    return False
        except Exception as e:
            self.logger.debug("Cache validation failed: %s", e)
            return False
        else:
            return isinstance(cached_result.metadata, dict)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats["requests"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        return {
            "cache_enabled": self.cache_service is not None,
            "requests": total_requests,
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "sets": self._cache_stats["sets"],
            "deletes": self._cache_stats["deletes"],
            "errors": self._cache_stats["errors"],
            "ttl_config": self._intent_ttl_config.copy(),
        }

    def configure_ttl(self, intent_type: str, ttl_seconds: int) -> None:
        """Configure TTL for specific intent type."""
        self._intent_ttl_config[intent_type.upper()] = ttl_seconds
        self.logger.info("Configured TTL for %s: %d seconds", intent_type, ttl_seconds)

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = dict.fromkeys(self._cache_stats, 0)
        self.logger.info("Reset cache statistics")
