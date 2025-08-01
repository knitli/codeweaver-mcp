# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service layer for CodeWeaver - connecting middleware with factory patterns."""

from codeweaver.services.manager import ServicesManager
from codeweaver.services.middleware_bridge import ServiceBridge, ServiceCoordinator
from codeweaver.services.providers import (
    BaseServiceProvider,
    BehavioralPatternLearningProvider,
    CacheConfig,
    CacheEntry,
    CachingService,
    ChunkingService,
    ContextAdequacyOptimizationProvider,
    ContextAdequacyPredictor,
    FastMCPContextAdequacyPredictor,
    FastMCPContextMiningProvider,
    FastMCPErrorHandlingProvider,
    FastMCPLoggingProvider,
    FastMCPRateLimitingProvider,
    FastMCPTimingProvider,
    FilteringService,
    LLMModelDetector,
    RateLimitConfig,
    RateLimitingService,
    SatisfactionSignalDetector,
    SessionPatternTracker,
    SuccessPatternDatabase,
    TokenBucket,
)


__all__ = [
    "BaseServiceProvider",
    "BehavioralPatternLearningProvider",
    "CacheConfig",
    "CacheEntry",
    "CachingService",
    "ChunkingService",
    "ContextAdequacyOptimizationProvider",
    "ContextAdequacyPredictor",
    "FastMCPContextAdequacyPredictor",
    "FastMCPContextMiningProvider",
    "FastMCPErrorHandlingProvider",
    "FastMCPLoggingProvider",
    "FastMCPRateLimitingProvider",
    "FastMCPTimingProvider",
    "FilteringService",
    "LLMModelDetector",
    "RateLimitConfig",
    "RateLimitingService",
    "SatisfactionSignalDetector",
    "ServiceBridge",
    "ServiceCoordinator",
    "ServicesManager",
    "SessionPatternTracker",
    "SuccessPatternDatabase",
    "TokenBucket",
]
