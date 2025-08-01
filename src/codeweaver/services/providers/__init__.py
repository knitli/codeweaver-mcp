# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service providers for CodeWeaver service layer."""

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.services.providers.caching import CacheConfig, CacheEntry, CachingService
from codeweaver.services.providers.chunking import ChunkingService
from codeweaver.services.providers.context_intelligence import (
    ContextAdequacyPredictor as FastMCPContextAdequacyPredictor,
)
from codeweaver.services.providers.context_intelligence import (
    FastMCPContextMiningProvider,
    LLMModelDetector,
)
from codeweaver.services.providers.file_filtering import FilteringService
from codeweaver.services.providers.implicit_learning import (
    BehavioralPatternLearningProvider,
    SatisfactionSignalDetector,
    SessionPatternTracker,
)
from codeweaver.services.providers.intent_orchestrator import IntentOrchestrator
from codeweaver.services.providers.middleware import (
    FastMCPErrorHandlingProvider,
    FastMCPLoggingProvider,
    FastMCPRateLimitingProvider,
    FastMCPTimingProvider,
)
from codeweaver.services.providers.rate_limiting import (
    RateLimitConfig,
    RateLimitingService,
    TokenBucket,
)
from codeweaver.services.providers.telemetry import PostHogTelemetryProvider
from codeweaver.services.providers.zero_shot_optimization import (
    ContextAdequacyOptimizationProvider,
    ContextAdequacyPredictor,
    SuccessPatternDatabase,
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
    "IntentOrchestrator",
    "LLMModelDetector",
    "PostHogTelemetryProvider",
    "RateLimitConfig",
    "RateLimitingService",
    "SatisfactionSignalDetector",
    "SessionPatternTracker",
    "SuccessPatternDatabase",
    "TokenBucket",
]
