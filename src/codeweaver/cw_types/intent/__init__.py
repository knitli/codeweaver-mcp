# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent types, enums, and exceptions for CodeWeaver."""

from codeweaver.cw_types.intent.base import IntentStrategy
from codeweaver.cw_types.intent.data import IntentResult, ParsedIntent
from codeweaver.cw_types.intent.enums import Complexity, IntentType, Scope
from codeweaver.cw_types.intent.exceptions import (
    IntentError,
    IntentParsingError,
    ServiceIntegrationError,
    StrategyExecutionError,
    StrategySelectionError,
)
from codeweaver.cw_types.intent.learning import (
    BehavioralPatterns,
    ContextAdequacy,
    ContextIntelligenceService,
    ImplicitFeedback,
    ImplicitLearningService,
    LearningSignal,
    LLMProfile,
    OptimizationStrategy,
    OptimizedIntentPlan,
    SatisfactionSignals,
    SessionSignals,
    SuccessPrediction,
    ZeroShotMetrics,
    ZeroShotOptimizationService,
)


__all__ = (
    # Phase 3 implicit learning types
    "BehavioralPatterns",
    # Base intent types
    "Complexity",
    "ContextAdequacy",
    "ContextIntelligenceService",
    "ImplicitFeedback",
    "ImplicitLearningService",
    "IntentError",
    "IntentParsingError",
    "IntentResult",
    "IntentStrategy",
    "IntentType",
    "LLMProfile",
    "LearningSignal",
    "OptimizationStrategy",
    "OptimizedIntentPlan",
    "ParsedIntent",
    "SatisfactionSignals",
    "Scope",
    "ServiceIntegrationError",
    "SessionSignals",
    "StrategyExecutionError",
    "StrategySelectionError",
    "SuccessPrediction",
    "ZeroShotMetrics",
    "ZeroShotOptimizationService",
)
