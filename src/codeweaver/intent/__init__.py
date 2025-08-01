# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent layer for CodeWeaver."""

from codeweaver.intent.caching import IntentCacheManager
from codeweaver.intent.middleware import IntentServiceBridge
from codeweaver.intent.parsing import BasicConfidenceScorer, IntentParserFactory, PatternBasedParser
from codeweaver.intent.recovery import FallbackChain, IntentErrorHandler
from codeweaver.intent.strategies import (
    AdaptiveStrategy,
    AnalysisWorkflowStrategy,
    SimpleSearchStrategy,
    StrategyRegistry,
)
from codeweaver.intent.workflows import (
    WorkflowDefinition,
    WorkflowOrchestrator,
    WorkflowResult,
    WorkflowStep,
)


__all__ = (
    "AdaptiveStrategy",
    "AnalysisWorkflowStrategy",
    "BasicConfidenceScorer",
    "FallbackChain",
    "IntentCacheManager",
    "IntentErrorHandler",
    "IntentParserFactory",
    "IntentServiceBridge",
    "PatternBasedParser",
    "SimpleSearchStrategy",
    "StrategyRegistry",
    "WorkflowDefinition",
    "WorkflowOrchestrator",
    "WorkflowResult",
    "WorkflowStep",
)
