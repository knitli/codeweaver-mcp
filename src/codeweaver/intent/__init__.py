# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent layer for CodeWeaver."""

from codeweaver.intent.middleware import IntentServiceBridge
from codeweaver.intent.parsing import BasicConfidenceScorer, IntentParserFactory, PatternBasedParser
from codeweaver.intent.strategies import (
    AdaptiveStrategy,
    AnalysisWorkflowStrategy,
    SimpleSearchStrategy,
    StrategyRegistry,
)


__all__ = [
    "AdaptiveStrategy",
    "AnalysisWorkflowStrategy",
    "BasicConfidenceScorer",
    "IntentParserFactory",
    "IntentServiceBridge",
    "PatternBasedParser",
    "SimpleSearchStrategy",
    "StrategyRegistry",
]
