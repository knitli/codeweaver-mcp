# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Data structures for Phase 3 implicit learning and zero-shot optimization."""

import hashlib

from datetime import UTC, datetime
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from codeweaver.cw_types.intent.data import IntentResult


@runtime_checkable
class ImplicitLearningService(Protocol):
    """Protocol for implicit learning service providers."""

    async def analyze_interaction_outcome(
        self,
        ctx: Any,  # FastMCPContext
        intent: str,
        result: IntentResult,
        response_metadata: dict[str, Any],
    ) -> "ImplicitFeedback":
        """Analyze interaction for implicit satisfaction signals."""
        ...

    async def extract_behavioral_patterns(
        self, session_data: dict[str, Any]
    ) -> "BehavioralPatterns":
        """Extract behavioral patterns from session data."""
        ...


@runtime_checkable
class ContextIntelligenceService(Protocol):
    """Protocol for context intelligence service providers."""

    async def extract_llm_characteristics(self, ctx: Any) -> "LLMProfile":
        """Extract LLM behavioral characteristics from context."""
        ...

    async def analyze_context_adequacy(
        self, intent: str, available_context: dict[str, Any]
    ) -> "ContextAdequacy":
        """Analyze adequacy of available context for intent processing."""
        ...


@runtime_checkable
class ZeroShotOptimizationService(Protocol):
    """Protocol for zero-shot optimization service providers."""

    async def optimize_for_zero_shot_success(
        self, ctx: Any, intent: str, available_context: dict[str, Any]
    ) -> "OptimizedIntentPlan":
        """Optimize intent processing for first-attempt success."""
        ...

    async def predict_success_probability(
        self, intent: str, context: dict[str, Any]
    ) -> "SuccessPrediction":
        """Predict likelihood of zero-shot success."""
        ...


class SessionSignals(BaseModel):
    """Session-level behavioral signals for implicit learning."""

    session_id: Annotated[str, Field(description="Session identifier hash")]
    user_agent: Annotated[str | None, Field(description="User agent string", default=None)]
    request_timing: Annotated[
        dict[str, float], Field(description="Request timing data", default_factory=dict)
    ]
    interaction_count: Annotated[
        int, Field(description="Number of interactions in session", default=1)
    ]
    follow_up_queries: Annotated[
        list[str], Field(description="Follow-up query patterns", default_factory=list)
    ]
    response_engagement: Annotated[
        float, Field(description="Response engagement score", default=0.0)
    ]


class SatisfactionSignals(BaseModel):
    """Satisfaction signals detected from behavioral patterns."""

    response_time_score: Annotated[
        float, Field(description="Response time satisfaction score", ge=0.0, le=1.0)
    ]
    result_quality_score: Annotated[
        float, Field(description="Result quality score", ge=0.0, le=1.0)
    ]
    follow_up_pattern_score: Annotated[
        float, Field(description="Follow-up pattern score", ge=0.0, le=1.0)
    ]
    session_completion_score: Annotated[
        float, Field(description="Session completion score", ge=0.0, le=1.0)
    ]
    overall_satisfaction: Annotated[
        float, Field(description="Overall satisfaction probability", ge=0.0, le=1.0)
    ]


class ImplicitFeedback(BaseModel):
    """Implicit feedback extracted from LLM behavioral patterns."""

    satisfaction_probability: Annotated[
        float, Field(description="Satisfaction probability", ge=0.0, le=1.0)
    ]
    improvement_signals: Annotated[
        dict[str, Any], Field(description="Areas for improvement", default_factory=dict)
    ]
    learning_weight: Annotated[
        float, Field(description="Weight for learning updates", ge=0.0, le=1.0)
    ]
    confidence: Annotated[
        float, Field(description="Confidence in feedback signals", ge=0.0, le=1.0)
    ]
    session_signals: Annotated[SessionSignals, Field(description="Session-level signals")]
    satisfaction_signals: Annotated[
        SatisfactionSignals, Field(description="Detailed satisfaction signals")
    ]
    created_at: Annotated[
        datetime,
        Field(description="When feedback was generated", default_factory=lambda: datetime.now(UTC)),
    ]


class BehavioralPatterns(BaseModel):
    """Behavioral patterns extracted from multiple sessions."""

    pattern_id: Annotated[str, Field(description="Unique pattern identifier")]
    pattern_type: Annotated[str, Field(description="Type of behavioral pattern")]
    frequency: Annotated[int, Field(description="Pattern occurrence frequency", ge=1)]
    confidence: Annotated[float, Field(description="Pattern confidence score", ge=0.0, le=1.0)]
    context_features: Annotated[
        dict[str, Any], Field(description="Context features for pattern", default_factory=dict)
    ]
    success_correlation: Annotated[
        float, Field(description="Correlation with success", ge=-1.0, le=1.0)
    ]
    identified_at: Annotated[
        datetime,
        Field(description="When pattern was identified", default_factory=lambda: datetime.now(UTC)),
    ]


class LLMProfile(BaseModel):
    """LLM behavioral profile extracted from context."""

    session_id: Annotated[str, Field(description="Session identifier")]
    identified_model: Annotated[str | None, Field(description="Identified LLM model", default=None)]
    confidence: Annotated[float, Field(description="Identification confidence", ge=0.0, le=1.0)]
    request_patterns: Annotated[
        list[str], Field(description="Request pattern signatures", default_factory=list)
    ]
    timing_characteristics: Annotated[
        dict[str, float], Field(description="Timing characteristics", default_factory=dict)
    ]
    behavioral_features: Annotated[
        dict[str, Any], Field(description="Behavioral features", default_factory=dict)
    ]
    created_at: Annotated[
        datetime,
        Field(description="Profile creation time", default_factory=lambda: datetime.now(UTC)),
    ]


class ContextAdequacy(BaseModel):
    """Assessment of context adequacy for intent processing."""

    score: Annotated[float, Field(description="Context adequacy score", ge=0.0, le=1.0)]
    missing_elements: Annotated[
        list[str], Field(description="Missing context elements", default_factory=list)
    ]
    richness_score: Annotated[float, Field(description="Context richness score", ge=0.0, le=1.0)]
    clarity_score: Annotated[float, Field(description="Context clarity score", ge=0.0, le=1.0)]
    recommendations: Annotated[
        list[str], Field(description="Context improvement recommendations", default_factory=list)
    ]
    assessed_at: Annotated[
        datetime,
        Field(description="Assessment timestamp", default_factory=lambda: datetime.now(UTC)),
    ]


class SuccessPrediction(BaseModel):
    """Prediction of zero-shot success probability."""

    probability: Annotated[float, Field(description="Success probability", ge=0.0, le=1.0)]
    context_adequacy: Annotated[float, Field(description="Context adequacy score", ge=0.0, le=1.0)]
    strategy_confidence: Annotated[float, Field(description="Strategy confidence", ge=0.0, le=1.0)]
    missing_context: Annotated[
        list[str], Field(description="Missing context elements", default_factory=list)
    ]
    recommended_strategies: Annotated[
        list[str], Field(description="Recommended strategies", default_factory=list)
    ]
    risk_factors: Annotated[
        list[str], Field(description="Identified risk factors", default_factory=list)
    ]
    predicted_at: Annotated[
        datetime,
        Field(description="Prediction timestamp", default_factory=lambda: datetime.now(UTC)),
    ]


class OptimizationStrategy(BaseModel):
    """Individual optimization strategy."""

    strategy_type: Annotated[str, Field(description="Type of optimization strategy")]
    description: Annotated[str, Field(description="Strategy description")]
    priority: Annotated[int, Field(description="Strategy priority", ge=1)]
    expected_improvement: Annotated[
        float, Field(description="Expected improvement score", ge=0.0, le=1.0)
    ]
    implementation_cost: Annotated[
        float, Field(description="Implementation cost score", ge=0.0, le=1.0)
    ]
    metadata: Annotated[
        dict[str, Any], Field(description="Strategy-specific metadata", default_factory=dict)
    ]


class OptimizedIntentPlan(BaseModel):
    """Optimized intent processing plan for zero-shot success."""

    original_intent: Annotated[str, Field(description="Original intent text")]
    success_probability: Annotated[
        float, Field(description="Predicted success probability", ge=0.0, le=1.0)
    ]
    optimizations: Annotated[
        list[OptimizationStrategy], Field(description="Applied optimizations", default_factory=list)
    ]
    enhanced_context: Annotated[
        dict[str, Any], Field(description="Enhanced context data", default_factory=dict)
    ]
    recommended_strategy: Annotated[
        str | None, Field(description="Recommended processing strategy", default=None)
    ]
    fallback_strategies: Annotated[
        list[str], Field(description="Fallback strategies", default_factory=list)
    ]
    optimization_metadata: Annotated[
        dict[str, Any], Field(description="Optimization metadata", default_factory=dict)
    ]
    created_at: Annotated[
        datetime, Field(description="Plan creation time", default_factory=lambda: datetime.now(UTC))
    ]


class LearningSignal(BaseModel):
    """Learning signal for telemetry integration."""

    intent_hash: Annotated[str, Field(description="Hashed intent identifier")]
    satisfaction_score: Annotated[float, Field(description="Satisfaction score", ge=0.0, le=1.0)]
    response_time: Annotated[
        float | None, Field(description="Response time in seconds", default=None)
    ]
    strategy_used: Annotated[str | None, Field(description="Strategy used", default=None)]
    success: Annotated[bool, Field(description="Whether operation succeeded")]
    learning_weight: Annotated[float, Field(description="Weight for learning", ge=0.0, le=1.0)]
    context_hash: Annotated[str | None, Field(description="Context hash for privacy", default=None)]
    created_at: Annotated[
        datetime,
        Field(description="Signal creation time", default_factory=lambda: datetime.now(UTC)),
    ]

    @classmethod
    def create_from_feedback(
        cls, feedback: ImplicitFeedback, intent: str, strategy: str | None = None
    ) -> "LearningSignal":
        """Create learning signal from implicit feedback."""
        intent_hash = hashlib.sha256(intent.encode()).hexdigest()[:16]
        return cls(
            intent_hash=intent_hash,
            satisfaction_score=feedback.satisfaction_probability,
            strategy_used=strategy,
            success=feedback.satisfaction_probability > 0.7,  # Threshold for success
            learning_weight=feedback.learning_weight,
        )


class ZeroShotMetrics(BaseModel):
    """Metrics for zero-shot optimization effectiveness."""

    optimization_type: Annotated[str, Field(description="Type of optimization")]
    before_metrics: Annotated[dict[str, float], Field(description="Metrics before optimization")]
    after_metrics: Annotated[dict[str, float], Field(description="Metrics after optimization")]
    improvement_score: Annotated[
        float, Field(description="Overall improvement score", ge=-1.0, le=1.0)
    ]
    metrics_delta: Annotated[dict[str, float], Field(description="Change in metrics")]
    measured_at: Annotated[
        datetime,
        Field(description="Measurement timestamp", default_factory=lambda: datetime.now(UTC)),
    ]
