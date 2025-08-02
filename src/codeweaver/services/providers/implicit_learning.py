# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Implicit learning service provider for behavioral pattern analysis."""

import hashlib
import logging
import uuid

from datetime import UTC, datetime
from typing import Any

from codeweaver.cw_types import (
    BehavioralPatterns,
    ImplicitFeedback,
    ImplicitLearningService,
    ImplicitLearningServiceConfig,
    IntentResult,
    LearningSignal,
    SatisfactionSignals,
    ServiceType,
    SessionSignals,
)
from codeweaver.services.providers.base_provider import BaseServiceProvider


class SessionPatternTracker:
    """Tracks behavioral patterns within and across sessions."""

    def __init__(self):
        """Initialize the session pattern tracker."""
        self._session_data: dict[str, dict[str, Any]] = {}
        self._patterns: dict[str, BehavioralPatterns] = {}
        self._pattern_frequencies: dict[str, int] = {}

    async def initialize(self) -> None:
        """Initialize the tracker."""
        # In a production system, this would load patterns from persistent storage

    async def track_session_interaction(
        self, session_id: str, intent: str, result: IntentResult, context: dict[str, Any]
    ) -> SessionSignals:
        """Track an interaction within a session."""
        if session_id not in self._session_data:
            self._session_data[session_id] = {
                "interactions": [],
                "start_time": datetime.now(UTC),
                "last_activity": datetime.now(UTC),
            }

        session = self._session_data[session_id]
        session["last_activity"] = datetime.now(UTC)

        interaction = {
            "intent": intent,
            "result": result,
            "context": context,
            "timestamp": datetime.now(UTC),
        }
        session["interactions"].append(interaction)

        # Extract behavioral signals
        return SessionSignals(
            session_id=session_id,
            user_agent=context.get("user_agent"),
            request_timing=context.get("timing", {}),
            interaction_count=len(session["interactions"]),
            follow_up_queries=self._extract_follow_up_patterns(session["interactions"]),
            response_engagement=self._calculate_engagement_score(session["interactions"]),
        )

    async def identify_patterns(self, min_frequency: int = 3) -> list[BehavioralPatterns]:
        """Identify behavioral patterns from tracked sessions."""
        patterns = []

        patterns.extend(
            self._patterns[pattern_id]
            for pattern_id, frequency in self._pattern_frequencies.items()
            if frequency >= min_frequency and pattern_id in self._patterns
        )
        return patterns

    def _extract_follow_up_patterns(self, interactions: list[dict[str, Any]]) -> list[str]:
        """Extract follow-up query patterns from interactions."""
        patterns = []
        for i in range(1, len(interactions)):
            prev_intent = interactions[i - 1]["intent"][:20]  # First 20 chars
            curr_intent = interactions[i]["intent"][:20]
            pattern = f"{prev_intent} -> {curr_intent}"
            patterns.append(pattern)
        return patterns

    def _calculate_engagement_score(self, interactions: list[dict[str, Any]]) -> float:
        """Calculate engagement score based on interaction patterns."""
        if not interactions:
            return 0.0

        # Simple engagement scoring based on:
        # - Number of interactions (more = higher engagement)
        # - Time between interactions (reasonable gaps = higher engagement)
        # - Success rate (higher success = higher engagement)

        interaction_score = min(len(interactions) / 10.0, 1.0)  # Max score at 10 interactions

        # Calculate success rate
        successful_interactions = sum(
            bool(interaction["result"].success) for interaction in interactions
        )
        success_rate = successful_interactions / len(interactions)

        # Combine scores
        return (interaction_score * 0.3) + (success_rate * 0.7)


class SatisfactionSignalDetector:
    """Detects satisfaction signals from behavioral patterns."""

    def __init__(self):
        """Initialize the satisfaction signal detector."""
        self._response_time_thresholds = {
            "excellent": 0.5,  # < 500ms
            "good": 2.0,  # < 2s
            "acceptable": 5.0,  # < 5s
            "poor": float("inf"),
        }

    async def initialize(self) -> None:
        """Initialize the detector."""

    async def calculate_satisfaction(
        self,
        intent: str,
        result: IntentResult,
        response_time: float,
        session_signals: SessionSignals,
    ) -> float:
        """Calculate satisfaction probability from various signals."""
        # Response time satisfaction
        response_time_score = self._score_response_time(response_time)

        # Result quality satisfaction (based on success and result data)
        result_quality_score = self._score_result_quality(result)

        # Follow-up pattern satisfaction (fewer follow-ups = higher satisfaction)
        follow_up_score = self._score_follow_up_patterns(session_signals.follow_up_queries)

        # Session completion satisfaction (engagement indicates satisfaction)
        session_completion_score = session_signals.response_engagement

        # Create detailed satisfaction signals
        satisfaction_signals = SatisfactionSignals(
            response_time_score=response_time_score,
            result_quality_score=result_quality_score,
            follow_up_pattern_score=follow_up_score,
            session_completion_score=session_completion_score,
            overall_satisfaction=0.0,  # Will be calculated below
        )

        # Weighted combination
        overall_satisfaction = (
            response_time_score * 0.25
            + result_quality_score * 0.35
            + follow_up_score * 0.25
            + session_completion_score * 0.15
        )

        satisfaction_signals.overall_satisfaction = overall_satisfaction
        return overall_satisfaction

    def _score_response_time(self, response_time: float) -> float:
        """Score satisfaction based on response time."""
        if response_time <= self._response_time_thresholds["excellent"]:
            return 1.0
        if response_time <= self._response_time_thresholds["good"]:
            return 0.8
        if response_time <= self._response_time_thresholds["acceptable"]:
            return 0.6
        return 0.3

    def _score_result_quality(self, result: IntentResult) -> float:
        """Score satisfaction based on result quality."""
        if not result.success:
            return 0.2

        # Check if result has meaningful data
        if not result.data:
            return 0.4

        # For search results, check result count
        if isinstance(result.data, list):
            result_count = len(result.data)
            if result_count == 0:
                return 0.3
            if result_count <= 5:
                return 0.7
            return 0.9 if result_count <= 20 else 0.8
        return 0.8  # Default for successful results with data

    def _score_follow_up_patterns(self, follow_up_queries: list[str]) -> float:
        """Score satisfaction based on follow-up patterns."""
        if not follow_up_queries:
            return 1.0  # No follow-ups = likely satisfied

        # Analyze follow-up patterns
        refinement_patterns = ["more", "another", "different", "better", "explain", "show", "find"]

        refinement_count = sum(
            any(pattern in query.lower() for pattern in refinement_patterns)
            for query in follow_up_queries
        )

        if refinement_count == 0:
            return 0.9  # Follow-ups but not refinements
        return 0.6 if refinement_count <= 2 else 0.3


class BehavioralPatternLearningProvider(BaseServiceProvider, ImplicitLearningService):
    """Implicit learning service provider using behavioral pattern analysis."""

    def __init__(
        self,
        service_type: ServiceType,
        config: ImplicitLearningServiceConfig,
        logger: logging.Logger | None = None,
    ):
        """Initialize the behavioral pattern learning provider."""
        super().__init__(service_type, config, logger)
        self._config: ImplicitLearningServiceConfig = config

        # Core components
        self.session_tracker = SessionPatternTracker()
        self.satisfaction_detector = SatisfactionSignalDetector()

        # Learning state
        self._learning_signals: list[LearningSignal] = []
        self._pattern_database: dict[str, BehavioralPatterns] = {}

        # Telemetry service will be injected by services manager
        self.telemetry_service = None

    async def _initialize_provider(self) -> None:
        """Initialize the behavioral pattern learning provider."""
        await self.session_tracker.initialize()
        await self.satisfaction_detector.initialize()

        self._logger.info(
            "Behavioral pattern learning provider initialized with config: %s",
            {
                "satisfaction_threshold": self._config.satisfaction_threshold,
                "learning_rate": self._config.learning_rate,
                "pattern_retention_days": self._config.pattern_retention_days,
            },
        )

    async def _shutdown_provider(self) -> None:
        """Shutdown the behavioral pattern learning provider."""
        # In production, save patterns to persistent storage
        self._logger.info(
            "Behavioral pattern learning provider shutdown. Learned %d patterns.",
            len(self._pattern_database),
        )

    async def _check_health(self) -> bool:
        """Check if the learning service is healthy."""
        # Check if we're collecting learning signals and patterns are being updated
        recent_signals = [
            signal
            for signal in self._learning_signals
            if (datetime.now(UTC) - signal.created_at).total_seconds() < 3600  # Last hour
        ]

        # Service is healthy if we have recent activity or if we're just starting
        return len(recent_signals) > 0 or len(self._learning_signals) < 10

    async def analyze_interaction_outcome(
        self,
        ctx: Any,  # FastMCPContext
        intent: str,
        result: IntentResult,
        response_metadata: dict[str, Any],
    ) -> ImplicitFeedback:
        """Analyze interaction for implicit satisfaction signals."""
        # Extract session information from context
        session_signals = await self._extract_session_signals(
            ctx, intent, result, response_metadata
        )

        # Analyze response timing and follow-up patterns
        satisfaction_probability = await self.satisfaction_detector.calculate_satisfaction(
            intent=intent,
            result=result,
            response_time=response_metadata.get("response_time", 0.0),
            session_signals=session_signals,
        )

        # Create satisfaction signals
        satisfaction_signals = SatisfactionSignals(
            response_time_score=self.satisfaction_detector._score_response_time(
                response_metadata.get("response_time", 0.0)
            ),
            result_quality_score=self.satisfaction_detector._score_result_quality(result),
            follow_up_pattern_score=self.satisfaction_detector._score_follow_up_patterns(
                session_signals.follow_up_queries
            ),
            session_completion_score=session_signals.response_engagement,
            overall_satisfaction=satisfaction_probability,
        )

        # Calculate learning weight based on confidence and novelty
        learning_weight = self._calculate_learning_weight(satisfaction_probability, session_signals)

        # Extract improvement signals
        improvement_signals = await self._extract_improvement_signals(
            intent, result, session_signals, satisfaction_probability
        )

        # Create implicit feedback
        feedback = ImplicitFeedback(
            satisfaction_probability=satisfaction_probability,
            improvement_signals=improvement_signals,
            learning_weight=learning_weight,
            confidence=self._calculate_confidence(session_signals, satisfaction_probability),
            session_signals=session_signals,
            satisfaction_signals=satisfaction_signals,
        )

        # Store learning signal for telemetry
        learning_signal = LearningSignal.create_from_feedback(
            feedback, intent, response_metadata.get("strategy_used")
        )
        self._learning_signals.append(learning_signal)

        # Update behavioral patterns
        await self._update_behavioral_patterns(intent, feedback, session_signals)

        # Send to telemetry service if available
        if self.telemetry_service:
            await self._track_learning_signal(learning_signal, response_metadata)

        return feedback

    async def extract_behavioral_patterns(self, session_data: dict[str, Any]) -> BehavioralPatterns:
        """Extract behavioral patterns from session data."""
        # Generate pattern ID based on session characteristics
        pattern_id = self._generate_pattern_id(session_data)

        # Analyze session for behavioral patterns
        pattern_type = self._classify_pattern_type(session_data)
        confidence = self._calculate_pattern_confidence(session_data)
        context_features = self._extract_context_features(session_data)
        success_correlation = self._calculate_success_correlation(session_data)

        pattern = BehavioralPatterns(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            frequency=session_data.get("frequency", 1),
            confidence=confidence,
            context_features=context_features,
            success_correlation=success_correlation,
        )

        # Store pattern in database
        self._pattern_database[pattern_id] = pattern

        return pattern

    async def _extract_session_signals(
        self, ctx: Any, intent: str, result: IntentResult, metadata: dict[str, Any]
    ) -> SessionSignals:
        """Extract session signals from FastMCP context."""
        # Generate or extract session ID
        session_id = self._get_session_id(ctx)

        # Extract user agent and timing from context
        user_agent = self._get_user_agent(ctx)
        request_timing = self._get_request_timing(ctx, metadata)

        # Track this interaction
        return await self.session_tracker.track_session_interaction(
            session_id=session_id,
            intent=intent,
            result=result,
            context={"user_agent": user_agent, "timing": request_timing},
        )

    def _get_session_id(self, ctx: Any) -> str:
        """Get or generate session ID from context."""
        if hasattr(ctx, "session_id") and ctx.session_id:
            return str(ctx.session_id)

        # Generate session ID from context attributes
        session_data = []
        if (
            hasattr(ctx, "request")
            and ctx.request
            and (hasattr(ctx.request, "client") and ctx.request.client)
        ):
            session_data.append(str(ctx.request.client.host))

        if hasattr(ctx, "user_agent"):
            session_data.append(str(ctx.user_agent))

        if session_data:
            joined_session = "|".join(session_data)
            return hashlib.sha256(joined_session.encode()).hexdigest()[:16]

        # Fallback to random session ID
        return str(uuid.uuid4())[:16]

    def _get_user_agent(self, ctx: Any) -> str | None:
        """Extract user agent from context."""
        if hasattr(ctx, "user_agent"):
            return str(ctx.user_agent)

        if hasattr(ctx, "request") and ctx.request and hasattr(ctx.request, "headers"):
            return ctx.request.headers.get("user-agent")

        return None

    def _get_request_timing(self, ctx: Any, metadata: dict[str, Any]) -> dict[str, float]:
        """Extract request timing information."""
        timing = {}

        # From metadata
        if "response_time" in metadata:
            timing["response_time"] = metadata["response_time"]

        # From context
        if hasattr(ctx, "start_time"):
            timing["start_time"] = ctx.start_time

        if hasattr(ctx, "request") and ctx.request and hasattr(ctx.request, "receive_time"):
            timing["receive_time"] = ctx.request.receive_time

        return timing

    def _calculate_learning_weight(
        self, satisfaction_probability: float, session_signals: SessionSignals
    ) -> float:
        """Calculate learning weight based on confidence and novelty."""
        # Higher weight for more confident satisfaction signals
        confidence_weight = (
            satisfaction_probability
            if satisfaction_probability > 0.5
            else (1.0 - satisfaction_probability)
        )

        # Higher weight for sessions with more interactions (more data)
        interaction_weight = min(session_signals.interaction_count / 10.0, 1.0)

        # Higher weight for engaged sessions
        engagement_weight = session_signals.response_engagement

        # Combine weights
        learning_weight = (
            (confidence_weight * 0.5) + (interaction_weight * 0.3) + (engagement_weight * 0.2)
        )

        return min(learning_weight, 1.0)

    def _calculate_confidence(
        self, session_signals: SessionSignals, satisfaction_probability: float
    ) -> float:
        """Calculate confidence in the satisfaction assessment."""
        # More interactions = higher confidence
        interaction_confidence = min(session_signals.interaction_count / 5.0, 1.0)

        # More extreme satisfaction scores = higher confidence
        satisfaction_confidence = abs(satisfaction_probability - 0.5) * 2.0

        # Engagement indicates more reliable signals
        engagement_confidence = session_signals.response_engagement

        return (
            (interaction_confidence * 0.4)
            + (satisfaction_confidence * 0.4)
            + (engagement_confidence * 0.2)
        )

    async def _extract_improvement_signals(
        self,
        intent: str,
        result: IntentResult,
        session_signals: SessionSignals,
        satisfaction_probability: float,
    ) -> dict[str, Any]:
        """Extract signals for areas of improvement."""
        improvement_signals = {}

        # Low satisfaction indicates need for improvement
        if satisfaction_probability < self._config.satisfaction_threshold:
            improvement_signals["overall_satisfaction"] = "low"

            # Analyze specific areas for improvement
            if not result.success:
                improvement_signals["result_quality"] = "failed_execution"
            elif not result.data:
                improvement_signals["result_quality"] = "empty_results"

            # Response time improvement
            if session_signals.request_timing.get("response_time", 0) > 2.0:
                improvement_signals["response_time"] = "slow_response"

            # Follow-up patterns indicate unclear results
            if len(session_signals.follow_up_queries) > 2:
                improvement_signals["result_clarity"] = "requires_followup"

        # Session engagement signals
        if session_signals.response_engagement < 0.5:
            improvement_signals["engagement"] = "low_engagement"

        return improvement_signals

    async def _update_behavioral_patterns(
        self, intent: str, feedback: ImplicitFeedback, session_signals: SessionSignals
    ) -> None:
        """Update behavioral patterns based on learning signals."""
        # Extract pattern features from intent and session
        pattern_features = {
            "intent_type": self._classify_intent_type(intent),
            "session_length": session_signals.interaction_count,
            "satisfaction_level": self._categorize_satisfaction(feedback.satisfaction_probability),
            "user_agent_hash": hashlib.sha256(
                (session_signals.user_agent or "unknown").encode()
            ).hexdigest()[:8],
        }

        # Create or update pattern
        pattern_id = self._generate_pattern_id(pattern_features)

        if pattern_id in self._pattern_database:
            # Update existing pattern
            pattern = self._pattern_database[pattern_id]
            pattern.frequency += 1

            # Update success correlation with learning rate
            new_success = 1.0 if feedback.satisfaction_probability > 0.7 else 0.0
            pattern.success_correlation = (
                pattern.success_correlation * (1 - self._config.learning_rate)
                + new_success * self._config.learning_rate
            )
        else:
            # Create new pattern
            pattern = BehavioralPatterns(
                pattern_id=pattern_id,
                pattern_type=pattern_features["intent_type"],
                frequency=1,
                confidence=feedback.confidence,
                context_features=pattern_features,
                success_correlation=1.0 if feedback.satisfaction_probability > 0.7 else 0.0,
            )
            self._pattern_database[pattern_id] = pattern

    def _classify_intent_type(self, intent: str) -> str:
        """Classify intent type for pattern analysis."""
        intent_lower = intent.lower()

        if any(word in intent_lower for word in ["find", "search", "look", "get"]):
            return "search"
        if any(word in intent_lower for word in ["explain", "describe", "what", "how"]):
            return "explanation"
        if any(word in intent_lower for word in ["analyze", "review", "check", "examine"]):
            return "analysis"
        return "other"

    def _categorize_satisfaction(self, satisfaction_probability: float) -> str:
        """Categorize satisfaction level."""
        if satisfaction_probability >= 0.8:
            return "high"
        return "medium" if satisfaction_probability >= 0.6 else "low"

    def _generate_pattern_id(self, pattern_data: dict[str, Any]) -> str:
        """Generate unique pattern ID from pattern data."""
        joined_pattern = "|".join(f"{k}:{v}" for k, v in sorted(pattern_data.items()))
        return hashlib.sha256(joined_pattern.encode()).hexdigest()[:16]

    def _classify_pattern_type(self, session_data: dict[str, Any]) -> str:
        """Classify the type of behavioral pattern."""
        # Simple classification based on session characteristics
        interaction_count = session_data.get("interaction_count", 0)

        if interaction_count == 1:
            return "single_query"
        if interaction_count <= 3:
            return "short_session"
        return "exploratory_session" if interaction_count <= 10 else "extended_session"

    def _calculate_pattern_confidence(self, session_data: dict[str, Any]) -> float:
        """Calculate confidence in pattern identification."""
        frequency = session_data.get("frequency", 1)
        interaction_count = session_data.get("interaction_count", 1)

        # More frequent patterns and longer sessions = higher confidence
        frequency_confidence = min(frequency / 10.0, 1.0)
        interaction_confidence = min(interaction_count / 5.0, 1.0)

        return (frequency_confidence * 0.7) + (interaction_confidence * 0.3)

    def _extract_context_features(self, session_data: dict[str, Any]) -> dict[str, Any]:
        """Extract contextual features from session data."""
        return {
            "interaction_count": session_data.get("interaction_count", 0),
            "avg_response_time": session_data.get("avg_response_time", 0.0),
            "success_rate": session_data.get("success_rate", 0.0),
            "session_duration": session_data.get("duration_minutes", 0.0),
        }

    def _calculate_success_correlation(self, session_data: dict[str, Any]) -> float:
        """Calculate correlation between pattern and success."""
        return session_data.get("success_rate", 0.5)

    async def _track_learning_signal(
        self, learning_signal: LearningSignal, metadata: dict[str, Any]
    ) -> None:
        """Track learning signal via telemetry service."""
        if not self.telemetry_service:
            return

        try:
            # Use the extended telemetry method we'll add later
            if hasattr(self.telemetry_service, "track_learning_signal"):
                await self.telemetry_service.track_learning_signal(
                    intent_hash=learning_signal.intent_hash,
                    satisfaction_score=learning_signal.satisfaction_score,
                    metadata={
                        "response_time": learning_signal.response_time,
                        "strategy": learning_signal.strategy_used,
                        "success": learning_signal.success,
                        "learning_weight": learning_signal.learning_weight,
                    },
                )
        except Exception as e:
            self._logger.warning("Failed to track learning signal: %s", e)

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for implicit learning operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "implicit_learning_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {
                "max_behavioral_patterns": self._config.max_behavioral_patterns,
                "pattern_decay_rate": self._config.pattern_decay_rate,
                "learning_rate": self._config.learning_rate,
                "min_pattern_confidence": self._config.min_pattern_confidence,
                "enable_adaptive_thresholds": self._config.enable_adaptive_thresholds,
                "context_sensitivity": self._config.context_sensitivity,
                "update_frequency": self._config.update_frequency,
            },
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
        })

        # Add runtime statistics
        context.update({
            "statistics": {
                "active_patterns": len(self._behavioral_patterns),
                "total_learning_signals": self._learning_stats["total_signals"],
                "successful_learning_events": self._learning_stats["successful_learning_events"],
                "failed_learning_events": self._learning_stats["failed_learning_events"],
                "pattern_updates": self._learning_stats["pattern_updates"],
                "avg_learning_time": self._learning_stats["avg_learning_time"],
                "confidence_improvements": self._learning_stats["confidence_improvements"],
                "patterns_learned": len([
                    p
                    for p in self._behavioral_patterns.values()
                    if p.confidence >= self._config.min_pattern_confidence
                ]),
            }
        })

        return context
