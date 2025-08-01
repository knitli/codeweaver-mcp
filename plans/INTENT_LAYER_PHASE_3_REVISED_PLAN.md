<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Phase 3: Revised Implementation Plan - Implicit Learning & Zero-Shot Optimization

## 🎯 Executive Summary

**Critical Insight**: The original Phase 3 plan fails to account for CodeWeaver's unique userbase - LLMs, not humans. This revised plan addresses the stateless nature of LLM interactions and focuses on implicit behavioral feedback rather than traditional user feedback mechanisms.

**Key Changes**:
- **Zero-Shot Focus**: Optimize for first-attempt success rather than iterative improvement
- **Implicit Feedback**: Learn from behavioral patterns rather than explicit feedback
- **Context Mining**: Leverage FastMCP Context and telemetry for intelligence gathering
- **Service Integration**: Full compliance with established service layer patterns

## 📊 Analysis Summary

### Current Phase 3 Issues
1. **User Feedback Assumption**: Assumes traditional feedback loops impossible with stateless LLMs
2. **A/B Testing Misalignment**: Traditional A/B testing doesn't work with anonymous sessions
3. **Telemetry Integration Gap**: Ignores existing TelemetryService infrastructure
4. **Context Harvesting**: Misses opportunities in FastMCP Context and HTTP Request objects
5. **Service Compliance**: Doesn't follow established service layer patterns

### Architecture Integration Points
- **FastMCP Context**: Provides request-level metadata (user agent, session info, timing)
- **TelemetryService**: Existing privacy-first analytics with PostHog integration
- **Service Layer Patterns**: Established BaseServiceProvider pattern for new services
- **Configuration System**: Hierarchical TOML configuration with environment overrides

## 🏗️ Revised Architecture

### 1. Implicit Learning Framework

Replace traditional feedback with behavioral pattern analysis:

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import ServiceType, ImplicitLearningConfig

class ImplicitLearningService(BaseServiceProvider):
    """Learn from LLM behavioral patterns without explicit feedback."""

    def __init__(self, config: ImplicitLearningConfig):
        super().__init__(ServiceType.IMPLICIT_LEARNING, config)
        self.session_tracker = SessionPatternTracker()
        self.satisfaction_detector = SatisfactionSignalDetector()
        self.telemetry_service = None  # Injected via services manager

    async def _initialize_provider(self) -> None:
        """Initialize behavioral pattern learning."""
        await self.session_tracker.initialize()
        await self.satisfaction_detector.initialize()

    async def analyze_interaction_outcome(
        self,
        ctx: FastMCPContext,
        intent: str,
        result: IntentResult,
        response_metadata: dict[str, Any]
    ) -> ImplicitFeedback:
        """Analyze interaction for implicit satisfaction signals."""

        # Extract session information from FastMCP context
        session_signals = await self._extract_session_signals(ctx)

        # Analyze response timing and follow-up patterns
        satisfaction_probability = await self.satisfaction_detector.calculate_satisfaction(
            intent=intent,
            result=result,
            response_time=response_metadata.get("response_time", 0),
            session_signals=session_signals
        )

        # Store learning data via telemetry service
        if self.telemetry_service:
            await self.telemetry_service.track_learning_signal(
                intent_hash=self._hash_intent(intent),
                satisfaction_score=satisfaction_probability,
                metadata=self._sanitize_metadata(response_metadata)
            )

        return ImplicitFeedback(
            satisfaction_probability=satisfaction_probability,
            improvement_signals=await self._extract_improvement_signals(
                intent, result, session_signals
            ),
            learning_weight=self._calculate_learning_weight(satisfaction_probability)
        )
```

### 2. Context Intelligence Mining

Leverage FastMCP Context and HTTP requests for intelligence:

```python
class ContextIntelligenceService(BaseServiceProvider):
    """Extract intelligence from FastMCP context and HTTP requests."""

    async def extract_llm_characteristics(
        self,
        ctx: FastMCPContext
    ) -> LLMProfile:
        """Extract LLM behavioral characteristics from context."""

        profile_data = {
            "session_id": self._get_session_id(ctx),
            "user_agent": self._get_user_agent(ctx),
            "request_timing": self._get_request_timing(ctx),
            "interaction_patterns": []
        }

        # Analyze request patterns if HTTP context available
        if hasattr(ctx, 'request') and ctx.request:
            http_profile = await self._analyze_http_patterns(ctx.request)
            profile_data.update(http_profile)

        # Build behavioral fingerprint
        behavioral_features = await self._extract_behavioral_features(profile_data)

        return LLMProfile(
            session_id=profile_data["session_id"],
            identified_model=await self._identify_model_type(behavioral_features),
            confidence=behavioral_features.get("identification_confidence", 0.0),
            request_patterns=behavioral_features.get("request_patterns", []),
            timing_characteristics=behavioral_features.get("timing", {})
        )

    async def _analyze_http_patterns(self, request) -> dict[str, Any]:
        """Analyze HTTP request for LLM behavioral patterns."""
        return {
            "headers": self._sanitize_headers(request.headers),
            "timing": {
                "connection_time": getattr(request, 'connection_time', None),
                "processing_start": getattr(request, 'start_time', None)
            },
            "client_info": {
                "ip_hash": self._hash_client_ip(request.client.host if request.client else None),
                "port": request.client.port if request.client else None
            }
        }
```

### 3. Telemetry-Integrated Learning

Integrate with existing TelemetryService following established patterns:

```python
class LearningTelemetryExtension:
    """Extension to existing TelemetryService for learning data."""

    def __init__(self, telemetry_service: TelemetryService):
        self.telemetry_service = telemetry_service

    async def track_learning_signal(
        self,
        intent_hash: str,
        satisfaction_score: float,
        metadata: dict[str, Any]
    ) -> None:
        """Track learning signals through existing telemetry."""

        if not self.telemetry_service.is_enabled():
            return

        # Use existing telemetry patterns for privacy compliance
        learning_event = {
            "event": "intent_learning_signal",
            "properties": {
                "intent_hash": intent_hash,
                "satisfaction_score": satisfaction_score,
                "response_time": metadata.get("response_time"),
                "strategy_used": metadata.get("strategy"),
                "success": metadata.get("success", False),
                # Sanitized metadata following existing patterns
                **self.telemetry_service._sanitize_metadata(metadata)
            }
        }

        await self.telemetry_service.track_event(learning_event)

    async def track_zero_shot_optimization(
        self,
        optimization_type: str,
        before_metrics: dict[str, Any],
        after_metrics: dict[str, Any]
    ) -> None:
        """Track zero-shot optimization effectiveness."""

        optimization_event = {
            "event": "zero_shot_optimization",
            "properties": {
                "optimization_type": optimization_type,
                "improvement_score": self._calculate_improvement_score(
                    before_metrics, after_metrics
                ),
                "metrics_delta": self._calculate_metrics_delta(
                    before_metrics, after_metrics
                )
            }
        }

        await self.telemetry_service.track_event(optimization_event)
```

### 4. Zero-Shot Optimization Engine

Focus on first-attempt success rather than iterative improvement:

```python
class ZeroShotOptimizationService(BaseServiceProvider):
    """Optimize for zero-shot intent resolution success."""

    def __init__(self, config: ZeroShotOptimizationConfig):
        super().__init__(ServiceType.ZERO_SHOT_OPTIMIZATION, config)
        self.context_adequacy_predictor = ContextAdequacyPredictor()
        self.success_pattern_db = SuccessPatternDatabase()

    async def optimize_for_zero_shot_success(
        self,
        ctx: FastMCPContext,
        intent: str,
        available_context: dict[str, Any]
    ) -> OptimizedIntentPlan:
        """Optimize intent processing for first-attempt success."""

        # Predict zero-shot success probability
        success_prediction = await self._predict_zero_shot_success(
            ctx, intent, available_context
        )

        optimization_plan = OptimizedIntentPlan(
            original_intent=intent,
            success_probability=success_prediction.probability,
            optimizations=[]
        )

        # Apply optimizations if success probability is low
        if success_prediction.probability < 0.8:
            # Context enrichment based on learned patterns
            if success_prediction.context_adequacy < 0.7:
                context_optimizations = await self._suggest_context_enrichment(
                    intent, available_context, success_prediction.missing_context
                )
                optimization_plan.optimizations.extend(context_optimizations)

            # Strategy selection optimization
            if success_prediction.strategy_confidence < 0.8:
                strategy_optimizations = await self._suggest_strategy_improvements(
                    intent, success_prediction.recommended_strategies
                )
                optimization_plan.optimizations.extend(strategy_optimizations)

        return optimization_plan

    async def _predict_zero_shot_success(
        self,
        ctx: FastMCPContext,
        intent: str,
        context: dict[str, Any]
    ) -> SuccessPrediction:
        """Predict likelihood of zero-shot success."""

        # Extract contextual features
        contextual_features = await self._extract_contextual_features(ctx, intent)

        # Assess context adequacy
        context_adequacy = await self.context_adequacy_predictor.assess(
            intent, context, contextual_features
        )

        # Get historical success patterns
        historical_patterns = await self.success_pattern_db.get_similar_patterns(
            intent, contextual_features
        )

        # Calculate composite success probability
        success_probability = self._calculate_success_probability(
            context_adequacy=context_adequacy.score,
            historical_success=historical_patterns.average_success_rate,
            intent_clarity=contextual_features.clarity_score,
            contextual_richness=contextual_features.richness_score
        )

        return SuccessPrediction(
            probability=success_probability,
            context_adequacy=context_adequacy.score,
            strategy_confidence=historical_patterns.strategy_confidence,
            missing_context=context_adequacy.missing_elements,
            recommended_strategies=historical_patterns.top_strategies
        )
```

## 📅 Revised Implementation Timeline

### Week 1: Context Intelligence & Implicit Learning Foundation

**Deliverables**:
1. **ContextIntelligenceService** - Extract LLM characteristics from FastMCP context
2. **ImplicitLearningService** - Behavioral pattern analysis without explicit feedback
3. **TelemetryService Extensions** - Learning data collection following existing patterns
4. **Service Integration** - Full compliance with BaseServiceProvider patterns

**Integration Points**:
- FastMCP Context object analysis for session data
- HTTP Request object mining when available (Starlette integration)
- Existing TelemetryService extension for privacy-compliant learning data
- ServicesManager integration following established patterns

### Week 2: Zero-Shot Optimization & Pattern Database

**Deliverables**:
1. **ZeroShotOptimizationService** - First-attempt success optimization
2. **SuccessPatternDatabase** - Historical pattern storage and retrieval
3. **ContextAdequacyPredictor** - Context completeness assessment
4. **LLM Behavioral Fingerprinting** - Model identification through behavior

**Integration Points**:
- Configuration through existing TOML hierarchy
- Health monitoring via existing service health patterns
- Telemetry integration for optimization effectiveness tracking

### Week 3: Advanced Analytics & Developer Tools

**Deliverables**:
1. **ImplicitAnalyticsService** - Advanced behavioral pattern analysis
2. **Zero-Shot Performance Monitor** - Real-time optimization effectiveness
3. **Developer Debugging Extensions** - Implicit learning visibility tools
4. **Privacy-Compliant Reporting** - Analytics following existing privacy patterns

**Integration Points**:
- Debugging tools integration with existing development patterns
- Analytics dashboards respecting privacy-first design
- Configuration management through existing hierarchical system

## 🔧 Configuration Integration

Leverage existing hierarchical configuration system:

```toml
# config.toml - Following existing patterns
[services.implicit_learning]
enabled = true
provider = "behavioral_pattern_learning"
satisfaction_threshold = 0.8
learning_rate = 0.1
pattern_retention_days = 30

[services.zero_shot_optimization]
enabled = true
provider = "context_adequacy_optimization"
success_threshold = 0.8
optimization_aggressiveness = "balanced"  # conservative, balanced, aggressive

[services.context_intelligence]
enabled = true
provider = "fastmcp_context_mining"
llm_identification_enabled = true
behavioral_fingerprinting = true
privacy_mode = "hash_identifiers"

# Environment variable overrides following existing patterns
# CW_IMPLICIT_LEARNING_ENABLED=false
# CW_ZERO_SHOT_OPTIMIZATION_SUCCESS_THRESHOLD=0.9
# CW_CONTEXT_INTELLIGENCE_PRIVACY_MODE=strict
```

## 🔒 Privacy & Compliance

Maintain privacy-first design following existing TelemetryService patterns:

- **Session-based hashing**: User identification through session-specific hashes
- **Data sanitization**: All contextual data sanitized before storage
- **Opt-out mechanisms**: Multiple opt-out paths following existing patterns
- **Anonymous tracking**: Default to anonymous behavioral pattern collection
- **Local-first processing**: Minimize data transmission, maximize local intelligence

## 📊 Success Metrics

### Implicit Learning Effectiveness
- **Zero-Shot Success Rate**: Target >90% (up from current ~75%)
- **Learning Signal Quality**: Satisfaction prediction accuracy >85%
- **Pattern Recognition**: Behavioral pattern identification accuracy >80%

### Context Intelligence
- **LLM Model Identification**: Behavioral fingerprinting accuracy >70%
- **Context Adequacy Prediction**: Context completeness prediction accuracy >85%
- **Session Pattern Recognition**: Pattern identification across sessions >75%

### Service Integration Compliance
- **Health Monitoring**: 100% compliance with service health patterns
- **Configuration Management**: 100% hierarchical configuration compliance
- **Privacy Compliance**: 100% compliance with existing privacy patterns

## 🚀 Architectural Benefits

### Zero-Shot Focus
- Optimizes for LLM workflow reality (stateless, first-attempt preference)
- Reduces need for follow-up queries and clarification cycles
- Improves overall user experience through better first-attempt results

### Implicit Intelligence
- Learns continuously without interrupting LLM workflow
- Builds intelligence database for future optimization
- Maintains privacy through behavioral pattern analysis

### Full Architectural Integration
- 100% compliance with existing service layer patterns
- Leverages existing TelemetryService infrastructure
- Follows established configuration and health monitoring patterns

## 🎯 Success Criteria

**Technical Excellence**:
- [ ] Zero-shot success rate improvement >15%
- [ ] Implicit learning accuracy >85%
- [ ] Full service layer compliance
- [ ] Privacy-first design maintained

**User Experience**:
- [ ] Reduced need for query reformulation
- [ ] Improved first-attempt result quality
- [ ] Transparent learning (no workflow interruption)

**Developer Experience**:
- [ ] Debugging tools for implicit learning visibility
- [ ] Analytics for optimization effectiveness
- [ ] Configuration flexibility for different deployment scenarios

---

*This revised Phase 3 plan addresses the unique challenges of CodeWeaver's LLM userbase while maintaining full architectural integrity and leveraging existing infrastructure for maximum effectiveness.*
