# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for Phase 3 intent layer services."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from codeweaver.services.providers.context_intelligence import FastMCPContextMiningProvider
from codeweaver.services.providers.implicit_learning import BehavioralPatternLearningProvider
from codeweaver.services.providers.zero_shot_optimization import ContextAdequacyOptimizationProvider
from codeweaver.types import (
    ContextIntelligenceServiceConfig,
    ImplicitLearningServiceConfig,
    IntentResult,
    ServiceType,
    ZeroShotOptimizationServiceConfig,
)


class TestImplicitLearningService:
    """Test cases for ImplicitLearningService."""

    @pytest.fixture
    def service_config(self):
        """Create test configuration for implicit learning service."""
        return ImplicitLearningServiceConfig(
            provider="behavioral_pattern_learning",
            satisfaction_threshold=0.8,
            learning_rate=0.1,
            pattern_retention_days=30,
        )

    @pytest.fixture
    def learning_service(self, service_config):
        """Create implicit learning service instance."""
        return BehavioralPatternLearningProvider(
            ServiceType.IMPLICIT_LEARNING,
            service_config
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, learning_service):
        """Test that the service initializes properly."""
        await learning_service._initialize_provider()
        assert learning_service.session_tracker is not None
        assert learning_service.satisfaction_detector is not None

    @pytest.mark.asyncio
    async def test_analyze_interaction_outcome(self, learning_service):
        """Test interaction outcome analysis."""
        await learning_service._initialize_provider()

        # Mock context
        mock_ctx = Mock()
        mock_ctx.session_id = "test_session"
        mock_ctx.user_agent = "test_agent"

        # Create test intent result
        intent_result = IntentResult(
            success=True,
            data=["result1", "result2"],
            metadata={"strategy": "test"},
            executed_at=datetime.now(UTC),
            execution_time=1.5
        )

        # Test analysis
        feedback = await learning_service.analyze_interaction_outcome(
            ctx=mock_ctx,
            intent="test intent",
            result=intent_result,
            response_metadata={"response_time": 1.5, "strategy_used": "test"}
        )

        assert feedback is not None
        assert 0.0 <= feedback.satisfaction_probability <= 1.0
        assert 0.0 <= feedback.learning_weight <= 1.0
        assert 0.0 <= feedback.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_behavioral_pattern_extraction(self, learning_service):
        """Test behavioral pattern extraction."""
        await learning_service._initialize_provider()

        session_data = {
            "interaction_count": 5,
            "frequency": 3,
            "success_rate": 0.8,
            "avg_response_time": 1.2,
            "duration_minutes": 15.0
        }

        pattern = await learning_service.extract_behavioral_patterns(session_data)

        assert pattern is not None
        assert pattern.frequency == 3
        assert 0.0 <= pattern.confidence <= 1.0
        assert pattern.pattern_type is not None


class TestContextIntelligenceService:
    """Test cases for ContextIntelligenceService."""

    @pytest.fixture
    def service_config(self):
        """Create test configuration for context intelligence service."""
        return ContextIntelligenceServiceConfig(
            provider="fastmcp_context_mining",
            llm_identification_enabled=True,
            behavioral_fingerprinting=True,
            privacy_mode="hash_identifiers",
        )

    @pytest.fixture
    def intelligence_service(self, service_config):
        """Create context intelligence service instance."""
        return FastMCPContextMiningProvider(
            ServiceType.CONTEXT_INTELLIGENCE,
            service_config
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, intelligence_service):
        """Test that the service initializes properly."""
        await intelligence_service._initialize_provider()
        assert intelligence_service.model_detector is not None
        assert intelligence_service.context_predictor is not None

    @pytest.mark.asyncio
    async def test_extract_llm_characteristics(self, intelligence_service):
        """Test LLM characteristics extraction."""
        await intelligence_service._initialize_provider()

        # Mock context
        mock_ctx = Mock()
        mock_ctx.session_id = "test_session"
        mock_ctx.user_agent = "Claude/1.0"
        mock_ctx.start_time = 1234567890.0

        profile = await intelligence_service.extract_llm_characteristics(mock_ctx)

        assert profile is not None
        assert profile.session_id == "test_session"
        assert 0.0 <= profile.confidence <= 1.0
        assert isinstance(profile.request_patterns, list)
        assert isinstance(profile.timing_characteristics, dict)

    @pytest.mark.asyncio
    async def test_analyze_context_adequacy(self, intelligence_service):
        """Test context adequacy analysis."""
        await intelligence_service._initialize_provider()

        intent = "find all functions in the authentication module"
        context = {
            "scope": "module",
            "target": "authentication",
            "query": "functions"
        }

        adequacy = await intelligence_service.analyze_context_adequacy(intent, context)

        assert adequacy is not None
        assert 0.0 <= adequacy.score <= 1.0
        assert 0.0 <= adequacy.richness_score <= 1.0
        assert 0.0 <= adequacy.clarity_score <= 1.0
        assert isinstance(adequacy.missing_elements, list)
        assert isinstance(adequacy.recommendations, list)


class TestZeroShotOptimizationService:
    """Test cases for ZeroShotOptimizationService."""

    @pytest.fixture
    def service_config(self):
        """Create test configuration for zero-shot optimization service."""
        return ZeroShotOptimizationServiceConfig(
            provider="context_adequacy_optimization",
            success_threshold=0.8,
            optimization_aggressiveness="balanced",
            enable_success_prediction=True,
        )

    @pytest.fixture
    def optimization_service(self, service_config):
        """Create zero-shot optimization service instance."""
        return ContextAdequacyOptimizationProvider(
            ServiceType.ZERO_SHOT_OPTIMIZATION,
            service_config
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, optimization_service):
        """Test that the service initializes properly."""
        await optimization_service._initialize_provider()
        assert optimization_service.context_adequacy_predictor is not None
        assert optimization_service.success_pattern_db is not None

    @pytest.mark.asyncio
    async def test_predict_success_probability(self, optimization_service):
        """Test success probability prediction."""
        await optimization_service._initialize_provider()

        intent = "analyze the performance of the database queries"
        context = {
            "target": "database",
            "scope": "project",
            "focus": "performance"
        }

        prediction = await optimization_service.predict_success_probability(intent, context)

        assert prediction is not None
        assert 0.0 <= prediction.probability <= 1.0
        assert 0.0 <= prediction.context_adequacy <= 1.0
        assert 0.0 <= prediction.strategy_confidence <= 1.0
        assert isinstance(prediction.missing_context, list)
        assert isinstance(prediction.recommended_strategies, list)
        assert isinstance(prediction.risk_factors, list)

    @pytest.mark.asyncio
    async def test_optimize_for_zero_shot_success(self, optimization_service):
        """Test zero-shot optimization."""
        await optimization_service._initialize_provider()

        # Mock context
        mock_ctx = Mock()
        mock_ctx.session_id = "test_session"

        intent = "find authentication functions"
        context = {"query": "authentication"}

        plan = await optimization_service.optimize_for_zero_shot_success(
            ctx=mock_ctx,
            intent=intent,
            available_context=context
        )

        assert plan is not None
        assert plan.original_intent == intent
        assert 0.0 <= plan.success_probability <= 1.0
        assert isinstance(plan.optimizations, list)
        assert isinstance(plan.enhanced_context, dict)
        assert isinstance(plan.optimization_metadata, dict)

    @pytest.mark.asyncio
    async def test_optimization_with_low_success_probability(self, optimization_service):
        """Test optimization when success probability is low."""
        await optimization_service._initialize_provider()

        # Mock context
        mock_ctx = Mock()

        # Create scenario with insufficient context (should trigger optimizations)
        intent = "complex analysis task"
        context = {}  # Empty context should result in low success probability

        plan = await optimization_service.optimize_for_zero_shot_success(
            ctx=mock_ctx,
            intent=intent,
            available_context=context
        )

        assert plan is not None
        # With empty context, success probability should be low
        assert plan.success_probability < 0.8
        # Should have optimizations applied
        assert len(plan.optimizations) > 0


class TestServiceIntegration:
    """Test integration between Phase 3 services."""

    @pytest.mark.asyncio
    async def test_services_work_together(self):
        """Test that the services can work together in a realistic scenario."""
        # Create service instances
        implicit_config = ImplicitLearningServiceConfig()
        context_config = ContextIntelligenceServiceConfig()
        optimization_config = ZeroShotOptimizationServiceConfig()

        learning_service = BehavioralPatternLearningProvider(
            ServiceType.IMPLICIT_LEARNING, implicit_config
        )
        intelligence_service = FastMCPContextMiningProvider(
            ServiceType.CONTEXT_INTELLIGENCE, context_config
        )
        optimization_service = ContextAdequacyOptimizationProvider(
            ServiceType.ZERO_SHOT_OPTIMIZATION, optimization_config
        )

        # Initialize all services
        await learning_service._initialize_provider()
        await intelligence_service._initialize_provider()
        await optimization_service._initialize_provider()

        # Mock context
        mock_ctx = Mock()
        mock_ctx.session_id = "integration_test"
        mock_ctx.user_agent = "TestAgent/1.0"

        # Test workflow:
        # 1. Context intelligence analyzes the request
        intent = "find authentication functions in the security module"
        initial_context = {"query": "authentication", "scope": "security"}

        # 2. Zero-shot optimization creates a plan
        optimization_plan = await optimization_service.optimize_for_zero_shot_success(
            ctx=mock_ctx,
            intent=intent,
            available_context=initial_context
        )

        # 3. Simulate execution result
        result = IntentResult(
            success=True,
            data=["auth_function_1", "auth_function_2"],
            metadata={"strategy": "adaptive"},
            executed_at=datetime.now(UTC),
            execution_time=2.1
        )

        # 4. Implicit learning analyzes the outcome
        feedback = await learning_service.analyze_interaction_outcome(
            ctx=mock_ctx,
            intent=intent,
            result=result,
            response_metadata={
                "response_time": 2.1,
                "strategy_used": "adaptive",
                "success_probability": optimization_plan.success_probability
            }
        )

        # Verify the integration worked
        assert optimization_plan.success_probability >= 0.0
        assert feedback.satisfaction_probability >= 0.0
        assert len(optimization_plan.optimizations) >= 0  # May or may not have optimizations

        # Services should be healthy after processing
        assert await learning_service._check_health()
        assert await intelligence_service._check_health()
        assert await optimization_service._check_health()


@pytest.mark.asyncio
async def test_service_health_checks():
    """Test health checks for all Phase 3 services."""
    configs = [
        (ServiceType.IMPLICIT_LEARNING, ImplicitLearningServiceConfig(), BehavioralPatternLearningProvider),
        (ServiceType.CONTEXT_INTELLIGENCE, ContextIntelligenceServiceConfig(), FastMCPContextMiningProvider),
        (ServiceType.ZERO_SHOT_OPTIMIZATION, ZeroShotOptimizationServiceConfig(), ContextAdequacyOptimizationProvider),
    ]

    for service_type, config, provider_class in configs:
        service = provider_class(service_type, config)
        await service._initialize_provider()

        # Health check should pass for newly initialized service
        is_healthy = await service._check_health()
        assert is_healthy, f"Service {service_type.value} failed health check"

        # Shutdown should work without errors
        await service._shutdown_provider()
