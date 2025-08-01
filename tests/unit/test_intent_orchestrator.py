# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Unit tests for Intent Orchestrator service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from codeweaver.services.providers.intent_orchestrator import IntentOrchestrator
from codeweaver.cw_types import (
    Complexity,
    HealthStatus,
    IntentResult,
    IntentServiceConfig,
    IntentType,
    ParsedIntent,
    Scope,
    ServiceHealth,
    ServiceType,
)


class TestIntentOrchestrator:
    """Test intent orchestrator using existing testing framework."""

    @pytest.fixture
    def intent_config(self):
        """Intent orchestrator configuration."""
        return IntentServiceConfig(
            provider="intent_orchestrator",
            enabled=True,
            default_strategy="adaptive",
            confidence_threshold=0.6,
            max_execution_time=30.0,
            debug_mode=False,
            cache_ttl=3600,
        )

    @pytest.fixture
    def intent_orchestrator(self, intent_config):
        """Intent orchestrator with mocked dependencies."""
        return IntentOrchestrator(intent_config)

    @pytest.fixture
    def mock_parsed_intent(self):
        """Mock parsed intent for testing."""
        from datetime import UTC, datetime

        return ParsedIntent(
            intent_type=IntentType.SEARCH,
            primary_target="authentication functions",
            scope=Scope.PROJECT,
            complexity=Complexity.MODERATE,
            confidence=0.8,
            filters={},
            metadata={"parser": "pattern_based"},
            parsed_at=datetime.now(UTC),
        )

    async def test_service_provider_compliance(self, intent_orchestrator):
        """Test compliance with BaseServiceProvider."""
        from codeweaver.services.providers.base_provider import BaseServiceProvider

        assert isinstance(intent_orchestrator, BaseServiceProvider)
        assert intent_orchestrator.service_type == ServiceType.INTENT

    async def test_initialization(self, intent_orchestrator):
        """Test orchestrator initialization."""
        with patch("codeweaver.intent.parsing.factory.IntentParserFactory.create") as mock_factory:
            mock_parser = Mock()
            mock_factory.return_value = mock_parser

            await intent_orchestrator._initialize_provider()

            assert intent_orchestrator.parser == mock_parser
            mock_factory.assert_called_once()

    async def test_health_check_healthy(self, intent_orchestrator):
        """Test health check when service is healthy."""
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=Mock(spec=ParsedIntent))
        intent_orchestrator.parser = mock_parser

        # Call health check through the base provider
        is_healthy = await intent_orchestrator._check_health()
        assert is_healthy is True

    async def test_health_check_unhealthy_no_parser(self, intent_orchestrator):
        """Test health check when parser is not available."""
        intent_orchestrator.parser = None

        is_healthy = await intent_orchestrator._check_health()
        assert is_healthy is False

    async def test_process_intent_basic_fallback(self, intent_orchestrator, mock_parsed_intent):
        """Test basic intent processing with fallback."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=mock_parsed_intent)
        intent_orchestrator.parser = mock_parser
        intent_orchestrator.strategy_registry = None  # No registry, should use fallback

        # Process intent
        result = await intent_orchestrator.process_intent("find auth functions", {})

        # Verify results
        assert isinstance(result, IntentResult)
        assert result.success is True
        assert result.strategy_used == "basic_fallback"
        assert "authentication functions" in result.data["target"]
        assert result.metadata["intent_type"] == "search"
        assert result.metadata["fallback_used"] is True

    async def test_process_intent_with_caching(self, intent_orchestrator, mock_parsed_intent):
        """Test intent processing with caching."""
        # Setup cache mock
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        intent_orchestrator.cache_service = mock_cache

        # Setup parser mock
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=mock_parsed_intent)
        intent_orchestrator.parser = mock_parser

        # Process intent
        result = await intent_orchestrator.process_intent("find auth functions", {})

        # Verify caching was attempted
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        assert result.success is True

    async def test_process_intent_cache_hit(self, intent_orchestrator):
        """Test intent processing with cache hit."""
        from datetime import UTC, datetime

        cached_result = IntentResult(
            success=True,
            data={"cached": True},
            metadata={"from_cache": True},
            executed_at=datetime.now(UTC),
            execution_time=0.1,
            strategy_used="cached",
        )

        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=cached_result)
        intent_orchestrator.cache_service = mock_cache

        result = await intent_orchestrator.process_intent("cached query", {})

        assert result == cached_result
        assert intent_orchestrator._intent_stats["cached_hits"] == 1

    async def test_no_index_intent_conversion(self, intent_orchestrator):
        """Test that INDEX intents are converted to SEARCH."""
        from datetime import UTC, datetime

        # Create a mock parsed intent with INDEX type
        index_intent = ParsedIntent(
            intent_type="INDEX",  # This should be converted
            primary_target="codebase",
            scope=Scope.PROJECT,
            complexity=Complexity.SIMPLE,
            confidence=0.9,
            filters={},
            metadata={},
            parsed_at=datetime.now(UTC),
        )

        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=index_intent)
        intent_orchestrator.parser = mock_parser

        result = await intent_orchestrator.process_intent("index this codebase", {})

        # Should be converted to SEARCH
        assert result.metadata.get("intent_type") != "INDEX"
        assert "background" in result.metadata.get("background_indexing_note", "").lower()

    async def test_get_capabilities(self, intent_orchestrator):
        """Test get capabilities method."""
        intent_orchestrator.parser = Mock()
        intent_orchestrator.cache_service = Mock()
        intent_orchestrator._intent_stats = {
            "total_processed": 10,
            "successful_intents": 8,
            "failed_intents": 2,
            "cached_hits": 3,
            "cache_misses": 7,
            "avg_processing_time": 2.5,
            "min_processing_time": 1.0,
            "max_processing_time": 5.0,
            "parsing_failures": 0,
            "strategy_failures": 0,
            "concurrent_requests": 0,
        }

        capabilities = await intent_orchestrator.get_capabilities()

        assert "intent_types" in capabilities
        assert "SEARCH" in capabilities["intent_types"]
        assert "UNDERSTAND" in capabilities["intent_types"]
        assert "ANALYZE" in capabilities["intent_types"]
        assert "INDEX" not in capabilities["intent_types"]  # Should not be exposed
        assert capabilities["services"]["cache_enabled"] is True
        assert capabilities["processing_stats"]["success_rate"] == 0.8
        assert capabilities["processing_stats"]["cache_hit_rate"] == 0.3

    async def test_error_handling_parsing_error(self, intent_orchestrator):
        """Test error handling for parsing errors."""
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(side_effect=Exception("Parse failed"))
        intent_orchestrator.parser = mock_parser

        result = await intent_orchestrator.process_intent("invalid intent", {})

        assert result.success is False
        assert "Intent processing failed" in result.error_message
        assert result.strategy_used == "error_fallback"
        assert len(result.suggestions) > 0

    async def test_statistics_tracking(self, intent_orchestrator, mock_parsed_intent):
        """Test that statistics are properly tracked."""
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=mock_parsed_intent)
        intent_orchestrator.parser = mock_parser

        # Process successful intent
        result = await intent_orchestrator.process_intent("test intent", {})

        assert intent_orchestrator._intent_stats["total_processed"] == 1
        assert intent_orchestrator._intent_stats["successful_intents"] == 1
        assert result.success is True

    async def test_health_check_with_metadata(self, intent_orchestrator):
        """Test health check returns proper structure."""
        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=Mock(spec=ParsedIntent))
        intent_orchestrator.parser = mock_parser

        health = await intent_orchestrator.health_check()

        assert isinstance(health, ServiceHealth)
        assert health.status == HealthStatus.HEALTHY
        assert health.service_type == ServiceType.INTENT
        assert health.response_time >= 0

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, intent_orchestrator, mock_parsed_intent):
        """Test concurrent intent processing."""
        import asyncio

        mock_parser = Mock()
        mock_parser.parse = AsyncMock(return_value=mock_parsed_intent)
        intent_orchestrator.parser = mock_parser

        # Process multiple intents concurrently
        tasks = [intent_orchestrator.process_intent(f"intent {i}", {}) for i in range(5)]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result.success for result in results)
        assert intent_orchestrator._intent_stats["total_processed"] == 5
        assert intent_orchestrator._intent_stats["successful_intents"] == 5


@pytest.mark.integration
class TestIntentOrchestratorIntegration:
    """Integration tests for intent orchestrator with real dependencies."""

    async def test_with_real_parser(self):
        """Test orchestrator with real intent parser."""
        config = IntentServiceConfig(enabled=True)
        orchestrator = IntentOrchestrator(config)

        with patch("codeweaver.intent.parsing.factory.IntentParserFactory.create") as mock_factory:
            from codeweaver.intent.parsing.pattern_matcher import PatternBasedParser

            mock_factory.return_value = PatternBasedParser()

            await orchestrator._initialize_provider()

            result = await orchestrator.process_intent("find authentication functions", {})

            assert result.success is True
            assert result.metadata["intent_type"] in ["search", "understand", "analyze"]
