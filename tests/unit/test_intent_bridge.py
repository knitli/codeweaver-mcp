"""Unit tests for Intent Service Bridge."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from codeweaver.intent.middleware.intent_bridge import IntentServiceBridge
from codeweaver.types import (
    IntentResult,
    ServiceHealth,
    HealthStatus,
    ServiceIntegrationError,
    ServiceType,
)


class TestIntentServiceBridge:
    """Test intent service bridge functionality."""

    @pytest.fixture
    def mock_services_manager(self):
        """Mock services manager."""
        manager = Mock()
        manager.get_service = AsyncMock()
        return manager

    @pytest.fixture
    def intent_bridge(self, mock_services_manager):
        """Intent service bridge with mocked dependencies."""
        return IntentServiceBridge(mock_services_manager)

    @pytest.fixture
    def mock_intent_orchestrator(self):
        """Mock intent orchestrator."""
        orchestrator = Mock()
        orchestrator.process_intent = AsyncMock()
        orchestrator.get_capabilities = AsyncMock()
        orchestrator.health_check = AsyncMock()
        return orchestrator

    @pytest.fixture
    def mock_auto_indexing_service(self):
        """Mock auto-indexing service."""
        service = Mock()
        service.trigger_indexing = AsyncMock()
        service.health_check = AsyncMock()
        return service

    async def test_service_provider_compliance(self, intent_bridge):
        """Test compliance with BaseServiceProvider."""
        from codeweaver.services.providers.base_provider import BaseServiceProvider
        
        assert isinstance(intent_bridge, BaseServiceProvider)
        assert intent_bridge.service_type == ServiceType.INTENT

    async def test_initialization_success(self, intent_bridge, mock_intent_orchestrator, mock_auto_indexing_service):
        """Test successful bridge initialization."""
        # Setup service manager mocks
        intent_bridge.services_manager.get_service.side_effect = lambda service_name: {
            "intent_orchestrator": mock_intent_orchestrator,
            "auto_indexing": mock_auto_indexing_service,
        }.get(service_name)
        
        await intent_bridge._initialize_provider()
        
        assert intent_bridge.bridge_initialized is True
        assert intent_bridge.intent_orchestrator == mock_intent_orchestrator
        assert intent_bridge.auto_indexing_service == mock_auto_indexing_service

    async def test_initialization_missing_orchestrator(self, intent_bridge):
        """Test initialization with missing orchestrator."""
        intent_bridge.services_manager.get_service.return_value = None
        
        with pytest.raises(ServiceIntegrationError, match="IntentOrchestrator service not available"):
            await intent_bridge._initialize_provider()

    async def test_initialization_missing_auto_indexing(self, intent_bridge, mock_intent_orchestrator):
        """Test initialization with missing auto-indexing (should not fail)."""
        intent_bridge.services_manager.get_service.side_effect = lambda service_name: {
            "intent_orchestrator": mock_intent_orchestrator,
            "auto_indexing": None,
        }.get(service_name)
        
        await intent_bridge._initialize_provider()
        
        assert intent_bridge.bridge_initialized is True
        assert intent_bridge.intent_orchestrator == mock_intent_orchestrator
        assert intent_bridge.auto_indexing_service is None

    async def test_health_check_healthy(self, intent_bridge, mock_intent_orchestrator, mock_auto_indexing_service):
        """Test health check when all services are healthy."""
        # Setup bridge
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        intent_bridge.auto_indexing_service = mock_auto_indexing_service
        
        # Mock healthy responses
        mock_orchestrator_health = Mock()
        mock_orchestrator_health.healthy = True
        mock_intent_orchestrator.health_check.return_value = mock_orchestrator_health
        
        mock_indexing_health = Mock()
        mock_indexing_health.healthy = True
        mock_auto_indexing_service.health_check.return_value = mock_indexing_health
        
        is_healthy = await intent_bridge._check_health()
        assert is_healthy is True

    async def test_health_check_unhealthy_orchestrator(self, intent_bridge, mock_intent_orchestrator):
        """Test health check when orchestrator is unhealthy."""
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        
        mock_orchestrator_health = Mock()
        mock_orchestrator_health.healthy = False
        mock_intent_orchestrator.health_check.return_value = mock_orchestrator_health
        
        is_healthy = await intent_bridge._check_health()
        assert is_healthy is False

    async def test_process_intent_success(self, intent_bridge, mock_intent_orchestrator):
        """Test successful intent processing."""
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        
        expected_result = IntentResult(
            success=True,
            data={"message": "Found functions"},
            metadata={"intent_type": "SEARCH"},
            strategy_used="simple_search",
        )
        mock_intent_orchestrator.process_intent.return_value = expected_result
        
        result = await intent_bridge.process_intent("find auth functions", {})
        
        assert result == expected_result
        mock_intent_orchestrator.process_intent.assert_called_once_with("find auth functions", {})

    async def test_process_intent_not_initialized(self, intent_bridge):
        """Test intent processing when bridge is not initialized."""
        intent_bridge.bridge_initialized = False
        
        with pytest.raises(ServiceIntegrationError, match="Intent service bridge not properly initialized"):
            await intent_bridge.process_intent("test intent", {})

    async def test_get_intent_capabilities_success(self, intent_bridge, mock_intent_orchestrator):
        """Test getting intent capabilities."""
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        intent_bridge.auto_indexing_service = Mock()
        
        mock_orchestrator_capabilities = {
            "intent_types": ["SEARCH", "UNDERSTAND"],
            "parser_type": "pattern",
        }
        mock_intent_orchestrator.get_capabilities.return_value = mock_orchestrator_capabilities
        
        capabilities = await intent_bridge.get_intent_capabilities()
        
        assert capabilities["available"] is True
        assert capabilities["intent_types"] == ["SEARCH", "UNDERSTAND", "ANALYZE"]
        assert capabilities["background_indexing"] is True
        assert "intent_types" in capabilities

    async def test_get_intent_capabilities_not_initialized(self, intent_bridge):
        """Test getting capabilities when not initialized."""
        intent_bridge.bridge_initialized = False
        
        capabilities = await intent_bridge.get_intent_capabilities()
        
        assert capabilities["available"] is False
        assert "Intent service bridge not initialized" in capabilities["error"]

    async def test_inject_services_into_context(self, intent_bridge, mock_intent_orchestrator, mock_auto_indexing_service):
        """Test service injection into FastMCP context."""
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        intent_bridge.auto_indexing_service = mock_auto_indexing_service
        
        context = {}
        
        with patch.object(intent_bridge, 'get_intent_capabilities') as mock_capabilities:
            mock_capabilities.return_value = {"available": True}
            
            await intent_bridge.inject_services_into_context(context)
        
        assert context["intent_orchestrator"] == mock_intent_orchestrator
        assert context["auto_indexing_service"] == mock_auto_indexing_service
        assert context["intent_bridge"] == intent_bridge
        assert "intent_capabilities" in context

    async def test_trigger_background_indexing_success(self, intent_bridge, mock_auto_indexing_service):
        """Test successful background indexing trigger."""
        intent_bridge.auto_indexing_service = mock_auto_indexing_service
        mock_auto_indexing_service.trigger_indexing.return_value = True
        
        result = await intent_bridge.trigger_background_indexing("/test/path")
        
        assert result is True
        mock_auto_indexing_service.trigger_indexing.assert_called_once_with("/test/path")

    async def test_trigger_background_indexing_not_available(self, intent_bridge):
        """Test background indexing when service not available."""
        intent_bridge.auto_indexing_service = None
        
        result = await intent_bridge.trigger_background_indexing("/test/path")
        
        assert result is False

    async def test_get_bridge_status(self, intent_bridge, mock_intent_orchestrator, mock_auto_indexing_service):
        """Test getting comprehensive bridge status."""
        intent_bridge.bridge_initialized = True
        intent_bridge.intent_orchestrator = mock_intent_orchestrator
        intent_bridge.auto_indexing_service = mock_auto_indexing_service
        
        # Mock health responses
        mock_orchestrator_health = Mock()
        mock_orchestrator_health.healthy = True
        mock_orchestrator_health.message = "Healthy"
        mock_intent_orchestrator.health_check.return_value = mock_orchestrator_health
        
        mock_indexing_health = Mock()
        mock_indexing_health.healthy = True
        mock_indexing_health.message = "Healthy"
        mock_auto_indexing_service.health_check.return_value = mock_indexing_health
        
        with patch.object(intent_bridge, '_check_health') as mock_health:
            mock_health.return_value = True
            
            status = await intent_bridge.get_bridge_status()
        
        assert status["bridge_initialized"] is True
        assert status["intent_orchestrator_available"] is True
        assert status["auto_indexing_available"] is True
        assert status["bridge_healthy"] is True
        assert status["intent_orchestrator_health"]["healthy"] is True
        assert status["auto_indexing_health"]["healthy"] is True

    async def test_service_discovery_methods(self, intent_bridge, mock_services_manager):
        """Test service discovery methods."""
        # Test intent orchestrator discovery
        mock_orchestrator = Mock()
        mock_services_manager.get_service.return_value = mock_orchestrator
        
        orchestrator = await intent_bridge._get_intent_orchestrator()
        assert orchestrator == mock_orchestrator
        mock_services_manager.get_service.assert_called_with("intent_orchestrator")
        
        # Test auto-indexing service discovery
        mock_services_manager.reset_mock()
        mock_indexing = Mock()
        mock_services_manager.get_service.return_value = mock_indexing
        
        indexing = await intent_bridge._get_auto_indexing_service()
        assert indexing == mock_indexing
        mock_services_manager.get_service.assert_called_with("auto_indexing")

    async def test_service_discovery_no_services_manager(self, intent_bridge):
        """Test service discovery when services manager is not available."""
        intent_bridge.services_manager = None
        
        orchestrator = await intent_bridge._get_intent_orchestrator()
        assert orchestrator is None
        
        indexing = await intent_bridge._get_auto_indexing_service()
        assert indexing is None


@pytest.mark.integration
class TestIntentServiceBridgeIntegration:
    """Integration tests for intent service bridge."""
    
    async def test_full_bridge_workflow(self):
        """Test complete bridge workflow with mocked services."""
        from codeweaver.services.manager import ServicesManager
        from codeweaver.types import ServicesConfig
        
        # This would require a more complex setup with actual services
        # For now, we'll keep it as a placeholder for future integration tests
        pass