"""Integration tests for Intent Layer components."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from codeweaver.services.manager import ServicesManager
from codeweaver.types import (
    ServicesConfig,
    IntentServiceConfig,
    AutoIndexingConfig,
    ServiceType,
    IntentResult,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestIntentLayerIntegration:
    """Integration tests for the complete intent layer."""

    @pytest.fixture
    def services_config(self):
        """Services configuration with intent services enabled."""
        return ServicesConfig(
            intent=IntentServiceConfig(
                enabled=True,
                provider="intent_orchestrator",
                default_strategy="adaptive", 
                confidence_threshold=0.6,
            ),
            auto_indexing=AutoIndexingConfig(
                enabled=True,
                provider="auto_indexing",
                watch_patterns=["**/*.py", "**/*.js"],
            ),
        )

    @pytest.fixture
    async def services_manager(self, services_config):
        """Services manager with intent services configured."""
        manager = ServicesManager(services_config)
        
        # Mock the FastMCP server for middleware registration
        mock_server = Mock()
        manager._fastmcp_server = mock_server
        
        return manager

    async def test_intent_services_registration(self, services_manager):
        """Test that intent services are properly registered."""
        # Initialize services manager (this registers providers)
        await services_manager.initialize()
        
        # Check that intent services are registered
        registry = services_manager._registry
        
        # Check provider registration
        assert registry.has_provider(ServiceType.INTENT, "intent_orchestrator")
        assert registry.has_provider(ServiceType.AUTO_INDEXING, "auto_indexing")
        assert registry.has_provider(ServiceType.INTENT, "intent_bridge")

    async def test_intent_services_creation(self, services_manager):
        """Test that intent services are created and initialized."""
        await services_manager.initialize()
        
        # Check that services are created
        intent_orchestrator = services_manager.get_service(ServiceType.INTENT)
        auto_indexing = services_manager.get_service(ServiceType.AUTO_INDEXING)
        intent_bridge = await services_manager.get_service("intent_bridge")
        
        assert intent_orchestrator is not None
        assert auto_indexing is not None
        assert intent_bridge is not None
        
        # Check that services are properly initialized
        from codeweaver.services.providers.intent_orchestrator import IntentOrchestrator
        from codeweaver.services.providers.auto_indexing import AutoIndexingService
        from codeweaver.intent.middleware.intent_bridge import IntentServiceBridge
        
        assert isinstance(intent_orchestrator, IntentOrchestrator)
        assert isinstance(auto_indexing, AutoIndexingService)
        assert isinstance(intent_bridge, IntentServiceBridge)

    async def test_intent_bridge_integration(self, services_manager):
        """Test intent bridge integration with services manager."""
        await services_manager.initialize()
        
        intent_bridge = await services_manager.get_service("intent_bridge")
        assert intent_bridge is not None
        
        # Test that bridge can access orchestrator through services manager
        orchestrator = await intent_bridge._get_intent_orchestrator()
        assert orchestrator is not None
        
        # Test that bridge can access auto-indexing through services manager
        auto_indexing = await intent_bridge._get_auto_indexing_service()
        assert auto_indexing is not None

    @patch('codeweaver.intent.parsing.factory.IntentParserFactory.create')
    async def test_end_to_end_intent_processing(self, mock_parser_factory, services_manager):
        """Test end-to-end intent processing through the bridge."""
        # Setup mock parser
        mock_parser = Mock()
        mock_parsed_intent = Mock()
        mock_parsed_intent.intent_type = "SEARCH"
        mock_parsed_intent.primary_target = "authentication"
        mock_parsed_intent.scope = "PROJECT"
        mock_parsed_intent.complexity = "MODERATE"
        mock_parsed_intent.confidence = 0.8
        mock_parsed_intent.filters = {}
        mock_parsed_intent.metadata = {"parser": "pattern"}
        
        mock_parser.parse = AsyncMock(return_value=mock_parsed_intent)
        mock_parser_factory.return_value = mock_parser
        
        await services_manager.initialize()
        
        # Get intent bridge
        intent_bridge = await services_manager.get_service("intent_bridge")
        
        # Process an intent
        result = await intent_bridge.process_intent("find authentication functions", {})
        
        # Verify result
        assert isinstance(result, IntentResult)
        assert result.success is True
        # Basic fallback should be used since no strategy registry is set up
        assert result.strategy_used == "basic_fallback"

    async def test_intent_capabilities_integration(self, services_manager):
        """Test intent capabilities through the integrated system."""
        await services_manager.initialize()
        
        intent_bridge = await services_manager.get_service("intent_bridge")
        capabilities = await intent_bridge.get_intent_capabilities()
        
        assert capabilities["available"] is True
        assert "SEARCH" in capabilities["intent_types"]
        assert "UNDERSTAND" in capabilities["intent_types"] 
        assert "ANALYZE" in capabilities["intent_types"]
        assert "INDEX" not in capabilities["intent_types"]  # Should not be exposed

    async def test_background_indexing_integration(self, services_manager):
        """Test background indexing service integration."""
        await services_manager.initialize()
        
        auto_indexing = services_manager.get_service(ServiceType.AUTO_INDEXING)
        assert auto_indexing is not None
        
        # Test trigger indexing method exists
        assert hasattr(auto_indexing, 'trigger_indexing')
        
        # Test through intent bridge
        intent_bridge = await services_manager.get_service("intent_bridge")
        
        with patch.object(auto_indexing, 'trigger_indexing') as mock_trigger:
            mock_trigger.return_value = True
            
            success = await intent_bridge.trigger_background_indexing("/test/path")
            assert success is True
            mock_trigger.assert_called_once_with("/test/path")

    async def test_services_health_integration(self, services_manager):
        """Test health monitoring integration for intent services."""
        await services_manager.initialize()
        
        # Get health report
        health_report = await services_manager.get_health_report()
        
        # Check that intent services are included in health report
        service_types = [service_health.service_type for service_health in health_report.services]
        
        assert ServiceType.INTENT in service_types
        assert ServiceType.AUTO_INDEXING in service_types

    async def test_service_dependency_injection(self, services_manager):
        """Test that intent services can access other services."""
        await services_manager.initialize()
        
        intent_orchestrator = services_manager.get_service(ServiceType.INTENT)
        
        # Check that orchestrator can potentially access cache service
        # (even if None, the method should exist)
        cache_service = await intent_orchestrator._get_cache_service()
        # Cache service might be None in test environment, that's OK
        assert cache_service is None or hasattr(cache_service, 'get')

    async def test_intent_service_shutdown(self, services_manager):
        """Test proper shutdown of intent services."""
        await services_manager.initialize()
        
        # Ensure services are running
        intent_bridge = await services_manager.get_service("intent_bridge")
        assert intent_bridge.bridge_initialized is True
        
        # Shutdown services
        await services_manager.shutdown()
        
        # Check that intent bridge is properly shut down
        assert intent_bridge.bridge_initialized is False


@pytest.mark.integration 
@pytest.mark.asyncio
class TestServerIntentIntegration:
    """Integration tests for intent layer with CodeWeaver server."""

    async def test_server_tool_registration(self):
        """Test that intent MCP tools are registered with server."""
        from codeweaver.server import CodeWeaverServer
        from codeweaver.config import CodeWeaverConfig
        
        # Create minimal config
        config = CodeWeaverConfig()
        config.services.intent.enabled = True
        
        server = CodeWeaverServer(config)
        
        # Mock FastMCP server
        mock_mcp_server = Mock()
        registered_tools = []
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools.append(func.__name__)
                return func
            return decorator
        
        mock_mcp_server.tool = mock_tool_decorator
        server.mcp = mock_mcp_server
        
        # Initialize server (this should register tools)
        with patch('codeweaver.server.CodeWeaverServer._setup_components'):
            with patch('codeweaver.server.CodeWeaverServer._setup_domain_middleware'):
                server._register_tools()
        
        # Check that intent tools are registered
        assert any("process_intent" in tool for tool in registered_tools)
        assert any("get_intent_capabilities" in tool for tool in registered_tools)

    @patch('codeweaver.server.CodeWeaverServer._get_intent_bridge')
    async def test_server_intent_handlers(self, mock_get_bridge):
        """Test server intent handler methods."""
        from codeweaver.server import CodeWeaverServer
        from codeweaver.config import CodeWeaverConfig
        
        config = CodeWeaverConfig()
        server = CodeWeaverServer(config)
        
        # Mock intent bridge
        mock_bridge = Mock()
        mock_result = IntentResult(
            success=True,
            data={"found": "functions"},
            metadata={"intent_type": "SEARCH"},
            strategy_used="simple_search",
        )
        mock_bridge.process_intent = AsyncMock(return_value=mock_result)
        mock_bridge.get_intent_capabilities = AsyncMock(return_value={
            "available": True,
            "intent_types": ["SEARCH", "UNDERSTAND", "ANALYZE"],
        })
        
        mock_get_bridge.return_value = mock_bridge
        
        # Test process intent handler
        ctx = Mock()
        result = await server._process_intent_handler(ctx, "find auth functions", {})
        
        assert result["success"] is True
        assert result["data"]["found"] == "functions"
        assert result["metadata"]["intent_type"] == "SEARCH"
        
        # Test capabilities handler
        capabilities = await server._get_intent_capabilities_handler(ctx)
        
        assert capabilities["available"] is True
        assert "SEARCH" in capabilities["intent_types"]

    async def test_server_fallback_behavior(self):
        """Test server behavior when intent services are not available."""
        from codeweaver.server import CodeWeaverServer
        from codeweaver.config import CodeWeaverConfig
        
        config = CodeWeaverConfig()
        server = CodeWeaverServer(config)
        
        # Mock missing intent bridge
        with patch.object(server, '_get_intent_bridge') as mock_get_bridge:
            mock_get_bridge.return_value = None
            
            ctx = Mock()
            result = await server._process_intent_handler(ctx, "test intent", {})
            
            assert result["success"] is False
            assert "Intent service not available" in result["error"]
            assert "fallback_suggestion" in result["metadata"]
            assert len(result["suggestions"]) > 0


@pytest.mark.slow
@pytest.mark.integration
class TestIntentLayerPerformance:
    """Performance tests for intent layer."""

    async def test_intent_processing_performance(self, services_manager):
        """Test performance of intent processing."""
        import time
        
        await services_manager.initialize()
        
        intent_bridge = await services_manager.get_service("intent_bridge")
        
        # Measure processing time for multiple intents
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            tasks.append(intent_bridge.process_intent(f"find function {i}", {}))
        
        # Process all intents
        import asyncio
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        
        # Should process intents reasonably quickly (less than 1 second each on average)
        assert avg_time < 1.0, f"Average intent processing time too slow: {avg_time}s"

    async def test_concurrent_intent_processing(self, services_manager):
        """Test concurrent intent processing doesn't cause issues."""
        await services_manager.initialize()
        
        intent_bridge = await services_manager.get_service("intent_bridge")
        
        # Process many intents concurrently
        import asyncio
        tasks = [
            intent_bridge.process_intent(f"concurrent intent {i}", {})
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully (no exceptions)
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"
            if hasattr(result, 'success'):
                # It's an IntentResult - success or failure is OK
                assert isinstance(result.success, bool)