# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Integration tests for FastMCP middleware integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeweaver._types.service_config import ServicesConfig
from codeweaver.config import CodeWeaverConfig
from codeweaver.server import CodeWeaverServer
from codeweaver.services.manager import ServicesManager


class TestFastMCPMiddlewareIntegration:
    """Test the complete FastMCP middleware integration."""

    @pytest.fixture
    def mock_fastmcp_server(self):
        """Create a mock FastMCP server."""
        mock = MagicMock()
        mock.add_middleware = MagicMock()
        return mock

    @pytest.fixture
    def services_config(self):
        """Create a test services configuration."""
        return ServicesConfig(
            # Enable all middleware services
            logging={"enabled": True, "provider": "fastmcp_logging"},
            timing={"enabled": True, "provider": "fastmcp_timing"},
            error_handling={"enabled": True, "provider": "fastmcp_error_handling"},
            rate_limiting={"enabled": True, "provider": "fastmcp_rate_limiting"},
            # Core services
            chunking={"enabled": True, "provider": "fastmcp_chunking"},
            filtering={"enabled": True, "provider": "fastmcp_filtering"},
        )

    @pytest.mark.asyncio
    async def test_services_manager_middleware_integration(self, mock_fastmcp_server, services_config):
        """Test that ServicesManager properly integrates middleware services."""
        # Create services manager with mock FastMCP server
        services_manager = ServicesManager(
            config=services_config,
            fastmcp_server=mock_fastmcp_server
        )

        # Initialize services manager
        await services_manager.initialize()

        try:
            # Verify middleware services were created
            middleware_services = services_manager.list_middleware_services()
            assert len(middleware_services) > 0

            # Check specific middleware services
            logging_service = services_manager.get_logging_service()
            assert logging_service is not None
            assert hasattr(logging_service, 'get_middleware_instance')

            timing_service = services_manager.get_timing_service()
            assert timing_service is not None
            assert hasattr(timing_service, 'get_middleware_instance')

            error_handling_service = services_manager.get_error_handling_service()
            assert error_handling_service is not None
            assert hasattr(error_handling_service, 'get_middleware_instance')

            rate_limiting_service = services_manager.get_rate_limiting_service()
            assert rate_limiting_service is not None
            assert hasattr(rate_limiting_service, 'get_middleware_instance')

            # Verify middleware was registered with FastMCP server
            assert mock_fastmcp_server.add_middleware.call_count >= 4

        finally:
            # Clean up
            await services_manager.shutdown()

    @pytest.mark.asyncio
    async def test_server_services_manager_integration(self):
        """Test that CodeWeaverServer properly uses ServicesManager."""
        # Create a minimal config for testing
        config = CodeWeaverConfig(
            services=ServicesConfig(
                logging={"enabled": True, "provider": "fastmcp_logging"},
                timing={"enabled": True, "provider": "fastmcp_timing"},
            )
        )

        # Mock the extensibility manager to avoid complex initialization
        with patch('codeweaver.server.ExtensibilityManager') as mock_ext_manager:
            mock_ext_manager.return_value.initialize = AsyncMock()
            mock_ext_manager.return_value.get_backend = AsyncMock(return_value=MagicMock())
            mock_ext_manager.return_value.get_embedding_provider = AsyncMock(return_value=MagicMock())
            mock_ext_manager.return_value.get_reranking_provider = AsyncMock(return_value=MagicMock())
            mock_ext_manager.return_value.get_data_sources = AsyncMock(return_value=[])
            mock_ext_manager.return_value.get_component_info = MagicMock(return_value={})

            # Create server
            server = CodeWeaverServer(config=config)

            try:
                # Initialize server (this should create and initialize ServicesManager)
                await server.initialize()

                # Verify services manager was created
                assert server.services_manager is not None
                assert isinstance(server.services_manager, ServicesManager)

                # Verify middleware services are available
                middleware_services = server.services_manager.list_middleware_services()
                assert len(middleware_services) > 0

            finally:
                # Clean up
                await server.shutdown()



    @pytest.mark.asyncio
    async def test_middleware_service_health_checks(self, mock_fastmcp_server, services_config):
        """Test that middleware services support health checks."""
        services_manager = ServicesManager(
            config=services_config,
            fastmcp_server=mock_fastmcp_server
        )

        await services_manager.initialize()

        try:
            # Get health report
            health_report = await services_manager.get_health_report()

            # Verify health report includes middleware services
            assert health_report.overall_status is not None

            # Check that middleware services are included in health monitoring
            middleware_services = services_manager.list_middleware_services()
            for service in middleware_services.values():
                # Each service should support health checks
                health = await service.health_check()
                assert health is not None
                assert hasattr(health, 'status')

        finally:
            await services_manager.shutdown()

    def test_middleware_service_configuration_validation(self):
        """Test that middleware service configurations are properly validated."""
        # Test valid configuration
        valid_config = ServicesConfig(
            logging={"enabled": True, "provider": "fastmcp_logging", "log_level": "INFO"},
            timing={"enabled": True, "provider": "fastmcp_timing"},
            error_handling={"enabled": True, "provider": "fastmcp_error_handling"},
            rate_limiting={"enabled": True, "provider": "fastmcp_rate_limiting", "max_requests_per_second": 1.0},
        )

        # Configuration should be created successfully
        assert valid_config.logging.enabled is True
        assert valid_config.timing.enabled is True
        assert valid_config.error_handling.enabled is True
        assert valid_config.rate_limiting.enabled is True

        # Verify provider names
        assert valid_config.logging.provider == "fastmcp_logging"
        assert valid_config.timing.provider == "fastmcp_timing"
        assert valid_config.error_handling.provider == "fastmcp_error_handling"
        assert valid_config.rate_limiting.provider == "fastmcp_rate_limiting"

    @pytest.mark.asyncio
    async def test_middleware_service_lifecycle(self, mock_fastmcp_server):
        """Test middleware service lifecycle management."""
        config = ServicesConfig(
            logging={"enabled": True, "provider": "fastmcp_logging"},
        )

        services_manager = ServicesManager(
            config=config,
            fastmcp_server=mock_fastmcp_server
        )

        # Test initialization
        await services_manager.initialize()

        # Verify service was created
        logging_service = services_manager.get_logging_service()
        assert logging_service is not None

        # Test service functionality
        await logging_service.log_request("test_method", {"param": "value"})
        await logging_service.log_response("test_method", {"result": "success"}, 0.1)

        # Test shutdown
        await services_manager.shutdown()

        # Verify services manager is properly shut down
        assert not services_manager._initialized


if __name__ == "__main__":
    pytest.main([__file__])
