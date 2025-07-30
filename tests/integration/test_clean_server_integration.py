# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration tests for CodeWeaverServer implementation.

Tests the new clean server implementation that uses the plugin system
and FastMCP middleware to ensure it works correctly with the refactored architecture.
"""

import asyncio

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeweaver.config import get_config
from codeweaver.server import CodeWeaverServer
from codeweaver.types import ExtensibilityConfig


class TestCodeWeaverServer:
    """Test CodeWeaverServer functionality."""

    def test_server_initialization(self) -> None:
        """Test server initializes correctly."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        assert server.config is not None
        assert server.mcp is not None
        assert server.extensibility_manager is not None
        assert server._initialized is False

    def test_server_initialization_with_extensibility_config(self) -> None:
        """Test server initialization with custom extensibility config."""
        config = get_config()
        extensibility_config = ExtensibilityConfig()

        server = CodeWeaverServer(config=config, extensibility_config=extensibility_config)

        assert server.config is not None
        assert server.extensibility_manager is not None

    @pytest.mark.asyncio
    async def test_server_initialization_process(self) -> None:
        """Test the server initialization process."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock the extensibility manager methods to avoid external dependencies
        with (
            patch.object(
                server.extensibility_manager, "initialize", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                server.extensibility_manager, "get_backend", new_callable=AsyncMock
            ) as mock_backend,
            patch.object(
                server.extensibility_manager, "get_embedding_provider", new_callable=AsyncMock
            ) as mock_embedding,
            patch.object(
                server.extensibility_manager, "get_reranking_provider", new_callable=AsyncMock
            ) as mock_reranking,
            patch.object(
                server.extensibility_manager, "get_data_sources", new_callable=AsyncMock
            ) as mock_data_sources,
        ):
            # Set up mock returns
            mock_data_sources.return_value = []

            await server.initialize()

            # Should have called extensibility manager initialization
            mock_init.assert_called_once()
            mock_backend.assert_called_once()
            mock_embedding.assert_called_once()
            mock_reranking.assert_called_once()
            mock_data_sources.assert_called_once()

            # Server should be marked as initialized
            assert server._initialized is True

    def test_double_initialization_warning(self) -> None:
        """Test that double initialization shows a warning."""
        config = get_config()
        server = CodeWeaverServer(config=config)
        server._initialized = True

        # This should complete without error but may log a warning
        # In a real test, we'd capture the log output
        asyncio.run(server.initialize())

    @pytest.mark.asyncio
    async def test_middleware_setup(self) -> None:
        """Test that middleware is set up correctly."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock the extensibility manager methods to avoid external dependencies
        with (
            patch.object(server.extensibility_manager, "initialize", new_callable=AsyncMock),
            patch.object(server.extensibility_manager, "get_backend", new_callable=AsyncMock),
            patch.object(
                server.extensibility_manager, "get_embedding_provider", new_callable=AsyncMock
            ),
            patch.object(
                server.extensibility_manager, "get_reranking_provider", new_callable=AsyncMock
            ),
            patch.object(
                server.extensibility_manager, "get_data_sources", new_callable=AsyncMock
            ) as mock_data_sources,
        ):
            # Set up mock returns
            mock_data_sources.return_value = []

            await server.initialize()

            # Check that services manager was created and initialized
            assert server.services_manager is not None
            assert server.services_manager._initialized

            # Check that core services are available
            chunking_service = server.services_manager.get_chunking_service()
            filtering_service = server.services_manager.get_filtering_service()
            assert chunking_service is not None
            assert filtering_service is not None

    @pytest.mark.asyncio
    async def test_component_initialization(self) -> None:
        """Test that plugin system components are initialized."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock the extensibility manager methods
        mock_backend = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_reranking_provider = MagicMock()
        MagicMock()
        mock_data_sources = [MagicMock()]

        server.extensibility_manager.get_backend = AsyncMock(return_value=mock_backend)
        server.extensibility_manager.get_embedding_provider = AsyncMock(
            return_value=mock_embedding_provider
        )
        server.extensibility_manager.get_reranking_provider = AsyncMock(
            return_value=mock_reranking_provider
        )
        server.extensibility_manager.get_data_sources = AsyncMock(return_value=mock_data_sources)
        server.extensibility_manager.initialize = AsyncMock()

        # Mock backend methods to avoid actual collection operations
        mock_backend.list_collections = AsyncMock(return_value=["existing_collection"])
        mock_backend.create_collection = AsyncMock()

        await server.initialize()

        # Check that components were initialized
        assert server._components["backend"] == mock_backend
        assert server._components["embedding_provider"] == mock_embedding_provider
        assert server._components["reranking_provider"] == mock_reranking_provider
        # Note: rate_limiter component removed - handled by middleware

    @pytest.mark.asyncio
    async def test_collection_creation(self) -> None:
        """Test that vector collection is created if it doesn't exist."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock components
        mock_backend = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.dimension = 1024

        # Collection doesn't exist
        mock_backend.list_collections = AsyncMock(return_value=[])
        mock_backend.create_collection = AsyncMock()

        server._components = {
            "backend": mock_backend,
            "embedding_provider": mock_embedding_provider,
        }

        await server._ensure_collection()

        # Should have created the collection
        mock_backend.create_collection.assert_called_once_with(
            name=config.backend.collection_name, dimension=1024, distance_metric="cosine"
        )

    @pytest.mark.asyncio
    async def test_collection_exists_no_creation(self) -> None:
        """Test that existing collection is not recreated."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock components
        mock_backend = MagicMock()
        mock_embedding_provider = MagicMock()

        # Collection already exists
        mock_backend.list_collections = AsyncMock(return_value=[config.backend.collection_name])
        mock_backend.create_collection = AsyncMock()

        server._components = {
            "backend": mock_backend,
            "embedding_provider": mock_embedding_provider,
        }

        await server._ensure_collection()

        # Should not have created the collection
        mock_backend.create_collection.assert_not_called()


class TestServerToolRegistration:
    """Test that MCP tools are registered correctly."""

    @pytest.mark.asyncio
    async def test_tools_registered(self) -> None:
        """Test that MCP tools are registered during initialization."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock extensibility manager methods to avoid external dependencies
        with (
            patch.object(server.extensibility_manager, "initialize", new_callable=AsyncMock),
            patch.object(server.extensibility_manager, "get_backend", new_callable=AsyncMock),
            patch.object(
                server.extensibility_manager, "get_embedding_provider", new_callable=AsyncMock
            ),
            patch.object(
                server.extensibility_manager, "get_reranking_provider", new_callable=AsyncMock
            ),
            patch.object(
                server.extensibility_manager, "get_data_sources", new_callable=AsyncMock
            ) as mock_data_sources,
        ):
            # Set up mock returns
            mock_data_sources.return_value = []

            # Mock the _ensure_collection to avoid backend operations
            with patch.object(server, "_ensure_collection", new_callable=AsyncMock):
                await server.initialize()

                # Check that the FastMCP server has tools registered
                # This is a bit tricky to test directly, but we can check that
                # the server completed initialization without error
                assert server._initialized is True

    def test_tool_registration_creates_functions(self) -> None:
        # sourcery skip: remove-assert-true
        """Test that tool registration creates the expected functions."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Call the registration method directly
        server._register_tools()

        # The tools should be registered with the FastMCP server
        # This is primarily testing that no exceptions are raised during registration
        assert True  # If we get here, registration succeeded


class TestServerBehaviorIntegration:
    """Test server behavior with mock components."""

    @pytest.mark.asyncio
    async def test_server_with_mock_components(self) -> None:
        """Test server behavior using mock components."""
        # This test demonstrates how the server would work in a real scenario
        # but uses mocks to avoid external dependencies

        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock all the dependencies
        mock_backend = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_reranking_provider = MagicMock()
        MagicMock()

        # Mock filesystem source with proper interface
        mock_filesystem_source = MagicMock()
        mock_filesystem_source.provider.value = "filesystem"

        # Set up the extensibility manager mocks
        server.extensibility_manager.initialize = AsyncMock()
        server.extensibility_manager.get_backend = AsyncMock(return_value=mock_backend)
        server.extensibility_manager.get_embedding_provider = AsyncMock(
            return_value=mock_embedding_provider
        )
        server.extensibility_manager.get_reranking_provider = AsyncMock(
            return_value=mock_reranking_provider
        )
        server.extensibility_manager.get_data_sources = AsyncMock(
            return_value=[mock_filesystem_source]
        )

        # Mock backend operations
        mock_backend.list_collections = AsyncMock(return_value=[config.backend.collection_name])
        mock_backend.create_collection = AsyncMock()

        # Initialize the server
        await server.initialize()

        # Verify the server is properly initialized
        assert server._initialized is True
        assert server._components["backend"] == mock_backend
        assert server._components["embedding_provider"] == mock_embedding_provider
        assert server._components["filesystem_source"] == mock_filesystem_source

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self) -> None:
        """Test that initialization errors are handled properly."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock extensibility manager to raise an error
        server.extensibility_manager.initialize = AsyncMock(side_effect=Exception("Test error"))

        # Initialization should raise the error
        with pytest.raises(Exception, match="Test error"):
            await server.initialize()

        # Server should not be marked as initialized
        assert server._initialized is False

    @pytest.mark.asyncio
    async def test_collection_creation_error_handling(self) -> None:
        """Test error handling during collection creation."""
        config = get_config()
        server = CodeWeaverServer(config=config)

        # Mock components with failing backend
        mock_backend = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.dimension = 1024

        mock_backend.list_collections = AsyncMock(side_effect=Exception("Backend error"))

        server._components = {
            "backend": mock_backend,
            "embedding_provider": mock_embedding_provider,
        }

        # Should raise the backend error
        with pytest.raises(Exception, match="Backend error"):
            await server._ensure_collection()


class TestServerConfiguration:
    """Test server configuration handling."""

    def test_server_uses_provided_config(self) -> None:
        """Test that server uses the provided configuration."""
        config = get_config()
        # Modify some config values to test
        original_collection_name = config.backend.collection_name
        config.backend.collection_name = "test_collection"

        server = CodeWeaverServer(config=config)

        assert server.config.backend.collection_name == "test_collection"

        # Reset for other tests
        config.backend.collection_name = original_collection_name

    def test_server_uses_default_config_when_none_provided(self) -> None:
        """Test that server uses default config when none provided."""
        server = CodeWeaverServer()

        # Should have loaded the default config
        assert server.config is not None
        assert hasattr(server.config, "backend")
        assert hasattr(server.config, "server")

    def test_extensibility_config_passed_to_manager(self) -> None:
        """Test that extensibility config is passed to the manager."""
        config = get_config()
        extensibility_config = ExtensibilityConfig()

        server = CodeWeaverServer(config=config, extensibility_config=extensibility_config)

        # The extensibility manager should have received the config
        # This is tested indirectly by ensuring the server initializes properly
        assert server.extensibility_manager is not None


# Integration test that would require real components (commented out for now)
"""
class TestFullServerIntegration:
    '''Test full server integration with real components.'''

    @pytest.mark.integration  # Mark as integration test
    @pytest.mark.asyncio
    async def test_full_server_workflow(self):
        '''Test complete server workflow with real components.'''
        # This would test the actual server with real backends, providers, etc.
        # Requires proper environment setup and external dependencies

        config = get_config()
        server = CodeWeaverServer(config=config)

        # Would need real API keys and services
        await server.initialize()

        # Test actual indexing and searching
        # This is what the existing test_server_functionality.py does
        pass
"""
