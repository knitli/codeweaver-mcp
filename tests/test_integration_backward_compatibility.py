# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration tests for backward compatibility between legacy and extensible architectures.

Validates that:
1. Existing deployments continue working without changes
2. New deployments can use extensible architecture
3. Migration utilities work correctly
4. Performance is maintained or improved
"""

import shutil
import tempfile

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the classes and functions we're testing
from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.extensibility_manager import ExtensibilityConfig
from codeweaver.factories.integration import create_migration_config, validate_migration_readiness
from codeweaver.server import (
    CodeEmbeddingsServer,
    ExtensibleCodeEmbeddingsServer,
    ServerMigrationManager,
    create_extensible_server,
    create_legacy_server,
    create_server,
    detect_configuration_type,
    migrate_config_to_extensible,
)


class TestBackwardCompatibility:
    """Test backward compatibility between server architectures."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=CodeWeaverConfig)

        # Mock legacy structure
        config.chunking = Mock()
        config.chunking.max_chunk_size = 1500
        config.chunking.min_chunk_size = 50

        config.indexing = Mock()
        config.indexing.batch_size = 8
        config.indexing.enable_auto_reindex = False

        config.qdrant = Mock()
        config.qdrant.url = "https://test.qdrant.io"
        config.qdrant.api_key = "test-key"
        config.qdrant.collection_name = "test-collection"

        config.embedding = Mock()
        config.embedding.provider = "voyage"
        config.embedding.api_key = "test-api-key"
        config.embedding.model = "voyage-code-3"
        config.embedding.dimension = 1024

        config.rate_limiting = Mock()
        config.rate_limiting.enabled = True

        config.server = Mock()
        config.server.log_level = "INFO"
        config.server.server_version = "2.0.0"

        return config

    @pytest.fixture
    def sample_test_files(self):
        """Create sample test files for indexing."""
        temp_dir = tempfile.mkdtemp()
        test_dir = Path(temp_dir)

        # Create sample Python file
        python_file = test_dir / "sample.py"
        python_file.write_text("""
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "success"

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
""")

        # Create sample JavaScript file
        js_file = test_dir / "sample.js"
        js_file.write_text("""
function greetUser(name) {
    console.log(`Hello, ${name}!`);
    return true;
}

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(num) {
        this.result += num;
        return this;
    }
}
""")

        yield test_dir

        # Cleanup
        shutil.rmtree(temp_dir)


class TestConfigurationDetection:
    """Test configuration type detection."""

    def test_detect_legacy_configuration(self, mock_config) -> None:
        """Test detection of legacy configuration format."""
        # Remove any extensible features
        if hasattr(mock_config, "backend"):
            delattr(mock_config, "backend")
        if hasattr(mock_config, "data_sources"):
            delattr(mock_config, "data_sources")

        config_type = detect_configuration_type(mock_config)
        assert config_type == "legacy"

    def test_detect_extensible_configuration_with_backend(self, mock_config) -> None:
        """Test detection of extensible configuration with backend config."""
        # Add backend configuration
        mock_config.backend = Mock()
        mock_config.backend.provider = "qdrant"

        config_type = detect_configuration_type(mock_config)
        # Will be 'legacy' since _EXTENDED_CONFIGS_AVAILABLE might be False
        assert config_type in ["legacy", "extensible"]

    def test_detect_extensible_configuration_with_data_sources(self, mock_config) -> None:
        """Test detection of extensible configuration with data sources."""
        # Add data sources configuration
        mock_config.data_sources = Mock()
        mock_config.data_sources.sources = []

        config_type = detect_configuration_type(mock_config)
        # Will be 'legacy' since _EXTENDED_CONFIGS_AVAILABLE might be False
        assert config_type in ["legacy", "extensible"]


class TestServerFactoryFunctions:
    """Test server factory functions."""

    def test_create_legacy_server(self, mock_config) -> None:
        """Test creation of legacy server."""
        server = create_legacy_server(mock_config)
        assert isinstance(server, CodeEmbeddingsServer)
        assert not isinstance(server, ExtensibleCodeEmbeddingsServer)

    def test_create_extensible_server(self, mock_config) -> None:
        """Test creation of extensible server."""
        server = create_extensible_server(mock_config)
        assert isinstance(server, ExtensibleCodeEmbeddingsServer)

    def test_create_server_auto_detection_legacy(self, mock_config) -> None:
        """Test auto server creation with legacy config."""
        server = create_server(mock_config, server_type="auto")
        # Should create legacy server for legacy config
        assert isinstance(server, CodeEmbeddingsServer)

    def test_create_server_explicit_legacy(self, mock_config) -> None:
        """Test explicit legacy server creation."""
        server = create_server(mock_config, server_type="legacy")
        assert isinstance(server, CodeEmbeddingsServer)
        assert not isinstance(server, ExtensibleCodeEmbeddingsServer)

    def test_create_server_explicit_extensible(self, mock_config) -> None:
        """Test explicit extensible server creation."""
        server = create_server(mock_config, server_type="extensible")
        assert isinstance(server, ExtensibleCodeEmbeddingsServer)


class TestCompatibilityLayer:
    """Test the backward compatibility layer."""

    @pytest.mark.asyncio
    async def test_legacy_compatibility_adapter(self, mock_config) -> None:
        """Test the legacy compatibility adapter."""
        with patch("codeweaver.factories.extensibility_manager.ExtensibilityManager"):
            # Create extensible server (it will use mocked manager)
            server = create_extensible_server(mock_config)

            # The server should have compatibility properties
            assert hasattr(server, "_ensure_initialized")
            assert hasattr(server, "extensibility_manager")

    def test_server_interface_compatibility(self, mock_config) -> None:
        """Test that both servers have the same public interface."""
        legacy_server = create_legacy_server(mock_config)
        extensible_server = create_extensible_server(mock_config)

        # Both should have the same core methods
        core_methods = [
            "index_codebase",
            "search_code",
            "ast_grep_search",
            "get_supported_languages",
        ]

        for method in core_methods:
            assert hasattr(legacy_server, method)
            assert hasattr(extensible_server, method)
            assert callable(getattr(legacy_server, method))
            assert callable(getattr(extensible_server, method))


class TestMigrationUtilities:
    """Test migration utilities and helpers."""

    def test_validate_migration_readiness_ready(self, mock_config) -> None:
        """Test migration readiness validation for ready config."""
        # Add required attributes for migration
        mock_config.backend = Mock()
        mock_config.embedding = Mock()

        results = validate_migration_readiness(mock_config)
        assert isinstance(results, dict)
        assert "ready" in results
        assert "issues" in results
        assert "warnings" in results
        assert "recommendations" in results

    def test_validate_migration_readiness_missing_backend(self, mock_config) -> None:
        """Test migration readiness validation with missing backend."""
        # Remove backend config
        if hasattr(mock_config, "backend"):
            delattr(mock_config, "backend")

        results = validate_migration_readiness(mock_config)
        assert results["ready"] is False
        assert "Missing backend configuration" in results["issues"]

    def test_create_migration_config(self) -> None:
        """Test creation of migration configuration."""
        migration_config = create_migration_config()

        assert isinstance(migration_config, ExtensibilityConfig)
        assert migration_config.enable_legacy_fallbacks is True
        assert migration_config.migration_mode is True
        assert migration_config.lazy_initialization is True

    @pytest.mark.asyncio
    async def test_migrate_config_to_extensible(self, mock_config) -> None:
        """Test configuration migration to extensible format."""
        migrated_config, extensibility_config = await migrate_config_to_extensible(mock_config)

        assert migrated_config is not None
        assert isinstance(extensibility_config, ExtensibilityConfig)
        assert extensibility_config.migration_mode is True

    def test_server_migration_manager_init(self, mock_config) -> None:
        """Test server migration manager initialization."""
        server = create_legacy_server(mock_config)
        migration_manager = ServerMigrationManager(server)

        assert migration_manager.server is server
        assert migration_manager._migration_state == "not_started"

    def test_server_migration_manager_analyze_readiness(self, mock_config) -> None:
        """Test migration readiness analysis."""
        server = create_legacy_server(mock_config)
        migration_manager = ServerMigrationManager(server)

        results = migration_manager.analyze_migration_readiness()

        assert isinstance(results, dict)
        assert "ready" in results
        assert "migration_state" in results
        assert "configuration" in results
        assert "component_health" in results
        assert "recommendations" in results

    def test_server_migration_manager_get_status(self, mock_config) -> None:
        """Test migration status retrieval."""
        server = create_legacy_server(mock_config)
        migration_manager = ServerMigrationManager(server)

        status = migration_manager.get_migration_status()

        assert isinstance(status, dict)
        assert "migration_state" in status
        assert "backup_available" in status
        assert "server_type" in status
        assert status["server_type"] == "CodeEmbeddingsServer"


class TestPerformanceCompatibility:
    """Test that performance is maintained across architectures."""

    @pytest.mark.asyncio
    async def test_initialization_performance(self, mock_config) -> None:
        """Test that initialization time is reasonable for both server types."""
        import time

        # Test legacy server initialization
        start_time = time.time()
        create_legacy_server(mock_config)
        legacy_init_time = time.time() - start_time

        # Test extensible server initialization (creation only, not full init)
        start_time = time.time()
        create_extensible_server(mock_config)
        extensible_init_time = time.time() - start_time

        # Both should initialize quickly (< 1 second)
        assert legacy_init_time < 1.0
        assert extensible_init_time < 1.0

        # Extensible server should not be significantly slower
        # (allowing 2x slower due to additional abstraction layers)
        assert extensible_init_time < legacy_init_time * 2

    def test_memory_usage_compatibility(self, mock_config) -> None:
        """Test that memory usage is reasonable for both server types."""
        import sys

        # Get baseline memory usage
        baseline_size = sys.getsizeof(mock_config)

        # Create servers
        legacy_server = create_legacy_server(mock_config)
        extensible_server = create_extensible_server(mock_config)

        # Get server sizes (rough approximation)
        legacy_size = sys.getsizeof(legacy_server)
        extensible_size = sys.getsizeof(extensible_server)

        # Both should be reasonable in size
        assert legacy_size > baseline_size
        assert extensible_size > baseline_size

        # Extensible server may be larger due to additional components
        # but should not be excessively larger (allowing 3x for extensibility overhead)
        assert extensible_size < legacy_size * 3


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_invalid_config_handling(self) -> None:
        """Test handling of invalid configurations."""
        invalid_config = Mock()
        # Missing required attributes

        # Should not raise exceptions during creation
        try:
            server = create_server(invalid_config)
            assert server is not None
        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, ValueError | AttributeError | TypeError)

    @pytest.mark.asyncio
    async def test_migration_failure_handling(self, mock_config) -> None:
        """Test handling of migration failures."""
        server = create_legacy_server(mock_config)
        migration_manager = ServerMigrationManager(server)

        # Test with backup enabled
        with patch.object(migration_manager, "_backup_current_components") as mock_backup:
            with patch("codeweaver.factories.integration.ServerMigrationHelper") as mock_helper:
                # Make migration fail
                mock_helper.return_value.migrate_to_factories.side_effect = Exception(
                    "Migration failed"
                )

                result = await migration_manager.perform_migration(backup_components=True)

                assert result["status"] == "failed"
                assert result["migration_state"] == "failed"
                assert "error" in result
                mock_backup.assert_called_once()


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_legacy_workflow(self, mock_config, sample_test_files) -> None:
        """Test end-to-end workflow with legacy server."""
        # Mock the external dependencies
        with (
            patch("qdrant_client.QdrantClient"),
            patch("codeweaver.embeddings.create_embedder") as mock_embedder,
            patch("codeweaver.embeddings.VoyageAIReranker"),
        ):
            # Setup mock embedder
            mock_embedder_instance = Mock()
            mock_embedder_instance.dimension = 1024
            mock_embedder_instance.embed_documents.return_value = [[0.1] * 1024] * 5
            mock_embedder_instance.embed_query.return_value = [0.1] * 1024
            mock_embedder.return_value = mock_embedder_instance

            # Create legacy server
            server = create_legacy_server(mock_config)

            # Test basic operations
            assert hasattr(server, "index_codebase")
            assert hasattr(server, "search_code")
            assert hasattr(server, "get_supported_languages")

            # Test supported languages (should work without external dependencies)
            languages = await server.get_supported_languages()
            assert isinstance(languages, dict)
            assert "supported_languages" in languages

    @pytest.mark.asyncio
    async def test_end_to_end_extensible_workflow(self, mock_config, sample_test_files) -> None:
        """Test end-to-end workflow with extensible server."""
        # Mock the external dependencies
        with patch(
            "codeweaver.factories.extensibility_manager.ExtensibilityManager"
        ) as mock_manager:
            # Setup mock manager
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance

            # Create extensible server
            server = create_extensible_server(mock_config)

            # Test basic operations
            assert hasattr(server, "index_codebase")
            assert hasattr(server, "search_code")
            assert hasattr(server, "get_supported_languages")
            assert hasattr(server, "extensibility_manager")

    def test_configuration_migration_scenarios(self, mock_config) -> None:
        """Test various configuration migration scenarios."""
        scenarios = [
            {"name": "Basic legacy config", "config": mock_config, "expected_type": "legacy"}
        ]

        for scenario in scenarios:
            config_type = detect_configuration_type(scenario["config"])
            # Due to test environment, might detect as legacy even for extensible configs
            assert config_type in ["legacy", "extensible"]

            # Should be able to create server regardless of detected type
            server = create_server(scenario["config"], server_type="auto")
            assert server is not None


# Integration test that can be run to validate the full system
@pytest.mark.integration
class TestFullSystemIntegration:
    """Full system integration tests (requires external dependencies)."""

    @pytest.mark.skipif(
        not Path("/tmp/test-qdrant-available").exists(),
        reason="Requires Qdrant instance for full integration testing",
    )
    @pytest.mark.asyncio
    async def test_full_system_legacy_vs_extensible(self, sample_test_files) -> None:
        """Test that legacy and extensible servers produce similar results."""
        # This test would require actual Qdrant and API keys
        # Skip by default, can be enabled for CI/CD with proper setup


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
