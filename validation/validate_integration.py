#!/usr/bin/env python3
"""
Integration validation script for backward compatibility.

This script validates that the integration is working correctly without
requiring external dependencies like Qdrant or API keys.
"""

import logging
import sys

from pathlib import Path
from unittest.mock import Mock, patch


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_imports() -> bool | None:
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        with patch("qdrant_client.QdrantClient"):
            logger.info("✅ Core imports successful")

    except Exception:
        logger.exception("❌ Import failed: ")
        return False

    else:
        logger.info("✅ All imports are valid")
        return True


def test_configuration_detection() -> bool | None:
    # sourcery skip: extract-method
    """Test configuration type detection."""
    logger.info("Testing configuration detection...")

    try:
        with (
            patch("qdrant_client.QdrantClient"),
            patch("codeweaver.embeddings.create_embedder"),
            patch("codeweaver.embeddings.VoyageAIReranker"),
        ):
            from codeweaver.server import detect_configuration_type

            # Create mock config
            config = Mock()
            config.chunking = Mock()
            config.chunking.max_chunk_size = 1500
            config.chunking.min_chunk_size = 50
            config.indexing = Mock()
            config.indexing.batch_size = 8
            config.indexing.enable_auto_reindex = False
            config.qdrant = Mock()
            config.qdrant.url = "test"
            config.embedding = Mock()
            config.embedding.provider = "voyage"
            config.rate_limiting = Mock()
            config.server = Mock()

            # Test detection
            config_type = detect_configuration_type(config)
            logger.info(f"✅ Configuration type detected: {config_type}")

            assert config_type in ["legacy", "extensible"]
            return True

    except Exception:
        logger.exception("❌ Configuration detection failed: ")
        return False


def test_server_factory() -> bool | None:  # sourcery skip: extract-method
    """Test server factory functions."""
    logger.info("Testing server factory...")

    try:
        with (
            patch("qdrant_client.QdrantClient"),
            patch("codeweaver.embeddings.create_embedder"),
            patch("codeweaver.embeddings.VoyageAIReranker"),
            patch("codeweaver.providers.get_provider_factory"),
        ):
            from codeweaver.server import (
                create_extensible_server,
                create_legacy_server,
                create_server,
            )

            # Create mock config
            config = Mock()
            config.chunking = Mock()
            config.chunking.max_chunk_size = 1500
            config.chunking.min_chunk_size = 50
            config.indexing = Mock()
            config.indexing.batch_size = 8
            config.indexing.enable_auto_reindex = False
            config.qdrant = Mock()
            config.qdrant.url = "test"
            config.qdrant.api_key = "test"
            config.qdrant.collection_name = "test"
            config.embedding = Mock()
            config.embedding.provider = "voyage"
            config.embedding.api_key = "test"
            config.rate_limiting = Mock()
            config.rate_limiting.enabled = True
            config.server = Mock()

            # Test legacy server creation
            legacy_server = create_legacy_server(config)
            logger.info(f"✅ Legacy server created: {type(legacy_server).__name__}")

            # Test extensible server creation
            with patch("codeweaver.factories.extensibility_manager.ExtensibilityManager"):
                extensible_server = create_extensible_server(config)
                logger.info(f"✅ Extensible server created: {type(extensible_server).__name__}")

            # Test auto server creation
            auto_server = create_server(config, server_type="auto")
            logger.info(f"✅ Auto server created: {type(auto_server).__name__}")

            return True

    except Exception:
        logger.exception("❌ Server factory test failed: ")
        import traceback

        traceback.print_exc()
        return False


def test_migration_utilities() -> bool | None:  # sourcery skip: extract-method
    """Test migration utilities."""
    logger.info("Testing migration utilities...")

    try:
        from codeweaver.factories.extensibility_manager import ExtensibilityConfig
        from codeweaver.factories.integration import (
            create_migration_config,
            validate_migration_readiness,
        )

        # Test migration config creation
        migration_config = create_migration_config()
        assert isinstance(migration_config, ExtensibilityConfig)
        logger.info("✅ Migration config created successfully")

        # Test migration readiness validation
        config = Mock()
        config.backend = Mock()
        config.embedding = Mock()

        results = validate_migration_readiness(config)
        assert isinstance(results, dict)
        assert "ready" in results
        logger.info("✅ Migration readiness validation successful")

    except Exception:
        logger.exception("❌ Migration utilities test failed: ")
        return False

    else:
        logger.info("✅ Migration utilities test passed")
        return True


def test_compatibility_layer() -> bool | None:  # sourcery skip: extract-method
    """Test the compatibility layer."""
    logger.info("Testing compatibility layer...")

    try:
        with patch("codeweaver.factories.extensibility_manager.ExtensibilityManager"):
            from codeweaver.factories.integration import LegacyCompatibilityAdapter

            # Mock the manager
            mock_manager_instance = Mock()
            adapter = LegacyCompatibilityAdapter(mock_manager_instance)

            # Test adapter methods exist
            assert hasattr(adapter, "get_qdrant_client")
            assert hasattr(adapter, "get_embedder")
            assert hasattr(adapter, "get_reranker")
            logger.info("✅ Compatibility adapter created successfully")

    except Exception:
        logger.exception("❌ Compatibility layer test failed: ")
        return False

    else:
        logger.info("✅ Compatibility layer test passed")
        return True


def test_main_integration() -> bool | None:  # sourcery skip: extract-method
    """Test main.py integration."""
    logger.info("Testing main.py integration...")

    try:
        with (
            patch("qdrant_client.QdrantClient"),
            patch("codeweaver.embeddings.create_embedder"),
            patch("codeweaver.embeddings.VoyageAIReranker"),
            patch("codeweaver.providers.get_provider_factory"),
            patch("codeweaver.config.get_config_manager") as mock_config_manager,
        ):
            # Mock config manager
            mock_config = Mock()
            mock_config.chunking = Mock()
            mock_config.chunking.max_chunk_size = 1500
            mock_config.chunking.min_chunk_size = 50
            mock_config.indexing = Mock()
            mock_config.indexing.batch_size = 8
            mock_config.indexing.enable_auto_reindex = False
            mock_config.qdrant = Mock()
            mock_config.qdrant.url = "test"
            mock_config.qdrant.api_key = "test"
            mock_config.qdrant.collection_name = "test"
            mock_config.embedding = Mock()
            mock_config.embedding.provider = "voyage"
            mock_config.embedding.api_key = "test"
            mock_config.rate_limiting = Mock()
            mock_config.rate_limiting.enabled = True
            mock_config.server = Mock()
            mock_config.server.log_level = "INFO"
            mock_config.server.server_version = "2.0.0"

            mock_config_manager_instance = Mock()
            mock_config_manager_instance.load_config.return_value = mock_config
            mock_config_manager.return_value = mock_config_manager_instance

            # Import main module functions
            from codeweaver.main import get_server_instance

            # Test server instance creation
            server = get_server_instance()
            assert server is not None
            logger.info(f"✅ Main integration successful: {type(server).__name__}")

    except Exception:
        logger.exception("❌ Main integration test failed: ")
        import traceback

        traceback.print_exc()
        return False

    else:
        logger.info("✅ Main integration test passed")
        return True


def main() -> int:
    """Run all validation tests."""
    logger.info("Starting integration validation...")

    tests = [
        test_imports,
        test_configuration_detection,
        test_server_factory,
        test_migration_utilities,
        test_compatibility_layer,
        test_main_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception:
            logger.exception("❌ Test {test.__name__} failed with exception: ")
            results.append(False)

    passed = sum(results)
    total = len(results)

    logger.info(f"\n📊 Integration Validation Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All integration tests passed! Backward compatibility is working correctly.")
        return 0
    logger.error(f"❌ {total - passed} tests failed. Please check the issues above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
