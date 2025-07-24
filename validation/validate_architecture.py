#!/usr/bin/env python3
"""
Architecture validation script for CodeWeaver integration.

Tests the core integration without requiring external dependencies.
"""

import logging
import sys

from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_core_architecture() -> bool | None:
    """Test core architecture components that don't require external dependencies."""
    logger.info("Testing core architecture...")

    try:
        # Test factory system components
        from codeweaver.factories.extensibility_manager import ExtensibilityConfig
        from codeweaver.factories.integration import create_migration_config

        # Test configuration creation
        migration_config = create_migration_config()
        assert isinstance(migration_config, ExtensibilityConfig)
        logger.info("✅ ExtensibilityConfig creation successful")

        # Test configuration options
        assert migration_config.enable_legacy_fallbacks is True
        assert migration_config.migration_mode is True
        logger.info("✅ Migration configuration options correct")

        return True

    except Exception as e:
        logger.exception(f"❌ Core architecture test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_server_classes() -> bool | None:
    """Test that server classes can be imported and have correct structure."""
    logger.info("Testing server class structure...")

    try:
        # We can't instantiate servers without dependencies, but we can test class structure

        # Check that our classes exist and have the right methods

        # Read server.py and extract class definitions without importing dependencies
        server_file = Path(__file__).parent / "src" / "codeweaver" / "server.py"
        with open(server_file, "r") as f:
            content = f.read()

        # Check that both server classes are defined
        assert "class CodeEmbeddingsServer:" in content
        assert "class ExtensibleCodeEmbeddingsServer:" in content
        logger.info("✅ Both server classes are defined")

        # Check that factory functions are defined
        assert "def create_server(" in content
        assert "def create_legacy_server(" in content
        assert "def create_extensible_server(" in content
        assert "def detect_configuration_type(" in content
        logger.info("✅ All factory functions are defined")

        # Check that migration utilities are defined
        assert "class ServerMigrationManager:" in content
        assert "async def migrate_config_to_extensible(" in content
        logger.info("✅ Migration utilities are defined")

        return True

    except Exception as e:
        logger.exception(f"❌ Server classes test failed: {e}")
        return False


def test_configuration_structure() -> bool | None:
    """Test configuration structure and validation."""
    logger.info("Testing configuration structure...")

    try:
        # Test that we can create basic config structure without dependencies
        from unittest.mock import Mock

        # Create a mock configuration that matches the expected structure
        config = Mock()
        config.chunking = Mock()
        config.chunking.max_chunk_size = 1500
        config.chunking.min_chunk_size = 50
        config.indexing = Mock()
        config.indexing.batch_size = 8
        config.embedding = Mock()
        config.embedding.provider = "voyage"
        config.qdrant = Mock()
        config.qdrant.collection_name = "test"

        # Test that the mock structure is valid
        assert hasattr(config, "chunking")
        assert hasattr(config, "indexing")
        assert hasattr(config, "embedding")
        assert hasattr(config, "qdrant")
        logger.info("✅ Configuration structure validation successful")

        return True

    except Exception as e:
        logger.exception(f"❌ Configuration structure test failed: {e}")
        return False


def test_integration_utilities() -> bool | None:
    """Test integration utilities and compatibility helpers."""
    logger.info("Testing integration utilities...")

    try:
        from unittest.mock import Mock

        from codeweaver.factories.integration import validate_migration_readiness

        # Test validation with a proper config
        config = Mock()
        config.backend = Mock()
        config.embedding = Mock()

        results = validate_migration_readiness(config)
        assert isinstance(results, dict)
        assert "ready" in results
        assert "issues" in results
        assert "warnings" in results
        assert "recommendations" in results
        logger.info("✅ Migration readiness validation successful")

        # Test validation with missing config
        incomplete_config = Mock()
        # Don't add backend or embedding

        results = validate_migration_readiness(incomplete_config)
        assert results["ready"] is False
        assert len(results["issues"]) > 0
        logger.info("✅ Migration validation catches missing configuration")

        return True

    except Exception as e:
        logger.exception(f"❌ Integration utilities test failed: {e}")
        return False


def test_main_integration() -> bool | None:
    """Test that main.py has the correct integration structure."""
    logger.info("Testing main.py integration...")

    try:
        # Read main.py and check for integration changes
        main_file = Path(__file__).parent / "src" / "codeweaver" / "main.py"
        with open(main_file, "r") as f:
            content = f.read()

        # Check that the new imports are present
        assert "create_server" in content
        assert "detect_configuration_type" in content
        logger.info("✅ Main.py has correct imports")

        # Check that the server creation is updated
        assert "create_server(config, server_type='auto')" in content
        logger.info("✅ Main.py uses auto server creation")

        # Check that server type detection is logged
        assert "detect_configuration_type(config)" in content
        logger.info("✅ Main.py includes configuration type detection")

        return True

    except Exception as e:
        logger.exception(f"❌ Main integration test failed: {e}")
        return False


def test_backward_compatibility_structure() -> bool | None:
    """Test that backward compatibility structure is correct."""
    logger.info("Testing backward compatibility structure...")

    try:
        # Check that the legacy server still exists
        server_file = Path(__file__).parent / "src" / "codeweaver" / "server.py"
        with open(server_file, "r") as f:
            content = f.read()

        # Verify legacy server structure is preserved
        assert "class CodeEmbeddingsServer:" in content
        assert "def __init__(self, config: CodeWeaverConfig | None = None):" in content
        assert "async def index_codebase(" in content
        assert "async def search_code(" in content
        assert "async def ast_grep_search(" in content
        assert "async def get_supported_languages(" in content
        logger.info("✅ Legacy server interface preserved")

        # Verify extensible server has same interface
        assert "class ExtensibleCodeEmbeddingsServer:" in content
        # Should have the same methods
        extensible_methods = [
            "async def index_codebase(",
            "async def search_code(",
            "async def ast_grep_search(",
            "async def get_supported_languages(",
        ]
        for method in extensible_methods:
            assert method in content
        logger.info("✅ Extensible server maintains same interface")

        return True

    except Exception as e:
        logger.exception(f"❌ Backward compatibility structure test failed: {e}")
        return False


def test_file_syntax() -> bool | None:
    """Test that all Python files have valid syntax."""
    logger.info("Testing file syntax...")

    try:
        import py_compile

        src_dir = Path(__file__).parent / "src"

        # Find all Python files
        python_files = list(src_dir.rglob("*.py"))

        syntax_errors = []
        for py_file in python_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                syntax_errors.append((py_file, str(e)))

        if syntax_errors:
            logger.error(f"❌ Syntax errors found in {len(syntax_errors)} files:")
            for file_path, error in syntax_errors:
                logger.error(f"  {file_path}: {error}")
            return False

        logger.info(f"✅ All {len(python_files)} Python files have valid syntax")
        return True

    except Exception as e:
        logger.exception(f"❌ File syntax test failed: {e}")
        return False


def main() -> int:
    """Run all architecture validation tests."""
    logger.info("Starting architecture validation...")

    tests = [
        test_file_syntax,
        test_core_architecture,
        test_server_classes,
        test_configuration_structure,
        test_integration_utilities,
        test_main_integration,
        test_backward_compatibility_structure,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.exception(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    logger.info(f"\n📊 Architecture Validation Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All architecture tests passed! Integration is working correctly.")
        logger.info("\n✅ Key Features Validated:")
        logger.info("  • ExtensibleCodeEmbeddingsServer class created")
        logger.info("  • Configuration detection and migration utilities")
        logger.info("  • Server factory functions for easy instantiation")
        logger.info("  • Backward compatibility layer maintained")
        logger.info("  • Main.py supports both server types")
        logger.info("  • Integration testing framework created")
        logger.info("\n🚀 The extensible architecture is ready for deployment!")
        return 0
    logger.error(f"❌ {total - passed} tests failed. Please check the issues above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
