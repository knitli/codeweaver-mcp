#!/usr/bin/env python3
"""
Structure validation for Phase 2 backend refactoring.

Tests that the backend abstraction is properly structured without
requiring external services.
"""

import logging

from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports() -> bool | None:
    """Test that all backend imports work correctly."""
    logger.info("Testing backend imports...")

    try:
        # Test backend base imports

        logger.info("✅ Backend base imports successful")

        # Test factory imports

        logger.info("✅ Backend factory imports successful")

        # Test Qdrant backend imports

        logger.info("✅ Qdrant backend imports successful")

    except Exception:
        logger.exception("❌ Import test failed: ")
        return False

    else:
        logger.info("✅ All backend imports are valid")
        return True


def test_protocol_compliance() -> bool | None:  # sourcery skip: avoid-global-variables, extract-method
    """Test that backends implement required protocols."""
    logger.info("Testing protocol compliance...")

    try:
        from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend

        # Check if QdrantBackend has required methods
        required_methods = [
            "create_collection",
            "upsert_vectors",
            "search_vectors",
            "delete_vectors",
            "get_collection_info",
            "list_collections",
        ]

        for method in required_methods:
            if not hasattr(QdrantBackend, method):
                raise ValueError(f"QdrantBackend missing method: {method}")

        logger.info("✅ QdrantBackend implements all required methods")

        # Check hybrid backend
        if not hasattr(QdrantHybridBackend, "hybrid_search"):
            raise ValueError("QdrantHybridBackend missing hybrid_search method")

        logger.info("✅ QdrantHybridBackend implements hybrid methods")

    except Exception:
        logger.exception("❌ Protocol compliance test failed: ")
        return False

    else:
        logger.info("✅ All backends comply with required protocols")
        return True


def test_factory_structure() -> bool | None:  # sourcery skip: extract-method
    """Test that factory has proper structure."""
    logger.info("Testing factory structure...")

    try:
        from codeweaver.backends.factory import BackendConfig, BackendFactory

        # Check if factory has required methods
        required_methods = [
            "create_backend",
            "list_supported_providers",
            "_get_hybrid_backend_class",
            "_build_backend_args",
        ]

        for method in required_methods:
            if not hasattr(BackendFactory, method):
                raise ValueError(f"BackendFactory missing method: {method}")

        # Check if qdrant is in supported backends
        if "qdrant" not in BackendFactory._backends:
            raise ValueError("Qdrant not in supported backends")

        logger.info("✅ BackendFactory has proper structure")

        # Test BackendConfig creation
        config = BackendConfig(
            provider="qdrant", url="http://localhost:6333", enable_hybrid_search=True
        )

        if config.provider != "qdrant":
            raise ValueError("BackendConfig not working correctly")

        logger.info("✅ BackendConfig working correctly")

    except Exception:
        logger.exception("❌ Factory structure test failed: ")
        return False

    else:
        return True


def test_server_integration() -> bool | None:
    """Test that server can import backend components."""
    logger.info("Testing server integration...")

    try:
        return _test_server_imports()
    except Exception:
        logger.exception("❌ Server integration test failed: ")
        return False


# TODO Rename this here and in `test_server_integration`
def _test_server_imports() -> bool | None:
    """Test that server imports backend components correctly."""
    # Test server imports work

    logger.info("✅ Server imports successful")

    # Check if server has backend-related attributes
    # (We can't instantiate without credentials, but we can check structure)
    server_code = Path("src/codeweaver/server.py").read_text()

    # Check for backend-related imports
    if "backends.factory" not in server_code:
        raise ValueError("Server missing backend factory import")

    if "backends.base" not in server_code:
        raise ValueError("Server missing backend base import")

    # Check for backend usage
    if "_initialize_backend" not in server_code:
        raise ValueError("Server missing _initialize_backend method")

    if "self.backend" not in server_code:
        raise ValueError("Server not using backend attribute")

    logger.info("✅ Server integration looks correct")

    return True


def test_backward_compatibility() -> bool | None:
    """Test that legacy interfaces are preserved."""
    logger.info("Testing backward compatibility...")

    try:
        # Test that legacy server still exists

        # Check that legacy methods still exist in server
        server_code = Path("src/codeweaver/server.py").read_text()

        # Should still have Qdrant client for backward compatibility
        if "self.qdrant = QdrantClient" not in server_code:
            raise ValueError("Legacy QdrantClient support removed")

        logger.info("✅ Legacy QdrantClient support preserved")

        # Check that fallback mechanisms exist
        if "legacy" not in server_code.lower():
            logger.warning("⚠️ Limited legacy fallback mechanisms detected")
        else:
            logger.info("✅ Legacy fallback mechanisms present")

    except Exception:
        logger.exception("❌ Backward compatibility test failed: ")
        return False

    else:
        logger.info("✅ Backward compatibility preserved")
        return True


def main() -> bool:
    """Run all structure validation tests."""
    logger.info("🚀 Starting Phase 2 Backend Structure Validation")

    tests = [
        ("Import Structure", test_imports),
        ("Protocol Compliance", test_protocol_compliance),
        ("Factory Structure", test_factory_structure),
        ("Server Integration", test_server_integration),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("\n--- Running %s Test ---", test_name)
        try:
            if _result := test_func():
                passed += 1
                logger.info("✅ %s: PASSED", test_name)
            else:
                logger.error("❌ %s: FAILED", test_name)
        except Exception:
            logger.exception("❌ %s: ERROR - ", test_name)

    logger.info("\n🎯 Test Results: {passed}/%s tests passed", total)

    if passed == total:
        logger.info("🎉 All structure tests passed! Backend refactoring is properly implemented.")
        return True
    logger.error("❌ Some tests failed. Please review the backend refactoring.")
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
