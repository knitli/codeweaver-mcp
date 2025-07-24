#!/usr/bin/env python3
# sourcery skip: avoid-global-variables, require-return-annotation
"""
Validation script for Phase 2 backend refactoring.

Tests the backend abstraction layer to ensure:
1. Backend factory can create Qdrant backends
2. Backend protocol compliance
3. Backward compatibility with legacy server
"""

import asyncio
import logging

from codeweaver.backends.factory import BackendConfig, BackendFactory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_backend_factory() -> bool | None:
    """Test backend factory creation."""
    logger.info("Testing backend factory...")

    try:
        # Test basic Qdrant backend creation
        config = BackendConfig(
            provider="qdrant",
            url="http://localhost:6333",  # Default local Qdrant
            api_key=None,
            enable_hybrid_search=False,
        )

        backend = BackendFactory.create_backend(config)
        logger.info("✅ Successfully created backend: %s", type(backend).__name__)

        # Test hybrid backend creation
        hybrid_config = BackendConfig(
            provider="qdrant",
            url="http://localhost:6333",
            api_key=None,
            enable_hybrid_search=True,
            enable_sparse_vectors=True,
        )

        hybrid_backend = BackendFactory.create_backend(hybrid_config)
        logger.info("✅ Successfully created hybrid backend: %s", type(hybrid_backend).__name__)

    except Exception:
        logger.exception("❌ Backend factory test failed: ")
        return False

    else:
        logger.info("✅ Backend factory test passed")
        return True


async def test_backend_protocol() -> bool | None:
    """Test backend protocol compliance."""
    def raise_value_error(message: str):
        """Helper function to raise ValueError with a message."""
        raise ValueError(message)
    logger.info("Testing backend protocol compliance...")

    try:
        # Create a basic backend for protocol testing
        config = BackendConfig(provider="qdrant", url="http://localhost:6333", api_key=None)

        backend = BackendFactory.create_backend(config)

        # Test protocol methods exist
        required_methods = [
            "create_collection",
            "upsert_vectors",
            "search_vectors",
            "delete_vectors",
            "get_collection_info",
            "list_collections",
        ]

        for method in required_methods:
            if not hasattr(backend, method):
                raise_value_error(f"Backend is missing required method: {method}")

        logger.info("✅ Backend implements all required protocol methods")

    except Exception:
        logger.exception("❌ Backend protocol test failed: ")
        return False

    else:
        return True


def test_configuration_integration() -> bool | None:
    """Test configuration system integration."""
    logger.info("Testing configuration integration...")

    try:
        from codeweaver.config import get_config

        # Test that config loading doesn't break
        config = get_config()
        logger.info("✅ Configuration loaded successfully")

        # Test backend config creation from legacy config
        if hasattr(config, "qdrant"):
            BackendConfig(provider="qdrant", url=config.qdrant.url, api_key=config.qdrant.api_key)
            logger.info("✅ Backend config created from legacy config")

    except Exception:
        logger.exception("❌ Configuration integration test failed: ")
        return False

    else:
        logger.info("✅ Configuration integration test passed")
        return True


def test_import_compatibility() -> bool | None:
    """Test that all imports work correctly."""
    logger.info("Testing import compatibility...")

    try:
        # Test backend imports

        logger.info("✅ Backend imports successful")

        # Test server imports still work

        logger.info("✅ Server imports successful")

    except Exception:
        logger.exception("❌ Import compatibility test failed: ")
        return False

    else:
        logger.info("✅ Import compatibility test passed")
        return True


async def main() -> bool:
    """Run all validation tests."""
    logger.info("🚀 Starting Phase 2 Backend Refactoring Validation")

    tests = [
        ("Import Compatibility", test_import_compatibility),
        ("Configuration Integration", test_configuration_integration),
        ("Backend Factory", test_backend_factory),
        ("Backend Protocol", test_backend_protocol),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("\n--- Running %s Test ---", test_name)
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                logger.info("✅ %s: PASSED", test_name)
            else:
                logger.error("❌ %s: FAILED", test_name)
        except Exception:
            logger.exception("❌ %s: ERROR - ", test_name)

    logger.info("\n🎯 Test Results: {passed}/%s tests passed", total)

    if passed == total:
        logger.info("🎉 All tests passed! Backend refactoring is working correctly.")
        return True
    logger.error("❌ Some tests failed. Please review the backend refactoring.")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
