#!/usr/bin/env python3
"""
Validation script for Phase 2 backend refactoring.

Tests the backend abstraction layer to ensure:
1. Backend factory can create Qdrant backends
2. Backend protocol compliance
3. Backward compatibility with legacy server
"""

import asyncio
import logging
from pathlib import Path

from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.backends.base import DistanceMetric, VectorPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_backend_factory():
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
        logger.info(f"✅ Successfully created backend: {type(backend).__name__}")
        
        # Test hybrid backend creation
        hybrid_config = BackendConfig(
            provider="qdrant",
            url="http://localhost:6333",
            api_key=None,
            enable_hybrid_search=True,
            enable_sparse_vectors=True,
        )
        
        hybrid_backend = BackendFactory.create_backend(hybrid_config)
        logger.info(f"✅ Successfully created hybrid backend: {type(hybrid_backend).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Backend factory test failed: {e}")
        return False


async def test_backend_protocol():
    """Test backend protocol compliance."""
    logger.info("Testing backend protocol compliance...")
    
    try:
        # Create a basic backend for protocol testing
        config = BackendConfig(
            provider="qdrant",
            url="http://localhost:6333",
            api_key=None,
        )
        
        backend = BackendFactory.create_backend(config)
        
        # Test protocol methods exist
        required_methods = [
            'create_collection',
            'upsert_vectors', 
            'search_vectors',
            'delete_vectors',
            'get_collection_info',
            'list_collections'
        ]
        
        for method in required_methods:
            if not hasattr(backend, method):
                raise ValueError(f"Backend missing required method: {method}")
                
        logger.info(f"✅ Backend implements all required protocol methods")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backend protocol test failed: {e}")
        return False


def test_configuration_integration():
    """Test configuration system integration."""
    logger.info("Testing configuration integration...")
    
    try:
        from codeweaver.config import get_config
        
        # Test that config loading doesn't break
        config = get_config()
        logger.info(f"✅ Configuration loaded successfully")
        
        # Test backend config creation from legacy config
        if hasattr(config, 'qdrant'):
            backend_config = BackendConfig(
                provider="qdrant",
                url=config.qdrant.url,
                api_key=config.qdrant.api_key,
            )
            logger.info(f"✅ Backend config created from legacy config")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration integration test failed: {e}")
        return False


def test_import_compatibility():
    """Test that all imports work correctly."""
    logger.info("Testing import compatibility...")
    
    try:
        # Test backend imports
        from codeweaver.backends.base import VectorBackend, VectorPoint, SearchResult
        from codeweaver.backends.factory import BackendFactory, BackendConfig
        from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend
        
        logger.info("✅ Backend imports successful")
        
        # Test server imports still work
        from codeweaver.server import CodeEmbeddingsServer
        logger.info("✅ Server imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import compatibility test failed: {e}")
        return False


async def main():
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
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Backend refactoring is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the backend refactoring.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)