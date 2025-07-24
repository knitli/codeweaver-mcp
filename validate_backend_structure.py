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


def test_imports():
    """Test that all backend imports work correctly."""
    logger.info("Testing backend imports...")
    
    try:
        # Test backend base imports
        from codeweaver.backends.base import (
            VectorBackend, 
            HybridSearchBackend,
            VectorPoint, 
            SearchResult,
            SearchFilter,
            FilterCondition,
            DistanceMetric,
            HybridStrategy,
            BackendError,
            ConnectionError,
            CollectionNotFoundError
        )
        logger.info("✅ Backend base imports successful")
        
        # Test factory imports
        from codeweaver.backends.factory import BackendFactory, BackendConfig
        logger.info("✅ Backend factory imports successful")
        
        # Test Qdrant backend imports
        from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend
        logger.info("✅ Qdrant backend imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_protocol_compliance():
    """Test that backends implement required protocols."""
    logger.info("Testing protocol compliance...")
    
    try:
        from codeweaver.backends.base import VectorBackend
        from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend
        
        # Check if QdrantBackend has required methods
        required_methods = [
            'create_collection',
            'upsert_vectors', 
            'search_vectors',
            'delete_vectors',
            'get_collection_info',
            'list_collections'
        ]
        
        for method in required_methods:
            if not hasattr(QdrantBackend, method):
                raise ValueError(f"QdrantBackend missing method: {method}")
                
        logger.info("✅ QdrantBackend implements all required methods")
        
        # Check hybrid backend
        if not hasattr(QdrantHybridBackend, 'hybrid_search'):
            raise ValueError("QdrantHybridBackend missing hybrid_search method")
            
        logger.info("✅ QdrantHybridBackend implements hybrid methods")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Protocol compliance test failed: {e}")
        return False


def test_factory_structure():
    """Test that factory has proper structure."""
    logger.info("Testing factory structure...")
    
    try:
        from codeweaver.backends.factory import BackendFactory, BackendConfig
        
        # Check if factory has required methods
        required_methods = [
            'create_backend',
            'list_supported_providers',
            '_get_hybrid_backend_class',
            '_build_backend_args'
        ]
        
        for method in required_methods:
            if not hasattr(BackendFactory, method):
                raise ValueError(f"BackendFactory missing method: {method}")
                
        # Check if qdrant is in supported backends
        if 'qdrant' not in BackendFactory._backends:
            raise ValueError("Qdrant not in supported backends")
            
        logger.info("✅ BackendFactory has proper structure")
        
        # Test BackendConfig creation
        config = BackendConfig(
            provider="qdrant",
            url="http://localhost:6333",
            enable_hybrid_search=True
        )
        
        if config.provider != "qdrant":
            raise ValueError("BackendConfig not working correctly")
            
        logger.info("✅ BackendConfig working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory structure test failed: {e}")
        return False


def test_server_integration():
    """Test that server can import backend components."""
    logger.info("Testing server integration...")
    
    try:
        # Test server imports work
        from codeweaver.server import CodeEmbeddingsServer
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
        
    except Exception as e:
        logger.error(f"❌ Server integration test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that legacy interfaces are preserved."""
    logger.info("Testing backward compatibility...")
    
    try:
        # Test that legacy server still exists
        from codeweaver.server import CodeEmbeddingsServer
        
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
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
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
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
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
        logger.info("🎉 All structure tests passed! Backend refactoring is properly implemented.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the backend refactoring.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)