# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Integration tests for the service layer implementation."""

import asyncio
import logging
import tempfile

from pathlib import Path

import pytest

from codeweaver.services import ServiceBridge, ServiceCoordinator, ServicesManager
from codeweaver.cw_types import ServicesConfig, ServiceType


@pytest.mark.asyncio
async def test_service_layer_integration():  # sourcery skip: avoid-global-variables, no-long-functions
    """Test the complete service layer integration."""
    print("🔧 Testing Service Layer Integration")

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_service_integration")

    try:
        # 1. Create services configuration
        print("📋 Creating services configuration...")
        services_config = ServicesConfig()

        # 2. Initialize services manager
        print("⚙️ Initializing services manager...")
        services_manager = ServicesManager(services_config, logger)
        await services_manager.initialize()

        # 3. Test service creation
        print("🛠️ Testing service creation...")
        chunking_service = services_manager.get_chunking_service()
        filtering_service = services_manager.get_filtering_service()

        print(f"   ✅ Chunking service: {chunking_service.__class__.__name__}")
        print(f"   ✅ Filtering service: {filtering_service.__class__.__name__}")

        # 4. Test service health checks
        print("💊 Testing service health checks...")
        health_report = await services_manager.get_health_report()
        print(f"   Overall status: {health_report.overall_status.value}")
        for service_type, health in health_report.services.items():
            print(f"   {service_type.value}: {health.status.value}")

        # 5. Test chunking service functionality
        print("📝 Testing chunking service functionality...")
        test_content = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

@pytest.mark.async_test
@pytest.mark.integration
@pytest.mark.services
class TestClass:
    def method(self):
        pass
        '''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            test_file = Path(f.name)

        try:
            chunks = await chunking_service.chunk_content(test_content, test_file)
            print(f"   ✅ Created {len(chunks)} chunks")

            # Test streaming
            chunk_count = 0
            async for _chunk in chunking_service.chunk_content_stream(test_content, test_file):
                chunk_count += 1
            print(f"   ✅ Streamed {chunk_count} chunks")

        finally:
            test_file.unlink()

        # 6. Test filtering service functionality
        print("📂 Testing filtering service functionality...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test.py").write_text("print('hello')")
            (temp_path / "test.js").write_text("console.log('hello')")
            (temp_path / "README.md").write_text("# Test")

            files = await filtering_service.discover_files(temp_path)
            print(f"   ✅ Discovered {len(files)} files")

            # Test file filtering
            python_files = await filtering_service.discover_files(
                temp_path, include_patterns=["*.py"]
            )
            print(f"   ✅ Found {len(python_files)} Python files")

        # 7. Test middleware bridge
        print("🌉 Testing middleware bridge...")
        ServiceBridge(services_manager)
        service_coordinator = ServiceCoordinator(services_manager)

        # Test coordination
        coordination_result = await service_coordinator.coordinate_indexing("/tmp")
        print(f"   ✅ Coordination result: {coordination_result}")

        # 8. Test service statistics
        print("📊 Testing service statistics...")
        chunking_stats = await chunking_service.get_chunking_stats()
        filtering_stats = await filtering_service.get_filtering_stats()

        print(f"   Chunking: {chunking_stats.total_files_processed} files processed")
        print(f"   Filtering: {filtering_stats.total_files_scanned} files scanned")

        # 9. Test service capabilities
        print("🔍 Testing service capabilities...")
        supported_languages = chunking_service.get_supported_languages()
        print(f"   ✅ Supports {len(supported_languages)} languages")

        active_patterns = filtering_service.get_active_patterns()
        print(f"   ✅ Active patterns: {len(active_patterns.get('include_patterns', []))}")

        print("✅ All service integration tests passed!")

        # 10. Cleanup
        print("🧹 Cleaning up...")
        await services_manager.shutdown()

    except Exception as e:
        logger.exception("Service integration test failed")
        print(f"❌ Test failed: {e}")
        return False
    else:
        print("✅ Service layer integration test completed successfully!")
        return True


@pytest.mark.asyncio
async def test_service_factory_integration():
    """Test service integration with the main factory."""
    print("\n🏭 Testing Service Factory Integration")

    try:
        from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
        from codeweaver.cw_types import ChunkingServiceConfig, FilteringServiceConfig

        # Create factory
        factory = CodeWeaverFactory()

        # Test service creation through factory
        chunking_config = ChunkingServiceConfig()
        filtering_config = FilteringServiceConfig()

        chunking_service = await factory.create_service(ServiceType.CHUNKING, chunking_config)
        filtering_service = await factory.create_service(ServiceType.FILTERING, filtering_config)

        print("   ✅ Services created through factory")

        # Test capabilities query
        available_components = factory.get_available_components()
        services = available_components.get("services", {})
        print(f"   ✅ Available services: {list(services.keys())}")

        # Cleanup services
        await chunking_service.shutdown()
        await filtering_service.shutdown()

        print("✅ Factory integration tests passed!")

    except Exception as e:
        print(f"❌ Factory integration test failed: {e}")
        return False
    else:
        return True


async def main():
    """Run all integration tests."""
    print("🚀 Starting Service Layer Integration Tests\n")

    test1_passed = await test_service_layer_integration()
    test2_passed = await test_service_factory_integration()

    if test1_passed and test2_passed:
        print("\n🎉 All integration tests passed successfully!")
        return 0
    print("\n💥 Some tests failed!")
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
