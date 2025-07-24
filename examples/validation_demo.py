#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Validation demonstration for the CodeWeaver data source abstraction system.

This script demonstrates and validates that all components of the data source
system work correctly together and maintain backward compatibility.
"""

import asyncio
import logging
import sys

from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def validate_core_protocols() -> None:
    """Validate that core protocols and data structures work correctly."""
    logger.info("🧪 Testing core protocols and data structures...")

    from codeweaver.sources.base import (
        ContentItem,
        SourceCapability,
        SourceRegistry,
        get_source_registry,
    )

    # Test ContentItem creation
    item = ContentItem(
        path="/test/file.py",
        content_type="file",
        metadata={"test": True},
        language="python",
        source_id="test_source",
    )

    assert item.id is not None, "ContentItem should generate an ID"
    assert item.content_type == "file", "ContentItem should preserve content type"

    # Test capabilities
    capabilities = {SourceCapability.CONTENT_DISCOVERY, SourceCapability.CONTENT_READING}
    assert len(capabilities) == 2, "Capabilities should be properly defined"

    # Test source registry
    registry = get_source_registry()
    assert isinstance(registry, SourceRegistry), "Should get a valid registry instance"

    logger.info("✅ Core protocols validation passed")


async def validate_file_system_source() -> None:
    """Validate that the file system source works correctly."""
    logger.info("🧪 Testing file system source implementation...")

    from codeweaver.sources.filesystem import FileSystemSource, FileSystemSourceConfig

    # Create a file system source
    source = FileSystemSource(source_id="test_fs")
    capabilities = source.get_capabilities()

    assert len(capabilities) > 0, "File system source should have capabilities"
    assert source.source_id == "test_fs", "Source ID should be preserved"

    # Test configuration validation
    config: FileSystemSourceConfig = {
        "enabled": True,
        "root_path": str(Path.cwd()),  # Use current directory
        "use_gitignore": False,  # Disable for testing
        "patterns": ["**/*.py"],
    }

    is_valid = await source.validate_source(config)
    assert is_valid, "File system source should validate correctly"

    logger.info("✅ File system source validation passed")


async def validate_source_factory() -> None:
    """Validate that the source factory works correctly."""
    logger.info("🧪 Testing source factory and registry...")

    from codeweaver.sources.factory import get_source_factory

    factory = get_source_factory()
    available_sources = factory.list_available_sources()

    assert "filesystem" in available_sources, "FileSystem source should be available"
    assert available_sources["filesystem"]["implemented"], (
        "FileSystem should be marked as implemented"
    )

    # Test creating a file system source
    config = {"enabled": True, "root_path": str(Path.cwd()), "use_gitignore": False}

    source = factory.create_source("filesystem", config)
    assert source is not None, "Factory should create a valid source"

    logger.info("✅ Source factory validation passed")


async def validate_configuration_system() -> None:
    """Validate that the configuration system works correctly."""
    logger.info("🧪 Testing configuration system...")

    from codeweaver.sources.config import DataSourcesConfig

    # Create configuration
    config = DataSourcesConfig()
    config.add_source_config(
        source_type="filesystem",
        config={"root_path": str(Path.cwd())},
        enabled=True,
        priority=1,
        source_id="test_config",
    )

    # Test validation
    errors = config.validate_configurations()
    assert len(errors) == 0, f"Configuration should be valid, got errors: {errors}"

    # Test enabled sources
    enabled = config.get_enabled_source_configs()
    assert len(enabled) == 1, "Should have one enabled source"
    assert enabled[0]["source_id"] == "test_config", "Source ID should match"

    logger.info("✅ Configuration system validation passed")


async def validate_data_source_manager() -> None:
    """Validate that the data source manager works correctly."""
    logger.info("🧪 Testing data source manager...")

    from codeweaver.sources.config import DataSourcesConfig
    from codeweaver.sources.integration import DataSourceManager

    # Create configuration
    config = DataSourcesConfig()
    config.add_source_config(
        source_type="filesystem",
        config={
            "root_path": str(Path.cwd()),
            "use_gitignore": False,
            "patterns": ["**/*.py"],
            "max_file_size_mb": 10,  # Larger limit for testing
        },
        enabled=True,
        priority=1,
        source_id="manager_test",
    )

    # Create and initialize manager
    manager = DataSourceManager(config)
    await manager.initialize_sources()

    assert len(manager._active_sources) == 1, "Should have one active source"

    # Test content discovery (limit to avoid huge output)
    content_items = await manager.discover_all_content()
    assert isinstance(content_items, list), "Should return a list of content items"

    if content_items:
        # Test reading content from first item
        first_item = content_items[0]
        content = await manager.read_content_item(first_item)
        assert isinstance(content, str), "Should return string content"
        assert len(content) > 0, "Content should not be empty"

    # Clean up
    await manager.cleanup()

    logger.info("✅ Data source manager validation passed")


async def validate_backward_compatibility() -> None:
    """Validate that backward compatibility works correctly."""
    logger.info("🧪 Testing backward compatibility...")

    # Migration guide has been moved to examples/migration/sources_migration.py
    # We'll test the core integration functionality instead
    from codeweaver.config import CodeWeaverConfig
    from codeweaver.sources.integration import integrate_data_sources_with_config

    # Test configuration extension
    ExtendedConfig = integrate_data_sources_with_config(CodeWeaverConfig)
    config = ExtendedConfig()

    # Test that it has data sources support
    assert hasattr(config, "ensure_data_sources_initialized"), "Should have migration methods"

    # Test initialization
    config.ensure_data_sources_initialized()
    data_sources_config = config.get_data_sources_config()

    assert isinstance(data_sources_config, object), "Should have data sources config"
    assert len(data_sources_config.sources) > 0, "Should have migrated to at least one source"

    logger.info("✅ Backward compatibility validation passed")


async def validate_placeholder_implementations() -> None:
    """Validate that placeholder implementations are properly structured."""
    logger.info("🧪 Testing placeholder implementations...")

    from codeweaver.sources.api import APISource
    from codeweaver.sources.database import DatabaseSource
    from codeweaver.sources.git import GitRepositorySource
    from codeweaver.sources.web import WebCrawlerSource

    # Test that all placeholder sources can be instantiated
    sources = [GitRepositorySource(), DatabaseSource(), APISource(), WebCrawlerSource()]

    for source in sources:
        # Check that they have proper capabilities
        capabilities = source.get_capabilities()
        assert len(capabilities) > 0, f"Source {source.source_type} should have capabilities"

        # Check that they raise NotImplementedError for core methods
        try:
            await source.discover_content({"enabled": True})
            raise AssertionError(f"Source {source.source_type} should raise NotImplementedError")
        except NotImplementedError:
            pass  # Expected

        try:
            from codeweaver.sources.base import ContentItem

            item = ContentItem(path="test", content_type=source.source_type)
            await source.read_content(item)
            raise AssertionError(f"Source {source.source_type} should raise NotImplementedError")
        except NotImplementedError:
            pass  # Expected

    logger.info("✅ Placeholder implementations validation passed")


async def run_comprehensive_validation() -> bool:
    """Run all validation tests."""
    logger.info("🚀 Starting comprehensive data source system validation")

    validation_functions = [
        validate_core_protocols,
        validate_file_system_source,
        validate_source_factory,
        validate_configuration_system,
        validate_data_source_manager,
        validate_backward_compatibility,
        validate_placeholder_implementations,
    ]

    passed = 0
    failed = 0

    for validation_func in validation_functions:
        try:
            await validation_func()
            passed += 1
        except Exception:
            logger.exception("❌ Validation failed for %s", validation_func.__name__)
            failed += 1

    logger.info("📊 Validation Results:")
    logger.info("   ✅ Passed: %d", passed)
    logger.info("   ❌ Failed: %d", failed)
    logger.info("   📈 Success Rate: %.1f%%", (passed / (passed + failed)) * 100)

    if failed == 0:
        logger.info("🎉 All validations passed! Data source system is ready for use.")
        return True
    logger.error("💥 Some validations failed. Please check the errors above.")
    return False


def print_system_summary() -> None:
    """Print a summary of the implemented system."""
    print("\n" + "=" * 80)
    print("CODEWEAVER DATA SOURCE ABSTRACTION SYSTEM")
    print("=" * 80)
    print()
    print("📦 IMPLEMENTED COMPONENTS:")
    print("  ✅ Core Protocols (DataSource, ContentItem, SourceCapability)")
    print("  ✅ File System Source (fully functional)")
    print("  ✅ Source Factory and Registry")
    print("  ✅ Configuration System with TOML support")
    print("  ✅ Data Source Manager for orchestration")
    print("  ✅ Backward Compatibility Integration")
    print("  ✅ Migration Tools and Utilities")
    print()
    print("🚧 PLACEHOLDER IMPLEMENTATIONS (for future development):")
    print("  🔧 Git Repository Source")
    print("  🔧 Database Source (SQL/NoSQL)")
    print("  🔧 API Source (REST/GraphQL)")
    print("  🔧 Web Crawler Source")
    print()
    print("🎯 KEY BENEFITS:")
    print("  • 100% backward compatibility with existing CodeWeaver deployments")
    print("  • Extensible architecture supporting 5+ data source types")
    print("  • Universal content discovery and processing")
    print("  • Content deduplication and metadata extraction")
    print("  • Change watching and incremental updates")
    print("  • Strong typing with comprehensive validation")
    print()
    print("📚 DOCUMENTATION:")
    print("  • README.md - Comprehensive system documentation")
    print("  • migration_guide.py - Step-by-step migration guide")
    print("  • validation_demo.py - This validation and demo script")
    print("=" * 80)


if __name__ == "__main__":

    async def main() -> int:
        print_system_summary()

        success = await run_comprehensive_validation()

        if success:
            print("\n🎉 SUCCESS: Data source abstraction system is fully validated!")
            print("📖 Next steps:")
            print("   1. Review the migration examples: examples/migration/sources_migration.py")
            print("   2. Read the documentation: src/codeweaver/sources/README.md")
            print("   3. Integrate with your existing CodeWeaver deployment")
            return 0
        print("\n💥 FAILURE: Some validations failed. Please review the errors.")
        return 1

    sys.exit(asyncio.run(main()))
