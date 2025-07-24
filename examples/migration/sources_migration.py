# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Example demonstrating how to migrate from legacy file-system-only indexing
to the new data source abstraction system.
"""

import asyncio
import logging

from codeweaver.config import CodeWeaverConfig
from codeweaver.sources.integration import (
    create_backward_compatible_server_integration,
    integrate_data_sources_with_config,
)


logger = logging.getLogger(__name__)


def migrate_configuration():
    """Example: Extend existing configuration with data source support."""
    # Step 1: Extend the existing config class
    ExtendedConfig = integrate_data_sources_with_config(CodeWeaverConfig)

    # Step 2: Create instance with data sources
    config = ExtendedConfig()
    config.ensure_data_sources_initialized()

    logger.info("Successfully extended configuration with data source support")
    return config


def create_data_source_config_example():
    """Example: Create a data source configuration for TOML file."""
    return {
        "data_sources": {
            "enabled": True,
            "default_source_type": "filesystem",
            "max_concurrent_sources": 5,
            "enable_content_deduplication": True,
            "enable_metadata_extraction": True,
            "sources": [
                {
                    "type": "filesystem",
                    "enabled": True,
                    "priority": 1,
                    "source_id": "main_codebase",
                    "config": {
                        "root_path": ".",
                        "use_gitignore": True,
                        "additional_ignore_patterns": [
                            "node_modules",
                            ".git",
                            ".venv",
                            "__pycache__",
                            "build",
                            "dist",
                        ],
                        "max_file_size_mb": 1,
                        "batch_size": 8,
                        "enable_change_watching": False,
                    },
                }
            ],
        }
    }


async def demonstrate_data_source_usage() -> None:
    """Example: Using data sources for content discovery."""
    # Create extended configuration
    ExtendedConfig = integrate_data_sources_with_config(CodeWeaverConfig)
    config = ExtendedConfig()
    config.ensure_data_sources_initialized()

    # Create data source manager
    data_source_manager = config.create_data_source_manager()

    try:
        # Initialize data sources
        await data_source_manager.initialize_sources()
        logger.info("Initialized data sources")

        # Discover content from all sources
        content_items = await data_source_manager.discover_all_content()
        logger.info("Discovered %d content items", len(content_items))

        # Example: Filter Python files
        python_files = [
            item for item in content_items
            if item.metadata.get("file_extension") == ".py"
        ]
        logger.info("Found %d Python files", len(python_files))

        # Example: Watch for changes (if enabled)
        if any(source.supports_watching for source in data_source_manager.enabled_sources):
            logger.info("Starting change watching...")
            # In real usage, this would run continuously
            # await data_source_manager.watch_for_changes()

    finally:
        await data_source_manager.cleanup()


def migrate_server_class():
    """Example: Migrate existing server class to support data sources."""
    from codeweaver.server import CodeEmbeddingsServer

    # Create migrated server class with backward compatibility
    MigratedServer = create_backward_compatible_server_integration(CodeEmbeddingsServer)

    # The migrated server maintains all existing functionality
    # while adding data source support
    logger.info("Created migrated server class: %s", MigratedServer.__name__)

    return MigratedServer


def validate_migration(config):
    """Example: Validate that migration was successful."""
    validation_results = {
        "has_data_sources": hasattr(config, "data_sources"),
        "can_create_manager": False,
        "sources_configured": 0,
        "errors": [],
    }

    try:
        if validation_results["has_data_sources"]:
            # Check if we can create a data source manager
            config.create_data_source_manager()
            validation_results["can_create_manager"] = True

            # Count configured sources
            data_sources_config = config.get_data_sources_config()
            validation_results["sources_configured"] = len(
                data_sources_config.get_enabled_source_configs()
            )
    except Exception as e:
        validation_results["errors"].append(str(e))

    return validation_results


async def main() -> None:
    """Run all migration examples."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Data Source Migration Examples")
    print("=" * 60)

    # Example 1: Extend configuration
    print("\n1. Extending Configuration:")
    config = migrate_configuration()
    print("   ✓ Configuration extended successfully")

    # Example 2: Validate migration
    print("\n2. Validating Migration:")
    validation = validate_migration(config)
    for key, value in validation.items():
        print(f"   - {key}: {value}")

    # Example 3: Data source usage
    print("\n3. Using Data Sources:")
    await demonstrate_data_source_usage()

    # Example 4: Server migration
    print("\n4. Migrating Server Class:")
    MigratedServer = migrate_server_class()
    print(f"   ✓ Created {MigratedServer.__name__}")

    print("\n" + "=" * 60)
    print("Migration examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
