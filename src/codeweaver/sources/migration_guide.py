# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Migration guide and utilities for transitioning to the data source abstraction system.

Provides step-by-step migration utilities and examples for integrating the new
data source system with existing CodeWeaver deployments.
"""

import logging

from pathlib import Path
from typing import Any

from codeweaver.config import CodeWeaverConfig
from codeweaver.sources.integration import (
    create_backward_compatible_server_integration,
    integrate_data_sources_with_config,
)


logger = logging.getLogger(__name__)


class MigrationHelper:
    """Helper class for migrating existing CodeWeaver deployments to use data sources."""

    @staticmethod
    def extend_existing_config() -> type:
        """Extend the existing CodeWeaverConfig class with data source support.

        This is the main migration entry point. Call this once to enable
        data source support in your existing configuration system.

        Returns:
            Extended CodeWeaverConfig class with data source support
        """
        # Import the existing config class
        extended_config = integrate_data_sources_with_config(CodeWeaverConfig)

        logger.info("Extended CodeWeaverConfig with data source support")
        return extended_config

    @staticmethod
    def create_default_data_source_config() -> dict[str, Any]:
        """Create a default data source configuration for migration.

        Returns:
            Dictionary with default data source configuration
        """
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
                        "source_id": "migrated_filesystem",
                        "config": {
                            "root_path": ".",
                            "use_gitignore": True,
                            "additional_ignore_patterns": [
                                "node_modules",
                                ".git",
                                ".venv",
                                "venv",
                                "__pycache__",
                                "target",
                                "build",
                                "dist",
                                ".next",
                                ".nuxt",
                                "coverage",
                            ],
                            "max_file_size_mb": 1,
                            "batch_size": 8,
                            "enable_change_watching": False,
                            "change_check_interval_seconds": 60,
                        },
                    }
                ],
            }
        }

    @staticmethod
    def migrate_legacy_config_file(config_path: Path, output_path: Path | None = None) -> None:
        """Migrate a legacy TOML configuration file to include data sources.

        Args:
            config_path: Path to existing configuration file
            output_path: Path for migrated configuration (defaults to config_path.with_suffix('.migrated.toml'))
        """
        if output_path is None:
            output_path = config_path.with_suffix(".migrated.toml")

        try:
            # Read existing configuration
            import tomllib

            with config_path.open("rb") as f:
                existing_config = tomllib.load(f)

            # Add data sources configuration if not present
            if "data_sources" not in existing_config:
                default_data_sources = MigrationHelper.create_default_data_source_config()
                existing_config.update(default_data_sources)

                # Migrate relevant settings from existing sections
                data_source_config = existing_config["data_sources"]["sources"][0]["config"]

                # Migrate from indexing section
                if "indexing" in existing_config:
                    indexing = existing_config["indexing"]
                    data_source_config["use_gitignore"] = indexing.get("use_gitignore", True)
                    data_source_config["additional_ignore_patterns"] = indexing.get(
                        "additional_ignore_patterns",
                        data_source_config["additional_ignore_patterns"],
                    )
                    data_source_config["batch_size"] = indexing.get("batch_size", 8)
                    data_source_config["enable_change_watching"] = indexing.get(
                        "enable_auto_reindex", False
                    )

                # Migrate from chunking section
                if "chunking" in existing_config:
                    chunking = existing_config["chunking"]
                    data_source_config["max_file_size_mb"] = chunking.get("max_file_size_mb", 1)

            # Write migrated configuration
            import tomli_w

            with output_path.open("wb") as f:
                tomli_w.dump(existing_config, f)

            logger.info("Migrated configuration from %s to %s", config_path, output_path)

        except Exception as e:
            logger.exception("Failed to migrate configuration file: %s", e)
            raise

    @staticmethod
    def validate_migration(config: CodeWeaverConfig) -> dict[str, Any]:
        """Validate that a configuration has been properly migrated.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "is_migrated": False,
            "has_data_sources": False,
            "validation_errors": [],
            "recommendations": [],
        }

        # Check if data sources are available
        if hasattr(config, "data_sources"):
            results["has_data_sources"] = True
            results["is_migrated"] = True

            # Validate data sources configuration
            data_sources = config.get_data_sources_config()
            validation_errors = data_sources.validate_configurations()
            results["validation_errors"] = validation_errors

            if not validation_errors:
                results["recommendations"].append("Data source configuration is valid")

            # Check if sources are enabled
            enabled_sources = data_sources.get_enabled_source_configs()
            if not enabled_sources:
                results["recommendations"].append("Enable at least one data source")
            else:
                results["recommendations"].append(
                    f"Found {len(enabled_sources)} enabled data sources"
                )

        else:
            results["recommendations"].extend([
                "Configuration not migrated to data source system",
                "Call MigrationHelper.extend_existing_config() to enable data sources",
                "Add data sources configuration to TOML file",
            ])

        return results


class ServerMigrationExample:
    """Example of how to migrate an existing server to use data sources."""

    @staticmethod
    def create_migrated_server_class(original_server_class: type) -> type:
        """Create a migrated server class with data source support.

        Args:
            original_server_class: Original server class to migrate

        Returns:
            Migrated server class with data source support
        """
        # Extend the server class with backward compatibility
        migrated_class = create_backward_compatible_server_integration(original_server_class)

        # Add additional migration helpers
        original_init = migrated_class.__init__

        def enhanced_init(self, config=None):
            """Enhanced initialization with migration logging."""
            # Ensure config has data sources support
            if config and hasattr(config, "ensure_data_sources_initialized"):
                config.ensure_data_sources_initialized()
                logger.info("Initialized data sources for migrated server")

            # Call original initialization
            original_init(self, config)

        migrated_class.__init__ = enhanced_init
        migrated_class.__name__ = f"Migrated{original_server_class.__name__}"

        return migrated_class

    @staticmethod
    async def demonstrate_migration_workflow():
        """Demonstrate the complete migration workflow."""
        logger.info("Starting data source migration demonstration")

        # Step 1: Extend configuration class
        ExtendedConfig = MigrationHelper.extend_existing_config()
        logger.info("✓ Extended configuration class")

        # Step 2: Load configuration with data sources
        config = ExtendedConfig()
        config.ensure_data_sources_initialized()
        logger.info("✓ Initialized data sources configuration")

        # Step 3: Validate migration
        validation_results = MigrationHelper.validate_migration(config)
        logger.info("✓ Migration validation: %s", validation_results)

        # Step 4: Create data source manager
        data_source_manager = config.create_data_source_manager()
        await data_source_manager.initialize_sources()
        logger.info("✓ Initialized data source manager")

        # Step 5: Demonstrate content discovery
        content_items = await data_source_manager.discover_all_content()
        logger.info("✓ Discovered %d content items", len(content_items))

        # Step 6: Clean up
        await data_source_manager.cleanup()
        logger.info("✓ Migration demonstration completed")


def get_migration_checklist() -> list[str]:
    """Get a checklist for migrating to the data source system.

    Returns:
        List of migration steps
    """
    return [
        "1. Extend CodeWeaverConfig class with data source support",
        "   ```python",
        "   from codeweaver.sources.migration_guide import MigrationHelper",
        "   ExtendedConfig = MigrationHelper.extend_existing_config()",
        "   ```",
        "",
        "2. Update configuration files to include data sources section",
        "   - Add [data_sources] section to TOML files",
        "   - Configure at least one data source (usually filesystem)",
        "   - Migrate existing settings to data source configuration",
        "",
        "3. Update server initialization code",
        "   ```python",
        "   config = ExtendedConfig()",
        "   config.ensure_data_sources_initialized()",
        "   server = CodeEmbeddingsServer(config)",
        "   ```",
        "",
        "4. (Optional) Extend server class for enhanced data source support",
        "   ```python",
        "   from codeweaver.sources.migration_guide import ServerMigrationExample",
        "   MigratedServer = ServerMigrationExample.create_migrated_server_class(CodeEmbeddingsServer)",
        "   server = MigratedServer(config)",
        "   ```",
        "",
        "5. Test the migration",
        "   - Verify existing functionality still works",
        "   - Test new data source features",
        "   - Check configuration validation",
        "",
        "6. (Future) Add additional data sources as needed",
        "   - Git repositories",
        "   - Databases",
        "   - APIs",
        "   - Web crawlers",
    ]


def print_migration_guide() -> None:
    """Print a comprehensive migration guide."""
    print("=" * 80)
    print("CodeWeaver Data Source Migration Guide")
    print("=" * 80)
    print()

    print("OVERVIEW")
    print("-" * 40)
    print("This guide helps you migrate from the legacy file-system-only indexing")
    print("to the new data source abstraction system with full backward compatibility.")
    print()

    print("MIGRATION CHECKLIST")
    print("-" * 40)
    checklist = get_migration_checklist()
    for item in checklist:
        print(item)
    print()

    print("CONFIGURATION EXAMPLE")
    print("-" * 40)
    print("Add this to your .code-weaver.toml file:")
    print()
    from codeweaver.sources.config import get_example_data_sources_config

    print(get_example_data_sources_config())
    print()

    print("BACKWARD COMPATIBILITY")
    print("-" * 40)
    print("✓ Existing configurations continue to work unchanged")
    print("✓ Automatic migration from legacy settings")
    print("✓ Same index_codebase() interface and behavior")
    print("✓ Zero breaking changes for existing deployments")
    print()

    print("BENEFITS OF MIGRATION")
    print("-" * 40)
    print("• Support for multiple data sources (git, databases, APIs, web)")
    print("• Enhanced content discovery and metadata extraction")
    print("• Improved change watching and incremental updates")
    print("• Content deduplication across sources")
    print("• Future-proof architecture for enterprise sources")
    print()

    print("NEED HELP?")
    print("-" * 40)
    print("• Check the migration validation: MigrationHelper.validate_migration()")
    print("• Run the demo: ServerMigrationExample.demonstrate_migration_workflow()")
    print("• Read the full documentation in sources/README.md")
    print("=" * 80)


if __name__ == "__main__":
    # Run migration demonstration if called directly
    import asyncio

    print_migration_guide()

    print("\nRunning migration demonstration...")
    asyncio.run(ServerMigrationExample.demonstrate_migration_workflow())
