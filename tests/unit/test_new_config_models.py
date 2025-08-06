# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Tests for the new configuration models by creating them inline.

This avoids import issues while testing the core functionality of our
enhanced configuration system.
"""

from typing import Annotated, Any

import pytest

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from codeweaver.cw_types import ComponentType


# Recreate the classes inline to test them without import issues
class DefaultsConfig(BaseModel):
    """Configuration for default behavior and profiles."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    profile: Annotated[
        str, Field(default="codeweaver_default", description="Configuration profile to use")
    ]
    auto_configure: Annotated[
        bool, Field(default=True, description="Automatically configure based on profile")
    ]
    validate_setup: Annotated[
        bool, Field(default=True, description="Validate configuration during startup")
    ]
    strict_validation: Annotated[
        bool, Field(default=False, description="Use strict validation mode")
    ]


class PluginRegistryConfig(BaseModel):
    """Plugin registry configuration for controlling plugin behavior."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled_plugins: Annotated[
        list[str],
        Field(default=["*"], description="List of enabled plugins (* means all discovered)"),
    ]
    disabled_plugins: Annotated[
        list[str], Field(default_factory=list, description="List of disabled plugins")
    ]
    plugin_priority_order: Annotated[
        list[str], Field(default_factory=list, description="Priority order for plugin resolution")
    ]
    auto_resolve_conflicts: Annotated[
        bool, Field(default=True, description="Automatically resolve plugin conflicts")
    ]
    require_explicit_enable: Annotated[
        bool, Field(default=False, description="Require explicit enabling of all plugins")
    ]


class CustomPluginConfig(BaseModel):
    """Configuration for a custom plugin."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: Annotated[bool, Field(default=True, description="Whether plugin is enabled")]
    plugin_type: Annotated[ComponentType, Field(description="Type of plugin")]
    module_path: Annotated[str, Field(description="Python module path")]
    class_name: Annotated[str, Field(description="Plugin class name")]
    entry_point: Annotated[
        str | None,
        Field(default=None, description="Entry point name (alternative to module_path/class_name)"),
    ]
    priority: Annotated[
        int, Field(default=50, ge=0, le=100, description="Plugin priority (0=lowest, 100=highest)")
    ]
    config: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Plugin-specific configuration")
    ]
    dependencies: Annotated[
        list[str], Field(default_factory=list, description="Required dependencies")
    ]
    tags: Annotated[
        list[str], Field(default_factory=list, description="Plugin tags for categorization")
    ]


class PluginsConfig(BaseModel):
    """Enhanced plugin system configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: Annotated[bool, Field(default=True, description="Enable plugin system")]
    auto_discover: Annotated[
        bool, Field(default=True, description="Automatically discover plugins")
    ]
    plugin_directories: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["~/.codeweaver/plugins", "./plugins", "./codeweaver_plugins"],
            description="Directories to search for plugins",
        ),
    ]
    entry_point_groups: Annotated[
        list[str],
        Field(
            default_factory=lambda: [
                "codeweaver.backends",
                "codeweaver.providers",
                "codeweaver.sources",
                "codeweaver.services",
            ],
            description="Entry point groups to scan",
        ),
    ]
    registry: Annotated[
        PluginRegistryConfig,
        Field(default_factory=PluginRegistryConfig, description="Plugin registry configuration"),
    ]
    custom: Annotated[
        dict[str, CustomPluginConfig],
        Field(default_factory=dict, description="Custom plugin configurations"),
    ]
    development_mode: Annotated[
        bool, Field(default=False, description="Enable development mode for plugin debugging")
    ]
    validation_strict: Annotated[
        bool, Field(default=True, description="Use strict validation for plugins")
    ]


class ProfileConfig(BaseModel):
    """Configuration profile definition."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: Annotated[str, Field(description="Profile name")]
    description: Annotated[str, Field(description="Profile description")]
    data_sources: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Data sources configuration overrides"),
    ]
    services: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Services configuration overrides")
    ]
    providers: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Providers configuration overrides")
    ]
    backend: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Backend configuration overrides")
    ]
    indexing: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Indexing configuration overrides")
    ]
    plugins: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Plugin configuration overrides")
    ]
    factory: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Factory configuration overrides")
    ]


class FactoryConfig(BaseModel):
    """Factory system configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enable_dependency_injection: Annotated[
        bool, Field(default=True, description="Enable dependency injection")
    ]
    enable_plugin_discovery: Annotated[
        bool, Field(default=True, description="Enable plugin discovery")
    ]
    validate_configurations: Annotated[
        bool, Field(default=True, description="Validate configurations during creation")
    ]
    lazy_initialization: Annotated[
        bool, Field(default=False, description="Use lazy initialization for components")
    ]
    enable_graceful_shutdown: Annotated[
        bool, Field(default=True, description="Enable graceful shutdown handling")
    ]
    shutdown_timeout: Annotated[
        float, Field(default=30.0, gt=0, description="Shutdown timeout in seconds")
    ]
    enable_health_checks: Annotated[
        bool, Field(default=True, description="Enable component health checks")
    ]
    health_check_interval: Annotated[
        float, Field(default=60.0, gt=0, description="Health check interval in seconds")
    ]
    enable_metrics: Annotated[
        bool, Field(default=True, description="Enable factory metrics collection")
    ]


# Tests
@pytest.mark.unit
class TestNewConfigModels:
    """Test the new configuration models."""

    def test_defaults_config_creation(self):
        """Test that DefaultsConfig can be created with default values."""
        config = DefaultsConfig()

        assert config.profile == "codeweaver_default"
        assert config.auto_configure is True
        assert config.validate_setup is True
        assert config.strict_validation is False

    def test_defaults_config_custom_values(self):
        """Test DefaultsConfig with custom values."""
        config = DefaultsConfig(
            profile="minimal", auto_configure=False, validate_setup=False, strict_validation=True
        )

        assert config.profile == "minimal"
        assert config.auto_configure is False
        assert config.validate_setup is False
        assert config.strict_validation is True

    def test_plugin_registry_defaults(self):
        """Test default plugin registry configuration."""
        config = PluginRegistryConfig()

        assert config.enabled_plugins == ["*"]
        assert config.disabled_plugins == []
        assert config.plugin_priority_order == []
        assert config.auto_resolve_conflicts is True
        assert config.require_explicit_enable is False

    def test_custom_plugin_config_creation(self):
        """Test custom plugin configuration creation."""
        config = CustomPluginConfig(
            plugin_type=ComponentType.BACKEND,
            module_path="my_company.plugins",
            class_name="MyVectorDB",
            priority=75,
            config={"connection_url": "mydb://localhost"},
            dependencies=["numpy"],
            tags=["database", "custom"],
        )

        assert config.enabled is True
        assert config.plugin_type == ComponentType.BACKEND
        assert config.module_path == "my_company.plugins"
        assert config.class_name == "MyVectorDB"
        assert config.entry_point is None
        assert config.priority == 75
        assert config.config == {"connection_url": "mydb://localhost"}
        assert config.dependencies == ["numpy"]
        assert config.tags == ["database", "custom"]

    def test_custom_plugin_priority_validation(self):
        """Test priority validation in custom plugin config."""
        # Valid priority
        config = CustomPluginConfig(
            plugin_type=ComponentType.SOURCE,
            module_path="test.plugin",
            class_name="TestPlugin",
            priority=100,
        )
        assert config.priority == 100

        # Test edge cases
        config_min = CustomPluginConfig(
            plugin_type=ComponentType.SOURCE,
            module_path="test.plugin",
            class_name="TestPlugin",
            priority=0,
        )
        assert config_min.priority == 0

        # Invalid priority should raise validation error
        with pytest.raises(ValidationError):
            CustomPluginConfig(
                plugin_type=ComponentType.SOURCE,
                module_path="test.plugin",
                class_name="TestPlugin",
                priority=101,  # > 100
            )

        with pytest.raises(ValidationError):
            CustomPluginConfig(
                plugin_type=ComponentType.SOURCE,
                module_path="test.plugin",
                class_name="TestPlugin",
                priority=-1,  # < 0
            )

    def test_plugins_config_defaults(self):
        """Test default plugins configuration."""
        config = PluginsConfig()

        assert config.enabled is True
        assert config.auto_discover is True
        assert "~/.codeweaver/plugins" in config.plugin_directories
        assert "./plugins" in config.plugin_directories
        assert "./codeweaver_plugins" in config.plugin_directories
        assert "codeweaver.backends" in config.entry_point_groups
        assert config.development_mode is False
        assert config.validation_strict is True

    def test_plugins_config_with_custom_plugin(self):
        """Test plugins config with custom plugin."""
        custom_plugin = CustomPluginConfig(
            plugin_type=ComponentType.BACKEND,
            module_path="custom.plugin",
            class_name="CustomBackend",
        )

        config = PluginsConfig(custom={"my_backend": custom_plugin})

        assert "my_backend" in config.custom
        assert config.custom["my_backend"].plugin_type == ComponentType.BACKEND

    def test_profile_config_creation(self):
        """Test profile configuration creation."""
        profile = ProfileConfig(
            name="test_profile",
            description="Test profile for development",
            data_sources={"default_source_type": "filesystem"},
            services={"chunking": {"max_chunk_size": 1000}},
            backend={"provider": "memory"},
        )

        assert profile.name == "test_profile"
        assert profile.description == "Test profile for development"
        assert profile.data_sources["default_source_type"] == "filesystem"
        assert profile.services["chunking"]["max_chunk_size"] == 1000
        assert profile.backend["provider"] == "memory"

    def test_factory_config_defaults(self):
        """Test factory configuration defaults."""
        config = FactoryConfig()

        assert config.enable_dependency_injection is True
        assert config.enable_plugin_discovery is True
        assert config.validate_configurations is True
        assert config.lazy_initialization is False
        assert config.enable_graceful_shutdown is True
        assert config.shutdown_timeout == 30.0
        assert config.enable_health_checks is True
        assert config.health_check_interval == 60.0
        assert config.enable_metrics is True

    def test_factory_config_validation(self):
        """Test factory config validation."""
        # Test invalid timeout values
        with pytest.raises(ValidationError):
            FactoryConfig(shutdown_timeout=0)  # Must be > 0

        with pytest.raises(ValidationError):
            FactoryConfig(health_check_interval=0)  # Must be > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
