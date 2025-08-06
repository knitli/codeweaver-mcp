# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Tests for just the new configuration models without importing the full system.

Tests only the new configuration classes to avoid any circular import or
typing issues with the backend systems.
"""

import pytest

from pydantic import ValidationError

from codeweaver.cw_types import ComponentType, ServicesConfig


# Import just the new config models we created
def test_component_type_import():
    """Test that ComponentType can be imported correctly."""
    assert ComponentType.BACKEND.value == "backend"
    assert ComponentType.PROVIDER.value == "provider"
    assert ComponentType.SOURCE.value == "source"


def test_services_config_import():
    """Test that ServicesConfig can be imported correctly."""
    config = ServicesConfig()
    assert hasattr(config, "chunking")
    assert hasattr(config, "filtering")
    assert config.chunking.provider == "fastmcp_chunking"


# Test creating the new config models directly
def test_defaults_config_direct():
    """Test DefaultsConfig creation directly."""
    from codeweaver.config import DefaultsConfig

    config = DefaultsConfig()
    assert config.profile == "codeweaver_default"
    assert config.auto_configure is True
    assert config.validate_setup is True
    assert config.strict_validation is False


def test_plugin_registry_config_direct():
    """Test PluginRegistryConfig creation directly."""
    from codeweaver.config import PluginRegistryConfig

    config = PluginRegistryConfig()
    assert config.enabled_plugins == ["*"]
    assert config.disabled_plugins == []
    assert config.auto_resolve_conflicts is True


def test_custom_plugin_config_direct():
    """Test CustomPluginConfig creation directly."""
    from codeweaver.config import CustomPluginConfig

    config = CustomPluginConfig(
        plugin_type=ComponentType.BACKEND, module_path="test.plugin", class_name="TestPlugin"
    )
    assert config.plugin_type == ComponentType.BACKEND
    assert config.module_path == "test.plugin"
    assert config.class_name == "TestPlugin"
    assert config.enabled is True
    assert config.priority == 50


def test_plugins_config_direct():
    """Test PluginsConfig creation directly."""
    from codeweaver.config import PluginsConfig

    config = PluginsConfig()
    assert config.enabled is True
    assert config.auto_discover is True
    assert len(config.plugin_directories) >= 3
    assert "~/.codeweaver/plugins" in config.plugin_directories


def test_profile_config_direct():
    """Test ProfileConfig creation directly."""
    from codeweaver.config import ProfileConfig

    profile = ProfileConfig(
        name="test_profile", description="Test profile", backend={"provider": "memory"}
    )
    assert profile.name == "test_profile"
    assert profile.backend["provider"] == "memory"


def test_factory_config_direct():
    """Test FactoryConfig creation directly."""
    from codeweaver.config import FactoryConfig

    config = FactoryConfig()
    assert config.enable_dependency_injection is True
    assert config.enable_plugin_discovery is True
    assert config.lazy_initialization is False


def test_configuration_errors():
    """Test configuration error classes."""
    from codeweaver.config import ConfigurationError, PluginConfigurationError, ProfileError

    # Test basic ConfigurationError
    error = ConfigurationError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)

    # Test ProfileError inheritance
    profile_error = ProfileError("Profile not found")
    assert isinstance(profile_error, ConfigurationError)
    assert isinstance(profile_error, Exception)

    # Test PluginConfigurationError inheritance
    plugin_error = PluginConfigurationError("Plugin config invalid")
    assert isinstance(plugin_error, ConfigurationError)
    assert isinstance(plugin_error, Exception)


def test_custom_plugin_priority_validation():
    """Test priority validation in CustomPluginConfig."""
    from codeweaver.config import CustomPluginConfig

    # Valid priorities
    config = CustomPluginConfig(
        plugin_type=ComponentType.BACKEND,
        module_path="test.plugin",
        class_name="TestPlugin",
        priority=100,
    )
    assert config.priority == 100

    config_min = CustomPluginConfig(
        plugin_type=ComponentType.BACKEND,
        module_path="test.plugin",
        class_name="TestPlugin",
        priority=0,
    )
    assert config_min.priority == 0

    # Invalid priorities should raise validation error
    with pytest.raises(ValidationError):
        CustomPluginConfig(
            plugin_type=ComponentType.BACKEND,
            module_path="test.plugin",
            class_name="TestPlugin",
            priority=101,  # > 100
        )

    with pytest.raises(ValidationError):
        CustomPluginConfig(
            plugin_type=ComponentType.BACKEND,
            module_path="test.plugin",
            class_name="TestPlugin",
            priority=-1,  # < 0
        )


def test_factory_config_validation():
    """Test FactoryConfig validation."""
    from codeweaver.config import FactoryConfig

    # Valid configuration
    config = FactoryConfig(shutdown_timeout=60.0, health_check_interval=30.0)
    assert config.shutdown_timeout == 60.0
    assert config.health_check_interval == 30.0

    # Invalid timeout values
    with pytest.raises(ValidationError):
        FactoryConfig(shutdown_timeout=0)  # Must be > 0

    with pytest.raises(ValidationError):
        FactoryConfig(health_check_interval=0)  # Must be > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
