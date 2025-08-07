# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Tests for enhanced configuration system functionality.

Tests the new configuration models, profile system, plugin configuration,
and factory integration introduced in the enhanced config system.
"""

import pytest

from pydantic import ValidationError

from codeweaver.config import (
    CodeWeaverConfig,
    ConfigurationError,
    CustomPluginConfig,
    DefaultsConfig,
    FactoryConfig,
    PluginConfigurationError,
    PluginRegistryConfig,
    PluginsConfig,
    ProfileConfig,
    ProfileError,
)
from codeweaver.cw_types import ComponentType


@pytest.mark.unit
@pytest.mark.config
class TestDefaultsConfig:
    """Test DefaultsConfig functionality."""

    def test_defaults_config_creation(self):
        """Test that DefaultsConfig can be created with default values."""
        config = DefaultsConfig()

        assert config.profile == "recommended"
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


@pytest.mark.unit
@pytest.mark.config
class TestPluginRegistryConfig:
    """Test PluginRegistryConfig functionality."""

    def test_plugin_registry_defaults(self):
        """Test default plugin registry configuration."""
        config = PluginRegistryConfig()

        assert config.enabled_plugins == ["*"]
        assert config.disabled_plugins == []
        assert config.plugin_priority_order == []
        assert config.auto_resolve_conflicts is True
        assert config.require_explicit_enable is False

    def test_plugin_registry_custom_config(self):
        """Test custom plugin registry configuration."""
        config = PluginRegistryConfig(
            enabled_plugins=["voyage_ai", "qdrant"],
            disabled_plugins=["legacy_plugin"],
            plugin_priority_order=["voyage_ai", "qdrant"],
            auto_resolve_conflicts=False,
            require_explicit_enable=True,
        )

        assert config.enabled_plugins == ["voyage_ai", "qdrant"]
        assert config.disabled_plugins == ["legacy_plugin"]
        assert config.plugin_priority_order == ["voyage_ai", "qdrant"]
        assert config.auto_resolve_conflicts is False
        assert config.require_explicit_enable is True


@pytest.mark.unit
@pytest.mark.config
class TestCustomPluginConfig:
    """Test CustomPluginConfig functionality."""

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

    def test_custom_plugin_with_entry_point(self):
        """Test custom plugin with entry point."""
        config = CustomPluginConfig(
            plugin_type=ComponentType.PROVIDER,
            entry_point="my_plugin_entry",
            module_path="ignored",  # Should be ignored when entry_point is set
            class_name="ignored",  # Should be ignored when entry_point is set
        )

        assert config.entry_point == "my_plugin_entry"
        assert config.module_path == "ignored"  # Still stored but would be ignored in loading
        assert config.class_name == "ignored"

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


class TestPluginsConfig:
    """Test PluginsConfig functionality."""

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


class TestProfileConfig:
    """Test ProfileConfig functionality."""

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

    def test_profile_config_empty_sections(self):
        """Test profile with empty configuration sections."""
        profile = ProfileConfig(name="minimal_profile", description="Minimal profile")

        assert profile.data_sources == {}
        assert profile.services == {}
        assert profile.providers == {}
        assert profile.backend == {}
        assert profile.indexing == {}
        assert profile.plugins == {}
        assert profile.factory == {}


@pytest.mark.unit
@pytest.mark.config
class TestFactoryConfig:
    """Test FactoryConfig functionality."""

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

    def test_factory_config_custom(self):
        """Test custom factory configuration."""
        config = FactoryConfig(
            enable_dependency_injection=False,
            lazy_initialization=True,
            shutdown_timeout=60.0,
            health_check_interval=30.0,
        )

        assert config.enable_dependency_injection is False
        assert config.lazy_initialization is True
        assert config.shutdown_timeout == 60.0
        assert config.health_check_interval == 30.0

    def test_factory_config_validation(self):
        """Test factory config validation."""
        # Test invalid timeout values
        with pytest.raises(ValidationError):
            FactoryConfig(shutdown_timeout=0)  # Must be > 0

        with pytest.raises(ValidationError):
            FactoryConfig(health_check_interval=0)  # Must be > 0


class TestEnhancedCodeWeaverConfig:
    """Test enhanced CodeWeaverConfig functionality."""

    def test_enhanced_config_creation(self):
        """Test that enhanced config includes new sections."""
        config = CodeWeaverConfig()

        # Test that new sections exist
        assert hasattr(config, "defaults")
        assert hasattr(config, "plugins")
        assert hasattr(config, "factory")
        assert hasattr(config, "services")
        assert hasattr(config, "profiles")

        # Test default values
        assert isinstance(config.defaults, DefaultsConfig)
        assert isinstance(config.plugins, PluginsConfig)
        assert isinstance(config.factory, FactoryConfig)
        assert config.profiles == {}

    def test_builtin_profiles_available(self):
        """Test that built-in profiles are available."""
        config = CodeWeaverConfig()
        builtin_profiles = config._get_builtin_profiles()

        assert "recommended" in builtin_profiles
        assert "minimal" in builtin_profiles
        assert "performance" in builtin_profiles

        # Test profile structure
        original_profile = builtin_profiles["recommended"]
        assert original_profile.name == "recommended"
        assert "Original CodeWeaver design" in original_profile.description
        assert original_profile.backend["provider"] == "qdrant"

    def test_profile_retrieval(self):
        """Test profile retrieval functionality."""
        config = CodeWeaverConfig()

        # Test built-in profile retrieval
        profile = config._get_profile("recommended")
        assert profile is not None
        assert profile.name == "recommended"

        # Test non-existent profile
        profile = config._get_profile("nonexistent")
        assert profile is None

    def test_profile_application_disabled(self):
        """Test that profile application can be disabled."""
        config = CodeWeaverConfig()
        config.defaults.auto_configure = False

        # Should not apply profile configuration
        # This test verifies the structure exists, actual application testing
        # would require more complex setup
        assert not config.defaults.auto_configure

    def test_original_defaults_setup(self):
        """Test original defaults setup."""
        config = CodeWeaverConfig()
        config.defaults.profile = "recommended"

        # Trigger the setup
        config._setup_original_defaults()

        # Verify original defaults are applied
        assert config.indexing.enable_auto_reindex is True

    def test_services_integration(self):
        """Test that services are properly integrated."""
        config = CodeWeaverConfig()

        # Test that services configuration is present and has expected structure
        assert hasattr(config.services, "chunking")
        assert hasattr(config.services, "filtering")
        assert hasattr(config.services, "validation")
        assert hasattr(config.services, "cache")
        assert hasattr(config.services, "monitoring")
        assert hasattr(config.services, "metrics")

    def test_config_with_custom_profile(self):
        """Test configuration with custom profile."""
        custom_profile = ProfileConfig(
            name="custom_test",
            description="Custom test profile",
            backend={"provider": "memory"},
            indexing={"batch_size": 32},
        )

        config = CodeWeaverConfig(
            profiles={"custom_test": custom_profile}, defaults={"profile": "custom_test"}
        )

        assert "custom_test" in config.profiles
        retrieved_profile = config._get_profile("custom_test")
        assert retrieved_profile is not None
        assert retrieved_profile.name == "custom_test"


class TestConfigurationErrors:
    """Test configuration error handling."""

    def test_configuration_error_inheritance(self):
        """Test that custom errors inherit correctly."""
        # Test basic ConfigurationError
        error = ConfigurationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

        # Test ProfileError
        profile_error = ProfileError("Profile not found")
        assert isinstance(profile_error, ConfigurationError)
        assert isinstance(profile_error, Exception)

        # Test PluginConfigurationError
        plugin_error = PluginConfigurationError("Plugin config invalid")
        assert isinstance(plugin_error, ConfigurationError)
        assert isinstance(plugin_error, Exception)


class TestProfileOverrides:
    """Test profile override functionality."""

    def test_apply_services_overrides(self):
        """Test applying services configuration overrides."""
        config = CodeWeaverConfig()

        # Test services override
        services_config = {
            "chunking": {"max_chunk_size": 2000, "provider": "custom_chunking"},
            "filtering": {"use_gitignore": False},
        }

        config._apply_services_overrides(services_config)

        # Verify overrides were applied
        assert config.services.chunking.max_chunk_size == 2000
        assert config.services.chunking.provider == "custom_chunking"
        assert config.services.filtering.use_gitignore is False

    def test_apply_backend_overrides(self):
        """Test applying backend configuration overrides."""
        config = CodeWeaverConfig()

        # Create a profile with backend overrides
        profile = ProfileConfig(
            name="test",
            description="Test profile",
            backend={"provider": "memory", "enable_hybrid_search": True},
        )

        config._apply_profile_overrides(profile)

        # Check that backend overrides were applied
        assert config.backend.provider == "memory"
        assert config.backend.enable_hybrid_search is True

    def test_apply_indexing_overrides(self):
        """Test applying indexing configuration overrides."""
        config = CodeWeaverConfig()

        profile = ProfileConfig(
            name="test",
            description="Test profile",
            indexing={"batch_size": 32, "enable_auto_reindex": False},
        )

        config._apply_profile_overrides(profile)

        assert config.indexing.batch_size == 32
        assert config.indexing.enable_auto_reindex is False


class TestBackwardsCompatibility:
    """Test backwards compatibility features."""

    def test_existing_config_structure_preserved(self):
        """Test that existing configuration structure is preserved."""
        config = CodeWeaverConfig()

        # Verify all original configuration sections still exist
        assert hasattr(config, "backend")
        assert hasattr(config, "providers")
        assert hasattr(config, "data_sources")
        assert hasattr(config, "chunking")
        assert hasattr(config, "indexing")
        assert hasattr(config, "rate_limiting")
        assert hasattr(config, "server")

        # Verify they have the expected types
        from codeweaver.backends.config import BackendConfigExtended
        from codeweaver.config import (
            ChunkingConfig,
            DataSourceConfig,
            IndexingConfig,
            ProviderConfig,
            RateLimitConfig,
            ServerConfig,
        )

        assert isinstance(config.backend, BackendConfigExtended)
        assert isinstance(config.providers, ProviderConfig)
        assert isinstance(config.data_sources, DataSourceConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.indexing, IndexingConfig)
        assert isinstance(config.rate_limiting, RateLimitConfig)
        assert isinstance(config.server, ServerConfig)

    def test_original_methods_preserved(self):
        """Test that original configuration methods are preserved."""
        config = CodeWeaverConfig()

        # Test that all original methods still exist
        assert hasattr(config, "get_effective_embedding_provider")
        assert hasattr(config, "get_effective_backend_provider")
        assert hasattr(config, "get_effective_backend_url")
        assert hasattr(config, "get_effective_backend_api_key")

        # Test that they return expected types
        backend_provider = config.get_effective_backend_provider()
        assert isinstance(backend_provider, str)


@pytest.mark.integration
@pytest.mark.config
@pytest.mark.slow
class TestConfigIntegration:
    """Integration tests for the enhanced configuration system."""

    def test_full_config_initialization(self):
        """Test full configuration initialization with all components."""
        config_data = {
            "defaults": {"profile": "minimal", "auto_configure": True},
            "plugins": {
                "enabled": True,
                "auto_discover": False,
                "custom": {
                    "test_plugin": {
                        "enabled": True,
                        "plugin_type": "backend",
                        "module_path": "test.plugin",
                        "class_name": "TestBackend",
                        "priority": 80,
                    }
                },
            },
            "factory": {"enable_plugin_discovery": True, "validate_configurations": True},
            "profiles": {
                "custom_minimal": {
                    "name": "custom_minimal",
                    "description": "Custom minimal setup",
                    "backend": {"provider": "memory"},
                }
            },
        }

        config = CodeWeaverConfig(**config_data)

        # Verify all sections are properly initialized
        assert config.defaults.profile == "minimal"
        assert config.plugins.enabled is True
        assert config.plugins.auto_discover is False
        assert "test_plugin" in config.plugins.custom
        assert config.factory.enable_plugin_discovery is True
        assert "custom_minimal" in config.profiles

    def test_profile_auto_application(self):
        """Test automatic profile application."""
        # This would require mocking the profile application since we don't
        # want to actually modify the config during testing
        config = CodeWeaverConfig()
        config.defaults.profile = "performance"
        config.defaults.auto_configure = True

        # Get the performance profile
        performance_profile = config._get_profile("performance")
        assert performance_profile is not None
        assert performance_profile.name == "performance"

        # The actual application would happen during model validation
        # but we can test the profile retrieval mechanism


if __name__ == "__main__":
    pytest.main([__file__])
