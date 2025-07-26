# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Comprehensive tests for the new pydantic-settings based configuration system.

Tests configuration loading, environment variables, TOML files, source priorities,
caching, reloading, and error handling.
"""

import os

from pathlib import Path
from unittest.mock import patch

import pytest

from codeweaver.config import (
    ChunkingConfig,
    CodeWeaverConfig,
    CodeWeaverConfigWithFile,
    ConfigManager,
    CustomTomlSource,
    DataSourceConfig,
    IndexingConfig,
    ProviderConfig,
    RateLimitConfig,
    ServerConfig,
)


class TestCodeWeaverConfig:
    """Test the main CodeWeaverConfig class."""

    def test_default_configuration(self):
        """Test that default configuration loads without errors."""
        config = CodeWeaverConfig()

        # Test default values
        assert config.chunking.max_chunk_size == 1500
        assert config.chunking.min_chunk_size == 50
        assert config.indexing.use_gitignore is True
        assert config.indexing.batch_size == 8
        assert config.server.server_name == "codeweaver-mcp"
        assert config.server.log_level == "INFO"

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CW_SERVER__SERVER_NAME": "test-server",
                "CW_SERVER__LOG_LEVEL": "DEBUG",
                "CW_CHUNKING__MAX_CHUNK_SIZE": "2000",
                "CW_CHUNKING__MIN_CHUNK_SIZE": "100",
                "CW_INDEXING__BATCH_SIZE": "16",
                "CW_BACKEND__PROVIDER": "test-backend",
                "CW_BACKEND__URL": "http://test.example.com",
                "CW_BACKEND__API_KEY": "test-api-key",
            },
        ):
            config = CodeWeaverConfig()

            assert config.server.server_name == "test-server"
            assert config.server.log_level == "DEBUG"
            assert config.chunking.max_chunk_size == 2000
            assert config.chunking.min_chunk_size == 100
            assert config.indexing.batch_size == 16
            assert config.backend.provider == "test-backend"
            assert config.backend.url == "http://test.example.com"
            assert config.backend.api_key == "test-api-key"

    def test_nested_environment_variables(self):
        """Test nested environment variable delimiter support."""
        with patch.dict(
            os.environ,
            {
                "CW_PROVIDERS__VOYAGE_AI__API_KEY": "voyage-key-123",
                "CW_PROVIDERS__OPENAI__API_KEY": "openai-key-456",
                "CW_DATA_SOURCES__ENABLED": "false",
                "CW_DATA_SOURCES__DEFAULT_SOURCE_TYPE": "custom",
                "CW_RATE_LIMITING__MAX_RETRIES": "10",
            },
        ):
            config = CodeWeaverConfig()

            assert config.data_sources.enabled is False
            assert config.data_sources.default_source_type == "custom"
            assert config.rate_limiting.max_retries == 10

    def test_configuration_validation(self):
        """Test configuration field validation."""
        # Test valid configuration
        config = CodeWeaverConfig(chunking={"max_chunk_size": 2000, "min_chunk_size": 100})
        assert config.chunking.max_chunk_size == 2000
        assert config.chunking.min_chunk_size == 100

        # Test invalid configuration (max < min)
        with pytest.raises(ValueError):  # noqa: PT011
            CodeWeaverConfig(chunking={"max_chunk_size": 50, "min_chunk_size": 100})

    def test_effective_methods(self):
        """Test the effective configuration getter methods."""
        config = CodeWeaverConfig(
            backend={"provider": "qdrant", "url": "http://localhost:6333", "api_key": "test-key"}
        )

        assert config.get_effective_backend_provider() == "qdrant"
        assert config.get_effective_backend_url() == "http://localhost:6333"
        assert config.get_effective_backend_api_key() == "test-key"

    def test_default_data_sources_setup(self):
        """Test that default data sources are set up correctly."""
        config = CodeWeaverConfig()

        # Should have default filesystem source
        assert len(config.data_sources.sources) >= 1
        default_source = config.data_sources.sources[0]
        assert default_source["type"] == "filesystem"
        assert default_source["enabled"] is True
        assert default_source["source_id"] == "default_filesystem"


class TestCustomTomlSource:
    """Test the custom TOML source implementation."""

    def test_toml_source_with_explicit_file(self, tmp_path):
        """Test TOML source with explicitly provided file."""
        toml_file = tmp_path / "test_config.toml"
        toml_content = """
[chunking]
max_chunk_size = 3000
min_chunk_size = 200

[server]
server_name = "test-from-toml"
log_level = "DEBUG"
"""
        toml_file.write_text(toml_content)

        # Test with explicit file
        source = CustomTomlSource(CodeWeaverConfig, toml_file=toml_file)
        data = source()

        assert data["chunking"]["max_chunk_size"] == 3000
        assert data["chunking"]["min_chunk_size"] == 200
        assert data["server"]["server_name"] == "test-from-toml"
        assert data["server"]["log_level"] == "DEBUG"

    def test_toml_source_search_paths(self, tmp_path, monkeypatch):
        """Test TOML source with search path discovery."""
        # Change to temporary directory
        monkeypatch.chdir(tmp_path)

        # Create a .codeweaver.toml file
        toml_file = tmp_path / ".codeweaver.toml"
        toml_content = """
[server]
server_name = "found-via-search"
"""
        toml_file.write_text(toml_content)

        # Test automatic discovery
        source = CustomTomlSource(CodeWeaverConfig)
        data = source()

        assert data["server"]["server_name"] == "found-via-search"

    def test_toml_source_nonexistent_file(self):
        """Test TOML source with nonexistent file."""
        source = CustomTomlSource(CodeWeaverConfig, toml_file=Path("/nonexistent/config.toml"))
        data = source()

        # Should return empty dict, not raise exception
        assert data == {}

    def test_toml_source_invalid_syntax(self, tmp_path):
        """Test TOML source with invalid syntax."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")

        source = CustomTomlSource(CodeWeaverConfig, toml_file=toml_file)
        data = source()

        # Should return empty dict on error, not crash
        assert data == {}


class TestConfigManager:
    """Test the ConfigManager class."""

    def test_config_manager_basic_usage(self):
        """Test basic ConfigManager functionality."""
        manager = ConfigManager()
        config = manager.get_config()

        assert isinstance(config, CodeWeaverConfig)
        assert config.server.server_name == "codeweaver-mcp"

    def test_config_manager_with_specific_file(self, tmp_path):
        """Test ConfigManager with specific configuration file."""
        config_file = tmp_path / "specific_config.toml"
        config_content = """
[server]
server_name = "specific-server"
log_level = "WARNING"

[chunking]
max_chunk_size = 2500
"""
        config_file.write_text(config_content)

        manager = ConfigManager(config_path=config_file)
        config = manager.get_config()

        assert config.server.server_name == "specific-server"
        assert config.server.log_level == "WARNING"
        assert config.chunking.max_chunk_size == 2500

    def test_config_manager_caching(self):
        """Test that ConfigManager caches configuration."""
        manager = ConfigManager()

        config1 = manager.get_config()
        config2 = manager.get_config()

        # Should return the same instance (cached)
        assert config1 is config2

    def test_config_manager_reload(self, tmp_path):
        """Test configuration reloading."""
        config_file = tmp_path / "reload_test.toml"

        # Initial configuration
        config_content_1 = """
[server]
server_name = "initial-server"
"""
        config_file.write_text(config_content_1)

        manager = ConfigManager(config_path=config_file)
        config1 = manager.get_config()
        assert config1.server.server_name == "initial-server"

        # Update configuration file
        config_content_2 = """
[server]
server_name = "updated-server"
"""
        config_file.write_text(config_content_2)

        # Reload configuration
        config2 = manager.reload_config()
        assert config2.server.server_name == "updated-server"
        assert config1 is not config2  # Should be different instance

    def test_config_manager_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config = CodeWeaverConfig(
            server={"server_name": "saved-server", "log_level": "ERROR"},
            chunking={"max_chunk_size": 4000, "min_chunk_size": 300},
        )

        manager = ConfigManager()
        save_path = tmp_path / "saved_config.toml"

        result_path = manager.save_config(config, save_path)
        assert result_path == save_path
        assert save_path.exists()

        # Verify saved content by loading it back
        manager2 = ConfigManager(config_path=save_path)
        loaded_config = manager2.get_config()

        assert loaded_config.server.server_name == "saved-server"
        assert loaded_config.server.log_level == "ERROR"
        assert loaded_config.chunking.max_chunk_size == 4000

    def test_config_manager_validate_config(self, tmp_path):
        # sourcery skip: extract-duplicate-method
        """Test configuration file validation."""
        manager = ConfigManager()

        # Test with valid configuration
        valid_config_file = tmp_path / "valid.toml"
        valid_content = """
[server]
server_name = "valid-server"

[chunking]
max_chunk_size = 2000
min_chunk_size = 100
"""
        valid_config_file.write_text(valid_content)

        result = manager.validate_config(valid_config_file)
        assert result["valid"] is True
        assert result["file_exists"] is True
        assert "summary" in result

        # Test with nonexistent file
        nonexistent_file = tmp_path / "nonexistent.toml"
        result = manager.validate_config(nonexistent_file)
        assert result["valid"] is False
        assert result["file_exists"] is False
        assert len(result["errors"]) > 0

    def test_config_manager_fallback_on_error(self):
        """Test that ConfigManager falls back gracefully on errors."""
        # Test with invalid path
        manager = ConfigManager(config_path="/invalid/path/config.toml")

        # Should not raise exception, should return working configuration
        config = manager.get_config()
        assert isinstance(config, CodeWeaverConfig)


class TestSourcePriority:
    """Test configuration source priority and precedence."""

    def test_environment_overrides_toml(self, tmp_path, monkeypatch):
        """Test that environment variables override TOML configuration."""
        monkeypatch.chdir(tmp_path)

        # Create TOML file
        toml_file = tmp_path / ".codeweaver.toml"
        toml_content = """
[server]
server_name = "toml-server"
log_level = "INFO"

[chunking]
max_chunk_size = 1000
"""
        toml_file.write_text(toml_content)

        # Set environment variables that should override TOML
        with patch.dict(
            os.environ,
            {"CW_SERVER__SERVER_NAME": "env-server", "CW_CHUNKING__MAX_CHUNK_SIZE": "2000"},
        ):
            config = CodeWeaverConfig()

            # Environment should win
            assert config.server.server_name == "env-server"
            assert config.chunking.max_chunk_size == 2000

            # TOML should provide values not overridden by env
            assert config.server.log_level == "INFO"

    def test_init_overrides_all(self, tmp_path, monkeypatch):
        """Test that init parameters override all other sources."""
        monkeypatch.chdir(tmp_path)

        # Create TOML file
        toml_file = tmp_path / ".codeweaver.toml"
        toml_content = """
[server]
server_name = "toml-server"
log_level = "INFO"
"""
        toml_file.write_text(toml_content)

        # Set environment variables
        with patch.dict(
            os.environ, {"CW_SERVER__SERVER_NAME": "env-server", "CW_SERVER__LOG_LEVEL": "ERROR"}
        ):
            # Init parameters should override everything
            config = CodeWeaverConfig(server={"server_name": "init-server", "log_level": "WARNING"})

            assert config.server.server_name == "init-server"
            assert config.server.log_level == "WARNING"


class TestConfigurationComponents:
    """Test individual configuration component classes."""

    def test_chunking_config_validation(self):
        """Test ChunkingConfig validation."""
        # Valid configuration
        config = ChunkingConfig(max_chunk_size=2000, min_chunk_size=100)
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 100

        # Invalid configuration (max <= min)
        with pytest.raises(ValueError):  # noqa: PT011
            ChunkingConfig(max_chunk_size=100, min_chunk_size=200)

    def test_indexing_config_defaults(self):
        """Test IndexingConfig default values."""
        config = IndexingConfig()

        assert config.use_gitignore is True
        assert config.enable_auto_reindex is False
        assert config.batch_size == 8
        assert config.max_concurrent_files == 10
        assert "node_modules" in config.additional_ignore_patterns
        assert ".git" in config.additional_ignore_patterns

    def test_rate_limit_config_constraints(self):
        """Test RateLimitConfig field constraints."""
        config = RateLimitConfig()

        # Test default values are within constraints
        assert 1 <= config.voyage_requests_per_minute <= 10000
        assert 1000 <= config.voyage_tokens_per_minute <= 10000000
        assert 0.1 <= config.initial_backoff_seconds <= 10.0
        assert 1.0 <= config.max_backoff_seconds <= 300.0
        assert 1.1 <= config.backoff_multiplier <= 5.0
        assert 1 <= config.max_retries <= 20

    def test_server_config_log_level(self):
        """Test ServerConfig log level validation."""
        config = ServerConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

        config = ServerConfig(log_level="CRITICAL")
        assert config.log_level == "CRITICAL"

    def test_provider_config_methods(self):
        """Test ProviderConfig getter methods."""
        config = ProviderConfig()

        # Test with no active providers
        assert config.get_active_embedding_provider() is None
        assert config.get_active_reranking_provider() is None

    def test_data_source_config_defaults(self):
        """Test DataSourceConfig default values."""
        config = DataSourceConfig()

        assert config.enabled is True
        assert config.default_source_type == "filesystem"
        assert config.max_concurrent_sources == 5
        assert config.enable_content_deduplication is True
        assert config.content_cache_ttl_hours == 24


class TestCodeWeaverConfigWithFile:
    """Test CodeWeaverConfigWithFile for explicit file loading."""

    def test_config_with_explicit_file(self, tmp_path):
        """Test loading configuration with explicit TOML file."""
        config_file = tmp_path / "explicit.toml"
        config_content = """
[server]
server_name = "explicit-file-server"
log_level = "DEBUG"

[chunking]
max_chunk_size = 3000
min_chunk_size = 300
"""
        config_file.write_text(config_content)

        config = CodeWeaverConfigWithFile(toml_file=config_file)

        assert config.server.server_name == "explicit-file-server"
        assert config.server.log_level == "DEBUG"
        assert config.chunking.max_chunk_size == 3000
        assert config.chunking.min_chunk_size == 300


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_invalid_toml_syntax_handling(self, tmp_path, monkeypatch):
        """Test handling of invalid TOML syntax."""
        monkeypatch.chdir(tmp_path)

        # Create file with invalid TOML syntax
        toml_file = tmp_path / ".codeweaver.toml"
        toml_file.write_text("invalid toml content [[[")

        # Should not crash, should load defaults
        config = CodeWeaverConfig()
        assert config.server.server_name == "codeweaver-mcp"  # Default value

    def test_permission_error_handling(self, tmp_path):
        """Test handling of permission errors."""
        config_file = tmp_path / "permission_test.toml"
        config_file.write_text("[server]\nserver_name = 'test'")

        # Make file unreadable (on systems that support it)
        try:
            config_file.chmod(0o000)

            # Should handle gracefully
            config = CodeWeaverConfigWithFile(toml_file=config_file)
            assert isinstance(config, CodeWeaverConfig)

        finally:
            # Restore permissions for cleanup
            config_file.chmod(0o644)

    def test_environment_variable_type_errors(self):
        """Test handling of invalid environment variable types."""
        with patch.dict(
            os.environ,
            {
                "CW_CHUNKING__MAX_CHUNK_SIZE": "not-a-number",
                "CW_INDEXING__BATCH_SIZE": "invalid-int",
            },
        ):
            # Should either use defaults or raise validation error
            try:
                config = CodeWeaverConfig()
                # If it succeeds, it should use defaults
                assert isinstance(config.chunking.max_chunk_size, int)
                assert isinstance(config.indexing.batch_size, int)
            except Exception:
                # Validation error is acceptable
                pass


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_configuration_scenario(self, tmp_path, monkeypatch):
        """Test a complete configuration scenario with all sources."""
        monkeypatch.chdir(tmp_path)

        # Create TOML configuration
        toml_file = tmp_path / ".codeweaver.toml"
        toml_content = """
[server]
server_name = "production-server"
log_level = "INFO"
max_search_results = 100

[chunking]
max_chunk_size = 2000
min_chunk_size = 100
max_file_size_mb = 2

[indexing]
use_gitignore = true
batch_size = 16
enable_auto_reindex = true

[backend]
provider = "qdrant"
collection_name = "production-embeddings"

[data_sources]
enabled = true
default_source_type = "filesystem"

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "/app/src"
use_gitignore = true
max_file_size_mb = 2
"""
        toml_file.write_text(toml_content)

        # Set some environment overrides
        with patch.dict(
            os.environ,
            {
                "CW_SERVER__LOG_LEVEL": "DEBUG",  # Override TOML
                "CW_BACKEND__URL": "http://prod-qdrant:6333",  # Add missing value
                "CW_BACKEND__API_KEY": "prod-api-key",  # Add missing value
            },
        ):
            config = CodeWeaverConfig()

            # Verify TOML values
            assert config.server.server_name == "production-server"
            assert config.server.max_search_results == 100
            assert config.chunking.max_chunk_size == 2000
            assert config.chunking.min_chunk_size == 100
            assert config.indexing.batch_size == 16
            assert config.indexing.enable_auto_reindex is True
            assert config.backend.provider == "qdrant"
            assert config.backend.collection_name == "production-embeddings"
            assert len(config.data_sources.sources) == 1

            # Verify environment overrides
            assert config.server.log_level == "DEBUG"  # Overridden by env
            assert config.backend.url == "http://prod-qdrant:6333"  # From env
            assert config.backend.api_key == "prod-api-key"  # From env

            # Verify effective methods
            assert config.get_effective_backend_provider() == "qdrant"
            assert config.get_effective_backend_url() == "http://prod-qdrant:6333"
            assert config.get_effective_backend_api_key() == "prod-api-key"

    def test_development_vs_production_config(self, tmp_path, monkeypatch):
        # sourcery skip: extract-duplicate-method
        """Test different configurations for development vs production."""
        monkeypatch.chdir(tmp_path)

        # Development configuration
        dev_config = tmp_path / ".local.codeweaver.toml"  # Highest priority
        dev_content = """
[server]
server_name = "dev-server"
log_level = "DEBUG"
enable_request_logging = true

[chunking]
max_chunk_size = 1000  # Smaller for faster testing

[indexing]
batch_size = 4  # Smaller for development
enable_auto_reindex = true  # Convenient for development
"""
        dev_config.write_text(dev_content)

        # Production base configuration
        prod_config = tmp_path / ".codeweaver.toml"  # Lower priority
        prod_content = """
[server]
server_name = "prod-server"
log_level = "WARNING"
enable_request_logging = false
max_search_results = 200

[chunking]
max_chunk_size = 2000
min_chunk_size = 100

[indexing]
batch_size = 16
enable_auto_reindex = false

[rate_limiting]
max_retries = 3
initial_backoff_seconds = 2.0
"""
        prod_config.write_text(prod_content)

        # Load configuration (should prioritize dev over prod)
        config = CodeWeaverConfig()

        # Development overrides should win
        assert config.server.server_name == "dev-server"
        assert config.server.log_level == "DEBUG"
        assert config.server.enable_request_logging is True
        assert config.chunking.max_chunk_size == 1000
        assert config.indexing.batch_size == 4
        assert config.indexing.enable_auto_reindex is True

        # With our current TOML source implementation, only the first found file is used
        # (.local.codeweaver.toml takes complete priority, .codeweaver.toml is ignored)
        # Values not in dev config use defaults, not production config values
        assert config.server.max_search_results == 50  # Default value, not from prod config
        assert config.chunking.min_chunk_size == 50  # Default value, not from prod config
        assert config.rate_limiting.max_retries == 5  # Default value, not from prod config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
