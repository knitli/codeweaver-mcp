<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Pydantic-Settings Integration Specification

## Overview

This specification defines the technical implementation details for integrating native pydantic-settings capabilities into CodeWeaver's configuration system. The focus is on API quality, maintainability, and leveraging pydantic-settings' built-in features to reduce manual implementations.

## Current vs. Improved Architecture

### Current Architecture Issues

```python
# Current: Manual TOML loading
class ConfigManager:
    def _load_config(self) -> CodeWeaverConfig:
        toml_data = {}
        for config_file in search_paths:
            if config_file.exists():
                try:
                    import tomllib
                    with config_file.open("rb") as f:
                        toml_data = tomllib.load(f)
                    break
                except Exception as e:
                    logger.warning("Failed to load config from %s: %s", config_file, e)
        return CodeWeaverConfig(**toml_data)
```

**Problems:**
- Manual file discovery and loading logic
- Basic error handling without context
- Limited to TOML format only
- No automatic source priority management
- Redundant with pydantic-settings capabilities

### Improved Architecture

```python
# Improved: Native pydantic-settings integration
class CodeWeaverConfig(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=[
            ".local.codeweaver.toml",
            ".codeweaver.toml",
            Path.home() / ".config" / "codeweaver" / "config.toml"
        ],
        env_prefix="CW_",
        env_nested_delimiter="__",
        extra="allow",
        validate_assignment=True,
    )

    @classmethod
    def settings_customize_sources(cls, ...):
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )
```

**Benefits:**
- Native TOML loading with automatic discovery
- Built-in error handling and validation
- Extensible source system
- Better environment variable support
- Reduced code complexity

## API Design Specification

### 1. Enhanced CodeWeaverConfig Class

#### 1.1 Core Configuration
```python
class CodeWeaverConfig(BaseSettings):
    """Enhanced configuration leveraging pydantic-settings native features.

    This class maintains full backward compatibility while adding new capabilities:
    - Native TOML file loading with automatic discovery
    - Enhanced environment variable handling with nested delimiters
    - Multiple configuration source support with customizable priority
    - Built-in validation and error handling
    - Optional support for JSON, YAML, and secrets
    """

    model_config = SettingsConfigDict(
        # File-based configuration
        toml_file=[
            ".local.codeweaver.toml",      # Workspace local (highest precedence)
            ".codeweaver.toml",            # Repository level
            Path.home() / ".config" / "codeweaver" / "config.toml"  # User level
        ],

        # Environment variable configuration
        env_prefix="CW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,

        # Validation and behavior
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        case_sensitive=False,
        validate_default=True,

        json_file="codeweaver.json",
        yaml_file="codeweaver.yaml",
        secrets_dir="/run/secrets",
    )
```

#### 1.2 Custom Source Configuration
```python
@classmethod
def settings_customize_sources(
    cls,
    settings_cls: type[BaseSettings],
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
) -> tuple[PydanticBaseSettingsSource, ...]:
    """Customize configuration source priority and add new sources.

    Priority (highest to lowest):
    1. Explicit initialization parameters
    2. TOML configuration files
    3. Environment variables
    4. .env files
    5. Secrets directory

    This ordering ensures that explicit parameters take precedence,
    followed by configuration files, then environment variables.
    """

    # Create TOML source with enhanced error handling
    toml_source = TomlConfigSettingsSource(
        settings_cls,
        toml_file=cls.model_config.get('toml_file')
    )

    # Optional: Add JSON/YAML sources (future enhancement)
    # json_source = JsonConfigSettingsSource(settings_cls)
    # yaml_source = YamlConfigSettingsSource(settings_cls)

    return (
        init_settings,        # Explicit parameters (highest priority)
        toml_source,          # TOML configuration files
        env_settings,         # Environment variables
        dotenv_settings,      # .env files
        file_secret_settings, # Secrets (lowest priority)
    )
```

#### 1.3 Enhanced Validation and Error Handling
```python
@model_validator(mode="after")
def validate_configuration_consistency(self) -> "CodeWeaverConfig":
    """Validate configuration consistency across all sources.

    This validator ensures that:
    - Required provider configurations are complete
    - Backend and provider configurations are compatible
    - Default data sources are properly configured
    - Configuration values are semantically valid
    """
    # Existing validation logic enhanced with better error messages
    return self

@classmethod
def create_with_validation(
    cls,
    config_path: str | Path | None = None,
    validate_files: bool = True
) -> "CodeWeaverConfig":
    """Create configuration with enhanced validation and error reporting.

    Args:
        config_path: Optional specific configuration file path
        validate_files: Whether to validate configuration file existence

    Returns:
        Validated configuration instance

    Raises:
        SettingsError: If configuration is invalid with detailed error context
    """
    try:
        if config_path:
            # Override TOML file discovery for specific path
            return cls(_toml_file=config_path)
        else:
            # Use automatic discovery
            return cls()
    except ValidationError as e:
        # Enhanced error reporting with source attribution
        raise SettingsError(f"Configuration validation failed: {e}") from e
```

### 2. Simplified ConfigManager Class

#### 2.1 Streamlined Implementation
```python
class ConfigManager:
    """Simplified configuration manager leveraging pydantic-settings.

    This class provides a clean interface for configuration management
    while delegating the heavy lifting to pydantic-settings' native capabilities.

    Key improvements:
    - Reduced from ~100 lines to ~30 lines
    - Automatic configuration discovery and loading
    - Enhanced error handling and validation
    - Better caching and reload semantics
    """

    def __init__(self, config_path: str | Path | None = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to specific configuration file.
                        If provided, this overrides automatic discovery.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: CodeWeaverConfig | None = None
        self._config_source: str | None = None

    def get_config(self) -> CodeWeaverConfig:
        """Get current configuration with automatic loading and caching.

        Returns:
            Configuration instance with all sources loaded and validated

        Note:
            Configuration is cached until explicitly reloaded.
            First call performs discovery and validation.
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> CodeWeaverConfig:
        """Load configuration using pydantic-settings native capabilities."""
        try:
            if self.config_path:
                # Use specific configuration file
                config = CodeWeaverConfig(_toml_file=self.config_path)
                self._config_source = str(self.config_path)
            else:
                # Use automatic discovery
                config = CodeWeaverConfig()
                self._config_source = "automatic_discovery"

            logger.info("Configuration loaded from: %s", self._config_source)
            return config

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def reload_config(self) -> CodeWeaverConfig:
        """Reload configuration, clearing cache.

        Returns:
            Freshly loaded configuration instance
        """
        self._config = None
        self._config_source = None
        return self.get_config()

    def get_config_info(self) -> dict[str, Any]:
        """Get information about current configuration sources and status.

        Returns:
            Dictionary with configuration metadata including:
            - Source information
            - Loading status
            - Validation results
            - Available configuration files
        """
        config = self.get_config()

        return {
            "source": self._config_source,
            "loaded": self._config is not None,
            "available_files": self._discover_config_files(),
            "environment_variables": self._get_env_vars(),
            "validation_status": "valid"  # Could be enhanced with detailed validation info
        }

    def _discover_config_files(self) -> list[str]:
        """Discover available configuration files."""
        search_paths = [
            Path(".local.codeweaver.toml"),
            Path(".codeweaver.toml"),
            Path.home() / ".config" / "codeweaver" / "config.toml"
        ]
        return [str(p) for p in search_paths if p.exists()]

    def _get_env_vars(self) -> dict[str, str]:
        """Get relevant environment variables."""
        import os
        return {k: v for k, v in os.environ.items() if k.startswith("CW_")}
```

#### 2.2 Enhanced Configuration Validation
```python
def validate_config(self, config_path: str | Path) -> dict[str, Any]:
    """Validate specific configuration file with detailed reporting.

    Args:
        config_path: Path to configuration file to validate

    Returns:
        Detailed validation results with errors, warnings, and suggestions
    """
    path = Path(config_path)
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "file_exists": path.exists(),
        "file_path": str(path),
        "sources_used": [],
    }

    if not path.exists():
        result["errors"].append(f"Configuration file does not exist: {path}")
        result["suggestions"].append("Create the file or check the path")
        return result

    try:
        # Use pydantic-settings validation
        config = CodeWeaverConfig(_toml_file=path)
        result["valid"] = True
        result["sources_used"] = ["toml_file", "environment", "defaults"]

        # Additional validation checks
        if not config.backend.url:
            result["warnings"].append("Backend URL not configured")
            result["suggestions"].append("Set CW_BACKEND__URL environment variable")

    except ValidationError as e:
        result["errors"].extend([str(err) for err in e.errors()])
        result["suggestions"].append("Check configuration file syntax and required fields")
    except Exception as e:
        result["errors"].append(f"Unexpected error: {e}")

    return result

def save_config(self, config: CodeWeaverConfig, config_path: str | Path | None = None) -> Path:
    """Save configuration to TOML file with validation.

    Args:
        config: Configuration instance to save
        config_path: Optional path to save to. If None, uses user config location.

    Returns:
        Path where configuration was saved

    Note:
        Uses model_dump with exclude_unset=True to only save non-default values.
        This keeps configuration files clean and focused.
    """
    if config_path is None:
        save_path = Path.home() / ".config" / "codeweaver" / "config.toml"
    else:
        save_path = Path(config_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Enhanced serialization with pydantic-settings support
    config_data = config.model_dump(
        exclude_unset=True,    # Only save explicitly set values
        exclude_defaults=False, # Include defaults that were explicitly set
        mode="json"            # Ensure JSON-serializable output
    )

    # Use tomli_w for TOML serialization
    import tomli_w
    with save_path.open("wb") as f:
        tomli_w.dump(config_data, f)

    logger.info("Configuration saved to: %s", save_path)
    return save_path
```

### 3. Advanced Features Specification

#### 3.1 Multiple Format Support (Future Enhancement)
```python
# Optional: Enable additional configuration formats
class CodeWeaverConfigExtended(CodeWeaverConfig):
    """Extended configuration with multiple format support."""

    model_config = SettingsConfigDict(
        **CodeWeaverConfig.model_config,

        # Additional format support
        json_file="codeweaver.json",
        yaml_file="codeweaver.yaml",
    )

    @classmethod
    def settings_customize_sources(cls, ...):
        """Extended source configuration with multiple formats."""
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            JsonConfigSettingsSource(settings_cls),
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
```

#### 3.2 Enterprise Features (Future Enhancement)
```python
# Optional: Enterprise configuration with secrets management
class CodeWeaverConfigEnterprise(CodeWeaverConfig):
    """Enterprise configuration with secrets management support."""

    @classmethod
    def settings_customize_sources(cls, ...):
        """Enterprise source configuration with secrets management."""

        # Optional secrets sources (can be enabled per deployment)
        azure_kv_source = None
        aws_secrets_source = None
        docker_secrets_source = None

        # Detect available secrets sources
        if os.getenv("AZURE_KEY_VAULT_URL"):
            azure_kv_source = AzureKeyVaultSettingsSource(
                settings_cls,
                os.environ["AZURE_KEY_VAULT_URL"],
                DefaultAzureCredential()
            )

        if os.getenv("AWS_SECRETS_MANAGER_SECRET_ID"):
            aws_secrets_source = AWSSecretsManagerSettingsSource(
                settings_cls,
                os.environ["AWS_SECRETS_MANAGER_SECRET_ID"]
            )

        if Path("/run/secrets").exists():
            docker_secrets_source = SecretsSettingsSource(
                settings_cls,
                secrets_dir="/run/secrets"
            )

        # Build source tuple dynamically
        sources = [
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]

        # Add secrets sources if available
        for source in [azure_kv_source, aws_secrets_source, docker_secrets_source]:
            if source:
                sources.append(source)

        return tuple(sources)
```

## Quality Assurance Specification

### 1. Backward Compatibility Requirements

#### 1.1 API Compatibility
- All existing `ConfigManager` methods must continue to work
- All existing `CodeWeaverConfig` attributes must remain accessible
- All existing environment variables must continue to work
- All existing TOML configuration files must load correctly

#### 1.2 Behavior Compatibility
- Configuration loading order must remain the same
- Default values must remain unchanged
- Validation rules must remain the same or become more permissive
- Error messages should improve but not break existing error handling

### 2. Performance Requirements

#### 2.1 Loading Performance
- Configuration loading time must not increase by more than 10%
- Memory usage should remain similar or decrease
- First-time loading should complete within 100ms for typical configurations

#### 2.2 Caching Performance
- Cached configuration access should remain sub-millisecond
- Configuration reload should complete within 50ms
- Multiple instances should not cause memory bloat

### 3. Error Handling Requirements

#### 3.1 Validation Errors
- Must provide clear, actionable error messages
- Should indicate which configuration source caused the error
- Must include suggestions for fixing common issues
- Should gracefully handle partial configuration failures

#### 3.2 File System Errors
- Must handle missing configuration files gracefully
- Should provide helpful guidance for file permission issues
- Must handle invalid TOML syntax with clear error reporting
- Should validate file paths and provide existence checks

## Testing Specification

### 1. Unit Test Requirements
- Test all existing functionality continues to work unchanged
- Test new pydantic-settings integration features
- Test error conditions and edge cases
- Test configuration source priority handling
- Test environment variable parsing with nested delimiters

### 2. Integration Test Requirements
- Test with real configuration files in various formats
- Test environment variable scenarios and precedence
- Test multi-source loading and priority resolution
- Test configuration validation and error reporting
- Test performance under various load conditions

### 3. Migration Test Requirements
- Test that existing configuration files load correctly
- Test that existing environment variables work unchanged
- Test that existing code using the configuration continues to work
- Test new features work as expected without breaking existing functionality

## Implementation Guidelines

### 1. Development Principles
- **Backward Compatibility First** - Never break existing functionality
- **Progressive Enhancement** - Add new features incrementally
- **Clear Error Messages** - Provide helpful, actionable error reporting
- **Performance Awareness** - Monitor and optimize configuration loading
- **Comprehensive Testing** - Test all scenarios thoroughly

### 2. Code Quality Standards
- Follow existing type annotation standards
- Use pydantic's modern typing features (Self, Literal, etc.)
- Implement comprehensive docstrings with examples
- Follow Google docstring convention
- Maintain line length limit of 100 characters

### 3. Documentation Requirements
- Update all docstrings to reflect new capabilities
- Create migration guide for users
- Document new configuration options and formats
- Provide examples for common use cases
- Update architectural documentation

This specification provides the technical foundation for implementing pydantic-settings integration while maintaining high quality, backward compatibility, and enhanced functionality.
