# Pydantic-Settings Migration Plan

## Executive Summary

This document outlines a comprehensive plan to leverage pydantic-settings' native capabilities to simplify CodeWeaver's configuration system. The migration will reduce manual implementations, improve maintainability, and add new features while maintaining full backward compatibility.

## Current State Analysis

### Existing Implementation ✅ **Good**
- Already uses `BaseSettings` as the foundation
- Hierarchical configuration with multiple TOML files
- Environment variable support with `CW_` prefix
- Nested configuration models using Pydantic
- Field validation and constraints
- Config saving and validation methods

### Current Issues ❌ **Needs Improvement**
- **Manual TOML loading** - Using `tomllib` directly instead of pydantic-settings built-in support
- **Complex ConfigManager** - 100+ lines of manual file discovery and loading logic
- **Manual environment handling** - Custom logic instead of leveraging pydantic-settings features
- **Limited source customization** - No ability to easily add new config sources
- **Error handling gaps** - Basic error handling without pydantic-settings' robust validation
- **No automatic discovery** - Manual file path management

## Migration Strategy

### Phase 1: Core Infrastructure ⚡ **High Impact**

#### 1.1 Replace Manual TOML Loading
**Current:**
```python
# Manual tomllib loading in ConfigManager._load_config()
import tomllib
with config_file.open("rb") as f:
    toml_data = tomllib.load(f)
return CodeWeaverConfig(**toml_data)
```

**Improved:**
```python
# Use pydantic-settings native TOML support
class CodeWeaverConfig(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=[
            ".local.codeweaver.toml",
            ".codeweaver.toml",
            Path.home() / ".config" / "codeweaver" / "config.toml"
        ]
    )
```

#### 1.2 Simplify ConfigManager Class
**Benefits:**
- Reduce from ~100 lines to ~30 lines
- Remove manual file discovery logic
- Leverage pydantic-settings' automatic loading
- Better error handling and validation

#### 1.3 Enhanced Source Customization
**Current:** Fixed priority and limited sources
**Improved:** Flexible source system with custom priorities

```python
@classmethod
def settings_customize_sources(cls, ...):
    return (
        init_settings,
        TomlConfigSettingsSource(settings_cls, toml_file=cls.model_config.toml_file),
        env_settings,
        file_secret_settings,
    )
```

### Phase 2: Advanced Features 🚀 **Medium Impact**

#### 2.1 Multiple Config Format Support
- **TOML** (current) - Primary format
- **JSON** - For programmatic generation
- **YAML** - For complex configurations
- **Environment-only** - For containerized deployments

#### 2.2 Enhanced Environment Variable Handling
**Current:** Basic `CW_` prefix support
**Improved:**
- Nested delimiter support (`CW_BACKEND__URL`)
- Better type coercion
- Automatic validation

#### 2.3 Secrets Management Integration
- Docker secrets support via `SecretsSettingsSource`
- Azure Key Vault integration for enterprise deployments
- AWS Secrets Manager support

#### 2.4 Configuration Discovery and Validation
- Automatic parent directory searching
- Configuration validation with detailed error reporting
- Schema validation and documentation generation

### Phase 3: Quality and Developer Experience 📈 **Low Impact**

#### 3.1 Enhanced Error Messages
- Detailed validation errors with field context
- Configuration source attribution in error messages
- Helpful suggestions for common misconfigurations

#### 3.2 Developer Tools
- Configuration schema export for documentation
- Example configuration generation
- Configuration debugging and introspection tools

## Technical Implementation Plan

### API Design Principles

1. **Backward Compatibility** - All existing APIs continue to work
2. **Progressive Enhancement** - New features available opt-in
3. **Clear Migration Path** - Easy transition from current implementation
4. **Performance** - No degradation in startup time or memory usage

### New Configuration Class Design

```python
class CodeWeaverConfig(BaseSettings):
    """Enhanced configuration using pydantic-settings native features."""

    model_config = SettingsConfigDict(
        # TOML file configuration
        toml_file=[
            ".local.codeweaver.toml",
            ".codeweaver.toml",
            Path.home() / ".config" / "codeweaver" / "config.toml"
        ],

        # Environment configuration
        env_prefix="CW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",

        # Validation configuration
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        case_sensitive=False,

        # New: Multiple format support
        json_file="codeweaver.json",  # Optional JSON config
        yaml_file="codeweaver.yaml",  # Optional YAML config
    )

    @classmethod
    def settings_customize_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Custom source priority: TOML -> JSON -> YAML -> ENV -> Secrets."""

        return (
            init_settings,  # Highest priority: explicit initialization
            TomlConfigSettingsSource(settings_cls),  # Primary config files
            JsonConfigSettingsSource(settings_cls),  # Secondary: JSON
            YamlConfigSettingsSource(settings_cls),  # Tertiary: YAML
            env_settings,  # Environment variables
            dotenv_settings,  # .env files
            file_secret_settings,  # Secrets (lowest priority)
        )
```

### Simplified ConfigManager

```python
class ConfigManager:
    """Simplified configuration manager using pydantic-settings."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: CodeWeaverConfig | None = None

    def get_config(self) -> CodeWeaverConfig:
        """Get configuration with automatic loading and caching."""
        if self._config is None:
            if self.config_path:
                # Override TOML file paths for specific config
                self._config = CodeWeaverConfig(
                    _toml_file=self.config_path
                )
            else:
                # Use default discovery
                self._config = CodeWeaverConfig()
        return self._config

    def reload_config(self) -> CodeWeaverConfig:
        """Reload configuration."""
        self._config = None
        return self.get_config()
```

## Risk Assessment and Mitigation

### High Risk: Breaking Changes
**Risk:** Existing configurations stop working
**Mitigation:**
- Comprehensive testing with existing config files
- Thorough implementation that considers dependencies

### Medium Risk: Performance Impact
**Risk:** Slower configuration loading
**Mitigation:**
- Performance benchmarking during development
- Caching optimizations
- Lazy loading where appropriate

## Testing Strategy

### Unit Tests
- Test all existing functionality continues to work
- Test new pydantic-settings integration
- Test error conditions and validation

### Integration Tests
- Test with real configuration files
- Test environment variable scenarios
- Test multi-source priority handling

### Migration Tests
- Test existing configs work unchanged
- Test new features work as expected
- Test error scenarios are properly handled

## Success Metrics

### Quantitative Goals
- **Reduce ConfigManager complexity:** From ~100 lines to ~30 lines (70% reduction)
- **Improve test coverage:** From current to 95%+ coverage
- **Maintain performance:** No degradation in config loading time
- **Zero breaking changes:** All existing APIs continue to work

### Qualitative Goals
- **Improved maintainability:** Easier to add new config sources
- **Better error messages:** More helpful validation and debugging
- **Enhanced developer experience:** Clearer configuration patterns
- **Future-proofing:** Easier to add enterprise features

## Implementation Timeline

### Week 1: Foundation
- [ ] Design and review new configuration class
- [ ] Implement simplified ConfigManager
- [ ] Create comprehensive test suite
- [ ] Ensure backward compatibility

### Week 2: Core Features
- [ ] Implement TOML source customization
- [ ] Add enhanced environment variable handling
- [ ] Implement error handling improvements
- [ ] Performance testing and optimization

### Week 3: Advanced Features
- [ ] Add JSON/YAML support (optional)
- [ ] Implement secrets management integration
- [ ] Add configuration validation tools
- [ ] Documentation and examples

### Week 4: Testing and Deployment
- [ ] Comprehensive testing across all scenarios
- [ ] Migration guide and documentation
- [ ] Code review and refinement
- [ ] Deployment and monitoring

## Conclusion

This migration to native pydantic-settings capabilities will significantly simplify the codebase while adding powerful new features. The phased approach ensures we can deliver value incrementally while maintaining stability and backward compatibility.

The end result will be a more maintainable, feature-rich configuration system that leverages the full power of pydantic-settings while preserving all existing functionality.
