# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Universal plugin interface protocols for the CodeWeaver factory system.

Provides plugin interfaces, discovery mechanisms, and validation frameworks
that support extensible backends, providers, and sources.
"""

import logging

from abc import abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from codeweaver.backends.base import VectorBackend
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.sources.base import DataSource
from codeweaver.types import (
    BaseCapabilities,
    BaseComponentConfig,
    BaseComponentInfo,
    ComponentType,
    PluginInfo,
    ValidationResult,
)


logger = logging.getLogger(__name__)


@runtime_checkable
class PluginInterface(Protocol):
    """Universal interface that all plugins must implement."""

    @classmethod
    @abstractmethod
    def get_plugin_name(cls) -> str:
        """Get the unique name for this plugin."""
        ...

    @classmethod
    @abstractmethod
    def get_component_type(cls) -> ComponentType:
        """Get the type of component this plugin provides."""
        ...

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> BaseCapabilities:
        """Get the capabilities this plugin provides."""
        ...

    @classmethod
    @abstractmethod
    def get_component_info(cls) -> BaseComponentInfo:
        """Get detailed information about this plugin."""
        ...

    @classmethod
    @abstractmethod
    def validate_config(cls, config: BaseComponentConfig) -> ValidationResult:
        """Validate configuration for this plugin."""
        ...

    @classmethod
    @abstractmethod
    def get_dependencies(cls) -> list[str]:
        """Get list of required dependencies for this plugin."""
        ...


class BackendPlugin(PluginInterface):
    """Specific interface for backend plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        """Get the type of component this plugin provides."""
        return ComponentType.BACKEND

    @classmethod
    @abstractmethod
    def get_backend_class(cls) -> type[VectorBackend]:
        """Get the backend implementation class."""
        ...


class ProviderPlugin(PluginInterface):
    """Specific interface for provider plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        """Get the type of component this plugin provides."""
        return ComponentType.PROVIDER

    @classmethod
    @abstractmethod
    def get_provider_class(cls) -> type[EmbeddingProvider | RerankProvider]:
        """Get the provider implementation class."""
        ...


class SourcePlugin(PluginInterface):
    """Specific interface for source plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        """Get the type of component this plugin provides."""
        return ComponentType.SOURCE

    @classmethod
    @abstractmethod
    def get_source_class(cls) -> type[DataSource]:
        """Get the source implementation class."""
        ...


class PluginValidator:
    """Comprehensive plugin validation."""

    def validate_plugin(self, plugin_info: PluginInfo) -> ValidationResult:
        """Multi-stage plugin validation."""
        errors = []
        warnings = []
        try:
            if not isinstance(plugin_info.plugin_class, type):
                errors.append("Plugin class must be a type")
                return ValidationResult(is_valid=False, errors=errors)
            if not hasattr(plugin_info.plugin_class, "get_plugin_name"):
                errors.append("Plugin must implement PluginInterface protocol")
                return ValidationResult(is_valid=False, errors=errors)
            validation_stages = [
                self._validate_plugin_interface,
                self._validate_plugin_capabilities,
                self._validate_plugin_dependencies,
                self._validate_plugin_configuration,
            ]
            for stage in validation_stages:
                result = stage(plugin_info)
                if not result.is_valid:
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    if result.errors:
                        break
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"Plugin validation failed: {e}"])
        else:
            return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_plugin_interface(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin interface compliance."""
        warnings = []
        required_methods = [
            "get_plugin_name",
            "get_component_type",
            "get_capabilities",
            "get_component_info",
            "validate_config",
            "get_dependencies",
        ]
        errors = [
            f"Plugin missing required method: {method}"
            for method in required_methods
            if not hasattr(plugin_info.plugin_class, method)
        ]
        if plugin_info.component_type == ComponentType.BACKEND:
            if not hasattr(plugin_info.plugin_class, "get_backend_class"):
                errors.append("Backend plugin missing get_backend_class method")
        elif plugin_info.component_type == ComponentType.PROVIDER:
            if not hasattr(plugin_info.plugin_class, "get_provider_class"):
                errors.append("Provider plugin missing get_provider_class method")
        elif plugin_info.component_type == ComponentType.SOURCE and (
            not hasattr(plugin_info.plugin_class, "get_source_class")
        ):
            errors.append("Source plugin missing get_source_class method")
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_plugin_capabilities(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin capabilities consistency."""
        errors = []
        warnings = []
        try:
            capabilities = plugin_info.plugin_class.get_capabilities()
            if not isinstance(capabilities, BaseCapabilities):
                errors.append("Plugin capabilities must inherit from BaseCapabilities")
        except Exception as e:
            errors.append(f"Failed to get plugin capabilities: {e}")
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_plugin_dependencies(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin dependencies."""
        errors = []
        warnings = []
        try:
            dependencies = plugin_info.plugin_class.get_dependencies()
            if not isinstance(dependencies, list):
                errors.append("Plugin dependencies must be a list")
                return ValidationResult(is_valid=False, errors=errors)
            for dep in dependencies:
                if not isinstance(dep, str):
                    errors.append(f"Dependency must be a string, got {type(dep)}")
                try:
                    __import__(dep)
                except ImportError:
                    warnings.append(f"Dependency '{dep}' may not be installed")
        except Exception as e:
            errors.append(f"Failed to validate plugin dependencies: {e}")
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_plugin_configuration(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin configuration handling."""
        errors = []
        warnings = []
        try:
            test_config = BaseComponentConfig(
                component_type=plugin_info.component_type, provider=plugin_info.name
            )
            result = plugin_info.plugin_class.validate_config(test_config)
            if not isinstance(result, ValidationResult):
                errors.append("Plugin validate_config must return ValidationResult")
        except Exception as e:
            errors.append(f"Plugin configuration validation failed: {e}")
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


class PluginSecurityValidator:
    """Security validation for plugins."""

    def validate_plugin_security(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate plugin security requirements."""
        errors = []
        warnings = []
        security_checks = [
            self._check_code_signing,
            self._check_dependency_security,
            self._check_permission_requirements,
            self._check_data_access_patterns,
        ]
        for check in security_checks:
            try:
                result = check(plugin_info)
                if not result.is_valid:
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
            except Exception as e:
                warnings.append(f"Security check failed: {e}")
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _check_code_signing(self, plugin_info: PluginInfo) -> ValidationResult:
        """Check plugin code signature validation."""
        return ValidationResult(is_valid=True, warnings=["Code signing validation not implemented"])

    def _check_dependency_security(self, plugin_info: PluginInfo) -> ValidationResult:
        """Check plugin dependency security."""
        return ValidationResult(
            is_valid=True, warnings=["Dependency security validation not implemented"]
        )

    def _check_permission_requirements(self, plugin_info: PluginInfo) -> ValidationResult:
        """Check plugin permission requirements."""
        return ValidationResult(is_valid=True, warnings=["Permission validation not implemented"])

    def _check_data_access_patterns(self, plugin_info: PluginInfo) -> ValidationResult:
        """Check plugin data access patterns."""
        return ValidationResult(is_valid=True, warnings=["Data access validation not implemented"])


class PluginDiscoveryEngine:
    """Advanced plugin discovery with multiple source support."""

    def __init__(
        self,
        plugin_directories: list[str] | None = None,
        *,
        enable_entry_points: bool = True,
        enable_directory_scan: bool = True,
        enable_module_scan: bool = True,
    ):
        """Initialize the plugin discovery engine."""
        self.plugin_directories = plugin_directories or []
        self.enable_entry_points = enable_entry_points
        self.enable_directory_scan = enable_directory_scan
        self.enable_module_scan = enable_module_scan
        self._validator = PluginValidator()
        self._security_validator = PluginSecurityValidator()

    def discover_all_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Comprehensive plugin discovery."""
        discovered = {
            ComponentType.BACKEND: [],
            ComponentType.PROVIDER: [],
            ComponentType.SOURCE: [],
        }
        if self.enable_entry_points:
            try:
                entry_point_plugins = self._discover_entry_point_plugins()
                self._merge_plugin_discoveries(discovered, entry_point_plugins)
            except Exception as e:
                logger.warning("Entry point plugin discovery failed: %s", e)
        if self.enable_directory_scan:
            try:
                directory_plugins = self._discover_directory_plugins()
                self._merge_plugin_discoveries(discovered, directory_plugins)
            except Exception as e:
                logger.warning("Directory plugin discovery failed: %s", e)
        if self.enable_module_scan:
            try:
                module_plugins = self._discover_module_plugins()
                self._merge_plugin_discoveries(discovered, module_plugins)
            except Exception as e:
                logger.warning("Module plugin discovery failed: %s", e)
        return discovered

    def discover_plugins_for_type(self, component_type: ComponentType) -> list[PluginInfo]:
        """Discover plugins for a specific component type."""
        all_plugins = self.discover_all_plugins()
        return all_plugins.get(component_type, [])

    def _discover_entry_point_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Discover plugins via setuptools entry points."""
        logger.debug("Entry point plugin discovery not yet implemented")
        return {ComponentType.BACKEND: [], ComponentType.PROVIDER: [], ComponentType.SOURCE: []}

    def _discover_directory_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Discover plugins by scanning directories."""
        discovered = {
            ComponentType.BACKEND: [],
            ComponentType.PROVIDER: [],
            ComponentType.SOURCE: [],
        }
        for directory_path in self.plugin_directories:
            directory = Path(directory_path)
            if not directory.exists():
                continue
            for py_file in directory.rglob("*.py"):
                if py_file.name.startswith("_") or "test" in py_file.name.lower():
                    continue
                try:
                    if plugin_info := self._load_plugin_from_file(py_file):
                        discovered[plugin_info.component_type].append(plugin_info)
                except Exception as e:
                    logger.warning("Failed to load plugin from %s: %s", py_file, e)
        return discovered

    def _discover_module_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Discover plugins within Python modules."""
        logger.debug("Module plugin discovery not yet implemented")
        return {ComponentType.BACKEND: [], ComponentType.PROVIDER: [], ComponentType.SOURCE: []}

    def _load_plugin_from_file(self, file_path: Path) -> PluginInfo | None:
        """Load a plugin from a Python file."""
        return None

    def _merge_plugin_discoveries(
        self,
        target: dict[ComponentType, list[PluginInfo]],
        source: dict[ComponentType, list[PluginInfo]],
    ) -> None:
        """Merge plugin discoveries, avoiding duplicates."""
        for component_type, plugins in source.items():
            for plugin in plugins:
                existing_names = {p.name for p in target[component_type]}
                if plugin.name not in existing_names:
                    target[component_type].append(plugin)
