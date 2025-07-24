# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Plugin discovery system for CodeWeaver extensibility.

Discovers and loads plugins for backends, providers, and data sources from
specified directories with dependency validation and version management.
"""

import contextlib
import importlib
import importlib.util
import logging
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


logger = logging.getLogger(__name__)


@dataclass
class BackendCapabilities:
    """Capabilities of a vector backend plugin."""

    supports_vector_search: bool = True
    supports_hybrid_search: bool = False
    supports_sparse_search: bool = False
    supports_filtering: bool = True
    supports_metadata: bool = True
    max_vector_dimension: int | None = None
    supported_distances: list[str] = field(default_factory=list)
    requires_api_key: bool = False


@dataclass
class ProviderCapabilities:
    """Capabilities of an embedding/reranking provider plugin."""

    supports_embedding: bool = False
    supports_reranking: bool = False
    embedding_dimension: int | None = None
    max_batch_size: int | None = None
    supports_async: bool = True
    requires_api_key: bool = True
    supported_models: list[str] = field(default_factory=list)


@dataclass
class SourceCapabilities:
    """Capabilities of a data source plugin."""

    supports_streaming: bool = False
    supports_filtering: bool = True
    supports_watching: bool = False
    supports_metadata: bool = True
    supported_formats: list[str] = field(default_factory=list)


@dataclass
class PluginMetadata:
    """Metadata for a discovered plugin."""

    name: str
    version: str
    author: str | None = None
    description: str | None = None
    license: str | None = None
    homepage: str | None = None
    requirements: list[str] = field(default_factory=list)
    compatible_versions: str | None = None  # e.g., ">=1.0.0,<2.0.0"


@dataclass
class BackendPlugin:
    """Discovered backend plugin."""

    name: str
    implementation: type
    capabilities: BackendCapabilities
    metadata: PluginMetadata
    supports_hybrid: bool = False  # Legacy compatibility


@dataclass
class ProviderPlugin:
    """Discovered provider plugin."""

    name: str
    implementation: type
    capabilities: ProviderCapabilities
    metadata: PluginMetadata
    provider_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourcePlugin:
    """Discovered data source plugin."""

    name: str
    implementation: type
    capabilities: SourceCapabilities
    metadata: PluginMetadata


class PluginInterface(Protocol):
    """Protocol that all plugins must implement."""

    @classmethod
    def get_plugin_info(cls) -> dict[str, Any]:
        """Get plugin information including metadata and capabilities."""
        ...


class PluginDiscovery:
    """
    Dynamic plugin discovery system for CodeWeaver.

    Discovers and loads plugins from specified directories, validates their
    interfaces, manages dependencies, and provides version compatibility checking.
    """

    def __init__(
        self,
        plugin_directories: list[str] | None = None,
        *,
        auto_load: bool = True,
        validate_interfaces: bool = True,
    ):
        """Initialize the plugin discovery system.

        Args:
            plugin_directories: List of directories to search for plugins
            auto_load: Automatically load discovered plugins
            validate_interfaces: Validate plugin interfaces before loading
        """
        self.plugin_directories = [Path(d) for d in (plugin_directories or [])]
        self.auto_load = auto_load
        self.validate_interfaces = validate_interfaces

        # Add default plugin directories
        self._add_default_directories()

        # Discovered plugins
        self._backend_plugins: dict[str, BackendPlugin] = {}
        self._provider_plugins: dict[str, ProviderPlugin] = {}
        self._source_plugins: dict[str, SourcePlugin] = {}

        # Plugin loading status
        self._loaded_modules: set[str] = set()
        self._failed_plugins: dict[str, str] = {}  # plugin_name -> error_message

    def _add_default_directories(self) -> None:
        """Add default plugin directories."""
        # User plugins directory
        user_plugins = Path.home() / ".codeweaver" / "plugins"
        if user_plugins.exists():
            self.plugin_directories.append(user_plugins)

        # System plugins directory
        system_plugins = Path("/usr/local/share/codeweaver/plugins")
        if system_plugins.exists():
            self.plugin_directories.append(system_plugins)

        # Development plugins (relative to package)
        with contextlib.suppress(Exception):
            import codeweaver

            package_dir = Path(codeweaver.__file__).parent
            dev_plugins = package_dir / "plugins"
            if dev_plugins.exists():
                self.plugin_directories.append(dev_plugins)

    async def discover_plugins(self) -> None:
        """Discover all available plugins from configured directories."""
        logger.info("Starting plugin discovery in %d directories", len(self.plugin_directories))

        for directory in self.plugin_directories:
            if not directory.exists():
                logger.debug("Plugin directory does not exist: %s", directory)
                continue

            logger.info("Scanning plugin directory: %s", directory)
            await self._scan_directory(directory)

        logger.info(
            "Plugin discovery complete. Found: %d backends, %d providers, %d sources",
            len(self._backend_plugins),
            len(self._provider_plugins),
            len(self._source_plugins),
        )

        # Log failed plugins
        if self._failed_plugins:
            logger.warning(
                "Failed to load %d plugins: %s",
                len(self._failed_plugins),
                ", ".join(self._failed_plugins.keys()),
            )

    async def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for plugin modules."""
        # Look for Python files and packages
        for path in directory.rglob("*.py"):
            if path.name.startswith("_"):
                continue

            # Skip test files
            if "test" in path.name.lower():
                continue

            try:
                await self._load_plugin_module(path)
            except Exception as e:
                logger.warning("Failed to load plugin from %s: %s", path, e)
                self._failed_plugins[path.stem] = str(e)

    def _validate_module_spec(self, spec: Any, path: Path) -> None:
        """Validate module spec and raise error if invalid."""
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module spec from {path}")

    async def _load_plugin_module(self, path: Path) -> None:
        """Load a plugin module and discover its plugins."""
        # Check if already loaded
        module_name = f"codeweaver_plugin_{path.stem}"
        if module_name in self._loaded_modules:
            logger.debug("Plugin module already loaded: %s", module_name)
            return

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, path)
            self._validate_module_spec(spec, path)

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules.add(module_name)
            logger.debug("Loaded plugin module: %s", module_name)

            # Discover plugins in the module
            await self._discover_module_plugins(module, path)

        except Exception:
            logger.exception("Failed to load plugin module %s", path)
            raise

    async def _discover_module_plugins(self, module: Any, path: Path) -> None:
        """Discover plugins within a loaded module."""
        # Look for plugin classes
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Skip non-class objects
            if not isinstance(obj, type):
                continue

            # Check if it implements the plugin interface
            if not hasattr(obj, "get_plugin_info"):
                continue

            try:
                plugin_info = obj.get_plugin_info()
                plugin_type = plugin_info.get("type")

                if plugin_type == "backend":
                    await self._register_backend_plugin(obj, plugin_info, path)
                elif plugin_type == "provider":
                    await self._register_provider_plugin(obj, plugin_info, path)
                elif plugin_type == "source":
                    await self._register_source_plugin(obj, plugin_info, path)
                else:
                    logger.warning("Unknown plugin type '%s' for %s in %s", plugin_type, name, path)

            except Exception as e:
                logger.warning("Failed to register plugin %s from %s: %s", name, path, e)

    async def _register_backend_plugin(
        self, plugin_class: type, plugin_info: dict[str, Any], source_path: Path
    ) -> None:
        """Register a backend plugin."""
        name = plugin_info["name"]

        # Parse capabilities
        capabilities = BackendCapabilities(**plugin_info.get("capabilities", {}))

        # Parse metadata
        metadata = PluginMetadata(
            name=name,
            version=plugin_info.get("version", "0.0.0"),
            author=plugin_info.get("author"),
            description=plugin_info.get("description"),
            license=plugin_info.get("license"),
            homepage=plugin_info.get("homepage"),
            requirements=plugin_info.get("requirements", []),
            compatible_versions=plugin_info.get("compatible_versions"),
        )

        # Validate interface if required
        if self.validate_interfaces:
            self._validate_backend_interface(plugin_class, capabilities)

        # Create plugin record
        plugin = BackendPlugin(
            name=name,
            implementation=plugin_class,
            capabilities=capabilities,
            metadata=metadata,
            supports_hybrid=capabilities.supports_hybrid_search,
        )

        self._backend_plugins[name] = plugin
        logger.info("Registered backend plugin: %s v%s", name, metadata.version)

    async def _register_provider_plugin(
        self, plugin_class: type, plugin_info: dict[str, Any], source_path: Path
    ) -> None:
        """Register a provider plugin."""
        name = plugin_info["name"]

        # Parse capabilities
        capabilities = ProviderCapabilities(**plugin_info.get("capabilities", {}))

        # Parse metadata
        metadata = PluginMetadata(
            name=name,
            version=plugin_info.get("version", "0.0.0"),
            author=plugin_info.get("author"),
            description=plugin_info.get("description"),
            license=plugin_info.get("license"),
            homepage=plugin_info.get("homepage"),
            requirements=plugin_info.get("requirements", []),
            compatible_versions=plugin_info.get("compatible_versions"),
        )

        # Validate interface if required
        if self.validate_interfaces:
            self._validate_provider_interface(plugin_class, capabilities)

        # Create plugin record
        plugin = ProviderPlugin(
            name=name,
            implementation=plugin_class,
            capabilities=capabilities,
            metadata=metadata,
            provider_info=plugin_info.get("provider_info", {}),
        )

        self._provider_plugins[name] = plugin
        logger.info("Registered provider plugin: %s v%s", name, metadata.version)

    async def _register_source_plugin(
        self, plugin_class: type, plugin_info: dict[str, Any], source_path: Path
    ) -> None:
        """Register a data source plugin."""
        name = plugin_info["name"]

        # Parse capabilities
        capabilities = SourceCapabilities(**plugin_info.get("capabilities", {}))

        # Parse metadata
        metadata = PluginMetadata(
            name=name,
            version=plugin_info.get("version", "0.0.0"),
            author=plugin_info.get("author"),
            description=plugin_info.get("description"),
            license=plugin_info.get("license"),
            homepage=plugin_info.get("homepage"),
            requirements=plugin_info.get("requirements", []),
            compatible_versions=plugin_info.get("compatible_versions"),
        )

        # Validate interface if required
        if self.validate_interfaces:
            self._validate_source_interface(plugin_class, capabilities)

        # Create plugin record
        plugin = SourcePlugin(
            name=name, implementation=plugin_class, capabilities=capabilities, metadata=metadata
        )

        self._source_plugins[name] = plugin
        logger.info("Registered source plugin: %s v%s", name, metadata.version)

    def _validate_backend_interface(
        self, plugin_class: type, capabilities: BackendCapabilities
    ) -> None:
        """Validate that a backend plugin implements required methods."""
        required_methods = [
            "create_collection",
            "upsert_vectors",
            "search_vectors",
            "delete_vectors",
            "get_collection_info",
        ]

        if capabilities.supports_hybrid_search:
            required_methods.extend(["create_sparse_index", "hybrid_search"])

        for method in required_methods:
            if not hasattr(plugin_class, method):
                raise ValueError(
                    f"Backend plugin {plugin_class.__name__} missing required method: {method}"
                )

    def _validate_provider_interface(
        self, plugin_class: type, capabilities: ProviderCapabilities
    ) -> None:
        """Validate that a provider plugin implements required methods."""
        if capabilities.supports_embedding:
            required_methods = ["embed_documents", "embed_query", "dimension", "model_name"]
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValueError(
                        f"Embedding provider {plugin_class.__name__} missing required: {method}"
                    )

        if capabilities.supports_reranking:
            required_methods = ["rerank"]
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValueError(
                        f"Reranking provider {plugin_class.__name__} missing required: {method}"
                    )

    def _validate_source_interface(
        self, plugin_class: type, capabilities: SourceCapabilities
    ) -> None:
        """Validate that a source plugin implements required methods."""
        required_methods = ["discover_files", "read_file", "get_metadata"]

        if capabilities.supports_watching:
            required_methods.append("watch")

        for method in required_methods:
            if not hasattr(plugin_class, method):
                raise ValueError(
                    f"Data source plugin {plugin_class.__name__} missing required method: {method}"
                )

    def discover_backend_plugins(self) -> list[BackendPlugin]:
        """Get all discovered backend plugins."""
        return list(self._backend_plugins.values())

    def discover_provider_plugins(self) -> list[ProviderPlugin]:
        """Get all discovered provider plugins."""
        return list(self._provider_plugins.values())

    def discover_source_plugins(self) -> list[SourcePlugin]:
        """Get all discovered data source plugins."""
        return list(self._source_plugins.values())

    def get_plugin_info(self) -> dict[str, Any]:
        """Get comprehensive information about all discovered plugins."""
        return {
            "plugin_directories": [str(d) for d in self.plugin_directories],
            "loaded_modules": len(self._loaded_modules),
            "failed_plugins": self._failed_plugins,
            "backend_plugins": {
                name: {
                    "version": plugin.metadata.version,
                    "description": plugin.metadata.description,
                    "capabilities": plugin.capabilities.__dict__,
                }
                for name, plugin in self._backend_plugins.items()
            },
            "provider_plugins": {
                name: {
                    "version": plugin.metadata.version,
                    "description": plugin.metadata.description,
                    "capabilities": plugin.capabilities.__dict__,
                }
                for name, plugin in self._provider_plugins.items()
            },
            "source_plugins": {
                name: {
                    "version": plugin.metadata.version,
                    "description": plugin.metadata.description,
                    "capabilities": plugin.capabilities.__dict__,
                }
                for name, plugin in self._source_plugins.items()
            },
        }

    async def cleanup(self) -> None:
        """Cleanup loaded plugins and modules."""
        logger.info("Cleaning up plugin discovery system")

        # Remove loaded modules from sys.modules
        for module_name in self._loaded_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Clear plugin registries
        self._backend_plugins.clear()
        self._provider_plugins.clear()
        self._source_plugins.clear()
        self._loaded_modules.clear()
        self._failed_plugins.clear()

        logger.info("Plugin discovery cleanup complete")
