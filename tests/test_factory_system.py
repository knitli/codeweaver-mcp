# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Comprehensive test suite for the CodeWeaver factory system.

Tests all aspects of the factory system including dependency injection,
plugin discovery, validation, and integration scenarios.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.dependency_injection import DependencyContainer, Lifecycle, ServiceLocator
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.factories.integration import (
    LegacyCompatibilityAdapter,
    ServerMigrationHelper,
    create_migration_config,
    validate_migration_readiness,
)
from codeweaver.factories.plugin_discovery import (
    BackendCapabilities,
    BackendPlugin,
    PluginDiscovery,
    PluginMetadata,
    ProviderCapabilities,
    ProviderPlugin,
)
from codeweaver.factories.unified_factory import UnifiedFactory
from codeweaver.factories.validation import CompatibilityLevel, FactoryValidator, ValidationLevel


class TestDependencyContainer:
    """Test dependency injection container functionality."""

    def test_basic_registration_and_resolution(self) -> None:
        """Test basic component registration and resolution."""
        container = DependencyContainer()

        # Register a simple factory
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return "test_instance"

        container.register("test", "component", factory)

        # Resolve the component
        instance = container.resolve("test", "component")

        assert instance == "test_instance"
        assert factory_called

    def test_singleton_lifecycle(self) -> None:
        """Test singleton lifecycle management."""
        container = DependencyContainer()

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        container.register("test", "singleton", factory, lifecycle=Lifecycle.SINGLETON)

        # Multiple resolutions should return same instance
        instance1 = container.resolve("test", "singleton")
        instance2 = container.resolve("test", "singleton")

        assert instance1 == instance2
        assert call_count == 1

    def test_transient_lifecycle(self) -> None:
        """Test transient lifecycle management."""
        container = DependencyContainer()

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        container.register("test", "transient", factory, lifecycle=Lifecycle.TRANSIENT)

        # Multiple resolutions should create new instances
        instance1 = container.resolve("test", "transient")
        instance2 = container.resolve("test", "transient")

        assert instance1 != instance2
        assert call_count == 2

    def test_scoped_lifecycle(self) -> None:
        """Test scoped lifecycle management."""
        container = DependencyContainer()

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        container.register("test", "scoped", factory, lifecycle=Lifecycle.SCOPED)

        # Create scopes
        container.create_scope("scope1")
        container.create_scope("scope2")

        # Same instance within scope
        instance1_s1 = container.resolve("test", "scoped", scope_id="scope1")
        instance2_s1 = container.resolve("test", "scoped", scope_id="scope1")
        assert instance1_s1 == instance2_s1

        # Different instance in different scope
        instance1_s2 = container.resolve("test", "scoped", scope_id="scope2")
        assert instance1_s1 != instance1_s2

        # Cleanup
        container.dispose_scope("scope1")
        container.dispose_scope("scope2")

    def test_dependency_resolution(self) -> None:
        """Test automatic dependency resolution."""
        container = DependencyContainer()

        # Register dependencies
        container.register("db", "postgres", lambda: "postgres_instance")
        container.register("cache", "redis", lambda: "redis_instance")

        # Register component with dependencies
        def service_factory(database, cache):
            return {"db": database, "cache": cache}

        container.register(
            "service",
            "api",
            service_factory,
            dependencies={"database": "db:postgres", "cache": "cache:redis"},
        )

        # Resolve should inject dependencies
        service = container.resolve("service", "api")

        assert service["db"] == "postgres_instance"
        assert service["cache"] == "redis_instance"

    def test_circular_dependency_detection(self) -> None:
        """Test circular dependency detection."""
        container = DependencyContainer()

        # Create circular dependency
        container.register("service", "a", lambda b: f"a_with_{b}", dependencies={"b": "service:b"})
        container.register("service", "b", lambda a: f"b_with_{a}", dependencies={"a": "service:a"})

        # Should raise ValueError for circular dependency
        with pytest.raises(ValueError, match="Circular dependency detected"):
            container.resolve("service", "a")

    def test_service_locator(self) -> None:
        """Test service locator pattern."""
        container = DependencyContainer()
        container.register("test", "component", lambda: "test_value")

        # Set global container
        ServiceLocator.set_container(container)

        # Resolve via service locator
        result = ServiceLocator.resolve("test", "component")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_cleanup_handlers(self) -> None:
        """Test cleanup handler registration and execution."""
        container = DependencyContainer()

        # Mock object with cleanup method
        class MockComponent:
            def __init__(self):
                self.cleaned_up = False

            async def close(self):
                self.cleaned_up = True

        component = MockComponent()
        container.register("test", "component", lambda: component, lifecycle=Lifecycle.SINGLETON)

        # Resolve to trigger registration
        resolved = container.resolve("test", "component")
        assert resolved == component

        # Cleanup should call the close method
        await container.cleanup()
        assert component.cleaned_up


class TestPluginDiscovery:
    """Test plugin discovery system."""

    @pytest.mark.asyncio
    async def test_plugin_discovery_initialization(self) -> None:
        """Test plugin discovery initialization."""
        discovery = PluginDiscovery(plugin_directories=["/tmp/test_plugins"], auto_load=False)

        assert len(discovery.plugin_directories) >= 1
        assert not discovery.auto_load

    def test_backend_plugin_structure(self) -> None:
        """Test backend plugin data structure."""
        capabilities = BackendCapabilities(
            supports_vector_search=True,
            supports_hybrid_search=True,
            max_vector_dimension=4096,
            supported_distances=["cosine", "euclidean"],
        )

        metadata = PluginMetadata(name="test-backend", version="1.0.0", author="Test Author")

        plugin = BackendPlugin(
            name="test-backend", implementation=Mock, capabilities=capabilities, metadata=metadata
        )

        assert plugin.name == "test-backend"
        assert plugin.capabilities.supports_hybrid_search
        assert plugin.metadata.version == "1.0.0"

    def test_provider_plugin_structure(self) -> None:
        """Test provider plugin data structure."""
        capabilities = ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=False,
            embedding_dimension=768,
            max_batch_size=128,
        )

        metadata = PluginMetadata(
            name="test-provider", version="2.0.0", requirements=["transformers>=4.0.0"]
        )

        plugin = ProviderPlugin(
            name="test-provider", implementation=Mock, capabilities=capabilities, metadata=metadata
        )

        assert plugin.name == "test-provider"
        assert plugin.capabilities.supports_embedding
        assert not plugin.capabilities.supports_reranking
        assert "transformers>=4.0.0" in plugin.metadata.requirements

    @pytest.mark.asyncio
    async def test_plugin_interface_validation(self) -> None:
        """Test plugin interface validation."""
        discovery = PluginDiscovery(validate_interfaces=True)

        # Mock plugin class with proper interface
        class ValidBackendPlugin:
            @classmethod
            def get_plugin_info(cls):
                return {
                    "type": "backend",
                    "name": "valid-backend",
                    "version": "1.0.0",
                    "capabilities": {"supports_vector_search": True},
                }

            async def create_collection(self, name, dimension):
                pass

            async def upsert_vectors(self, vectors):
                pass

            async def search_vectors(self, query, filters, limit):
                pass

            async def delete_vectors(self, ids):
                pass

            async def get_collection_info(self, name):
                pass

        # Should not raise
        with patch.object(discovery, "_validate_backend_interface") as mock_validate:
            await discovery._register_backend_plugin(
                ValidBackendPlugin, ValidBackendPlugin.get_plugin_info(), Path("/test/plugin.py")
            )
            mock_validate.assert_called_once()


class TestExtensibilityManager:
    """Test extensibility manager functionality."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self) -> None:
        """Test extensibility manager initialization."""
        config = Mock(spec=CodeWeaverConfig)
        config.rate_limiting = Mock()
        config.backend = Mock(provider="qdrant")
        config.embedding = Mock(provider="voyage", api_key="test")

        manager = ExtensibilityManager(config)

        assert manager.config == config
        assert not manager._initialized

        # Initialize
        with patch.object(manager, "_unified_factory", Mock()):
            await manager.initialize()
            assert manager._initialized

    @pytest.mark.asyncio
    async def test_lazy_initialization(self) -> None:
        """Test lazy initialization of components."""
        config = Mock(spec=CodeWeaverConfig)
        config.rate_limiting = Mock()

        extensibility_config = ExtensibilityConfig(
            lazy_initialization=True,
            enable_plugin_discovery=False,
            enable_dependency_injection=False,
        )

        manager = ExtensibilityManager(config, extensibility_config)

        # Initialize should not create components
        with patch.object(manager, "_initialize_core_components") as mock_init:
            await manager.initialize()
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_component_lifecycle(self) -> None:
        """Test component creation and lifecycle."""
        config = Mock(spec=CodeWeaverConfig)
        config.rate_limiting = Mock()
        config.backend = Mock(provider="qdrant")
        config.embedding = Mock(provider="voyage", api_key="test")

        manager = ExtensibilityManager(config)

        # Mock factory
        mock_backend = Mock()
        mock_factory = Mock()
        mock_factory.backends.create_backend.return_value = mock_backend

        with patch.object(manager, "_unified_factory", mock_factory):
            backend = await manager._create_backend()
            assert backend == mock_backend
            mock_factory.backends.create_backend.assert_called_once()

    def test_component_info(self) -> None:
        """Test getting component information."""
        config = Mock(spec=CodeWeaverConfig)
        manager = ExtensibilityManager(config)

        # Without initialization
        info = manager.get_component_info()
        assert "error" in info

        # With mock factory
        mock_factory = Mock()
        mock_factory.get_component_info.return_value = {
            "backends": {"qdrant": {}},
            "providers": {"voyage": {}},
        }
        manager._unified_factory = mock_factory
        manager._initialized = True

        info = manager.get_component_info()
        assert "backends" in info
        assert "extensibility_manager" in info


class TestFactoryValidator:
    """Test factory validation system."""

    def test_validation_levels(self) -> None:
        """Test different validation levels."""
        # Minimal validation
        validator_min = FactoryValidator(level=ValidationLevel.MINIMAL)
        assert len(validator_min._validation_checks) < 5

        # Standard validation
        validator_std = FactoryValidator(level=ValidationLevel.STANDARD)
        assert len(validator_std._validation_checks) >= 5

        # Comprehensive validation
        validator_full = FactoryValidator(level=ValidationLevel.COMPREHENSIVE)
        assert len(validator_full._validation_checks) > len(validator_std._validation_checks)

    @pytest.mark.asyncio
    async def test_configuration_validation(self) -> None:
        """Test configuration validation."""
        validator = FactoryValidator()

        # Invalid config (missing backend)
        config = Mock(spec=CodeWeaverConfig)
        config.backend = None

        results = await validator.validate_configuration(config)

        # Should have validation errors
        assert any(not r.passed for r in results)
        assert any(r.component == "backend" for r in results)

    def test_compatibility_checking(self) -> None:
        """Test component compatibility checking."""
        validator = FactoryValidator()

        # Mock config with optimal pairing
        config = Mock(spec=CodeWeaverConfig)
        config.backend = Mock(provider="qdrant")
        config.embedding = Mock(provider="voyage", api_key="test")

        result = validator._check_backend_provider_compatibility(config)

        assert result is not None
        assert result.level == CompatibilityLevel.OPTIMAL

    @pytest.mark.asyncio
    async def test_health_report_generation(self) -> None:
        """Test system health report generation."""
        validator = FactoryValidator()

        # Mock config
        config = Mock(spec=CodeWeaverConfig)
        config.backend = Mock(provider="qdrant")
        config.embedding = Mock(provider="voyage", api_key="test", dimension=1536)
        config.rate_limiting = Mock(requests_per_minute=100)

        report = await validator.generate_health_report(config)

        assert report.overall_health in ["healthy", "degraded", "critical"]
        assert isinstance(report.validation_results, list)
        assert isinstance(report.recommendations, list)


class TestIntegrationUtilities:
    """Test integration and migration utilities."""

    def test_migration_config_creation(self) -> None:
        """Test migration configuration creation."""
        config = create_migration_config(
            enable_plugins=True, enable_legacy_fallback=True, lazy_init=False
        )

        assert config.enable_plugin_discovery
        assert config.enable_legacy_fallbacks
        assert config.migration_mode
        assert not config.lazy_initialization

    def test_migration_readiness_validation(self) -> None:
        """Test migration readiness validation."""
        # Valid config
        config = Mock(spec=CodeWeaverConfig)
        config.backend = Mock()
        config.embedding = Mock()
        config.qdrant = None

        results = validate_migration_readiness(config)
        assert results["ready"]

        # Invalid config (missing backend)
        config.backend = None
        results = validate_migration_readiness(config)
        assert not results["ready"]
        assert "Missing backend configuration" in results["issues"]

    @pytest.mark.asyncio
    async def test_legacy_compatibility_adapter(self) -> None:
        """Test legacy compatibility adapter."""
        # Mock extensibility manager
        manager = Mock(spec=ExtensibilityManager)
        mock_backend = Mock()
        mock_embedder = Mock()

        manager.get_backend = AsyncMock(return_value=mock_backend)
        manager.get_embedding_provider = AsyncMock(return_value=mock_embedder)
        manager.get_rate_limiter = Mock(return_value="rate_limiter")

        # Create adapter
        adapter = LegacyCompatibilityAdapter(manager)

        # Test getting components
        embedder = await adapter.get_embedder()
        assert embedder == mock_embedder

        rate_limiter = adapter.get_rate_limiter()
        assert rate_limiter == "rate_limiter"

    @pytest.mark.asyncio
    async def test_server_migration_helper(self) -> None:
        """Test server migration helper."""
        # Mock server
        server = Mock()
        server.config = Mock(spec=CodeWeaverConfig)
        server.config.rate_limiting = Mock()
        server.qdrant = "old_qdrant"
        server.embedder = "old_embedder"

        # Create helper
        helper = ServerMigrationHelper(server)

        # Mock manager creation
        with patch("codeweaver.factories.integration.ExtensibilityManager") as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            mock_manager.initialize = AsyncMock()
            mock_adapter = Mock()
            mock_adapter.get_qdrant_client = AsyncMock(return_value="new_qdrant")
            mock_adapter.get_embedder = AsyncMock(return_value="new_embedder")

            with patch(
                "codeweaver.factories.integration.LegacyCompatibilityAdapter",
                return_value=mock_adapter,
            ):
                await helper.migrate_to_factories()

        # Server should have new components
        assert server.qdrant == "new_qdrant"
        assert server.embedder == "new_embedder"


class TestUnifiedFactory:
    """Test unified factory functionality."""

    def test_factory_initialization(self) -> None:
        """Test unified factory initialization."""
        container = Mock(spec=DependencyContainer)
        discovery = Mock(spec=PluginDiscovery)

        factory = UnifiedFactory(container, discovery)

        assert factory._container == container
        assert factory._plugin_discovery == discovery
        assert hasattr(factory, "backends")
        assert hasattr(factory, "providers")
        assert hasattr(factory, "sources")

    def test_component_info_aggregation(self) -> None:
        """Test getting aggregated component information."""
        factory = UnifiedFactory()

        # Mock component lists
        factory.backends.list_supported_providers = Mock(
            return_value={"qdrant": {}, "pinecone": {}}
        )
        factory.providers.list_available_embedding_providers = Mock(
            return_value={"voyage": {}, "openai": {}}
        )
        factory.sources.list_available_sources = Mock(return_value={"filesystem": {}, "s3": {}})

        info = factory.get_component_info()

        assert "backends" in info
        assert "embedding_providers" in info
        assert "data_sources" in info
        assert len(info["backends"]) == 2
        assert len(info["embedding_providers"]) == 2

    def test_configuration_validation(self) -> None:
        """Test configuration validation across components."""
        factory = UnifiedFactory()

        # Valid configuration
        config = {
            "backend": {"provider": "qdrant", "url": "http://localhost:6333"},
            "embedding": {"provider": "voyage", "api_key": "test"},
        }

        # Mock available providers
        factory.providers.list_available_embedding_providers = Mock(return_value={"voyage": {}})

        results = factory.validate_configuration(config)

        assert results["valid"]
        assert len(results["issues"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
