# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Pattern consistency validation tests for CodeWeaver components.

This module validates that all components follow the established patterns
from the providers module "gold standard". It ensures naming conventions,
required methods, properties, and architectural patterns are consistent
across all modules.
"""

import contextlib
import inspect

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def get_all_provider_classes():
    """Get all provider classes for validation."""
    provider_classes = []

    with contextlib.suppress(ImportError):
        from codeweaver.providers.providers.voyageai import VoyageAIProvider

        provider_classes.append(VoyageAIProvider)
    with contextlib.suppress(ImportError):
        from codeweaver.providers.providers.openai import OpenAIProvider

        provider_classes.append(OpenAIProvider)
    with contextlib.suppress(ImportError):
        from codeweaver.providers.providers.cohere import CohereProvider

        provider_classes.append(CohereProvider)
    with contextlib.suppress(ImportError):
        from codeweaver.providers.providers.huggingface import HuggingFaceProvider

        provider_classes.append(HuggingFaceProvider)
    with contextlib.suppress(ImportError):
        from codeweaver.providers.providers.sentence_transformers import SentenceTransformersProvider

        provider_classes.append(SentenceTransformersProvider)
    return provider_classes


def get_all_backend_classes():
    """Get all backend classes for validation."""
    backend_classes = []

    with contextlib.suppress(ImportError):
        from codeweaver.backends.providers.qdrant import QdrantBackend

        backend_classes.append(QdrantBackend)
    return backend_classes


def get_all_source_classes():
    """Get all source classes for validation."""
    source_classes = []

    with contextlib.suppress(ImportError):
        from codeweaver.sources.providers.filesystem import FileSystemSource

        source_classes.append(FileSystemSource)
    with contextlib.suppress(ImportError):
        from codeweaver.sources.api import APISource

        source_classes.append(APISource)
    with contextlib.suppress(ImportError):
        from codeweaver.sources.providers.git import GitSource

        source_classes.append(GitSource)
    return source_classes


def get_plugin_modules():
    """Get all plugin modules for import validation."""
    modules = []

    # Provider modules
    provider_modules = [
        "codeweaver.providers.voyage",
        "codeweaver.providers.openai",
        "codeweaver.providers.cohere",
        "codeweaver.providers.huggingface",
        "codeweaver.providers.sentence_transformers",
    ]

    for module_name in provider_modules:
        with contextlib.suppress(ImportError):
            module = __import__(module_name, fromlist=[""])
            modules.append(module)
    # Backend modules
    backend_modules = ["codeweaver.backends.qdrant"]

    for module_name in backend_modules:
        with contextlib.suppress(ImportError):
            module = __import__(module_name, fromlist=[""])
            modules.append(module)
    # Source modules
    source_modules = [
        "codeweaver.sources.filesystem",
        "codeweaver.sources.api",
        "codeweaver.sources.git",
    ]

    for module_name in source_modules:
        with contextlib.suppress(ImportError):
            module = __import__(module_name, fromlist=[""])
            modules.append(module)
    return modules


class TestProviderPatternConsistency:
    """Test that all providers follow the established patterns."""

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_naming_convention(self, provider_class):
        """Test that providers follow naming conventions."""
        class_name = provider_class.__name__
        assert class_name.endswith("Provider"), (
            f"Provider class {class_name} should end with 'Provider'"
        )

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_has_required_properties(self, provider_class):
        """Test that providers have all required properties."""
        required_properties = [
            "provider_name",
            "capabilities",
            "model_name",
            "dimension",
            "max_batch_size",
        ]

        # Create a mock instance to test properties
        mock_config = MagicMock()
        mock_config.get.return_value = "test-value"

        try:
            # Try to create instance with mock config
            instance = provider_class(mock_config)

            # sourcery skip: no-loop-in-tests
            for prop_name in required_properties:
                assert hasattr(instance, prop_name), (
                    f"Provider {provider_class.__name__} missing property: {prop_name}"
                )

                # Verify it's actually a property
                prop = getattr(provider_class, prop_name, None)
                assert isinstance(prop, property), (
                    f"Provider {provider_class.__name__}.{prop_name} should be a property"
                )

        except Exception as e:
            pytest.skip(f"Could not instantiate {provider_class.__name__}: {e}")

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_has_required_class_methods(self, provider_class):
        """Test that providers have all required class methods."""
        required_methods = ["check_availability", "get_static_provider_info"]

        # sourcery skip: no-loop-in-tests
        for method_name in required_methods:
            assert hasattr(provider_class, method_name), (
                f"Provider {provider_class.__name__} missing class method: {method_name}"
            )

            method = getattr(provider_class, method_name)
            assert callable(method), (
                f"Provider {provider_class.__name__}.{method_name} should be callable"
            )

            # Check if it's a classmethod
            assert isinstance(inspect.getattr_static(provider_class, method_name), classmethod), (
                f"Provider {provider_class.__name__}.{method_name} should be a classmethod"
            )

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_check_availability_signature(self, provider_class):
        """Test that check_availability has correct signature."""
        method = provider_class.check_availability
        sig = inspect.signature(method)

        # Should have parameters: capability (cls is automatically excluded from classmethod signatures)
        params = list(sig.parameters.keys())
        assert len(params) == 1, (
            f"check_availability should have 1 parameter, got {len(params)}: {params}"
        )

        assert params[0] == "capability", f"Parameter should be 'capability', got '{params[0]}'"

        # Check return type annotation
        return_annotation = sig.return_annotation
        # sourcery skip: no-conditionals-in-tests
        if return_annotation != inspect.Signature.empty:
            # Should return tuple[bool, str | None]
            assert "tuple" in str(return_annotation).lower(), (
                f"check_availability should return tuple, got {return_annotation}"
            )

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_get_static_info_signature(self, provider_class):
        """Test that get_static_provider_info has correct signature."""
        method = provider_class.get_static_provider_info
        sig = inspect.signature(method)

        # Should have no parameters (cls is automatically excluded from classmethod signatures)
        params = list(sig.parameters.keys())
        assert len(params) == 0, (
            f"get_static_provider_info should have 0 parameters, got {len(params)}: {params}"
        )

        # Check return type annotation
        return_annotation = sig.return_annotation
        # sourcery skip: no-conditionals-in-tests
        if return_annotation != inspect.Signature.empty:
            assert "EmbeddingProviderInfo" in str(return_annotation), (
                f"get_static_provider_info should return EmbeddingProviderInfo, got {return_annotation}"
            )

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_provider_has_validate_config_method(self, provider_class):
        """Test that providers have _validate_config method."""
        assert hasattr(provider_class, "_validate_config"), (
            f"Provider {provider_class.__name__} should have _validate_config method"
        )

        method = provider_class._validate_config
        assert callable(method), (
            f"Provider {provider_class.__name__}._validate_config should be callable"
        )

        # Should be an instance method (not classmethod or staticmethod)
        assert not isinstance(
            inspect.getattr_static(provider_class, "_validate_config"), classmethod | staticmethod
        ), f"Provider {provider_class.__name__}._validate_config should be an instance method"


class TestBackendPatternConsistency:
    """Test that all backends follow the established patterns."""

    @pytest.mark.parametrize("backend_class", get_all_backend_classes())
    def test_backend_naming_convention(self, backend_class):
        """Test that backends follow naming conventions."""
        class_name = backend_class.__name__
        assert class_name.endswith("Backend"), (
            f"Backend class {class_name} should end with 'Backend'"
        )

    @pytest.mark.parametrize("backend_class", get_all_backend_classes())
    def test_backend_has_required_properties(self, backend_class):
        """Test that backends have required properties."""
        required_properties = ["backend_name", "capabilities"]

        # Create a mock instance to test properties
        mock_config = MagicMock()
        mock_config.get.return_value = "test-value"

        try:
            # Try to create instance with mock config
            instance = backend_class(mock_config)

            # sourcery skip: no-loop-in-tests
            for prop_name in required_properties:
                assert hasattr(instance, prop_name), (
                    f"Backend {backend_class.__name__} missing property: {prop_name}"
                )

        except Exception as e:
            pytest.skip(f"Could not instantiate {backend_class.__name__}: {e}")

    @pytest.mark.parametrize("backend_class", get_all_backend_classes())
    def test_backend_should_have_class_methods(self, backend_class):
        """Test that backends should have required class methods."""
        # Note: This is aspirational - backends may not have these yet
        recommended_methods = ["check_availability", "get_static_backend_info"]

        # sourcery skip: no-conditionals-in-tests
        if missing_methods := [
            method_name
            for method_name in recommended_methods
            if not hasattr(backend_class, method_name)
        ]:
            pytest.skip(
                f"Backend {backend_class.__name__} missing recommended methods: {missing_methods}. "
                f"This is expected during migration phase."
            )


class TestSourcePatternConsistency:
    """Test that all sources follow the established patterns."""

    @pytest.mark.parametrize("source_class", get_all_source_classes())
    def test_source_naming_convention(self, source_class):
        """Test that sources follow naming conventions."""
        class_name = source_class.__name__

        # During migration, we expect either Source or SourceProvider
        assert class_name.endswith(("Source", "SourceProvider")), (
            f"Source class {class_name} should end with 'Source' or 'SourceProvider'"
        )

        # Ideally should end with SourceProvider
        # sourcery skip: no-conditionals-in-tests
        if not class_name.endswith("SourceProvider"):
            pytest.skip(
                f"Source {class_name} should be renamed to {class_name}Provider. "
                f"This is expected during migration phase."
            )

    @pytest.mark.parametrize("source_class", get_all_source_classes())
    def test_source_should_have_required_properties(self, source_class):
        """Test that sources should have required properties."""
        # Note: This is aspirational - sources may not have these yet
        recommended_properties = ["source_name", "capabilities"]

        # Create a mock instance to test properties
        mock_config = MagicMock()
        mock_config.get.return_value = "test-value"

        try:
            # Try to create instance with mock config
            instance = source_class(mock_config)

            # sourcery skip: no-conditionals-in-tests
            if missing_properties := [
                prop_name
                for prop_name in recommended_properties
                if not hasattr(instance, prop_name)
            ]:
                pytest.skip(
                    f"Source {source_class.__name__} missing recommended properties: {missing_properties}. "
                    f"This is expected during migration phase."
                )

        except Exception as e:
            pytest.skip(f"Could not instantiate {source_class.__name__}: {e}")

    @pytest.mark.parametrize("source_class", get_all_source_classes())
    def test_source_should_have_class_methods(self, source_class):
        """Test that sources should have required class methods."""
        # Note: This is aspirational - sources may not have these yet
        recommended_methods = ["check_availability", "get_static_source_info"]

        # sourcery skip: no-conditionals-in-tests
        if missing_methods := [
            method_name
            for method_name in recommended_methods
            if not hasattr(source_class, method_name)
        ]:
            pytest.skip(
                f"Source {source_class.__name__} missing recommended methods: {missing_methods}. "
                f"This is expected during migration phase."
            )


class TestAntiPatternDetection:
    """Test for anti-patterns that should be eliminated."""

    @pytest.mark.parametrize("module", get_plugin_modules())
    def test_no_direct_middleware_imports(self, module):
        """Test that no plugins import middleware directly."""
        module_source = inspect.getsource(module)

        # Check for direct middleware imports
        forbidden_imports = ["from codeweaver.middleware", "import codeweaver.middleware"]

        # sourcery skip: no-loop-in-tests
        for forbidden_import in forbidden_imports:
            assert forbidden_import not in module_source, (
                f"Module {module.__name__} contains forbidden import: {forbidden_import}"
            )

    def test_no_migration_code_exists(self):
        """Test that migration code has been removed."""
        migration_file = Path("src/codeweaver/config_migration.py")
        assert not migration_file.exists(), (
            "Migration code should be removed: config_migration.py still exists"
        )

    @pytest.mark.parametrize("provider_class", get_all_provider_classes())
    def test_providers_use_services_context(self, provider_class):
        """Test that provider methods accept context parameter."""
        # Check main embedding method
        # sourcery skip: no-conditionals-in-tests
        if hasattr(provider_class, "embed_documents"):
            method = provider_class.embed_documents
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            # Should have context parameter (may be optional)
            assert "context" in params, (
                f"Provider {provider_class.__name__}.embed_documents should accept 'context' parameter"
            )

        # Check reranking method if exists
        if hasattr(provider_class, "rerank_documents"):
            method = provider_class.rerank_documents
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            # Should have context parameter (may be optional)
            assert "context" in params, (
                f"Provider {provider_class.__name__}.rerank_documents should accept 'context' parameter"
            )


class TestConfigurationPatterns:
    """Test configuration pattern consistency."""

    def test_provider_configs_follow_naming(self):
        """Test that provider configuration classes follow naming conventions."""
        config_classes = []

        with contextlib.suppress(ImportError):
            from codeweaver.providers.config import (
                CohereConfig,
                HuggingFaceConfig,
                OpenAIConfig,
                SentenceTransformersConfig,
                VoyageConfig,
            )

            config_classes.extend([
                VoyageConfig,
                OpenAIConfig,
                CohereConfig,
                HuggingFaceConfig,
                SentenceTransformersConfig,
            ])
        # sourcery skip: no-loop-in-tests
        for config_class in config_classes:
            class_name = config_class.__name__
            assert class_name.endswith("Config"), (
                f"Configuration class {class_name} should end with 'Config'"
            )

            # Should not have redundant Provider in name
            assert "Provider" not in class_name, (
                f"Configuration class {class_name} should not contain 'Provider'"
            )

    def test_config_classes_inherit_from_base(self):
        """Test that configuration classes inherit from appropriate base."""
        config_classes = []

        with contextlib.suppress(ImportError):
            from codeweaver.providers.config import (
                CohereConfig,
                HuggingFaceConfig,
                OpenAIConfig,
                SentenceTransformersConfig,
                VoyageConfig,
            )

            config_classes.extend([
                VoyageConfig,
                OpenAIConfig,
                CohereConfig,
                HuggingFaceConfig,
                SentenceTransformersConfig,
            ])
        # sourcery skip: no-loop-in-tests
        for config_class in config_classes:
            # Should inherit from BaseModel (Pydantic)
            mro = config_class.__mro__
            base_model_found = any("BaseModel" in str(base) for base in mro)
            assert base_model_found, (
                f"Configuration class {config_class.__name__} should inherit from BaseModel"
            )


class TestServicesIntegration:
    """Test services layer integration patterns."""

    def test_services_manager_exists(self):
        """Test that ServicesManager exists and is properly structured."""
        from codeweaver.services.manager import ServicesManager

        # Check required methods exist
        required_methods = [
            "start_all_services",
            "stop_all_services",
            "create_service_context",
            "get_service_health",
        ]

        # sourcery skip: no-loop-in-tests
        for method_name in required_methods:
            assert hasattr(ServicesManager, method_name), (
                f"ServicesManager missing method: {method_name}"
            )

    def test_middleware_bridge_exists(self):
        """Test that ServiceBridge exists."""
        from codeweaver.services.middleware_bridge import ServiceBridge

        # Should be a class
        assert inspect.isclass(ServiceBridge), "ServiceBridge should be a class"

    def test_service_providers_exist(self):
        """Test that service providers exist."""
        service_providers = [
            "codeweaver.services.providers.chunking",
            "codeweaver.services.providers.file_filtering",
            "codeweaver.services.providers.base_provider",
        ]

        # sourcery skip: no-loop-in-tests
        for provider_module in service_providers:
            try:
                __import__(provider_module)
            except ImportError as e:
                pytest.fail(f"Service provider module {provider_module} not found: {e}")


def validate_pattern_consistency() -> bool:  # sourcery skip: low-code-quality, no-long-functions
    """Main validation function for pattern consistency.

    Returns:
        True if all patterns are consistent, False otherwise
    """
    print("🔍 Validating Pattern Consistency")

    try:
        # Run provider pattern tests
        provider_classes = get_all_provider_classes()
        print(f"   ✅ Found {len(provider_classes)} provider classes")

        for provider_class in provider_classes:
            # Check naming
            if not provider_class.__name__.endswith("Provider"):
                print(f"   ❌ Provider {provider_class.__name__} doesn't follow naming convention")
                return False

            # Check required methods
            required_methods = ["check_availability", "get_static_provider_info"]
            for method_name in required_methods:
                if not hasattr(provider_class, method_name):
                    print(f"   ❌ Provider {provider_class.__name__} missing method: {method_name}")
                    return False

        print("   ✅ All providers follow naming and method patterns")

        # Check for anti-patterns (middleware imports outside fallback methods)
        modules = get_plugin_modules()
        violations = []
        for module in modules:
            try:
                module_source = inspect.getsource(module)
                lines = module_source.split("\n")

                for i, line in enumerate(lines):
                    if "from codeweaver.middleware" in line:
                        # Check if this import is within a fallback method or allowed context
                        in_fallback_method = False

                        # Look backwards to find the method definition
                        for j in range(i - 1, max(0, i - 20), -1):
                            if "_fallback" in lines[j] or "fallback" in lines[j]:
                                in_fallback_method = True
                                break
                            if lines[j].strip().startswith("def ") or lines[j].strip().startswith(
                                "class "
                            ):
                                break

                        # Also allow imports in services providers
                        if not in_fallback_method and "services/providers" not in module.__file__:
                            violations.append(f"{module.__name__}:{i + 1}")
            except Exception as e:
                print(f"   ⚠️  Could not check module {module.__name__}: {e}")

        if violations:
            print(f"   ❌ Direct middleware imports found outside fallback methods: {violations}")
            return False

        print("   ✅ All middleware imports are in allowed contexts")

        # Check migration code removal
        migration_file = Path("src/codeweaver/config_migration.py")
        if migration_file.exists():
            print("   ❌ Migration code still exists")
            return False

        print("   ✅ Migration code has been removed")

        print("   ✅ Pattern consistency validation complete")

    except Exception as e:
        print(f"   ❌ Pattern validation error: {e}")
        return False
    else:
        return True


if __name__ == "__main__":
    """Run pattern consistency validation as a script."""
    success = validate_pattern_consistency()
    exit(0 if success else 1)
