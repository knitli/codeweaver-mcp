# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Unit tests for DocArray backend integration."""

import pytest

from codeweaver.backends.factory import BackendFactory


# Check if DocArray is actually available
try:
    import docarray  # noqa: F401

    DOCARRAY_AVAILABLE = True
except ImportError:
    DOCARRAY_AVAILABLE = False


@pytest.mark.embeddings
@pytest.mark.external_api
@pytest.mark.qdrant
@pytest.mark.unit
class TestDocArrayIntegration:
    """Test DocArray backend integration with CodeWeaver."""

    def test_docarray_backend_availability(self):
        """Test DocArray backend availability matches dependency status."""
        providers = BackendFactory.list_supported_providers()
        docarray_providers = [p for p in providers.keys() if "docarray" in p]

        # sourcery skip: no-conditionals-in-tests
        if DOCARRAY_AVAILABLE:
            # Should have DocArray backends when dependencies are available
            assert docarray_providers
            assert "docarray_qdrant" in providers
        else:
            # Should be empty without docarray installed
            assert not docarray_providers

    @pytest.mark.skipif(DOCARRAY_AVAILABLE, reason="DocArray is installed")
    def test_docarray_imports_fail_gracefully_when_not_installed(self):
        """Test that DocArray imports fail gracefully without dependencies."""
        with pytest.raises(ImportError):
            from codeweaver.backends.docarray.config import DocArrayConfigFactory  # noqa: F401

        with pytest.raises(ImportError):
            from codeweaver.backends.docarray.schema import SchemaTemplates  # noqa: F401

        with pytest.raises(ImportError):
            from codeweaver.backends.docarray.qdrant import QdrantDocArrayBackend  # noqa: F401

    @pytest.mark.skipif(not DOCARRAY_AVAILABLE, reason="DocArray not installed")
    def test_docarray_imports_work_when_installed(self):
        """Test that DocArray imports work when dependencies are available."""
        # These should not raise ImportError when DocArray is installed
        from codeweaver.backends.docarray.config import DocArrayConfigFactory  # noqa: F401
        from codeweaver.backends.docarray.qdrant import QdrantDocArrayBackend  # noqa: F401
        from codeweaver.backends.docarray.schema import SchemaTemplates  # noqa: F401

    def test_main_factory_always_works(self):
        """Test that main factory works regardless of DocArray availability."""
        providers = BackendFactory.list_supported_providers()

        # Should always have the standard backends
        assert "qdrant" in providers
        assert providers["qdrant"]["available"]

        # sourcery skip: no-conditionals-in-tests
        if DOCARRAY_AVAILABLE:
            # Should have DocArray backends when available
            assert "docarray_qdrant" in providers
        else:
            # Should not have DocArray backends when not available
            assert "docarray_qdrant" not in providers

    def test_backend_creation_still_works(self):
        """Test that backend creation still works for non-DocArray backends."""
        from codeweaver.backends.factory import BackendConfig
        from codeweaver.types import ProviderKind

        # This should work for standard Qdrant
        config = BackendConfig(
            provider="qdrant", kind=ProviderKind.COMBINED, url="http://localhost:6333"
        )

        # Should be able to create the config (actual backend creation would require Qdrant)
        assert config.provider == "qdrant"
        assert config.url == "http://localhost:6333"


@pytest.mark.skipif(
    condition=not DOCARRAY_AVAILABLE,
    reason="DocArray not installed - this tests the implementation with dependencies",
)
@pytest.mark.embeddings
@pytest.mark.external_api
@pytest.mark.qdrant
@pytest.mark.unit
class TestDocArrayWithDependencies:
    """Test DocArray backend when dependencies are available."""

    def test_docarray_config_factory(self):
        """Test DocArray configuration factory."""
        from codeweaver.backends.docarray.config import DocArrayConfigFactory
        from codeweaver.types import ProviderKind

        backends = DocArrayConfigFactory.get_supported_backends()
        assert "docarray_qdrant" in backends

        config = DocArrayConfigFactory.create_config(
            "docarray_qdrant",
            kind=ProviderKind.COMBINED,
            url="http://localhost:6333",
            api_key="test-key",
        )
        assert config.provider == "docarray_qdrant"

    def test_schema_templates(self):
        """Test document schema templates."""
        from codeweaver.backends.docarray.schema import SchemaTemplates

        # Test code search schema
        code_schema = SchemaTemplates.code_search_schema(512)
        assert hasattr(code_schema, "model_fields")

        # Test semantic search schema
        semantic_schema = SchemaTemplates.semantic_search_schema(512)
        assert hasattr(semantic_schema, "model_fields")

    def test_qdrant_backend_dependencies(self):
        """Test Qdrant backend dependency checking."""
        from codeweaver.backends.docarray.qdrant import QdrantDocArrayBackend

        missing_deps = QdrantDocArrayBackend._check_dependencies()
        assert isinstance(missing_deps, list)
        # Would be empty if all dependencies are installed
