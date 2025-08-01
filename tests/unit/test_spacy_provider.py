# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for spaCy NLP provider."""

from unittest.mock import MagicMock, patch

import pytest

from codeweaver.providers.config import SpaCyProviderConfig
from codeweaver.providers.nlp.spacy import SpaCyProvider
from codeweaver.cw_types import ProviderType


class TestSpaCyProvider:
    """Test suite for spaCy NLP provider following development patterns."""

    @pytest.fixture
    def config(self) -> SpaCyProviderConfig:
        """Standard configuration for testing."""
        return SpaCyProviderConfig(
            model="en_core_web_sm",
            enable_intent_classification=True,
            intent_labels=["SEARCH", "DOCUMENTATION", "ANALYSIS"],
            confidence_threshold=0.7,
        )

    @pytest.fixture
    def provider(self, config: SpaCyProviderConfig) -> SpaCyProvider:
        """Provider instance for testing."""
        return SpaCyProvider(config)

    # Pattern compliance tests
    def test_provider_follows_naming_convention(self, provider: SpaCyProvider):
        """Test that provider follows naming conventions."""
        assert provider.__class__.__name__.endswith("Provider")
        assert hasattr(provider, "provider_name")
        assert provider.provider_name == "spaCy"

    def test_provider_has_required_properties(self, provider: SpaCyProvider):
        """Test that provider has all required properties."""
        required_properties = ["provider_name", "model_name", "max_batch_size"]
# sourcery skip: no-loop-in-tests
        for prop_name in required_properties:
            assert hasattr(provider, prop_name), f"Missing property: {prop_name}"

    def test_provider_has_required_methods(self):
        """Test that provider has all required methods."""
        required_methods = ["initialize", "shutdown", "process_text", "health_check"]
# sourcery skip: no-loop-in-tests
        for method_name in required_methods:
            assert hasattr(SpaCyProvider, method_name), f"Missing method: {method_name}"
            assert callable(getattr(SpaCyProvider, method_name))

    # Configuration tests
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = SpaCyProviderConfig(
            model="en_core_web_sm", enable_intent_classification=True, confidence_threshold=0.8
        )
        # Should not raise
        provider = SpaCyProvider(config)
        assert provider.config.model == "en_core_web_sm"

    def test_config_validation_failure(self):
        """Test configuration validation failures."""
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            SpaCyProviderConfig(
                model="en_core_web_sm",
                confidence_threshold=-0.1,  # Invalid threshold
            )

        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
            SpaCyProviderConfig(
                model="en_core_web_sm",
                confidence_threshold=1.5,  # Invalid threshold
            )

    # Provider registration tests
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        from codeweaver.cw_types.providers.registry import PROVIDER_REGISTRY

        # Check that spaCy provider is in registry
        assert ProviderType.SPACY in PROVIDER_REGISTRY
        registry_entry = PROVIDER_REGISTRY[ProviderType.SPACY]

        # Verify provider class is set
        assert registry_entry.provider_class is not None
        assert registry_entry.provider_class == SpaCyProvider

    # Mock tests for functionality (since spaCy models may not be available)
    @patch("spacy.load")
    async def test_initialize_success(self, mock_spacy_load, provider: SpaCyProvider):
        """Test successful provider initialization."""
        # Mock spaCy language object
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["tok2vec", "tagger", "parser", "ner"]
        mock_nlp.meta = {"name": "en_core_web_sm"}
        mock_nlp.add_pipe = MagicMock()
        mock_spacy_load.return_value = mock_nlp

        await provider.initialize()

        # Verify spacy.load was called with correct model
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
        assert provider.nlp is not None

    @patch("spacy.load")
    async def test_initialize_model_not_found(self, mock_spacy_load, provider: SpaCyProvider):
        """Test initialization with missing spaCy model."""
        mock_spacy_load.side_effect = OSError("Model 'en_core_web_sm' not found")

        with pytest.raises(RuntimeError, match="spaCy model initialization failed"):
            await provider.initialize()

    @patch("spacy.load")
    async def test_process_text_mock(self, mock_spacy_load, provider: SpaCyProvider):
        """Test text processing with mocked spaCy."""
        # Mock spaCy components
        mock_doc = MagicMock()
        mock_doc.cats = {"SEARCH": 0.8, "DOCUMENTATION": 0.2}
        mock_doc.ents = []
        mock_doc.tensor = None
        mock_doc.vector = [0.1, 0.2, 0.3]

        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_nlp.pipe_names = ["textcat"]
        mock_nlp.meta = {"name": "en_core_web_sm"}
        mock_spacy_load.return_value = mock_nlp

        provider.nlp = mock_nlp

        result = await provider.process_text("search for python functions")

        assert result is not None
        assert result.confidence == 0.8  # Max confidence from cats
        assert result.embeddings == [0.1, 0.2, 0.3]

    async def test_health_check(self, provider: SpaCyProvider):
        """Test provider health check."""
        health = await provider.health_check()

        assert isinstance(health, dict)
        assert "provider_name" in health
        assert health["provider_name"] == "spaCy"
        assert "model_loaded" in health
        assert "capabilities" in health

    async def test_shutdown(self, provider: SpaCyProvider):
        """Test provider shutdown."""
        # Should not raise any exceptions
        await provider.shutdown()
        assert provider.nlp is None
