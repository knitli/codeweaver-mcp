# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
spaCy provider for natural language processing.

Implements NLP capabilities using spaCy's industrial-strength NLP library,
including intent classification, entity recognition, and text embeddings.
"""

import itertools
import logging

from typing import Any

from codeweaver.cw_types import (
    EmbeddingProviderInfo,
    IntentType,
    NLPCapability,
    NLPModelSize,
    ProviderCapabilities,
)
from codeweaver.providers.base import LocalNLPProvider
from codeweaver.providers.config import SpaCyProviderConfig


logger = logging.getLogger(__name__)


class SpaCyProvider(LocalNLPProvider):
    """spaCy-based NLP provider with 3.7+ features."""

    def __init__(self, config: SpaCyProviderConfig | dict[str, Any]):
        """Initialize spaCy provider.

        Args:
            config: SpaCy provider configuration
        """
        if isinstance(config, dict):
            config = SpaCyProviderConfig(**config)
        super().__init__(config)
        self.nlp = None
        self._default_patterns = self._get_default_domain_patterns()

    def _validate_config(self) -> None:
        """Validate spaCy provider configuration."""
        if not isinstance(self.config, SpaCyProviderConfig):
            raise TypeError("Config must be SpaCyProviderConfig instance")
        if not self.config.model:
            raise ValueError("Model name is required")
        if self.config.confidence_threshold < 0.0 or self.config.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.config.batch_size < 1:
            raise ValueError("Batch size must be positive")

    async def initialize(self) -> None:
        """Initialize spaCy pipeline with 3.7+ features."""
        try:
            import spacy

            if self.config.use_transformers:
                try:
                    self.nlp = spacy.load("en_core_web_trf")
                    logger.info("Loaded transformer-based spaCy model")
                except (ImportError, OSError) as e:
                    logger.warning(
                        "Transformers not available, falling back to standard model: %s", e
                    )
                    self.nlp = spacy.load(self.config.model)
            else:
                self.nlp = spacy.load(self.config.model)
            await self._setup_domain_entity_ruler()
            await self._setup_intent_classifier()
            logger.info("spaCy provider initialized with model: %s", self.nlp.meta["name"])
        except OSError as e:
            error_msg = ("Failed to load spaCy model %s", self.config.model)
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e
        except ImportError as e:
            error_msg = "spaCy not available"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    async def shutdown(self) -> None:
        """Shutdown spaCy provider."""
        if self.nlp:
            self.nlp = None
            logger.info("spaCy provider shutdown complete")

    async def _setup_domain_entity_ruler(self) -> None:
        """Set up domain-specific entity ruler with code patterns."""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self._default_patterns)
            logger.info("Added %d domain-specific entity patterns", len(self._default_patterns))

    async def _setup_intent_classifier(self) -> None:
        """Set up text categorizer for intent classification."""
        if self.config.enable_intent_classification and "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.add_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
            )
            for intent in self.config.intent_labels:
                textcat.add_label(intent)
            logger.info("Added text categorizer for intent classification")

    async def process_text(
        self, text: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process text using spaCy pipeline."""
        if not self.nlp:
            raise RuntimeError("spaCy provider not initialized")
        doc = self.nlp(text)
        intent_type = None
        confidence = 0.0
        if doc.cats:
            parsed_intent = max(doc.cats, key=doc.cats.get)
            confidence = doc.cats[parsed_intent]
            try:
                intent_type = IntentType(parsed_intent.lower())
            except ValueError:
                intent_type = None
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent._, "confidence", 0.95),
            }
            for ent in doc.ents
        ]
        primary_target = self._extract_primary_target(entities)
        embeddings = None
        if hasattr(doc, "tensor") and doc.tensor is not None:
            embeddings = doc.tensor.tolist()
        elif hasattr(doc, "vector") and doc.vector is not None:
            embeddings = doc.vector.tolist()
        return {
            "intent_type": intent_type,
            "confidence": confidence,
            "entities": entities,
            "primary_target": primary_target,
            "embeddings": embeddings,
            "metadata": {
                "model": self.nlp.meta["name"],
                "pipeline": list(self.nlp.pipe_names),
                "has_transformer": "transformer" in self.nlp.pipe_names,
                "tokens": len(doc),
                "pos_tags": [(token.text, token.pos_) for token in doc[:10]],
            },
        }

    async def classify_intent(self, text: str) -> tuple[IntentType | None, float]:
        """Classify intent with confidence score."""
        if not self.nlp:
            raise RuntimeError("spaCy provider not initialized")
        doc = self.nlp(text)
        if doc.cats:
            parsed_intent = max(doc.cats, key=doc.cats.get)
            confidence = doc.cats[parsed_intent]
            try:
                intent_type = IntentType(parsed_intent.lower())
            except ValueError:
                return (None, confidence)
            else:
                return (intent_type, confidence)
        return (None, 0.0)

    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""
        if not self.nlp:
            raise RuntimeError("spaCy provider not initialized")
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent._, "confidence", 0.95),
            }
            for ent in doc.ents
        ]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using spaCy pipeline."""
        if not self.nlp:
            raise RuntimeError("spaCy provider not initialized")
        embeddings = []
        for doc in self.nlp.pipe(texts, batch_size=self.config.batch_size):
            if hasattr(doc, "tensor") and doc.tensor is not None:
                embeddings.append(doc.tensor.tolist())
            elif hasattr(doc, "vector") and doc.vector is not None:
                embeddings.append(doc.vector.tolist())
            else:
                embeddings.append([0.0] * 300)
        return embeddings

    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "spacy"

    @property
    def model_name(self) -> str:
        """Get the current NLP model name."""
        return self.nlp.meta["name"] if self.nlp else self.config.model

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        if self.nlp and "lang" in self.nlp.meta:
            return [self.nlp.meta["lang"]]
        return ["en"]

    @property
    def max_text_length(self) -> int | None:
        """Get the maximum text length in characters."""
        return self.config.max_length

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get available spaCy models."""
        return [
            {
                "name": "en_core_web_sm",
                "language": "en",
                "capabilities": [
                    NLPCapability.TOK2VEC.value,
                    NLPCapability.TAGGER.value,
                    NLPCapability.DEPENDENCY_PARSER.value,
                    NLPCapability.SENTENCE_RECOGNIZER.value,
                    NLPCapability.ATTRIBUTE_RULER.value,
                    NLPCapability.LEMMATIZER.value,
                    NLPCapability.NAMED_ENTITY_RECOGNIZER.value,
                ],
                "model_size": NLPModelSize.SMALL.value,
                "requires_download": True,
                "description": "Small English model with basic capabilities",
            },
            {
                "name": "en_core_web_md",
                "language": "en",
                "capabilities": [
                    NLPCapability.TOK2VEC.value,
                    NLPCapability.TAGGER.value,
                    NLPCapability.DEPENDENCY_PARSER.value,
                    NLPCapability.SENTENCE_RECOGNIZER.value,
                    NLPCapability.ATTRIBUTE_RULER.value,
                    NLPCapability.LEMMATIZER.value,
                    NLPCapability.NAMED_ENTITY_RECOGNIZER.value,
                ],
                "model_size": NLPModelSize.MEDIUM.value,
                "requires_download": True,
                "description": "Medium English model with word vectors",
            },
            {
                "name": "en_core_web_lg",
                "language": "en",
                "capabilities": [
                    NLPCapability.TOK2VEC.value,
                    NLPCapability.TAGGER.value,
                    NLPCapability.DEPENDENCY_PARSER.value,
                    NLPCapability.SENTENCE_RECOGNIZER.value,
                    NLPCapability.ATTRIBUTE_RULER.value,
                    NLPCapability.LEMMATIZER.value,
                    NLPCapability.NAMED_ENTITY_RECOGNIZER.value,
                ],
                "model_size": NLPModelSize.LARGE.value,
                "requires_download": True,
                "description": "Large English model with large word vectors",
            },
            {
                "name": "en_core_web_trf",
                "language": "en",
                "capabilities": [
                    NLPCapability.TRANSFORMER.value,
                    NLPCapability.TAGGER.value,
                    NLPCapability.DEPENDENCY_PARSER.value,
                    NLPCapability.ATTRIBUTE_RULER.value,
                    NLPCapability.LEMMATIZER.value,
                    NLPCapability.NAMED_ENTITY_RECOGNIZER.value,
                ],
                "model_size": NLPModelSize.TRANSFORMER.value,
                "requires_download": True,
                "description": "Transformer-based English model with high accuracy",
            },
        ]

    async def switch_model(self, model_name: str) -> bool:
        """Switch spaCy model at runtime."""
        try:
            old_model = self.config.model
            self.config.model = model_name
            await self.initialize()
            logger.info("Successfully switched from %s to %s", old_model, model_name)
        except Exception:
            logger.exception("Failed to switch to model %s", model_name)
            return False
        else:
            return True

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        if not self.nlp:
            return {
                "name": self.config.model,
                "language": "en",
                "capabilities": [],
                "pipeline": [],
                "metadata": {},
            }
        return {
            "name": self.nlp.meta["name"],
            "language": self.nlp.meta.get("lang", "en"),
            "capabilities": list(self.nlp.pipe_names),
            "pipeline": list(self.nlp.pipe_names),
            "metadata": {
                "version": self.nlp.meta.get("version", "unknown"),
                "description": self.nlp.meta.get("description", ""),
                "author": self.nlp.meta.get("author", ""),
                "license": self.nlp.meta.get("license", ""),
            },
        }

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        return EmbeddingProviderInfo(
            name=self.provider_name,
            display_name="spaCy",
            description="Industrial-strength natural language processing with local inference",
            supported_capabilities=[],
            capabilities=ProviderCapabilities(
                supports_nlp=True,
                supports_embedding=True,
                supports_local_inference=True,
                supports_multiple_models=True,
                supports_model_switching=True,
                max_batch_size=self.config.batch_size,
                max_input_length=self.config.max_length,
                requires_api_key=False,
                required_dependencies=["spacy"],
                optional_dependencies=["spacy-curated-transformers"],
                default_nlp_model=self.config.model,
                supported_nlp_models=[
                    "en_core_web_sm",
                    "en_core_web_md",
                    "en_core_web_lg",
                    "en_core_web_trf",
                ],
                native_dimensions={
                    "en_core_web_sm": 96,
                    "en_core_web_md": 300,
                    "en_core_web_lg": 300,
                    "en_core_web_trf": 768,
                },
            ),
            requires_api_key=False,
            max_batch_size=self.config.batch_size,
            max_input_length=self.config.max_length,
        )

    def _extract_primary_target(self, entities: list[dict[str, Any]]) -> str | None:
        """Extract primary target from entities."""
        priority_labels = ["CODE_ELEMENT", "LANGUAGE", "FRAMEWORK", "OPERATION", "DATABASE"]
        return next(
            (
                entity["text"]
                for label, entity in itertools.product(priority_labels, entities)
                if entity["label"] == label
            ),
            entities[0]["text"] if entities else None,
        )

    def _get_default_domain_patterns(self) -> list[dict[str, Any]]:
        """Get default code domain patterns."""
        return [
            {
                "label": "LANGUAGE",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "python",
                                "javascript",
                                "typescript",
                                "java",
                                "go",
                                "rust",
                                "c++",
                                "csharp",
                                "ruby",
                                "php",
                                "swift",
                                "kotlin",
                                "scala",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "CODE_ELEMENT",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "function",
                                "class",
                                "method",
                                "variable",
                                "constant",
                                "interface",
                                "module",
                                "package",
                                "component",
                                "service",
                                "endpoint",
                                "route",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "FRAMEWORK",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "react",
                                "vue",
                                "angular",
                                "django",
                                "flask",
                                "express",
                                "spring",
                                "fastapi",
                                "nextjs",
                                "nuxt",
                                "laravel",
                                "rails",
                                "dotnet",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "DATABASE",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "mysql",
                                "postgresql",
                                "mongodb",
                                "redis",
                                "elasticsearch",
                                "sqlite",
                                "oracle",
                                "cassandra",
                                "dynamodb",
                                "firestore",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "OPERATION",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "authentication",
                                "authorization",
                                "validation",
                                "logging",
                                "caching",
                                "monitoring",
                                "deployment",
                                "testing",
                                "debugging",
                                "optimization",
                                "refactoring",
                                "migration",
                                "backup",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "FILE_TYPE",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "json",
                                "yaml",
                                "xml",
                                "csv",
                                "sql",
                                "dockerfile",
                                "makefile",
                                "config",
                                "env",
                                "properties",
                                "manifest",
                                "schema",
                            ]
                        }
                    }
                ],
            },
            {
                "label": "TOOL",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "git",
                                "docker",
                                "kubernetes",
                                "jenkins",
                                "github",
                                "gitlab",
                                "vscode",
                                "intellij",
                                "webpack",
                                "babel",
                                "eslint",
                                "pytest",
                            ]
                        }
                    }
                ],
            },
        ]

    async def health_check(self) -> bool:
        """Check provider health and model availability."""
        return self.nlp is not None


# Register the provider in the global registry
from codeweaver.cw_types import ProviderType, register_provider_class


register_provider_class(ProviderType.SPACY, SpaCyProvider)
