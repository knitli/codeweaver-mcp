# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider enums for embedding and reranking capabilities.

This module defines all enums related to provider capabilities, types, and model families.
Replaces string literals and TypedDict usage with proper type-safe enums.
"""

from codeweaver.cw_types.base_enum import BaseEnum


class ProviderCapability(BaseEnum):
    """Provider capability types.

    This enum defines the different capabilities that providers can support.
    Replaces the broken TypedDict that was causing runtime failures.
    """

    EMBEDDING = "embedding"  # Fixed naming to follow singular pattern
    RERANKING = "reranking"
    BATCH_PROCESSING = "batch_processing"
    RATE_LIMITING = "rate_limiting"
    STREAMING = "streaming"
    CUSTOM_DIMENSIONS = "custom_dimensions"
    LOCAL_INFERENCE = "local_inference"


class ProviderKind(BaseEnum):
    """Provider kind/role classification.

    Defines what kind of provider this is based on capabilities.
    """

    EMBEDDING = "embedding"  # Embedding-only provider
    NLP = "nlp"  # NLP-specific provider
    RERANKING = "reranking"  # Reranking-only provider
    COMBINED = "combined"  # Both embedding and reranking
    LOCAL = "local"  # Local inference provider


class ProviderType(BaseEnum):
    """Provider types.

    Enum of all supported provider implementations.
    """

    VOYAGE_AI = "voyage-ai"  # Fixed naming for consistency
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"  # Generic OpenAI-compatible provider
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    NLP = "nlp"  # NLP-specific provider
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    SPACY = "spacy"
    CUSTOM = "custom"

    # Test/Mock providers
    MOCK_EMBEDDING = "mock_embedding"
    MOCK_RERANK = "mock_rerank"


class ModelFamily(BaseEnum):
    """Model families across providers.

    Categorizes models by their primary use case and capabilities.
    """

    CODE_EMBEDDING = "code_embedding"
    TEXT_EMBEDDING = "text_embedding"
    RERANKING = "reranking"
    NLP = "nlp"  # General NLP models
    MULTIMODAL = "multimodal"


class ModelCapabilityEnum(BaseEnum):
    """A common interface for model capabilities."""

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        raise NotImplementedError("Subclasses must implement dimension property")

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        raise NotImplementedError("Subclasses must implement context_length property")

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        raise NotImplementedError("Subclasses must implement model_family property")

    @classmethod
    def member_dimension_map(cls) -> dict[type["ModelCapabilityEnum"], int]:
        """Get a mapping of model names to their native dimensions."""
        return {member: member.dimension for member in cls.members()}

    @classmethod
    def member_context_length_map(cls) -> dict[type["ModelCapabilityEnum"], int]:
        """Get a mapping of model names to their native context lengths."""
        return {member: member.context_length for member in cls.members()}


class VoyageModel(ModelCapabilityEnum):
    """VoyageAI supported models."""

    CODE_3 = "voyage-code-3"
    CONTEXT_3 = "voyage-context-3"
    VOYAGE_3_LARGE = "voyage-3-large"
    VOYAGE_35 = "voyage-3.5"
    VOYAGE_35_LITE = "voyage-3.5-lite"

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        return {
            self.CODE_3: 1024,
            self.CONTEXT_3: 1024,
            self.VOYAGE_35: 1024,
            self.VOYAGE_35_LITE: 1024,
            self.VOYAGE_3_LARGE: 1024,
        }[self]

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.CODE_3: 32000,
            self.CONTEXT_3: 32000,
            self.VOYAGE_35: 32000,
            self.VOYAGE_35_LITE: 32000,
            self.VOYAGE_3_LARGE: 32000,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        if "code" in self.value:
            return ModelFamily.CODE_EMBEDDING
        return ModelFamily.TEXT_EMBEDDING


class VoyageRerankModel(ModelCapabilityEnum):
    """VoyageAI reranking models."""

    RERANK_25 = "voyage-rerank-2.5"
    RERANK_25_LITE = "voyage-rerank-2.5-lite"
    RERANK_2 = "voyage-rerank-2"
    RERANK_2_LITE = "voyage-rerank-2-lite"

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.RERANK_25: 32000,
            self.RERANK_25_LITE: 32000,
            self.RERANK_2: 16000,
            self.RERANK_2_LITE: 8000,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING


class OpenAIModel(ModelCapabilityEnum):
    """OpenAI supported models."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        return {
            self.TEXT_EMBEDDING_3_SMALL: 1536,
            self.TEXT_EMBEDDING_3_LARGE: 3072,
            self.TEXT_EMBEDDING_ADA_002: 1536,
        }[self]

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.TEXT_EMBEDDING_3_SMALL: 8192,
            self.TEXT_EMBEDDING_3_LARGE: 8192,
            self.TEXT_EMBEDDING_ADA_002: 8192,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereModel(ModelCapabilityEnum):
    """Cohere supported models."""

    EMBED_V4 = "embed-v4.0"
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        return {
            self.EMBED_V4: 1536,
            self.EMBED_ENGLISH_V3: 1024,
            self.EMBED_MULTILINGUAL_V4: 1024,
            self.EMBED_ENGLISH_LIGHT_V4: 384,
            self.EMBED_MULTILINGUAL_LIGHT_V4: 384,
        }[self]

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.EMBED_V4: 128000,
            self.EMBED_ENGLISH_V3: 512,
            self.EMBED_MULTILINGUAL_V3: 512,
            self.EMBED_ENGLISH_LIGHT_V3: 512,
            self.EMBED_MULTILINGUAL_LIGHT_V3: 512,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereRerankModel(ModelCapabilityEnum):
    """Cohere reranking models."""

    RERANK_35 = "rerank-3.5"
    RERANK_MULTILINGUAL_3 = "rerank-multilingual-3"
    RERANK_ENGLISH_3 = "rerank-english-3"

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.RERANK_35: 4096,
            self.RERANK_MULTILINGUAL_3: 4096,
            self.RERANK_ENGLISH_3: 4096,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING


class SentenceTransformerModel(ModelCapabilityEnum):
    """Sentence Transformers models."""

    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    ALL_DISTILROBERTA_V1 = "all-distilroberta-v1"

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        return {
            self.ALL_MINI_LM_L6_V2: 384,
            self.ALL_MPNET_BASE_V2: 768,
            self.ALL_DISTILROBERTA_V1: 768,
        }[self]

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.ALL_MINI_LM_L6_V2: 512,
            self.ALL_MPNET_BASE_V2: 512,
            self.ALL_DISTILROBERTA_V1: 512,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class SpaCyModel(ModelCapabilityEnum):
    """Spacy supported models."""

    EN_CORE_WEB_SM = "en_core_web_sm"
    EN_CORE_WEB_MD = "en_core_web_md"
    EN_CORE_WEB_LG = "en_core_web_lg"
    EN_CORE_WEB_TRF = "en_core_web_trf"

    @property
    def dimension(self) -> int:
        """Get native dimension for this model."""
        return {
            self.EN_CORE_WEB_SM: 96,
            self.EN_CORE_WEB_MD: 300,
            self.EN_CORE_WEB_LG: 300,
            self.EN_CORE_WEB_TRF: 768,
        }[self]

    @property
    def context_length(self) -> int:
        """Get context length for this model."""
        return {
            self.EN_CORE_WEB_SM: 512,
            self.EN_CORE_WEB_MD: 512,
            self.EN_CORE_WEB_LG: 512,
            self.EN_CORE_WEB_TRF: 8092,  # default to conservative low for a transformer model
        }

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.NLP


class NLPCapability(BaseEnum):
    """NLP capabilities for models."""

    ATTRIBUTE_RULER = "attribute_ruler"  # Attribute-based entity recognition
    CONREFERENCE_RESOLVER = "coreference_resolver"  # Coreference resolution
    CURATED_TRANSFORMER = "curated_transformer"  # Curated transformer models
    DEPENDENCY_MATCHER = "dependency_matcher"  # Dependency matching
    DEPENDENCY_PARSER = "dependency_parser"  # Dependency parsing
    EDIT_TREE_LEMMATIZER = "edit_tree_lemmatizer"  # Edit tree lemmatization
    ENTITY_LINKER = "entity_linker"  # Entity linking
    ENTITY_RECOGNIZER = "entity_recognizer"  # Named entity recognition
    ENTITY_RULER = "entity_ruler"  # Rule-based entity recognition
    LARGE_LANGUAGE_MODEL = "large_language_model"  # Large language model capabilities
    LEMMATIZER = "lemmatizer"  # Lemmatization
    MATCHER = "matcher"  # General matching capabilities
    MORPHOLOGIZER = "morphologizer"  # Morphological analysis
    NAMED_ENTITY_RECOGNIZER = "named_entity_recognizer"  # Named entity recognition
    OTHER = "other"  # Other NLP capabilities not listed
    PHRASE_MATCHER = "phrase_matcher"  # Phrase matching
    SENTENCE_RECOGNIZER = "sentence_recognizer"  # Sentence boundary detection
    SENTENCIZER = "sentencizer"  # Sentence segmentation
    SPAN_CATEGORIZER = "span_categorizer"  # Span categorization
    SPAN_FINDER = "span_finder"  # Span finding
    TAGGER = "tagger"  # Part-of-speech tagging
    TEXT_CATEGORIZER = "text_categorizer"  # Text categorization
    TOK2VEC = "tok2vec"  # Token-to-vector transformation
    TRAINABLE_PIPE = "trainable_pipe"  # Trainable pipeline components
    TRANSFORMER = "transformer"  # Transformer-based models

    CUSTOM_TRAINING = "custom_training"  # Custom training capabilities

    @property
    def shortform(self) -> str:
        """Get a short form of the capability name."""
        match self:
            case NLPCapability.NAMED_ENTITY_RECOGNIZER:
                return "ner"
            case NLPCapability.SENTENCE_RECOGNIZER:
                return "senter"
            case NLPCapability.LARGE_LANGUAGE_MODEL:
                return "llm"
            case NLPCapability.TEXT_CATEGORIZER:
                return "textcat"
            case NLPCapability.DEPENDENCY_PARSER:
                return "parser"
            case _:
                return self.value.replace("_", " ").lower().replace(" ", "")


class NLPModelSize(BaseEnum):
    """NLP model sizes."""

    SMALL = "sm"
    MEDIUM = "md"
    LARGE = "lg"
    TRANSFORMER = "trf"  # Transformer-based models
