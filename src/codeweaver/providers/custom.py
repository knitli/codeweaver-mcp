# sourcery skip: do-not-use-staticmethod, no-complex-if-expressions
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Enhanced custom provider support system for CodeWeaver.

Provides comprehensive custom provider registration, validation, and plugin-style
integration for embedding and reranking providers with SDK-style development tools.
"""

import contextlib
import importlib.util
import inspect
import logging

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeweaver.providers.base import (
    CombinedProvider,
    EmbeddingProvider,
    EmbeddingProviderBase,
    RerankProvider,
    RerankProviderBase,
)
from codeweaver.types import EmbeddingProviderInfo, ProviderCapabilities, ProviderCapability


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of provider validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    capabilities: list[ProviderCapability] = Field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class CustomProviderRegistration:
    """Registration information for a custom provider."""

    provider_name: str
    provider_class: type[EmbeddingProviderBase | RerankProviderBase | CombinedProvider]
    capabilities: ProviderCapabilities
    display_name: str
    description: str
    validation_result: ValidationResult
    is_available: bool = True
    unavailable_reason: str | None = None

    @property
    def provider_info(self) -> EmbeddingProviderInfo:
        """Get provider info for this registration."""
        # Convert capabilities to list of ProviderCapability enums
        available_capabilities = []
        if self.capabilities.supports_embedding:
            available_capabilities.append(ProviderCapability.EMBEDDING)
        if self.capabilities.supports_reranking:
            available_capabilities.append(ProviderCapability.RERANKING)

        return EmbeddingProviderInfo(
            name=self.provider_name,
            display_name=self.display_name,
            description=self.description,
            supported_capabilities=available_capabilities,
            capabilities=self.capabilities,
            requires_api_key=self.capabilities.requires_api_key,
            max_batch_size=self.capabilities.max_batch_size,
            max_input_length=self.capabilities.max_input_length,
            default_models={
                "embedding": self.capabilities.default_embedding_model,
                "reranking": self.capabilities.default_reranking_model,
            }
            if self.capabilities.default_embedding_model
            or self.capabilities.default_reranking_model
            else None,
            supported_models={
                "embedding": self.capabilities.supported_embedding_models or [],
                "reranking": self.capabilities.supported_reranking_models or [],
            },
            rate_limits={
                "requests_per_minute": self.capabilities.requests_per_minute,
                "tokens_per_minute": self.capabilities.tokens_per_minute,
            }
            if self.capabilities.requests_per_minute or self.capabilities.tokens_per_minute
            else None,
            native_dimensions=self.capabilities.native_dimensions,
        )


class ProviderImplementationValidator:
    """Validates custom provider implementations against required interfaces."""

    @staticmethod
    def validate_embedding_provider(provider_class: type) -> ValidationResult:
        """Validate an embedding provider implementation.

        Args:
            provider_class: The provider class to validate

        Returns:
            Validation result with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check if it's a proper class
        if not inspect.isclass(provider_class):
            result.add_error("Provider must be a class")
            return result

        # Check protocol compliance - protocols with properties can't use issubclass directly
        try:
            is_base_subclass = issubclass(provider_class, EmbeddingProviderBase)
        except TypeError:
            is_base_subclass = False

        if not is_base_subclass and not ProviderImplementationValidator._implements_protocol(
            provider_class, EmbeddingProvider
        ):
            result.add_error(
                "Provider must implement EmbeddingProvider protocol or inherit from "
                "EmbeddingProviderBase"
            )

        # Check required methods
        required_methods = {
            "embed_documents": (
                "async def embed_documents(self, texts: list[str]) -> list[list[float]]"
            ),
            "embed_query": "async def embed_query(self, text: str) -> list[float]",
            "get_provider_info": "def get_provider_info(self) -> EmbeddingProviderInfo",
        }

        ProviderImplementationValidator._validate_methods(provider_class, required_methods, result)
        required_properties = {
            "provider_name": "str",
            "model_name": "str",
            "dimension": "int",
            "max_batch_size": "int | None",
            "max_input_length": "int | None",
        }
        ProviderImplementationValidator._validate_properties(
            provider_class, required_properties, result
        )

        if result.is_valid:
            result.capabilities.append(ProviderCapability.EMBEDDING)

        return result

    @staticmethod
    def validate_reranking_provider(provider_class: type) -> ValidationResult:
        """Validate a reranking provider implementation.

        Args:
            provider_class: The provider class to validate

        Returns:
            Validation result with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check if it's a proper class
        if not inspect.isclass(provider_class):
            result.add_error("Provider must be a class")
            return result

        # Check protocol compliance - protocols with properties can't use issubclass directly
        try:
            is_base_subclass = issubclass(provider_class, RerankProviderBase)
        except TypeError:
            is_base_subclass = False

        if not is_base_subclass and not ProviderImplementationValidator._implements_protocol(
            provider_class, RerankProvider
        ):
            result.add_error(
                "Provider must implement RerankProvider protocol or inherit from RerankProviderBase"
            )

        # Check required methods
        required_methods = {
            "rerank": (
                "async def rerank(self, query: str, documents: list[str], "
                "top_k: int | None = None) -> list[RerankResult]"
            ),
            "get_provider_info": "def get_provider_info(self) -> EmbeddingProviderInfo",
        }

        ProviderImplementationValidator._validate_methods(provider_class, required_methods, result)
        required_properties = {
            "provider_name": "str",
            "model_name": "str",
            "max_documents": "int | None",
            "max_query_length": "int | None",
        }
        ProviderImplementationValidator._validate_properties(
            provider_class, required_properties, result
        )

        if result.is_valid:
            result.capabilities.append(ProviderCapability.RERANKING)

        return result

    @staticmethod
    def validate_combined_provider(provider_class: type) -> ValidationResult:
        """Validate a combined provider implementation.

        Args:
            provider_class: The provider class to validate

        Returns:
            Validation result with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check if it's a proper class
        if not inspect.isclass(provider_class):
            result.add_error("Provider must be a class")
            return result

        # Check if it inherits from CombinedProvider or implements both protocols
        if not (
            issubclass(provider_class, CombinedProvider)
            or (
                ProviderImplementationValidator._implements_protocol(
                    provider_class, EmbeddingProvider
                )
                and ProviderImplementationValidator._implements_protocol(
                    provider_class, RerankProvider
                )
            )
        ):
            result.add_error(
                "Combined provider must inherit from CombinedProvider or implement both "
                "EmbeddingProvider and RerankProvider protocols"
            )

        # Validate both embedding and reranking capabilities
        embedding_result = ProviderImplementationValidator.validate_embedding_provider(
            provider_class
        )
        reranking_result = ProviderImplementationValidator.validate_reranking_provider(
            provider_class
        )

        # Combine results
        result.errors.extend(embedding_result.errors)
        result.errors.extend(reranking_result.errors)
        result.warnings.extend(embedding_result.warnings)
        result.warnings.extend(reranking_result.warnings)
        result.capabilities.extend(embedding_result.capabilities)
        result.capabilities.extend(reranking_result.capabilities)

        result.is_valid = embedding_result.is_valid and reranking_result.is_valid

        return result

    @staticmethod
    def _implements_protocol(provider_class: type, protocol: type) -> bool:
        """Check if a class implements a protocol."""
        try:
            return isinstance(provider_class(), protocol)
        except Exception:
            # If we can't instantiate, check structurally
            return hasattr(provider_class, "__annotations__") and all(
                hasattr(provider_class, attr) for attr in getattr(protocol, "__annotations__", {})
            )

    @staticmethod
    def _validate_methods(
        provider_class: type, required_methods: dict[str, str], result: ValidationResult
    ) -> None:
        """Validate that required methods exist."""
        for method_name, signature in required_methods.items():
            if not hasattr(provider_class, method_name):
                result.add_error(f"Missing required method: {signature}")
            else:
                method = getattr(provider_class, method_name)
                if not callable(method):
                    result.add_error(f"'{method_name}' must be callable")

    @staticmethod
    def _validate_properties(
        provider_class: type, required_properties: dict[str, str], result: ValidationResult
    ) -> None:
        """Validate that required properties exist."""
        for prop_name, prop_type in required_properties.items():
            if not hasattr(provider_class, prop_name):
                result.add_error(f"Missing required property: {prop_name}: {prop_type}")


class ProviderCapabilityDetector:
    """Automatically detects provider capabilities from implementation."""

    @staticmethod
    def detect_capabilities(provider_class: type) -> ProviderCapabilities:
        """Detect provider capabilities from class implementation.

        Args:
            provider_class: The provider class to analyze

        Returns:
            Detected capabilities
        """
        capabilities = ProviderCapabilities()

        # Check embedding support
        if hasattr(provider_class, "embed_documents") and hasattr(provider_class, "embed_query"):
            capabilities.supports_embedding = True

        # Check reranking support
        if hasattr(provider_class, "rerank"):
            capabilities.supports_reranking = True

        # Check batch processing support
        if hasattr(provider_class, "max_batch_size"):
            with contextlib.suppress(Exception):
                max_batch = provider_class.max_batch_size
                if max_batch is not None and max_batch > 1:
                    capabilities.supports_batch_processing = True
                    capabilities.max_batch_size = max_batch

        # Check API key requirement
        if hasattr(provider_class, "requires_api_key"):
            try:
                capabilities.requires_api_key = provider_class.requires_api_key
            except Exception:
                capabilities.requires_api_key = True  # Default to True for safety

        # Try to detect rate limiting support
        if any(
            hasattr(provider_class, attr)
            for attr in ["rate_limiter", "_rate_limiter", "requests_per_minute"]
        ):
            capabilities.supports_rate_limiting = True

        # Try to detect local inference support
        if hasattr(provider_class, "supports_local_inference"):
            with contextlib.suppress(Exception):
                capabilities.supports_local_inference = provider_class.supports_local_inference

        return capabilities


class EnhancedProviderRegistry:
    """Enhanced registry with comprehensive custom provider support."""

    _custom_providers: ClassVar[dict[str, CustomProviderRegistration]] = {}
    _plugin_directories: ClassVar[list[Path]] = []
    _validation_rules: ClassVar[dict[str, Callable[[type], ValidationResult]]] = {}

    @classmethod
    def register_custom_provider(
        cls,
        provider_name: str,
        provider_class: type[EmbeddingProviderBase | RerankProviderBase | CombinedProvider],
        capabilities: ProviderCapabilities | None = None,
        display_name: str | None = None,
        description: str | None = None,
        *,
        validate_implementation: bool = True,
        auto_detect_capabilities: bool = True,
    ) -> ValidationResult:
        """Register a custom provider with comprehensive validation.

        Args:
            provider_name: Unique name for the provider
            provider_class: Provider implementation class
            capabilities: Provider capabilities (auto-detected if None)
            display_name: Human-readable provider name (defaults to provider_name)
            description: Provider description
            validate_implementation: Whether to validate implementation
            auto_detect_capabilities: Whether to auto-detect capabilities

        Returns:
            Validation result indicating success or failure

        Raises:
            ValueError: If provider_name is already registered
        """
        if provider_name in cls._custom_providers:
            raise ValueError(f"Provider '{provider_name}' is already registered")

        # Use defaults
        display_name = display_name or provider_name.replace("_", " ").title()
        description = description or f"Custom {display_name} provider"

        # Auto-detect capabilities if not provided
        if capabilities is None and auto_detect_capabilities:
            capabilities = ProviderCapabilityDetector.detect_capabilities(provider_class)
        elif capabilities is None:
            capabilities = ProviderCapabilities()

        # Validate implementation
        validation_result = ValidationResult(is_valid=True)
        if validate_implementation:
            validation_result = cls._validate_provider_implementation(provider_class, capabilities)

        # Check availability
        is_available = validation_result.is_valid
        unavailable_reason = None
        if not validation_result.is_valid:
            unavailable_reason = "; ".join(validation_result.errors)

        # Create registration
        registration = CustomProviderRegistration(
            provider_name=provider_name,
            provider_class=provider_class,
            capabilities=capabilities,
            display_name=display_name,
            description=description,
            validation_result=validation_result,
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )

        cls._custom_providers[provider_name] = registration

        # Log registration
        if is_available:
            logger.info("Registered custom provider: %s (%s)", provider_name, display_name)
        else:
            logger.warning(
                "Registered custom provider %s with validation errors: %s",
                provider_name,
                unavailable_reason,
            )

        return validation_result

    @classmethod
    def unregister_custom_provider(cls, provider_name: str) -> bool:
        """Unregister a custom provider.

        Args:
            provider_name: Name of the provider to unregister

        Returns:
            True if provider was unregistered, False if not found
        """
        if provider_name in cls._custom_providers:
            del cls._custom_providers[provider_name]
            logger.info("Unregistered custom provider: %s", provider_name)
            return True
        return False

    @classmethod
    def get_custom_provider_registration(
        cls, provider_name: str
    ) -> CustomProviderRegistration | None:
        """Get custom provider registration by name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider registration or None if not found
        """
        return cls._custom_providers.get(provider_name)

    @classmethod
    def get_all_custom_providers(cls) -> dict[str, CustomProviderRegistration]:
        """Get all custom provider registrations."""
        return cls._custom_providers.copy()

    @classmethod
    def get_available_custom_providers(cls) -> dict[str, CustomProviderRegistration]:
        """Get all available custom provider registrations."""
        return {name: reg for name, reg in cls._custom_providers.items() if reg.is_available}

    @classmethod
    def add_plugin_directory(cls, directory: Path | str) -> None:
        """Add a directory to search for plugin providers.

        Args:
            directory: Directory path to search for plugins
        """
        dir_path = Path(directory)
        if dir_path.is_dir():
            cls._plugin_directories.append(dir_path)
            logger.info("Added plugin directory: %s", dir_path)
        else:
            logger.warning("Plugin directory does not exist: %s", dir_path)

    @classmethod
    def discover_plugins(cls, *, auto_register: bool = True) -> list[str]:
        """Discover and optionally register plugin providers.

        Args:
            auto_register: Whether to automatically register discovered providers

        Returns:
            List of discovered provider names
        """
        discovered = []

        for plugin_dir in cls._plugin_directories:
            try:
                discovered.extend(
                    cls._discover_plugins_in_directory(plugin_dir, auto_register=auto_register)
                )
            except Exception:
                logger.exception("Failed to discover plugins in %s")

        return discovered

    @classmethod
    def add_validation_rule(
        cls, rule_name: str, validator: Callable[[type], ValidationResult]
    ) -> None:
        """Add a custom validation rule.

        Args:
            rule_name: Name of the validation rule
            validator: Validation function that takes a provider class and returns ValidationResult
        """
        cls._validation_rules[rule_name] = validator
        logger.info("Added custom validation rule: %s", rule_name)

    @classmethod
    def _validate_provider_implementation(
        cls, provider_class: type, capabilities: ProviderCapabilities
    ) -> ValidationResult:
        """Validate a provider implementation against capabilities.

        Args:
            provider_class: Provider class to validate
            capabilities: Expected capabilities

        Returns:
            Validation result
        """
        results = []

        # Validate based on capabilities
        if capabilities.supports_embedding:
            results.append(
                ProviderImplementationValidator.validate_embedding_provider(provider_class)
            )

        if capabilities.supports_reranking:
            results.append(
                ProviderImplementationValidator.validate_reranking_provider(provider_class)
            )

        if capabilities.supports_embedding and capabilities.supports_reranking:
            results.append(
                ProviderImplementationValidator.validate_combined_provider(provider_class)
            )

        # Apply custom validation rules
        for rule_name, validator in cls._validation_rules.items():
            try:
                rule_result = validator(provider_class)
                results.append(rule_result)
                logger.debug(
                    "Applied validation rule '%s' to %s", rule_name, provider_class.__name__
                )
            except Exception:
                logger.exception("Validation rule '%s' failed")

        # Combine all results
        combined_result = ValidationResult(is_valid=True)
        for result in results:
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
            combined_result.capabilities.extend(result.capabilities)
            if not result.is_valid:
                combined_result.is_valid = False

        return combined_result

    @classmethod
    def _discover_plugins_in_directory(cls, directory: Path, *, auto_register: bool) -> list[str]:
        """Discover plugins in a directory.

        Args:
            directory: Directory to search
            auto_register: Whether to auto-register discovered plugins

        Returns:
            List of discovered provider names
        """
        discovered = []

        # Look for Python files
        for py_file in directory.glob("*.py"):
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for registration functions
                    if hasattr(module, "register_provider"):
                        provider_info = module.register_provider()
                        if auto_register and isinstance(provider_info, dict):
                            cls.register_custom_provider(**provider_info)
                            discovered.append(provider_info.get("provider_name", py_file.stem))

                    # Look for provider classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            name.endswith("Provider")
                            and obj.__module__ == module.__name__
                            and any(
                                issubclass(obj, base)
                                for base in [
                                    EmbeddingProviderBase,
                                    RerankProviderBase,
                                    CombinedProvider,
                                ]
                            )
                        ) and auto_register:
                            cls.register_custom_provider(
                                provider_name=name.lower().replace("provider", ""),
                                provider_class=obj,
                            )
                            discovered.append(name.lower().replace("provider", ""))

            except Exception:
                logger.exception("Failed to process plugin file %s")

        return discovered


@runtime_checkable
class ProviderTemplate(Protocol):
    """Template interface for custom provider development."""

    @abstractmethod
    def get_template_info(self) -> dict[str, Any]:
        """Get template information."""
        ...


class ProviderSDK:
    """SDK for custom provider development with templates and utilities."""

    EMBEDDING_TEMPLATE = '''"""
Custom embedding provider template.

Generated by CodeWeaver ProviderSDK.
"""

from typing import Any
from codeweaver.providers.base import EmbeddingProviderBase
from codeweaver.types import EmbeddingProviderInfo
from codeweaver.types import ProviderCapabilities
from codeweaver.types import ProviderCapability


class {class_name}(EmbeddingProviderBase):
    """Custom embedding provider: {display_name}."""

    def __init__(self, config: dict[str, Any], **kwargs):
        """Initialize the provider.

        Args:
            config: Provider configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)
        # Initialize your provider here

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        # Add your configuration validation here
        pass

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "{provider_name}"

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self.config.get("model_name", "{default_model}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return {dimension}

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size."""
        return {max_batch_size}

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length."""
        return {max_input_length}

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # Implement your embedding logic here
        embeddings = []
        for text in texts:
            embedding = await self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        # Implement your embedding logic here
        # This is a placeholder - replace with actual implementation
        import random
        return [random.random() for _ in range(self.dimension)]

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider."""
        return EmbeddingProviderInfo(
            name=self.provider_name,
            display_name="{display_name}",
            description="{description}",
            supported_capabilities=[ProviderCapability.EMBEDDING],
            capabilities=ProviderCapabilities(
                supports_embedding=True,
                max_batch_size={max_batch_size},
                max_input_length={max_input_length},
                requires_api_key={requires_api_key},
                default_embedding_model="{default_model}",
                native_dimensions={{"{default_model}": {dimension}}},
            ),
            requires_api_key={requires_api_key},
            max_batch_size={max_batch_size},
            max_input_length={max_input_length},
        )


# Registration function for plugin discovery
def register_provider() -> dict[str, Any]:
    """Register this provider for plugin discovery."""
    return {{
        "provider_name": "{provider_name}",
        "provider_class": {class_name},
        "display_name": "{display_name}",
        "description": "{description}",
    }}
'''

    RERANKING_TEMPLATE = '''"""
Custom reranking provider template.

Generated by CodeWeaver ProviderSDK.
"""

from typing import Any
from codeweaver.providers.base import RerankProviderBase
from codeweaver.types import EmbeddingProviderInfo
from codeweaver.types import ProviderCapabilities
from codeweaver.types import ProviderCapability
from codeweaver.types import RerankResult


class {class_name}(RerankProviderBase):
    """Custom reranking provider: {display_name}."""

    def __init__(self, config: dict[str, Any], **kwargs):
        """Initialize the provider.

        Args:
            config: Provider configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)
        # Initialize your provider here

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        # Add your configuration validation here
        pass

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "{provider_name}"

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self.config.get("model_name", "{default_model}")

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents."""
        return {max_documents}

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length."""
        return {max_query_length}

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return

        Returns:
            List of rerank results ordered by relevance
        """
        # Implement your reranking logic here
        # This is a placeholder - replace with actual implementation
        import random

        results = []
        for i, doc in enumerate(documents):
            # Generate a random relevance score (replace with actual logic)
            relevance_score = random.random()
            results.append(RerankResult(
                index=i,
                relevance_score=relevance_score,
                document=doc
            ))

        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider."""
        return EmbeddingProviderInfo(
            name=self.provider_name,
            display_name="{display_name}",
            description="{description}",
            supported_capabilities=[ProviderCapability.RERANKING],
            capabilities=ProviderCapabilities(
                supports_reranking=True,
                max_documents={max_documents},
                max_query_length={max_query_length},
                requires_api_key={requires_api_key},
                default_reranking_model="{default_model}",
            ),
            requires_api_key={requires_api_key},
        )


# Registration function for plugin discovery
def register_provider() -> dict[str, Any]:
    """Register this provider for plugin discovery."""
    return {{
        "provider_name": "{provider_name}",
        "provider_class": {class_name},
        "display_name": "{display_name}",
        "description": "{description}",
    }}
'''

    @classmethod
    def generate_embedding_provider_template(
        cls,
        provider_name: str,
        display_name: str | None = None,
        description: str | None = None,
        dimension: int = 384,
        max_batch_size: int | None = 32,
        max_input_length: int | None = 512,
        *,
        requires_api_key: bool = False,
        default_model: str = "default",
    ) -> str:
        """Generate an embedding provider template.

        Args:
            provider_name: Unique provider name
            display_name: Human-readable name
            description: Provider description
            dimension: Embedding dimension
            max_batch_size: Maximum batch size
            max_input_length: Maximum input length
            requires_api_key: Whether API key is required
            default_model: Default model name

        Returns:
            Generated provider template code
        """
        display_name = display_name or provider_name.replace("_", " ").title()
        description = description or f"Custom {display_name} embedding provider"
        class_name = f"{provider_name.replace('_', '').title()}Provider"

        return cls.EMBEDDING_TEMPLATE.format(
            class_name=class_name,
            provider_name=provider_name,
            display_name=display_name,
            description=description,
            dimension=dimension,
            max_batch_size=max_batch_size,
            max_input_length=max_input_length,
            requires_api_key=requires_api_key,
            default_model=default_model,
            supports_batch=max_batch_size is not None and max_batch_size > 1,
        )

    @classmethod
    def generate_reranking_provider_template(
        cls,
        provider_name: str,
        display_name: str | None = None,
        description: str | None = None,
        max_documents: int | None = 100,
        max_query_length: int | None = 1000,
        *,
        requires_api_key: bool = False,
        default_model: str = "default",
    ) -> str:
        """Generate a reranking provider template.

        Args:
            provider_name: Unique provider name
            display_name: Human-readable name
            description: Provider description
            max_documents: Maximum documents to rerank
            max_query_length: Maximum query length
            requires_api_key: Whether API key is required
            default_model: Default model name

        Returns:
            Generated provider template code
        """
        display_name = display_name or provider_name.replace("_", " ").title()
        description = description or f"Custom {display_name} reranking provider"
        class_name = f"{provider_name.replace('_', '').title()}Provider"

        return cls.RERANKING_TEMPLATE.format(
            class_name=class_name,
            provider_name=provider_name,
            display_name=display_name,
            description=description,
            max_documents=max_documents,
            max_query_length=max_query_length,
            requires_api_key=requires_api_key,
            default_model=default_model,
        )

    @classmethod
    def create_provider_file(
        cls, output_path: Path | str, provider_type: str, **template_kwargs
    ) -> None:
        """Create a provider file from template.

        Args:
            output_path: Path to write the provider file
            provider_type: Type of provider ('embedding' or 'reranking')
            **template_kwargs: Arguments for template generation
        """
        output_path = Path(output_path)

        if provider_type == "embedding":
            content = cls.generate_embedding_provider_template(**template_kwargs)
        elif provider_type == "reranking":
            content = cls.generate_reranking_provider_template(**template_kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        output_path.write_text(content)
        logger.info("Created %s provider template at %s", provider_type, output_path)

    @classmethod
    def validate_provider_implementation(cls, provider_class: type) -> ValidationResult:
        """Validate a provider implementation.

        Args:
            provider_class: Provider class to validate

        Returns:
            Validation result
        """
        # Try to detect capabilities first
        capabilities = ProviderCapabilityDetector.detect_capabilities(provider_class)

        # Validate based on detected capabilities
        if capabilities.supports_embedding and capabilities.supports_reranking:
            return ProviderImplementationValidator.validate_combined_provider(provider_class)
        if capabilities.supports_embedding:
            return ProviderImplementationValidator.validate_embedding_provider(provider_class)
        if capabilities.supports_reranking:
            return ProviderImplementationValidator.validate_reranking_provider(provider_class)
        result = ValidationResult(is_valid=True)
        result.add_error("Provider does not implement any recognized capabilities")
        return result


# importlib.util imported at top for plugin discovery


# Helper functions for easy registration
def register_embedding_provider(
    provider_name: str, provider_class: type[EmbeddingProviderBase], **kwargs
) -> ValidationResult:
    """Helper function to register an embedding provider.

    Args:
        provider_name: Unique provider name
        provider_class: Provider implementation class
        **kwargs: Additional registration arguments

    Returns:
        Validation result
    """
    return EnhancedProviderRegistry.register_custom_provider(
        provider_name=provider_name, provider_class=provider_class, **kwargs
    )


def register_reranking_provider(
    provider_name: str, provider_class: type[RerankProviderBase], **kwargs
) -> ValidationResult:
    """Helper function to register a reranking provider.

    Args:
        provider_name: Unique provider name
        provider_class: Provider implementation class
        **kwargs: Additional registration arguments

    Returns:
        Validation result
    """
    return EnhancedProviderRegistry.register_custom_provider(
        provider_name=provider_name, provider_class=provider_class, **kwargs
    )


def register_combined_provider(
    provider_name: str, provider_class: type[CombinedProvider], **kwargs
) -> ValidationResult:
    """Helper function to register a combined provider.

    Args:
        provider_name: Unique provider name
        provider_class: Provider implementation class
        **kwargs: Additional registration arguments

    Returns:
        Validation result
    """
    return EnhancedProviderRegistry.register_custom_provider(
        provider_name=provider_name, provider_class=provider_class, **kwargs
    )
