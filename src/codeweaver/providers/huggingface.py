# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
HuggingFace provider implementation for embeddings.

Provides HuggingFace Inference API and local transformers models using the unified provider interface.
Supports both cloud inference and local model loading with GPU acceleration.
"""

import logging

from typing import Any, ClassVar

from codeweaver.providers.base import EmbeddingProviderBase, ProviderCapability, ProviderInfo


try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import torch
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers = None
    torch = None


logger = logging.getLogger(__name__)


class HuggingFaceProvider(EmbeddingProviderBase):
    """HuggingFace provider for embeddings with support for API and local models."""

    # Provider metadata
    PROVIDER_NAME: ClassVar[str] = "huggingface"
    DISPLAY_NAME: ClassVar[str] = "HuggingFace"
    DESCRIPTION: ClassVar[str] = "HuggingFace embeddings via Inference API or local transformers"

    # Popular embedding models on HuggingFace Hub
    POPULAR_MODELS: ClassVar[dict[str, int]] = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
        "microsoft/codebert-base": 768,
        "microsoft/graphcodebert-base": 768,
        "huggingface/CodeBERTa-small-v1": 768,
    }

    # Rate limits for Inference API (requests per hour)
    RATE_LIMITS: ClassVar[dict[str, int]] = {
        "embed_requests": 1000  # Depends on HuggingFace tier
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize HuggingFace provider.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name (default: sentence-transformers/all-MiniLM-L6-v2)
                - api_key: HuggingFace API key (optional for public models)
                - use_local: Whether to use local model loading (default: False)
                - device: Device for local models ('cpu', 'cuda', 'auto') (default: auto)
                - batch_size: Batch size for local processing (default: 16)
        """
        super().__init__(config)

        # Configuration
        self._model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self._api_key = self.config.get("api_key")
        self._use_local = self.config.get("use_local", False)
        self._device = self.config.get("device", "auto")
        self._batch_size = self.config.get("batch_size", 16)

        # Initialize based on mode
        if self._use_local:
            self._init_local_model()
        else:
            self._init_api_client()

        # Get dimension (estimate if unknown)
        self._dimension = self.POPULAR_MODELS.get(self._model_name, 768)

    def _validate_config(self) -> None:
        """Validate HuggingFace configuration."""
        if self._use_local and not TRANSFORMERS_AVAILABLE:
            raise ValueError(
                "Local model mode requires transformers. Install with: uv add transformers torch"
            )

        if not self._use_local and not REQUESTS_AVAILABLE:
            raise ValueError("API mode requires requests. Install with: uv add requests")

        if not self._use_local and not self._api_key:
            logger.warning(
                "No HuggingFace API key provided. Rate limits will apply for public inference."
            )

    def _init_local_model(self) -> None:
        """Initialize local transformers model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch required for local models")

        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)

            # Move to device
            if self._device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)

            logger.info("Loaded local HuggingFace model: %s on %s", self._model_name, self._device)

        except Exception as e:
            raise RuntimeError(f"Failed to load local model {self._model_name}: {e}") from e

    def _init_api_client(self) -> None:
        """Initialize API client for HuggingFace Inference API."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests required for HuggingFace API")

        self._api_url = (
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self._model_name}"
        )
        self._headers = {}

        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

        logger.info("Initialized HuggingFace API client for: %s", self._model_name)

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.PROVIDER_NAME

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """Batch size depends on mode."""
        if self._use_local:
            return self._batch_size
        return 1  # API processes one at a time typically

    @property
    def max_input_length(self) -> int | None:
        """Input length depends on model tokenizer limits."""
        return 8000  # Conservative estimate for most models

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        try:
            if self._use_local:
                return await self._embed_local(texts)
            return await self._embed_api(texts)
        except Exception:
            logger.exception("Error generating HuggingFace embeddings")
            raise

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""
        try:
            embeddings = await self.embed_documents([text])
            return embeddings[0]
        except Exception:
            logger.exception("Error generating HuggingFace query embedding")
            raise

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local model."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Local model not available")

        import torch

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i : i + self._batch_size]

            # Tokenize
            encoded = self._tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            )

            # Move to device
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**encoded)

                # Use mean pooling of last hidden states
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Mean pooling
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                batch_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                # Convert to list
                batch_embeddings = batch_embeddings.cpu().numpy().tolist()
                embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using HuggingFace Inference API."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("API client not available")

        embeddings = []

        for text in texts:
            response = requests.post(
                self._api_url, headers=self._headers, json={"inputs": text}, timeout=30
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace API error: {response.status_code} - {response.text}"
                )

            embedding = response.json()

            # Handle different response formats
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                # Already in correct format
                embeddings.append(embedding[0])
            elif isinstance(embedding, list):
                # Single embedding vector
                embeddings.append(embedding)
            else:
                raise RuntimeError(f"Unexpected API response format: {type(embedding)}")

        return embeddings

    # Provider info methods

    def get_provider_info(self) -> ProviderInfo:
        """Get information about HuggingFace capabilities."""
        return self.get_static_provider_info()

    @classmethod
    def get_static_provider_info(cls) -> ProviderInfo:
        """Get static provider information without instantiation."""
        return ProviderInfo(
            name=cls.PROVIDER_NAME,
            display_name=cls.DISPLAY_NAME,
            description=cls.DESCRIPTION,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.LOCAL_INFERENCE,
                ProviderCapability.BATCH_PROCESSING,
            ],
            default_models={"embedding": "sentence-transformers/all-MiniLM-L6-v2"},
            supported_models={"embedding": list(cls.POPULAR_MODELS.keys())},
            rate_limits=cls.RATE_LIMITS,
            requires_api_key=False,  # Optional for public models
            supports_batch_processing=True,
            max_batch_size=16,
            max_input_length=8000,
            native_dimensions=cls.POPULAR_MODELS,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if HuggingFace is available for the given capability."""
        # Check for API mode
        if not REQUESTS_AVAILABLE:
            api_available = False
            api_reason = "requests package not installed"
        else:
            api_available = True
            api_reason = None

        # Check for local mode
        if not TRANSFORMERS_AVAILABLE:
            local_available = False
            local_reason = "transformers/torch packages not installed"
        else:
            local_available = True
            local_reason = None

        # Only embedding is supported
        if capability == ProviderCapability.EMBEDDING:
            if api_available or local_available:
                return True, None
            return (
                False,
                f"Neither API ({api_reason}) nor local ({local_reason}) mode available",
            )

        return False, f"Capability {capability.value} not supported by HuggingFace"
