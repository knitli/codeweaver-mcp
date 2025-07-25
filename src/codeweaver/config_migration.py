# sourcery skip: avoid-global-variables, do-not-use-staticmethod
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Configuration migration utilities for CodeWeaver.

Provides tools to migrate between configuration formats and validate
configuration compatibility across different backend/provider combinations.
"""

import logging
import os

from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Literal


try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    import tomli as tomllib

try:
    import tomlkit
except ImportError:
    tomlkit = None

from codeweaver.config import CodeWeaverConfig


logger = logging.getLogger(__name__)


class ConfigMigrationError(Exception):
    """Exception raised during configuration migration."""


BackendType = Literal["qdrant", "pinecone", "huggingface", "chroma", "weaviate", "pgvector", "mulvus", "elasticsearch"]

EmbeddingProvider = Literal["voyage", "openai", "cohere", "sentence-transformers", "huggingface"]


class ConfigValidator:
    """Validates configuration combinations and compatibility."""

    # Supported combinations matrix
    BACKEND_EMBEDDING_COMPATIBILITY: ClassVar[dict[BackendType, list[EmbeddingProvider]]] = {
        "qdrant": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "pinecone": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "chroma": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "weaviate": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "pgvector": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "milvus": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
        "elasticsearch": ["voyage", "openai", "cohere", "sentence-transformers", "huggingface"],
    }

    HYBRID_SEARCH_BACKENDS: ClassVar[list[BackendType]] = ["qdrant", "weaviate", "milvus"]
    RERANKING_PROVIDERS: ClassVar[list[EmbeddingProvider]] = ["voyage", "cohere"]
    LOCAL_PROVIDERS: ClassVar[list[EmbeddingProvider]] = ["sentence-transformers", "huggingface"]

    @classmethod
    def validate_backend_provider_combination(cls, backend: str, provider: str) -> list[str]:
        """Validate backend and embedding provider combination.

        Args:
            backend: Backend provider name
            provider: Embedding provider name

        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []

        if backend not in cls.BACKEND_EMBEDDING_COMPATIBILITY:
            warnings.append(f"Unknown backend provider: {backend}")
        elif provider not in cls.BACKEND_EMBEDDING_COMPATIBILITY[backend]:
            warnings.append(
                f"Embedding provider '{provider}' may not be fully compatible with backend '{backend}'. "
                f"Recommended providers: {', '.join(cls.BACKEND_EMBEDDING_COMPATIBILITY[backend])}"
            )

        return warnings

    @classmethod
    def validate_hybrid_search_config(
        cls, backend: str, *, enable_hybrid: bool, enable_sparse: bool
    ) -> list[str]:
        """Validate hybrid search configuration.

        Args:
            backend: Backend provider name
            enable_hybrid: Whether hybrid search is enabled
            enable_sparse: Whether sparse vectors are enabled

        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []

        if enable_hybrid and backend not in cls.HYBRID_SEARCH_BACKENDS:
            warnings.append(
                f"Hybrid search is not supported by backend '{backend}'. "
                f"Supported backends: {', '.join(cls.HYBRID_SEARCH_BACKENDS)}"
            )

        if enable_sparse and backend not in cls.HYBRID_SEARCH_BACKENDS:
            warnings.append(
                f"Sparse vectors are not supported by backend '{backend}'. "
                f"Supported backends: {', '.join(cls.HYBRID_SEARCH_BACKENDS)}"
            )

        if enable_hybrid and not enable_sparse and backend in cls.HYBRID_SEARCH_BACKENDS:
            warnings.append(
                "Hybrid search is enabled but sparse vectors are disabled. "
                "Consider enabling sparse vectors for better hybrid search performance."
            )

        return warnings

    @classmethod
    def validate_reranking_config(cls, provider: str, rerank_provider: str | None) -> list[str]:
        """Validate reranking configuration.

        Args:
            provider: Embedding provider name
            rerank_provider: Reranking provider name

        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []

        if rerank_provider and rerank_provider not in cls.RERANKING_PROVIDERS:
            warnings.append(
                f"Reranking provider '{rerank_provider}' is not supported. "
                f"Supported providers: {', '.join(cls.RERANKING_PROVIDERS)}"
            )

        if not rerank_provider and provider in cls.RERANKING_PROVIDERS:
            warnings.append(
                f"Embedding provider '{provider}' supports reranking but rerank_provider is not set. "
                f"Consider enabling reranking for better search quality."
            )

        return warnings

    @classmethod
    def validate_local_provider_config(
        cls, provider: str, *, use_local: bool, api_key: str | None
    ) -> list[str]:
        """Validate local provider configuration.

        Args:
            provider: Embedding provider name
            use_local: Whether to use local models
            api_key: API key

        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []

        if provider in cls.LOCAL_PROVIDERS and not use_local and not api_key:
            warnings.append(
                f"Provider '{provider}' requires either use_local=true or an API key. "
                f"Set USE_LOCAL_MODELS=true or provide an API key."
            )

        if provider not in cls.LOCAL_PROVIDERS and use_local:
            warnings.append(
                f"Provider '{provider}' does not support local models. "
                f"Local model providers: {', '.join(cls.LOCAL_PROVIDERS)}"
            )

        return warnings

    @classmethod
    def validate_full_config(cls, config: CodeWeaverConfig) -> list[str]:
        """Validate entire configuration for compatibility issues.

        Args:
            config: Configuration to validate

        Returns:
            List of validation warnings and errors
        """
        warnings = []

        # Get effective values
        backend = config.get_effective_backend_provider()
        provider = config.get_effective_embedding_provider()

        # Validate backend/provider combination
        warnings.extend(cls.validate_backend_provider_combination(backend, provider))

        # Validate hybrid search configuration
        warnings.extend(
            cls.validate_hybrid_search_config(
                backend=backend,
                enable_hybrid=config.backend.enable_hybrid_search,
                enable_sparse=config.backend.enable_sparse_vectors,
            )
        )

        # Validate reranking configuration
        rerank_provider = config.provider.rerank_provider or config.embedding.rerank_provider
        warnings.extend(cls.validate_reranking_config(provider, rerank_provider))

        # Validate local provider configuration
        api_key = config.provider.embedding_api_key or config.embedding.api_key
        use_local = config.provider.use_local or config.embedding.use_local
        warnings.extend(cls.validate_local_provider_config(provider, use_local, api_key))

        # Validate data sources
        if config.data_sources.enabled and not config.data_sources.sources:
            warnings.append("Data sources are enabled but no sources are configured")

        return warnings


class ConfigMigrator:
    """Handles migration between configuration formats."""

    @staticmethod
    def detect_config_format(config_data: dict[str, Any]) -> Literal["legacy", "new", "mixed"]:
        """Detect the format of a configuration dictionary.

        Args:
            config_data: Configuration dictionary

        Returns:
            Configuration format type
        """
        has_legacy = any(key in config_data for key in ["embedding", "qdrant"])
        has_new = any(key in config_data for key in ["backend", "provider", "data_sources"])

        if has_legacy and has_new:
            return "mixed"
        if has_legacy:
            return "legacy"
        return "new" if has_new else "legacy"

    @staticmethod
    def migrate_legacy_to_new(legacy_dict: dict[str, Any]) -> dict[str, Any]:
        """Migrate legacy configuration dictionary to new format.

        Args:
            legacy_dict: Legacy configuration dictionary

        Returns:
            New format configuration dictionary
        """
        new_data = {}

        # Migrate embedding configuration to provider configuration
        if "embedding" in legacy_dict:
            embedding_config = legacy_dict["embedding"]
            if isinstance(embedding_config, dict):
                new_data["provider"] = {
                    "embedding_provider": embedding_config.get("provider", "voyage"),
                    "embedding_api_key": embedding_config.get("api_key"),
                    "embedding_model": embedding_config.get("model", "voyage-code-3"),
                    "embedding_dimension": embedding_config.get("dimension", 1024),
                    "embedding_batch_size": embedding_config.get("batch_size", 8),
                    "rerank_provider": embedding_config.get("rerank_provider"),
                    "rerank_model": embedding_config.get("rerank_model"),
                    "base_url": embedding_config.get("base_url"),
                    "custom_headers": embedding_config.get("custom_headers", {}),
                    "use_local": embedding_config.get("use_local", False),
                    "device": embedding_config.get("device", "auto"),
                    "normalize_embeddings": embedding_config.get("normalize_embeddings", True),
                }

        # Migrate Qdrant configuration to backend configuration
        if "qdrant" in legacy_dict:
            qdrant_config = legacy_dict["qdrant"]
            if isinstance(qdrant_config, dict):
                new_data["backend"] = {
                    "provider": "qdrant",
                    "url": qdrant_config.get("url"),
                    "api_key": qdrant_config.get("api_key"),
                    "collection_name": qdrant_config.get("collection_name", "code-embeddings"),
                    "enable_sparse_vectors": qdrant_config.get("enable_sparse_vectors", False),
                    "enable_hybrid_search": qdrant_config.get("enable_sparse_vectors", False),
                }

        # Create default data sources configuration
        if "data_sources" not in legacy_dict:
            new_data["data_sources"] = {
                "enabled": True,
                "default_source_type": "filesystem",
                "sources": [
                    {
                        "type": "filesystem",
                        "enabled": True,
                        "priority": 1,
                        "source_id": "default_filesystem",
                        "config": {
                            "root_path": ".",
                            "use_gitignore": legacy_dict.get("indexing", {}).get(
                                "use_gitignore", True
                            ),
                            "additional_ignore_patterns": legacy_dict.get("indexing", {}).get(
                                "additional_ignore_patterns", []
                            ),
                            "max_file_size_mb": legacy_dict.get("chunking", {}).get(
                                "max_file_size_mb", 1
                            ),
                        },
                    }
                ],
            }

        # Copy other sections as-is
        for key, value in legacy_dict.items():
            if key not in ["embedding", "qdrant"] and key not in new_data:
                new_data[key] = value

        # Add configuration version
        new_data["_config_version"] = "2.0"

        return new_data

    @staticmethod
    def create_migration_script(
        input_path: str | Path,
        output_path: str | Path | None = None,
        fmt: Literal["toml", "dict"] = "toml",
    ) -> str:
        """Create a migration script from legacy to new configuration.

        Args:
            input_path: Path to legacy configuration file
            output_path: Path for new configuration file (optional)
            fmt: Output fmt (toml or dict)

        Returns:
            Migration script content or new configuration

        Raises:
            ConfigMigrationError: If migration fails
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise ConfigMigrationError(f"Configuration file not found: {input_path}")

        try:
            with input_path.open("rb") as f:
                legacy_config = tomllib.load(f)
        except Exception as e:
            raise ConfigMigrationError("Failed to load configuration file") from e

        config_format = ConfigMigrator.detect_config_format(legacy_config)

        if config_format == "new":
            logger.info("Configuration is already in new format")
            return "# Configuration is already in new format"

        # Migrate to new format
        new_config = ConfigMigrator.migrate_legacy_to_new(legacy_config)

        if fmt == "dict":
            return str(new_config)

        # Generate TOML output
        if tomlkit is None:
            raise ConfigMigrationError(
                "tomlkit is required for TOML output. Install with: pip install tomlkit"
            )

        try:
            toml_content = tomlkit.dumps(new_config)

            # Add migration header
            migration_header = dedent(f"""\
            # Code Weaver Configuration (Migrated from {input_path})
            # Migration Date: {os.environ.get("USER", "system")}@{os.uname().nodename if hasattr(os, "uname") else "unknown"}
            # Original Format: {config_format}
            # New Format: v2.0

            """)

            full_content = migration_header + toml_content

            # Write to output file if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(full_content)
                logger.info("Migrated configuration written to: %s", output_path)

        except Exception as e:
            raise ConfigMigrationError("Failed to generate TOML output") from e

        else:
            return full_content


def validate_configuration_file(config_path: str | Path) -> tuple[CodeWeaverConfig, list[str]]:
    """Validate a configuration file and return config with warnings.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (configuration object, list of validation warnings)

    Raises:
        ConfigMigrationError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigMigrationError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)
    except Exception as e:
        raise ConfigMigrationError("Failed to load configuration file.") from e

    # Create configuration object
    config = CodeWeaverConfig()
    config.merge_from_dict(config_data)
    config.merge_from_env()

    # Validate configuration
    warnings = ConfigValidator.validate_full_config(config)

    return config, warnings


def generate_deployment_configs() -> dict[str, str]:
    """Generate example configurations for different deployment scenarios.

    Returns:
        Dictionary mapping scenario names to TOML configuration strings
    """
    return {
        "local_development": """# Local Development Configuration
[backend]
provider = "qdrant"
url = "http://localhost:6333"
collection_name = "code-embeddings-dev"

[provider]
embedding_provider = "sentence-transformers"
embedding_model = "all-MiniLM-L6-v2"
use_local = true
device = "cpu"

[data_sources]
enabled = true

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "."
use_gitignore = true
""",
        "production_cloud": """# Production Cloud Configuration
[backend]
provider = "qdrant"
url = "https://your-cluster.qdrant.io"
api_key = "your-qdrant-api-key"
collection_name = "code-embeddings"
enable_hybrid_search = true
enable_sparse_vectors = true
batch_size = 100

[provider]
embedding_provider = "voyage"
embedding_api_key = "your-voyage-api-key"
embedding_model = "voyage-code-3"
rerank_provider = "voyage"
rerank_model = "voyage-rerank-2"

[data_sources]
enabled = true
enable_content_deduplication = true

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "/app/code"
use_gitignore = true
""",
        "enterprise_multi_source": """# Enterprise Multi-Source Configuration
[backend]
provider = "qdrant"
url = "https://enterprise-cluster.qdrant.io"
api_key = "enterprise-api-key"
collection_name = "enterprise-code-embeddings"
enable_hybrid_search = true
enable_failover = true
replica_urls = ["https://replica1.qdrant.io", "https://replica2.qdrant.io"]

[provider]
embedding_provider = "voyage"
embedding_api_key = "voyage-enterprise-key"
embedding_model = "voyage-code-3"
rerank_provider = "voyage"
enable_caching = true

[data_sources]
enabled = true
max_concurrent_sources = 10
enable_content_deduplication = true
enable_metadata_extraction = true

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "/enterprise/code"
patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.go"]

[[data_sources.sources]]
type = "git"
enabled = true
priority = 2
source_id = "shared_libraries"

[data_sources.sources.config]
repository_url = "https://github.com/enterprise/shared-libs.git"
branch = "main"
local_clone_path = "/tmp/shared-libs"
""",
        "pinecone_setup": """# Pinecone Configuration Example
[backend]
provider = "pinecone"
api_key = "your-pinecone-api-key"
collection_name = "code-embeddings"

[backend.provider_options]
environment = "us-west1-gcp"
index_name = "code-embeddings"

[provider]
embedding_provider = "openai"
embedding_api_key = "your-openai-api-key"
embedding_model = "text-embedding-3-large"
embedding_dimension = 3072
""",
        "weaviate_hybrid": """# Weaviate Hybrid Search Configuration
[backend]
provider = "weaviate"
url = "https://your-cluster.weaviate.network"
api_key = "your-weaviate-api-key"
enable_hybrid_search = true

[provider]
embedding_provider = "cohere"
embedding_api_key = "your-cohere-api-key"
embedding_model = "embed-english-v3.0"
rerank_provider = "cohere"
rerank_model = "rerank-english-v3.0"
""",
    }


# Convenience functions for common operations
def migrate_config_file(input_path: str | Path, output_path: str | Path | None = None) -> str:
    """Migrate a configuration file from legacy to new format."""
    return ConfigMigrator.create_migration_script(input_path, output_path, fmt="toml")


def validate_config_file(config_path: str | Path) -> list[str]:
    """Validate a configuration file and return warnings."""
    _, warnings = validate_configuration_file(config_path)
    return warnings


def check_backend_compatibility(backend: str, provider: str) -> bool:
    """Check if backend and provider are compatible."""
    warnings = ConfigValidator.validate_backend_provider_combination(backend, provider)
    return len(warnings) == 0
