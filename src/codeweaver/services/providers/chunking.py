# sourcery skip: name-type-suffix
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""FastMCP chunking service provider."""

import time

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from codeweaver.cw_types import (
    ChunkingError,
    ChunkingService,
    ChunkingServiceConfig,
    ChunkingStats,
    ChunkingStrategy,
    CodeChunk,
    Language,
    ServiceCapabilities,
    ServiceType,
    UnsupportedLanguageError,
)
from codeweaver.middleware.chunking import ChunkingMiddleware
from codeweaver.services.providers.base_provider import BaseServiceProvider


class ChunkingService(BaseServiceProvider, ChunkingService):
    """FastMCP-based chunking service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: ChunkingServiceConfig):
        """Initialize chunking provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: ChunkingMiddleware | None = None
        self._stats = ChunkingStats()

        # Language mapping
        self._language_mapping = {
            k: v for k, v in Language.members_to_values().items() if k.supports_ast_grep
        }

    @property
    def capabilities(self) -> ServiceCapabilities:
        """Provider capabilities."""
        return ServiceCapabilities(
            supports_streaming=True,
            supports_batch=True,
            supports_async=True,
            max_concurrency=self._config.metadata.get("max_concurrency", 10),
            memory_usage="medium",
            performance_profile=self._config.performance_mode.value,
        )

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP chunking middleware."""
        middleware_config = {
            "max_chunk_size": self._config.max_chunk_size,
            "min_chunk_size": self._config.min_chunk_size,
            "ast_grep_enabled": self._config.ast_grep_enabled,
        }

        self._middleware = ChunkingMiddleware(middleware_config)
        self._logger.info("FastMCP chunking provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown chunking provider."""
        self._middleware = None
        self._logger.info("FastMCP chunking provider shut down")

    async def _check_health(self) -> bool:
        """Check if chunking service is healthy."""
        return self._middleware is not None

    async def chunk_content(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO,
    ) -> list[CodeChunk]:
        """Chunk content into code segments."""

        def raise_chunking_error(message: str) -> None:
            """Raise a chunking error with a specific message."""
            raise ChunkingError(file_path, message)

        if not self._middleware:
            raise ChunkingError(file_path, "Chunking service not initialized")

        start_time = time.time()

        try:
            # Detect language and validate support
            language = self.detect_language(file_path, content)

            # Apply language-specific configuration if available
            if language and language in self._config.language_configs:
                self._config.language_configs[language]
                # Could override middleware settings here if needed
            if strategy == ChunkingStrategy.AUTO or strategy not in [
                ChunkingStrategy.AST,
                ChunkingStrategy.SIMPLE,
            ]:
                chunks = await self._middleware.chunk_file(file_path, content)
            elif strategy == ChunkingStrategy.AST:
                if not self._config.ast_grep_enabled:
                    raise_chunking_error("AST chunking is not enabled in the configuration")
                chunks = await self._chunk_with_ast_strategy(content, file_path)
            else:
                chunks = await self._chunk_with_simple_strategy(content, file_path)
            # Apply post-processing if configured
            if self._config.respect_code_structure:
                chunks = self._respect_code_boundaries(chunks)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(file_path, len(chunks), processing_time, success=True)

            self.record_operation(True)

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(file_path, 0, processing_time, success=False)

            error_msg = f"Chunking failed: {e}"
            self.record_operation(False, error_msg)
            self._logger.exception("Chunking failed for %s")

            raise ChunkingError(file_path, str(e)) from e
        else:
            return chunks

    async def chunk_content_stream(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO,
    ) -> AsyncGenerator[CodeChunk]:
        """Stream chunks for large files."""
        chunks = await self.chunk_content(content, file_path, metadata, strategy)
        for chunk in chunks:
            yield chunk

    def detect_language(self, file_path: Path, content: str | None = None) -> Language | None:
        """Detect programming language."""
        if not self._middleware:
            return None

        lang_str = self._middleware._detect_language(file_path)

        return next(
            (
                lang_enum
                for lang_enum, lang_name in self._language_mapping.items()
                if lang_name == lang_str
            ),
            None,
        )

    def get_supported_languages(self) -> dict[Language, dict[str, Any]]:
        """Get supported languages and capabilities."""
        if not self._middleware:
            return {}

        capabilities = self._middleware.get_supported_languages()
        return {
            lang_enum: {
                "name": lang_name,
                "ast_supported": lang_name in capabilities.get("chunk_patterns", {}),
                "extensions": [
                    ext
                    for ext, lang in capabilities["language_extensions"].items()
                    if lang == lang_name
                ],
                "patterns": capabilities.get("chunk_patterns", {}).get(lang_name, []),
            }
            for lang_enum, lang_name in self._language_mapping.items()
            if lang_name in capabilities["supported_languages"]
        }

    def get_language_config(self, language: Language) -> dict[str, Any] | None:
        """Get configuration for a specific language."""
        if lang_name := self._language_mapping.get(language):
            return self._config.language_configs.get(
                lang_name,
                {
                    "max_chunk_size": self._config.max_chunk_size,
                    "min_chunk_size": self._config.min_chunk_size,
                    "respect_structure": self._config.respect_code_structure,
                    "preserve_comments": self._config.preserve_comments,
                    "include_imports": self._config.include_imports,
                },
            )
        return None

    def get_available_strategies(self) -> dict[ChunkingStrategy, dict[str, Any]]:
        """Get all available chunking strategies."""
        return {
            ChunkingStrategy.AUTO: {
                "description": "Automatic strategy selection based on file type",
                "supports_ast": self._config.ast_grep_enabled,
                "fallback": "simple",
            },
            ChunkingStrategy.AST: {
                "description": "AST-based semantic chunking",
                "enabled": self._config.ast_grep_enabled,
                "requires": "ast-grep",
            },
            ChunkingStrategy.SIMPLE: {
                "description": "Line-based chunking with size limits",
                "enabled": True,
                "fallback": True,
            },
        }

    def validate_chunk_size(self, size: int, language: Language = None) -> bool:
        """Validate if a chunk size is appropriate."""
        if size < self._config.min_chunk_size:
            return False
        if size > self._config.max_chunk_size:
            return False

        # Language-specific validation
        if language and (lang_config := self.get_language_config(language)):
            lang_min = lang_config.get("min_chunk_size", self._config.min_chunk_size)
            lang_max = lang_config.get("max_chunk_size", self._config.max_chunk_size)
            return lang_min <= size <= lang_max

        return True

    async def get_chunking_stats(self) -> ChunkingStats:
        """Get chunking performance statistics."""
        return self._stats

    async def reset_stats(self) -> None:
        """Reset chunking statistics."""
        self._stats = ChunkingStats()
        self._logger.info("Chunking statistics reset")

    async def _chunk_with_ast_strategy(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Force AST-based chunking."""
        if not self._middleware:
            raise ChunkingError(file_path, "Middleware not available")

        language = self._middleware._detect_language(file_path)
        if language not in self._middleware.CHUNK_PATTERNS:
            raise UnsupportedLanguageError(file_path, language)

        return await self._middleware._chunk_with_ast_grep(content, language, file_path)

    async def _chunk_with_simple_strategy(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Force simple line-based chunking."""
        if not self._middleware:
            raise ChunkingError(file_path, "Middleware not available")

        language = self._middleware._detect_language(file_path)
        return await self._middleware._chunk_with_fallback(content, file_path, language)

    def _respect_code_boundaries(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Apply code structure respect rules."""
        # TODO: Implement logic to respect function/class boundaries
        # For now, return chunks as-is
        return chunks

    def _update_stats(
        self, file_path: Path, chunk_count: int, processing_time: float, *, success: bool
    ) -> None:
        """Update chunking statistics."""
        if success:
            self._stats.total_files_processed += 1
            self._stats.total_chunks_created += chunk_count
            self._stats.total_processing_time += processing_time

            # Update average chunk size
            if chunk_count > 0:
                current_avg = self._stats.average_chunk_size
                total_chunks = self._stats.total_chunks_created
                # Weighted average update
                self._stats.average_chunk_size = (
                    current_avg * (total_chunks - chunk_count)
                    + (chunk_count * self._config.max_chunk_size / 2)
                ) / total_chunks

            if lang := self.detect_language(file_path):
                lang_name = self._language_mapping.get(lang, "unknown")
                self._stats.languages_processed[lang_name] = (
                    self._stats.languages_processed.get(lang_name, 0) + 1
                )
        else:
            self._stats.error_count += 1

        # Update success rate
        total_operations = self._stats.total_files_processed + self._stats.error_count
        if total_operations > 0:
            self._stats.success_rate = self._stats.total_files_processed / total_operations
