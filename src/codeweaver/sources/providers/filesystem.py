# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
File system data source implementation for CodeWeaver.

Refactors the existing file system indexing logic into the new data source
abstraction while maintaining full backward compatibility.
"""

import asyncio
import contextlib
import logging

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from codeweaver.cw_types import (
    CodeChunk,
    ContentItem,
    ContentType,
    SourceCapabilities,
    SourceProvider,
)
from codeweaver.sources.base import AbstractDataSource, SourceConfig, SourceWatcher


logger = logging.getLogger(__name__)


class FileSystemSourceConfig(SourceConfig):
    """Configuration specific to file system data sources."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    root_path: str = Field(".", description="Root directory path")
    follow_symlinks: bool = Field(False, description="Follow symbolic links")
    use_gitignore: bool = Field(True, description="Respect .gitignore files")
    additional_ignore_patterns: list[str] = Field(
        default_factory=list, description="Additional ignore patterns"
    )
    recursive_discovery: bool = Field(True, description="Recursive directory discovery")
    file_extensions: list[str] = Field(
        default_factory=list, description="Specific file extensions to include"
    )

    @field_validator("root_path")
    @classmethod
    def validate_root_path(cls, v: str) -> str:
        """Validate that root path exists and is a directory."""
        from pathlib import Path

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Root path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Root path is not a directory: {v}")
        return str(path.resolve())


class FileSystemSourceWatcher(SourceWatcher):
    """File system specific watcher implementation."""

    def __init__(
        self,
        source_id: str,
        callback: Callable[[list[ContentItem]], None],
        root_path: Path,
        config: FileSystemSourceConfig,
    ):
        """Initialize file system watcher.

        Args:
            source_id: Source identifier
            callback: Change notification callback
            root_path: Root directory to watch
            config: File system source configuration
        """
        super().__init__(source_id, callback)
        self.root_path = root_path
        self.config = config
        self._watch_task: asyncio.Task | None = None
        self._last_scan_time = datetime.now(UTC)

    async def start(self) -> bool:
        """Start watching for file system changes."""
        if self.is_active:
            return True
        try:
            self.is_active = True
            self._watch_task = asyncio.create_task(self._watch_loop())
            logger.info("Started file system watching: %s", self.root_path)
        except Exception:
            logger.exception("Failed to start file system watching")
            self.is_active = False
            return False
        else:
            return True

    async def stop(self) -> bool:
        """Stop watching for file system changes."""
        if not self.is_active:
            return True
        try:
            self.is_active = False
            if self._watch_task:
                self._watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._watch_task
                self._watch_task = None
            logger.info("Stopped file system watching: %s", self.root_path)
        except Exception:
            logger.exception("Error stopping file system watching")
            return False
        else:
            return True

    async def _watch_loop(self) -> None:
        """Main watching loop for detecting file changes."""
        interval = getattr(self.config, "change_check_interval_seconds", 60)
        while self.is_active:
            try:
                await asyncio.sleep(interval)
                if not self.is_active:
                    break
                changed_items = await self._detect_changes()
                if changed_items:
                    await self.notify_changes(changed_items)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in file system watch loop")
                await asyncio.sleep(interval)

    async def _detect_changes(self) -> list[ContentItem]:
        """Detect files that have changed since last scan."""
        changed_items = []
        current_time = datetime.now(UTC)
        try:
            for file_path in self.root_path.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    stat = file_path.stat()
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    if modified_time > self._last_scan_time and (
                        item := self._path_to_content_item(file_path, stat)
                    ):
                        changed_items.append(item)
                except OSError:
                    continue
            self._last_scan_time = current_time
            if changed_items:
                logger.info("Detected %d changed files in %s", len(changed_items), self.root_path)
        except Exception:
            logger.exception("Error detecting file changes")
        return changed_items

    def _path_to_content_item(self, file_path: Path, stat: Any = None) -> ContentItem | None:
        """Convert a file path to a ContentItem."""
        try:
            if stat is None:
                stat = file_path.stat()
            language = self._detect_language(file_path)
        except Exception:
            logger.debug("Failed to create ContentItem for %s", file_path)
            return None
        else:
            return ContentItem(
                path=str(file_path),
                content_type=ContentType.FILE,
                metadata={
                    "file_extension": file_path.suffix,
                    "relative_path": str(file_path.relative_to(self.root_path)),
                    "parent_directory": str(file_path.parent),
                },
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                language=language,
                source_id=self.source_id,
                checksum=None,
            )

    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".ps1": "powershell",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".md": "markdown",
            ".rst": "rst",
            ".tex": "latex",
            ".sql": "sql",
            ".r": "r",
            ".R": "r",
            ".m": "matlab",
            ".pl": "perl",
            ".lua": "lua",
            ".vim": "vim",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile",
        }
        return language_map.get(suffix)


class FileSystemSource(AbstractDataSource):
    """File system data source implementation.

    Provides content discovery from local file systems with intelligent
    file filtering, gitignore support, and change watching capabilities.
    """

    def __init__(self, source_id: str | None = None):
        """Initialize file system data source.

        Args:
            source_id: Unique identifier for this source instance
        """
        super().__init__(SourceProvider.FILESYSTEM, source_id)

    CAPABILITIES = SourceCapabilities(
        supports_content_discovery=True,
        supports_content_reading=True,
        supports_change_watching=True,
        supports_metadata_extraction=True,
        supports_batch_processing=True,
        supports_content_deduplication=True,
        max_content_size_mb=100,
        max_concurrent_requests=8,
        required_dependencies=[],
        optional_dependencies=["watchdog"],
    )

    def get_capabilities(self) -> SourceCapabilities:
        """Get capabilities - no hardcoding needed."""
        return self.CAPABILITIES

    async def discover_content(
        self, config: FileSystemSourceConfig, context: dict[str, Any] | None = None
    ) -> list[ContentItem]:
        """Discover files from the file system.

        Args:
            config: File system source configuration
            context: Optional context containing services

        Returns:
            List of discovered file content items
        """
        if not getattr(config, "enabled", True):
            return []
        root = config.root_path
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")
        logger.info("Discovering content in: %s", root_path)
        if context and "filtering_service" in context:
            self._filtering_service = context["filtering_service"]
        files = await self._discover_files(root_path, config)
        content_items = []
        for file_path in files:
            item = await self._file_to_content_item(file_path, root_path)
            if item:
                content_items.append(item)
        filtered_items = self._apply_content_filters(content_items, config)
        logger.info(
            "File system discovery complete: %d files found, %d after filtering",
            len(content_items),
            len(filtered_items),
        )
        return filtered_items

    async def index_content(
        self, path: Path, context: dict[str, Any] | None = None
    ) -> list[CodeChunk]:
        """Index content using middleware services from context.

        Args:
            path: Path to index (file or directory)
            context: Optional context containing middleware services

        Returns:
            List of CodeChunk objects representing indexed content
        """
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        logger.info("Indexing content: %s", path)
        filtering_service = context.get("filtering_service") if context else None
        chunking_service = context.get("chunking_service") if context else None
        if filtering_service:
            files = await filtering_service.find_files(path)
            logger.info("Found %d files using filtering service", len(files))
        else:
            files = await self._fallback_file_discovery(path)
            logger.info("Found %d files using fallback discovery", len(files))
        all_chunks = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    logger.debug("Skipping empty file: %s", file_path)
                    continue
                if chunking_service:
                    chunks = await chunking_service.chunk_file(file_path, content)
                else:
                    chunks = await self._fallback_chunking(file_path, content)
                all_chunks.extend(chunks)
                logger.debug("Processed %s: %d chunks", file_path.name, len(chunks))
            except Exception as e:
                logger.warning("Failed to process file %s: %s", file_path, e)
                continue
        logger.info("Indexing complete: %d chunks from %d files", len(all_chunks), len(files))
        return all_chunks

    async def _fallback_file_discovery(self, base_path: Path) -> list[Path]:
        """Fallback file discovery when filtering service is not available."""
        config = FileSystemSourceConfig(root_path=str(base_path))
        return await self._discover_files(base_path, config)

    async def _fallback_chunking(self, file_path: Path, content: str) -> list[CodeChunk]:
        """Fallback chunking when chunking service is not available."""
        from codeweaver.middleware.chunking import ChunkingMiddleware

        config = {"max_chunk_size": 1500, "min_chunk_size": 50, "ast_grep_enabled": True}
        chunker = ChunkingMiddleware(config)
        legacy_chunks = await chunker.chunk_file(file_path, content)
        pydantic_chunks = []
        for legacy_chunk in legacy_chunks:
            pydantic_chunk = CodeChunk.create_with_hash(
                content=legacy_chunk.content,
                file_path=legacy_chunk.file_path,
                start_line=legacy_chunk.start_line,
                end_line=legacy_chunk.end_line,
                chunk_type=legacy_chunk.chunk_type,
                language=legacy_chunk.language,
                node_kind=legacy_chunk.node_kind,
                metadata={"fallback_chunking": True},
            )
            pydantic_chunks.append(pydantic_chunk)
        return pydantic_chunks

    async def structural_search(
        self,
        pattern: str,
        language: str | None = None,
        root_path: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform structural search using ast-grep patterns.

        Args:
            pattern: AST-grep pattern to search for
            language: Programming language to search in (optional)
            root_path: Root path to search in (optional, defaults to configured root)
            context: Optional context containing middleware services

        Returns:
            List of search result dictionaries
        """
        try:
            from ast_grep_py import SgRoot
        except ImportError as e:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py") from e
        if root_path is None:
            root_path = Path.cwd()
        elif isinstance(root_path, str):
            root_path = Path(root_path)
        if not root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        logger.info(
            "Performing structural search: pattern='%s', language=%s, root=%s",
            pattern,
            language,
            root_path,
        )
        filtering_service = context.get("filtering_service") if context else None
        if filtering_service:
            file_patterns = self._get_file_patterns_for_language(language)
            files = await filtering_service.find_files(root_path, file_patterns)
        else:
            files = await self._discover_files_for_language(root_path, language)
        logger.debug("Found %d files for structural search", len(files))
        results = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                detected_language = language or self._detect_language_from_extension(file_path)
                if not detected_language:
                    continue
                sg_root = SgRoot(content, detected_language)
                tree = sg_root.root()
                matches = tree.find_all(pattern)
                for match in matches:
                    range_info = match.range()
                    results.append({
                        "file_path": str(file_path),
                        "match_content": match.text(),
                        "start_line": range_info.start.line + 1,
                        "end_line": range_info.end.line + 1,
                        "start_column": range_info.start.column + 1,
                        "end_column": range_info.end.column + 1,
                        "language": detected_language,
                        "pattern": pattern,
                    })
            except Exception as e:
                logger.warning("Error searching %s: %s", file_path, e)
                continue
        logger.info("Structural search complete: %d matches found", len(results))
        return results

    def _get_file_patterns_for_language(self, language: str | None = None) -> list[str]:
        """Get file patterns for a specific language."""
        if not language:
            return ["*"]
        lang_extensions = {
            "python": ["*.py", "*.py3", "*.pyi"],
            "javascript": ["*.js", "*.mjs", "*.cjs"],
            "typescript": ["*.ts", "*.cts", "*.mts"],
            "tsx": ["*.tsx"],
            "rust": ["*.rs"],
            "go": ["*.go"],
            "java": ["*.java"],
            "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh"],
            "c": ["*.c", "*.h"],
            "csharp": ["*.cs", "*.csx"],
            "css": ["*.css", "*.scss"],
            "html": ["*.html", "*.htm", "*.xhtml"],
            "json": ["*.json"],
            "yaml": ["*.yaml", "*.yml"],
            "bash": ["*.sh", "*.bash"],
            "ruby": ["*.rb", "*.gemspec"],
            "php": ["*.php"],
            "swift": ["*.swift"],
            "kotlin": ["*.kt", "*.ktm", "*.kts"],
            "scala": ["*.scala", "*.sbt", "*.sc"],
            "elixir": ["*.ex", "*.exs"],
            "lua": ["*.lua"],
        }
        return lang_extensions.get(language.lower(), [f"*.{language}"])

    async def _discover_files_for_language(
        self, root_path: Path, language: str | None = None
    ) -> list[Path]:
        """Discover files for a specific language using fallback method."""
        patterns = self._get_file_patterns_for_language(language)
        files = []
        try:
            import rignore

            for entry in rignore.walk(str(root_path)):
                if entry.is_file():
                    file_path = Path(entry.path)
                    import fnmatch

                    for pattern in patterns:
                        if fnmatch.fnmatch(file_path.name, pattern):
                            files.append(file_path)
                            break
        except ImportError:
            for pattern in patterns:
                files.extend(root_path.rglob(pattern))
        return files

    def _detect_language_from_extension(self, file_path: Path) -> str | None:
        """Detect language from file extension for structural search."""
        extension = file_path.suffix.lower().lstrip(".")
        ext_lang_map = {
            "py": "python",
            "py3": "python",
            "pyi": "python",
            "js": "javascript",
            "mjs": "javascript",
            "cjs": "javascript",
            "ts": "typescript",
            "cts": "typescript",
            "mts": "typescript",
            "tsx": "tsx",
            "rs": "rust",
            "go": "go",
            "java": "java",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "hpp": "cpp",
            "hh": "cpp",
            "c": "c",
            "h": "c",
            "cs": "csharp",
            "csx": "csharp",
            "css": "css",
            "scss": "css",
            "html": "html",
            "htm": "html",
            "xhtml": "html",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "sh": "bash",
            "bash": "bash",
            "rb": "ruby",
            "gemspec": "ruby",
            "php": "php",
            "swift": "swift",
            "kt": "kotlin",
            "ktm": "kotlin",
            "kts": "kotlin",
            "scala": "scala",
            "sbt": "scala",
            "sc": "scala",
            "ex": "elixir",
            "exs": "elixir",
            "lua": "lua",
        }
        return ext_lang_map.get(extension)

    async def read_content(self, item: ContentItem) -> str:
        """Read content from a file.

        Args:
            item: Content item representing a file

        Returns:
            Text content of the file
        """
        if item.content_type != ContentType.FILE:
            raise ValueError(
                f"Unsupported content type for file system source: {item.content_type}"
            )
        file_path = Path(item.path)
        if not file_path.exists():
            raise FileNotFoundError(f"File no longer exists: {file_path}")
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading file: {file_path}") from e
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}") from e

    async def watch_changes(
        self, config: FileSystemSourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Set up file system change watching.

        Args:
            config: File system source configuration
            callback: Function to call when changes are detected

        Returns:
            File system watcher instance
        """
        if not getattr(config, "enable_change_watching", False):
            raise NotImplementedError("Change watching is disabled in configuration")
        root = config.root_path
        root_path = Path(root).resolve()
        watcher = FileSystemSourceWatcher(
            source_id=self.source_id, callback=callback, root_path=root_path, config=config
        )
        self._watchers.append(watcher)
        return watcher

    async def validate_source(self, config: FileSystemSourceConfig) -> bool:
        """Validate file system source configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not await super().validate_source(config):
                return False
            root = config.root_path
            if not root:
                logger.warning("Missing root_path in file system source configuration")
                return False
            root_path = Path(root)
            if not root_path.exists():
                logger.warning("Root path does not exist: %s", root_path)
                return False
            if not root_path.is_dir():
                logger.warning("Root path is not a directory: %s", root_path)
                return False
            try:
                list(root_path.iterdir())
            except PermissionError:
                logger.warning("Permission denied accessing root path: %s", root_path)
                return False
        except Exception:
            logger.exception("Error validating file system source configuration")
            return False
        else:
            logger.info("File system source configuration is valid: %s", root_path)
            return True

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get detailed metadata for a file.

        Args:
            item: Content item representing a file

        Returns:
            Dictionary with detailed file metadata
        """
        metadata = await super().get_content_metadata(item)
        try:
            file_path = Path(item.path)
            if file_path.exists():
                stat = file_path.stat()
                metadata.update({
                    "file_size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "accessed_at": datetime.fromtimestamp(stat.st_atime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                    "is_symlink": file_path.is_symlink(),
                    "absolute_path": str(file_path.resolve()),
                })
                if (language := self._detect_language(file_path)) and (not item.language):
                    metadata["detected_language"] = language
        except Exception:
            logger.debug("Failed to get extended metadata for %s", item.path)
        return metadata

    async def _discover_files(self, root_path: Path, config: FileSystemSourceConfig) -> list[Path]:
        """Discover files using services-based filtering.

        Args:
            root_path: Root directory to search
            config: Source configuration

        Returns:
            List of discovered file paths
        """
        if filtering_service := getattr(self, "_filtering_service", None):
            logger.info("Using FilteringService for file discovery")
            return await self._discover_files_with_service(root_path, config, filtering_service)
        logger.info("FilteringService not available, using fallback discovery")
        return await self._discover_files_fallback(root_path, config)

    async def _discover_files_with_service(
        self, root_path: Path, config: FileSystemSourceConfig, filtering_service
    ) -> list[Path]:
        """Discover files using the FilteringService."""
        try:
            include_patterns = []
            if config.file_extensions:
                include_patterns.extend([f"*.{ext.lstrip('.')}" for ext in config.file_extensions])
            if not include_patterns:
                include_patterns = ["*"]
            exclude_patterns = config.additional_ignore_patterns or []
            follow_symlinks = config.follow_symlinks
            files = await filtering_service.discover_files(
                base_path=root_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                follow_symlinks=follow_symlinks,
            )
        except Exception as e:
            logger.warning("FilteringService failed: %s, falling back to manual discovery", e)
            return await self._discover_files_fallback(root_path, config)
        else:
            logger.info("Discovered %d files with FilteringService", len(files))
            return files

    async def _discover_files_fallback(
        self, root_path: Path, config: FileSystemSourceConfig
    ) -> list[Path]:
        """Fallback file discovery when FilteringService is not available."""
        from codeweaver.middleware.filtering import FileFilteringMiddleware

        middleware_config = {
            "use_gitignore": config.use_gitignore,
            "max_file_size": 10 * 1024 * 1024,
            "excluded_dirs": ["__pycache__", ".git", "node_modules", ".pytest_cache"],
            "included_extensions": config.file_extensions or None,
            "additional_ignore_patterns": config.additional_ignore_patterns,
        }
        middleware = FileFilteringMiddleware(middleware_config)
        patterns = config.file_extensions or ["*"]
        files = await middleware.find_files(
            base_path=root_path, patterns=patterns, recursive=config.recursive_discovery
        )
        logger.info("Discovered %d files with fallback discovery", len(files))
        return files

    async def _file_to_content_item(self, file_path: Path, root_path: Path) -> ContentItem | None:
        """Convert a file path to a ContentItem.

        Args:
            file_path: Path to the file
            root_path: Root directory for relative path calculation

        Returns:
            ContentItem or None if file cannot be processed
        """
        try:
            stat = file_path.stat()
            try:
                relative_path = file_path.relative_to(root_path)
            except ValueError:
                relative_path = file_path
            language = self._detect_language(file_path)
        except Exception:
            logger.debug("Failed to create ContentItem for %s", file_path)
            return None
        else:
            return ContentItem(
                path=str(file_path),
                content_type=ContentType.FILE,
                metadata={
                    "file_extension": file_path.suffix,
                    "relative_path": str(relative_path),
                    "parent_directory": str(file_path.parent),
                    "root_path": str(root_path),
                },
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                language=language,
                source_id=self.source_id,
            )

    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".ps1": "powershell",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".md": "markdown",
            ".rst": "rst",
            ".tex": "latex",
            ".sql": "sql",
            ".r": "r",
            ".R": "r",
            ".m": "matlab",
            ".pl": "perl",
            ".lua": "lua",
            ".vim": "vim",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile",
        }
        return language_map.get(suffix)

    async def health_check(self) -> bool:
        """Check filesystem data source health by verifying root path accessibility.

        Returns:
            True if source is healthy and operational, False otherwise
        """
        try:
            if not hasattr(self, "source_id") or not self.source_id:
                logger.warning("Filesystem source missing source_id")
                return False
            test_path = Path()
            if test_path.exists() and test_path.is_dir():
                list(test_path.iterdir())
                logger.debug("Filesystem source health check passed")
                return True
            logger.warning("Filesystem source cannot access test directory")
        except Exception:
            logger.exception("Filesystem source health check failed")
            return False
        else:
            return False
