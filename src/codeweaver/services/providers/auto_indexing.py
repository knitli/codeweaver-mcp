# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Auto-indexing background service provider."""

import asyncio
import logging

from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from codeweaver.cw_types import (
    AutoIndexingConfig,
    ChunkingService,
    ContentItem,
    FilteringService,
    HealthStatus,
    ServiceHealth,
    ServiceIntegrationError,
    ServiceType,
)
from codeweaver.services.providers.base_provider import BaseServiceProvider


class CodebaseChangeHandler(FileSystemEventHandler):
    """File system event handler for codebase changes."""

    def __init__(self, auto_indexing_service: "AutoIndexingService"):
        """Initialize with reference to the auto-indexing service."""
        super().__init__()
        self.service = auto_indexing_service
        self._logger = auto_indexing_service._logger
        self._debounce_tasks: dict[str, asyncio.Task] = {}
        self._background_tasks: set[asyncio.Task] = set()

    def on_modified(self, event: FileSystemEventHandler) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self.create_and_store_task(file_path)

    def on_created(self, event: FileSystemEventHandler) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self.create_and_store_task(file_path)

    def create_and_store_task(self, file_path: FileSystemEventHandler) -> None:
        """Create and store a debounced indexing task."""
        task = asyncio.create_task(self._debounced_index_file(file_path))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def on_deleted(self, event: FileSystemEventHandler) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            task = asyncio.create_task(self._remove_from_index(file_path))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed using FilteringService if available."""
        # Use FilteringService if available for consistent filtering logic
        if self.service.filtering_service:
            return self.service.filtering_service.should_include_file(
                file_path,
                include_patterns=self.service._auto_indexing_config.watch_patterns,
                exclude_patterns=self.service._auto_indexing_config.ignore_patterns
            )
        
        # Fallback to custom logic if FilteringService is not available
        return self._fallback_filtering(file_path)
    
    def _fallback_filtering(self, file_path: Path) -> bool:
        """Fallback filtering logic when FilteringService is unavailable."""
        raw_path = str(file_path)
        for pattern in self.service._auto_indexing_config.ignore_patterns:
            if pattern in raw_path:
                return False
        return any(
            file_path.match(pattern)
            for pattern in self.service._auto_indexing_config.watch_patterns
        )

    async def _debounced_index_file(self, file_path: Path):
        """Index file with debouncing to avoid excessive processing."""
        file_key = str(file_path)
        if file_key in self._debounce_tasks:
            self._debounce_tasks[file_key].cancel()

        async def _delayed_index():
            await asyncio.sleep(self.service._auto_indexing_config.debounce_delay)
            await self.service._index_single_file(file_path)

        self._debounce_tasks[file_key] = asyncio.create_task(_delayed_index())

    async def _remove_from_index(self, file_path: Path):
        """Remove file from index."""
        try:
            await self.service._remove_file_from_index(file_path)
        except Exception as e:
            self._logger.warning("Failed to remove file from index %s: %s", file_path, e)


class AutoIndexingService(BaseServiceProvider):
    """
    Background indexing service - NEVER exposed to LLM users.

    This service operates transparently in the background to keep the
    codebase indexed and searchable. It:

    1. Monitors filesystem changes using watchdog
    2. Uses existing chunking and filtering services
    3. Integrates with vector backends for storage
    4. Provides health monitoring and error recovery
    5. Is controlled by framework developers, not LLM users

    Key Design Principles:
    - NO INDEX intent exposed to LLMs
    - Background operation only
    - Uses existing service dependencies
    - Follows BaseServiceProvider patterns
    - Framework developer control only
    """

    def __init__(self, config: AutoIndexingConfig, logger: logging.Logger | None = None):
        """Initialize the auto-indexing service."""
        super().__init__(ServiceType.AUTO_INDEXING, config, logger)
        self._auto_indexing_config = config
        self.observer: Observer | None = None
        self.watched_paths: set[str] = set()
        self.chunking_service: ChunkingService | None = None
        self.filtering_service: FilteringService | None = None
        self.backend_registry = None
        self._indexing_queue: asyncio.Queue = asyncio.Queue(maxsize=config.indexing_queue_size)
        self._indexing_workers: list[asyncio.Task] = []
        self._indexing_stats = {
            "files_indexed": 0,
            "files_failed": 0,
            "total_chunks_created": 0,
            "last_indexing_time": None,
        }

    async def _initialize_provider(self) -> None:
        """Initialize with existing service dependencies."""
        self._logger.info("Initializing auto-indexing service with service dependencies")
        try:
            self.chunking_service = await self._get_chunking_service()
            self.filtering_service = await self._get_filtering_service()
            self.backend_registry = await self._get_backend_registry()
            self.observer = Observer()
            await self._start_indexing_workers()
            self._logger.info("Auto-indexing service initialized successfully")
        except Exception as e:
            self._logger.exception("Failed to initialize auto-indexing service")
            raise ServiceIntegrationError(
                f"Auto-indexing service initialization failed: {e}"
            ) from e

    async def _shutdown_provider(self) -> None:
        """Shutdown provider-specific resources."""
        self._logger.info("Shutting down auto-indexing service")
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        for worker in self._indexing_workers:
            worker.cancel()
        if self._indexing_workers:
            await asyncio.gather(*self._indexing_workers, return_exceptions=True)
        self._logger.info(
            "Auto-indexing statistics: %d files indexed, %d failed, %d chunks created",
            self._indexing_stats["files_indexed"],
            self._indexing_stats["files_failed"],
            self._indexing_stats["total_chunks_created"],
        )
        self.watched_paths.clear()
        self.observer = None

    async def _check_health(self) -> bool:
        """Perform provider-specific health check."""
        try:
            if not self.chunking_service:
                self._logger.warning("Chunking service not available")
                return False
            if not self.filtering_service:
                self._logger.warning("Filtering service not available")
                return False
            if self.watched_paths and (not (self.observer and self.observer.is_alive())):
                self._logger.warning("File observer not running but paths are being watched")
                return False
            chunking_health = await self.chunking_service.health_check()
            filtering_health = await self.filtering_service.health_check()
            if chunking_health.status == HealthStatus.UNHEALTHY:
                self._logger.warning("Chunking service is unhealthy")
                return False
            if filtering_health.status == HealthStatus.UNHEALTHY:
                self._logger.warning("Filtering service is unhealthy")
                return False
        except Exception as e:
            self._logger.warning("Health check failed: %s", e)
            return False
        else:
            return True

    async def start_monitoring(self, path: str) -> None:
        """
        Start background monitoring - exposed to framework developers only.

        This method is NOT exposed to LLM users. It's only available to
        framework developers who control the indexing service.

        Args:
            path: Directory path to monitor for changes
        """
        if path in self.watched_paths:
            self._logger.info("Path already being monitored: %s", path)
            return
        self._logger.info("Starting monitoring for path: %s", path)
        try:
            if self._auto_indexing_config.initial_scan_enabled:
                await self._index_path_initial(path)
            event_handler = CodebaseChangeHandler(self)
            self.observer.schedule(
                event_handler, path, recursive=self._auto_indexing_config.recursive_monitoring
            )
            if not self.observer.is_alive():
                self.observer.start()
            self.watched_paths.add(path)
            self._logger.info("Started monitoring path: %s", path)
        except Exception as e:
            self._logger.exception("Failed to start monitoring path %s.", path)
            raise ServiceIntegrationError(f"Failed to start monitoring {path}: {e}") from e

    async def stop_monitoring(self, path: str | None = None) -> None:
        """
        Stop monitoring - exposed to framework developers only.

        Args:
            path: Specific path to stop monitoring, or None to stop all
        """
        if path:
            self.watched_paths.discard(path)
            self._logger.info("Stopped monitoring path: %s", path)
        else:
            self.watched_paths.clear()
            self._logger.info("Stopped monitoring all paths")
        if not self.watched_paths and self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

    async def trigger_indexing(self, path: str) -> bool:
        """
        Trigger background indexing for a given path.

        This method can be called from the intent bridge to start background
        indexing operations without exposing indexing to LLM users.

        Args:
            path: Path to index

        Returns:
            True if indexing was triggered successfully
        """
        try:
            self._logger.info("Triggering background indexing for path: %s", path)
            await self.start_monitoring(path)
        except Exception:
            self._logger.exception("Failed to trigger background indexing for %s", path)
            return False
        else:
            return True

    async def _index_path_initial(self, path: str) -> None:
        """Perform initial indexing of a path using existing services."""
        if not all([self.filtering_service, self.chunking_service]):
            self._logger.warning("Required services not available for indexing")
            return
        self._logger.info("Starting initial indexing of path: %s", path)
        try:
            # Use FilteringService for consistent file discovery
            files = await self.filtering_service.discover_files(
                Path(path),
                include_patterns=self._auto_indexing_config.watch_patterns,
                exclude_patterns=self._auto_indexing_config.ignore_patterns
            )
            self._logger.info("Found %d files to index in %s", len(files), path)
            for file_path in files:
                try:
                    await self._indexing_queue.put(file_path)
                except asyncio.QueueFull:
                    self._logger.warning("Indexing queue full, skipping file: %s", file_path)
                    break
        except Exception:
            self._logger.exception("Failed to perform initial indexing of %s", path)

    async def _index_single_file(self, file_path: Path) -> None:
        """Index a single file."""
        try:
            await self._indexing_queue.put(file_path)
        except asyncio.QueueFull:
            self._logger.warning("Indexing queue full, skipping file: %s", file_path)

    async def _remove_file_from_index(self, file_path: Path) -> None:
        """Remove a file from the index."""
        try:
            self._logger.debug("File removed from index: %s", file_path)
        except Exception as e:
            self._logger.warning("Failed to remove file from index %s: %s", file_path, e)

    async def _start_indexing_workers(self) -> None:
        """Start background indexing workers."""
        num_workers = self._auto_indexing_config.max_concurrent_indexing
        for i in range(num_workers):
            worker = asyncio.create_task(self._indexing_worker(f"worker-{i}"))
            self._indexing_workers.append(worker)
        self._logger.info("Started %d indexing workers", num_workers)

    async def _indexing_worker(self, worker_name: str) -> None:
        """Background worker for processing indexing queue."""
        self._logger.debug("Starting indexing worker: %s", worker_name)
        while True:
            try:
                file_path = await asyncio.wait_for(self._indexing_queue.get(), timeout=30.0)
                await self._process_file_for_indexing(file_path, worker_name)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                self._logger.debug("Indexing worker %s cancelled", worker_name)
                break
            except Exception:
                self._logger.exception("Indexing worker %s error.", worker_name)

    async def _process_file_for_indexing(self, file_path: Path, worker_name: str) -> None:
        """Process a single file for indexing."""
        try:
            # Use FilteringService metadata if available for enhanced file checking
            if self.filtering_service:
                try:
                    metadata = await self.filtering_service.get_file_metadata(file_path)
                    if metadata.is_binary:
                        self._logger.debug("Skipping binary file: %s", file_path)
                        return
                    if metadata.size > self._auto_indexing_config.max_file_size:
                        self._logger.debug("Skipping large file: %s (%d bytes)", file_path, metadata.size)
                        return
                except Exception as e:
                    self._logger.warning("Failed to get file metadata for %s: %s", file_path, e)
                    # Continue with fallback logic
            
            # Fallback to basic size check
            if file_path.stat().st_size > self._auto_indexing_config.max_file_size:
                self._logger.debug("Skipping large file: %s", file_path)
                return
            content = await self._read_file_content(file_path)
            if not content.strip():
                self._logger.debug("Skipping empty file: %s", file_path)
                return
            chunks = await self.chunking_service.chunk_content(content, str(file_path))
            await self._store_chunks_via_backend(file_path, chunks)
            self._indexing_stats["files_indexed"] += 1
            self._indexing_stats["total_chunks_created"] += len(chunks)
            self._indexing_stats["last_indexing_time"] = asyncio.get_event_loop().time()
            self._logger.debug(
                "Worker %s indexed file %s (%d chunks)", worker_name, file_path, len(chunks)
            )
        except Exception as e:
            self._indexing_stats["files_failed"] += 1
            self._logger.warning("Worker %s failed to index file %s: %s", worker_name, file_path, e)

    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content with error handling."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with file_path.open("r", encoding="latin1") as f:
                return f.read()

    async def _store_chunks_via_backend(self, file_path: Path, chunks: list[ContentItem]) -> None:
        """Store chunks using existing backend patterns."""
        self._logger.debug("Storing %d chunks for file: %s", len(chunks), file_path)

    async def _get_chunking_service(self) -> ChunkingService | None:
        """Get chunking service through dependency injection."""
        # Service injection should be provided by the service manager during initialization
        # This will be updated when the auto-indexing service is properly integrated
        # with the ServicesManager dependency injection system
        return self.chunking_service

    async def _get_filtering_service(self) -> FilteringService | None:
        """Get filtering service through dependency injection."""
        # Service injection should be provided by the service manager during initialization  
        # This will be updated when the auto-indexing service is properly integrated
        # with the ServicesManager dependency injection system
        return self.filtering_service

    async def _get_backend_registry(self):
        """Get backend registry through dependency injection."""
        return

    async def health_check(self) -> ServiceHealth:
        """Enhanced health check with auto-indexing specific metrics."""
        base_health = await super().health_check()
        base_health.metadata = {
            "watched_paths_count": len(self.watched_paths),
            "watched_paths": list(self.watched_paths),
            "files_indexed": self._indexing_stats["files_indexed"],
            "files_failed": self._indexing_stats["files_failed"],
            "total_chunks_created": self._indexing_stats["total_chunks_created"],
            "indexing_workers_active": len([w for w in self._indexing_workers if not w.done()]),
            "queue_size": self._indexing_queue.qsize(),
            "observer_running": bool(self.observer and self.observer.is_alive()),
            "chunking_service_available": bool(self.chunking_service),
            "filtering_service_available": bool(self.filtering_service),
        }
        return base_health

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for auto-indexing operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "auto_indexing_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {
                "auto_indexing_enabled": self._config.auto_indexing_enabled,
                "watch_patterns": self._config.watch_patterns,
                "debounce_delay": self._config.debounce_delay,
                "max_workers": self._config.max_workers,
                "batch_size": self._config.batch_size,
            },
            "watched_paths": list(self.watched_paths),
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
            "metadata": health.metadata,
        })

        # Add runtime statistics
        context.update({
            "statistics": {
                "files_indexed": self._indexing_stats["files_indexed"],
                "files_failed": self._indexing_stats["files_failed"],
                "total_chunks_created": self._indexing_stats["total_chunks_created"],
                "indexing_workers_active": len([w for w in self._indexing_workers if not w.done()]),
                "queue_size": self._indexing_queue.qsize(),
                "observer_running": bool(self.observer and self.observer.is_alive()),
            }
        })

        return context
