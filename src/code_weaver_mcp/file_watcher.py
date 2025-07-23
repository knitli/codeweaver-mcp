# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
File watching functionality with automatic re-indexing.

Provides intelligent file monitoring using watchdog to automatically 
re-index code changes with debouncing and filtering capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Set
from datetime import datetime, timedelta

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("watchdog not available, auto-reindexing disabled")

from .config import CodeWeaverConfig
from .file_filter import FileFilter

logger = logging.getLogger(__name__)


class CodeFileEventHandler(FileSystemEventHandler):
    """Handles file system events for code files with intelligent filtering."""
    
    def __init__(self, watcher: 'FileWatcher'):
        super().__init__()
        self.watcher = watcher
    
    def on_modified(self, event):
        if not event.is_directory:
            self.watcher.handle_file_change(Path(event.src_path), 'modified')
    
    def on_created(self, event):
        if not event.is_directory:
            self.watcher.handle_file_change(Path(event.src_path), 'created')
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.watcher.handle_file_change(Path(event.src_path), 'deleted')


class FileWatcher:
    """Monitors file changes and triggers re-indexing with debouncing."""
    
    def __init__(self, 
                 config: CodeWeaverConfig,
                 root_path: Path,
                 reindex_callback: Callable[[Path, str], None]):
        self.config = config
        self.root_path = Path(root_path).resolve()
        self.reindex_callback = reindex_callback
        self.enabled = config.indexing.enable_auto_reindex and WATCHDOG_AVAILABLE
        
        # Debouncing state
        self.pending_changes: Dict[Path, str] = {}
        self.debounce_task: Optional[asyncio.Task] = None
        self.debounce_seconds = config.indexing.watch_debounce_seconds
        
        # File filtering
        self.file_filter = FileFilter(config, root_path)
        
        # Watchdog components
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[CodeFileEventHandler] = None
        
        if not WATCHDOG_AVAILABLE and config.indexing.enable_auto_reindex:
            logger.warning("watchdog not available, install with: uv add watchdog")
    
    def start(self) -> bool:
        """Start watching for file changes."""
        if not self.enabled:
            logger.info("File watching disabled in configuration")
            return False
        
        if not WATCHDOG_AVAILABLE:
            logger.error("Cannot start file watching: watchdog not available")
            return False
        
        try:
            self.observer = Observer()
            self.event_handler = CodeFileEventHandler(self)
            
            # Watch the root directory recursively
            self.observer.schedule(
                self.event_handler,
                str(self.root_path),
                recursive=True
            )
            
            self.observer.start()
            logger.info(f"Started watching for file changes in: {self.root_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            return False
    
    def stop(self):
        """Stop watching for file changes."""
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5.0)
            logger.info("Stopped file watching")
        
        if self.debounce_task and not self.debounce_task.done():
            self.debounce_task.cancel()
    
    def handle_file_change(self, file_path: Path, change_type: str):
        """Handle a file change event with debouncing."""
        if not self.enabled:
            return
        
        try:
            # Check if file should be processed
            if not self._should_process_file(file_path, change_type):
                return
            
            # Add to pending changes
            self.pending_changes[file_path] = change_type
            
            logger.debug(f"File {change_type}: {file_path}")
            
            # Cancel existing debounce task
            if self.debounce_task and not self.debounce_task.done():
                self.debounce_task.cancel()
            
            # Schedule debounced processing
            self.debounce_task = asyncio.create_task(self._debounced_process())
            
        except Exception as e:
            logger.warning(f"Error handling file change {file_path}: {e}")
    
    def _should_process_file(self, file_path: Path, change_type: str) -> bool:
        """Check if a file change should trigger re-indexing."""
        try:
            # Skip if outside root path
            try:
                file_path.resolve().relative_to(self.root_path)
            except ValueError:
                return False
            
            # For deletions, we can't check file properties
            if change_type == 'deleted':
                # Check if it was likely a code file based on extension
                return self._has_code_extension(file_path)
            
            # Use file filter for other changes
            return self.file_filter.should_include_file(file_path)
            
        except Exception as e:
            logger.debug(f"Error checking file {file_path}: {e}")
            return False
    
    def _has_code_extension(self, file_path: Path) -> bool:
        """Check if a file has a code file extension."""
        from .chunker import AstGrepChunker
        chunker = AstGrepChunker()
        return file_path.suffix.lower() in chunker.SUPPORTED_LANGUAGES
    
    async def _debounced_process(self):
        """Process pending changes after debounce delay."""
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_seconds)
            
            if not self.pending_changes:
                return
            
            # Get snapshot of pending changes
            changes = dict(self.pending_changes)
            self.pending_changes.clear()
            
            logger.info(f"Processing {len(changes)} file changes after debounce")
            
            # Group changes by type
            created_files = [f for f, t in changes.items() if t == 'created']
            modified_files = [f for f, t in changes.items() if t == 'modified']
            deleted_files = [f for f, t in changes.items() if t == 'deleted']
            
            # Process changes
            await self._process_file_changes(created_files, modified_files, deleted_files)
            
        except asyncio.CancelledError:
            logger.debug("Debounced processing cancelled")
        except Exception as e:
            logger.error(f"Error in debounced processing: {e}")
    
    async def _process_file_changes(self, created: list, modified: list, deleted: list):
        """Process the accumulated file changes."""
        try:
            # Log change summary
            change_summary = []
            if created:
                change_summary.append(f"{len(created)} created")
            if modified:
                change_summary.append(f"{len(modified)} modified")  
            if deleted:
                change_summary.append(f"{len(deleted)} deleted")
            
            logger.info(f"Auto-reindexing triggered by: {', '.join(change_summary)}")
            
            # Determine if we need full or partial reindex
            significant_changes = len(created) + len(deleted)
            
            if significant_changes > 10:
                # Many files created/deleted - do full reindex
                logger.info("Performing full reindex due to significant changes")
                await self.reindex_callback(self.root_path, 'full')
            else:
                # Incremental reindex for specific files
                all_changed_files = created + modified
                if all_changed_files:
                    logger.info(f"Performing incremental reindex for {len(all_changed_files)} files")
                    await self.reindex_callback(all_changed_files, 'incremental')
                
                # Handle deletions separately
                if deleted:
                    logger.info(f"Removing {len(deleted)} deleted files from index")
                    await self.reindex_callback(deleted, 'delete')
            
        except Exception as e:
            logger.error(f"Error processing file changes: {e}")
    
    def get_status(self) -> Dict[str, any]:
        """Get the current status of the file watcher."""
        return {
            "enabled": self.enabled,
            "watchdog_available": WATCHDOG_AVAILABLE,
            "watching": self.observer.is_alive() if self.observer else False,
            "root_path": str(self.root_path),
            "debounce_seconds": self.debounce_seconds,
            "pending_changes": len(self.pending_changes),
            "filter_stats": self.file_filter.get_filtering_stats()
        }


class FileWatcherManager:
    """Manages file watchers for multiple root paths."""
    
    def __init__(self, config: CodeWeaverConfig):
        self.config = config
        self.watchers: Dict[Path, FileWatcher] = {}
    
    def add_watcher(self, root_path: Path, reindex_callback: Callable) -> bool:
        """Add a file watcher for a root path."""
        if root_path in self.watchers:
            logger.warning(f"Watcher already exists for {root_path}")
            return False
        
        watcher = FileWatcher(self.config, root_path, reindex_callback)
        if watcher.start():
            self.watchers[root_path] = watcher
            return True
        return False
    
    def remove_watcher(self, root_path: Path) -> bool:
        """Remove a file watcher for a root path."""
        if root_path not in self.watchers:
            return False
        
        self.watchers[root_path].stop()
        del self.watchers[root_path]
        return True
    
    def stop_all(self):
        """Stop all file watchers."""
        for watcher in self.watchers.values():
            watcher.stop()
        self.watchers.clear()
    
    def get_status(self) -> Dict[str, any]:
        """Get status of all watchers."""
        return {
            "total_watchers": len(self.watchers),
            "watchers": {
                str(path): watcher.get_status() 
                for path, watcher in self.watchers.items()
            }
        }