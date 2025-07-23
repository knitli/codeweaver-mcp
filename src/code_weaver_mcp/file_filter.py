# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
File filtering functionality with gitignore awareness.

Provides intelligent file filtering using rignore for .gitignore pattern matching
and additional custom ignore patterns from configuration.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Set

try:
    from rignore import Gitignore
    RIGNORE_AVAILABLE = True
except ImportError:
    RIGNORE_AVAILABLE = False
    logging.warning("rignore not available, falling back to simple filtering")

from .config import CodeWeaverConfig

logger = logging.getLogger(__name__)


class FileFilter:
    """Handles file filtering with gitignore awareness and custom patterns."""
    
    def __init__(self, config: CodeWeaverConfig, root_path: Path):
        self.config = config
        self.root_path = Path(root_path).resolve()
        self.use_gitignore = config.indexing.use_gitignore and RIGNORE_AVAILABLE
        
        # Initialize gitignore if available and enabled
        self.gitignore: Optional[Gitignore] = None
        if self.use_gitignore:
            try:
                self.gitignore = Gitignore(str(self.root_path))
                logger.info(f"Loaded .gitignore patterns from {self.root_path}")
            except Exception as e:
                logger.warning(f"Failed to load .gitignore: {e}, falling back to simple filtering")
                self.use_gitignore = False
        
        # Convert additional ignore patterns to a set for fast lookup
        self.ignore_patterns = set(config.indexing.additional_ignore_patterns)
        
        if not RIGNORE_AVAILABLE and config.indexing.use_gitignore:
            logger.warning("rignore not available, install with: uv add rignore")
    
    def should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on filtering rules."""
        try:
            # Convert to absolute path for consistent handling
            abs_path = file_path.resolve()
            
            # Check if file exists and is readable
            if not abs_path.exists() or not abs_path.is_file():
                return False
            
            # Check against gitignore patterns
            if self.use_gitignore and self.gitignore:
                try:
                    # Get relative path from project root for gitignore matching
                    rel_path = abs_path.relative_to(self.root_path)
                    if self.gitignore.is_ignored(str(rel_path)):
                        logger.debug(f"File ignored by .gitignore: {rel_path}")
                        return False
                except ValueError:
                    # File is outside the project root
                    logger.debug(f"File outside project root: {abs_path}")
                    return False
                except Exception as e:
                    logger.debug(f"Error checking gitignore for {abs_path}: {e}")
            
            # Check against additional ignore patterns
            if self._matches_ignore_patterns(abs_path):
                return False
            
            # Check file size limits
            if not self._check_file_size(abs_path):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error filtering file {file_path}: {e}")
            return False
    
    def _matches_ignore_patterns(self, file_path: Path) -> bool:
        """Check if file matches any of the additional ignore patterns."""
        # Check each part of the path against ignore patterns
        for part in file_path.parts:
            if part in self.ignore_patterns:
                logger.debug(f"File ignored by pattern '{part}': {file_path}")
                return True
        
        # Check filename against patterns
        if file_path.name in self.ignore_patterns:
            logger.debug(f"File ignored by name pattern: {file_path}")
            return True
        
        return False
    
    def _check_file_size(self, file_path: Path) -> bool:
        """Check if file size is within limits."""
        try:
            max_size = self.config.chunking.max_file_size_mb * 1024 * 1024
            file_size = file_path.stat().st_size
            
            if file_size > max_size:
                logger.debug(f"File too large ({file_size / 1024 / 1024:.1f}MB): {file_path}")
                return False
            
            if file_size == 0:
                logger.debug(f"Empty file: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking file size for {file_path}: {e}")
            return False
    
    def filter_files(self, files: List[Path]) -> List[Path]:
        """Filter a list of files using all filtering rules."""
        filtered_files = []
        total_files = len(files)
        
        for file_path in files:
            if self.should_include_file(file_path):
                filtered_files.append(file_path)
        
        filtered_count = len(filtered_files)
        excluded_count = total_files - filtered_count
        
        logger.info(
            f"File filtering: {filtered_count} included, {excluded_count} excluded "
            f"(gitignore: {'enabled' if self.use_gitignore else 'disabled'})"
        )
        
        return filtered_files
    
    def find_files(self, patterns: Optional[List[str]] = None) -> List[Path]:
        """Find and filter files matching the given patterns."""
        if patterns is None:
            # Use default patterns for supported languages
            from .chunker import AstGrepChunker
            chunker = AstGrepChunker()
            patterns = [f"**/*{ext}" for ext in chunker.SUPPORTED_LANGUAGES.keys()]
        
        # Collect files matching patterns
        files = []
        for pattern in patterns:
            try:
                matched_files = list(self.root_path.glob(pattern))
                files.extend(matched_files)
                logger.debug(f"Pattern '{pattern}' matched {len(matched_files)} files")
            except Exception as e:
                logger.warning(f"Error globbing pattern '{pattern}': {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        logger.info(f"Found {len(unique_files)} files before filtering")
        
        # Apply filtering
        return self.filter_files(unique_files)
    
    def get_filtering_stats(self) -> dict:
        """Get statistics about the filtering configuration."""
        return {
            "gitignore_enabled": self.use_gitignore,
            "gitignore_available": RIGNORE_AVAILABLE,
            "additional_patterns": list(self.ignore_patterns),
            "max_file_size_mb": self.config.chunking.max_file_size_mb,
            "root_path": str(self.root_path)
        }