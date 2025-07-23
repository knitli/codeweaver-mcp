# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Data models for the Code Weaver MCP server.

Contains core data structures used throughout the codebase.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CodeChunk:
    """Represents a semantic chunk of code."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'struct', 'enum', 'block'
    language: str
    hash: str
    node_kind: Optional[str] = None  # ast-grep node kind
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to Qdrant metadata format."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "hash": self.hash,
            "node_kind": self.node_kind or "",
            "content": self.content  # Store content for reranking
        }

    @classmethod
    def create_with_hash(cls, content: str, file_path: str, start_line: int, 
                        end_line: int, chunk_type: str, language: str, 
                        node_kind: Optional[str] = None) -> 'CodeChunk':
        """Create a CodeChunk with automatically generated hash."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return cls(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=language,
            hash=content_hash,
            node_kind=node_kind
        )