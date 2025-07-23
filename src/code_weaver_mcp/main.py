#!/usr/bin/env python3
"""
Voyage AI Code Embeddings MCP Server with ast-grep Integration

A Model Context Protocol server that provides semantic code search using:
- Voyage AI's voyage-code-3 embeddings (best-in-class for code)  
- Voyage AI's voyage-rerank-2 reranker
- Qdrant Cloud vector database
- ast-grep tree-sitter parsing for 20+ languages
- Bonus: Direct ast-grep structural search capabilities

Usage:
    python server.py

Environment Variables:
    VOYAGE_API_KEY: Your Voyage AI API key
    QDRANT_URL: Your Qdrant Cloud URL  
    QDRANT_API_KEY: Your Qdrant Cloud API key
    COLLECTION_NAME: Name for the code collection (default: code-embeddings)
"""

import asyncio
import json
import logging
import os
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# Third party imports
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue
)

# ast-grep for proper tree-sitter parsing
try:
    from ast_grep_py import SgRoot
    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False
    logging.warning("ast-grep-py not available, falling back to simple chunking")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AstGrepChunker:
    """Handles semantic chunking using ast-grep's tree-sitter parsers for 20+ languages."""
    
    # Languages supported by ast-grep
    SUPPORTED_LANGUAGES = {
        # Web technologies
        '.html': 'html',
        '.htm': 'html', 
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.css': 'css',
        '.scss': 'css',
        '.sass': 'css',
        '.vue': 'vue',
        '.svelte': 'svelte',
        
        # Systems programming
        '.rs': 'rust',
        '.go': 'go',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.cs': 'csharp',
        '.zig': 'zig',
        
        # High-level languages
        '.py': 'python',
        '.java': 'java',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.dart': 'dart',
        '.lua': 'lua',
        '.perl': 'perl',
        '.r': 'r',
        '.jl': 'julia',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.elm': 'elm',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',
        '.clj': 'clojure',
        '.cljs': 'clojure',
        
        # Data and config
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        '.sql': 'sql',
        '.thrift': 'thrift',
        '.proto': 'protobuf',
        
        # Documentation
        '.md': 'markdown',
        '.rst': 'markdown',
        '.tex': 'latex',
        
        # Shell and scripts
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.fish': 'bash',
        '.ps1': 'powershell',
        '.bat': 'bat',
        '.dockerfile': 'docker',
        '.containerfile': 'docker',
        
        # Build systems
        '.cmake': 'cmake',
        '.mk': 'make',
        '.makefile': 'make',
        '.gradle': 'kotlin',  # Gradle uses Kotlin DSL
        '.bazel': 'python',   # Bazel uses Python-like syntax
        '.bzl': 'python',
    }
    
    # Chunk types we look for in different languages
    CHUNK_PATTERNS = {
        'python': [
            ('function_definition', 'function'),
            ('class_definition', 'class'),
            ('async_function_definition', 'async_function'),
        ],
        'javascript': [
            ('function_declaration', 'function'),
            ('function_expression', 'function'),
            ('arrow_function', 'function'),
            ('class_declaration', 'class'),
            ('method_definition', 'method'),
        ],
        'typescript': [
            ('function_declaration', 'function'),
            ('function_expression', 'function'),
            ('arrow_function', 'function'),
            ('class_declaration', 'class'),
            ('method_definition', 'method'),
            ('interface_declaration', 'interface'),
            ('type_alias_declaration', 'type'),
            ('enum_declaration', 'enum'),
        ],
        'rust': [
            ('function_item', 'function'),
            ('impl_item', 'impl'),
            ('struct_item', 'struct'),
            ('enum_item', 'enum'),
            ('trait_item', 'trait'),
            ('mod_item', 'module'),
            ('macro_definition', 'macro'),
        ],
        'go': [
            ('function_declaration', 'function'),
            ('method_declaration', 'method'),
            ('type_declaration', 'type'),
            ('interface_type', 'interface'),
            ('struct_type', 'struct'),
        ],
        'java': [
            ('method_declaration', 'method'),
            ('class_declaration', 'class'),
            ('interface_declaration', 'interface'),
            ('enum_declaration', 'enum'),
            ('constructor_declaration', 'constructor'),
        ],
        'c': [
            ('function_definition', 'function'),
            ('struct_specifier', 'struct'),
            ('union_specifier', 'union'),
            ('enum_specifier', 'enum'),
        ],
        'cpp': [
            ('function_definition', 'function'),
            ('class_specifier', 'class'),
            ('struct_specifier', 'struct'),
            ('namespace_definition', 'namespace'),
            ('template_declaration', 'template'),
        ],
    }
    
    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        if not AST_GREP_AVAILABLE:
            logger.warning("ast-grep not available, install with: pip install ast-grep-py")
    
    def chunk_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk a file into semantic pieces using ast-grep."""
        if not content.strip():
            return []
        
        language = self._detect_language(file_path)
        
        if not AST_GREP_AVAILABLE or language not in self.CHUNK_PATTERNS:
            return self._fallback_chunk(file_path, content, language)
        
        try:
            return self._ast_grep_chunk(file_path, content, language)
        except Exception as e:
            logger.warning(f"ast-grep chunking failed for {file_path}: {e}, falling back")
            return self._fallback_chunk(file_path, content, language)
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        # Handle special cases
        if file_path.name.lower() in ('dockerfile', 'containerfile'):
            return 'docker'
        if file_path.name.lower() in ('makefile', 'gnumakefile'):
            return 'make'
        if file_path.name.lower().endswith('.cmake'):
            return 'cmake'
        
        return self.SUPPORTED_LANGUAGES.get(suffix, 'text')
    
    def _ast_grep_chunk(self, file_path: Path, content: str, language: str) -> List[CodeChunk]:
        """Use ast-grep to semantically chunk code."""
        try:
            # Parse with ast-grep
            root = SgRoot(content, language)
            tree = root.root()
            
            chunks = []
            patterns = self.CHUNK_PATTERNS.get(language, [])
            
            # Find semantic chunks using ast-grep patterns
            for node_kind, chunk_type in patterns:
                matches = self._find_nodes_by_kind(tree, node_kind)
                
                for match in matches:
                    chunk = self._create_chunk_from_node(
                        file_path, match, content, chunk_type, language, node_kind
                    )
                    if chunk and len(chunk.content.strip()) >= self.min_chunk_size:
                        chunks.append(chunk)
            
            # If no semantic chunks found, fall back to block-level chunking
            if not chunks:
                chunks = self._block_chunk_with_ast_grep(file_path, content, language, tree)
            
            return self._merge_overlapping_chunks(chunks)
            
        except Exception as e:
            logger.error(f"ast-grep parsing failed for {language}: {e}")
            raise
    
    def _find_nodes_by_kind(self, root_node, kind: str) -> List:
        """Find all nodes of a specific kind in the AST."""
        matches = []
        
        # Use ast-grep's find method with pattern matching
        pattern = f"$_:{kind}"  # Generic pattern to match node kind
        
        # For specific known patterns, use more precise matching
        if kind == 'function_definition':
            pattern = "def $_($$$_): $$$_"
        elif kind == 'class_definition':
            pattern = "class $_($$$_): $$$_"
        elif kind == 'function_declaration':
            pattern = "function $_($$$_) { $$$_ }"
        elif kind == 'class_declaration':
            pattern = "class $_ { $$$_ }"
        
        try:
            found = root_node.find(pattern=pattern)
            if found:
                matches.append(found)
        except:
            # If pattern matching fails, traverse manually
            matches = self._traverse_for_kind(root_node, kind)
        
        return matches
    
    def _traverse_for_kind(self, node, target_kind: str) -> List:
        """Manually traverse AST to find nodes of target kind."""
        matches = []
        
        # This is a simplified traversal - ast-grep has better methods
        # but this provides a fallback
        try:
            if hasattr(node, 'kind') and node.kind() == target_kind:
                matches.append(node)
            
            # Recursively check children
            if hasattr(node, 'children'):
                for child in node.children():
                    matches.extend(self._traverse_for_kind(child, target_kind))
        except:
            pass
        
        return matches
    
    def _create_chunk_from_node(self, file_path: Path, node, content: str, 
                               chunk_type: str, language: str, node_kind: str) -> Optional[CodeChunk]:
        """Create a CodeChunk from an ast-grep node."""
        try:
            # Get node range
            range_info = node.range()
            start_pos = range_info.start
            end_pos = range_info.end
            
            # Extract content
            lines = content.split('\n')
            chunk_lines = lines[start_pos.line:end_pos.line + 1]
            
            if not chunk_lines:
                return None
            
            # Handle partial first and last lines
            if len(chunk_lines) == 1:
                chunk_content = chunk_lines[0][start_pos.column:end_pos.column]
            else:
                chunk_lines[0] = chunk_lines[0][start_pos.column:]
                chunk_lines[-1] = chunk_lines[-1][:end_pos.column]
                chunk_content = '\n'.join(chunk_lines)
            
            if not chunk_content.strip():
                return None
            
            # Create hash
            content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
            
            return CodeChunk(
                content=chunk_content,
                file_path=str(file_path),
                start_line=start_pos.line + 1,  # 1-indexed
                end_line=end_pos.line + 1,      # 1-indexed  
                chunk_type=chunk_type,
                language=language,
                hash=content_hash,
                node_kind=node_kind
            )
            
        except Exception as e:
            logger.warning(f"Failed to create chunk from node: {e}")
            return None
    
    def _block_chunk_with_ast_grep(self, file_path: Path, content: str, 
                                  language: str, tree) -> List[CodeChunk]:
        """Create larger block chunks when no semantic chunks found."""
        lines = content.split('\n')
        chunks = []
        
        # Create reasonably sized chunks
        chunk_size = min(self.max_chunk_size // 20, 50)  # ~50 lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            end_idx = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end_idx]
            
            if chunk_lines and any(line.strip() for line in chunk_lines):
                chunk_content = '\n'.join(chunk_lines)
                content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=i + 1,
                    end_line=end_idx,
                    chunk_type='block',
                    language=language,
                    hash=content_hash,
                    node_kind='block'
                ))
        
        return chunks
    
    def _fallback_chunk(self, file_path: Path, content: str, language: str) -> List[CodeChunk]:
        """Simple fallback chunking when ast-grep is not available."""
        lines = content.split('\n')
        chunks = []
        
        # Look for common patterns with regex
        import re
        
        patterns = {
            'python': [
                (r'^(class\s+\w+.*?:)', 'class'),
                (r'^(def\s+\w+.*?:)', 'function'),
                (r'^(async\s+def\s+\w+.*?:)', 'async_function'),
            ],
            'javascript': [
                (r'^(function\s+\w+.*?\{)', 'function'),
                (r'^(class\s+\w+.*?\{)', 'class'),
                (r'^(\w+\s*=\s*function.*?\{)', 'function'),
                (r'^(\w+\s*=>\s*\{)', 'function'),
            ],
            'rust': [
                (r'^(fn\s+\w+.*?\{)', 'function'),
                (r'^(struct\s+\w+.*?\{)', 'struct'),
                (r'^(impl\s+.*?\{)', 'impl'),
                (r'^(enum\s+\w+.*?\{)', 'enum'),
            ]
        }
        
        lang_patterns = patterns.get(language, [])
        
        if lang_patterns:
            current_chunk = []
            current_start = 0
            current_type = 'block'
            
            for i, line in enumerate(lines):
                # Check if line starts a new semantic block
                for pattern, chunk_type in lang_patterns:
                    if re.match(pattern, line.strip()):
                        # Save previous chunk
                        if current_chunk:
                            chunks.append(self._create_fallback_chunk(
                                file_path, current_chunk, current_start, i - 1, 
                                current_type, language
                            ))
                        
                        # Start new chunk
                        current_chunk = [line]
                        current_start = i
                        current_type = chunk_type
                        break
                else:
                    current_chunk.append(line)
            
            # Add final chunk
            if current_chunk:
                chunks.append(self._create_fallback_chunk(
                    file_path, current_chunk, current_start, len(lines) - 1,
                    current_type, language
                ))
        
        else:
            # Pure line-based chunking
            chunk_size = 30  # Lines per chunk
            for i in range(0, len(lines), chunk_size):
                end_idx = min(i + chunk_size, len(lines))
                chunk_lines = lines[i:end_idx]
                
                if chunk_lines and any(line.strip() for line in chunk_lines):
                    chunks.append(self._create_fallback_chunk(
                        file_path, chunk_lines, i, end_idx - 1, 'block', language
                    ))
        
        return chunks
    
    def _create_fallback_chunk(self, file_path: Path, lines: List[str], 
                              start_line: int, end_line: int, chunk_type: str,
                              language: str) -> CodeChunk:
        """Create a CodeChunk from lines (fallback method)."""
        content = '\n'.join(lines)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return CodeChunk(
            content=content,
            file_path=str(file_path),
            start_line=start_line + 1,  # 1-indexed
            end_line=end_line + 1,      # 1-indexed
            chunk_type=chunk_type,
            language=language,
            hash=content_hash,
            node_kind=None
        )
    
    def _merge_overlapping_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Merge chunks that are too small or overlapping."""
        if not chunks:
            return chunks
        
        # Sort by file path and start line
        chunks.sort(key=lambda c: (c.file_path, c.start_line))
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Merge if chunks are from same file and very small or overlapping
            if (current.file_path == next_chunk.file_path and
                (len(current.content) < self.min_chunk_size or
                 current.end_line >= next_chunk.start_line - 1)):
                
                # Merge chunks
                merged_content = current.content + '\n' + next_chunk.content
                current = CodeChunk(
                    content=merged_content,
                    file_path=current.file_path,
                    start_line=current.start_line,
                    end_line=next_chunk.end_line,
                    chunk_type=current.chunk_type,
                    language=current.language,
                    hash=hashlib.md5(merged_content.encode()).hexdigest()[:8],
                    node_kind=current.node_kind
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged

class VoyageAIEmbedder:
    """Handles Voyage AI embeddings for code."""
    
    def __init__(self, api_key: str, model: str = "voyage-code-3"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = 1024  # Default dimension for voyage-code-3
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents (code chunks)."""
        try:
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="document",
                output_dimension=self.dimension
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="query",
                output_dimension=self.dimension
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

class VoyageAIReranker:
    """Handles Voyage AI reranking."""
    
    def __init__(self, api_key: str, model: str = "voyage-rerank-2"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents for a query."""
        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k
            )
            return result.results
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            raise

class AstGrepStructuralSearch:
    """Provides direct ast-grep structural search capabilities."""
    
    def __init__(self):
        self.available = AST_GREP_AVAILABLE
        if not self.available:
            logger.warning("ast-grep not available for structural search")
    
    async def structural_search(self, pattern: str, language: str, 
                               root_path: str) -> List[Dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        if not self.available:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")
        
        results = []
        root = Path(root_path)
        
        # Find files for the language
        extensions = self._get_extensions_for_language(language)
        files = []
        
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
        
        # Search each file
        for file_path in files[:100]:  # Limit to avoid overload
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse and search with ast-grep
                sg_root = SgRoot(content, language)
                tree = sg_root.root()
                
                matches = tree.find(pattern=pattern)
                if matches:
                    range_info = matches.range()
                    results.append({
                        "file_path": str(file_path),
                        "match_content": matches.text(),
                        "start_line": range_info.start.line + 1,
                        "end_line": range_info.end.line + 1,
                        "start_column": range_info.start.column + 1,
                        "end_column": range_info.end.column + 1,
                    })
                
            except Exception as e:
                logger.warning(f"Error searching {file_path}: {e}")
                continue
        
        return results
    
    def _get_extensions_for_language(self, language: str) -> List[str]:
        """Get file extensions for a language."""
        lang_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'rust': ['.rs'],
            'go': ['.go'],
            'java': ['.java'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp', '.cc', '.cxx'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass'],
        }
        return lang_map.get(language, [f'.{language}'])

class CodeEmbeddingsServer:
    """Main MCP server for code embeddings with ast-grep integration."""
    
    def __init__(self):
        # Environment configuration
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("COLLECTION_NAME", "code-embeddings")
        
        if not self.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY environment variable is required")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")
        
        # Initialize components
        self.embedder = VoyageAIEmbedder(self.voyage_api_key)
        self.reranker = VoyageAIReranker(self.voyage_api_key)
        self.chunker = AstGrepChunker()
        self.structural_search = AstGrepStructuralSearch()
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    async def index_codebase(self, root_path: str, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Index a codebase for semantic search using ast-grep."""
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        
        # Use ast-grep supported extensions if no patterns specified
        if patterns is None:
            patterns = list(self.chunker.SUPPORTED_LANGUAGES.keys())
            patterns = [f"**/*{ext}" for ext in patterns]
        
        # Collect files
        files = []
        for pattern in patterns:
            files.extend(root.glob(pattern))
        
        # Filter out common ignore patterns
        ignore_patterns = {
            'node_modules', '.git', '.venv', 'venv', '__pycache__', 
            'target', 'build', 'dist', '.next', '.nuxt', 'coverage',
            '.pytest_cache', '.mypy_cache', '.vscode', '.idea'
        }
        
        files = [f for f in files if not any(part in ignore_patterns for part in f.parts)]
        
        logger.info(f"Found {len(files)} files to index")
        
        # Process files in batches
        batch_size = 8  # Smaller batches for ast-grep processing
        total_chunks = 0
        processed_languages = set()
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_chunks = []
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files or files larger than 1MB
                    if not content.strip() or len(content) > 1024 * 1024:
                        continue
                    
                    # Chunk the file using ast-grep
                    chunks = self.chunker.chunk_file(file_path, content)
                    batch_chunks.extend(chunks)
                    
                    # Track processed languages
                    if chunks:
                        processed_languages.add(chunks[0].language)
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
            
            if batch_chunks:
                await self._index_chunks(batch_chunks)
                total_chunks += len(batch_chunks)
                logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
        
        return {
            "status": "success",
            "files_processed": len(files),
            "total_chunks": total_chunks,
            "languages_found": list(processed_languages),
            "collection": self.collection_name,
            "ast_grep_available": AST_GREP_AVAILABLE
        }
    
    async def _index_chunks(self, chunks: List[CodeChunk]):
        """Index a batch of code chunks."""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_documents(texts)
        
        # Create points for Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=hash(f"{chunk.file_path}:{chunk.start_line}:{chunk.hash}") & ((1 << 63) - 1),
                vector=embedding,
                payload=chunk.to_metadata()
            )
            points.append(point)
        
        # Upload to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def search_code(self, query: str, limit: int = 10, 
                         file_filter: Optional[str] = None,
                         language_filter: Optional[str] = None,
                         chunk_type_filter: Optional[str] = None,
                         rerank: bool = True) -> List[Dict[str, Any]]:
        """Search for code using semantic similarity with advanced filtering."""
        
        # Generate query embedding
        query_vector = await self.embedder.embed_query(query)
        
        # Build filters
        filter_conditions = []
        
        if file_filter:
            filter_conditions.append(
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_filter)
                )
            )
        
        if language_filter:
            filter_conditions.append(
                FieldCondition(
                    key="language", 
                    match=MatchValue(value=language_filter)
                )
            )
        
        if chunk_type_filter:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type_filter)
                )
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search Qdrant
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit * 2 if rerank else limit  # Get more for reranking
        )
        
        results = []
        for hit in search_result:
            result = {
                "content": hit.payload["content"],
                "file_path": hit.payload["file_path"],
                "start_line": hit.payload["start_line"],
                "end_line": hit.payload["end_line"],
                "chunk_type": hit.payload["chunk_type"],
                "language": hit.payload["language"],
                "node_kind": hit.payload.get("node_kind", ""),
                "similarity_score": hit.score
            }
            results.append(result)
        
        # Rerank if requested
        if rerank and len(results) > 1:
            try:
                documents = [r["content"] for r in results]
                rerank_results = await self.reranker.rerank(query, documents, top_k=limit)
                
                # Reorder results based on reranking
                reranked = []
                for rerank_item in rerank_results:
                    original_result = results[rerank_item["index"]]
                    original_result["rerank_score"] = rerank_item["relevance_score"]
                    reranked.append(original_result)
                
                results = reranked
            except Exception as e:
                logger.warning(f"Reranking failed, using similarity search only: {e}")
        
        return results[:limit]
    
    async def ast_grep_search(self, pattern: str, language: str, 
                             root_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        if not AST_GREP_AVAILABLE:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")
        
        results = await self.structural_search.structural_search(
            pattern=pattern,
            language=language, 
            root_path=root_path
        )
        
        return results[:limit]
    
    async def get_supported_languages(self) -> Dict[str, Any]:
        """Get information about supported languages and capabilities."""
        return {
            "ast_grep_available": AST_GREP_AVAILABLE,
            "supported_languages": list(set(self.chunker.SUPPORTED_LANGUAGES.values())),
            "language_extensions": self.chunker.SUPPORTED_LANGUAGES,
            "chunk_patterns": {
                lang: [pattern[1] for pattern in patterns] 
                for lang, patterns in self.chunker.CHUNK_PATTERNS.items()
            },
            "voyage_models": {
                "embedding": "voyage-code-3",
                "reranker": "voyage-rerank-2"
            }
        }

# MCP Server Setup
app = Server("voyage-code-embeddings-astgrep")

@app.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    tools = [
        types.Tool(
            name="index_codebase",
            description="Index a codebase for semantic search using Voyage AI embeddings and ast-grep parsing",
            inputSchema={
                "type": "object",
                "properties": {
                    "root_path": {
                        "type": "string",
                        "description": "Root directory path of the codebase to index"
                    },
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns to include (optional, defaults to all supported languages)"
                    }
                },
                "required": ["root_path"]
            }
        ),
        types.Tool(
            name="search_code",
            description="Search indexed code using natural language queries with advanced filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Optional file path filter (e.g., 'src/api')"
                    },
                    "language_filter": {
                        "type": "string",
                        "description": "Optional language filter (e.g., 'python', 'javascript')"
                    },
                    "chunk_type_filter": {
                        "type": "string",
                        "description": "Optional chunk type filter (e.g., 'function', 'class')"
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Whether to use Voyage AI reranker for better results",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="ast_grep_search",
            description="Perform structural code search using ast-grep patterns (requires ast-grep-py)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "ast-grep pattern (e.g., 'function $_($$_) { $$_ }' for JS functions)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (python, javascript, rust, etc.)"
                    },
                    "root_path": {
                        "type": "string",
                        "description": "Root directory to search in"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20
                    }
                },
                "required": ["pattern", "language", "root_path"]
            }
        ),
        types.Tool(
            name="get_supported_languages",
            description="Get information about supported languages and ast-grep capabilities",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]
    
    return tools

# Global server instance
server_instance = None

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls."""
    global server_instance
    
    if server_instance is None:
        server_instance = CodeEmbeddingsServer()
    
    try:
        if name == "index_codebase":
            result = await server_instance.index_codebase(
                root_path=arguments["root_path"],
                patterns=arguments.get("patterns")
            )
            
            status_text = f"? Indexing completed successfully!\n\n"
            status_text += f"?? Files processed: {result['files_processed']}\n"
            status_text += f"?? Total chunks: {result['total_chunks']}\n" 
            status_text += f"??? Collection: {result['collection']}\n"
            status_text += f"?? Languages found: {', '.join(result['languages_found'])}\n"
            status_text += f"?? ast-grep parsing: {'? Available' if result['ast_grep_available'] else '?? Fallback mode'}"
            
            return [types.TextContent(type="text", text=status_text)]
        
        elif name == "search_code":
            results = await server_instance.search_code(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
                file_filter=arguments.get("file_filter"),
                language_filter=arguments.get("language_filter"),
                chunk_type_filter=arguments.get("chunk_type_filter"),
                rerank=arguments.get("rerank", True)
            )
            
            if not results:
                return [types.TextContent(
                    type="text",
                    text="No results found for your query. Try broader search terms or check if your codebase is indexed."
                )]
            
            # Format results with enhanced information
            response_text = f"?? Found {len(results)} results for: **'{arguments['query']}'**\n\n"
            
            for i, result in enumerate(results, 1):
                score_info = f"Similarity: {result['similarity_score']:.3f}"
                if 'rerank_score' in result:
                    score_info += f", Rerank: {result['rerank_score']:.3f}"
                
                response_text += f"## ?? Result {i}\n"
                response_text += f"**File:** `{result['file_path']}` (lines {result['start_line']}-{result['end_line']})\n"
                response_text += f"**Type:** {result['chunk_type']} ({result['language']})"
                if result.get('node_kind'):
                    response_text += f"  AST: {result['node_kind']}"
                response_text += f"\n**Score:** {score_info}\n\n"
                response_text += f"```{result['language']}\n{result['content']}\n```\n\n"
            
            return [types.TextContent(type="text", text=response_text)]
        
        elif name == "ast_grep_search":
            if not AST_GREP_AVAILABLE:
                return [types.TextContent(
                    type="text",
                    text="? ast-grep not available. Install with: `pip install ast-grep-py`"
                )]
            
            results = await server_instance.ast_grep_search(
                pattern=arguments["pattern"],
                language=arguments["language"],
                root_path=arguments["root_path"],
                limit=arguments.get("limit", 20)
            )
            
            if not results:
                return [types.TextContent(
                    type="text",
                    text=f"No structural matches found for pattern: `{arguments['pattern']}`"
                )]
            
            response_text = f"?? Found {len(results)} structural matches for pattern: **`{arguments['pattern']}`**\n\n"
            
            for i, result in enumerate(results, 1):
                response_text += f"## ?? Match {i}\n"
                response_text += f"**File:** `{result['file_path']}`\n"
                response_text += f"**Location:** lines {result['start_line']}-{result['end_line']}, "
                response_text += f"columns {result['start_column']}-{result['end_column']}\n\n"
                response_text += f"```{arguments['language']}\n{result['match_content']}\n```\n\n"
            
            return [types.TextContent(type="text", text=response_text)]
        
        elif name == "get_supported_languages":
            info = await server_instance.get_supported_languages()
            
            response_text = "## ?? Supported Languages & Capabilities\n\n"
            response_text += f"**ast-grep Status:** {'? Available' if info['ast_grep_available'] else '?? Not installed'}\n"
            response_text += f"**Voyage Models:** {info['voyage_models']['embedding']} + {info['voyage_models']['reranker']}\n\n"
            
            response_text += "### Supported Languages:\n"
            for lang in sorted(info['supported_languages']):
                chunk_types = info['chunk_patterns'].get(lang, ['block'])
                extensions = [ext for ext, l in info['language_extensions'].items() if l == lang]
                response_text += f"- **{lang}**: {', '.join(extensions)}  {', '.join(chunk_types)}\n"
            
            response_text += "\n### ast-grep Pattern Examples:\n"
            examples = {
                'python': 'def $_($$_): $$_',
                'javascript': 'function $_($$_) { $$_ }',
                'rust': 'fn $_($$_) -> $_ { $$_ }',
                'java': 'public $_ $_($$_) { $$_ }'
            }
            
            for lang, pattern in examples.items():
                response_text += f"- **{lang}**: `{pattern}`\n"
            
            return [types.TextContent(type="text", text=response_text)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error handling tool call {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"? Error: {str(e)}\n\nPlease check your configuration and try again."
        )]

async def main():
    """Main entry point."""
    # Verify environment
    required_vars = ["VOYAGE_API_KEY", "QDRANT_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"? Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\n?? Required environment variables:")
        print("  VOYAGE_API_KEY: Your Voyage AI API key")
        print("  QDRANT_URL: Your Qdrant Cloud URL")
        print("  QDRANT_API_KEY: Your Qdrant Cloud API key (optional if not using auth)")
        print("  COLLECTION_NAME: Collection name (optional, defaults to 'code-embeddings')")
        print("\n?? Optional: Install ast-grep for better parsing:")
        print("  pip install ast-grep-py")
        return
    
    # Check ast-grep availability
    if AST_GREP_AVAILABLE:
        print("? ast-grep available - using tree-sitter parsing for 20+ languages")
    else:
        print("??  ast-grep not available - using fallback parsing")
        print("   Install with: pip install ast-grep-py")
    
    print("?? Starting Voyage AI Code Embeddings MCP Server...")
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="voyage-code-embeddings-astgrep",
                server_version="2.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
