# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main MCP server implementation for code embeddings and search.

Orchestrates semantic search, chunking, and indexing functionality
using Voyage AI embeddings and Qdrant vector database.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue
)

from .chunker import AstGrepChunker, AST_GREP_AVAILABLE
from .config import get_config, CodeWeaverConfig
from .embeddings import VoyageAIEmbedder, VoyageAIReranker
from .file_filter import FileFilter
from .file_watcher import FileWatcherManager
from .models import CodeChunk
from .rate_limiter import RateLimiter
from .search import AstGrepStructuralSearch

logger = logging.getLogger(__name__)


class CodeEmbeddingsServer:
    """Main MCP server for code embeddings with ast-grep integration."""
    
    def __init__(self, config: Optional[CodeWeaverConfig] = None):
        # Load configuration
        self.config = config or get_config()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.config.rate_limiting)
        
        # Initialize components with configuration and rate limiting
        self.embedder = VoyageAIEmbedder(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            rate_limiter=self.rate_limiter
        )
        self.reranker = VoyageAIReranker(
            api_key=self.config.embedding.api_key,
            rate_limiter=self.rate_limiter
        )
        self.chunker = AstGrepChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            min_chunk_size=self.config.chunking.min_chunk_size
        )
        self.structural_search = AstGrepStructuralSearch()
        
        # Initialize file watcher manager
        self.file_watcher_manager = FileWatcherManager(self.config)
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            collection_name = self.config.qdrant.collection_name
            if collection_name not in collection_names:
                logger.info(f"Creating collection: {collection_name}")
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    async def index_codebase(self, root_path: str, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Index a codebase for semantic search using ast-grep."""
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        
        # Use file filter for intelligent file discovery
        file_filter = FileFilter(self.config, root)
        files = file_filter.find_files(patterns)
        
        logger.info(f"Found {len(files)} files to index after filtering")
        
        # Process files in batches
        batch_size = self.config.indexing.batch_size
        total_chunks = 0
        processed_languages = set()
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_chunks = []
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files (size check already done by FileFilter)
                    if not content.strip():
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
        
        # Set up file watching if enabled
        watcher_started = False
        if self.config.indexing.enable_auto_reindex:
            watcher_started = self.file_watcher_manager.add_watcher(
                root_path=root,
                reindex_callback=self._handle_file_changes
            )
        
        return {
            "status": "success",
            "files_processed": len(files),
            "total_chunks": total_chunks,
            "languages_found": list(processed_languages),
            "collection": self.config.qdrant.collection_name,
            "ast_grep_available": AST_GREP_AVAILABLE,
            "filtering_stats": file_filter.get_filtering_stats(),
            "file_watching": {
                "enabled": self.config.indexing.enable_auto_reindex,
                "started": watcher_started
            }
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
            collection_name=self.config.qdrant.collection_name,
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
            collection_name=self.config.qdrant.collection_name,
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