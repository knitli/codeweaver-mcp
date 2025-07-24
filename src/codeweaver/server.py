# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main MCP server implementation for code embeddings and search.

Orchestrates semantic search, chunking, and indexing functionality
using configurable embedding providers and vector database backends.
"""

import logging

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.backends.base import VectorBackend, VectorPoint, DistanceMetric
from codeweaver.chunker import AST_GREP_AVAILABLE, AstGrepChunker
from codeweaver.config import _EXTENDED_CONFIGS_AVAILABLE, CodeWeaverConfig, get_config
from codeweaver.embeddings import VoyageAIReranker, create_embedder
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.factories.integration import (
    LegacyCompatibilityAdapter,
    create_migration_config,
    validate_migration_readiness,
)
from codeweaver.file_filter import FileFilter
from codeweaver.file_watcher import FileWatcherManager
from codeweaver.models import CodeChunk
from codeweaver.providers import RerankProvider, get_provider_factory
from codeweaver.rate_limiter import RateLimiter
from codeweaver.search import AstGrepStructuralSearch
from codeweaver.task_search import TaskSearchCoordinator


logger = logging.getLogger(__name__)


class _RerankProviderAdapter:
    """Adapter to make new reranking providers compatible with legacy interface."""

    def __init__(self, provider: RerankProvider):
        """Initialize adapter with a reranking provider instance."""
        self.provider = provider

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank documents and convert to legacy format."""
        results = await self.provider.rerank(query, documents, top_k)

        # Convert RerankResult objects to legacy dict format
        return [
            {
                "index": result.index,
                "relevance_score": result.relevance_score,
                "document": result.document,
            }
            for result in results
        ]


class CodeEmbeddingsServer:
    """Main MCP server for code embeddings with ast-grep integration."""

    def __init__(self, config: CodeWeaverConfig | None = None):
        """Initialize the code embeddings server.

        Args:
            config: Optional configuration object. If None, loads from default sources.
        """
        # Load configuration
        self.config = config or get_config()

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.config.rate_limiting)

        # Initialize components with configuration and rate limiting
        # Use new provider system with fallback to legacy for compatibility
        try:
            provider_factory = get_provider_factory()

            # Create embedding provider
            embedding_provider = provider_factory.create_embedding_provider(
                config=self.config.embedding, rate_limiter=self.rate_limiter
            )

            # Wrap in legacy interface for backward compatibility
            from codeweaver.embeddings import _ProviderAdapter

            self.embedder = _ProviderAdapter(embedding_provider, self.config.embedding)

            # Create reranking provider (with fallback)
            rerank_provider = provider_factory.get_default_reranking_provider(
                embedding_provider_name=self.config.embedding.provider,
                api_key=self.config.embedding.api_key,
                rate_limiter=self.rate_limiter,
            )

            if rerank_provider is not None:
                # Wrap in legacy interface adapter
                self.reranker = _RerankProviderAdapter(rerank_provider)
            else:
                logger.warning("No reranking provider available, using legacy VoyageAIReranker")
                self.reranker = VoyageAIReranker(
                    api_key=self.config.embedding.api_key, rate_limiter=self.rate_limiter
                )

        except Exception as e:
            logger.warning("Failed to initialize new provider system, using legacy: %s", e)
            # Fallback to legacy system
            self.embedder = create_embedder(
                config=self.config.embedding, rate_limiter=self.rate_limiter
            )
            self.reranker = VoyageAIReranker(
                api_key=self.config.embedding.api_key, rate_limiter=self.rate_limiter
            )
        self.chunker = AstGrepChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )
        self.structural_search = AstGrepStructuralSearch()

        # Initialize Task search coordinator
        self.task_coordinator = TaskSearchCoordinator(task_tool_available=True)

        # Initialize file watcher manager
        self.file_watcher_manager = FileWatcherManager(self.config)

        # Initialize vector backend (with backward compatibility)
        self.backend = self._initialize_backend()
        self.collection_name = self.config.qdrant.collection_name

        # Initialize legacy Qdrant client for backward compatibility
        self.qdrant = QdrantClient(url=self.config.qdrant.url, api_key=self.config.qdrant.api_key)

        # Ensure collection exists
        self._ensure_collection()

    def _initialize_backend(self) -> VectorBackend:
        """Initialize the vector database backend."""
        # Create backend configuration from legacy Qdrant config
        backend_config = BackendConfig(
            provider="qdrant",
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key,
            enable_hybrid_search=False,  # Default to basic for backward compatibility
            enable_sparse_vectors=False,
        )
        
        try:
            backend = BackendFactory.create_backend(backend_config)
            logger.info("Successfully initialized backend: %s", backend_config.provider)
            return backend
        except Exception as e:
            logger.exception("Failed to initialize backend, falling back to legacy mode: %s", e)
            # For now, we'll still maintain the legacy QdrantClient for fallback
            # but in the future this could raise an error
            return None

    def _ensure_collection(self):
        """Ensure the vector collection exists using legacy client (sync)."""
        try:
            # For now, use legacy Qdrant client for synchronous initialization
            # Collection creation via backend will be done lazily when needed
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info("Creating collection using legacy client: %s", self.collection_name)
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension, distance=Distance.COSINE
                    ),
                )
            else:
                logger.info("Collection %s already exists", self.collection_name)
        except Exception:
            logger.exception("Error ensuring collection")
            raise
    
    async def _ensure_collection_async(self):
        """Ensure the vector collection exists using backend abstraction (async)."""
        try:
            if self.backend is not None:
                # Use new backend abstraction
                try:
                    # Check if collection exists
                    collections = await self.backend.list_collections()
                    if self.collection_name not in collections:
                        logger.info("Creating collection using backend: %s", self.collection_name)
                        await self.backend.create_collection(
                            name=self.collection_name,
                            dimension=self.embedder.dimension,
                            distance_metric=DistanceMetric.COSINE
                        )
                    else:
                        logger.info("Collection %s already exists", self.collection_name)
                    return
                except Exception as e:
                    logger.warning("Backend collection creation failed, falling back to legacy: %s", e)
            
            # Fallback using legacy collection creation logic
            self._ensure_collection()
        except Exception:
            logger.exception("Error ensuring collection async")
            raise

    async def index_codebase(
        self, root_path: str, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        """Index a codebase for semantic search using ast-grep."""
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        # Use file filter for intelligent file discovery
        file_filter = FileFilter(self.config, root)
        files = file_filter.find_files(patterns)

        logger.info("Found %d files to index after filtering", len(files))

        # Process files in batches
        batch_size = self.config.indexing.batch_size
        total_chunks = 0
        processed_languages = set()

        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            batch_chunks = []

            for file_path in batch_files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

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
                    logger.warning("Error processing %s: %s", file_path, e)
                    continue

            if batch_chunks:
                await self._index_chunks(batch_chunks)
                total_chunks += len(batch_chunks)
                logger.info("Indexed batch %d: %d chunks", i // batch_size + 1, len(batch_chunks))

        # Set up file watching if enabled
        watcher_started = False
        if self.config.indexing.enable_auto_reindex:
            watcher_started = self.file_watcher_manager.add_watcher(
                root_path=root, reindex_callback=self._handle_file_changes
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
                "started": watcher_started,
            },
        }

    async def _handle_file_changes(self, changed_files: list[Path]):
        """Handle file changes for auto-reindexing."""
        logger.info("Handling changes for %d files", len(changed_files))
        # This is a placeholder - implement actual reindexing logic as needed

    async def _index_chunks(self, chunks: list[CodeChunk]):
        """Index a batch of code chunks."""
        if not chunks:
            return

        # Ensure collection exists (async)
        await self._ensure_collection_async()

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_documents(texts)

        # Create vector points for backend abstraction
        vector_points = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            vector_point = VectorPoint(
                id=hash(f"{chunk.file_path}:{chunk.start_line}:{chunk.hash}") & ((1 << 63) - 1),
                vector=embedding,
                payload=chunk.to_metadata(),
            )
            vector_points.append(vector_point)

        # Upload using backend abstraction with fallback to legacy
        try:
            if self.backend is not None:
                await self.backend.upsert_vectors(self.collection_name, vector_points)
                logger.debug("Uploaded %d vectors using backend", len(vector_points))
            else:
                raise Exception("Backend not available, using fallback")
        except Exception as e:
            logger.warning("Backend upload failed, using legacy Qdrant client: %s", e)
            # Fallback to legacy Qdrant client
            points = []
            for vector_point in vector_points:
                point = PointStruct(
                    id=vector_point.id,
                    vector=vector_point.vector,
                    payload=vector_point.payload,
                )
                points.append(point)
            self.qdrant.upsert(collection_name=self.collection_name, points=points)

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        *,
        file_filter: str | None = None,
        language_filter: str | None = None,
        chunk_type_filter: str | None = None,
        rerank: bool = True,
        use_task_delegation: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code using semantic similarity with advanced filtering.

        Automatically assesses query complexity and suggests Task tool delegation
        for comprehensive or uncertain-scope searches.
        """
        # Assess search complexity
        estimated_files = self._estimate_file_count(file_filter, language_filter)
        assessment = self.task_coordinator.assess_search_complexity(
            query=query,
            file_filter=file_filter,
            language_filter=language_filter,
            estimated_files=estimated_files,
        )

        # Log assessment for transparency
        logger.info(
            "Search assessment: %s complexity, confidence: %.2f, scope: %s",
            assessment.complexity.value,
            assessment.confidence,
            assessment.estimated_scope,
        )

        # Return assessment info along with results
        if assessment.should_use_task and use_task_delegation is not False:
            # Include recommendation in results
            return [
                {
                    "type": "search_recommendation",
                    "message": f"This search has {assessment.complexity.value} complexity. "
                    f"Consider using Task tool for comprehensive results.",
                    "assessment": {
                        "complexity": assessment.complexity.value,
                        "confidence": assessment.confidence,
                        "scope": assessment.estimated_scope,
                        "reasoning": assessment.reasoning,
                        "task_prompt": self.task_coordinator.create_task_prompt_for_semantic_search(
                            query=query,
                            limit=limit,
                            file_filter=file_filter,
                            language_filter=language_filter,
                            chunk_type_filter=chunk_type_filter,
                            rerank=rerank,
                        ),
                    },
                }
            ]

        # Generate query embedding
        query_vector = await self.embedder.embed_query(query)

        # Build universal search filters
        from codeweaver.backends.base import SearchFilter, FilterCondition
        
        filter_conditions = []

        if file_filter:
            filter_conditions.append(
                FilterCondition(field="file_path", operator="eq", value=file_filter)
            )

        if language_filter:
            filter_conditions.append(
                FilterCondition(field="language", operator="eq", value=language_filter)
            )

        if chunk_type_filter:
            filter_conditions.append(
                FilterCondition(field="chunk_type", operator="eq", value=chunk_type_filter)
            )

        universal_filter = SearchFilter(conditions=filter_conditions) if filter_conditions else None

        # Search using backend abstraction with fallback to legacy
        try:
            if self.backend is not None:
                search_result = await self.backend.search_vectors(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    search_filter=universal_filter,
                    limit=limit * 2 if rerank else limit,  # Get more for reranking
                )
                logger.debug("Search using backend returned %d results", len(search_result))
            else:
                raise Exception("Backend not available, using fallback")
        except Exception as e:
            logger.warning("Backend search failed, using legacy Qdrant client: %s", e)
            # Fallback to legacy Qdrant search
            legacy_filter_conditions = []

            if file_filter:
                legacy_filter_conditions.append(
                    FieldCondition(key="file_path", match=MatchValue(value=file_filter))
                )

            if language_filter:
                legacy_filter_conditions.append(
                    FieldCondition(key="language", match=MatchValue(value=language_filter))
                )

            if chunk_type_filter:
                legacy_filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter))
                )

            legacy_filter = Filter(must=legacy_filter_conditions) if legacy_filter_conditions else None

            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=legacy_filter,
                limit=limit * 2 if rerank else limit,  # Get more for reranking
            )

        results = []
        for hit in search_result:
            # Handle both backend abstraction SearchResult and legacy Qdrant hits
            if hasattr(hit, 'payload') and hit.payload is not None:
                # Backend abstraction SearchResult or legacy Qdrant hit
                payload = hit.payload
                score = hit.score
            else:
                # This shouldn't happen, but handle gracefully
                logger.warning("Unexpected search result format: %s", type(hit))
                continue
                
            result = {
                "content": payload["content"],
                "file_path": payload["file_path"],
                "start_line": payload["start_line"],
                "end_line": payload["end_line"],
                "chunk_type": payload["chunk_type"],
                "language": payload["language"],
                "node_kind": payload.get("node_kind", ""),
                "similarity_score": score,
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
                logger.warning("Reranking failed, using similarity search only: %s", e)

        return results[:limit]

    async def ast_grep_search(
        self,
        pattern: str,
        language: str,
        root_path: str,
        limit: int = 20,
        *,
        use_task_delegation: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Perform structural search using ast-grep patterns.

        Automatically assesses pattern complexity and suggests Task tool delegation
        for comprehensive structural searches across large codebases.
        """
        if not AST_GREP_AVAILABLE:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")

        # Assess if this is a comprehensive search
        root = Path(root_path)
        is_large_scope = root.is_dir() and len(list(root.rglob("*"))) > 100
        is_complex_pattern = len(pattern) > 50 or pattern.count("$") > 3

        if (is_large_scope or is_complex_pattern) and use_task_delegation is not False:
            # Suggest Task tool for comprehensive structural search
            return [
                {
                    "type": "search_recommendation",
                    "message": "This structural search covers a large scope or complex pattern. "
                    "Consider using Task tool for comprehensive results.",
                    "assessment": {
                        "pattern_complexity": "complex" if is_complex_pattern else "simple",
                        "scope": f"{'large' if is_large_scope else 'moderate'} codebase at {root_path}",
                        "task_prompt": self.task_coordinator.create_task_prompt_for_structural_search(
                            pattern=pattern, language=language, root_path=root_path, limit=limit
                        ),
                    },
                }
            ]

        results = await self.structural_search.structural_search(
            pattern=pattern, language=language, root_path=root_path
        )

        return results[:limit]

    async def get_supported_languages(self) -> dict[str, Any]:
        """Get information about supported languages and capabilities."""
        return {
            "ast_grep_available": AST_GREP_AVAILABLE,
            "supported_languages": list(set(self.chunker.SUPPORTED_LANGUAGES.values())),
            "language_extensions": self.chunker.SUPPORTED_LANGUAGES,
            "chunk_patterns": {
                lang: [pattern[1] for pattern in patterns]
                for lang, patterns in self.chunker.CHUNK_PATTERNS.items()
            },
            "voyage_models": {"embedding": "voyage-code-3", "reranker": "voyage-rerank-2"},
        }

    def _estimate_file_count(
        self, file_filter: str | None = None, language_filter: str | None = None
    ) -> int:
        """Estimate number of files that would be searched."""
        try:
            # For now, use legacy Qdrant client for synchronous collection info
            # Backend abstraction collection info can be implemented later as async version
            collection_info = self.qdrant.get_collection(self.collection_name)
            total_points = collection_info.points_count

            # Rough estimate based on average chunks per file
            estimated_files = total_points // 5  # Assume ~5 chunks per file on average

            # Adjust based on filters
            if file_filter:
                # Specific path reduces scope significantly
                estimated_files = estimated_files // 10
            if language_filter:
                # Language filter typically reduces to ~20% of files
                estimated_files = estimated_files // 5

            return max(10, estimated_files)  # At least 10 files
        except Exception:
            # Default estimate if we can't get collection info
            return 100


class ExtensibleCodeEmbeddingsServer:
    """
    Next-generation MCP server using the extensible factory architecture.

    Features:
    - Factory-based component creation with dependency injection
    - Plugin support for backends, providers, and data sources
    - Advanced configuration with backward compatibility
    - Graceful lifecycle management with proper cleanup
    - Performance optimization through component caching
    """

    def __init__(
        self,
        config: CodeWeaverConfig | None = None,
        extensibility_config: ExtensibilityConfig | None = None,
    ):
        """Initialize the extensible code embeddings server.

        Args:
            config: Optional configuration object. If None, loads from default sources.
            extensibility_config: Optional extensibility-specific configuration.
        """
        # Load configuration
        self.config = config or get_config()

        # Create extensibility manager
        self.extensibility_manager = ExtensibilityManager(
            config=self.config, extensibility_config=extensibility_config
        )

        # Component instances (populated lazily)
        self._components: dict[str, Any] = {}
        self._initialized = False

        # Initialize components that don't depend on factories
        self.chunker = AstGrepChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )
        self.structural_search = AstGrepStructuralSearch()
        self.task_coordinator = TaskSearchCoordinator(task_tool_available=True)
        self.file_watcher_manager = FileWatcherManager(self.config)

    async def _ensure_initialized(self) -> None:
        """Ensure the server is properly initialized."""
        if self._initialized:
            return

        logger.info("Initializing extensible code embeddings server")

        # Initialize extensibility manager
        await self.extensibility_manager.initialize()

        # Get components through the extensibility system
        self._components["backend"] = await self.extensibility_manager.get_backend()
        self._components[
            "embedding_provider"
        ] = await self.extensibility_manager.get_embedding_provider()
        self._components[
            "reranking_provider"
        ] = await self.extensibility_manager.get_reranking_provider()
        self._components["rate_limiter"] = self.extensibility_manager.get_rate_limiter()

        # Create legacy compatibility adapters
        compatibility_adapter = LegacyCompatibilityAdapter(self.extensibility_manager)
        self._components["qdrant"] = await compatibility_adapter.get_qdrant_client()
        self._components["embedder"] = await compatibility_adapter.get_embedder()
        self._components["reranker"] = await compatibility_adapter.get_reranker()

        # Ensure collection exists
        await self._ensure_collection()

        self._initialized = True
        logger.info("Extensible server initialization complete")

    async def _ensure_collection(self) -> None:
        """Ensure the vector collection exists."""
        try:
            qdrant = self._components["qdrant"]
            collections = await qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]

            collection_name = self.config.qdrant.collection_name
            if collection_name not in collection_names:
                logger.info("Creating collection: %s", collection_name)
                embedder = self._components["embedder"]
                await qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embedder.dimension, distance=Distance.COSINE),
                )
            else:
                logger.info("Collection %s already exists", collection_name)
        except Exception:
            logger.exception("Error ensuring collection")
            raise

    # Provide compatibility properties for existing methods
    @property
    async def qdrant(self):
        """Get Qdrant client for backward compatibility."""
        await self._ensure_initialized()
        return self._components["qdrant"]

    @property
    async def embedder(self):
        """Get embedder for backward compatibility."""
        await self._ensure_initialized()
        return self._components["embedder"]

    @property
    async def reranker(self):
        """Get reranker for backward compatibility."""
        await self._ensure_initialized()
        return self._components["reranker"]

    @property
    async def rate_limiter(self):
        """Get rate limiter for backward compatibility."""
        await self._ensure_initialized()
        return self._components["rate_limiter"]

    # Delegate all the core methods to the legacy implementation for now
    # This ensures 100% compatibility while using the new architecture under the hood

    async def index_codebase(
        self, root_path: str, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        """Index a codebase for semantic search using ast-grep."""
        await self._ensure_initialized()

        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        # Use file filter for intelligent file discovery
        file_filter = FileFilter(self.config, root)
        files = file_filter.find_files(patterns)

        logger.info("Found %d files to index after filtering", len(files))

        # Process files in batches
        batch_size = self.config.indexing.batch_size
        total_chunks = 0
        processed_languages = set()

        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            batch_chunks = []

            for file_path in batch_files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

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
                    logger.warning("Error processing %s: %s", file_path, e)
                    continue

            if batch_chunks:
                await self._index_chunks(batch_chunks)
                total_chunks += len(batch_chunks)
                logger.info("Indexed batch %d: %d chunks", i // batch_size + 1, len(batch_chunks))

        # Set up file watching if enabled
        watcher_started = False
        if self.config.indexing.enable_auto_reindex:
            watcher_started = self.file_watcher_manager.add_watcher(
                root_path=root, reindex_callback=self._handle_file_changes
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
                "started": watcher_started,
            },
            "extensibility_info": {
                "server_type": "extensible",
                "components_info": self.extensibility_manager.get_component_info(),
            },
        }

    async def _handle_file_changes(self, changed_files: list[Path]):
        """Handle file changes for auto-reindexing."""
        logger.info("Handling changes for %d files", len(changed_files))
        # This is a placeholder - implement actual reindexing logic as needed

    async def _index_chunks(self, chunks: list[CodeChunk]):
        """Index a batch of code chunks."""
        if not chunks:
            return

        # Generate embeddings using the factory-created embedder
        embedder = self._components["embedder"]
        texts = [chunk.content for chunk in chunks]
        embeddings = await embedder.embed_documents(texts)

        # Create points for Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            point = PointStruct(
                id=hash(f"{chunk.file_path}:{chunk.start_line}:{chunk.hash}") & ((1 << 63) - 1),
                vector=embedding,
                payload=chunk.to_metadata(),
            )
            points.append(point)

        # Upload to Qdrant using the factory-created client
        qdrant = self._components["qdrant"]
        await qdrant.upsert(collection_name=self.config.qdrant.collection_name, points=points)

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        *,
        file_filter: str | None = None,
        language_filter: str | None = None,
        chunk_type_filter: str | None = None,
        rerank: bool = True,
        use_task_delegation: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code using semantic similarity with advanced filtering."""
        await self._ensure_initialized()

        # Assess search complexity
        estimated_files = await self._estimate_file_count(file_filter, language_filter)
        assessment = self.task_coordinator.assess_search_complexity(
            query=query,
            file_filter=file_filter,
            language_filter=language_filter,
            estimated_files=estimated_files,
        )

        # Log assessment for transparency
        logger.info(
            "Search assessment: %s complexity, confidence: %.2f, scope: %s",
            assessment.complexity.value,
            assessment.confidence,
            assessment.estimated_scope,
        )

        # Return assessment info along with results
        if assessment.should_use_task and use_task_delegation is not False:
            # Include recommendation in results
            return [
                {
                    "type": "search_recommendation",
                    "message": f"This search has {assessment.complexity.value} complexity. "
                    f"Consider using Task tool for comprehensive results.",
                    "assessment": {
                        "complexity": assessment.complexity.value,
                        "confidence": assessment.confidence,
                        "scope": assessment.estimated_scope,
                        "reasoning": assessment.reasoning,
                        "task_prompt": self.task_coordinator.create_task_prompt_for_semantic_search(
                            query=query,
                            limit=limit,
                            file_filter=file_filter,
                            language_filter=language_filter,
                            chunk_type_filter=chunk_type_filter,
                            rerank=rerank,
                        ),
                    },
                }
            ]

        # Generate query embedding
        embedder = self._components["embedder"]
        query_vector = await embedder.embed_query(query)

        # Build filters
        filter_conditions = []

        if file_filter:
            filter_conditions.append(
                FieldCondition(key="file_path", match=MatchValue(value=file_filter))
            )

        if language_filter:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language_filter))
            )

        if chunk_type_filter:
            filter_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter))
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Search using the factory-created backend
        qdrant = self._components["qdrant"]
        search_result = await qdrant.search(
            collection_name=self.config.qdrant.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit * 2 if rerank else limit,  # Get more for reranking
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
                "similarity_score": hit.score,
            }
            results.append(result)

        # Rerank if requested
        if rerank and len(results) > 1:
            try:
                reranker = self._components["reranker"]
                if reranker:
                    documents = [r["content"] for r in results]
                    rerank_results = await reranker.rerank(query, documents, top_k=limit)

                    # Reorder results based on reranking
                    reranked = []
                    for rerank_item in rerank_results:
                        original_result = results[rerank_item["index"]]
                        original_result["rerank_score"] = rerank_item["relevance_score"]
                        reranked.append(original_result)

                    results = reranked
            except Exception as e:
                logger.warning("Reranking failed, using similarity search only: %s", e)

        return results[:limit]

    async def ast_grep_search(
        self,
        pattern: str,
        language: str,
        root_path: str,
        limit: int = 20,
        *,
        use_task_delegation: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        if not AST_GREP_AVAILABLE:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")

        # Assess if this is a comprehensive search
        root = Path(root_path)
        is_large_scope = root.is_dir() and len(list(root.rglob("*"))) > 100
        is_complex_pattern = len(pattern) > 50 or pattern.count("$") > 3

        if (is_large_scope or is_complex_pattern) and use_task_delegation is not False:
            # Suggest Task tool for comprehensive structural search
            return [
                {
                    "type": "search_recommendation",
                    "message": "This structural search covers a large scope or complex pattern. "
                    "Consider using Task tool for comprehensive results.",
                    "assessment": {
                        "pattern_complexity": "complex" if is_complex_pattern else "simple",
                        "scope": f"{'large' if is_large_scope else 'moderate'} codebase at {root_path}",
                        "task_prompt": self.task_coordinator.create_task_prompt_for_structural_search(
                            pattern=pattern, language=language, root_path=root_path, limit=limit
                        ),
                    },
                }
            ]

        results = await self.structural_search.structural_search(
            pattern=pattern, language=language, root_path=root_path
        )

        return results[:limit]

    async def get_supported_languages(self) -> dict[str, Any]:
        """Get information about supported languages and capabilities."""
        await self._ensure_initialized()

        # Get base capabilities
        base_info = {
            "ast_grep_available": AST_GREP_AVAILABLE,
            "supported_languages": list(set(self.chunker.SUPPORTED_LANGUAGES.values())),
            "language_extensions": self.chunker.SUPPORTED_LANGUAGES,
            "chunk_patterns": {
                lang: [pattern[1] for pattern in patterns]
                for lang, patterns in self.chunker.CHUNK_PATTERNS.items()
            },
            "voyage_models": {"embedding": "voyage-code-3", "reranker": "voyage-rerank-2"},
        }

        # Add extensibility information
        component_info = self.extensibility_manager.get_component_info()
        base_info.update({
            "extensibility": {
                "server_type": "extensible",
                "available_backends": component_info.get("backends", {}),
                "available_providers": component_info.get("providers", {}),
                "available_sources": component_info.get("sources", {}),
                "plugin_support": component_info.get("plugin_discovery", {}).get("enabled", False),
            }
        })

        return base_info

    async def _estimate_file_count(
        self, file_filter: str | None = None, language_filter: str | None = None
    ) -> int:
        """Estimate number of files that would be searched."""
        try:
            # Get collection statistics if available
            qdrant = self._components["qdrant"]
            collection_info = await qdrant.get_collection(self.config.qdrant.collection_name)
            total_points = collection_info.points_count

            # Rough estimate based on average chunks per file
            estimated_files = total_points // 5  # Assume ~5 chunks per file on average

            # Adjust based on filters
            if file_filter:
                # Specific path reduces scope significantly
                estimated_files = estimated_files // 10
            if language_filter:
                # Language filter typically reduces to ~20% of files
                estimated_files = estimated_files // 5

            return max(10, estimated_files)  # At least 10 files
        except Exception:
            # Default estimate if we can't get collection info
            return 100

    async def shutdown(self) -> None:
        """Gracefully shutdown the server and cleanup resources."""
        logger.info("Shutting down extensible code embeddings server")

        # Shutdown extensibility manager
        await self.extensibility_manager.shutdown()

        # Clear component cache
        self._components.clear()
        self._initialized = False

        logger.info("Extensible server shutdown complete")


# Server Factory Functions


def detect_configuration_type(config: CodeWeaverConfig) -> str:
    """Detect whether configuration is legacy or extensible format.

    Args:
        config: Configuration object to analyze

    Returns:
        'legacy' or 'extensible' based on configuration structure
    """
    # Check for new extensibility features
    if _EXTENDED_CONFIGS_AVAILABLE:
        # Check if backend config exists and is not just legacy qdrant
        if hasattr(config, "backend") and config.backend:
            if hasattr(config.backend, "provider") and config.backend.provider != "qdrant_legacy":
                return "extensible"

        # Check for new provider configuration format
        if hasattr(config, "embedding") and config.embedding:
            if hasattr(config.embedding, "provider_config"):
                return "extensible"

        # Check for data sources configuration
        if hasattr(config, "data_sources") and config.data_sources:
            return "extensible"

    # Default to legacy for backward compatibility
    return "legacy"


def create_server(
    config: CodeWeaverConfig | None = None,
    server_type: str = "auto",
    extensibility_config: ExtensibilityConfig | None = None,
) -> CodeEmbeddingsServer | ExtensibleCodeEmbeddingsServer:
    """Create a CodeWeaver server instance with automatic type detection.

    Args:
        config: Optional configuration object
        server_type: 'auto', 'legacy', or 'extensible'
        extensibility_config: Optional extensibility configuration

    Returns:
        Appropriate server instance based on configuration
    """
    # Load config if not provided
    if config is None:
        config = get_config()

    # Auto-detect server type if requested
    if server_type == "auto":
        detected_type = detect_configuration_type(config)
        logger.info("Auto-detected server type: %s", detected_type)
        server_type = detected_type

    # Create appropriate server instance
    if server_type == "extensible":
        logger.info("Creating extensible server with factory architecture")
        return ExtensibleCodeEmbeddingsServer(config, extensibility_config)
    logger.info("Creating legacy server for backward compatibility")
    return CodeEmbeddingsServer(config)


def create_legacy_server(config: CodeWeaverConfig | None = None) -> CodeEmbeddingsServer:
    """Create a legacy CodeEmbeddingsServer instance.

    Args:
        config: Optional configuration object

    Returns:
        Legacy CodeEmbeddingsServer instance
    """
    return CodeEmbeddingsServer(config)


def create_extensible_server(
    config: CodeWeaverConfig | None = None, extensibility_config: ExtensibilityConfig | None = None
) -> ExtensibleCodeEmbeddingsServer:
    """Create an ExtensibleCodeEmbeddingsServer instance.

    Args:
        config: Optional configuration object
        extensibility_config: Optional extensibility configuration

    Returns:
        ExtensibleCodeEmbeddingsServer instance
    """
    return ExtensibleCodeEmbeddingsServer(config, extensibility_config)


async def migrate_config_to_extensible(
    legacy_config: CodeWeaverConfig, enable_plugins: bool = False
) -> tuple[CodeWeaverConfig, ExtensibilityConfig]:
    """Migrate a legacy configuration to extensible format.

    Args:
        legacy_config: Legacy configuration to migrate
        enable_plugins: Whether to enable plugin discovery

    Returns:
        Tuple of (migrated_config, extensibility_config)
    """
    logger.info("Migrating legacy configuration to extensible format")

    # Create extensibility config for migration
    extensibility_config = create_migration_config(
        enable_plugins=enable_plugins, enable_legacy_fallback=True, lazy_init=True
    )

    # For now, return the original config with extensibility config
    # In a full implementation, you would transform the config structure
    migrated_config = legacy_config

    # If extended configs are available, perform actual migration
    if _EXTENDED_CONFIGS_AVAILABLE:
        try:
            # Import the extended configuration classes
            from codeweaver.backends.config import create_backend_config_from_legacy
            from codeweaver.sources.config import extend_config_with_data_sources

            # Create backend config from legacy qdrant config
            if hasattr(legacy_config, "qdrant") and legacy_config.qdrant:
                backend_config = create_backend_config_from_legacy(legacy_config.qdrant)

                # Update the config (in a real implementation, you'd create a new config object)
                migrated_config.backend = backend_config
                logger.info("Migrated Qdrant config to backend config")

            # Extend with data sources if needed
            migrated_config = extend_config_with_data_sources(migrated_config)

        except ImportError:
            logger.warning("Extended configuration classes not available, using legacy fallback")

    logger.info("Configuration migration complete")
    return migrated_config, extensibility_config


class ServerMigrationManager:
    """
    Manages the migration of existing server instances to the new architecture.

    Provides utilities for:
    - Analyzing configuration readiness
    - Performing incremental migrations
    - Validating migration success
    - Rolling back failed migrations
    """

    def __init__(self, server: CodeEmbeddingsServer):
        """Initialize migration manager with a server instance.

        Args:
            server: The server instance to migrate
        """
        self.server = server
        self._backup_components: dict[str, Any] = {}
        self._migration_state = "not_started"

    def analyze_migration_readiness(self) -> dict[str, Any]:
        """Analyze if the server is ready for migration.

        Returns:
            Analysis results with readiness status and recommendations
        """
        logger.info("Analyzing migration readiness for server instance")

        # Validate configuration
        config_validation = validate_migration_readiness(self.server.config)

        # Check component health
        component_health = {
            "qdrant_connected": hasattr(self.server, "qdrant") and self.server.qdrant is not None,
            "embedder_available": hasattr(self.server, "embedder")
            and self.server.embedder is not None,
            "reranker_available": hasattr(self.server, "reranker")
            and self.server.reranker is not None,
        }

        # Determine overall readiness
        overall_ready = (
            config_validation["ready"]
            and component_health["qdrant_connected"]
            and component_health["embedder_available"]
        )

        results = {
            "ready": overall_ready,
            "migration_state": self._migration_state,
            "configuration": config_validation,
            "component_health": component_health,
            "recommendations": [],
        }

        # Add recommendations
        if overall_ready:
            results["recommendations"].append(
                "Server is ready for migration. Use perform_migration() to proceed."
            )
        else:
            if not config_validation["ready"]:
                results["recommendations"].append("Fix configuration issues before migration")
            if not component_health["qdrant_connected"]:
                results["recommendations"].append("Ensure Qdrant connection is established")
            if not component_health["embedder_available"]:
                results["recommendations"].append("Ensure embedder is properly initialized")

        return results

    async def perform_migration(
        self,
        extensibility_config: ExtensibilityConfig | None = None,
        backup_components: bool = True,
    ) -> dict[str, Any]:
        """Perform the actual migration to extensible architecture.

        Args:
            extensibility_config: Optional extensibility configuration
            backup_components: Whether to backup current components

        Returns:
            Migration results and status
        """
        logger.info("Starting server migration to extensible architecture")
        self._migration_state = "in_progress"

        try:
            # Backup current components if requested
            if backup_components:
                self._backup_current_components()

            # Create migration helper
            from codeweaver.factories.integration import ServerMigrationHelper

            migration_helper = ServerMigrationHelper(self.server)

            # Perform the migration
            await migration_helper.migrate_to_factories()

            self._migration_state = "completed"
            logger.info("Server migration completed successfully")

            return {
                "status": "success",
                "migration_state": self._migration_state,
                "message": "Server successfully migrated to extensible architecture",
                "components_migrated": list(self._backup_components.keys())
                if backup_components
                else [],
            }

        except Exception as e:
            logger.exception("Migration failed: %s", e)
            self._migration_state = "failed"

            # Attempt rollback if backup exists
            if backup_components and self._backup_components:
                await self._rollback_components()

            return {
                "status": "failed",
                "migration_state": self._migration_state,
                "error": str(e),
                "message": "Migration failed, components restored from backup",
            }

    def _backup_current_components(self) -> None:
        """Backup current server components."""
        logger.info("Backing up current server components")

        backup_attrs = ["qdrant", "embedder", "reranker", "rate_limiter"]
        for attr in backup_attrs:
            if hasattr(self.server, attr):
                self._backup_components[attr] = getattr(self.server, attr)

        logger.info("Backed up %d components", len(self._backup_components))

    async def _rollback_components(self) -> None:
        """Rollback to backed up components."""
        logger.info("Rolling back to backed up components")

        for attr, component in self._backup_components.items():
            setattr(self.server, attr, component)

        self._migration_state = "rolled_back"
        logger.info("Rollback completed")

    def get_migration_status(self) -> dict[str, Any]:
        """Get current migration status.

        Returns:
            Current migration state and statistics
        """
        return {
            "migration_state": self._migration_state,
            "backup_available": bool(self._backup_components),
            "backed_up_components": list(self._backup_components.keys()),
            "server_type": type(self.server).__name__,
        }
