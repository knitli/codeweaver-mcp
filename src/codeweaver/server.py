# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Main MCP server implementation for code embeddings and search.

Orchestrates semantic search, chunking, and indexing functionality
using configurable embedding providers and vector database backends.
"""

import logging

from pathlib import Path
from typing import Any

from codeweaver.backends.base import DistanceMetric, VectorBackend, VectorPoint
from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.chunker import AST_GREP_AVAILABLE, AstGrepChunker
from codeweaver.config import CodeWeaverConfig, get_config
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.file_filter import FileFilter
from codeweaver.file_watcher import FileWatcherManager
from codeweaver.models import CodeChunk
from codeweaver.providers import RerankProvider, get_provider_factory
from codeweaver.rate_limiter import RateLimiter
from codeweaver.search import AstGrepStructuralSearch
from codeweaver.task_search import TaskSearchCoordinator


logger = logging.getLogger(__name__)


class BackendNotAvailableError(Exception):
    """Custom exception for when the backend is not available."""

    def __init__(self, message: str):
        """Initialize with a custom message."""
        super().__init__(message)
        self.message = "Backend not available: %s", message


def raise_if_backend_none() -> None:
    """Raise an error if the backend is not available."""
    raise BackendNotAvailableError("Backend not available, using fallback")


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

        except Exception as e:
            logger.warning("Failed to initialize new provider system, using legacy: %s", e)

        self.chunker = AstGrepChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )
        self.structural_search = AstGrepStructuralSearch()

        # Initialize Task search coordinator
        self.task_coordinator = TaskSearchCoordinator(task_tool_available=True)

        # Initialize file watcher manager
        self.file_watcher_manager = FileWatcherManager(self.config)

        # Initialize vector backend using factory pattern
        self.backend = self._initialize_backend()
        self.collection_name = self.config.qdrant.collection_name

        # Collection creation will happen lazily in async methods when first needed

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
        except Exception:
            logger.exception("Failed to initialize backend")
            raise  # Fail fast - no legacy fallback
        else:
            return backend

    async def _ensure_collection(self):
        """Ensure the vector collection exists using backend abstraction."""
        try:
            # Use backend abstraction for collection management
            collections = await self.backend.list_collections()
            if self.collection_name not in collections:
                logger.info("Creating collection using backend: %s", self.collection_name)
                await self.backend.create_collection(
                    name=self.collection_name,
                    dimension=self.embedder.dimension,
                    distance_metric=DistanceMetric.COSINE,
                )
            else:
                logger.info("Collection %s already exists", self.collection_name)
        except Exception:
            logger.exception("Error ensuring collection")
            raise

    # _ensure_collection_async removed - using unified _ensure_collection method

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
        await self._ensure_collection()

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
        def _raise_backend_error():
            raise RuntimeError("Backend not available, using fallback")

        # Use backend abstraction for vector upload
        await self.backend.upsert_vectors(self.collection_name, vector_points)
        logger.debug("Uploaded %d vectors using backend", len(vector_points))

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
        # Check if task delegation should be used
        task_recommendation = await self._assess_search_complexity(
            query,
            file_filter,
            language_filter,
            use_task_delegation,
            limit,
            chunk_type_filter,
            rerank,
        )
        if task_recommendation:
            return task_recommendation

        # Generate query embedding
        query_vector = await self.embedder.embed_query(query)

        # Perform search with backend/legacy fallback
        search_results = await self._perform_vector_search(
            query_vector, file_filter, language_filter, chunk_type_filter, limit, rerank
        )

        # Process search results
        results = self._process_search_results(search_results)

        # Apply reranking if requested
        if rerank and len(results) > 1:
            results = await self._apply_reranking(query, results, limit)

        return results[:limit]

    async def _assess_search_complexity(
        self,
        query: str,
        file_filter: str | None,
        language_filter: str | None,
        *,
        use_task_delegation: bool | None,
        limit: int,
        chunk_type_filter: str | None,
        rerank: bool,
    ) -> list[dict[str, Any]] | None:
        """Assess search complexity and return task recommendation if needed."""
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

        # Return assessment info if task delegation should be used
        if assessment.should_use_task and use_task_delegation is not False:
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
        return None

    async def _perform_vector_search(
        self,
        query_vector: list[float],
        file_filter: str | None,
        language_filter: str | None,
        chunk_type_filter: str | None,
        limit: int,
        *,
        rerank: bool,
    ) -> list:
        """Perform vector search using backend abstraction with fallback to legacy."""
        # Build universal search filters
        from codeweaver.backends.base import FilterCondition, SearchFilter

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
        def _raise_search_backend_error():
            raise RuntimeError("Backend not available, using fallback")

        try:
            if self.backend is None:
                _raise_search_backend_error()
            search_result = await self.backend.search_vectors(
                collection_name=self.collection_name,
                query_vector=query_vector,
                search_filter=universal_filter,
                limit=limit * 2 if rerank else limit,  # Get more for reranking
            )
            logger.debug("Search using backend returned %d results", len(search_result))
        except Exception as e:
            logger.warning("Backend search failed, using legacy Qdrant client: %s", e)

        return search_result

    def _process_search_results(self, search_result: list) -> list[dict[str, Any]]:
        """Process search results from backend or legacy client."""
        results = []
        for hit in search_result:
            # Handle both backend abstraction SearchResult and legacy Qdrant hits
            if hasattr(hit, "payload") and hit.payload is not None:
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
        return results

    async def _apply_reranking(
        self, query: str, results: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Apply reranking to search results."""
        try:
            documents = [r["content"] for r in results]
            rerank_results = await self.reranker.rerank(query, documents, top_k=limit)

            # Reorder results based on reranking
            reranked = []
            for rerank_item in rerank_results:
                original_result = results[rerank_item["index"]]
                original_result["rerank_score"] = rerank_item["relevance_score"]
                reranked.append(original_result)

        except Exception as e:
            logger.warning("Reranking failed, using similarity search only: %s", e)
            return results
        else:
            return reranked

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
            # Use backend abstraction for collection info
            collection_info = self.backend.get_collection_info(self.collection_name)
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
    """Next-generation MCP server using the extensible factory architecture.

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

        # Ensure collection exists
        await self._ensure_collection()

        self._initialized = True
        logger.info("Extensible server initialization complete")

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
        results = []
        search_results = self.ast_grep_search(
            pattern=query,
            language=language_filter or "all",  # Use 'all' if no specific language
            root_path=self.config.indexing.root_path,
            limit=limit,
            use_task_delegation=use_task_delegation,
        )
        for hit in search_results:
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
                if reranker := self._components["reranker"]:
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
        base_info["extensibility"] = {
            "server_type": "extensible",
            "available_backends": component_info.get("backends", {}),
            "available_providers": component_info.get("providers", {}),
            "available_sources": component_info.get("sources", {}),
            "plugin_support": component_info.get("plugin_discovery", {}).get("enabled", False),
        }

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

    # Create appropriate server instance
    logger.info("Creating extensible server with factory architecture")
    return ExtensibleCodeEmbeddingsServer(config, extensibility_config)


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
