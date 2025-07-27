# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Clean CodeWeaver server implementation using plugin system and FastMCP middleware.

This is the new clean server implementation that integrates:
- FastMCP middleware stack (chunking, filtering, rate limiting, logging)
- Enhanced plugin system with factory pattern
- New Pydantic models and types
- Integrated FilesystemSource with AST-grep support
"""

import logging

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import Context, FastMCP

from codeweaver._types import ContentSearchResult, ExtensibilityConfig
from codeweaver.factories.extensibility_manager import ExtensibilityManager

# CodeWeaver domain-specific middleware
from codeweaver.middleware import ChunkingMiddleware, FileFilteringMiddleware
from codeweaver.services.manager import ServicesManager


if TYPE_CHECKING:
    from codeweaver.config import CodeWeaverConfig


logger = logging.getLogger(__name__)


class CodeWeaverServer:
    """Clean CodeWeaver server using plugin system and FastMCP middleware.

    Features:
    - FastMCP middleware integration for cross-cutting concerns
    - Plugin system for extensible backends, providers, and sources
    - Clean architecture with no legacy compatibility layers
    - Configuration-driven initialization
    - Factory pattern for component creation
    """

    def __init__(
        self,
        config: "CodeWeaverConfig | None" = None,
        extensibility_config: ExtensibilityConfig | None = None,
    ):
        """Initialize the clean CodeWeaver server.

        Args:
            config: Main configuration object
            extensibility_config: Extensibility configuration
        """
        # Load configuration - import at runtime to avoid circular import
        from codeweaver.config import get_config

        self.config = config or get_config()

        # Create FastMCP server instance
        self.mcp = FastMCP("CodeWeaver")

        # Initialize plugin system
        self.extensibility_manager = ExtensibilityManager(
            config=self.config, extensibility_config=extensibility_config
        )

        # Component instances (populated during initialization)
        self._components: dict[str, Any] = {}
        self._initialized = False

        # Services manager (created during initialization)
        self.services_manager: ServicesManager | None = None

    async def initialize(self) -> None:
        """Initialize server with FastMCP middleware and plugin system."""
        if self._initialized:
            logger.warning("Server already initialized")
            return

        logger.info("Initializing clean CodeWeaver server")

        # Initialize plugin system first
        await self.extensibility_manager.initialize()

        # NEW: Create and initialize services manager with FastMCP server reference
        self.services_manager = ServicesManager(
            config=self.config.services,
            fastmcp_server=self.mcp
        )
        await self.services_manager.initialize()

        # Setup domain-specific middleware (chunking, filtering)
        await self._setup_domain_middleware()

        # Get plugin system components
        await self._initialize_components()

        # Register MCP tools
        self._register_tools()

        self._initialized = True
        logger.info("Clean CodeWeaver server initialization complete")

    async def _setup_domain_middleware(self) -> None:
        """Setup domain-specific middleware (chunking, filtering)."""
        # NOTE: FastMCP builtin middleware now handled by ServicesManager
        # Only setup domain-specific middleware here

        logger.info("Setting up domain-specific middleware")

        # Chunking middleware (using existing chunking config)
        chunking_config = {
            "max_chunk_size": self.config.chunking.max_chunk_size,
            "min_chunk_size": self.config.chunking.min_chunk_size,
            "ast_grep_enabled": True,
        }
        chunking_middleware = ChunkingMiddleware(chunking_config)
        self.mcp.add_middleware(chunking_middleware)
        logger.info("Added chunking middleware")

        # File filtering middleware (using existing indexing config)
        filtering_config = {
            "use_gitignore": self.config.indexing.use_gitignore,
            "max_file_size": f"{self.config.chunking.max_file_size_mb}MB",
            "excluded_dirs": self.config.indexing.additional_ignore_patterns,
            "included_extensions": None,
        }
        filtering_middleware = FileFilteringMiddleware(filtering_config)
        self.mcp.add_middleware(filtering_middleware)
        logger.info("Added file filtering middleware")

        logger.info("Domain-specific middleware setup complete")

    async def _initialize_components(self) -> None:
        """Initialize plugin system components."""
        logger.info("Initializing plugin system components")

        # Get components through the extensibility system
        self._components["backend"] = await self.extensibility_manager.get_backend()
        self._components[
            "embedding_provider"
        ] = await self.extensibility_manager.get_embedding_provider()
        self._components[
            "reranking_provider"
        ] = await self.extensibility_manager.get_reranking_provider()

        # Note: Rate limiting is handled by FastMCP's RateLimitingMiddleware
        # No need for a separate rate_limiter component

        # Get data sources and find filesystem source
        data_sources = await self.extensibility_manager.get_data_sources()
        filesystem_source = next(
            (
                source
                for source in data_sources
                if hasattr(source, "provider") and source.provider.value == "filesystem"
            ),
            None,
        )
        if not filesystem_source:
            # Create a fallback filesystem source
            from codeweaver.sources.filesystem import FileSystemSource

            filesystem_source = FileSystemSource()

        self._components["filesystem_source"] = filesystem_source

        # Ensure vector collection exists
        await self._ensure_collection()

        logger.info("Plugin system components initialized")

    async def _ensure_collection(self) -> None:
        """Ensure the vector collection exists."""
        backend = self._components["backend"]
        embedding_provider = self._components["embedding_provider"]
        collection_name = self.config.backend.collection_name

        try:
            collections = await backend.list_collections()
            if collection_name not in collections:
                logger.info("Creating collection: %s", collection_name)
                await backend.create_collection(
                    name=collection_name,
                    dimension=embedding_provider.dimension,
                    distance_metric="cosine",
                )
            else:
                logger.info("Collection %s already exists", collection_name)
        except Exception:
            logger.exception("Error ensuring collection")
            raise

    async def _index_codebase_handler(self, ctx: Context, path: str) -> dict[str, Any]:
        """Index a codebase using middleware services and plugin system."""
        # Get filesystem source from plugin system
        filesystem_source = self._components["filesystem_source"]
        backend = self._components["backend"]
        embedding_provider = self._components["embedding_provider"]

        # Create context with services for source to use
        source_context = {
            "chunking_service": self.services_manager.get_chunking_service() if self.services_manager else None,
            "filtering_service": self.services_manager.get_filtering_service() if self.services_manager else None,
        }

        # Index using enhanced filesystem source
        chunks = await filesystem_source.index_content(Path(path), source_context)

        if chunks:
            # Generate embeddings
            embeddings = await embedding_provider.embed_batch([c.content for c in chunks])

            # Store in backend
            vector_points = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                vector_point = {
                    "id": hash(chunk.unique_id) & ((1 << 63) - 1),
                    "vector": embedding,
                    "payload": chunk.to_metadata(),
                }
                vector_points.append(vector_point)

            await backend.upsert_vectors(self.config.backend.collection_name, vector_points)

        return {
            "status": "success",
            "indexed_chunks": len(chunks),
            "collection": self.config.backend.collection_name,
            "services_used": {
                "chunking": source_context["chunking_service"] is not None,
                "filtering": source_context["filtering_service"] is not None,
                "middleware_services": list(self.services_manager.list_middleware_services().keys()) if self.services_manager else [],
            },
        }

    def _register_tools(self) -> None:
        """Register MCP tools with FastMCP server."""
        logger.info("Registering MCP tools")

        @self.mcp.tool()
        async def index_codebase(ctx: Context, path: str) -> dict[str, Any]:
            """Index a codebase using middleware services and plugin system."""
            return await self._index_codebase_handler(ctx, path)

        @self.mcp.tool()
        async def search_code(
            ctx: Context,
            query: str,
            limit: int = 10,
            file_filter: str | None = None,
            language_filter: str | None = None,
            chunk_type_filter: str | None = None,
            *,
            rerank: bool = True,
        ) -> list[dict[str, Any]]:
            """Search code using plugin system components."""
            return await self._search_code_handler(
                ctx, query, limit, file_filter, language_filter, chunk_type_filter, rerank=rerank
            )

        @self.mcp.tool()
        async def ast_grep_search(
            ctx: Context, pattern: str, language: str, root_path: str, limit: int = 20
        ) -> list[dict[str, Any]]:
            """Perform structural search using ast-grep patterns."""
            return await self._ast_grep_search_handler(ctx, pattern, language, root_path, limit)

        @self.mcp.tool()
        async def get_supported_languages(ctx: Context) -> dict[str, Any]:
            """Get information about supported languages and capabilities."""
            return await self._get_supported_languages_handler(ctx)

        logger.info("MCP tools registered successfully")

    async def _search_code_handler(
        self,
        ctx: Context,
        query: str,
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None,
        chunk_type_filter: str | None = None,
        *,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Search code using plugin system components."""
        # Get components
        backend = self._components["backend"]
        embedding_provider = self._components["embedding_provider"]
        reranking_provider = self._components["reranking_provider"]

        # Generate query embedding
        query_vector = await embedding_provider.embed_query(query)

        # Build search filters
        search_filter = self._build_search_filters(file_filter, language_filter, chunk_type_filter)

        # Search vectors
        search_results = await backend.search_vectors(
            collection_name=self.config.backend.collection_name,
            query_vector=query_vector,
            search_filter=search_filter,
            limit=limit * 2 if rerank else limit,  # Get more for reranking
        )

        # Convert to SearchResult objects
        results = self._convert_search_results(search_results)

        # Apply reranking if requested
        if rerank and reranking_provider and len(results) > 1:
            results = await self._apply_reranking(query, results, reranking_provider, limit)

        return [r.to_dict() for r in results[:limit]]

    def _build_search_filters(
        self, file_filter: str | None, language_filter: str | None, chunk_type_filter: str | None
    ) -> dict[str, Any] | None:
        """Build search filter conditions."""
        filter_conditions = []

        if file_filter:
            filter_conditions.append({
                "field": "file_path",
                "operator": "contains",
                "value": file_filter,
            })
        if language_filter:
            filter_conditions.append({
                "field": "language",
                "operator": "eq",
                "value": language_filter,
            })
        if chunk_type_filter:
            filter_conditions.append({
                "field": "chunk_type",
                "operator": "eq",
                "value": chunk_type_filter,
            })

        return {"conditions": filter_conditions} if filter_conditions else None

    def _convert_search_results(self, search_results: list[Any]) -> list[ContentSearchResult]:
        """Convert backend search results to ContentSearchResult objects."""
        results = []
        for hit in search_results:
            payload = hit.payload
            search_result = ContentSearchResult(
                content=payload["content"],
                file_path=payload["file_path"],
                start_line=payload["start_line"],
                end_line=payload["end_line"],
                chunk_type=payload["chunk_type"],
                language=payload["language"],
                node_kind=payload.get("node_kind", ""),
                similarity_score=hit.score,
            )
            results.append(search_result)
        return results

    async def _apply_reranking(
        self, query: str, results: list[ContentSearchResult], reranking_provider: Any, limit: int
    ) -> list[ContentSearchResult]:
        """Apply reranking to search results."""
        try:
            documents = [r.content for r in results]
            rerank_results = await reranking_provider.rerank(query, documents, top_k=limit)

            # Reorder results based on reranking
            reranked = []
            for rerank_result in rerank_results:
                original_result = results[rerank_result.index]
                original_result.rerank_score = rerank_result.relevance_score
                reranked.append(original_result)

        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return results
        else:
            return reranked

    async def _ast_grep_search_handler(
        self, ctx: Context, pattern: str, language: str, root_path: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        # Get filesystem source
        filesystem_source = self._components["filesystem_source"]

        # Create context with services
        source_context = {
            "filtering_service": self.services_manager.get_filtering_service() if self.services_manager else None
        }

        # Perform structural search using enhanced filesystem source
        results = await filesystem_source.structural_search(
            pattern=pattern, language=language, root_path=Path(root_path), context=source_context
        )

        return results[:limit]

    async def _get_supported_languages_handler(self, ctx: Context) -> dict[str, Any]:
        """Get information about supported languages and capabilities."""
        chunking_info = {}
        filtering_info = {}
        middleware_info = {}

        if self.services_manager:
            # Get service information
            if chunking_service := self.services_manager.get_chunking_service():
                chunking_info = chunking_service.get_supported_languages()
            if filtering_service := self.services_manager.get_filtering_service():
                filtering_info = await filtering_service.get_filtering_stats()

            # Get middleware service information
            middleware_services = self.services_manager.list_middleware_services()
            middleware_info = {str(service_type): service.name for service_type, service in middleware_services.items()}

        # Get plugin system component info
        component_info = self.extensibility_manager.get_component_info()

        return {
            "server_type": "services_integrated",
            "chunking": chunking_info,
            "filtering": filtering_info,
            "middleware_services": middleware_info,
            "extensibility": component_info,
        }

    async def run(self) -> None:
        """Run the server."""
        await self.initialize()
        await self.mcp.run()

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Shutting down clean CodeWeaver server")

        # Shutdown services manager
        if self.services_manager:
            await self.services_manager.shutdown()

        # Shutdown extensibility manager
        await self.extensibility_manager.shutdown()

        # Clear component cache
        self._components.clear()
        self._initialized = False

        logger.info("Clean server shutdown complete")


def create_clean_server(
    config: "CodeWeaverConfig | None" = None,
    extensibility_config: ExtensibilityConfig | None = None,
) -> CodeWeaverServer:
    """Create a clean CodeWeaver server instance.

    Args:
        config: Optional configuration object
        extensibility_config: Optional extensibility configuration

    Returns:
        Clean server instance
    """
    return CodeWeaverServer(config, extensibility_config)
