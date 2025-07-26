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

from codeweaver._types import ExtensibilityConfig, SearchResult
from codeweaver.factories.extensibility_manager import ExtensibilityManager
# FastMCP built-in middleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

# CodeWeaver domain-specific middleware
from codeweaver.middleware import (
    ChunkingMiddleware,
    FileFilteringMiddleware,
)


if TYPE_CHECKING:
    from codeweaver.config import CodeWeaverConfig


logger = logging.getLogger(__name__)


class CleanCodeWeaverServer:
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
            config=self.config,
            extensibility_config=extensibility_config,
        )

        # Component instances (populated during initialization)
        self._components: dict[str, Any] = {}
        self._middleware_instances: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize server with FastMCP middleware and plugin system."""
        if self._initialized:
            logger.warning("Server already initialized")
            return

        logger.info("Initializing clean CodeWeaver server")

        # Initialize plugin system first
        await self.extensibility_manager.initialize()

        # Setup FastMCP middleware stack
        await self._setup_middleware()

        # Get plugin system components
        await self._initialize_components()

        # Register MCP tools
        self._register_tools()

        self._initialized = True
        logger.info("Clean CodeWeaver server initialization complete")

    async def _setup_middleware(self) -> None:
        """Setup FastMCP middleware stack."""
        logger.info("Setting up FastMCP middleware stack")

        # Error handling middleware (first in chain for proper error catching)
        error_handler = ErrorHandlingMiddleware(
            include_traceback=False,  # Don't expose internal details
            transform_errors=True  # Convert to MCP errors
        )
        self.mcp.add_middleware(error_handler)
        self._middleware_instances["error_handling"] = error_handler
        logger.info("Added error handling middleware")

        # Rate limiting middleware (using FastMCP built-in)
        # Convert requests_per_minute to requests_per_second
        requests_per_second = 60.0 / 60.0  # 60 req/min = 1 req/sec
        rate_limiter = RateLimitingMiddleware(
            max_requests_per_second=requests_per_second,
            burst_capacity=10,  # Allow short bursts
            global_limit=True  # Apply globally, not per-client
        )
        self.mcp.add_middleware(rate_limiter)
        self._middleware_instances["rate_limiting"] = rate_limiter
        logger.info("Added rate limiting middleware")

        # Logging middleware (using FastMCP built-in)
        import logging as logging_module
        log_level = getattr(logging_module, self.config.server.log_level.upper(), logging_module.INFO)
        log_middleware = LoggingMiddleware(
            log_level=log_level,
            include_payloads=self.config.server.enable_request_logging,
            max_payload_length=1000,
            methods=None  # Log all methods
        )
        self.mcp.add_middleware(log_middleware)
        self._middleware_instances["logging"] = log_middleware
        logger.info("Added logging middleware")

        # Timing middleware (for performance monitoring)
        timing_middleware = TimingMiddleware(log_level=log_level)
        self.mcp.add_middleware(timing_middleware)
        self._middleware_instances["timing"] = timing_middleware
        logger.info("Added timing middleware")

        # Chunking middleware (using existing chunking config)
        chunking_config = {
            "max_chunk_size": self.config.chunking.max_chunk_size,
            "min_chunk_size": self.config.chunking.min_chunk_size,
            "ast_grep_enabled": True,  # Enable by default
        }
        chunking_middleware = ChunkingMiddleware(chunking_config)
        self.mcp.add_middleware(chunking_middleware)
        self._middleware_instances["chunking"] = chunking_middleware
        logger.info("Added chunking middleware")

        # File filtering middleware (using existing indexing config)
        filtering_config = {
            "use_gitignore": self.config.indexing.use_gitignore,
            "max_file_size": f"{self.config.chunking.max_file_size_mb}MB",
            "excluded_dirs": self.config.indexing.additional_ignore_patterns,
            "included_extensions": None,  # Allow all supported extensions
        }
        filtering_middleware = FileFilteringMiddleware(filtering_config)
        self.mcp.add_middleware(filtering_middleware)
        self._middleware_instances["filtering"] = filtering_middleware
        logger.info("Added file filtering middleware")

        logger.info("FastMCP middleware stack setup complete")

    async def _initialize_components(self) -> None:
        """Initialize plugin system components."""
        logger.info("Initializing plugin system components")

        # Get components through the extensibility system
        self._components["backend"] = await self.extensibility_manager.get_backend()
        self._components["embedding_provider"] = await self.extensibility_manager.get_embedding_provider()
        self._components["reranking_provider"] = await self.extensibility_manager.get_reranking_provider()
        self._components["rate_limiter"] = self.extensibility_manager.get_rate_limiter()

        # Get data sources and find filesystem source
        data_sources = await self.extensibility_manager.get_data_sources()
        filesystem_source = None
        for source in data_sources:
            if hasattr(source, 'provider') and source.provider.value == "filesystem":
                filesystem_source = source
                break

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

    def _register_tools(self) -> None:
        """Register MCP tools with FastMCP server."""
        logger.info("Registering MCP tools")

        @self.mcp.tool()
        async def index_codebase(ctx: Context, path: str) -> dict[str, Any]:
            """Index a codebase using middleware services and plugin system."""

            # Get filesystem source from plugin system
            filesystem_source = self._components["filesystem_source"]
            backend = self._components["backend"]
            embedding_provider = self._components["embedding_provider"]

            # Create context with middleware services for source to use
            source_context = {
                "chunking_service": ctx.get_state_value("chunking_service"),
                "filtering_service": ctx.get_state_value("filtering_service"),
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
                "middleware_services_used": {
                    "chunking": ctx.get_state_value("chunking_service") is not None,
                    "filtering": ctx.get_state_value("filtering_service") is not None,
                },
            }

        @self.mcp.tool()
        async def search_code(
            ctx: Context,
            query: str,
            limit: int = 10,
            file_filter: str | None = None,
            language_filter: str | None = None,
            chunk_type_filter: str | None = None,
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
            filter_conditions = []
            if file_filter:
                filter_conditions.append({"field": "file_path", "operator": "contains", "value": file_filter})
            if language_filter:
                filter_conditions.append({"field": "language", "operator": "eq", "value": language_filter})
            if chunk_type_filter:
                filter_conditions.append({"field": "chunk_type", "operator": "eq", "value": chunk_type_filter})

            search_filter = {"conditions": filter_conditions} if filter_conditions else None

            # Search vectors
            search_results = await backend.search_vectors(
                collection_name=self.config.backend.collection_name,
                query_vector=query_vector,
                search_filter=search_filter,
                limit=limit * 2 if rerank else limit,  # Get more for reranking
            )

            # Convert to SearchResult objects
            results = []
            for hit in search_results:
                payload = hit.payload
                search_result = SearchResult(
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

            # Apply reranking if requested
            if rerank and reranking_provider and len(results) > 1:
                try:
                    documents = [r.content for r in results]
                    rerank_results = await reranking_provider.rerank(query, documents, top_k=limit)

                    # Reorder results based on reranking
                    reranked = []
                    for rerank_result in rerank_results:
                        original_result = results[rerank_result.index]
                        original_result.rerank_score = rerank_result.relevance_score
                        reranked.append(original_result)

                    results = reranked
                except Exception as e:
                    logger.warning("Reranking failed: %s", e)

            return [r.to_dict() for r in results[:limit]]

        @self.mcp.tool()
        async def ast_grep_search(
            ctx: Context,
            pattern: str,
            language: str,
            root_path: str,
            limit: int = 20
        ) -> list[dict[str, Any]]:
            """Perform structural search using ast-grep patterns."""

            # Get filesystem source
            filesystem_source = self._components["filesystem_source"]

            # Create context with middleware services
            source_context = {
                "filtering_service": ctx.get_state_value("filtering_service"),
            }

            # Perform structural search using enhanced filesystem source
            results = await filesystem_source.structural_search(
                pattern=pattern,
                language=language,
                root_path=Path(root_path),
                context=source_context,
            )

            return results[:limit]

        @self.mcp.tool()
        async def get_supported_languages(ctx: Context) -> dict[str, Any]:
            """Get information about supported languages and capabilities."""

            # Get chunking middleware info
            chunking_service = ctx.get_state_value("chunking_service")
            chunking_info = {}
            if chunking_service:
                chunking_info = chunking_service.get_supported_languages()

            # Get filtering middleware info
            filtering_service = ctx.get_state_value("filtering_service")
            filtering_info = {}
            if filtering_service:
                filtering_info = filtering_service.get_filtering_stats()

            # Get plugin system component info
            component_info = self.extensibility_manager.get_component_info()

            return {
                "server_type": "clean_plugin_system",
                "chunking": chunking_info,
                "filtering": filtering_info,
                "extensibility": component_info,
                "middleware_stack": list(self._middleware_instances.keys()),
            }

        logger.info("MCP tools registered successfully")

    async def run(self) -> None:
        """Run the server."""
        await self.initialize()
        await self.mcp.run()

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Shutting down clean CodeWeaver server")

        # Shutdown extensibility manager
        await self.extensibility_manager.shutdown()

        # Clear component cache
        self._components.clear()
        self._middleware_instances.clear()
        self._initialized = False

        logger.info("Clean server shutdown complete")


def create_clean_server(
    config: "CodeWeaverConfig | None" = None,
    extensibility_config: ExtensibilityConfig | None = None,
) -> CleanCodeWeaverServer:
    """Create a clean CodeWeaver server instance.

    Args:
        config: Optional configuration object
        extensibility_config: Optional extensibility configuration

    Returns:
        Clean server instance
    """
    return CleanCodeWeaverServer(config, extensibility_config)
