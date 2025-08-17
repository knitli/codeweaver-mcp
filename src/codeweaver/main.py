# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Main FastMCP server implementation for CodeWeaver."""

from __future__ import annotations

import time

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import ConfigDict, Field, NonNegativeInt
from pydantic.dataclasses import dataclass

from codeweaver import __version__ as version
from codeweaver._settings_registry import get_provider_registry
from codeweaver._statistics import SessionStatistics
from codeweaver.exceptions import CodeWeaverError
from codeweaver.models.core import FindCodeResponse
from codeweaver.settings import get_settings
from codeweaver.tools.find_code import find_code_implementation


if TYPE_CHECKING:
    from fastmcp.server.middleware import Middleware

    from codeweaver.language import SemanticSearchLanguage
    from codeweaver.models.intent import IntentType
    from codeweaver.settings import CodeWeaverSettings


@dataclass(
    order=True,
    kw_only=True,
    config=ConfigDict(extra="forbid", str_strip_whitespace=True, arbitrary_types_allowed=True),
)
class AppState:
    """Application state for CodeWeaver server."""

    initialized: Annotated[
        bool, Field(description="Indicates if the server has been initialized")
    ] = False
    config_path: Annotated[
        Path | None, Field(description="Path to the configuration file, if any")
    ] = None
    settings: Annotated[
        CodeWeaverSettings | None, Field(description="CodeWeaver configuration settings")
    ] = None

    # Provider registry integration
    provider_registry: Annotated[
        Any, Field(description="Provider registry for dynamic provider management")
    ] = None

    # Statistics and performance tracking
    statistics: Annotated[
        SessionStatistics, Field(description="Session statistics and performance tracking")
    ] = Field(default_factory=SessionStatistics)

    # Request tracking
    request_count: Annotated[
        NonNegativeInt, Field(description="Total number of requests processed")
    ] = 0

    # Health status
    health: Annotated[dict[str, Any], Field(description="Health status information")] = Field(
        default_factory=dict
    )

    # TODO: Future implementations
    indexer: None = None  # Placeholder for background indexer

    middleware_stack: Annotated[
        tuple[type[Middleware], ...], Field(description="Loaded middleware stack")
    ] = Field(default_factory=tuple)


@asynccontextmanager
async def lifespan(app: FastMCP[AppState]) -> AsyncIterator[AppState]:
    """Context manager for application lifespan with proper initialization."""
    # Initialize application state
    state = AppState(initialized=False)

    try:
        # Load settings
        state.settings = get_settings()

        # Initialize provider registry
        state.provider_registry = get_provider_registry()

        # Initialize health status
        state.health = {
            "status": "initializing",
            "version": version,
            "startup_time": time.time(),
            "features": ["basic_search", "file_discovery", "provider_registry", "statistics"],
        }

        # Mark as initialized
        state.initialized = True
        state.health["status"] = "healthy"

        # Yield the initialized state
        yield state

    except Exception as e:
        # Handle initialization errors
        state.health["status"] = "unhealthy"
        state.health["error"] = str(e)
        state.initialized = False
        raise

    finally:
        # Cleanup resources
        state.initialized = False


# Create FastMCP application
# TODO: Add middleware, dependency injection, etc.
app: FastMCP[AppState] = initialize_app()


@app.tool(tags={"user", "external", "code"})
async def find_code(
    query: str,
    intent: IntentType | None = None,
    token_limit: int = 10000,
    *,
    include_tests: bool = False,
    focus_languages: tuple[SemanticSearchLanguage, ...] | None = None,
    context: Context | None = None,
) -> FindCodeResponse:
    """Intelligently discover and retrieve relevant codebase context.

    Phase 1 implementation provides basic file discovery and keyword-based search.
    Future phases will add semantic embeddings and AI-powered intent analysis.

    Args:
        query: Natural language description of information needed
        intent: Optional hint about the type of task (auto-detected if None)
        token_limit: Maximum tokens to include in response
        include_tests: Whether to include test files in results
        focus_languages: Limit search to specific programming languages
        context: FastMCP context (injected automatically)

    Returns:
        Structured response with relevant code matches and metadata

    Examples:
        query="authentication middleware setup"
        query="database connection pooling configuration"
        query="user registration validation logic"
    """
    try:
        # Execute the find_code implementation
        response = await find_code_implementation(
            query=query,
            settings=settings,
            intent=intent,
            token_limit=token_limit,
            include_tests=include_tests,
            focus_languages=focus_languages,
            statistics=app_state.statistics if app_state else None,
        )

        # Record successful request
        if app_state and app_state.statistics:
            app_state.statistics.add_successful_request()

    except CodeWeaverError:
        # Record failed request
        if app_state and app_state.statistics:
            app_state.statistics.add_failed_request()
        # Re-raise CodeWeaver errors as-is
        raise
    except Exception as e:
        # Record failed request
        if app_state and app_state.statistics:
            app_state.statistics.add_failed_request()

        # Wrap other exceptions in CodeWeaver error
        from codeweaver.exceptions import QueryError

        raise QueryError(
            f"Unexpected error in find_code: {e!s}",
            suggestions=["Try a simpler query", "Check server logs for details"],
        ) from e
    else:
        return response


# TODO: This shouldn't be a tool, but a proper health check endpoint. We can also expose it as a Resource. But not a tool.
@app.tool()
async def health() -> dict[str, Any]:
    """Health check endpoint for monitoring server status.

    Returns:
        Health status information with statistics
    """
    try:
        # Get app state
        app_state: AppState | None = app.state if hasattr(app, "state") else None
        if not app_state:
            return {
                "status": "unhealthy",
                "error": "App state not initialized",
                "version": "0.1.0-phase1",
                "timestamp": time.time(),
            }

        health_info = {
            "status": "healthy",
            "version": "0.1.0-phase1",
            "features": ["basic_search", "file_discovery", "provider_registry", "statistics"],
        } | {
            "initialized": app_state.initialized,
            "request_count": app_state.request_count,
            "uptime_seconds": time.time() - app_state.health.get("startup_time", time.time()),
        }
        # Add statistics if available
        if hasattr(app_state, "statistics"):
            stats = app_state.statistics
            health_info["statistics"] = {
                "total_requests": stats.total_requests or 0,
                "successful_requests": stats.successful_requests or 0,
                "failed_requests": stats.failed_requests or 0,
                "success_rate": stats.get_success_rate(),
                "average_response_time_ms": stats.average_response_time_ms,
                "max_response_time_ms": stats.max_response_time_ms,
                "min_response_time_ms": stats.min_response_time_ms,
                "total_files_indexed": stats.index_statistics.total_unique_files,
                "total_operations": stats.index_statistics.total_operations,
                "token_usage": {
                    "total_generated": stats.token_statistics.total_generated,
                    "total_used": stats.token_statistics.total_used,
                    "context_saved": stats.token_statistics.context_saved,
                },
            }

        # Add provider registry info if available
        if app_state.provider_registry:
            from codeweaver._settings import ProviderKind

            health_info["providers"] = {
                "embedding_providers": len(
                    app_state.provider_registry.list_providers(ProviderKind.EMBEDDING)
                ),
                "vector_store_providers": len(
                    app_state.provider_registry.list_providers(ProviderKind.VECTOR_STORE)
                ),
            }

        if settings := get_settings():
            health_info["project_path"] = str(settings.project_path)

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "version": "0.1.0-phase1",
            "timestamp": time.time(),
        }
    else:
        return health_info


def initialize_app() -> FastMCP[AppState]:
    """Initialize the FastMCP application."""
    base_fast_mcp_settings = {
        "instructions": "Ask a question, describe what you're trying to do, and get the exact context you need. CodeWeaver is an advanced code search and code context tool. It keeps an updated vector, AST, and text index of your codebase, and uses intelligent intent analysis to provide the most relevant context for AI Agents to complete tasks. It's just one easy-to-use tool - the `find_code` tool. To use it, you only need to provide a plain language description of what you want to find, and what you are trying to do. CodeWeaver will return the most relevant code matches, along with their context and precise locations.",
        "version": version,
        "include_tags": {"external", "user", "context"},
        "exclude_tags": {"internal", "system", "admin"},
        "include_fastmcp_meta": True,
        "middleware": [],
    }
    settings = get_settings()

    return app


# Server startup function for CLI
async def start_server(host: str = "localhost", port: int = 8080, *, debug: bool = False) -> None:
    """Start the FastMCP server.

    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    import uvicorn

    # Get app state from context
    app_state = app.state if hasattr(app, "state") else None

    # Update request statistics
    if app_state and app_state.statistics:
        app_state.request_count += 1

    # Get settings (use from app state if available)
    settings = app_state.settings if app_state and app_state.settings else get_settings()

    # TODO: the typechecker doesn't like this and probably neither does uvicorn.
    config = uvicorn.Config(app, host=host, port=port, log_level="debug" if debug else "info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Main entry point for MCP
    import asyncio

    asyncio.run(start_server(debug=True))
