# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Main FastMCP server implementation for CodeWeaver."""

from __future__ import annotations

import asyncio
import time

from typing import TYPE_CHECKING, Any

from fastmcp import Context, FastMCP

from codeweaver._server import initialize_app
from codeweaver.exceptions import CodeWeaverError
from codeweaver.models.core import FindCodeResponse
from codeweaver.tools.find_code import find_code_implementation


if TYPE_CHECKING:
    from codeweaver._server import AppState
    from codeweaver.language import SemanticSearchLanguage
    from codeweaver.models.intent import IntentType


# Create FastMCP application
app: FastMCP[AppState] = asyncio.run(initialize_app())


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
            settings=app.state.settings if app.state else None,
            intent=intent,
            token_limit=token_limit,
            include_tests=include_tests,
            focus_languages=focus_languages,
            statistics=app.state.statistics if app.state else None,
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
            "features": ["basic_search", "file_discovery", "registry", "statistics"],
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
        if app_state.registry:
            from codeweaver._settings import ProviderKind

            health_info["providers"] = {
                "embedding_providers": len(
                    app_state.registry.list_providers(ProviderKind.EMBEDDING)
                ),
                "vector_store_providers": len(
                    app_state.registry.list_providers(ProviderKind.VECTOR_STORE)
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


# Server startup function for CLI
async def start_server(host: str = "localhost", port: int = 8080, *, debug: bool = False) -> None:
    """Start the FastMCP server.

    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    import uvicorn

    # Get settings (use from app state if available)
    settings = app_state.settings if app_state and app_state.settings else get_settings()

    # TODO: the typechecker doesn't like this and probably neither does uvicorn.
    config = uvicorn.Config(app, host=host, port=port, log_level="debug" if debug else "info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Main entry point for MCP
    import asyncio

    asyncio.run(app.run_http_async())
    asyncio.run(start_server(debug=True))
