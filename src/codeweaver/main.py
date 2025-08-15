# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Main FastMCP server implementation for CodeWeaver."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import ConfigDict, Field, NonNegativeInt
from pydantic.dataclasses import dataclass

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
    """
    Application state for CodeWeaver server.
    """

    initialized: Annotated[
        bool, Field(description="Indicates if the server has been initialized")
    ] = False
    config_path: Annotated[
        Path | None, Field(description="Path to the configuration file, if any")
    ] = None
    settings: Annotated[
        CodeWeaverSettings | None, Field(description="CodeWeaver configuration settings")
    ] = get_settings()
    # TODO: Integrate fastmcp middleware here -- error handling, logging, timing, rate_limiting, etc.
    loaded_middleware: Annotated[
        tuple[type[Middleware], ...], Field(description="Tuple of loaded middleware")
    ] = Field(default_factory=tuple)
    indexer: None = None  # Placeholder for future indexer implementation

    statistics: Annotated[SessionStatistics, Field(description="Session statistics tracking")] = (
        SessionStatistics()
    )

    embedding_tokens_generated: Annotated[
        NonNegativeInt, Field(description="Number of tokens generated for embeddings this session.")
    ] = 0

    reranking_tokens_generated: Annotated[
        NonNegativeInt, Field(description="Number of tokens generated for reranking this session.")
    ] = 0
    # TODO: We need a proper health check system -- this doesn't do anything
    health: Annotated[dict[str, Any], Field(description="Health status information")] = Field(
        default_factory=dict
    )


@asynccontextmanager
async def lifespan(app: FastMCP[AppState]) -> AsyncIterator[AppState]:
    """Context manager for application lifespan."""
    state = AppState(initialized=False)
    # TODO: setup application state
    state.initialized = True
    try:
        yield state
    finally:
        # TODO: teardown application state
        state.initialized = False


# Create FastMCP application
# TODO: Add middleware, dependency injection, etc.
app: FastMCP[AppState] = FastMCP("CodeWeaver", lifespan=lifespan)


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
        # TODO: handle context properly and inject it into app state
        settings = get_settings()

        return await find_code_implementation(
            query=query,
            settings=settings,
            intent=intent,
            token_limit=token_limit,
            include_tests=include_tests,
            focus_languages=focus_languages,
        )

    except CodeWeaverError:
        # Re-raise CodeWeaver errors as-is
        raise
    except Exception as e:
        # Wrap other exceptions in CodeWeaver error
        from codeweaver.exceptions import QueryError

        raise QueryError(
            f"Unexpected error in find_code: {e!s}",
            suggestions=["Try a simpler query", "Check server logs for details"],
        ) from e


# TODO: This shouldn't be a tool, but a proper health check endpoint. We can also expose it as a Resource. But not a tool.
@app.tool()
async def health() -> dict[str, str]:
    """Health check endpoint for monitoring server status.

    Returns:
        Health status information
    """
    try:
        settings = get_settings()

        return {
            "status": "healthy",
            "version": "0.1.0-phase1",
            "project_path": str(settings.project_path),
            "features": "basic_search,file_discovery",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "version": "0.1.0-phase1"}


# Server startup function for CLI
async def start_server(host: str = "localhost", port: int = 8080, *, debug: bool = False) -> None:
    """Start the FastMCP server.

    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    import uvicorn

    # TODO: the typechecker doesn't like this and probably neither does uvicorn.
    config = uvicorn.Config(app, host=host, port=port, log_level="debug" if debug else "info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Direct server execution for development
    import asyncio

    asyncio.run(start_server(debug=True))
