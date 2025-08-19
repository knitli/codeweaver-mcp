# sourcery skip: snake-case-variable-declarations
"""Initialize the FastMCP application with default middleware and settings."""

import logging
import re
import time

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import FunctionType
from typing import Annotated, Any, Literal, cast

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware, RetryMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware
from fastmcp.server.middleware.middleware import Middleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from pydantic import ConfigDict, Field, NonNegativeInt
from pydantic.dataclasses import dataclass

from codeweaver import __version__ as version
from codeweaver._common import BaseEnum
from codeweaver._logger import setup_logger
from codeweaver._settings import (
    ErrorHandlingMiddlewareSettings,
    FastMCPHTTPRunArgs,
    FastMCPServerSettingsType,
    LoggingMiddlewareSettings,
    MiddlewareSettings,
    RateLimitingMiddlewareSettings,
    RetryMiddlewareSettings,
)
from codeweaver._settings_registry import get_provider_registry
from codeweaver._statistics import SessionStatistics
from codeweaver._utils import rpartial
from codeweaver.middleware import StatisticsMiddleware
from codeweaver.settings import (
    CodeWeaverSettings,
    FastMCPServerSettings,
    FileFilterSettings,
    get_settings,
)


_STORE: dict[Literal["settings", "server", "lifespan_func", "app_start_func"], Any] = {}


@dataclass(order=True, kw_only=True, config=ConfigDict(extra="forbid", str_strip_whitespace=True))
class StoredSettings:
    """A simple container for storing/caching."""

    settings: Annotated[CodeWeaverSettings, Field(description="Resolved CodeWeaver settings")]
    server: Annotated[
        FastMCPServerSettingsType, Field(description="Resolved FastMCP server settings")
    ]
    lifespan_func: Annotated[
        FunctionType, Field(description="Lifespan function for the application")
    ]
    app_start_func: Annotated[
        FunctionType, Field(description="Function to start the FastMCP application")
    ]

    def __post_init__(self) -> None:
        from pydantic import TypeAdapter

        global _STORE
        _STORE = TypeAdapter(self).dump_python(mode="python")


class Feature(BaseEnum):
    """Enum for features supported by the CodeWeaver server."""

    BASIC_SEARCH = "basic_search"
    SEMANTIC_SEARCH = "semantic_search"
    VECTOR_SEARCH = "vector_search"
    HYBRID_SEARCH = "hybrid_search"

    WEB_SEARCH = "web_search"  # tavily, duckduckgo

    SPARSE_INDEXING = "sparse_indexing"
    VECTOR_INDEXING = "vector_indexing"

    FILE_DISCOVERY = "file_discovery"
    PROVIDER_REGISTRY = "registry"

    MCP_CONTEXT_AGENT = "mcp_context_agent"
    NON_MCP_CONTEXT_AGENT = "non_mcp_context_agent"

    HEALTH = "health"
    LOGGING = "logging"
    ERROR_HANDLING = "error_handling"
    RATE_LIMITING = "rate_limiting"
    STATISTICS = "statistics"


class HealthStatus(BaseEnum):
    """Enum for health status of the CodeWeaver server."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass(order=True, kw_only=True, config=ConfigDict(extra="forbid", str_strip_whitespace=True))
class HealthInfo:
    """Health information for the CodeWeaver server."""

    status: Annotated[HealthStatus, Field(description="Health status of the server")] = (
        HealthStatus.HEALTHY
    )
    version: Annotated[str, Field(description="Version of the CodeWeaver server")] = version
    startup_time: Annotated[float, Field(description="Startup time of the server")] = time.time()
    features: Annotated[
        tuple[Feature],
        Field(default_factory=tuple, description="List of features supported by the server"),
    ] = (
        Feature.BASIC_SEARCH,
        Feature.FILE_DISCOVERY,
        Feature.PROVIDER_REGISTRY,
        Feature.STATISTICS,
    )  # type: ignore
    error: Annotated[str | None, Field(description="Error message if any")] = None


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
    registry: Annotated[
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
    health: Annotated[HealthInfo, Field(description="Health status information")] = HealthInfo()

    # TODO: Future implementation
    indexer: None = None  # Placeholder for background indexer

    middleware_stack: Annotated[
        tuple[Middleware, ...], Field(description="Loaded middleware stack")
    ] = Field(default_factory=tuple)


@asynccontextmanager
async def lifespan(
    app: FastMCP[AppState],
    settings: CodeWeaverSettings | None,
    statistics: SessionStatistics | None = None,
) -> AsyncIterator[AppState]:
    """Context manager for application lifespan with proper initialization."""
    statistics = statistics or SessionStatistics()
    settings = settings or get_settings()
    if not hasattr(app, "state"):
        setattr(  # noqa: B010
            app,
            "state",
            AppState(
                initialized=False,
                config_path=settings.config_file if settings else None,
                registry=get_provider_registry(),
                middleware_stack=tuple(getattr(app, "middleware", ())),
            ),
        )  # type: ignore
    state: AppState = app.state  # type: ignore
    try:
        # Initialize health status
        state.health = HealthInfo(
            status=HealthStatus.HEALTHY,
            version=version,
            startup_time=time.time(),
            features=(
                Feature.BASIC_SEARCH,
                Feature.FILE_DISCOVERY,
                Feature.PROVIDER_REGISTRY,
                Feature.STATISTICS,
            ),  # type: ignore
        )

        state.initialized = True
        # Yield the initialized state
        yield state

    except Exception as e:
        # Handle initialization errors
        state.health.status = HealthStatus.UNHEALTHY  # type: ignore
        import json

        state.health.error = json.dumps({"error": e})  # type: ignore
        state.initialized = False
        raise

    finally:
        # Cleanup resources
        state.initialized = False


def get_default_middleware_settings(logger: logging.Logger, log_level: int) -> MiddlewareSettings:
    """Get the default middleware settings."""
    return MiddlewareSettings(
        error_handling=ErrorHandlingMiddlewareSettings(
            logger=logger, include_traceback=True, error_callback=None, transform_errors=False
        ),
        retry=RetryMiddlewareSettings(
            max_retries=5, base_delay=1.0, max_delay=60.0, backoff_multiplier=2.0, logger=logger
        ),
        logging=LoggingMiddlewareSettings(
            logger=logger, log_level=log_level, include_payloads=False
        ),
        rate_limiting=RateLimitingMiddlewareSettings(
            max_requests_per_second=75, get_client_id=None, burst_capacity=150, global_limit=True
        ),
    )


def resolve_globs(path_string: str, repo_root: Path) -> set[Path]:
    """Resolve glob patterns in a path string."""
    if "*" in path_string or "?" in path_string or re.search(r"\[.+\]", path_string):
        return set(repo_root.glob(path_string))
    if (path := (repo_root / path_string)) and path.exists():
        return {path} if path.is_file() else set(path.glob("**/*"))
    return set()


def resolve_includes_and_excludes(
    filter_settings: FileFilterSettings, repo_root: Path
) -> tuple[frozenset[Path], frozenset[Path]]:
    """Resolve included and excluded files based on filter settings.

    Resolves glob patterns for include and exclude paths, filtering includes for excluded extensions.
    If a file is specifically included in the `forced_includes`, it will not be excluded even if it matches an excluded extension or excludes.
    "Specifically included" means that it was defined directly in the `forced_includes`, and **not** as a glob pattern.
    """
    settings = filter_settings.model_dump(mode="python")
    other_files: set[Path] = set()
    specifically_included_files = {
        Path(file)
        for file in settings.get("forced_includes", set())
        if file
        and "*" not in file
        and "?" not in file
        and Path(file).exists()
        and Path(file).is_file()
    }
    for include in settings.get("forced_includes", set()):
        other_files |= resolve_globs(include, repo_root)
    for ext in settings.get("excluded_extensions", set()):
        # we only exclude `other_files` if the file was not specifically included (not by globs)
        if not ext:
            continue
        ext = ext.lstrip("*?[]")
        ext = ext if ext.startswith(".") else f".{ext}"
        other_files -= {
            file
            for file in other_files
            if file.suffix == ext and file not in specifically_included_files
        }
    excludes: set[Path] = set()
    excluded_files = settings.get("excluded_files", set())
    for exclude in excluded_files:
        if exclude:
            excludes |= resolve_globs(exclude, repo_root)
    excludes |= specifically_included_files
    other_files -= {exclude for exclude in excludes if exclude not in specifically_included_files}
    other_files -= {None, Path(), Path("./"), Path("./.")}  # Remove empty paths
    excludes -= {None, Path(), Path("./"), Path("./.")}  # Remove empty paths
    return frozenset(other_files), frozenset(excludes)


def _setup_logger(settings: CodeWeaverSettings) -> tuple[logging.Logger, int]:
    """Set up the logger from settings.

    Returns:
        Tuple of (logger, log_level)
    """
    logger_settings = settings.logging or {}
    level = logger_settings.get("level", 20)
    rich = logger_settings.get("use_rich", True)
    rich_kwargs = logger_settings.get("rich_kwargs", {})
    logging_kwargs = logger_settings.get("dict_config", None)
    logger = setup_logger(
        name="codeweaver",
        level=level,
        rich=rich,
        rich_kwargs=rich_kwargs,
        logging_kwargs=logging_kwargs,
    )
    return logger, level


def _configure_middleware(
    settings: CodeWeaverSettings, logger: logging.Logger, level: int
) -> tuple[MiddlewareSettings, Any]:
    """Configure middleware settings and determine logging middleware type.

    Returns:
        Tuple of (middleware_settings, logging_middleware_class)
    """
    middleware_settings = settings.middleware_settings or {}
    middleware_logging_settings = middleware_settings.get("logging", {}) or {}
    use_structured_logging = middleware_logging_settings.get("use_structured_logging", False)
    logging_middleware = (
        StructuredLoggingMiddleware if use_structured_logging else LoggingMiddleware
    )
    middleware_defaults: MiddlewareSettings = get_default_middleware_settings(logger, level)
    if middleware_settings := settings.middleware_settings or None:  # type: ignore
        middleware_defaults |= middleware_settings
    middleware_settings: MiddlewareSettings = middleware_defaults
    return middleware_settings, logging_middleware


def _create_base_fastmcp_settings(
    session_statistics: SessionStatistics,
    logger: logging.Logger,
    level: int,
    middleware_settings: MiddlewareSettings,
    logging_middleware: type[LoggingMiddleware | StructuredLoggingMiddleware],
) -> FastMCPServerSettingsType:
    """Create the base FastMCP settings dictionary.

    Returns:
        Dictionary with base FastMCP configuration
    """
    return {
        "instructions": "Ask a question, describe what you're trying to do, and get the exact context you need. CodeWeaver is an advanced code search and code context tool. It keeps an updated vector, AST, and text index of your codebase, and uses intelligent intent analysis to provide the most relevant context for AI Agents to complete tasks. It's just one easy-to-use tool - the `find_code` tool. To use it, you only need to provide a plain language description of what you want to find, and what you are trying to do. CodeWeaver will return the most relevant code matches, along with their context and precise locations.",
        "version": version,
        "lifespan": lifespan,
        "include_tags": {"external", "user", "context"},
        "exclude_tags": {"internal", "system", "admin"},
        "include_fastmcp_meta": True,
        "middleware": [
            StatisticsMiddleware(session_statistics, logger=logger, log_level=level),
            logging_middleware(**middleware_settings["logging"]),  # type: ignore
            ErrorHandlingMiddleware(**middleware_settings["error_handling"]),  # type: ignore
            RetryMiddleware(**middleware_settings["retry"]),  # type: ignore
            RateLimitingMiddleware(**middleware_settings["rate_limiting"]),  # type: ignore
        ],
        "tools": [],
        "resources": [],
    }


type SettingsKey = Literal["dependencies", "middleware", "tools"]


def _integrate_user_settings(
    settings: FastMCPServerSettings, base_fast_mcp_settings: FastMCPServerSettingsType
) -> FastMCPServerSettingsType:
    """Integrate user-provided settings with base FastMCP settings.

    Args:
        settings: CodeWeaver settings containing user preferences
        base_fast_mcp_settings: Base FastMCP configuration to extend

    Returns:
        Updated FastMCP settings dictionary
    """
    additional_keys = ("additional_dependencies", "additional_middleware", "additional_tools")
    for key in additional_keys:
        if (value := getattr(settings, key, None)) and isinstance(value, list):
            if key in {"additional_dependencies", "additional_tools"} and (
                all(isinstance(item, str) for item in value)  # type: ignore
            ):
                # If it's a list of strings, we can directly append it
                settings_key: SettingsKey = cast(SettingsKey, key.replace("additional_", ""))
                base_fast_mcp_settings[settings_key].extend(value)  # type: ignore
                continue
            if key == "additional_middleware" and (
                all(isinstance(item, Middleware | Callable) for item in value)  # type: ignore
            ):
                base_fast_mcp_settings["middleware"].extend(value)  # type: ignore

    server_settings = settings.model_dump(
        mode="python", exclude_defaults=True, exclude_unset=True, exclude_none=True
    )

    return {**base_fast_mcp_settings, **cast(FastMCPServerSettingsType, server_settings)}


def _setup_file_filters_and_lifespan(
    settings: CodeWeaverSettings, session_statistics: SessionStatistics
) -> Any:
    """Set up file filters and create lifespan function.

    Args:
        settings: CodeWeaver settings
        session_statistics: Session statistics instance

    Returns:
        Configured lifespan function
    """
    settings.filter_settings.forced_includes, settings.filter_settings.excludes = (
        resolve_includes_and_excludes(settings.filter_settings, settings.project_path)
    )
    return rpartial(lifespan, settings, session_statistics)


def _filter_server_settings(server_settings: FastMCPServerSettings) -> FastMCPServerSettingsType:
    """Filter server settings to remove keys not recognized by FastMCP."""
    filtered_settings = server_settings.model_dump(mode="python")
    to_remove = ("additional_middleware", "additional_tools", "additional_dependencies")
    for key in to_remove:
        filtered_settings.pop(key, None)
    return cast(FastMCPServerSettingsType, filtered_settings)


def _create_start_method(fastmcp_settings: dict[str, Any]) -> str:
    """Create the start method for the FastMCP application.

    Args:
        settings: CodeWeaver settings
        app_state: Application state instance

    Returns:
        Function to start the FastMCP application
    """
    return (
        "run_http_async"
        if fastmcp_settings.get("transport", "http") == "http"
        else "run_stdio_async"
    )


def _get_fastmcp_run_args(server_settings: FastMCPServerSettingsType) -> FastMCPHTTPRunArgs | None:
    """Get the FastMCP run arguments from the server settings.

    Args:
        server_settings: The server settings to extract run arguments from.

    Returns:
        The FastMCP run arguments, or None if not found.
    """
    if not server_settings:
        return None
    if server_settings.get("transport", "http") != "http":
        return None

    return FastMCPHTTPRunArgs(
        transport=server_settings.get("transport", "http"),  # type: ignore
        host=server_settings.get("host"),
        port=server_settings.get("port"),
        log_level=server_settings.get("log_level"),
        path=server_settings.get("path"),
        uvicorn_config=server_settings.get("uvicorn_config"),
        middleware=[],
    )


async def initialize_app() -> FastMCP[AppState]:
    """Initialize the FastMCP application."""
    session_statistics = SessionStatistics()
    settings = get_settings()
    logger, level = _setup_logger(settings)
    middleware_settings, logging_middleware = _configure_middleware(settings, logger, level)
    filtered_server_settings = _filter_server_settings(settings.server or {})
    base_fast_mcp_settings = _create_base_fastmcp_settings(
        session_statistics, logger, level, middleware_settings, logging_middleware
    )
    base_fast_mcp_settings = _integrate_user_settings(settings.server, filtered_server_settings)
    lifespan_fn = _setup_file_filters_and_lifespan(settings, session_statistics)
    base_fast_mcp_settings["lifespan"] = lifespan_fn
    run_args = _get_fastmcp_run_args(settings.server)
    start_method = getattr(FastMCP, _create_start_method(base_fast_mcp_settings), None)  # type: ignore
    return FastMCP[AppState](**base_fast_mcp_settings)
