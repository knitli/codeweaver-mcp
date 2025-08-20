# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# We need to override our generic models with specific types, and type overrides for narrower values is a good thing.
# pyright: reportIncompatibleMethodOverride=false,reportIncompatibleVariableOverride=false
"""Core settings and provider definitions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import ssl

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal, LiteralString, NotRequired, Required, Self, TypedDict

from fastmcp.contrib.bulk_tool_caller.bulk_tool_caller import BulkToolCaller
from fastmcp.server.auth.auth import OAuthProvider
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware, RetryMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.server import DuplicateBehavior
from fastmcp.tools.tool import Tool
from mcp.server.lowlevel.server import LifespanResultT
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, SecretStr
from pydantic_ai.settings import ModelSettings as AgentModelSettings
from starlette.middleware import Middleware as ASGIMiddleware
from uvicorn.config import (
    SSL_PROTOCOL_VERSION,
    HTTPProtocolType,
    InterfaceType,
    LifespanType,
    LoopSetupType,
    WSProtocolType,
)

from codeweaver._common import BaseEnum
from codeweaver.exceptions import ConfigurationError


AVAILABLE_MIDDLEWARE = (
    BulkToolCaller,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    StructuredLoggingMiddleware,
    DetailedTimingMiddleware,
    RateLimitingMiddleware,
    RetryMiddleware,
)

# ===========================================================================
# *  TypedDict classes for Python Stdlib Logging Configuration (`dictConfig``)
# ===========================================================================

type FormatterID = str


class FormattersDict(TypedDict, total=False):
    """A dictionary of formatters for logging configuration.

    This is used to define custom formatters for logging in a dictionary format.
    Each formatter can have a `format`, `date_format`, `style`, and other optional fields.

    [See the Python documentation for more details](https://docs.python.org/3/library/logging.html#logging.Formatter).
    """

    format: NotRequired[str]
    date_format: NotRequired[str]
    style: NotRequired[str]
    validate: NotRequired[bool]
    defaults: NotRequired[
        Annotated[
            dict[str, Any],
            Field(
                default_factory=dict,
                description="Default values for the formatter. [See the Python documentation for more details](https://docs.python.org/3/library/logging.html#logging.Formatter).",
            ),
        ]
    ]
    class_name: NotRequired[
        Annotated[
            str,
            Field(
                description="The class name of the formatter in the form of an import path, like `logging.Formatter` or `rich.logging.RichFormatter`.",
                alias="class",
            ),
        ]
    ]


type FilterID = str

type FiltersDict = dict[FilterID, dict[Literal["name"] | str, Any]]

type HandlerID = str


class HandlersDict(TypedDict, total=False):
    """A dictionary of handlers for logging configuration.

    This is used to define custom handlers for logging in a dictionary format.
    Each handler can have a `class_name`, `level`, and other optional fields.

    [See the Python documentation for more details](https://docs.python.org/3/library/logging.html#logging.Handler).
    """

    class_name: Required[
        Annotated[
            str,
            Field(
                description="The class name of the handler in the form of an import path, like `logging.StreamHandler` or `rich.logging.RichHandler`.",
                alias="class",
            ),
        ]
    ]
    level: NotRequired[Literal[0, 10, 20, 30, 40, 50]]  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    formatter: NotRequired[FormatterID]  # The ID of the formatter to use for this handler
    filters: NotRequired[list[FilterID | logging.Filter]]


type LoggerName = str


class LoggersDict(TypedDict, total=False):
    """A dictionary of loggers for logging configuration.

    This is used to define custom loggers for logging in a dictionary format.
    Each logger can have a `level`, `handlers`, and other optional fields.

    [See the Python documentation for more details](https://docs.python.org/3/library/logging.html#logging.Logger).
    """

    level: NotRequired[Literal[0, 10, 20, 30, 40, 50]]  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    propagate: NotRequired[bool]  # Whether to propagate messages to the parent logger
    handlers: NotRequired[list[HandlerID]]  # The IDs of the handlers to use for this logger
    filters: NotRequired[
        list[FilterID | logging.Filter]
    ]  # The IDs of the filters to use for this logger, or filter instances


class LoggingConfigDict(TypedDict, total=False):
    """Logging configuration settings. You may optionally use this to customize logging in a very granular way.

    `LoggingConfigDict` is structured to match the format expected by Python's `logging.config.dictConfig` function. You can use this to define loggers, handlers, and formatters in a dictionary format -- either programmatically or in your CodeWeaver settings file.
    [See the Python documentation for more details](https://docs.python.org/3/library/logging.config.html).
    """

    version: Required[Literal[1]]
    formatters: NotRequired[dict[FormatterID, FormattersDict]]
    filters: NotRequired[FiltersDict]
    handlers: NotRequired[dict[HandlerID, HandlersDict]]
    loggers: NotRequired[dict[str, LoggersDict]]
    root: NotRequired[Annotated[LoggersDict, Field(description="The root logger configuration.")]]
    incremental: NotRequired[
        Annotated[
            bool,
            Field(
                description="Whether to apply this configuration incrementally or replace the existing configuration. [See the Python documentation for more details](https://docs.python.org/3/library/logging.config.html#logging-config-dict-incremental)."
            ),
        ]
    ]
    disable_existing_loggers: NotRequired[
        Annotated[
            bool,
            Field(
                description="Whether to disable all existing loggers when configuring logging. If not present, defaults to `True`."
            ),
        ]
    ]


class LoggingSettings(TypedDict, total=False):
    """Global logging settings."""

    level: NotRequired[Literal[0, 10, 20, 30, 40, 50]]  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    use_rich: NotRequired[bool]
    dict_config: NotRequired[
        Annotated[
            LoggingConfigDict,
            Field(
                description="Logging configuration in dictionary format that matches the format expected by [`logging.config.dictConfig`](https://docs.python.org/3/library/logging.config.html)."
            ),
        ]
    ]
    rich_kwargs: NotRequired[
        Annotated[
            dict[str, Any],
            Field(
                description="Additional keyword arguments for the `rich` logging handler, [`rich.logging.RichHandler`], if enabled."
            ),
        ]
    ]


# ===========================================================================
# *          TypedDict classes for Middleware Settings
# ===========================================================================


class ErrorHandlingMiddlewareSettings(TypedDict, total=False):
    """Settings for error handling middleware."""

    logger: NotRequired[logging.Logger | None]
    include_traceback: NotRequired[bool]
    error_callback: NotRequired[Callable[[Exception, MiddlewareContext[Any]], None] | None]
    transform_errors: NotRequired[bool]


class RetryMiddlewareSettings(TypedDict, total=False):
    """Settings for retry middleware."""

    max_retries: NotRequired[int]
    base_delay: NotRequired[float]
    max_delay: NotRequired[float]
    backoff_multiplier: NotRequired[float]
    retry_exceptions: NotRequired[tuple[type[Exception], ...]]
    logger: NotRequired[logging.Logger | None]


class LoggingMiddlewareSettings(TypedDict, total=False):
    """Settings for logging middleware (both structured and unstructured)."""

    logger: Annotated[NotRequired[logging.Logger | None], Field(exclude=True)]
    log_level: NotRequired[int]
    include_payloads: NotRequired[bool]
    max_payload_length: NotRequired[int]
    methods: NotRequired[list[str] | None]

    use_structured_logging: NotRequired[bool]


class RateLimitingMiddlewareSettings(TypedDict, total=False):
    """Settings for rate limiting middleware."""

    max_requests_per_second: NotRequired[PositiveInt]
    burst_capacity: NotRequired[PositiveInt | None]
    get_client_id: NotRequired[Callable[[MiddlewareContext[Any]], str] | None]
    global_limit: NotRequired[bool]


class MiddlewareOptions(TypedDict, total=False):
    """Settings for middleware."""

    error_handling: ErrorHandlingMiddlewareSettings | None
    retry: RetryMiddlewareSettings | None
    logging: LoggingMiddlewareSettings | None
    rate_limiting: RateLimitingMiddlewareSettings | None


# ===========================================================================
# *            Provider Settings classes
# ===========================================================================


class BaseProviderSettings(TypedDict, total=False):
    """Base settings for all providers."""

    provider: Required[Provider]
    enabled: Required[bool]
    api_key: NotRequired[LiteralString | None]
    extra: NotRequired[dict[str, Any] | None]


class DataProviderSettings(BaseProviderSettings):
    """Settings for data providers."""


class EmbeddingModelSettings:
    """Embedding model settings stub."""


class RerankModelSettings:
    """Rerank model settings stub."""


class EmbeddingProviderSettings(BaseProviderSettings):
    """Settings for embedding models."""

    model: Required[str]
    model_settings: NotRequired[EmbeddingModelSettings | None]


class RerankProviderSettings(BaseProviderSettings):
    """Settings for re-ranking models."""

    models: Required[str | tuple[str, ...]]  # Tuple of model names
    """A model name or a tuple of model names to use for re-ranking in order of preference."""
    model_settings: NotRequired[RerankModelSettings | tuple[RerankModelSettings, ...] | None]
    """Settings for the re-ranking model(s)."""
    extra: NotRequired[dict[str, Any] | None]


class AgentProviderSettings(BaseProviderSettings):
    """Settings for agent models."""

    models: Required[str | tuple[str, ...]]
    """A model name or a tuple of model names to use for agent in order of preference."""
    model_settings: NotRequired[AgentModelSettings | tuple[AgentModelSettings, ...] | None]
    """Settings for the agent model(s)."""


class FastMcpHttpRunArgs(TypedDict, total=False):
    transport: Literal["http"]
    host: str | None
    port: PositiveInt | None
    log_level: Literal["debug", "info", "warning", "error"] | None
    path: str | None
    uvicorn_config: UvicornServerSettingsType | None
    middleware: list[ASGIMiddleware] | None


class FastMcpServerSettingsType(TypedDict, total=False):
    """TypedDict for FastMCP server settings.

    Not intended to be used directly; used for internal type checking and validation.
    """

    name: str
    instructions: str | None
    version: str | None
    lifespan: LifespanResultT | None  # type: ignore  # it's purely for context
    include_tags: set[str] | None
    exclude_tags: set[str] | None
    transport: Literal["stdio", "http"] | None
    host: str | None
    port: PositiveInt | None
    path: str | None
    auth: OAuthProvider | None
    cache_expiration_seconds: float | None
    on_duplicate_tools: DuplicateBehavior | None
    on_duplicate_resources: DuplicateBehavior | None
    on_duplicate_prompts: DuplicateBehavior | None
    resource_prefix_format: Literal["protocol", "path"] | None
    middleware: list[Middleware | Callable[..., Any]] | None
    tools: list[Tool | Callable[..., Any]] | None
    dependencies: list[str] | None


# ===========================================================================
# *                        UVICORN Server Settings
# ===========================================================================


class UvicornServerSettings(BaseModel):
    """
    Uvicorn server settings. Besides the port, these are all defaults for uvicorn.

    We expose them so you can configure them for advanced deployments inside your codeweaver.toml (or yaml or json).
    """

    # For the following, we just want to track if it's the default value or not (True/False), not the actual value.
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "TelemetryBoolProps": [
                "host",
                "name",
                "ssl_keyfile",
                "ssl_certfile",
                "ssl_keyfile_password",
                "ssl_version",
                "ssl_cert_reqs",
                "ssl_ca_certs",
                "ssl_ciphers",
                "root_path",
                "headers",
                "server_header",
                "data_header",
                "forwarded_allow_ips",
                "env_file",
                "log_config",
            ]
        },
    )

    name: Annotated[str, Field(exclude=True)] = "CodeWeaver_http"
    host: str = "127.0.0.1"
    port: PositiveInt = 9328
    uds: str | None = None
    fd: int | None = None
    loop: LoopSetupType | str = "auto"
    http: type[asyncio.Protocol] | HTTPProtocolType | str = "auto"
    ws: type[asyncio.Protocol] | WSProtocolType | str = "auto"
    ws_max_size: PositiveInt = 16777216  # 16 MiB
    ws_max_queue: PositiveInt = 32
    ws_ping_interval: PositiveFloat = 20.0
    ws_ping_timeout: PositiveFloat = 20.0
    ws_per_message_deflate: bool = True
    lifespan: LifespanType = "auto"
    env_file: str | os.PathLike[str] | None = None
    log_config: LoggingConfigDict | None = None
    log_level: str | int | None = "info"
    access_log: bool = True
    use_colors: bool | None = None
    interface: InterfaceType = "auto"
    reload: bool = False  # TODO: We should add it, but we need to manage handling it mid-request.
    reload_dirs: list[str] | str | None = None
    reload_delay: PositiveFloat = 0.25
    reload_includes: list[str] | str | None = None
    reload_excludes: list[str] | str | None = None
    workers: int | None = None
    proxy_headers: bool = True
    server_header: bool = True
    data_header: bool = True
    forwarded_allow_ips: str | list[str] | None = None
    root_path: str = ""
    limit_concurrency: PositiveInt | None = None
    limit_max_requests: PositiveInt | None = None
    backlog: PositiveInt = 2048
    timeout_keep_alive: PositiveInt = 5
    timeout_notify: PositiveInt = 30
    timeout_graceful_shutdown: PositiveInt | None = None
    callback_notify: Callable[..., Awaitable[None]] | None = None
    ssl_keyfile: str | os.PathLike[str] | None = None
    ssl_certfile: str | os.PathLike[str] | None = None
    ssl_keyfile_password: SecretStr | None = None
    ssl_version: int | None = SSL_PROTOCOL_VERSION
    ssl_cert_reqs: int = ssl.CERT_NONE
    ssl_ca_certs: SecretStr | None = None
    ssl_ciphers: str = "TLSv1"
    headers: list[tuple[str, str]] | None = None
    factory: bool = False
    h11_max_incomplete_event_size: int | None = None


class UvicornServerSettingsType(TypedDict, total=False):
    """TypedDict for Uvicorn server settings.

    Not intended to be used directly; used for internal type checking and validation.
    """

    name: str
    host: str
    port: PositiveInt
    uds: str | None
    fd: int | None
    loop: LoopSetupType | str
    http: type[asyncio.Protocol] | HTTPProtocolType | str
    ws: type[asyncio.Protocol] | WSProtocolType | str
    ws_max_size: PositiveInt
    ws_max_queue: PositiveInt
    ws_ping_interval: PositiveFloat
    ws_ping_timeout: PositiveFloat
    ws_per_message_deflate: bool
    lifespan: LifespanType
    env_file: str | os.PathLike[str] | None
    log_config: LoggingConfigDict | None
    log_level: str | int | None
    access_log: bool
    use_colors: bool | None
    interface: InterfaceType
    reload: bool
    reload_dirs: list[str] | str | None
    reload_delay: PositiveFloat
    reload_includes: list[str] | str | None
    reload_excludes: list[str] | str | None
    workers: int | None
    proxy_headers: bool
    server_header: bool
    data_header: bool
    forwarded_allow_ips: str | list[str] | None
    root_path: str
    limit_concurrency: PositiveInt | None
    limit_max_requests: PositiveInt | None
    backlog: PositiveInt
    timeout_keep_alive: PositiveInt
    timeout_notify: PositiveInt
    timeout_graceful_shutdown: PositiveInt | None
    callback_notify: Callable[..., Awaitable[None]] | None
    ssl_keyfile: str | os.PathLike[str] | None
    ssl_certfile: str | os.PathLike[str] | None
    ssl_keyfile_password: SecretStr | None
    ssl_version: int | None
    ssl_cert_reqs: int
    ssl_ca_certs: SecretStr | None
    ssl_ciphers: str
    headers: list[tuple[str, str]] | None
    factory: bool
    h11_max_incomplete_event_size: int | None


# ===========================================================================
# *     PROVIDER ENUM - main provider enum for all Codeweaver providers
# ===========================================================================


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"
    FASTEMBED = "fastembed"

    QDRANT = "qdrant"
    FASTEMBED_VECTORSTORE = "fastembed_vectorstore"

    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    GOOGLE = "google"
    X_AI = "x_ai"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    OPENAI = "openai"

    # OpenAI Compatible with OpenAIModel
    AZURE = "azure"  # supports rerank, but not w/ OpenAI API
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GITHUB = "github"
    GROQ = "groq"  # yes, it's different from Grok...
    HEROKU = "heroku"
    MOONSHOT = "moonshot"
    OLLAMA = "ollama"  # supports rerank, but not w/ OpenAI API
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"
    TOGETHER = "together"
    VERCEL = "vercel"

    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"

    _UNSET = "unset"

    @classmethod
    def validate(cls, value: str) -> Self:
        """Validate provider-specific settings."""
        with contextlib.suppress(AttributeError, KeyError, ValueError):
            if value_in_self := cls.from_string(value.strip()):
                return value_in_self
        # TODO: We need to allow for dynamic providers in the future, we would check if there's a provider class registered for the value, then register the provider here with `cls.add_member("NEW_PROVIDER", "new_provider")`.
        raise ConfigurationError(f"Invalid provider: {value}")


class ProviderKind(BaseEnum):
    """Enumeration of available provider kinds."""

    DATA = "data"
    """Provider for data retrieval and processing (e.g. Tavily)"""
    EMBEDDING = "embedding"
    """Provider for text embedding (e.g. Voyage)"""
    RERANKING = "reranking"
    """Provider for re-ranking (e.g. Voyage)"""
    VECTOR_STORE = "vector_store"
    """Provider for vector storage (e.g. Qdrant)"""
    AGENT = "agent"
    """Provider for agents (e.g. OpenAI or Anthropic)"""

    _UNSET = "unset"
    """A sentinel setting to identify when a `ProviderKind` is not set or is configured."""

    @property
    def settings_object(self) -> object:
        """Get the settings object for this provider kind."""
        if self == ProviderKind.DATA:
            return DataProviderSettings
        if self == ProviderKind.EMBEDDING:
            return EmbeddingProviderSettings
        if self == ProviderKind.RERANKING:
            return RerankProviderSettings
        if self == ProviderKind.AGENT:
            return AgentProviderSettings
        raise ConfigurationError(f"ProviderKind {self} does not have a settings object.")


def default_config_file_locations(
    *, as_yaml: bool = False, as_json: bool = False
) -> tuple[str, ...]:
    """Get default file locations for configuration files."""
    # Determine base extensions
    extensions = (
        ["yaml", "yml"] if not as_yaml and not as_json else ["yaml", "yml"] if as_yaml else ["json"]
    )
    # Get user config directory
    user_config_dir = (
        os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        if platform.system() == "Windows"
        else os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    )

    # Build file paths maintaining precedence order
    base_paths = [
        (Path.cwd(), ".codeweaver.local"),
        (Path.cwd(), ".codeweaver"),
        (Path(user_config_dir) / "codeweaver", "settings"),
    ]

    # Generate all file paths using list comprehension
    file_paths = [
        str(base_dir / f"{filename}.{ext}")
        for base_dir, filename in base_paths
        for ext in extensions
    ]

    return tuple(file_paths)
