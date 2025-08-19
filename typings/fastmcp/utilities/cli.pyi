

from typing import TYPE_CHECKING, Any, Literal

from fastmcp import FastMCP

if TYPE_CHECKING: ...
LOGO_ASCII = ...

def log_server_banner(
    server: FastMCP[Any],
    transport: Literal["stdio", "http", "sse", "streamable-http"],
    *,
    host: str | None = ...,
    port: int | None = ...,
    path: str | None = ...,
) -> None:
    ...
