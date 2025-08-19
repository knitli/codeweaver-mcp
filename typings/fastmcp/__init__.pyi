

from fastmcp.client import Client
from fastmcp.server.context import Context
from fastmcp.server.server import FastMCP

from . import client

"""FastMCP - An ergonomic MCP interface."""
settings = ...
__version__ = ...
if settings.deprecation_warnings: ...

def __getattr__(name: str):  # -> type[Image]:
    ...

__all__ = ["Client", "Context", "FastMCP", "client", "settings"]
