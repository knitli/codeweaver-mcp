

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from mcp import ClientSession
from mcp.client.session import ElicitationFnT
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult as MCPElicitResult

__all__ = ["ElicitRequestParams", "ElicitResult", "ElicitationHandler"]
T = TypeVar("T")

class ElicitResult[T](MCPElicitResult):
    content: T | None = ...

type ElicitationHandler[T] = Callable[
    [str, type[T], ElicitRequestParams, RequestContext[ClientSession, LifespanContextT]],
    Awaitable[T | dict[str, Any] | ElicitResult[T | dict[str, Any]]],
]

def create_elicitation_callback(elicitation_handler: ElicitationHandler) -> ElicitationFnT: ...
