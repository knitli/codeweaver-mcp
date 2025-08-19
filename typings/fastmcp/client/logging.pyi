

from collections.abc import Awaitable, Callable

from mcp.client.session import LoggingFnT
from mcp.types import LoggingMessageNotificationParams

logger = ...
type LogMessage = LoggingMessageNotificationParams
type LogHandler = Callable[[LogMessage], Awaitable[None]]

async def default_log_handler(message: LogMessage) -> None: ...
def create_log_callback(handler: LogHandler | None = ...) -> LoggingFnT: ...
