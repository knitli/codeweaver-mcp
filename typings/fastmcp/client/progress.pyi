


from mcp.shared.session import ProgressFnT

logger = ...
type ProgressHandler = ProgressFnT

async def default_progress_handler(
    progress: float, total: float | None, message: str | None
) -> None:
    ...
