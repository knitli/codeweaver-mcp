# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Auto-indexing service commands.

Provides CLI commands for managing the auto-indexing service including
starting, stopping, monitoring progress, and configuring indexing behavior.
"""

import asyncio
import json
import logging

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from codeweaver.cli.types import OutputFormat
from codeweaver.cli.utils import ServerManager, raise_cli_error


if TYPE_CHECKING:
    from codeweaver.services.manager import ServicesManager
    from codeweaver.services.providers.auto_indexing import AutoIndexingService


logger = logging.getLogger(__name__)
app = App(name="index", help="Auto-indexing service management - control and monitor code indexing")


async def _handle_service_start(
    auto_indexing_service: "AutoIndexingService",
    services_manager: "ServicesManager",
    paths: Sequence[Path],
    *,
    watch: bool,
) -> "AutoIndexingService":
    """Handle starting the auto-indexing service with optional paths and watching."""
    if auto_indexing_service:
        await services_manager.stop_service("auto_indexing")
        await asyncio.sleep(1.0)
    await services_manager.start_service("auto_indexing")
    auto_indexing_service = await ServerManager.get_service("auto_indexing")
    if paths and hasattr(auto_indexing_service, "add_paths"):
        for path in paths:
            await auto_indexing_service.add_path(Path(path))
    if hasattr(auto_indexing_service, "enable_watching"):
        if watch:
            await auto_indexing_service.enable_watching()
        else:
            await auto_indexing_service.disable_watching()
    return auto_indexing_service


@app.command
async def start(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    paths: Annotated[
        list[str] | None,
        Parameter("--paths", "-p", help="Specific paths to index (overrides config)"),
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    watch: Annotated[
        bool, Parameter("--watch", "-w", help="Enable file watching for real-time updates")
    ] = True,
    force: Annotated[bool, Parameter("--force", help="Force restart if already running")] = False,
) -> None:
    """Start the auto-indexing service."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if auto_indexing_service and (not force):
            result = {
                "status": "already_running",
                "message": "Auto-indexing service is already running",
                "service_info": {
                    "type": type(auto_indexing_service).__name__,
                    "watching": getattr(auto_indexing_service, "watching", False),
                },
            }
        else:
            services_manager = await ServerManager.get_services_manager(config_path)
            auto_indexing_service = await _handle_service_start(
                auto_indexing_service, services_manager, paths, watch
            )
            result = {
                "status": "started",
                "message": "Auto-indexing service started successfully",
                "service_info": {
                    "type": type(auto_indexing_service).__name__,
                    "watching": watch,
                    "paths": paths or [],
                },
            }
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            status_icon = "✓" if result["status"] == "started" else "ℹ"  # noqa: RUF001
            print(f"{status_icon} {result['message']}")
            if result.get("service_info"):
                info = result["service_info"]
                print(f"  Type: {info['type']}")
                print(f"  Watching: {('Yes' if info.get('watching') else 'No')}")
                if info.get("paths"):
                    print(f"  Paths: {', '.join(info['paths'])}")
    except Exception as e:
        logger.exception("Failed to start auto-indexing service")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to start auto-indexing service: {e}")


@app.command
async def stop(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    force: Annotated[
        bool, Parameter("--force", help="Force stop without graceful shutdown")
    ] = False,
) -> None:
    """Stop the auto-indexing service."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if not auto_indexing_service:
            result = {"status": "not_running", "message": "Auto-indexing service is not running"}
        else:
            services_manager = await ServerManager.get_services_manager(config_path)
            await services_manager.stop_service("auto_indexing")
            result = {"status": "stopped", "message": "Auto-indexing service stopped successfully"}
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            status_icon = "✓" if result["status"] == "stopped" else "ℹ"  # noqa: RUF001
            print(f"{status_icon} {result['message']}")
    except Exception as e:
        logger.exception("Failed to stop auto-indexing service")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to stop auto-indexing service: {e}")


async def _format_status_data(
    auto_indexing_service: "AutoIndexingService", *, detailed: bool
) -> dict[str, Any]:
    """Format status data for auto-indexing service."""
    status_data = {"running": False, "status": "stopped"}
    if auto_indexing_service:
        status_data = {
            "running": True,
            "status": "running",
            "type": type(auto_indexing_service).__name__,
        }
        if detailed:
            return await auto_indexing_service.create_service_context(status_data)
    return status_data


def _print_status_intro(status_data: "AutoIndexingService") -> None:
    """Print introductory status information for auto-indexing service."""
    print("Auto-indexing Service Status")
    print("=" * 30)
    if not status_data["running"]:
        print("○ Stopped")
        return
    print("✓ Running")
    print(f"  Type: {status_data.get('type', 'Unknown')}")
    print(f"  Watching: {('Yes' if status_data.get('watching') else 'No')}")


@app.command
async def status(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    detailed: Annotated[
        bool, Parameter("--detailed", "-d", help="Show detailed indexing statistics")
    ] = False,
) -> None:
    """Show auto-indexing service status and statistics."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        status_data = _format_status_data(auto_indexing_service, detailed=detailed)
        if fmt == OutputFormat.JSON:
            print(json.dumps(status_data, indent=2, default=str))
            return
        _print_status_intro(status_data)
        for key, value in status_data.items():
            if isinstance(value, str | int | bool):
                print(f"  {key.capitalize()}: {value}")
            else:
                print(f"  {key.capitalize()}: {json.dumps(value, indent=2)}")
    except Exception as e:
        logger.exception("Failed to get auto-indexing status")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "running": False}))
        else:
            print(f"Failed to get auto-indexing status: {e}")


@app.command
async def add_path(
    path: Annotated[str, Parameter(help="Path to add to indexing")],
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    recursive: Annotated[
        bool, Parameter("--recursive", "-r", help="Index path recursively")
    ] = True,
) -> None:
    """Add a path to auto-indexing."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if not auto_indexing_service:
            raise_cli_error(
                "Auto-indexing service is not running. Start it first with 'codeweaver index start'"
            )
        target_path = Path(path).resolve()
        if hasattr(auto_indexing_service, "add_path"):
            await auto_indexing_service.add_path(target_path, recursive=recursive)
            result = {
                "status": "added",
                "message": f"Path '{target_path}' added to indexing",
                "path": str(target_path),
                "recursive": recursive,
            }
        else:
            raise_cli_error("Auto-indexing service does not support dynamic path addition")
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ {result['message']}")
            print(f"  Recursive: {('Yes' if recursive else 'No')}")
    except Exception as e:
        logger.exception("Failed to add path %s", path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to add path: {e}")


@app.command
async def remove_path(
    path: Annotated[str, Parameter(help="Path to remove from indexing")],
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Remove a path from auto-indexing."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if not auto_indexing_service:
            raise_cli_error("Auto-indexing service is not running")
        target_path = Path(path).resolve()
        if hasattr(auto_indexing_service, "remove_path"):
            await auto_indexing_service.remove_path(target_path)
            result = {
                "status": "removed",
                "message": f"Path '{target_path}' removed from indexing",
                "path": str(target_path),
            }
        else:
            raise_cli_error("Auto-indexing service does not support dynamic path removal")
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ {result['message']}")
    except Exception as e:
        logger.exception("Failed to remove path %s", path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to remove path: {e}")


async def _do_reindex(
    auto_indexing_service: "AutoIndexingService",
    target_paths: Sequence[Path],
    *,
    wait: bool,
    connection_timeout: float,
) -> None:
    """Perform reindexing on specified paths."""
    await auto_indexing_service.reindex(target_paths)
    if wait:
        # Wait for completion logic (simplified for clarity)
        import time

        start = time.time()
        while True:
            progress = auto_indexing_service.get_progress()
            if progress.get("completed", 0) >= progress.get("total", 0):
                break
            if time.time() - start > connection_timeout:
                break
            await asyncio.sleep(1)


@app.command
async def reindex(
    paths: Annotated[
        list[str] | None,
        Parameter(help="Specific paths to reindex (if not provided, reindexes all)"),
    ] = None,
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    wait: Annotated[
        bool, Parameter("--wait", "-w", help="Wait for reindexing to complete")
    ] = False,
    connection_timeout: Annotated[
        float,
        Parameter(
            "--timeout", "-t", alias="timeout", help="Timeout in seconds to wait for completion"
        ),
    ] = 300.0,
) -> None:
    """Trigger reindexing of paths."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if not auto_indexing_service:
            raise_cli_error("Auto-indexing service is not running")
        if not hasattr(auto_indexing_service, "reindex"):
            raise_cli_error("Auto-indexing service does not support manual reindexing")
        target_paths = [Path(p).resolve() for p in paths] if paths else None
        await _do_reindex(auto_indexing_service, target_paths, wait, connection_timeout)
        result = {
            "status": "reindexed",
            "message": "Reindexing triggered successfully",
            "paths": [str(p) for p in target_paths] if target_paths else "all",
        }
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ {result['message']}")
            if target_paths:
                print(f"  Paths: {', '.join([str(p) for p in target_paths])}")
            else:
                print("  Paths: all")
    except Exception as e:
        logger.exception("Failed to trigger reindexing")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to trigger reindexing: {e}")


@app.command
async def watch(
    enable: Annotated[bool, Parameter(help="Enable (true) or disable (false) file watching")],
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Enable or disable file watching for real-time indexing."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if not auto_indexing_service:
            raise_cli_error("Auto-indexing service is not running")
        if hasattr(auto_indexing_service, "enable_watching") and hasattr(
            auto_indexing_service, "disable_watching"
        ):
            if enable:
                await auto_indexing_service.enable_watching()
                action = "enabled"
            else:
                await auto_indexing_service.disable_watching()
                action = "disabled"
            result = {
                "status": "configured",
                "message": f"File watching {action}",
                "watching": enable,
            }
        else:
            raise_cli_error("Auto-indexing service does not support watching configuration")
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ {result['message']}")
    except Exception as e:
        logger.exception("Failed to configure watching")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to configure watching: {e}")


if __name__ == "__main__":
    app.parse_args()
