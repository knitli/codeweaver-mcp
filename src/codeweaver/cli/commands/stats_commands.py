# sourcery skip: avoid-global-variables, require-parameter-annotation
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Health monitoring and statistics commands.

Provides CLI commands for monitoring CodeWeaver health, collecting statistics,
and generating performance reports.
"""

import asyncio
import contextlib
import json
import logging

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

from codeweaver.cli.types import OutputFormat
from codeweaver.cli.utils import ServerManager


if TYPE_CHECKING:
    from codeweaver.server import CodeWeaverServer
    from codeweaver.services.auto_indexing import AutoIndexingService

logger = logging.getLogger(__name__)

# Create stats command group
app = App(
    name="stats", help="Health monitoring and statistics - track CodeWeaver performance and status"
)


def format_health_output(health_data: dict, fmt: OutputFormat) -> str:
    """Format health data for output."""
    if fmt == OutputFormat.JSON:
        return json.dumps(health_data, indent=2, default=str)

    # Overall status
    server_running = health_data.get("server_running", False)
    lines = [
        "CodeWeaver Health Status",
        "=" * 25,
        f"Server: {'✓ Running' if server_running else '✗ Not running'}",
    ]
    # Services status
    services = health_data.get("services", {})
    if services:
        lines.append("\nServices:")
        for service_name, service_info in services.items():
            if service_info.get("healthy", False):
                status_icon = "✓"
                status_text = "Healthy"
            elif service_info.get("status") == "stopped":
                status_icon = "○"
                status_text = "Stopped"
            else:
                status_icon = "✗"
                status_text = f"Unhealthy: {service_info.get('error', 'Unknown error')}"

            lines.append(f"  {status_icon} {service_name}: {status_text}")

    # Overall health summary
    total_services = len(services)
    healthy_services = sum(bool(s.get("healthy", False)) for s in services.values())

    lines.append(f"\nOverall: {healthy_services}/{total_services} services healthy")

    return "\n".join(lines)


async def _get_health_data(config_path: Path, *, detailed: bool) -> dict[str, Any]:
    """Collect health data for CodeWeaver server and services."""
    health_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "server_running": ServerManager.is_running(),
        "services": {},
    }
    if health_data["server_running"]:
        try:
            server = await ServerManager.ensure_server("health check", config_path)
            if detailed:
                health_data["server_info"] = {
                    "type": type(server).__name__,
                    "initialized": getattr(server, "_initialized", False),
                }
            if server.services_manager:
                available_services = await server.services_manager.list_services()
                for service_name in available_services:
                    health_data["services"][service_name] = await _get_service_health(
                        server, service_name, detailed=detailed
                    )
        except Exception as e:
            health_data["server_error"] = str(e)
            health_data["server_running"] = False
    return health_data


async def _get_service_health(
    server: "CodeWeaverServer", service_name, *, detailed: bool
) -> dict[str, Any]:
    """Get health status of a specific service."""
    try:
        service = await server.services_manager.get_service(service_name)
        service_health = {
            "status": "running" if service else "stopped",
            "healthy": service is not None,
        }
        if detailed and service:
            service_health |= {
                "type": type(service).__name__,
                "uptime": getattr(service, "uptime", None),
            }
            if hasattr(service, "get_health"):
                service_health["health_details"] = service.get_health()
            elif hasattr(service, "get_stats"):
                service_health["stats"] = service.get_stats()
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}
    else:
        return service_health


@app.command
async def health(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    detailed: Annotated[
        bool, Parameter("--detailed", "-d", help="Show detailed health information")
    ] = False,
) -> None:
    """Show comprehensive health status of CodeWeaver server and services."""
    try:
        config_path = str(config) if config else None
        health_data = await _get_health_data(config_path, detailed=detailed)
        output = format_health_output(health_data, fmt)
        print(output)
    except Exception as e:
        logger.exception("Health check failed")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "healthy": False}))
        else:
            print(f"Health check failed: {e}")


async def _get_services_stats(services_manager, available_services, include_stats):
    """Collect statistics for all available services."""
    services_stats = {}
    for service_name in available_services:
        try:
            service = await services_manager.get_service(service_name)
            if service:
                service_data = {"status": "running", "type": type(service).__name__}
                if include_stats and hasattr(service, "get_stats"):
                    service_data["stats"] = service.get_stats()
                if hasattr(service, "uptime"):
                    service_data["uptime"] = service.uptime
                services_stats[service_name] = service_data
            else:
                services_stats[service_name] = {"status": "stopped"}
        except Exception as e:
            services_stats[service_name] = {"status": "error", "error": str(e)}
    return services_stats


@app.command
async def services(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    include_stats: Annotated[
        bool, Parameter("--stats", "-s", help="Include service statistics")
    ] = False,
) -> None:
    """Show statistics for all services."""
    try:
        config_path = str(config) if config else None
        services_manager = await ServerManager.get_services_manager(config_path)
        available_services = await services_manager.list_services()
        services_stats = await _get_services_stats(
            services_manager, available_services, include_stats
        )
        if fmt == OutputFormat.JSON:
            print(json.dumps(services_stats, indent=2, default=str))
        else:
            print("Service Statistics")
            print("=" * 18)
            for service_name, data in services_stats.items():
                status = data["status"]
                status_icon = {"running": "✓", "stopped": "○", "error": "✗"}.get(status, "?")
                print(f"{status_icon} {service_name}: {status.title()}")
                if data.get("type"):
                    print(f"    Type: {data['type']}")
                if data.get("uptime"):
                    print(f"    Uptime: {data['uptime']}")
                if data.get("stats"):
                    print("    Statistics:")
                    for key, value in data["stats"].items():
                        print(f"      {key}: {value}")
                if data.get("error"):
                    print(f"    Error: {data['error']}")
                print()
    except Exception as e:
        logger.exception("Failed to get service statistics")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to get service statistics: {e}")


def _get_indexing_stats(
    auto_indexing_service: "AutoIndexingService", *, detailed: bool
) -> dict[str, Any]:
    """Collect statistics for auto-indexing service."""
    if not auto_indexing_service:
        return {"status": "not_running", "message": "Auto-indexing service is not running"}
    indexing_stats = {"status": "running", "type": type(auto_indexing_service).__name__}
    if hasattr(auto_indexing_service, "get_stats"):
        indexing_stats["stats"] = auto_indexing_service.get_stats()
    if hasattr(auto_indexing_service, "get_indexed_paths"):
        indexed_paths = auto_indexing_service.get_indexed_paths()
        indexing_stats["indexed_paths"] = indexed_paths
        indexing_stats["paths_count"] = len(indexed_paths)
    if hasattr(auto_indexing_service, "get_progress"):
        indexing_stats["progress"] = auto_indexing_service.get_progress()
    if hasattr(auto_indexing_service, "watching"):
        indexing_stats["watching"] = auto_indexing_service.watching
    if hasattr(auto_indexing_service, "get_performance_metrics"):
        indexing_stats["performance"] = auto_indexing_service.get_performance_metrics()
    return indexing_stats


def indexing_output(indexing_stats: dict[str, Any], fmt: OutputFormat, *, detailed: bool) -> str:
    """Format the output for indexing stats.

    Args:
        indexing_stats: Stats dictionary.
        fmt: Output format.
        detailed: Whether to show detailed output.

    Returns:
        Formatted string output.
    """
    if not indexing_stats.get("service_running"):
        return "○ Service not running"
    output_lines = [
        "✓ Service running",
        f"  Type: {indexing_stats.get('type', 'Unknown')}",
        f"  Watching: {'Yes' if indexing_stats.get('watching') else 'No'}",
    ]
    if detailed and "paths" in indexing_stats:
        output_lines.extend([f"  • {path}" for path in indexing_stats["paths"]])
    return "\n".join(output_lines)


@app.command
async def indexing(
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
    """Show auto-indexing service statistics."""
    try:
        config_path = str(config) if config else None
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)

        # Extracted logic for collecting indexing stats
        def _collect_index_stats():
            return _get_indexing_stats(auto_indexing_service, detailed=detailed)

        # Extracted logic for formatting indexing output
        def _format_index_output(stats):
            return indexing_output(stats, fmt, detailed=detailed)

        # Main logic now delegates to helpers
        stats = _collect_index_stats()
        output = _format_index_output(stats)
        print(output)

    except Exception as e:
        logger.exception("Failed to get indexing statistics")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to get indexing statistics: {e}")


async def _get_performance_data(server, duration) -> dict[str, Any]:
    """Collect performance data for server and services."""
    performance_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "server": {"running": True, "type": type(server).__name__},
        "services": {},
    }
    start_time = datetime.now(UTC)
    if server.services_manager:
        available_services = await server.services_manager.list_services()
        for service_name in available_services:
            try:
                service = await server.services_manager.get_service(service_name)
                if service:
                    service_metrics = {"status": "running", "type": type(service).__name__}
                    if hasattr(service, "get_performance_metrics"):
                        service_metrics["metrics"] = service.get_performance_metrics()
                    elif hasattr(service, "get_stats"):
                        service_metrics["stats"] = service.get_stats()
                    performance_data["services"][service_name] = service_metrics
            except Exception as e:
                performance_data["services"][service_name] = {"status": "error", "error": str(e)}
    return performance_data, start_time


async def _collect_end_metrics(server, performance_data):
    """Collect end metrics for all services after performance monitoring."""
    for service_name in performance_data["services"]:
        if performance_data["services"][service_name]["status"] == "running":
            with contextlib.suppress(Exception):
                service = await server.services_manager.get_service(service_name)
                if service and hasattr(service, "get_performance_metrics"):
                    end_metrics = service.get_performance_metrics()
                    performance_data["services"][service_name]["end_metrics"] = end_metrics


@app.command
async def performance(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    duration: Annotated[
        int, Parameter("--duration", "-d", help="Duration in seconds to collect metrics")
    ] = 60,
) -> None:
    """Show performance metrics and resource usage."""
    try:
        config_path = str(config) if config else None
        server = await ServerManager.ensure_server("performance monitoring", config_path)

        async def _gather_performance_metrics():
            nonlocal server, duration
            performance_data, start_time = await _get_performance_data(server, duration)
            if duration > 0:
                print(f"Collecting performance metrics for {duration} seconds...")
                await asyncio.sleep(duration)
                end_time = datetime.now(UTC)
                performance_data["actual_duration"] = (end_time - start_time).total_seconds()
                await _collect_end_metrics(server, performance_data)
            return performance_data

        def _format_performance_output(performance_data) -> str:
            """Format the output for performance metrics."""
            if performance_data.get("fmt") == OutputFormat.JSON:
                return json.dumps(performance_data, indent=2, default=str)
            output_lines = [
                f"Duration: {performance_data.get('actual_duration', performance_data.get('duration')):.1f} seconds",
                f"Timestamp: {performance_data['timestamp']}",
            ]
            # Add more lines as needed
            return "\n".join(output_lines)

        # Main logic now delegates to helpers
        metrics = await _gather_performance_metrics()
        output = _format_performance_output(metrics)
        print(output)

    except Exception as e:
        logger.exception("Failed to collect performance metrics")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to collect performance metrics: {e}")


async def _get_server_status(config_path) -> tuple[Any, str, str | None]:
    """Get server status for summary command."""
    try:
        server = await ServerManager.ensure_server("summary", config_path)
    except Exception as e:
        return None, "error", str(e)
    else:
        return server, "running", None


async def _get_services_summary(server) -> dict[str, int]:
    """Collect summary of services status."""
    summary = {"total": 0, "running": 0, "stopped": 0}
    if server and server.services_manager:
        available_services = await server.services_manager.list_services()
        running_count = 0
        total_count = len(available_services)
        for service_name in available_services:
            with contextlib.suppress(Exception):
                service = await server.services_manager.get_service(service_name)
                if service:
                    running_count += 1
        summary = {
            "total": total_count,
            "running": running_count,
            "stopped": total_count - running_count,
        }
    return summary


async def _get_indexing_summary(config_path: Path) -> dict[str, Any]:
    """Get summary of auto-indexing service."""
    try:
        auto_indexing_service = await ServerManager.get_service("auto_indexing", config_path)
        if auto_indexing_service:
            indexing_summary = {
                "status": "running",
                "watching": getattr(auto_indexing_service, "watching", False),
            }
            if hasattr(auto_indexing_service, "get_indexed_paths"):
                indexed_paths = auto_indexing_service.get_indexed_paths()
                indexing_summary["paths_count"] = len(indexed_paths)
            if hasattr(auto_indexing_service, "get_stats"):
                stats = auto_indexing_service.get_stats()
                indexing_summary["stats"] = stats
            return indexing_summary
    except Exception:
        return {"status": "error"}
    else:
        return {"status": "stopped"}


def _get_overall_health(services_summary, indexing_summary):
    services_healthy = services_summary.get("running", 0) > 0
    indexing_healthy = indexing_summary.get("status") == "running"
    if services_healthy and indexing_healthy:
        return "healthy"
    return "partial" if services_healthy or indexing_healthy else "unhealthy"


def _format_summary_output(summary_data, fmt):
    if fmt == OutputFormat.JSON:
        return json.dumps(summary_data, indent=2, default=str)
    output_lines = [
        "CodeWeaver Summary",
        "=" * 17,
        f"Timestamp: {summary_data['timestamp']}",
        "",
        f"Server: {summary_data['server_status'].title()}",
    ]
    if summary_data.get("server_error"):
        output_lines.append(f"  Error: {summary_data['server_error']}")
    if services := summary_data.get("services_summary", {}):
        output_lines.append(
            f"\nServices: {services.get('running', 0)}/{services.get('total', 0)} running"
        )
        if services.get("stopped", 0) > 0:
            output_lines.append(f"  Stopped: {services['stopped']}")
    if indexing := summary_data.get("indexing_summary", {}):
        _format_indexing_status(indexing, output_lines)
    health = summary_data["overall_health"]
    health_icon = {"healthy": "✓", "partial": "⚠", "unhealthy": "✗", "unknown": "?"}.get(
        health, "?"
    )
    output_lines.append(f"\nOverall Health: {health_icon} {health.title()}")
    return "\n".join(output_lines)


def _format_indexing_status(indexing, output_lines):
    indexing_status = indexing.get("status", "unknown")
    indexing_icon = {"running": "✓", "stopped": "○", "error": "✗"}.get(indexing_status, "?")
    output_lines.append(f"\nIndexing: {indexing_icon} {indexing_status.title()}")
    if indexing.get("watching"):
        output_lines.append("  Watching: Yes")
    if indexing.get("paths_count") is not None:
        output_lines.append(f"  Paths: {indexing['paths_count']}")
    if indexing.get("stats"):
        stats = indexing["stats"]
        if isinstance(stats, dict):
            output_lines.extend(f"  {key}: {value}" for key, value in list(stats.items())[:3])


@app.command
async def summary(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> str:
    """Summarize stats and format output."""
    summary_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "server_status": "unknown",
        "services_summary": {},
        "indexing_summary": {},
        "overall_health": "unknown",
    }
    config_path = str(config) if config else None
    server, server_status, server_error = await _get_server_status(config_path)
    summary_data["server_status"] = server_status
    if server_error:
        summary_data["server_error"] = server_error
        summary_data["overall_health"] = "unhealthy"
        return _format_summary_output(summary_data, fmt)
    summary_data["services_summary"] = await _get_services_summary(server)
    summary_data["indexing_summary"] = await _get_indexing_summary(config_path)
    summary_data["overall_health"] = _get_overall_health(
        summary_data["services_summary"], summary_data["indexing_summary"]
    )
    return _format_summary_output(summary_data, fmt)


if __name__ == "__main__":
    # Allow running individual command module for testing
    app.parse_args()
