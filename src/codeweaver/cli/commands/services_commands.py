# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Service management commands.

Provides CLI commands for managing CodeWeaver services including starting,
stopping, restarting, and monitoring individual services or all services.
"""

import asyncio
import json
import logging

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeweaver.cli.types import OutputFormat, ServiceType
from codeweaver.cli.utils import ServerManager


logger = logging.getLogger(__name__)
app = App(name="services", help="Service management - start, stop, and monitor CodeWeaver services")


# --- Private helpers for service commands ---


async def _resolve_service_names(services, services_manager, mode="start"):
    """Resolve the list of service names to operate on."""
    if services is not None:
        return [s.value if hasattr(s, "value") else str(s) for s in services]
    if mode == "start":
        return [s.value for s in ServiceType]
    if mode == "stop":
        running_services = await services_manager.list_services()
        return [s for s in running_services if s in ServiceType.get_values()]
    return []


async def _wait_for_service_state(
    services_manager, service_name, should_exist, connection_timeout: float = 30.0
):
    """Wait for a service to reach the desired state (existence or not)."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < connection_timeout:
        service = await services_manager.get_service(service_name)
        if (service is not None) == should_exist:
            return True
        await asyncio.sleep(0.5)
    return False


def _format_service_results(results, fmt, action):
    """Format and print service action results."""
    if fmt == OutputFormat.JSON:
        print(json.dumps(results, indent=2))
    else:
        print(f"Service {action} results:")
        for service_name, result in results.items():
            status_icon = {
                "started": "✓",
                "already_running": "ℹ",  # noqa: RUF001
                "starting": "⏳",
                "timeout": "⚠",
                "error": "✗",
                "stopped": "✓",
                "force_stopped": "✓",
                "not_running": "ℹ",  # noqa: RUF001
            }.get(result["status"], "?")
            print(f"  {status_icon} {service_name}: {result['message']}")


@app.command
async def start(
    services: Annotated[
        list[ServiceType] | None,
        Parameter(help="Specific services to start (if not provided, starts all services)"),
    ] = None,
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    wait: Annotated[
        bool, Parameter("--wait", "-w", help="Wait for services to fully start before returning")
    ] = True,
    connection_timeout: Annotated[
        float,
        Parameter(
            "--timeout",
            "-t",
            alias="timeout",
            help="Timeout in seconds to wait for service startup",
        ),
    ] = 30.0,
) -> None:
    """Start CodeWeaver services."""
    try:
        config_path = str(config) if config else None
        services_manager = await ServerManager.get_services_manager(config_path)
        results = {}
        services_to_start = await _resolve_service_names(services, services_manager, mode="start")
        for raw_service_name in services_to_start:
            try:
                existing_service = await services_manager.get_service(raw_service_name)
                if existing_service:
                    results[raw_service_name] = {
                        "status": "already_running",
                        "message": f"Service '{raw_service_name}' is already running",
                    }
                    continue
                await services_manager.start_service(raw_service_name)
                if wait:
                    started = await _wait_for_service_state(
                        services_manager, raw_service_name, True, connection_timeout
                    )
                    if started:
                        results[raw_service_name] = {
                            "status": "started",
                            "message": f"Service '{raw_service_name}' started successfully",
                        }
                    else:
                        results[raw_service_name] = {
                            "status": "timeout",
                            "message": f"Service '{raw_service_name}' start timed out",
                        }
                else:
                    results[raw_service_name] = {
                        "status": "starting",
                        "message": f"Service '{raw_service_name}' start initiated",
                    }
            except Exception as e:
                logger.exception("Failed to start service %s", raw_service_name)
                results[raw_service_name] = {
                    "status": "error",
                    "message": f"Failed to start service '{raw_service_name}': {e}",
                }
        _format_service_results(results, fmt, "start")
    except Exception as e:
        logger.exception("Failed to start services")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to start services: {e}")


@app.command
async def stop(
    services: Annotated[
        list[ServiceType] | None,
        Parameter(help="Specific services to stop (if not provided, stops all services)"),
    ] = None,
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    force: Annotated[
        bool, Parameter("--force", help="Force stop services without graceful shutdown")
    ] = False,
    connection_timeout: Annotated[
        float,
        Parameter(
            "--timeout",
            "-t",
            alias="timeout",
            help="Timeout in seconds to wait for graceful shutdown",
        ),
    ] = 30.0,
) -> None:
    """Stop CodeWeaver services."""
    try:
        config_path = str(config) if config else None
        services_manager = await ServerManager.get_services_manager(config_path)
        results = {}
        services_to_stop = await _resolve_service_names(services, services_manager, mode="stop")
        for raw_service_name in services_to_stop:
            try:
                existing_service = await services_manager.get_service(raw_service_name)
                if not existing_service:
                    results[raw_service_name] = {
                        "status": "not_running",
                        "message": f"Service '{raw_service_name}' is not running",
                    }
                    continue
                await services_manager.stop_service(raw_service_name)
                stopped = await _wait_for_service_state(
                    services_manager, raw_service_name, False, connection_timeout
                )
                if stopped:
                    results[raw_service_name] = {
                        "status": "stopped",
                        "message": f"Service '{raw_service_name}' stopped successfully",
                    }
                elif force:
                    try:
                        results[raw_service_name] = {
                            "status": "force_stopped",
                            "message": f"Service '{raw_service_name}' force stopped",
                        }
                    except Exception:
                        results[raw_service_name] = {
                            "status": "timeout",
                            "message": f"Service '{raw_service_name}' stop timed out",
                        }
                else:
                    results[raw_service_name] = {
                        "status": "timeout",
                        "message": f"Service '{raw_service_name}' stop timed out (use --force to force stop)",
                    }
            except Exception as e:
                logger.exception("Failed to stop service %s", raw_service_name)
                results[raw_service_name] = {
                    "status": "error",
                    "message": f"Failed to stop service '{raw_service_name}': {e}",
                }
        _format_service_results(results, fmt, "stop")
    except Exception as e:
        logger.exception("Failed to stop services")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to stop services: {e}")


@app.command
async def restart(
    services: Annotated[
        list[ServiceType] | None,
        Parameter(help="Specific services to restart (if not provided, restarts all services)"),
    ] = None,
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    restart_timeout: Annotated[
        float,
        Parameter(
            "--timeout", "-t", alias="timeout", help="Timeout in seconds for stop/start operations"
        ),
    ] = 30.0,
) -> None:
    """Restart CodeWeaver services."""
    try:
        print("Stopping services...")
        await stop(services, config, OutputFormat.TEXT, False, restart_timeout)
        await asyncio.sleep(1.0)
        print("\nStarting services...")
        await start(services, config, format, True, restart_timeout)
    except Exception as e:
        logger.exception("Failed to restart services")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to restart services: {e}")


async def _gather_service_status(services_manager, all_services, detailed):
    """Gather status info for all services."""
    status_data = {"server_running": ServerManager.is_running(), "services": {}}
    for service_name in all_services:
        try:
            service = await services_manager.get_service(service_name)
            is_running = service is not None
            service_info = {"running": is_running, "status": "running" if is_running else "stopped"}
            if detailed and is_running:
                service_info |= {
                    "type": type(service).__name__,
                    "uptime": getattr(service, "uptime", None),
                    "stats": getattr(service, "get_stats", dict)(),
                }
            status_data["services"][service_name] = service_info
        except Exception as e:
            status_data["services"][service_name] = {
                "running": False,
                "status": "error",
                "error": str(e),
            }
    return status_data


@app.command
async def status(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    detailed: Annotated[
        bool, Parameter("--detailed", "-d", help="Show detailed service information")
    ] = False,
) -> None:
    """Show status of CodeWeaver services."""
    try:
        config_path = str(config) if config else None
        services_manager = await ServerManager.get_services_manager(config_path)
        all_services = ServiceType.get_values()
        await services_manager.list_services()
        status_data = await _gather_service_status(services_manager, all_services, detailed)
        if fmt == OutputFormat.JSON:
            print(json.dumps(status_data, indent=2, default=str))
        else:
            print("CodeWeaver Services Status")
            print("=" * 30)
            print(f"Server: {('✓ Running' if status_data['server_running'] else '✗ Not running')}")
            print()
            print("Services:")
            for service_name, info in status_data["services"].items():
                if info["running"]:
                    status_icon = "✓"
                    status_text = "Running"
                elif info["status"] == "error":
                    status_icon = "✗"
                    status_text = f"Error: {info.get('error', 'Unknown error')}"
                else:
                    status_icon = "○"
                    status_text = "Stopped"
                print(f"  {status_icon} {service_name}: {status_text}")
                if detailed and info["running"] and info.get("stats"):
                    for key, value in info["stats"].items():
                        print(f"    {key}: {value}")
    except Exception as e:
        logger.exception("Failed to get service status")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to get service status: {e}")


@app.command(alias="list")
async def list_services(
    config: Annotated[
        Path | None, Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    running_only: Annotated[
        bool, Parameter("--running-only", "-r", help="Show only running services")
    ] = False,
) -> None:
    """List available CodeWeaver services."""
    try:
        if running_only:
            config_path = str(config) if config else None
            services_manager = await ServerManager.get_services_manager(config_path)
            running_services = await services_manager.list_services()
            all_running_services = running_services
        else:
            all_running_services = ServiceType.get_values()
        if fmt == OutputFormat.JSON:
            print(json.dumps({"services": all_running_services}, indent=2))
        else:
            title = "Running Services" if running_only else "Available Services"
            print(f"{title}:")
            for service in all_running_services:
                print(f"  • {service}")
            if not all_running_services:
                print("  No services found")
    except Exception as e:
        logger.exception("Failed to list services")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "services": []}))
        else:
            print(f"Failed to list services: {e}")


if __name__ == "__main__":
    app.parse_args()
