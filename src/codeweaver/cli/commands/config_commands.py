# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Configuration management commands.

Provides CLI commands for generating configuration files, managing MCP client
integrations, and handling CodeWeaver configuration settings.
"""

import json
import logging

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeweaver.cli.types import MCPTarget, OutputFormat
from codeweaver.cli.utils import (
    ConfigHelper,
    generate_config_file,
    insert_mcp_server_config,
    list_mcp_client_configs,
    raise_cli_error,
)


logger = logging.getLogger(__name__)

# Create config command group
app = App(
    name="config", help="Configuration management - generate configs and manage MCP integrations"
)


@app.command
async def generate(
    output: Annotated[str, Parameter(help="Output path for configuration file")],
    template: Annotated[
        str, Parameter("--template", "-t", help="Configuration template to use")
    ] = "default",
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    overwrite: Annotated[
        bool, Parameter("--overwrite", help="Overwrite existing configuration file")
    ] = False,
) -> None:
    """Generate a CodeWeaver configuration file."""
    try:
        output_path = Path(output).expanduser().resolve()

        # Check if file exists and overwrite is not set
        if output_path.exists() and not overwrite:
            raise_cli_error(
                "Configuration file already exists: %s. Use --overwrite to replace it.", output_path
            )

        # Generate configuration
        config_path = generate_config_file(str(output_path), template)

        result = {
            "status": "generated",
            "message": "Configuration file generated successfully",
            "path": str(config_path),
            "template": template,
        }

        # Format output
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        else:
            print(f"✓ {result['message']}")
            print(f"  Path: {result['path']}")
            print(f"  Template: {result['template']}")

            # Show available templates
            print("\nAvailable templates: default, minimal, development, production")

    except Exception as e:
        logger.exception("Failed to generate configuration file")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to generate configuration file: {e}")


@app.command
async def validate(
    config_path: Annotated[str, Parameter(help="Path to configuration file to validate")],
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Validate a CodeWeaver configuration file."""
    try:
        from codeweaver.config.manager import get_config_manager

        config_file = Path(config_path).expanduser().resolve()

        if not config_file.exists():
            raise_cli_error("Configuration file not found: %s", config_file)

        # Attempt to load and validate configuration
        try:
            config_manager = get_config_manager(str(config_file))
            config = config_manager.get_config()

            result = {
                "status": "valid",
                "message": "Configuration file is valid",
                "path": str(config_file),
                "config_type": type(config).__name__,
            }

            # Add configuration summary
            if hasattr(config, "model_dump"):
                config_obj = config.model_dump()
                result["summary"] = {
                    "sections": list(config_obj.keys()),
                    "services": list(config_obj.get("services", {}).keys())
                    if "services" in config_obj
                    else [],
                }

        except Exception as validation_error:
            result = {
                "status": "invalid",
                "message": "Configuration file is invalid",
                "path": str(config_file),
                "error": str(validation_error),
            }

        # Format output
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        elif result["status"] == "valid":
            print(f"✓ {result['message']}")
            print(f"  Path: {result['path']}")
            print(f"  Type: {result['config_type']}")

            if result.get("summary"):
                summary = result["summary"]
                print(f"  Sections: {', '.join(summary['sections'])}")
                if summary.get("services"):
                    print(f"  Services: {', '.join(summary['services'])}")
        else:
            print(f"✗ {result['message']}")
            print(f"  Path: {result['path']}")
            print(f"  Error: {result['error']}")

    except Exception as e:
        logger.exception("Failed to validate configuration file")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "status": "error"}))
        else:
            print(f"Failed to validate configuration file: {e}")


@app.command
async def mcp_insert(
    target: Annotated[MCPTarget, Parameter(help="Target MCP client to configure")],
    server_name: Annotated[
        str, Parameter("--name", "-n", help="Name for the MCP server entry")
    ] = "codeweaver",
    server_path: Annotated[
        str | None, Parameter("--path", "-p", help="Path to CodeWeaver server executable")
    ] = None,
    args: Annotated[
        list[str] | None, Parameter("--args", "-a", help="Command line arguments for server")
    ] = None,
    env_vars: Annotated[
        list[str] | None, Parameter("--env", "-e", help="Environment variables (KEY=VALUE format)")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    backup: Annotated[
        bool, Parameter("--backup", "-b", help="Create backup of existing configuration")
    ] = True,
    force: Annotated[
        bool, Parameter("--force", help="Force insertion even if server already exists")
    ] = False,
) -> None:
    """Insert CodeWeaver MCP server configuration into target client."""
    try:
        # Parse environment variables
        env_obj = {}
        if env_vars:
            for env_var in env_vars:
                if "=" not in env_var:
                    raise_cli_error(
                        "Invalid environment variable format: %s. Use KEY=VALUE format.", env_var
                    )
                key, value = env_var.split("=", 1)
                env_obj[key] = value

        # Insert MCP configuration
        result = insert_mcp_server_config(
            target=target,
            server_name=server_name,
            server_path=server_path,
            args=args,
            env=env_obj or None,
            backup=backup,
        )

        # Format output
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        elif result["success"]:
            print("✓ Successfully inserted MCP server configuration")
            print(f"  Target: {target.value}")
            print(f"  Server name: {result['server_name']}")
            print(f"  Config path: {result['config_path']}")

            if result.get("backup_path"):
                print(f"  Backup created: {result['backup_path']}")
        else:
            print("✗ Failed to insert MCP server configuration")
            print(f"  Target: {target.value}")
            print(f"  Error: {result['error']}")

    except Exception as e:
        logger.exception("Failed to insert MCP configuration")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"Failed to insert MCP configuration: {e}")


@app.command
async def mcp_remove(
    target: Annotated[MCPTarget, Parameter(help="Target MCP client to modify")],
    server_name: Annotated[
        str, Parameter("--name", "-n", help="Name of MCP server entry to remove")
    ] = "codeweaver",
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    backup: Annotated[
        bool, Parameter("--backup", "-b", help="Create backup before removal")
    ] = True,
) -> None:
    """Remove CodeWeaver MCP server configuration from target client."""
    try:
        result = ConfigHelper.remove_mcp_config(
            target=target, server_name=server_name, backup=backup
        )

        # Format output
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        elif result["success"]:
            if result.get("removed", False):
                print("✓ Successfully removed MCP server configuration")
                print(f"  Target: {target.value}")
                print(f"  Server name: {server_name}")
                print(f"  Config path: {result['config_path']}")

                if result.get("backup_path"):
                    print(f"  Backup created: {result['backup_path']}")
            else:
                print("ℹ MCP server configuration not found")  # noqa: RUF001
                print(f"  Target: {target.value}")
                print(f"  Server name: {server_name}")
                print(f"  Message: {result.get('message', 'Server not found in configuration')}")
        else:
            print("✗ Failed to remove MCP server configuration")
            print(f"  Target: {target.value}")
            print(f"  Error: {result['error']}")

    except Exception as e:
        logger.exception("Failed to remove MCP configuration")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"Failed to remove MCP configuration: {e}")


@app.command
def _filter_mcp_configs(configs, show_all):
    """Filter configs based on show_all flag."""
    if not show_all:
        return {target: info for target, info in configs.items() if info.get("exists", False)}
    return configs


def _mcp_list_json(configs):
    """Format configs for JSON output."""
    print(json.dumps({target.value: info for target, info in configs.items()}, indent=2))


def _print_mcp_list_text(configs, show_all):
    """Print MCP client configurations in text format."""
    print("MCP Client Configurations")
    print("=" * 25)

    if not configs:
        print("No MCP client configurations found.")
        if not show_all:
            print("Use --all to show all potential clients.")
        return

    for target, info in configs.items():
        if info.get("exists", False):
            has_codeweaver = info.get("has_codeweaver", False)
            server_count = info.get("server_count", 0)

            status_icon = "✓" if has_codeweaver else "○"
            codeweaver_status = "Has CodeWeaver" if has_codeweaver else "No CodeWeaver"

            print(f"{status_icon} {target.value}: {codeweaver_status}")
            print(f"    Path: {info['config_path']}")
            print(f"    Servers: {server_count}")

            if info.get("error"):
                print(f"    Error: {info['error']}")
        else:
            print(f"○ {target.value}: No configuration file")
            if info.get("config_path"):
                print(f"    Expected path: {info['config_path']}")

        print()


@app.command
async def mcp_list(
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    show_all: Annotated[
        bool,
        Parameter("--all", "-a", help="Show all MCP clients, even if config files don't exist"),
    ] = False,
) -> None:
    """List MCP client configurations and their status."""
    try:
        configs = list_mcp_client_configs()
        configs = _filter_mcp_configs(configs, show_all)

        if fmt == OutputFormat.JSON:
            _mcp_list_json(configs)
        else:
            _print_mcp_list_text(configs, show_all)

    except Exception as e:
        logger.exception("Failed to list MCP configurations")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to list MCP configurations: {e}")


def _build_mcp_status_for_target(mcp_target):
    """Build the status dictionary for a single MCP target."""
    try:
        config_path = ConfigHelper._find_target_config_path(mcp_target)
        if config_path and config_path.exists():
            config_data = ConfigHelper._load_mcp_config(config_path)
            mcp_servers = config_data.get("mcpServers", {})
            codeweaver_servers = {
                name: config for name, config in mcp_servers.items() if "codeweaver" in name.lower()
            }
            return {
                "exists": True,
                "config_path": str(config_path),
                "total_servers": len(mcp_servers),
                "codeweaver_servers": codeweaver_servers,
                "has_codeweaver": len(codeweaver_servers) > 0,
            }
        return {
            "exists": False,
            "config_path": str(config_path) if config_path else None,
            "total_servers": 0,
            "codeweaver_servers": {},
            "has_codeweaver": False,
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "total_servers": 0,
            "codeweaver_servers": {},
            "has_codeweaver": False,
        }


def _mcp_status_json(status_results):
    """Print MCP status in JSON format."""
    print(json.dumps({target.value: info for target, info in status_results.items()}, indent=2))


def _print_mcp_status_text(status_results):
    """Print MCP status in text format."""
    print("MCP Client Status")
    print("=" * 17)

    for mcp_target, info in status_results.items():
        if info.get("exists", False):
            has_codeweaver = info.get("has_codeweaver", False)
            total_servers = info.get("total_servers", 0)
            codeweaver_count = len(info.get("codeweaver_servers", {}))

            status_icon = "✓" if has_codeweaver else "○"
            print(f"{status_icon} {mcp_target.value}")
            print(f"    Config: {info['config_path']}")
            print(f"    Total servers: {total_servers}")
            print(f"    CodeWeaver servers: {codeweaver_count}")

            # Show CodeWeaver server details
            if info.get("codeweaver_servers"):
                print("    CodeWeaver configurations:")
                for name, config in info["codeweaver_servers"].items():
                    print(f"      • {name}")
                    print(f"        Command: {config.get('command', 'N/A')}")
                    if config.get("args"):
                        print(f"        Args: {' '.join(config['args'])}")
                    if config.get("env"):
                        print(f"        Env vars: {len(config['env'])} defined")
        else:
            print(f"○ {mcp_target.value}: No configuration")
            if info.get("config_path"):
                print(f"    Expected: {info['config_path']}")
            if info.get("error"):
                print(f"    Error: {info['error']}")

        print()


@app.command
async def mcp_status(
    target: Annotated[
        MCPTarget | None,
        Parameter(help="Specific MCP client to check (if not provided, checks all)"),
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Show detailed status of MCP client configurations."""
    try:
        targets = [target] if target else list(MCPTarget)
        status_results = {
            mcp_target: _build_mcp_status_for_target(mcp_target) for mcp_target in targets
        }

        if fmt == OutputFormat.JSON:
            _mcp_status_json(status_results)
        else:
            _print_mcp_status_text(status_results)

    except Exception as e:
        logger.exception("Failed to get MCP status")
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Failed to get MCP status: {e}")


@app.command
async def templates(
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """List available configuration templates."""
    templates_info = {
        "default": {
            "name": "Default",
            "description": "Standard configuration with balanced settings",
            "features": ["All services enabled", "Moderate logging", "Standard telemetry"],
        },
        "minimal": {
            "name": "Minimal",
            "description": "Lightweight configuration for basic usage",
            "features": ["Auto-indexing disabled", "Warning level logging", "No telemetry"],
        },
        "development": {
            "name": "Development",
            "description": "Configuration optimized for development",
            "features": ["Auto-indexing enabled", "Debug logging", "No telemetry"],
        },
        "production": {
            "name": "Production",
            "description": "Configuration optimized for production use",
            "features": ["All services enabled", "Info logging", "Full telemetry"],
        },
    }

    if fmt == OutputFormat.JSON:
        print(json.dumps(templates_info, indent=2))
    else:
        print("Available Configuration Templates")
        print("=" * 33)

        for template_id, info in templates_info.items():
            print(f"• {template_id} ({info['name']})")
            print(f"  Description: {info['description']}")
            print("  Features:")
            for feature in info["features"]:
                print(f"    - {feature}")
            print()

        print("Usage: codeweaver config generate <output_path> --template <template_name>")


if __name__ == "__main__":
    # Allow running individual command module for testing
    app.parse_args()
