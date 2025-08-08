# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Configuration helper for CLI operations.

Handles configuration file generation, MCP config insertion,
and integration with common MCP client applications.
"""

import json
import logging

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, LiteralString, TypedDict

from codeweaver.cli.types import CLIError, MCPTarget
from codeweaver.config import get_config_manager


logger = logging.getLogger(__name__)


class MCPConfigFiles(TypedDict):
    """Typed dictionary for MCP configuration file paths."""

    project: LiteralString
    windows: LiteralString
    macos: LiteralString
    linux: LiteralString


class ConfigHelper:
    """Helper for configuration management and MCP integration."""

    MCP_CONFIG_PATHS: ClassVar[dict[MCPTarget, MCPConfigFiles]] = {
        MCPTarget.CLAUDE_DESKTOP: {
            "project": ".mcp.json",
            "windows": "~/AppData/Roaming/Claude/claude_desktop_config.json",
            "macos": "~/Library/Application Support/Claude/claude_desktop_config.json",
            "linux": "~/.config/claude/claude_desktop_config.json",
        },
        MCPTarget.CLAUDE_CODE: {
            "project": ".mcp.json",
            "windows": "~/.claude.json",
            "macos": "~/.claude.json",
            "linux": "~/.claude.json",
        },
        MCPTarget.CURSOR: {
            "project": ".cursor/mcp.json",
            "windows": "~/AppData/Roaming/Cursor/cursor_config.json",
            "macos": "~/Library/Application Support/Cursor/cursor_config.json",
            "linux": "~/.config/cursor/cursor_config.json",
        },
        MCPTarget.MCP_JSON: {
            "project": ".mcp.json",
            # can be anywhere... this is just a placeholder
            "windows": "~/mcp.json",
            "macos": "~/mcp.json",
            "linux": "~/mcp.json",
        },
        MCPTarget.VSCODE: {
            "project": ".vscode/mcp.json",
            "windows": "~/.vscode/mcp.json",
            "macos": "~/Library/Application Support/Code/User/mcp.json",
            "linux": "~/.config/Code/User/mcp.json",
        },
        MCPTarget.ROO: {
            "project": ".roo/mcp.json",
            "windows": "~/.vscode/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json",
            "macos": "~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json",
            "linux": "~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json",
        },
    }

    @classmethod
    def generate_config(cls, output_path: str, template: str = "default") -> Path:
        """
        Generate CodeWeaver configuration file.

        Args:
            output_path: Path where config file should be written
            template: Configuration template to use

        Returns:
            Path to generated configuration file

        Raises:
            CLIError: If config generation fails
        """
        try:
            output_path = Path(output_path).expanduser().resolve()
            config_manager = get_config_manager()
            default_config = config_manager.get_config()
            config_data = cls._apply_template(default_config, template)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                config_data.model_dump_json(indent=2, exclude_unset=True, exclude_none=True),
                encoding="utf-8",
            )
            logger.info("Generated configuration file: %s", output_path)
        except Exception as e:
            logger.exception("Failed to generate configuration file")
            raise CLIError(f"Failed to generate config: {e}") from e
        else:
            return output_path

    @classmethod
    def _apply_template(cls, config, template: str):
        """Apply template-specific modifications to configuration."""
        if template == "minimal":
            config.logging.level = "WARNING"
            config.services.auto_indexing.enabled = False
            config.services.telemetry.enabled = False
        elif template == "development":
            config.logging.level = "DEBUG"
            config.services.auto_indexing.enabled = True
            config.services.telemetry.enabled = False
        elif template == "production":
            config.logging.level = "INFO"
            config.services.auto_indexing.enabled = True
            config.services.telemetry.enabled = True
        return config

    @classmethod
    def insert_mcp_config(
        cls,
        target: MCPTarget,
        server_name: str = "codeweaver",
        server_path: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        *,
        backup: bool = True,
    ) -> dict[str, str | bool]:
        """
        Insert CodeWeaver MCP server configuration into target client.

        Args:
            target: Target MCP client (claude-desktop, claude-code, etc.)
            server_name: Name for the MCP server entry
            server_path: Path to CodeWeaver server executable
            args: Command line arguments for server
            env: Environment variables for server
            backup: Whether to create backup of existing config

        Returns:
            Dictionary with operation results
        """
        try:
            config_path = cls._find_target_config_path(target)
            if not config_path:
                return {
                    "success": False,
                    "error": f"Could not find configuration file for {target.value}",
                    "config_path": None,
                    "backup_path": None,
                }
            backup_path = None
            if backup and config_path.exists():
                backup_path = cls._create_backup(config_path)
            mcp_config = cls._load_mcp_config(config_path)
            server_config = cls._generate_server_config(
                server_path=server_path, args=args or [], env=env or {}
            )
            if "mcpServers" not in mcp_config:
                mcp_config["mcpServers"] = {}
            mcp_config["mcpServers"][server_name] = server_config
            cls._write_mcp_config(config_path, mcp_config)
            logger.info("Successfully inserted MCP config for %s", target.value)
        except Exception as e:
            logger.exception("Failed to insert MCP config for %s", target.value)
            return {"success": False, "error": str(e), "config_path": None, "backup_path": None}
        else:
            return {
                "success": True,
                "config_path": str(config_path),
                "backup_path": str(backup_path) if backup_path else None,
                "server_name": server_name,
            }

    @classmethod
    def _find_target_config_path(cls, target: MCPTarget) -> Path | None:
        """Find configuration file path for target MCP client."""
        import platform

        system = platform.system().lower()
        if system == "darwin":
            system = "macos"
        elif system not in ["windows", "linux"]:
            system = "linux"
        path_templates = cls.MCP_CONFIG_PATHS.get(target, {})
        path_template = path_templates.get(system)
        if not path_template:
            return None
        if "*" in path_template:
            expanded_paths = Path.glob(str(Path(path_template).expanduser()))
            return Path(expanded_paths[0]) if expanded_paths else None
        path = Path(path_template).expanduser()
        return path if path.parent.exists() else None

    @classmethod
    def _create_backup(cls, config_path: Path) -> Path:
        """Create backup of existing configuration file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".backup_{timestamp}.json")
        import shutil

        shutil.copy2(config_path, backup_path)
        logger.debug("Created backup: %s", backup_path)
        return backup_path

    @classmethod
    def _load_mcp_config(cls, config_path: Path) -> dict:
        """Load existing MCP configuration file."""
        if not config_path.exists():
            return {}
        try:
            with config_path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load existing config %s: %s", config_path, e)
            return {}

    @classmethod
    def _write_mcp_config(cls, config_path: Path, config_data: dict) -> None:
        """Write MCP configuration file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def _generate_server_config(
        cls,
        server_path: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        """Generate MCP server configuration block."""
        if not server_path:
            server_path = "codeweaver"
        config = {"command": server_path, "args": args or []}
        if env:
            config["env"] = env
        return config

    @classmethod
    def list_mcp_configs(cls) -> dict[MCPTarget, dict[str, str | bool]]:
        """
        List MCP client configurations and their status.

        Returns:
            Dictionary mapping targets to their config status
        """
        results = {}
        for target in MCPTarget:
            config_path = cls._find_target_config_path(target)
            if config_path and config_path.exists():
                try:
                    config_data = cls._load_mcp_config(config_path)
                    has_codeweaver = "mcpServers" in config_data and any(
                        "codeweaver" in name.lower() for name in config_data["mcpServers"]
                    )
                    results[target] = {
                        "config_path": str(config_path),
                        "exists": True,
                        "has_codeweaver": has_codeweaver,
                        "server_count": len(config_data.get("mcpServers", {})),
                    }
                except Exception as e:
                    results[target] = {
                        "config_path": str(config_path),
                        "exists": True,
                        "has_codeweaver": False,
                        "error": str(e),
                    }
            else:
                results[target] = {
                    "config_path": str(config_path) if config_path else None,
                    "exists": False,
                    "has_codeweaver": False,
                }
        return results

    @classmethod
    def remove_mcp_config(
        cls, target: MCPTarget, server_name: str = "codeweaver", *, backup: bool = True
    ) -> dict[str, str | bool]:
        """
        Remove CodeWeaver MCP server configuration from target client.

        Args:
            target: Target MCP client
            server_name: Name of server entry to remove
            backup: Whether to create backup before removal

        Returns:
            Dictionary with operation results
        """
        try:
            config_path = cls._find_target_config_path(target)
            if not config_path or not config_path.exists():
                return {
                    "success": False,
                    "error": f"Configuration file not found for {target.value}",
                    "config_path": str(config_path) if config_path else None,
                }
            backup_path = cls._create_backup(config_path) if backup else None
            mcp_config = cls._load_mcp_config(config_path)
            if "mcpServers" in mcp_config and server_name in mcp_config["mcpServers"]:
                del mcp_config["mcpServers"][server_name]
                cls._write_mcp_config(config_path, mcp_config)
                logger.info("Removed MCP server '%s' from %s", server_name, target.value)
                return {
                    "success": True,
                    "config_path": str(config_path),
                    "backup_path": str(backup_path) if backup_path else None,
                    "removed": True,
                }
        except Exception as e:
            logger.exception("Failed to remove MCP config for %s", target.value)
            return {"success": False, "error": str(e), "config_path": None, "backup_path": None}
        else:
            return {
                "success": True,
                "config_path": str(config_path),
                "backup_path": None,
                "removed": False,
                "message": f"Server '{server_name}' not found in configuration",
            }


def generate_config_file(output_path: str, template: str = "default") -> Path:
    """Generate CodeWeaver configuration file."""
    return ConfigHelper.generate_config(output_path, template)


def insert_mcp_server_config(
    target: MCPTarget, server_name: str = "codeweaver", **kwargs
) -> dict[str, str | bool]:
    """Insert MCP server configuration into target client."""
    return ConfigHelper.insert_mcp_config(target, server_name, **kwargs)


def list_mcp_client_configs() -> dict[MCPTarget, dict[str, str | bool]]:
    """List MCP client configurations and status."""
    return ConfigHelper.list_mcp_configs()
