# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
MCP client wrapper commands.

Provides CLI commands for interacting with MCP servers using FastMCP client
capabilities, including connection testing, tool listing, and tool execution.
"""

import asyncio
import json
import logging

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeweaver.cli.types import CLIError, OutputFormat
from codeweaver.cli.utils import ClientManager, test_mcp_connection


logger = logging.getLogger(__name__)
app = App(name="client", help="MCP client operations - connect to and interact with MCP servers")


@app.command
async def test(
    server_path: Annotated[str, Parameter(help="Path or URL to MCP server")],
    connection_timeout: Annotated[
        float, Parameter("--timeout", "-t", help="Connection timeout in seconds")
    ] = 30.0,  # noqa: PT028
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,  # noqa: PT028
) -> None:
    """Test connection to an MCP server."""
    try:
        result = await test_mcp_connection(server_path, connection_timeout)
        if fmt == OutputFormat.JSON:
            print(json.dumps(result, indent=2))
        elif result["connected"]:
            print(f"✓ Successfully connected to {server_path}")
            print(f"  Tools available: {result['tools_count']}")
            if result["tools"]:
                print("  Tool names:")
                for tool in result["tools"]:
                    print(f"    - {tool}")
            if result.get("server_info"):
                print(f"  Server info: {result['server_info']}")
        else:
            print(f"✗ Failed to connect to {server_path}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.exception("Connection test failed for %s", server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"connected": False, "error": str(e)}))
        else:
            print(f"✗ Connection test failed: {e}")


@app.command
async def list_tools(
    server_path: Annotated[str, Parameter(help="Path or URL to MCP server")],
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
    detailed: Annotated[
        bool, Parameter("--detailed", "-d", help="Show detailed tool information including schemas")
    ] = False,
) -> None:
    """List tools available from an MCP server."""
    try:
        tools = await ClientManager.list_tools(server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps(tools, indent=2))
        else:
            if not tools:
                print(f"No tools available from {server_path}")
                return
            print(f"Tools available from {server_path}:")
            print(f"Found {len(tools)} tool(s)")
            print()
            for tool in tools:
                print(f"• {tool['name']}")
                if tool.get("description"):
                    print(f"  Description: {tool['description']}")
                if detailed and tool.get("input_schema"):
                    print(f"  Input schema: {json.dumps(tool['input_schema'], indent=4)}")
                print()
    except Exception as e:
        logger.exception("Failed to list tools for %s", server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "tools": []}))
        else:
            print(f"Failed to list tools: {e}")


@app.command
async def call_tool(
    server_path: Annotated[str, Parameter(help="Path or URL to MCP server")],
    tool_name: Annotated[str, Parameter(help="Name of tool to call")],
    arguments: Annotated[
        str | None, Parameter("--args", "-a", help="Tool arguments as JSON string")
    ] = None,
    args_file: Annotated[
        Path | None, Parameter("--args-file", help="Path to file containing tool arguments as JSON")
    ] = None,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Call a specific tool on an MCP server."""
    try:
        tool_args = _parse_tool_arguments(arguments, args_file)
        result = await ClientManager.call_tool(server_path, tool_name, tool_args)
        _print_tool_result(result, tool_name, fmt)
    except Exception as e:
        logger.exception("Failed to call tool %s on %s", tool_name, server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"Tool execution failed: {e}")


def _parse_tool_arguments(arguments: str | None, args_file: Path | None) -> dict:
    """Parse tool arguments from a JSON string or file."""
    tool_args = {}
    if arguments:
        try:
            tool_args = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise CLIError(f"Invalid JSON in arguments: {e}") from e
    elif args_file:
        try:
            with args_file.open("r") as f:
                tool_args = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise CLIError(f"Failed to load arguments from file: {e}") from e
    return tool_args


def _print_tool_result(result: dict, tool_name: str, fmt: OutputFormat) -> None:
    """Print the result of a tool call in the requested format."""
    if fmt == OutputFormat.JSON:
        print(json.dumps(result, indent=2))
    elif result.get("success"):
        print(f"✓ Tool '{tool_name}' executed successfully")
        if result.get("result"):
            print("Result:")
            if isinstance(result["result"], (dict, list)):
                print(json.dumps(result["result"], indent=2))
            else:
                print(result["result"])
        if result.get("structured_content"):
            print("\nStructured content:")
            print(json.dumps(result["structured_content"], indent=2))
    else:
        print(f"✗ Tool '{tool_name}' execution failed")
        print(f"Error: {result.get('error', 'Unknown error')}")


@app.command
async def list_resources(
    server_path: Annotated[str, Parameter(help="Path or URL to MCP server")],
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """List resources available from an MCP server."""
    try:
        resources = await ClientManager.list_resources(server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps(resources, indent=2))
        else:
            if not resources:
                print(f"No resources available from {server_path}")
                return
            print(f"Resources available from {server_path}:")
            print(f"Found {len(resources)} resource(s)")
            print()
            for resource in resources:
                print(f"• {resource['uri']}")
                if resource.get("name"):
                    print(f"  Name: {resource['name']}")
                if resource.get("description"):
                    print(f"  Description: {resource['description']}")
                if resource.get("mime_type"):
                    print(f"  MIME type: {resource['mime_type']}")
                print()
    except Exception as e:
        logger.exception("Failed to list resources for %s", server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "resources": []}))
        else:
            print(f"Failed to list resources: {e}")


@app.command
async def inspect(
    server_path: Annotated[str, Parameter(help="Path or URL to MCP server")],
    connection_timeout: Annotated[
        float, Parameter("--timeout", "-t", help="Connection timeout in seconds")
    ] = 10.0,
    fmt: Annotated[
        OutputFormat, Parameter("--format", "-f", alias="format", help="Output format for results")
    ] = OutputFormat.TEXT,
) -> None:
    """Inspect an MCP server's capabilities (tools and resources)."""
    try:
        connection_result = await test_mcp_connection(server_path, connection_timeout)
        if not connection_result["connected"]:
            _print_inspect_connection_error(server_path, connection_result, fmt)
            return
        tools, resources = await _gather_tools_and_resources(server_path)
        inspection_data = _build_inspection_data(server_path, connection_result, tools, resources)
        _print_inspection_output(inspection_data, fmt)
    except Exception as e:
        logger.exception("Failed to inspect server %s", server_path)
        if fmt == OutputFormat.JSON:
            print(json.dumps({"error": str(e), "connected": False}))
        else:
            print(f"Inspection failed: {e}")


def _print_inspect_connection_error(
    server_path: str, connection_result: dict, fmt: OutputFormat
) -> None:
    if fmt == OutputFormat.JSON:
        print(json.dumps({"error": "Cannot connect to server", "details": connection_result}))
    else:
        print(f"✗ Cannot connect to {server_path}")
        print(f"Error: {connection_result.get('error', 'Unknown error')}")


async def _gather_tools_and_resources(server_path: str) -> tuple[list, list]:
    """Fetch tools and resources concurrently, with fallback on error."""
    tools_task = asyncio.create_task(ClientManager.list_tools(server_path))
    resources_task = asyncio.create_task(ClientManager.list_resources(server_path))
    try:
        tools, resources = await asyncio.gather(tools_task, resources_task)
    except Exception:
        try:
            tools = await ClientManager.list_tools(server_path)
        except Exception:
            tools = []
        try:
            resources = await ClientManager.list_resources(server_path)
        except Exception:
            resources = []
    return tools, resources


def _build_inspection_data(
    server_path: str, connection_result: dict, tools: list, resources: list
) -> dict:
    """Build the inspection data dictionary."""
    return {
        "server_path": server_path,
        "connected": True,
        "tools_count": len(tools),
        "tools": tools,
        "resources_count": len(resources),
        "resources": resources,
        "server_info": connection_result.get("server_info"),
    }


def _print_inspection_output(inspection_data: dict, fmt: OutputFormat) -> None:
    """Print inspection data in the requested format."""
    if fmt == OutputFormat.JSON:
        print(json.dumps(inspection_data, indent=2))
    else:
        print(f"MCP Server Inspection: {inspection_data['server_path']}")
        print("=" * 50)
        print("Connection: ✓ Connected")
        if inspection_data.get("server_info"):
            print(f"Server info: {inspection_data['server_info']}")
        print()
        print(f"Tools ({inspection_data['tools_count']}):")
        tools = inspection_data.get("tools", [])
        if tools:
            for tool in tools:
                print(f"  • {tool['name']}")
                if tool.get("description"):
                    print(f"    {tool['description']}")
        else:
            print("  No tools available")
        print()
        print(f"Resources ({inspection_data['resources_count']}):")
        resources = inspection_data.get("resources", [])
        if resources:
            for resource in resources:
                print(f"  • {resource['uri']}")
                if resource.get("name"):
                    print(f"    Name: {resource['name']}")
                if resource.get("description"):
                    print(f"    Description: {resource['description']}")
        else:
            print("  No resources available")


@app.command
async def close(
    server_path: Annotated[
        str | None,
        Parameter(help="Path or URL to MCP server (if not provided, closes all connections)"),
    ] = None,
) -> None:
    """Close MCP client connection(s)."""
    try:
        if server_path:
            await ClientManager.close_client(server_path)
            print(f"Closed connection to {server_path}")
        else:
            await ClientManager.close_all()
            print("Closed all MCP client connections")
    except Exception as e:
        logger.exception("Failed to close client connection(s)")
        print(f"Failed to close connection(s): {e}")


if __name__ == "__main__":
    app.parse_args()
