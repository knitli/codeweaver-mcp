# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: avoid-global-variables
"""
Defines *client* log handler for FastMCP.
"""

import logging
import os

from typing import Any, Literal

from fastmcp.client.logging import LogHandler, LogMessage
from pydantic import ConfigDict
from rich.console import Console
from rich.logging import RichHandler


console = Console(stderr=True, markup=True, emoji=True)

LogLevel = Literal[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

IS_CI: bool = os.environ.get("GH_ACTIONS", "false").lower() == "true"
DEBUG: bool = os.environ.get("CW_DEBUG", "false").lower() == "true"
DEFAULT_LOG_LEVEL: LogLevel = logging.DEBUG if DEBUG else logging.INFO


async def get_logger() -> logging.Logger:  # noqa: RUF029
    """
    Get the logger for CodeWeaver.

    Returns:
        A logger instance configured for CodeWeaver.
    """
    logger = logging.getLogger("[orange]codeweaver[/orange]")
    if not logger.hasHandlers():
        # Create a proper logging handler, not the FastMCP handler function
        handler = RichHandler(console=console, markup=True, rich_tracebacks=DEBUG)

        if IS_CI:
            # Use StreamHandler for CI environments
            handler = logging.StreamHandler()

        handler.setLevel(DEFAULT_LOG_LEVEL)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(DEFAULT_LOG_LEVEL)
    return logger


class CodeWeaverLogMessage(LogMessage):
    """Custom log message for CodeWeaver with enhanced formatting and metadata."""

    level: LogLevel
    """The severity of this log message."""
    logger: str
    """An optional name of the logger issuing this message."""
    data: Any
    """
    The data to be logged, such as a string message or an object. Any JSON serializable
    type is allowed here.
    """
    model_config = ConfigDict(extra="allow")

    def __init__(
        self, level: LogLevel | str | None = None, logger: str | None = None, data: Any = None
    ):
        """
        Initialize a CodeWeaverLogMessage.

        Args:
            level: The severity of this log message.
            data: The data to be logged.
            logger: An optional name of the logger issuing this message.
        """
        # Process level
        if not level:
            level = DEFAULT_LOG_LEVEL
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Process logger
        if not logger:
            logger = "~codeweaver~" if IS_CI else "[orange]codeweaver[/orange]"

        # Process data
        if data is None:
            data = ""

        # Call parent constructor with processed values
        super().__init__(level=level, logger=logger, data=data)


def get_handler() -> LogHandler:
    """
    Get the log handler for CodeWeaver.

    Returns:
        A log handler instance that can be registered with FastMCP.
    """
    # Return the actual handler function for FastMCP registration
    return async_log_handler


async def async_log_handler(message: CodeWeaverLogMessage) -> None:  # noqa: RUF029
    """
    Handle log messages from FastMCP.

    Args:
        message: The log message to handle.
    """
    level = message.level or DEFAULT_LOG_LEVEL
    logger_name = message.logger or "[orange]codeweaver[/orange]"
    data = message.data

    # Get the actual logger instance
    logger = logging.getLogger(logger_name)

    # Use RichHandler for better formatting in console
    handler = RichHandler(console=console, markup=True, rich_tracebacks=DEBUG)

    if IS_CI:
        # If running in GitHub Actions, use the default logging
        handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Add handler to logger if it doesn't have any handlers
    if not logger.hasHandlers():
        logger.addHandler(handler)
        logger.setLevel(level)

    # Log the message using the logger
    logger.log(level, data)
