# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: avoid-global-variables
"""
Defines *client* log handler for FastMCP.
"""

import logging

from typing import Any

from fastmcp.client.logging import LogHandler, LogMessage
from pydantic import ConfigDict
from rich.console import Console
from rich.logging import RichHandler


console = Console(stderr=True, markup=True, emoji=True)

DEFAULT_LOG_LEVEL = logging.INFO


async def get_logger() -> logging.Logger:
    """
    Get the logger for CodeWeaver.

    Returns:
        A logger instance configured for CodeWeaver.
    """
    logger = logging.getLogger("[orange]codeweaver[/orange]")
    if not logger.hasHandlers():
        logger.addHandler(async_log_handler)
    return logger


class CodeWeaverLogMessage(LogMessage):
    """Custom log message for CodeWeaver with enhanced formatting and metadata."""

    level: logging.LoggingLevel
    """The severity of this log message."""
    logger: str
    """An optional name of the logger issuing this message."""
    data: Any
    """
    The data to be logged, such as a string message or an object. Any JSON serializable
    type is allowed here.
    """
    model_config = ConfigDict(extra="allow")

    def __init__(self, level: logging.LoggingLevel, data: Any, logger: str = "~~codeweaver~~"):
        """
        Initialize a CodeWeaverLogMessage.

        Args:
            level: The severity of this log message.
            data: The data to be logged.
            logger: An optional name of the logger issuing this message.
        """
        super().__init__(level=level, data=data, logger=logger)


def get_handler() -> LogHandler:
    """
    Get the log handler for CodeWeaver.

    Returns:
        A log handler instance that can be registered with FastMCP.
    """
    return async_log_handler


@LogHandler.register(CodeWeaverLogMessage)
async def async_log_handler(message: CodeWeaverLogMessage) -> None:
    """
    Handle log messages from FastMCP.

    Args:
        message: The log message to handle.
    """
    level = message.level.upper() or DEFAULT_LOG_LEVEL
    logger = message.logger or "[orange]codeweaver[/orange]"
    data = message.data

    # Use RichHandler for better formatting in console
    rich_handler = RichHandler(console=console)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    # Log the message
    logger.add_handler(rich_handler)
    rich_handler.emit(
        logging.LogRecord(
            name=logger, level=level, pathname="", lineno=0, msg=data, args=(), exc_info=None
        )
    )
