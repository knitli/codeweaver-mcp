# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# ruff: noqa: RUF029
# sourcery skip: avoid-global-variables
"""
Defines *client* log handler for FastMCP.
"""

import logging
import os

from enum import IntEnum, unique
from typing import Any, Literal, Self, cast

from fastmcp.client.logging import LogHandler, LogMessage
from pydantic import ConfigDict
from rich.console import Console
from rich.logging import RichHandler


@unique
class LogLevel(IntEnum):
    """Enumeration of log levels for CodeWeaver."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def __str__(self) -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        """Return the string representation of the log level."""
        return self.name

    @classmethod
    def from_string(cls, value: str) -> "LogLevel":
        """
        Convert a string to the corresponding LogLevel.

        Args:
            value: The string representation of the log level.

        Returns:
            The corresponding LogLevel enum member.
        """
        # If the value is a two-digit number, treat it as an integer log level
        if len(value) == 2 and value.isnumeric() and (integer_value := int(value)) in cls._value2member_map_:
            return cast(Self, cls._value2member_map_[integer_value])
        return cls.__members__.get(value.strip().upper(), cls.INFO)


# Create a Rich console for logging output
console = Console(stderr=True, markup=True, emoji=True)

IS_CI: bool = os.environ.get("GH_ACTIONS", "false").lower() == "true"
DEBUG: bool = os.environ.get("CW_DEBUG", "false").lower() == "true"
DEFAULT_LOG_LEVEL: LogLevel = LogLevel.DEBUG if DEBUG else LogLevel.INFO


async def get_logger() -> logging.Logger:
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

    _level: LogLevel
    """The severity of this log message."""
    _logger: str
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
        level = LogLevel.from_string(level) if isinstance(level, str) else LogLevel(level)

        # Process logger
        if not logger:
            logger = "~codeweaver~" if IS_CI else "[orange]codeweaver[/orange]"

        # Process data
        if data is None:
            data = ""

        # Call parent constructor with processed values
        super().__init__(level=str(level), logger=logger, data=data)


def get_handler() -> LogHandler:
    """
    Get the log handler for CodeWeaver.

    Returns:
        A log handler instance that can be registered with FastMCP.
    """
    # Return the actual handler function for FastMCP registration
    return async_log_handler


async def async_log_handler(message: CodeWeaverLogMessage) -> None:
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
