

import logging

from typing import Literal

"""Logging utilities for FastMCP."""

def get_logger(name: str) -> logging.Logger:
    ...

def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int = ...,
    logger: logging.Logger | None = ...,
    enable_rich_tracebacks: bool = ...,
) -> None:
    ...
