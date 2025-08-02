# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Utility functions for CodeWeaver CLI operations."""

from codeweaver.cli.types import CLIError


def raise_cli_error(message: str) -> None:
    """
    Raise a CLIError with the given message.

    Args:
        message: Error message to raise

    Raises:
        CLIError: Always raises with the provided message
    """
    raise CLIError(message)
