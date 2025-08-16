# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""MCP tools for CodeWeaver."""

import contextlib


DUCKDUCKGO_AVAILABLE = False
TAVILY_AVAILABLE = False

# TODO: replace these constants with registrations to the providers registry
with contextlib.suppress(ImportError):
    from pydantic_ai.common_tools import duckduckgo as duckduckgo_tool

    DUCKDUCKGO_AVAILABLE = True  # type: ignore

with contextlib.suppress(ImportError):
    from pydantic_ai.common_tools import tavily as tavily_tool

    TAVILY_AVAILABLE = True  # type: ignore


from codeweaver.tools.find_code import basic_text_search, find_code_implementation


__all__ = ("basic_text_search", "duckduckgo_tool", "find_code_implementation", "tavily_tool")
