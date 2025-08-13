# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Helper functions for CodeWeaver utilities.
"""

from functools import cache
from pathlib import Path


def walk_down_to_git_root(path: Path | None = None) -> Path:
    """Walk up the directory tree until a .git directory is found."""
    if path is None:
        path = Path.cwd()
    if path.is_file():
        path = path.parent
    while path != path.parent:
        if (path / ".git").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")


def in_codeweaver_clone(path: Path) -> bool:
    """Check if the current repo is CodeWeaver."""
    return "codeweaver" in str(path).lower() or "code-weaver" in str(path).lower()


def estimate_tokens(text: str | bytes, encoder: str = "cl100k_base") -> int:
    """Estimate the number of tokens in a text."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoder)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return len(encoding.encode(text))


@cache
def normalize_ext(ext: str) -> str:
    """Normalize a file extension to a standard format."""
    return ext.lower().strip() if ext.startswith(".") else f".{ext.lower().strip()}"
