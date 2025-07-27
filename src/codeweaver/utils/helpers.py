# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Helper functions for CodeWeaver utilities.
"""

from pathlib import Path


def walk_down_to_git_root(path: Path) -> Path:
    """Walk up the directory tree until a .git directory is found."""
    if path.is_file():
        path = path.parent
    while path != path.parent:
        if (path / ".git").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")
