#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: snake-case-functions
"""
Convert logging f-strings to % format using AST parsing.
Handles G004 violations that ast-grep can't easily transform.
"""

import ast
import sys

from pathlib import Path


class FStringConverter(ast.NodeTransformer):
    """Convert f-strings in logging calls to % format."""

    def __init__(self) -> None:
        """Initialize FStringConverter."""
        self.changes_made = False

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Convert f-strings in logging method calls."""
        self.generic_visit(node)

        # Check if this is a logging call
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("logger", "logging", "log")
            and node.func.attr in ("debug", "info", "warning", "error", "critical", "exception")
            and node.args
            and isinstance(node.args[0], ast.JoinedStr)
        ):
            format_str, args = self._convert_fstring(node.args[0])

            # Replace f-string with format string
            node.args[0] = ast.Constant(value=format_str)

            # Add extracted variables as additional arguments
            node.args.extend(args)

            self.changes_made = True

        return node

    def _convert_fstring(self, fstring: ast.JoinedStr) -> tuple[str, list[ast.expr]]:
        """Convert JoinedStr (f-string) to format string and argument list."""
        format_parts = []
        args = []

        for value in fstring.values:
            if isinstance(value, ast.Constant):
                # String literal part
                format_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                # Variable part - convert to %s
                format_parts.append("%s")
                args.append(value.value)

        format_string = "".join(format_parts)
        return format_string, args


def convert_file(file_path: Path) -> bool:
    """Convert f-strings in a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        converter = FStringConverter()
        new_tree = converter.visit(tree)

        if converter.changes_made:
            # Convert back to source
            new_content = ast.unparse(new_tree)
            file_path.write_text(new_content, encoding="utf-8")
            return True

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not process {file_path}: {e}", file=sys.stderr)

    return False


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fstring_converter.py <file_or_directory> [file2] ...")
        sys.exit(1)

    files_changed = 0
    total_files = 0

    for arg in sys.argv[1:]:
        path = Path(arg)

        if path.is_file() and path.suffix == ".py":
            total_files += 1
            if convert_file(path):
                files_changed += 1
                print(f"✅ Converted f-strings in: {path}")

        elif path.is_dir():
            for py_file in path.rglob("*.py"):
                total_files += 1
                if convert_file(py_file):
                    files_changed += 1
                    print(f"✅ Converted f-strings in: {py_file}")

        else:
            print(f"Warning: {path} is not a Python file or directory", file=sys.stderr)

    print(f"\n📊 Processed {total_files} files, converted f-strings in {files_changed} files")


if __name__ == "__main__":
    main()
