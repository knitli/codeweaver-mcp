#!/usr/bin/env python3
"""
Fix TRY300 violations by moving return statements from try blocks to else blocks.
This handles complex cases that ast-grep rules can't handle reliably.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class TryReturnFixer(ast.NodeTransformer):
    """Move return statements from try blocks to else blocks."""

    def __init__(self) -> None:
        self.changes_made = False

    def visit_Try(self, node: ast.Try) -> ast.Try:
        """Transform try blocks that have return statements."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Check if this try block has return statements
        returns_in_try = self._extract_returns_from_try(node.body)
        
        if not returns_in_try:
            return node  # No returns to move

        # Don't transform if there's already an else block with returns
        if node.orelse and self._has_returns(node.orelse):
            return node

        # Don't transform if there's a finally block (more complex case)
        if node.finalbody:
            return node

        # Create new try block without the return statements
        new_body = []
        for stmt in node.body:
            if not isinstance(stmt, ast.Return):
                new_body.append(stmt)

        # If we removed all statements, don't transform
        if not new_body:
            return node

        # Create else block with the return statements
        else_block = returns_in_try

        # Create new try node
        new_try = ast.Try(
            body=new_body,
            handlers=node.handlers,
            orelse=else_block,
            finalbody=node.finalbody,
            lineno=node.lineno,
            col_offset=node.col_offset
        )

        self.changes_made = True
        return new_try

    def _extract_returns_from_try(self, body: List[ast.stmt]) -> List[ast.Return]:
        """Extract return statements from the end of a try block."""
        returns = []
        
        # Look for return statements at the end of the try block
        for stmt in reversed(body):
            if isinstance(stmt, ast.Return):
                returns.insert(0, stmt)
            else:
                # Stop at the first non-return statement
                break
        
        return returns

    def _has_returns(self, stmts: List[ast.stmt]) -> bool:
        """Check if a list of statements contains return statements."""
        return any(isinstance(stmt, ast.Return) for stmt in stmts)


class TryReturnComplexFixer(ast.NodeTransformer):
    """Handle more complex TRY300 cases with conditional returns."""

    def __init__(self) -> None:
        self.changes_made = False

    def visit_Try(self, node: ast.Try) -> ast.Try:
        """Transform try blocks with conditional returns."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Check for complex patterns like if/elif/else with returns
        if self._has_conditional_returns(node.body):
            return self._transform_conditional_returns(node)

        return node

    def _has_conditional_returns(self, body: List[ast.stmt]) -> bool:
        """Check if the try body has conditional return patterns."""
        for stmt in body:
            if isinstance(stmt, ast.If):
                # Check if this if statement has returns in all branches
                if self._if_has_returns(stmt):
                    return True
        return False

    def _if_has_returns(self, if_stmt: ast.If) -> bool:
        """Check if an if statement has returns in its branches."""
        # Check if body has return
        has_body_return = any(isinstance(s, ast.Return) for s in if_stmt.body)
        
        # Check orelse (else/elif)
        has_else_return = False
        if if_stmt.orelse:
            if len(if_stmt.orelse) == 1 and isinstance(if_stmt.orelse[0], ast.If):
                # This is an elif
                has_else_return = self._if_has_returns(if_stmt.orelse[0])
            else:
                # This is an else
                has_else_return = any(isinstance(s, ast.Return) for s in if_stmt.orelse)

        return has_body_return and has_else_return

    def _transform_conditional_returns(self, node: ast.Try) -> ast.Try:
        """Transform try block with conditional returns."""
        # For now, handle simple cases - more complex logic can be added later
        # This is a placeholder for more sophisticated transformations
        return node


def fix_try_returns(source_code: str) -> Tuple[str, bool]:
    """Fix TRY300 violations in source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"❌ Syntax error in source code: {e}")
        return source_code, False

    # Apply simple transformations first
    simple_fixer = TryReturnFixer()
    tree = simple_fixer.visit(tree)

    # Apply complex transformations
    complex_fixer = TryReturnComplexFixer()
    tree = complex_fixer.visit(tree)

    changes_made = simple_fixer.changes_made or complex_fixer.changes_made

    if changes_made:
        # Fix missing locations
        ast.fix_missing_locations(tree)
        
        try:
            return ast.unparse(tree), True
        except Exception as e:
            print(f"❌ Error unparsing AST: {e}")
            return source_code, False
    else:
        return source_code, False


def process_file(file_path: Path) -> bool:
    """Process a single Python file."""
    try:
        original_content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

    fixed_content, changes_made = fix_try_returns(original_content)

    if changes_made:
        try:
            file_path.write_text(fixed_content, encoding='utf-8')
            print(f"✅ Fixed try/return patterns in: {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error writing {file_path}: {e}")
            return False
    else:
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 try_return_fixer.py <file_or_directory> [file_or_directory ...]")
        sys.exit(1)

    total_files = 0
    files_changed = 0

    for arg in sys.argv[1:]:
        path = Path(arg)
        
        if path.is_file() and path.suffix == '.py':
            total_files += 1
            if process_file(path):
                files_changed += 1
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                total_files += 1
                if process_file(py_file):
                    files_changed += 1
        else:
            print(f"⚠️  Skipping {path}: not a Python file or directory")

    print(f"\n📊 Processed {total_files} files, fixed try/return patterns in {files_changed} files")

    if files_changed == 0 and total_files > 0:
        print("✨ No try/return violations found!")


if __name__ == "__main__":
    main()
