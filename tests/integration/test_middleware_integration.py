# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration tests for FastMCP middleware components.

Tests the FastMCP middleware integration including chunking and filtering
middleware to ensure they work correctly with the new plugin architecture.
"""

import tempfile

from pathlib import Path

import pytest

from codeweaver.middleware.chunking import ChunkingMiddleware
from codeweaver.middleware.filtering import FileFilteringMiddleware


class TestChunkingMiddleware:
    """Test ChunkingMiddleware functionality."""

    def test_middleware_initialization(self) -> None:
        """Test chunking middleware initializes correctly."""
        config = {"max_chunk_size": 1500, "min_chunk_size": 50, "ast_grep_enabled": True}

        middleware = ChunkingMiddleware(config)

        assert middleware.max_chunk_size == 1500
        assert middleware.min_chunk_size == 50
        assert middleware.ast_grep_enabled is True

    def test_middleware_initialization_defaults(self) -> None:
        """Test chunking middleware with default configuration."""
        middleware = ChunkingMiddleware()

        assert middleware.max_chunk_size == 1500
        assert middleware.min_chunk_size == 50
        assert isinstance(middleware.ast_grep_enabled, bool)

    @pytest.mark.asyncio
    async def test_chunk_python_file(self) -> None:
        """Test chunking a Python file."""
        middleware = ChunkingMiddleware()

        python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

# Global variable
CONSTANT = 42
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            temp_path = Path(f.name)

        try:
            chunks = await middleware.chunk_file(temp_path, python_code)

            # Should have chunks for function and class
            assert len(chunks) >= 1

            # Check chunk properties
            for chunk in chunks:
                assert chunk.content
                assert chunk.file_path == str(temp_path)
                assert chunk.language == "python"
                assert chunk.chunk_type in ["function", "class", "fallback_chunk"]
                assert isinstance(chunk.metadata, dict)

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_chunk_javascript_file(self) -> None:
        """Test chunking a JavaScript file."""
        middleware = ChunkingMiddleware()

        js_code = """
function greetUser(name) {
    console.log(`Hello, ${name}!`);
    return `Hello, ${name}!`;
}

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }

    async fetchUser(id) {
        const response = await fetch(`${this.apiUrl}/users/${id}`);
        return response.json();
    }
}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            temp_path = Path(f.name)

        try:
            chunks = await middleware.chunk_file(temp_path, js_code)

            # Should have chunks for function and class
            assert len(chunks) >= 1

            # Check chunk properties
            for chunk in chunks:
                assert chunk.content
                assert chunk.file_path == str(temp_path)
                assert chunk.language == "javascript"
                assert chunk.chunk_type in ["function", "class", "fallback_chunk"]

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_chunk_unsupported_language(self) -> None:
        """Test chunking file with unsupported language falls back correctly."""
        middleware = ChunkingMiddleware()

        unknown_code = """
Some random text file content
that doesn't match any known programming language
but should still be chunked using fallback method.

This is another paragraph to ensure
we get proper chunking behavior.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(unknown_code)
            temp_path = Path(f.name)

        try:
            chunks = await middleware.chunk_file(temp_path, unknown_code)

            # Should have at least one fallback chunk
            assert len(chunks) >= 1

            # Check chunk properties
            for chunk in chunks:
                assert chunk.content
                assert chunk.file_path == str(temp_path)
                assert chunk.language == "unknown"
                assert chunk.chunk_type == "fallback_chunk"
                assert chunk.metadata.get("ast_grep_used") is False

        finally:
            temp_path.unlink(missing_ok=True)

    def test_get_supported_languages(self) -> None:
        """Test getting supported languages information."""
        middleware = ChunkingMiddleware()

        info = middleware.get_supported_languages()

        assert isinstance(info, dict)
        assert "ast_grep_available" in info
        assert "ast_grep_enabled" in info
        assert "supported_languages" in info
        assert "language_extensions" in info
        assert "chunk_patterns" in info
        assert "config" in info

        # Check that we have expected languages
        supported_langs = info["supported_languages"]
        assert "python" in supported_langs
        assert "javascript" in supported_langs
        assert "typescript" in supported_langs


class TestFileFilteringMiddleware:
    """Test FileFilteringMiddleware functionality."""

    def test_middleware_initialization(self) -> None:
        """Test filtering middleware initializes correctly."""
        config = {
            "use_gitignore": True,
            "max_file_size": "2MB",
            "excluded_dirs": ["node_modules", ".git"],
            "included_extensions": [".py", ".js"],
        }

        middleware = FileFilteringMiddleware(config)

        assert middleware.use_gitignore is True
        assert middleware.max_file_size == 2 * 1024 * 1024  # 2MB in bytes
        assert "node_modules" in middleware.excluded_dirs
        assert ".git" in middleware.excluded_dirs

    def test_middleware_initialization_defaults(self) -> None:
        """Test filtering middleware with default configuration."""
        middleware = FileFilteringMiddleware()

        assert middleware.use_gitignore is True
        assert middleware.max_file_size == 1024 * 1024  # 1MB default
        assert isinstance(middleware.excluded_dirs, set)

    @pytest.mark.asyncio
    async def test_find_files_basic(self) -> None:
        """Test basic file finding functionality."""
        middleware = FileFilteringMiddleware()

        # Create a temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test.py").write_text("print('hello')")
            (temp_path / "test.js").write_text("console.log('hello');")
            (temp_path / "README.md").write_text("# Test Project")

            # Create excluded directory
            excluded_dir = temp_path / "node_modules"
            excluded_dir.mkdir()
            (excluded_dir / "package.js").write_text("// should be excluded")

            files = await middleware.find_files(temp_path)

            # Should find the main files but not the excluded ones
            file_names = [f.name for f in files]
            assert "test.py" in file_names
            assert "test.js" in file_names
            assert "README.md" in file_names
            assert "package.js" not in file_names  # Should be excluded

    @pytest.mark.asyncio
    async def test_find_files_with_patterns(self) -> None:
        """Test file finding with specific patterns."""
        middleware = FileFilteringMiddleware()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "app.py").write_text("# Python file")
            (temp_path / "script.js").write_text("// JavaScript file")
            (temp_path / "config.json").write_text('{"key": "value"}')
            (temp_path / "README.txt").write_text("Text file")

            # Find only Python files
            python_files = await middleware.find_files(temp_path, ["*.py"])
            file_names = [f.name for f in python_files]
            assert "app.py" in file_names
            assert "script.js" not in file_names
            assert "config.json" not in file_names

            # Find Python and JavaScript files
            code_files = await middleware.find_files(temp_path, ["*.py", "*.js"])
            file_names = [f.name for f in code_files]
            assert "app.py" in file_names
            assert "script.js" in file_names
            assert "config.json" not in file_names

    def test_parse_file_size(self) -> None:
        """Test file size parsing functionality."""
        middleware = FileFilteringMiddleware()

        # Test various size formats
        assert middleware._parse_size("1024") == 1024
        assert middleware._parse_size("1KB") == 1024
        assert middleware._parse_size("2MB") == 2 * 1024 * 1024
        assert middleware._parse_size("1GB") == 1024 * 1024 * 1024
        assert middleware._parse_size("1.5MB") == int(1.5 * 1024 * 1024)


class TestMiddlewareIntegration:
    """Test middleware components working together."""

    @pytest.mark.asyncio
    async def test_chunking_and_filtering_integration(self) -> None:
        """Test chunking and filtering middleware working together."""
        chunking_middleware = ChunkingMiddleware({"max_chunk_size": 500, "min_chunk_size": 20})

        filtering_middleware = FileFilteringMiddleware({
            "excluded_dirs": ["__pycache__", ".git"],
            "max_file_size": "1MB",
        })

        # Create a test project structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Python files
            (temp_path / "main.py").write_text('''
def main():
    """Main function."""
    print("Hello from main!")

class Application:
    """Main application class."""

    def run(self):
        """Run the application."""
        print("Running application...")
        return True
''')

            (temp_path / "utils.py").write_text('''
def helper_function():
    """A helper function."""
    return "helper"

def another_function():
    """Another function."""
    return "another"
''')

            # Create excluded directory
            excluded_dir = temp_path / "__pycache__"
            excluded_dir.mkdir()
            (excluded_dir / "cache.pyc").write_text("cached data")

            # Step 1: Filter files
            filtered_files = await filtering_middleware.find_files(temp_path, ["*.py"])

            # Should find Python files but not cache files
            file_names = [f.name for f in filtered_files]
            assert "main.py" in file_names
            assert "utils.py" in file_names
            assert "cache.pyc" not in file_names

            # Step 2: Chunk the filtered files
            all_chunks = []
            for file_path in filtered_files:
                content = file_path.read_text()
                chunks = await chunking_middleware.chunk_file(file_path, content)
                all_chunks.extend(chunks)

            # Should have chunks from both files
            assert len(all_chunks) >= 2  # At least some chunks

            # Verify chunk properties
            file_paths = {chunk.file_path for chunk in all_chunks}
            assert len(file_paths) == 2  # Should have chunks from both files

            for chunk in all_chunks:
                assert chunk.content
                assert chunk.language == "python"
                assert len(chunk.content) >= chunking_middleware.min_chunk_size
                assert len(chunk.content) <= chunking_middleware.max_chunk_size
