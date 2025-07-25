#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Test script for Voyage AI Code Embeddings MCP Server.

This script tests the core functionality without requiring MCP client setup.
Run this to verify your API keys and basic functionality work correctly.

Usage:
    python test_server.py /path/to/test/codebase
"""

import asyncio
import os
import sys
import tempfile

from pathlib import Path
from typing import Literal


# Add the server to path
sys.path.insert(0, Path(__file__).parent.parent.parent.as_posix())

from codeweaver.server import CodeChunker, CodeEmbeddingsServer


def test_environment() -> bool:
    """Test environment variables and API connections."""
    print("🧪 Testing Environment Setup...")

    # Check environment variables
    required_vars = ["CW_EMBEDDING_API_KEY", "CW_VECTOR_BACKEND_URL"]
    missing_vars = []

    missing_vars.extend(var for var in required_vars if not os.getenv(var))
    if missing_vars:
        return _print_env_missing_vars_message(missing_vars)
    print("✅ Environment variables configured")

    # Test Voyage AI connection
    try:
        import voyageai

        client = voyageai.Client(api_key=os.getenv("CW_VOYAGE_API_KEY"))

        # Test embedding
        result = client.embed(
            texts=["def hello_world():\n    print('Hello, World!')"],
            model="voyage-code-3",
            input_type="document",
            output_dimension=1024,
        )

        if result.embeddings and len(result.embeddings[0]) == 1024:
            print("✅ Voyage AI API connection successful")
        else:
            print("❌ Voyage AI API returned unexpected response")
            return False

    except Exception as e:
        print(f"❌ Voyage AI API connection failed: {e}")
        return False

    # Test Qdrant connection
    try:
        from qdrant_client import QdrantClient

        qdrant = QdrantClient(
            url=os.getenv("CW_VECTOR_BACKEND_URL"), api_key=os.getenv("CW_VECTOR_BACKEND_API_KEY")
        )

        # Test connection
        qdrant.get_collections()
        print("✅ Qdrant connection successful")

    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

    return True


def _print_env_missing_vars_message(missing_vars):
    print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    print("\nSet these environment variables:")
    print("  export CW_VOYAGE_API_KEY='your-voyage-ai-key'")
    print("  export CW_VECTOR_BACKEND_URL='https://your-cluster.qdrant.tech:6333'")
    print("  export CW_VECTOR_BACKEND_API_KEY='your-qdrant-key'  # Optional")
    return False


def test_chunker() -> bool:
    """Test the code chunker with sample files."""
    print("\n🔧 Testing Code Chunker...")

    chunker = CodeChunker()

    # Test Python chunking
    python_code = '''
import os
from typing import list

class DatabaseManager:
    """Manages database connections."""

    def __init__(self, url: str):
        self.url = url

    def connect(self) -> bool:
        """Connect to database."""
        try:
            # Connection logic here
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

def get_user_data(user_id: int) -> dict:
    """Fetch user data from database."""
    manager = DatabaseManager("postgresql://localhost")

    if manager.connect():
        # Query logic here
        return {"id": user_id, "name": "Test User"}

    return {}

# Global configuration
CONFIG = {
    "database_url": "postgresql://localhost",
    "debug": True
}
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_code)
        temp_path = Path(f.name)

    try:
        chunks = chunker.chunk_file(temp_path, python_code)

        print(f"✅ Python chunking: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i + 1}: {chunk.chunk_type} ({chunk.start_line}-{chunk.end_line})")

        # Test JavaScript chunking
        js_code = """
class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }

    async fetchUser(id) {
        try {
            const response = await fetch(`${this.apiUrl}/users/${id}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching user:', error);
            throw error;
        }
    }
}

function validateEmail(email) {
    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return regex.test(email);
}

const userService = new UserService('https://api.example.com');
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            js_temp_path = Path(f.name)

        js_chunks = chunker.chunk_file(js_temp_path, js_code)
        print(f"✅ JavaScript chunking: {len(js_chunks)} chunks created")

    finally:
        # Clean up temp files
        temp_path.unlink(missing_ok=True)
        js_temp_path.unlink(missing_ok=True)

    return True


async def test_full_workflow(test_path: str | None = None) -> bool:  # noqa: PT028
    # sourcery skip: avoid-global-variables, no-long-functions
    """Test the complete indexing and search workflow."""
    print("\n🚀 Testing Full Workflow...")

    # Create test codebase if none provided
    if not test_path:
        test_dir = Path(tempfile.mkdtemp(prefix="voyage_test_"))
        print(f"Creating test codebase in: {test_dir}")

        # Create sample Python files
        (test_dir / "main.py").write_text('''
from auth import authenticate_user
from db import get_connection

def main():
    """Main application entry point."""
    user = authenticate_user("admin", "password")
    if user:
        conn = get_connection()
        print("Application started successfully")
    else:
        print("Authentication failed")

if __name__ == "__main__":
    main()
''')

        (test_dir / "auth.py").write_text('''
import hashlib
from typing import Optional

class User:
    def __init__(self, username: str, password_hash: str):
        self.username = username
        self.password_hash = password_hash

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    # Mock authentication logic
    expected_hash = hash_password(password)

    if username == "admin" and password == "password":
        return User(username, expected_hash)

    return None

def validate_session(session_token: str) -> bool:
    """Validate a session token."""
    # Mock session validation
    return len(session_token) > 10
''')

        (test_dir / "db.py").write_text('''
import sqlite3
from typing import Optional

class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Connect to the database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a SQL query."""
        if not self.connection:
            raise ValueError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

def get_connection() -> Optional[DatabaseConnection]:
    """Get a database connection."""
    conn = DatabaseConnection("app.db")
    if conn.connect():
        return conn
    return None

def create_user_table(conn: DatabaseConnection):
    """Create the users table."""
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    conn.execute_query(query)
''')

        # Create a JavaScript file
        (test_dir / "utils.js").write_text("""
/**
 * Utility functions for the application
 */

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }
}

module.exports = { debounce, ApiClient };
""")

        test_path = str(test_dir)
    else:
        test_dir = Path(test_path)
        if not test_dir.exists():
            print(f"❌ Test path does not exist: {test_path}")
            return False

    # Initialize server
    try:
        server = CodeEmbeddingsServer()
        print("✅ Server initialized successfully")
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False

    # Index the codebase
    try:
        print("📚 Indexing codebase...")
        result = await server.index_codebase(test_path)
        print(
            f"✅ Indexing completed: {result['total_chunks']} chunks from {result['files_processed']} files"
        )
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False

    # Test searches
    test_queries = [
        "authentication and user login",
        "database connection setup",
        "password hashing functions",
        "API client for HTTP requests",
        "main application entry point",
    ]

    print("\n🔍 Testing search queries...")
    for query in test_queries:
        try:
            results = await server.search_code(query, limit=3, rerank=True)
            print(f"✅ Query '{query}': {len(results)} results")

            if results:
                top_result = results[0]
                print(f"   Best match: {top_result['file_path']} ({top_result['chunk_type']})")
                if "rerank_score" in top_result:
                    print(
                        f"   Scores: similarity={top_result['similarity_score']:.3f}, rerank={top_result['rerank_score']:.3f}"
                    )
                else:
                    print(f"   Similarity: {top_result['similarity_score']:.3f}")
        except Exception as e:
            print(f"❌ Search failed for '{query}': {e}")

    # Clean up test directory if we created it
    if not sys.argv[1:]:  # Only clean up if we created the test dir
        import shutil

        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"🧹 Cleaned up test directory: {test_dir}")

    return True


async def main() -> Literal[0, 1]:
    """Main test runner."""
    print("🧪 Voyage AI Code Embeddings MCP Server Test Suite")
    print("=" * 60)

    # Test environment setup
    if not await test_environment():
        print("\n❌ Environment test failed. Please fix the issues above.")
        return 1

    # Test chunker
    if not await test_chunker():
        print("\n❌ Chunker test failed.")
        return 1

    # Test full workflow
    test_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not await test_full_workflow(test_path):
        print("\n❌ Full workflow test failed.")
        return 1

    print("\n🎉 All tests passed! Your Voyage AI MCP Server is ready to use.")
    print("\nNext steps:")
    print("1. Configure your MCP client (Claude Desktop, VS Code, etc.)")
    print("2. Add the server to your client configuration")
    print("3. Start using semantic code search!")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
