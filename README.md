<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver

**Extensible semantic code search with plugin architecture**

CodeWeaver is a next-generation Model Context Protocol (MCP) server built on an extensible plugin architecture. Navigate your entire codebase using natural language queries with your choice of embedding providers (Voyage AI, OpenAI, Cohere, HuggingFace), vector databases (Qdrant, Pinecone, Weaviate, ChromaDB), and data sources (filesystem, git, database, API, web). Enhanced with ast-grep's structural search capabilities across 20+ programming languages.

![CodeWeaver Demo](https://img.shields.io/badge/MCP-Compatible-blue) ![Languages](https://img.shields.io/badge/Languages-20+-green) ![Providers](https://img.shields.io/badge/Providers-Extensible-purple) ![License](https://img.shields.io/badge/License-MIT-yellow) ![License](https://img.shields.io/badge/License-Apache-2.0-yellow)

## 🌟 What Makes CodeWeaver Special

🔌 **Extensible Plugin Architecture**
Built on factory patterns and protocol-based interfaces. Mix and match embedding providers, vector databases, and data sources to fit your exact needs.

🧠 **Universal Provider Support**
Choose from best-in-class providers: Voyage AI (`voyage-code-3`), OpenAI, Cohere, HuggingFace, or bring your own. Each optimized for code understanding with automatic fallbacks.

🗃️ **Multiple Backend Options**
Support for Qdrant and easily extensible to other providers (more planned). Start local and scale to cloud with zero code changes.

📚 **Rich Data Sources**
Index from filesystem, git repositories, databases, APIs, and web sources. Unified interface across all source types.

🔍 **Powered by Multiple Search Strategies**
- **Semantic Search**: Natural language queries with embedding reranking
- **Structural Search**: ast-grep patterns for precise code structure matching
- **Hybrid Search**: Combines semantic and structural search for best results
- **Support for multiple index and search strategies**: CodeWeaver intelligently selects the best approach based on your query intent, and can combined results from multiple strategies.

⚙️ **Configuration-Driven**
- **Configure *everything* with TOML or environment variables**. No code changes needed to switch providers, backends, data sources, services, or even search and intent strategies.
- **Hierarchical TOML configuration** with environment variable overrides. No code changes needed to switch providers or backends.

🎯 **Production Ready**
Smart batching, advanced filtering, robust error handling, and comprehensive testing framework.[^1]

[^1]: We're still in alpha though, so expect some rough edges. But we have a solid foundation and are actively improving!

## Our Design Philosophy

### **Developer Experience First**

CodeWeaver is designed to be intuitive and easy to use. We prioritize developer experience with:
- **Simple Configuration**: Hierarchical TOML files or environment variables
- **Unified Interface**: Consistent API across all providers and backends
- **Extensive Documentation**: Clear examples, tutorials, and best practices
- **Community Driven**: Open source with active discussions and contributions and responsive core dev team (of one...)
- **MCP Integration**: Seamless compatibility with AI assistants like Claude.
- **Make AI Useful for Your Codebase**: We built CodeWeaver to help AI assistants like Claude understand your codebase better, enabling them to answer questions, find patterns, and assist with development tasks. Less tokens. More context. Better results.

### **Revolutionary Intent-Based Interface**

- **Natural Language First**. CodeWeaver uses a single `get_context` tool that understands natural language queries like "find authentication functions" or "analyze database patterns". No complex tool selection or parameter configuration required.
  - Traditional MCP servers force AI assistants to choose between multiple tools (`index_codebase`, `search_code`, `ast_grep_search`). CodeWeaver eliminates this complexity with intelligent intent processing.
- **Automatic Background Indexing**. CodeWeaver automatically indexes your codebase in the background using file system monitoring. No manual indexing required - just start asking questions about your code.
- **Intent-Driven Architecture**. Our sophisticated intent processing pipeline understands what you want to accomplish and automatically selects the best combination of search strategies, providers, and processing techniques.
  - Built on enterprise-grade service architecture with health monitoring, auto-recovery, and circuit breakers
- **AI Assistant Optimized**. Every aspect is designed for AI efficiency: unified responses, minimal token usage, contextual results, and zero configuration complexity.
- **Focus on Code Understanding**. Stop re-explaining your codebase. CodeWeaver maintains persistent understanding so AI assistants can dive deep into architecture, patterns, and relationships without starting from scratch every time.

## 🚀 Quick Start

### Install with uv (Recommended)
```bash
# Install CodeWeaver
uv add codeweaver

# Or install from source
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp
uv sync
```

### Install with pip
```bash
pip install codeweaver
```

### Configuration

CodeWeaver uses hierarchical TOML configuration. Create a `.codeweaver.toml` file:

```toml
[embedding]
provider = "voyage"
api_key = "your-voyage-key"  # or use CW_EMBEDDING_API_KEY env var

[backend]
type = "qdrant"
url = "https://your-cluster.qdrant.tech:6333"
api_key = "your-qdrant-key"  # or use CW_VECTOR_BACKEND_API_KEY env var
collection = "my-codebase"

[source]
type = "filesystem"
path = "/path/to/your/codebase"
```

### Environment Variables (Alternative)
```bash
# Required
export CW_EMBEDDING_API_KEY="your-voyage-ai-key"
export CW_VECTOR_BACKEND_URL="https://your-cluster.qdrant.tech:6333"

# Optional
export CW_VECTOR_BACKEND_API_KEY="your-qdrant-key"
export CW_VECTOR_BACKEND_COLLECTION="my-codebase"
export CW_EMBEDDING_PROVIDER="voyage"  # voyage, openai, cohere, huggingface
export CW_BACKEND_TYPE="qdrant"        # qdrant, pinecone, weaviate, chromadb
```

### Add to Claude Desktop
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-voyage-ai-key",
        "CW_VECTOR_BACKEND_URL": "https://your-cluster.qdrant.tech:6333",
        "CW_VECTOR_BACKEND_API_KEY": "your-qdrant-key"
      }
    }
  }
}
```

### Start Exploring
```bash
uv run codeweaver
```

Then in Claude (using natural language):
- *"Analyze the authentication system in my codebase"*
- *"Find all database connection patterns"*
- *"Show me error handling across the TypeScript files"*
- *"What are the main architectural components?"*
- *"Find potential security vulnerabilities"*

CodeWeaver automatically indexes your codebase in the background and understands your intent!

## 🔧 Provider Configurations

### Embedding Providers

**Voyage AI (Default - Best for Code)**
```toml
[embedding]
provider = "voyage"
api_key = "your-voyage-key"
model = "voyage-code-3"  # Best-in-class code embeddings
dimensions = 1024        # 256, 512, 1024, 2048

[reranking]
provider = "voyage"
model = "voyage-rerank-3"
```

**OpenAI**
```toml
[embedding]
provider = "openai"
api_key = "your-openai-key"
model = "text-embedding-3-large"
dimensions = 3072
```

**Cohere**
```toml
[embedding]
provider = "cohere"
api_key = "your-cohere-key"
model = "embed-english-v3.0"
input_type = "search_document"
```

**HuggingFace**
```toml
[embedding]
provider = "huggingface"
api_key = "your-hf-key"
model = "microsoft/codebert-base"
```

### Vector Backends

**Qdrant (Default)**
```toml
[backend]
type = "qdrant"
url = "https://your-cluster.qdrant.tech:6333"
api_key = "your-api-key"
collection = "codebase"
```

**Pinecone**
```toml
[backend]
type = "pinecone"
api_key = "your-pinecone-key"
environment = "us-west1-gcp"
index_name = "codebase"
```

**Weaviate**
```toml
[backend]
type = "weaviate"
url = "https://your-cluster.weaviate.network"
api_key = "your-weaviate-key"
class_name = "CodeChunk"
```

**ChromaDB**
```toml
[backend]
type = "chromadb"
path = "./chroma_db"          # Local
# url = "http://localhost:8000"  # Remote
collection = "codebase"
```

### Data Sources

**Filesystem (Default)**
```toml
[source]
type = "filesystem"
path = "/path/to/codebase"
include_patterns = ["*.py", "*.js", "*.ts"]
exclude_patterns = ["node_modules/**", "*.pyc"]
```

**Git Repository**
```toml
[source]
type = "git"
url = "https://github.com/owner/repo.git"
branch = "main"
token = "your-github-token"  # For private repos
```

**Database**
```toml
[source]
type = "database"
connection_string = "postgresql://user:pass@localhost/db"
query = "SELECT content, metadata FROM code_files"
```

## 📝 Example Searches

### Semantic Search (Natural Language)
Ask Claude (or any other AI assistant) to search your code using plain English:

```
"Find authentication middleware functions"
"Show me database connection patterns"
"Where is error handling implemented?"
"Find API endpoint definitions"
"Show me React components that use hooks"
```

### Structural Search (ast-grep Patterns)
Find exact code structures across your codebase:

```python
# Python: Find all functions
"def $_($$$_): $$$_"

# JavaScript: Find all classes
"class $_ { $$$_ }"

# Rust: Find error handling
"match $_ { Ok($_) => $$$_, Err($_) => $$$_ }"

# TypeScript: Find interfaces
"interface $_ { $$$_ }"

# Go: Find error checks
"if err != nil { $$$_ }"
```

## 🌐 Supported Languages

Thanks to ast-grep's tree-sitter integration, CodeWeaver supports intelligent parsing for:

| **Web Technologies** | **Systems Programming** | **Popular Languages** |
|----------------------|------------------------|----------------------|
| HTML, CSS | Rust, Go, C/C++ | Python, Java, Kotlin |
| JavaScript, TypeScript | C#, Swift | Ruby, PHP, Scala |
| React TSX, Json, Yaml | Bash | Lua, Nix, Solidity |

## ⚙️ Advanced Features

### Smart Filtering
```python
# Search only Python functions
{
  "query": "authentication logic",
  "language_filter": "python",
  "chunk_type_filter": "function"
}

# Search specific directory
{
  "query": "API endpoints",
  "file_filter": "src/api"
}
```

### Hybrid Search
CodeWeaver automatically combines:
1. **Provider embeddings** for semantic understanding (Voyage AI, OpenAI, etc.)
2. **Provider reranking** for result quality (when available)
3. **Hybrid search** for exact matches (e.g., `grep`-like functionality) if the embedding provider supports it
3. **ast-grep parsing** for precise code structure

### Cost Optimization
Choose embedding dimensions based on your needs:
- `256` dimensions: 4x cost reduction, minimal quality loss
- `512` dimensions: 2x cost reduction, slight quality loss
- `1024` dimensions: Default - optimal quality/cost balance
- `2048` dimensions: Maximum quality, highest cost

## 🏗️ Architecture - **INTENT-BASED WITH SERVICES LAYER**

```
┌─────────────────┐    ┌─────────────────────────────────────────┐    ┌─────────────────┐
│   Your AI       │───►│         CodeWeaver MCP Server           │───►│  Vector Backend │
│   Assistant     │    │      (Intent Processing System)         │    │ (Qdrant/Pinecone│
│  (Claude, etc.) │    │                                         │    │ /Weaviate/etc.) │
└─────────────────┘    │  ┌─────────────────────────────────┐    │    └─────────────────┘
                       │  │     Intent Orchestrator         │    │
                       │  │  - Natural Language Processing  │    │
                       │  │  - Auto Background Indexing     │    │
                       │  │  - Strategy Selection           │    │
                       │  │  - Circuit Breakers & Caching  │    │
                       │  └─────────────────────────────────┘    │
                       │                                         │
                       │  ┌─────────────────────────────────┐    │
                       │  │      Advanced Services Layer    │    │
                       │  │  - Health Monitoring           │    │
                       │  │  - Auto Recovery               │    │
                       │  │  - Performance Optimization    │    │
                       │  │  - Chunking & Filtering        │    │
                       │  └─────────────────────────────────┘    │
                       └─────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ Embedding       │              │    ast-grep     │              │  Data Sources   │
│ Providers       │              │   Tree-sitter   │              │  (Filesystem/   │
│ (Voyage/OpenAI/ │              │    Parsing      │              │   Git/DB/API)   │
│  Cohere/HF)     │              │                 │              │                 │
└─────────────────┘              └─────────────────┘              └─────────────────┘
```

## 📚 Configuration System

CodeWeaver uses a hierarchical configuration system that searches for config files in this order:

1. **Workspace local**: `.local.codeweaver.toml`
2. **Repository**: `.codeweaver/.codeweaver.toml` or `.codeweaver.toml`
3. **User config**: `~/.config/codeweaver/config.toml`[^1] (Linux/Mac) or `%LOCALAPPDATA%/codeweaver/config.toml`[^2] (Windows)
4. **Environment variables**: `CW_*` prefixed variables

[^1]: If `XDG_CONFIG_HOME` is set, it will use that directory instead of `~/.config/` (if it's not set to ~/.config...).
[^2]: `%LOCALAPPDATA%` is usually `C:\Users\<YourUser>\AppData\Local`

### Complete Configuration Example
```toml
# .codeweaver.toml

[embedding]
provider = "voyage"
api_key = "your-voyage-key"
model = "voyage-code-3"
dimensions = 1024
batch_size = 8

[reranking]
provider = "voyage"
model = "voyage-rerank-2"
top_k = 10

[backend]
type = "qdrant"
url = "https://your-cluster.qdrant.tech:6333"
api_key = "your-qdrant-key"
collection = "my-codebase"
timeout = 30.0

[source]
type = "filesystem"
path = "/path/to/codebase"
include_patterns = ["*.py", "*.js", "*.ts", "*.rs", "*.go"]
exclude_patterns = [
    "node_modules/**",
    "target/**",
    "dist/**",
    "__pycache__/**",
    "*.pyc"
]

[chunking]
max_chunk_size = 1500
min_chunk_size = 50
overlap_size = 100

[search]
default_limit = 20
max_limit = 100
similarity_threshold = 0.7

[server]
port = 8000
host = "localhost"
```

## 📖 Documentation

### MCP Tools - **INTENT-BASED INTERFACE**

| Tool | Description | Status |
|------|-------------|--------|
| `get_context` | **Primary Tool** - Natural language intent processing for all codebase interactions | ✅ Active |
| `get_context_capabilities` | Query available intent capabilities and supported operations | ✅ Active |
| `index_codebase` | **Legacy** - Manual indexing (replaced by automatic background indexing) | ⚠️ Deprecated |
| `search_code` | **Legacy** - Direct search (replaced by intent processing) | ⚠️ Deprecated |
| `ast_grep_search` | **Legacy** - Structural search (integrated into intent processing) | ⚠️ Deprecated |

**Note**: CodeWeaver now uses a single intent-based interface. Simply describe what you want in natural language!

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CW_EMBEDDING_API_KEY` | ✅ | - | Your embedding provider API key |
| `CW_VECTOR_BACKEND_URL` | ✅ | - | Vector database URL |
| `CW_VECTOR_BACKEND_API_KEY` | ❌ | - | Vector database API key (if auth enabled) |
| `CW_VECTOR_BACKEND_COLLECTION` | ❌ | `codeweaver-{uuid}` | Collection name |
| `CW_EMBEDDING_PROVIDER` | ❌ | `voyage` | Embedding provider (voyage/openai/cohere/huggingface) |
| `CW_BACKEND_TYPE` | ❌ | `qdrant` | Vector backend (qdrant/pinecone/weaviate/chromadb) |

### Getting API Keys

**Voyage AI** (Recommended): Sign up at [dash.voyageai.com](https://dash.voyageai.com/) for best-in-class code embeddings.

**OpenAI**: Get your API key at [platform.openai.com](https://platform.openai.com/api-keys)

**Cohere**: Sign up at [dashboard.cohere.ai](https://dashboard.cohere.ai/)

**HuggingFace**: Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Vector Databases**:
- **Qdrant Cloud**: Free tier at [cloud.qdrant.io](https://cloud.qdrant.io/) or run locally with Docker: `docker run -p 6333:6333 qdrant/qdrant`
- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io/)
- **Weaviate**: Cloud at [weaviate.io](https://weaviate.io/) or self-hosted
- **ChromaDB**: Run locally or use hosted version

## 🔧 Development

### Project Structure
```
codeweaver/
├── src/codeweaver/           # Main source code
│   ├── main.py              # MCP server entry point
│   ├── server.py            # Core server implementation
│   ├── config.py            # Configuration management
│   ├── providers/           # Embedding/reranking providers
│   │   ├── voyage.py        # Voyage AI implementation
│   │   ├── openai.py        # OpenAI implementation
│   │   ├── cohere.py        # Cohere implementation
│   │   └── huggingface.py   # HuggingFace implementation
│   ├── backends/            # Vector database backends
│   │   ├── qdrant.py        # Qdrant implementation
│   │   ├── pinecone.py      # Pinecone implementation
│   │   ├── weaviate.py      # Weaviate implementation
│   │   └── chromadb.py      # ChromaDB implementation
│   ├── sources/             # Data source implementations
│   │   ├── filesystem.py    # Filesystem source
│   │   ├── git.py           # Git repository source
│   │   ├── database.py      # Database source
│   │   └── api.py           # API source
│   └── factories/           # Factory pattern implementations
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── validation/         # Architecture validation
├── examples/               # Usage examples
└── docs/                  # Documentation
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/

# Test with coverage
uv run pytest --cov=codeweaver

# Test specific functionality
uv run python tests/integration/test_server_functionality.py /path/to/test/codebase
```

### Code Quality
```bash
# Run linting with ruff
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Contributing

We welcome contributions! Areas where help is especially appreciated:

- **Provider Integrations**: Add more embedding providers and vector databases
- **Data Sources**: Implement new data source types (S3, Azure Blob, etc.)
- **Language Support**: Improve ast-grep patterns for better chunking. We have not yet implemented ast-grep patterns for all languages supported by ast-grep.
- **Performance**: Optimization for large codebases
- **Documentation**: More examples, tutorials, best practices

Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📊 Performance

Based on published benchmarks and our testing:

- **Voyage AI**: 13.80% better than OpenAI text-embedding-3-large on code retrieval tasks
- **Sub-second search** on codebases with 100,000+ code chunks
- **ast-grep parsing**: 10-100x faster than manual AST traversal
- **Supports codebases** up to millions of lines with efficient batching
- **Plugin architecture**: <10ms overhead for provider/backend switching

## 🆚 Comparison

| Feature | CodeWeaver | GitHub Copilot | Sourcegraph | RooCode |
|---------|-------------|----------------|-------------|---------|
| **Extensible Architecture** | ✅ Plugin-based | ❌ | ❌ | ✅ |
| **Provider Choice** | ✅ 4+ providers | ❌ Fixed | ❌ Fixed | ✅ Configurable |
| **Vector Backends** | ✅ 4+ backends | ❌ | ❌ | ✅ Limited |
| **Structural Search** | ✅ ast-grep patterns | ❌ | ✅ Manual | ❌ |
| **Language Support** | ✅ 20+ with AST | ✅ Many | ✅ Many | ✅ Many |
| **Local Deployment** | ✅ | ❌ | ✅ Enterprise | ✅ |
| **MCP Integration** | ✅ Native | ❌ | ❌ | ❌ |
| **Open Source** | ✅ MIT | ❌ | ❌ | ✅ |
| **Configuration** | ✅ TOML + Env | ❌ | ✅ Complex | ✅ Simple |

## 🚨 Troubleshooting

### Common Issues

**"ast-grep not available"**
Install with: `uv add ast-grep-py`. Server works in fallback mode without it, but ast-grep provides much better results.

**"No provider configured"**
Set your embedding provider in config file or via `CW_EMBEDDING_API_KEY` environment variable.

**"Backend connection failed"**
Check your vector database URL and credentials. For Qdrant, ensure your cluster is accessible.

**"No results found"**
CodeWeaver automatically indexes your codebase in the background. If you're not getting results, try broader natural language queries or check that your codebase path is correctly configured.

**"Configuration not found"**
CodeWeaver looks for config files in several locations. Create a `.codeweaver.toml` file in your project root or use environment variables.

### Getting Help

- 📚 Check our [documentation](docs/)
- 🐛 Report bugs on [GitHub Issues](https://github.com/knitli/codeweaver-mcp/issues)
- 💬 Join discussions on [GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions)
- 📧 Email support: support@knit.li

## 🙏 Acknowledgments

CodeWeaver was inspired by and builds upon the excellent work of several open-source projects:

### 🎯 Key Inspirations

**[RooCode](https://github.com/RooCodeInc/Roo-Code)** - Their sophisticated approach to codebase indexing and semantic chunking provided the foundation for our indexing strategy. RooCode's integration of multiple embedding providers and focus on developer experience shaped our design philosophy.

**[ast-grep](https://ast-grep.github.io/)** - This incredible tool by Herrington Darkholme powers our structural search capabilities. ast-grep's tree-sitter integration and pattern matching system enables CodeWeaver to support 20+ languages with proper AST awareness.[^1]

**[Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)** - The official Qdrant MCP server provided the blueprint for MCP protocol implementation and vector database integration patterns.

### 🛠️ Technologies Used

CodeWeaver is can be extended to use **any** *embedding provider*, *vector database*, *traditional search tool*, or *data source*. But if you just want to get started, the default configuration uses:

- **[Voyage AI](https://www.voyageai.com/)** - Best-in-class code embeddings (`voyage-code-3`) and reranking (`voyage-rerank-2`)
- **[ast-grep](https://ast-grep.github.io/)** - Tree-sitter based structural search and AST parsing
- **[Qdrant](https://qdrant.tech/)** - High-performance vector similarity search
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Standardized AI-to-tool communication
- **[Tree-sitter](https://tree-sitter.github.io/)** - Incremental parsing for multiple languages
- **[FastMCP](https://github.com/jlowin/fastmcp)** - High-performance MCP implementation

## 📄 License

MIT OR Apache-2.0 License - see [LICENSE](LICENSE) for details.

## ⭐ Star History

If CodeWeaver helps you navigate your codebase more effectively, give us a star! ⭐ If it doesn't, please open an issue or contribute to make it better.

[![Star History Chart](https://api.star-history.com/svg?repos=knitli/codeweaver-mcp&type=Date)](https://star-history.com/#knitli/codeweaver-mcp&Date) <-- look how sad it is! Please help us change that! 🙏

---

**Built with 💚 for developers who want to understand their code better.**

*CodeWeaver - Navigate your codebase with extensible semantic intelligence.*

[^1]: What's AST? AST stands for Abstract Syntax Tree, a data structure that represents the structure of source code. In plain terms, it lets us create a map of your code, seeing how everything connects without getting lost in the details. This is crucial for understanding complex codebases, especially when searching for specific patterns or structures.
