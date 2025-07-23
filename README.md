<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver

**Semantic code search powered by Voyage AI and ast-grep**

CodeWeaver is a Model Context Protocol (MCP) server that brings best-in-class semantic code search to your AI assistant. Navigate your entire codebase using natural language queries, powered by Voyage AI's superior code embeddings and enhanced with ast-grep's structural search capabilities.

![CodeWeaver Demo](https://img.shields.io/badge/MCP-Compatible-blue) ![Languages](https://img.shields.io/badge/Languages-20+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ? What Makes CodeWeaver Special

?? **Best-in-Class Code Understanding**
Uses Voyage AI's `voyage-code-3` embeddings, which outperform OpenAI by 13.80% on code retrieval tasks, plus `voyage-rerank-2` for hybrid search.

?? **Massive Language Support**
Supports 20+ programming languages with proper AST-aware chunking thanks to ast-grep's tree-sitter integration.

?? **Dual Search Modes**
- **Semantic Search**: "Find authentication middleware"  Understands meaning and context
- **Structural Search**: `"def $_($$$_): $$$_"`  Finds exact code patterns using ast-grep

? **Production Ready**
Smart batching, advanced filtering, cloud-native with Qdrant, and robust error handling.

## ?? Quick Start

### Install
```bash
# Create virtual environment
python -m venv codevoyager-env
source codevoyager-env/bin/activate

# Install dependencies
pip install voyageai qdrant-client mcp ast-grep-py
```

### Configure
```bash
# Set your API keys
export VOYAGE_API_KEY="your-voyage-ai-key"
export QDRANT_URL="https://your-cluster.qdrant.tech:6333"
export QDRANT_API_KEY="your-qdrant-key"
```

### Add to Claude Desktop
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "codevoyager": {
      "command": "python",
      "args": ["/path/to/codevoyager/server.py"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-ai-key",
        "QDRANT_URL": "https://your-cluster.qdrant.tech:6333",
        "QDRANT_API_KEY": "your-qdrant-key"
      }
    }
  }
}
```

### Start Exploring
```bash
python server.py
```

Then in Claude:
- *"Index my Python project at /home/user/myapp"*
- *"Find all authentication functions in my codebase"*
- *"Search for error handling patterns in TypeScript"*
- *"Use ast-grep to find all Rust functions with error handling"*

## ?? Example Searches

### Semantic Search (Natural Language)
Ask Claude to search your code using plain English:

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

## ?? Supported Languages

Thanks to ast-grep's tree-sitter integration, CodeWeaver supports intelligent parsing for:

| **Web Technologies** | **Systems Programming** | **Popular Languages** |
|----------------------|------------------------|----------------------|
| HTML, CSS | Rust, Go, C/C++ | Python, Java, Kotlin |
| JavaScript, TypeScript | C#, Swift | Ruby, PHP, Scala |
| React TSX, Json, Yaml | Bash | Lua, Nix, Solidity |


## ??? Advanced Features

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
1. **Voyage AI embeddings** for semantic understanding
2. **Voyage AI reranker** for result quality
3. **ast-grep parsing** for precise code structure

### Cost Optimization
Choose embedding dimensions based on your needs:
- `256` dimensions: 4x cost reduction, minimal quality loss
- `512` dimensions: 2x cost reduction, slight quality loss
- `1024` dimensions: Default - optimal quality/cost balance
- `2048` dimensions: Maximum quality, highest cost

## ??? Architecture

```
�����������������Ŀ    �����������������Ŀ    �����������������Ŀ
�   Your AI       ����?�   CodeWeaver   ����?�   Qdrant Cloud  �
�   Assistant     �    �   MCP Server    �    �   Vector DB     �
�  (Claude, etc.) �    �                 �    �                 �
�������������������    �������������������    �������������������
                              �
                              
                       �����������������Ŀ
                       �   Voyage AI     �
                       �   Embeddings    �
                       �   + Reranker    �
                       �������������������
                              �
                              
                       �����������������Ŀ
                       �    ast-grep     �
                       �   Tree-sitter   �
                       �    Parsing      �
                       �������������������
```

## ?? Documentation

### MCP Tools

| Tool | Description |
|------|-------------|
| `index_codebase` | Semantically chunk and embed your codebase |
| `search_code` | Natural language search with advanced filtering |
| `ast_grep_search` | Structural search using ast-grep patterns |
| `get_supported_languages` | List all supported languages and capabilities |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOYAGE_API_KEY` | ? | - | Your Voyage AI API key |
| `QDRANT_URL` | ? | - | Qdrant instance URL |
| `QDRANT_API_KEY` | ? | - | Qdrant API key (if auth enabled) |
| `COLLECTION_NAME` | ? | `code-embeddings` | Collection name in Qdrant |

### Getting API Keys

**Voyage AI**: Sign up at [dash.voyageai.com](https://dash.voyageai.com/) for best-in-class code embeddings.

**Qdrant Cloud** (recommended): Qdrant cloud offers a free tier that will handle most open source projects. Create a cluster at [cloud.qdrant.io](https://cloud.qdrant.io/) or run locally with Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## ?? Acknowledgments

CodeWeaver was inspired by and builds upon the excellent work of several open-source projects:

### ?? Key Inspirations

**[RooCode](https://github.com/RooCodeInc/Roo-Code)** - Their sophisticated approach to codebase indexing and semantic chunking provided the foundation for our indexing strategy. RooCode's integration of multiple embedding providers and focus on developer experience shaped our design philosophy.

**[ast-grep](https://ast-grep.github.io/)** - This incredible tool by Herrington Darkholme powers our structural search capabilities. ast-grep's tree-sitter integration and pattern matching system enables CodeWeaver to support 20+ languages with proper AST awareness.

**[Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)** - The official Qdrant MCP server provided the blueprint for MCP protocol implementation and vector database integration patterns.

### ??? Technologies Used

- **[Voyage AI](https://www.voyageai.com/)** - Best-in-class code embeddings (`voyage-code-3`) and reranking (`voyage-rerank-2`)
- **[ast-grep](https://ast-grep.github.io/)** - Tree-sitter based structural search and AST parsing
- **[Qdrant](https://qdrant.tech/)** - High-performance vector similarity search
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Standardized AI-to-tool communication
- **[Tree-sitter](https://tree-sitter.github.io/)** - Incremental parsing for multiple languages

## ?? Development

### Project Structure
```
codevoyager/
��� server.py              # Main MCP server
��� test_server.py         # Test suite
��� requirements.txt       # Dependencies
��� README.md             # This file
��� examples/             # Usage examples
    ��� patterns.md       # ast-grep pattern library
    ��� search_examples.md # Search query examples
```

### Running Tests
```bash
# Test with your own codebase
python test_server.py /path/to/your/project

# Test with sample codebase (creates temporary files)
python test_server.py
```

### Contributing

We welcome contributions! Areas where help is especially appreciated:

- **Language Support**: Add more ast-grep patterns for better chunking
- **Pattern Libraries**: Create domain-specific search patterns (security, performance, etc.)
- **Integrations**: Additional vector databases, embedding providers
- **Performance**: Optimization for large codebases
- **Documentation**: More examples, tutorials, best practices

Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## ?? Performance

Based on published benchmarks and our testing:

- **13.80% better** than OpenAI text-embedding-3-large on code retrieval tasks
- **16.81% better** than CodeSage-large on code understanding
- **Sub-second search** on codebases with 100,000+ code chunks
- **ast-grep parsing**: 10-100x faster than manual AST traversal
- **Supports codebases** up to millions of lines with efficient batching

## ?? Comparison

| Feature | CodeWeaver | GitHub Copilot | Sourcegraph | RooCode |
|---------|-------------|----------------|-------------|---------|
| **Code Embeddings** | ? Voyage AI (best-in-class) | ? Generic | ? Generic | ? Configurable |
| **Reranking** | ? Voyage rerank-2 | ? | ? | ? |
| **Structural Search** | ? ast-grep patterns | ? | ? Manual | ? |
| **Language Support** | ? 20+ with AST | ? Many | ? Many | ? Many |
| **Local Deployment** | ? | ? | ? Enterprise | ? |
| **MCP Integration** | ? Native | ? | ? | ? |
| **Open Source** | ? MIT | ? | ? | ? |
| **Cloud Ready** | ? Qdrant | ? GitHub | ? Cloud | ? |

## ?? Troubleshooting

### Common Issues

**"ast-grep not available"**
Install with: `pip install ast-grep-py`. Server works in fallback mode without it, but ast-grep provides much better results.

**"No results found"**
Ensure your codebase is indexed first using the `index_codebase` tool. Try broader search terms.

**"API key errors"**
Double-check your Voyage AI API key at [dash.voyageai.com](https://dash.voyageai.com/) and Qdrant credentials.

**Performance issues**
Start with smaller codebases (<100k lines) and use appropriate embedding dimensions (512 or 1024).

### Getting Help

- ?? Check our [FAQ](docs/FAQ.md)
- ?? Report bugs on [GitHub Issues](https://github.com/yourusername/codevoyager/issues)
- ?? Join discussions on [GitHub Discussions](https://github.com/yourusername/codevoyager/discussions)
- ?? Email support: codevoyager@yourcompany.com

## ?? License

MIT License - see [LICENSE](LICENSE) for details.

## ?? Star History

If CodeWeaver helps you navigate your codebase more effectively, please consider giving us a star! ?

---

**Built with ?? for developers who want to understand their code better.**

*CodeWeaver - Voyage through your codebase with semantic understanding.*
