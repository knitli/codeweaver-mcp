<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Quick Start Guide

!!! tip "5-Minute Setup"
    Get CodeWeaver running with Claude Desktop in under 5 minutes. This guide covers the fastest path to semantic code search.

Get CodeWeaver up and running with Claude Desktop in 5 minutes. By the end of this guide, you'll be searching your codebase using natural language.

## Prerequisites

- **Python 3.11+** with `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Claude Desktop** application ([download here](https://claude.ai/download))
- **API Keys**:
  - Voyage AI API key ([get one free](https://voyage.ai/))
  - Qdrant Cloud cluster ([create free cluster](https://cloud.qdrant.io/))

!!! note "Why these providers?"
    This quick start uses Voyage AI (best-in-class code embeddings) and Qdrant Cloud (easiest vector database setup). CodeWeaver supports many other providers - see [Configuration Guide](../configuration/providers.md) for alternatives.

## Step 1: Install CodeWeaver

=== "Using uv (Recommended)"
    ```bash
    # Install CodeWeaver
    uv add codeweaver
    
    # Verify installation
    uv run codeweaver --version
    ```

=== "Using pip"
    ```bash
    # Install CodeWeaver
    pip install codeweaver
    
    # Verify installation
    codeweaver --version
    ```

=== "From Source"
    ```bash
    # Clone repository
    git clone https://github.com/knitli/code-weaver-mcp.git
    cd code-weaver-mcp
    
    # Install with uv
    uv sync
    
    # Verify installation
    uv run codeweaver --version
    ```

## Step 2: Get Your API Keys

### Voyage AI API Key
1. Visit [voyage.ai](https://voyage.ai/) and sign up
2. Navigate to your API dashboard
3. Create a new API key
4. Copy the key (starts with `pa-...`)

### Qdrant Cloud Cluster
1. Visit [cloud.qdrant.io](https://cloud.qdrant.io/) and sign up
2. Create a new cluster (free tier available)
3. Copy your cluster URL (format: `https://xxx-xxx-xxx.us-east.aws.cloud.qdrant.io:6333`)
4. Copy your API key from cluster settings

## Step 3: Configure Environment

Set your API keys as environment variables:

=== "Linux/macOS"
    ```bash
    export CW_EMBEDDING_API_KEY="pa-your-voyage-key"
    export CW_VECTOR_BACKEND_URL="https://your-cluster.qdrant.io:6333"
    export CW_VECTOR_BACKEND_API_KEY="your-qdrant-key"
    ```

=== "Windows"
    ```cmd
    set CW_EMBEDDING_API_KEY=pa-your-voyage-key
    set CW_VECTOR_BACKEND_URL=https://your-cluster.qdrant.io:6333
    set CW_VECTOR_BACKEND_API_KEY=your-qdrant-key
    ```

=== "PowerShell"
    ```powershell
    $env:CW_EMBEDDING_API_KEY="pa-your-voyage-key"
    $env:CW_VECTOR_BACKEND_URL="https://your-cluster.qdrant.io:6333"
    $env:CW_VECTOR_BACKEND_API_KEY="your-qdrant-key"
    ```

!!! tip "Persistent Environment Variables"
    Add these to your `.bashrc`, `.zshrc`, or equivalent shell configuration file to make them permanent.

## Step 4: Configure Claude Desktop

Add CodeWeaver to your Claude Desktop configuration:

=== "macOS"
    Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
    
    ```json
    {
      "mcpServers": {
        "code-weaver": {
          "command": "uv",
          "args": ["run", "codeweaver"],
          "env": {
            "CW_EMBEDDING_API_KEY": "pa-your-voyage-key",
            "CW_VECTOR_BACKEND_URL": "https://your-cluster.qdrant.io:6333",
            "CW_VECTOR_BACKEND_API_KEY": "your-qdrant-key"
          }
        }
      }
    }
    ```

=== "Windows"
    Edit `%APPDATA%\Claude\claude_desktop_config.json`:
    
    ```json
    {
      "mcpServers": {
        "code-weaver": {
          "command": "uv",
          "args": ["run", "codeweaver"],
          "env": {
            "CW_EMBEDDING_API_KEY": "pa-your-voyage-key",
            "CW_VECTOR_BACKEND_URL": "https://your-cluster.qdrant.io:6333",
            "CW_VECTOR_BACKEND_API_KEY": "your-qdrant-key"
          }
        }
      }
    }
    ```

!!! warning "File Location"
    If the file doesn't exist, create it. Make sure to use the exact file path for your operating system.

## Step 5: Test the Connection

1. **Restart Claude Desktop** completely (quit and reopen)
2. **Start a new conversation** in Claude Desktop
3. **Verify MCP connection** by asking: "What MCP tools do you have available?"

You should see CodeWeaver tools listed:
- `index_codebase`
- `search_code` 
- `ast_grep_search`
- `get_supported_languages`

## Step 6: Index Your First Codebase

Now let's index a codebase and try semantic search:

### Index a Project
In Claude Desktop, ask:
```
Can you index my codebase at /path/to/your/project?
```

Claude will use the `index_codebase` tool to:
- Discover source code files
- Chunk code intelligently using AST parsing
- Generate embeddings using Voyage AI
- Store vectors in your Qdrant cluster

### Your First Search
Try these semantic searches:

```
Search for "authentication middleware" in the codebase
```

```
Find code that handles user registration
```

```
Show me error handling patterns
```

### Structural Search
Try pattern-based searches:

```
Find all React functional components using the pattern "const $_ = () => { $$$ }"
```

```
Search for Python classes that inherit from BaseModel
```

## 🎉 Success! What's Next?

Congratulations! You now have CodeWeaver running with Claude Desktop. Here's what you can do next:

<div class="grid cards" markdown>

-   :material-search-web: **[Master Search Strategies](../user-guide/search-strategies.md)**
    
    Learn advanced semantic and structural search techniques

-   :material-tune: **[Optimize Performance](../user-guide/performance.md)**
    
    Tune indexing and search for large codebases

-   :material-cog: **[Advanced Configuration](../configuration/config-file.md)**
    
    Explore configuration options and alternative providers

-   :material-school: **[Try Tutorials](../tutorials/index.md)**
    
    Step-by-step guides for specific use cases

</div>

## Troubleshooting

### Common Issues

**"No MCP tools available"**
- Restart Claude Desktop completely
- Check that `claude_desktop_config.json` is in the correct location
- Verify JSON syntax is valid

**"API key invalid"**
- Verify your Voyage AI API key starts with `pa-`
- Check your Qdrant cluster URL format
- Ensure environment variables are set correctly

**"No search results"**
- Make sure you've indexed your codebase first
- Check that the indexed path contains source code files
- Verify your Qdrant cluster is accessible

**"ast-grep not available"**
- This is normal - CodeWeaver falls back to regex parsing
- For optimal performance, install ast-grep: `uv add ast-grep-py`

### Getting Help

- **[Troubleshooting Guide](../user-guide/troubleshooting.md)** - Comprehensive troubleshooting
- **[Community Support](../community/support.md)** - Discord and GitHub discussions
- **[Configuration Guide](../configuration/index.md)** - Detailed configuration options

---

**Next:** [Your First Search →](first-search.md)