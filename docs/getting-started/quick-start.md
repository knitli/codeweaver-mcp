<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Quick Setup Guide

!!! tip "5-Minute Setup"
    Get CodeWeaver running with Claude Desktop in 5 minutes. Enable semantic code search for any AI assistant.

!!! info "Telemetry Notice"
    CodeWeaver includes optional usage analytics to help us improve the platform. [Learn more about our privacy-first telemetry →](telemetry.md)

**CodeWeaver** is the first full-stack MCP platform: a powerful server, extensible framework, and natural language interface for semantic code search.

Get semantic code search working with Claude Desktop in 5 minutes. This guide covers installation, configuration, and first usage.

## Prerequisites

- **Python 3.11+** with `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Claude Desktop** application ([download here](https://claude.ai/download))
- **API Keys**:
  - Voyage AI API key ([get one free](https://voyage.ai/))
  - Qdrant Cloud cluster ([create free cluster](https://cloud.qdrant.io/))

!!! note "Provider Options"
    This guide uses Voyage AI and Qdrant Cloud for simplicity. CodeWeaver supports multiple providers - see [Configuration Guide](../configuration/providers.md) for alternatives including OpenAI, Cohere, Pinecone, and Weaviate.

## Step 1: Install CodeWeaver

=== "Using uv (Recommended)"
    ```bash
    # Install CodeWeaver
    uv add codeweaver-mcp

    # Verify installation
    uv run python -c "import codeweaver; print('CodeWeaver', codeweaver.__version__, 'installed successfully')"
    ```

=== "Using pip"
    ```bash
    # Install CodeWeaver
    pip install codeweaver-mcp

    # Verify installation
    python -c "import codeweaver; print('CodeWeaver', codeweaver.__version__, 'installed successfully')"
    ```

=== "From Source"
    ```bash
    # Clone repository
    git clone https://github.com/knitli/codeweaver-mcp.git
    cd codeweaver-mcp

    # Install with uv
    uv sync

    # Verify installation
    uv run python -c "import codeweaver; print('CodeWeaver', codeweaver.__version__, 'installed successfully')"
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
        "codeweaver": {
          "command": "uv",
          "args": ["run", "python", "-m", "codeweaver.main"],
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
        "codeweaver": {
          "command": "uv",
          "args": ["run", "python", "-m", "codeweaver.main"],
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

You should see CodeWeaver's intent processing tools:
- `get_context` - Natural language codebase interaction
- `get_context_capabilities` - Available intent types and features

!!! note "Interface Design"
    CodeWeaver uses an intent-based interface designed for AI assistants. Instead of calling multiple tools directly, you simply describe what you want to accomplish in natural language.

## Step 6: Try It Out

Test semantic code search with your codebase. **No manual indexing required** - CodeWeaver automatically handles indexing when AI assistants explore code.

### First Search

In Claude Desktop, try:

```plaintext
Can you help me understand the authentication system in /path/to/your/project?
```

CodeWeaver will:
- Automatically index relevant files in your codebase
- Search for authentication-related code semantically
- Provide comprehensive analysis of how authentication works

### Available CLI Commands

CodeWeaver also provides a rich CLI interface for direct interaction:

```bash
# Client operations
uv run python -m codeweaver.cli client search "authentication logic"

# Configuration management
uv run python -m codeweaver.cli config show

# Service management
uv run python -m codeweaver.cli services status

# Indexing operations
uv run python -m codeweaver.cli index create /path/to/project
```

### Example Queries

Natural language queries work across any codebase:

```plaintext
Find all API endpoints in this Express.js project
```

```plaintext
Show me error handling patterns in the codebase
```

```plaintext
What are the main database models?
```

```plaintext
Explain the user registration flow
```

```plaintext
Find potential security issues in auth code
```

### How Queries Work

CodeWeaver processes natural language in multiple ways:

- **Semantic Search**: "authentication functions" finds login, verify, authorize functions
- **Structural Patterns**: "API endpoints" finds route definitions, handlers, middleware
- **Cross-File Analysis**: "user registration flow" traces code across multiple files
- **Domain Understanding**: "security issues" applies security-specific pattern matching

## Success! What's Next?

You now have semantic code search working with Claude Desktop. Here's what to explore:

<div class="grid cards" markdown>

-   :material-cog: **[User Guide](../user-guide/how-it-works.md)**

    Understand how CodeWeaver integrates with your development workflow

-   :material-tune: **[Configuration](../configuration/environment.md)**

    Customize providers, backends, and performance settings

-   :material-puzzle: **[Extension Development](../architecture/index.md)**

    Build custom providers and extend CodeWeaver's capabilities

-   :material-help: **[Troubleshooting](troubleshooting.md)**

    Solve common issues and optimize performance

</div>

## How This Differs from Manual Search

**Traditional Approach:**
```plaintext
# Manual steps required
1. grep -r "auth" .
2. Find relevant files
3. Read and understand context
4. Piece together the system
```

**With CodeWeaver:**
```plaintext
# Single natural language query
"Explain the authentication system in this project"
# → Automatic semantic search + structural analysis + result synthesis
```

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

**"Failed to process intent"**
- Make sure the specified codebase path exists and contains source code
- Verify your Qdrant cluster is accessible
- Check that file permissions allow reading the codebase

**"Background indexing slow"**
- This is normal for large codebases on first access
- Subsequent queries will be much faster due to caching
- Consider using smaller directory paths for testing

### Getting Help

- **[User Guide](../user-guide/how-it-works.md)** - How CodeWeaver works with your development workflow
- **[Configuration Guide](../configuration/environment.md)** - Advanced setup and customization
- **[Community Support](../community/support.md)** - GitHub discussions and help
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

**Next:** [User Guide →](../user-guide/how-it-works.md)
