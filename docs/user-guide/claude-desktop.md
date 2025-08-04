<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Claude Desktop Integration

**Complete setup guide for integrating CodeWeaver with Claude Desktop**

This guide walks you through configuring CodeWeaver as an MCP server in Claude Desktop, from initial setup through advanced configuration and troubleshooting.

## Prerequisites

Before starting, ensure you have:

- **Claude Desktop** installed ([download here](https://claude.ai/download))
- **Python 3.11+** with uv package manager
- **API credentials** for your chosen embedding provider
- **Vector database access** (Qdrant recommended for local development)

## Step 1: Install CodeWeaver

### Using uv (Recommended)
```bash
# Install CodeWeaver globally
uv tool install codeweaver

# Or install in a virtual environment
uv sync
source .venv/bin/activate
uv pip install codeweaver
```

### Using pip
```bash
pip install codeweaver
```

### Verify Installation
```bash
# Test the installation
codeweaver --help

# Check version
uv run codeweaver --version
```

## Step 2: Set Up Environment Variables

Create a dedicated environment file for CodeWeaver configuration:

### Option 1: System Environment (Recommended)
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export CW_EMBEDDING_API_KEY="your-voyage-ai-api-key"
export CW_VECTOR_BACKEND_URL="http://localhost:6333"
export CW_VECTOR_BACKEND_API_KEY="your-qdrant-api-key"  # Optional for local Qdrant
export CW_VECTOR_BACKEND_COLLECTION="my-project-codebase"
```

### Option 2: Environment File
```bash
# Create ~/.codeweaver/.env
mkdir -p ~/.codeweaver
cat > ~/.codeweaver/.env << EOF
CW_EMBEDDING_API_KEY=your-voyage-ai-api-key
CW_VECTOR_BACKEND_URL=http://localhost:6333
CW_VECTOR_BACKEND_API_KEY=your-qdrant-api-key
CW_VECTOR_BACKEND_COLLECTION=my-project-codebase
EOF
```

### Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CW_EMBEDDING_API_KEY` | ✅ | - | API key for embedding provider (Voyage AI, OpenAI, etc.) |
| `CW_VECTOR_BACKEND_URL` | ✅ | - | Vector database URL (e.g., Qdrant, Pinecone) |
| `CW_VECTOR_BACKEND_API_KEY` | ❌ | - | API key for vector database (if required) |
| `CW_VECTOR_BACKEND_COLLECTION` | ❌ | `codeweaver-{uuid}` | Collection name for storing embeddings |
| `CW_EMBEDDING_PROVIDER` | ❌ | `voyage` | Embedding provider: `voyage`, `openai`, `cohere`, `huggingface` |
| `CW_VECTOR_BACKEND_TYPE` | ❌ | `qdrant` | Backend type: `qdrant`, `pinecone`, `weaviate`, `chroma` |
| `CW_LOG_LEVEL` | ❌ | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Step 3: Configure Claude Desktop

### Locate Configuration File

**macOS:**
```bash
~/.claude_desktop_config.json
```

**Windows:**
```bash
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/claude_desktop_config.json
```

### Basic Configuration

Add CodeWeaver to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-voyage-ai-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333",
        "CW_VECTOR_BACKEND_COLLECTION": "my-project-codebase"
      }
    }
  }
}
```

### Advanced Configuration

For production or team setups:

```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "https://your-qdrant-cluster.qdrant.cloud:6333",
        "CW_VECTOR_BACKEND_API_KEY": "your-qdrant-api-key",
        "CW_VECTOR_BACKEND_COLLECTION": "team-codebase",
        "CW_EMBEDDING_PROVIDER": "voyage",
        "CW_VECTOR_BACKEND_TYPE": "qdrant",
        "CW_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Alternative Installation Methods

#### Using Direct Python Path
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "python",
      "args": ["-m", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

#### Using Virtual Environment
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["-m", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Step 4: Restart Claude Desktop

After updating the configuration:

1. **Close Claude Desktop completely**
2. **Wait 5-10 seconds**
3. **Restart Claude Desktop**
4. **Look for CodeWeaver in the server status** (bottom of the chat interface)

## Step 5: Verify Integration

### Check Server Status
In Claude Desktop, you should see:
- 🟢 **CodeWeaver** server connected
- Available tools: `index_codebase`, `search_code`, `ast_grep_search`, `get_supported_languages`

### Test Basic Functionality
Try these commands in Claude Desktop:

```plaintext
Can you list the supported languages in CodeWeaver?
```

```plaintext
Please index the codebase in /path/to/your/project
```

```plaintext
Search for "authentication logic" in the codebase
```

## Provider-Specific Setup

### Voyage AI (Recommended)
```bash
# Get API key from https://dash.voyageai.com/
export CW_EMBEDDING_API_KEY="pa-your-voyage-key"
export CW_EMBEDDING_PROVIDER="voyage"
```

**Benefits:** Best code understanding, competitive pricing, high rate limits

### OpenAI
```bash
# Get API key from https://platform.openai.com/
export CW_EMBEDDING_API_KEY="sk-your-openai-key"
export CW_EMBEDDING_PROVIDER="openai"
```

**Benefits:** Familiar ecosystem, good performance, widely supported

### Cohere
```bash
# Get API key from https://dashboard.cohere.com/
export CW_EMBEDDING_API_KEY="your-cohere-key"
export CW_EMBEDDING_PROVIDER="cohere"
```

**Benefits:** Good multilingual support, competitive pricing

### HuggingFace
```bash
# Get API key from https://huggingface.co/settings/tokens
export CW_EMBEDDING_API_KEY="hf_your-huggingface-key"
export CW_EMBEDDING_PROVIDER="huggingface"
```

**Benefits:** Open source models, no vendor lock-in, customizable

## Vector Database Setup

### Qdrant (Recommended for Local Development)

#### Local Qdrant with Docker
```bash
# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Configuration
export CW_VECTOR_BACKEND_URL="http://localhost:6333"
export CW_VECTOR_BACKEND_TYPE="qdrant"
```

#### Qdrant Cloud
```bash
# Get cluster URL from https://cloud.qdrant.io/
export CW_VECTOR_BACKEND_URL="https://your-cluster.qdrant.cloud:6333"
export CW_VECTOR_BACKEND_API_KEY="your-api-key"
export CW_VECTOR_BACKEND_TYPE="qdrant"
```

### Pinecone
```bash
# Get credentials from https://app.pinecone.io/
export CW_VECTOR_BACKEND_URL="https://your-index.pinecone.io"
export CW_VECTOR_BACKEND_API_KEY="your-pinecone-api-key"
export CW_VECTOR_BACKEND_TYPE="pinecone"
```

### Weaviate
```bash
# Local or cloud Weaviate instance
export CW_VECTOR_BACKEND_URL="http://localhost:8080"
export CW_VECTOR_BACKEND_API_KEY="your-weaviate-key"  # If authentication enabled
export CW_VECTOR_BACKEND_TYPE="weaviate"
```

## Troubleshooting

### Common Issues and Solutions

#### ❌ "CodeWeaver server not found"
**Symptoms:** Server doesn't appear in Claude Desktop status

**Solutions:**
1. Verify `claude_desktop_config.json` syntax with a JSON validator
2. Check that the command path is correct (`uv`, `python`, etc.)
3. Restart Claude Desktop completely
4. Check file permissions on the config file

#### ❌ "Failed to start CodeWeaver server"
**Symptoms:** Server appears but shows error status

**Solutions:**
1. Test CodeWeaver command manually: `uv run codeweaver --help`
2. Check environment variables are set correctly
3. Verify API credentials are valid
4. Ensure vector database is accessible

#### ❌ "Connection refused" errors
**Symptoms:** CodeWeaver starts but can't connect to vector database

**Solutions:**
1. Verify vector database URL and port
2. Check firewall/network settings
3. Test connection manually: `curl http://localhost:6333/health`
4. Ensure API keys are correct for cloud services

#### ❌ "Authentication failed" errors
**Symptoms:** Invalid API key errors

**Solutions:**
1. Verify API key format and validity
2. Check key permissions and usage limits
3. Test API key with provider's CLI or curl
4. Ensure environment variables are properly loaded

### Debug Mode

Enable debug logging for detailed troubleshooting:

```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_LOG_LEVEL": "DEBUG",
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Logs and Diagnostics

**Claude Desktop Logs:**
- macOS: `~/Library/Logs/Claude/`
- Windows: `%APPDATA%\Claude\logs\`
- Linux: `~/.config/claude/logs/`

**CodeWeaver Logs:**
Check the MCP server logs in Claude Desktop's developer console or log files.

### Performance Tuning

#### For Large Codebases
```json
{
  "env": {
    "CW_CHUNK_SIZE": "1200",
    "CW_BATCH_SIZE": "16",
    "CW_MAX_FILE_SIZE": "2097152",
    "CW_PARALLEL_PROCESSING": "true"
  }
}
```

#### For Resource-Constrained Environments
```json
{
  "env": {
    "CW_CHUNK_SIZE": "800",
    "CW_BATCH_SIZE": "4",
    "CW_MAX_FILE_SIZE": "524288",
    "CW_PARALLEL_PROCESSING": "false"
  }
}
```

## Security Considerations

### API Key Security
- Never commit API keys to version control
- Use environment variables or secure secret management
- Rotate keys regularly
- Monitor API usage for unexpected activity

### Network Security
- Use HTTPS for all external connections
- Consider VPN for team setups
- Implement proper firewall rules
- Use strong authentication for vector databases

### Data Privacy
- Understand that code is sent to embedding providers
- Consider on-premises solutions for sensitive code
- Review provider data handling policies
- Implement data retention policies

## Team Configuration

### Shared Vector Database
For team setups, use a shared vector database:

```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "team-api-key",
        "CW_VECTOR_BACKEND_URL": "https://team-qdrant.company.com:6333",
        "CW_VECTOR_BACKEND_API_KEY": "team-qdrant-key",
        "CW_VECTOR_BACKEND_COLLECTION": "main-codebase",
        "CW_EMBEDDING_PROVIDER": "voyage"
      }
    }
  }
}
```

### Multiple Projects
Configure different collections for different projects:

```json
{
  "mcpServers": {
    "codeweaver-frontend": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_VECTOR_BACKEND_COLLECTION": "frontend-app",
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    },
    "codeweaver-backend": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_VECTOR_BACKEND_COLLECTION": "backend-api",
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Next Steps

Once you have CodeWeaver integrated with Claude Desktop:

- [**Development Workflows**](workflows.md) - Learn practical usage patterns
- [**Performance Optimization**](performance.md) - Tune for your specific needs
- [**Troubleshooting Guide**](../getting-started/troubleshooting.md) - Resolve common issues
- [**Configuration Reference**](../getting-started/configuration.md) - Complete configuration options