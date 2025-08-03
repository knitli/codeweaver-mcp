<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Troubleshooting Guide

**Common issues and solutions for CodeWeaver setup and operation**

This comprehensive troubleshooting guide covers the most common issues encountered when setting up and using CodeWeaver, with step-by-step solutions and debugging techniques.

## Quick Diagnostics

### Health Check Commands

Before diving into specific issues, run these commands to check your setup:

```bash
# 1. Verify CodeWeaver installation
codeweaver --version

# 2. Test basic functionality
codeweaver --help

# 3. Check environment variables
env | grep CW_

# 4. Test API connectivity
curl -H "Authorization: Bearer $CW_EMBEDDING_API_KEY" \
  https://api.voyageai.com/v1/embeddings \
  -d '{"input":["test"],"model":"voyage-code-2"}' \
  -H "Content-Type: application/json"

# 5. Test vector database connection
curl $CW_VECTOR_BACKEND_URL/collections
```

## Installation Issues

### ❌ "Command not found: codeweaver"

**Symptoms:**
```bash
$ codeweaver --help
bash: codeweaver: command not found
```

**Solutions:**

**Option 1: Install with uv (Recommended)**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install CodeWeaver
uv tool install codeweaver

# Verify installation
uv tool list
```

**Option 2: Install with pip**
```bash
pip install codeweaver

# If using virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
pip install codeweaver
```

**Option 3: Development Installation**
```bash
git clone https://github.com/knitli/codeweaver-mcp
cd codeweaver-mcp
uv sync
uv run codeweaver --help
```

### ❌ "Permission denied" errors

**Symptoms:**
```bash
$ uv tool install codeweaver
Permission denied: /usr/local/bin/codeweaver
```

**Solutions:**

**Option 1: User-level installation**
```bash
# Install for current user only
uv tool install --user codeweaver

# Add to PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Option 2: Use virtual environment**
```bash
# Create isolated environment
python -m venv ~/.codeweaver-env
source ~/.codeweaver-env/bin/activate
pip install codeweaver
```

### ❌ Python version incompatibility

**Symptoms:**
```bash
ERROR: Package 'codeweaver' requires a different Python version
```

**Solutions:**

**Check Python version:**
```bash
python --version  # Should be 3.11+
```

**Install compatible Python:**
```bash
# Using pyenv (recommended)
pyenv install 3.11.8
pyenv global 3.11.8

# Using uv
uv python install 3.11
uv python pin 3.11
```

## Claude Desktop Integration Issues

### ❌ CodeWeaver server not appearing in Claude Desktop

**Symptoms:**
- Server not listed in Claude Desktop status
- No CodeWeaver tools available

**Diagnostic Steps:**

**1. Verify configuration file location:**
```bash
# macOS
ls -la ~/.claude_desktop_config.json

# Windows
dir "%APPDATA%\Claude\claude_desktop_config.json"

# Linux
ls -la ~/.config/claude_desktop_config.json
```

**2. Validate JSON syntax:**
```bash
# Check for JSON syntax errors
python -m json.tool ~/.claude_desktop_config.json
```

**3. Test CodeWeaver command manually:**
```bash
# Test the exact command from config
uv run codeweaver --help

# Or if using direct path
python -m codeweaver --help
```

**Solutions:**

**Fix 1: Correct configuration format**
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Fix 2: Use absolute paths**
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "/usr/local/bin/python",
      "args": ["-m", "codeweaver"],
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Fix 3: Restart Claude Desktop properly**
```bash
# macOS - Force quit and restart
pkill -f "Claude Desktop"
open -a "Claude Desktop"

# Windows - Task Manager or
taskkill /F /IM claude.exe
# Then restart from Start Menu

# Linux
pkill claude
claude-desktop &
```

### ❌ "Failed to start CodeWeaver server"

**Symptoms:**
- Server appears in status but shows error
- Red indicator next to CodeWeaver in Claude Desktop

**Diagnostic Steps:**

**1. Check Claude Desktop logs:**
```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp.log

# Windows
type "%APPDATA%\Claude\logs\mcp.log"

# Linux
tail -f ~/.config/claude/logs/mcp.log
```

**2. Test environment variables:**
```bash
# Test in same environment as Claude Desktop
CW_EMBEDDING_API_KEY=your-key \
CW_VECTOR_BACKEND_URL=http://localhost:6333 \
uv run codeweaver --help
```

**Solutions:**

**Fix 1: Environment variable issues**
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333",
        "CW_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

**Fix 2: Working directory issues**
```json
{
  "mcpServers": {
    "codeweaver": {
      "command": "uv",
      "args": ["run", "codeweaver"],
      "cwd": "/home/user/codeweaver-mcp",
      "env": {
        "CW_EMBEDDING_API_KEY": "your-api-key",
        "CW_VECTOR_BACKEND_URL": "http://localhost:6333"
      }
    }
  }
}
```

## API Authentication Issues

### ❌ "Invalid API key" errors

**Symptoms:**
```
Error: Authentication failed with embedding provider
HTTP 401: Unauthorized
```

**Diagnostic Steps:**

**1. Verify API key format:**
```bash
# Voyage AI keys start with "pa-"
echo $CW_EMBEDDING_API_KEY | head -c 3

# OpenAI keys start with "sk-"
echo $CW_EMBEDDING_API_KEY | head -c 3
```

**2. Test API key directly:**
```bash
# Test Voyage AI
curl -H "Authorization: Bearer $CW_EMBEDDING_API_KEY" \
  https://api.voyageai.com/v1/models

# Test OpenAI
curl -H "Authorization: Bearer $CW_EMBEDDING_API_KEY" \
  https://api.openai.com/v1/models
```

**Solutions:**

**Fix 1: Get valid API key**
- **Voyage AI**: https://dash.voyageai.com/
- **OpenAI**: https://platform.openai.com/api-keys
- **Cohere**: https://dashboard.cohere.com/api-keys
- **HuggingFace**: https://huggingface.co/settings/tokens

**Fix 2: Check key permissions**
```bash
# Ensure API key has embedding permissions
# Check usage limits and billing status
```

**Fix 3: Environment variable loading**
```bash
# Add to shell profile
echo 'export CW_EMBEDDING_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc

# Or use .env file
echo 'CW_EMBEDDING_API_KEY=your-key' > .env
export $(cat .env)
```

### ❌ "Rate limit exceeded" errors

**Symptoms:**
```
Error: Rate limit exceeded for embedding provider
HTTP 429: Too Many Requests
```

**Solutions:**

**Fix 1: Reduce batch size**
```bash
export CW_BATCH_SIZE=4          # Reduce from default 8
export CW_REQUEST_DELAY=2       # Add delay between requests
export CW_MAX_RETRIES=5         # Increase retry attempts
```

**Fix 2: Use different provider**
```bash
# Switch to provider with higher limits
export CW_EMBEDDING_PROVIDER=openai
export CW_EMBEDDING_API_KEY=your-openai-key
```

**Fix 3: Implement exponential backoff**
```bash
export CW_RETRY_STRATEGY=exponential
export CW_RETRY_BASE_DELAY=1.0
export CW_RETRY_MAX_DELAY=60.0
```

## Vector Database Connection Issues

### ❌ "Connection refused" to vector database

**Symptoms:**
```
Error: Failed to connect to vector database
Connection refused: http://localhost:6333
```

**Diagnostic Steps:**

**1. Check if vector database is running:**
```bash
# For Qdrant
curl http://localhost:6333/health

# For Pinecone
curl -H "Api-Key: $CW_VECTOR_BACKEND_API_KEY" \
  https://controller.pinecone.io/databases

# For Weaviate
curl http://localhost:8080/v1/meta
```

**2. Check network connectivity:**
```bash
# Test port connectivity
telnet localhost 6333

# Check if port is in use
netstat -tuln | grep 6333
```

**Solutions:**

**Fix 1: Start vector database**
```bash
# Start local Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start local Weaviate
docker run -p 8080:8080 semitechnologies/weaviate
```

**Fix 2: Check firewall settings**
```bash
# Linux - allow port
sudo ufw allow 6333

# macOS - check firewall settings
sudo pfctl -sr

# Windows - Windows Defender Firewall settings
```

**Fix 3: Use cloud vector database**
```bash
# Qdrant Cloud
export CW_VECTOR_BACKEND_URL=https://your-cluster.qdrant.cloud:6333
export CW_VECTOR_BACKEND_API_KEY=your-api-key

# Pinecone
export CW_VECTOR_BACKEND_URL=https://your-index.pinecone.io
export CW_VECTOR_BACKEND_API_KEY=your-api-key
```

### ❌ "Collection not found" errors

**Symptoms:**
```
Error: Collection 'codeweaver-xyz' not found
HTTP 404: Collection does not exist
```

**Solutions:**

**Fix 1: Let CodeWeaver create collection**
```bash
# Remove collection specification to auto-create
unset CW_VECTOR_BACKEND_COLLECTION
# Or use default naming
export CW_VECTOR_BACKEND_COLLECTION=codeweaver-$(uuidgen)
```

**Fix 2: Create collection manually**
```bash
# Qdrant - create collection
curl -X PUT http://localhost:6333/collections/my-collection \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
```

**Fix 3: Check collection permissions**
```bash
# Verify read/write permissions for collection
# Check API key permissions for cloud services
```

## Indexing Issues

### ❌ "No files found to index"

**Symptoms:**
```
Warning: No files found in directory
Indexed 0 files, created 0 chunks
```

**Diagnostic Steps:**

**1. Check directory path and permissions:**
```bash
# Verify directory exists and is readable
ls -la /path/to/codebase
find /path/to/codebase -name "*.py" | head -10
```

**2. Check file filtering settings:**
```bash
# Review filtering configuration
env | grep CW_INCLUDE
env | grep CW_EXCLUDE
```

**Solutions:**

**Fix 1: Correct path and permissions**
```bash
# Use absolute path
codeweaver index /absolute/path/to/codebase

# Fix permissions
chmod -R 755 /path/to/codebase
```

**Fix 2: Adjust file filtering**
```bash
# Include more file types
export CW_INCLUDE_LANGUAGES=python,javascript,typescript,rust,go,java
export CW_MAX_FILE_SIZE=5242880  # 5MB

# Reduce exclusions
unset CW_EXCLUDE_PATTERNS
```

**Fix 3: Debug file discovery**
```bash
# Enable debug logging
export CW_LOG_LEVEL=DEBUG
codeweaver index /path/to/codebase
```

### ❌ "Indexing stuck or very slow"

**Symptoms:**
- Indexing process appears frozen
- Very slow progress on large codebases

**Solutions:**

**Fix 1: Check resource usage**
```bash
# Monitor CPU and memory
htop
free -h

# Check disk I/O
iotop
```

**Fix 2: Optimize performance settings**
```bash
# Reduce batch size
export CW_BATCH_SIZE=4
export CW_MAX_CONCURRENT_CHUNKS=2

# Enable parallel processing
export CW_PARALLEL_PROCESSING=true
export CW_INDEXING_WORKERS=4
```

**Fix 3: Skip problematic files**
```bash
# Reduce max file size
export CW_MAX_FILE_SIZE=1048576  # 1MB

# Exclude large directories
export CW_EXCLUDE_DIRS=node_modules,target,build,dist,.git
```

## Search Issues

### ❌ "No results found" for valid queries

**Symptoms:**
- Search returns empty results
- Expected files not appearing in results

**Diagnostic Steps:**

**1. Verify indexing completed:**
```bash
# Check if collection has vectors
curl http://localhost:6333/collections/your-collection
```

**2. Test with different query types:**
```bash
# Try exact text search
# Try broader semantic search
# Test with file names or function names
```

**Solutions:**

**Fix 1: Re-index with better settings**
```bash
# Use smaller chunk size for better granularity
export CW_CHUNK_SIZE=800
export CW_MIN_CHUNK_SIZE=50

# Re-index the codebase
codeweaver index /path/to/codebase --force
```

**Fix 2: Adjust search parameters**
```bash
# Lower similarity threshold
export CW_SEARCH_SCORE_THRESHOLD=0.5

# Increase result count
export CW_SEARCH_TOP_K=50
```

**Fix 3: Use hybrid search**
```bash
# Enable both semantic and keyword search
export CW_ENABLE_HYBRID_SEARCH=true
export CW_HYBRID_WEIGHTS=0.7,0.3
```

### ❌ "Search timeout" errors

**Symptoms:**
```
Error: Search request timed out
HTTP 504: Gateway Timeout
```

**Solutions:**

**Fix 1: Increase timeout settings**
```bash
export CW_REQUEST_TIMEOUT=60
export CW_SEARCH_TIMEOUT=30
```

**Fix 2: Optimize vector database**
```bash
# For Qdrant - enable HNSW index
curl -X PUT http://localhost:6333/collections/your-collection \
  -H "Content-Type: application/json" \
  -d '{"optimizers_config": {"default_segment_number": 2}}'
```

**Fix 3: Reduce search scope**
```bash
# Limit search results
export CW_SEARCH_TOP_K=20
export CW_SEARCH_SCORE_THRESHOLD=0.7
```

## Performance Issues

### ❌ High memory usage

**Symptoms:**
- System running out of memory
- CodeWeaver process killed by OOM killer

**Solutions:**

**Fix 1: Limit memory usage**
```bash
export CW_MEMORY_LIMIT_MB=1024
export CW_CHUNK_CACHE_SIZE=500
export CW_VECTOR_CACHE_SIZE=1000
```

**Fix 2: Enable streaming processing**
```bash
export CW_STREAM_PROCESSING=true
export CW_BATCH_SIZE=4
export CW_MAX_CONCURRENT_CHUNKS=2
```

**Fix 3: Use swap if necessary**
```bash
# Add swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### ❌ Slow search performance

**Symptoms:**
- Search queries take >5 seconds
- UI becomes unresponsive

**Solutions:**

**Fix 1: Enable caching**
```bash
export CW_ENABLE_QUERY_CACHE=true
export CW_QUERY_CACHE_SIZE=1000
export CW_EMBEDDING_CACHE_SIZE=5000
```

**Fix 2: Optimize vector database**
```bash
# Use SSD storage for vector database
# Enable indexing optimizations
# Consider using fewer dimensions if possible
```

**Fix 3: Use result streaming**
```bash
export CW_ENABLE_STREAMING=true
export CW_STREAM_CHUNK_SIZE=10
```

## Debugging Tools and Techniques

### Enable Debug Logging

```bash
# Maximum verbosity
export CW_LOG_LEVEL=DEBUG
export CW_LOG_TIMING=true
export CW_LOG_API_CALLS=true
export CW_LOG_PERFORMANCE=true

# Save logs to file
codeweaver index /path 2>&1 | tee codeweaver-debug.log
```

### Health Check Script

Create a comprehensive health check:

```bash
#!/bin/bash
# codeweaver-health-check.sh

echo "=== CodeWeaver Health Check ==="

# 1. Installation check
echo "1. Checking installation..."
if command -v codeweaver &> /dev/null; then
    echo "✅ CodeWeaver installed: $(codeweaver --version)"
else
    echo "❌ CodeWeaver not found"
    exit 1
fi

# 2. Environment variables
echo "2. Checking environment variables..."
if [ -n "$CW_EMBEDDING_API_KEY" ]; then
    echo "✅ Embedding API key set"
else
    echo "❌ CW_EMBEDDING_API_KEY not set"
fi

if [ -n "$CW_VECTOR_BACKEND_URL" ]; then
    echo "✅ Vector backend URL set: $CW_VECTOR_BACKEND_URL"
else
    echo "❌ CW_VECTOR_BACKEND_URL not set"
fi

# 3. API connectivity
echo "3. Checking API connectivity..."
if curl -s -H "Authorization: Bearer $CW_EMBEDDING_API_KEY" \
   https://api.voyageai.com/v1/models > /dev/null; then
    echo "✅ Embedding API accessible"
else
    echo "❌ Cannot reach embedding API"
fi

# 4. Vector database connectivity
echo "4. Checking vector database..."
if curl -s "$CW_VECTOR_BACKEND_URL/health" > /dev/null; then
    echo "✅ Vector database accessible"
else
    echo "❌ Cannot reach vector database"
fi

echo "=== Health check complete ==="
```

### Performance Profiling

```bash
# Enable Python profiling
export CW_ENABLE_PROFILING=true
export CW_PROFILE_OUTPUT_DIR=/tmp/codeweaver-profiles

# Run with profiling
codeweaver index /path/to/codebase

# Analyze profiles
python -m pstats /tmp/codeweaver-profiles/indexing.prof
```

## Getting Help

### Log Collection for Support

```bash
# Collect comprehensive logs
mkdir codeweaver-debug
cd codeweaver-debug

# System information
uname -a > system-info.txt
python --version > python-version.txt
pip list > pip-packages.txt

# Environment
env | grep CW_ > environment.txt

# Test output
codeweaver --help > codeweaver-help.txt 2>&1
codeweaver index --dry-run /path > indexing-test.txt 2>&1

# Logs
cp ~/.config/claude/logs/mcp.log claude-desktop.log
```

### Community Resources

- **GitHub Issues**: https://github.com/knitli/codeweaver-mcp/issues
- **Documentation**: https://docs.codeweaver.dev
- **Discord Community**: [Link to Discord]
- **Stack Overflow**: Tag with `codeweaver`

### Creating Bug Reports

Include this information in bug reports:

1. **System Information**
   - Operating system and version
   - Python version
   - CodeWeaver version

2. **Configuration**
   - Environment variables (sanitized)
   - Claude Desktop configuration
   - Vector database setup

3. **Steps to Reproduce**
   - Exact commands run
   - Input data (if possible)
   - Expected vs actual behavior

4. **Logs and Error Messages**
   - Complete error messages
   - Debug logs (if available)
   - Stack traces

5. **Attempted Solutions**
   - What troubleshooting steps were tried
   - Any partial solutions found

## Next Steps

If you've resolved your issues:

- [**Configuration Reference**](configuration.md) - Complete configuration options
- [**Performance Optimization**](../user-guide/performance.md) - Optimize for your use case
- [**Development Workflows**](../user-guide/workflows.md) - Learn practical usage patterns
- [**Extension Development**](../extension-dev/) - Build custom integrations