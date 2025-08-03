<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Installation Guide

**Multiple installation methods for CodeWeaver on different platforms**

This guide covers all available installation methods for CodeWeaver, from simple package manager installation to development setups, with platform-specific instructions and troubleshooting.

## Quick Installation (Recommended)

### Using uv (Fast Python Package Manager)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install CodeWeaver as a tool
uv tool install codeweaver

# Verify installation
codeweaver --version
```

**Benefits:**
- ✅ Fastest installation method
- ✅ Automatic dependency resolution
- ✅ Isolated environment
- ✅ Easy updates with `uv tool upgrade codeweaver`

## Standard Installation Methods

### Using pip

```bash
# Install from PyPI
pip install codeweaver

# Verify installation
codeweaver --version
```

**For virtual environments:**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install CodeWeaver
pip install codeweaver
```

### Using pipx (Isolated Installation)

```bash
# Install pipx if not already installed
python -m pip install --user pipx
python -m pipx ensurepath

# Install CodeWeaver
pipx install codeweaver

# Verify installation
codeweaver --version
```

**Benefits:**
- ✅ Isolated from system Python
- ✅ Available globally
- ✅ Easy to manage and uninstall

## Platform-Specific Installation

### macOS

#### Method 1: Using Homebrew (Coming Soon)
```bash
# Will be available soon
brew install codeweaver
```

#### Method 2: Using uv
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Install CodeWeaver
uv tool install codeweaver
```

#### Method 3: Using pip with system Python
```bash
# Install for current user
pip3 install --user codeweaver

# Add to PATH if needed
echo 'export PATH="$HOME/Library/Python/3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Linux (Ubuntu/Debian)

#### Method 1: Using uv (Recommended)
```bash
# Install dependencies
sudo apt update
sudo apt install -y curl build-essential

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install CodeWeaver
uv tool install codeweaver
```

#### Method 2: Using system pip
```bash
# Install Python and pip
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Install CodeWeaver
pip3 install --user codeweaver

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Method 3: Using virtual environment
```bash
# Create project directory
mkdir codeweaver-setup
cd codeweaver-setup

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CodeWeaver
pip install codeweaver
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Install Python and pip (RHEL/CentOS)
sudo yum install -y python3 python3-pip

# Or for Fedora
sudo dnf install -y python3 python3-pip

# Install development tools
sudo yum groupinstall -y "Development Tools"  # RHEL/CentOS
sudo dnf groupinstall -y "Development Tools"  # Fedora

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install CodeWeaver
uv tool install codeweaver
```

### Linux (Arch Linux)

```bash
# Install dependencies
sudo pacman -S python python-pip base-devel

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install CodeWeaver
uv tool install codeweaver
```

### Windows

#### Method 1: Using uv (Recommended)
```powershell
# Install uv using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart shell or run
$env:PATH = "$env:USERPROFILE\.cargo\bin;" + $env:PATH

# Install CodeWeaver
uv tool install codeweaver
```

#### Method 2: Using pip
```cmd
# Install Python from python.org if not already installed
# Then install CodeWeaver
pip install codeweaver

# Verify installation
codeweaver --version
```

#### Method 3: Using Windows Subsystem for Linux (WSL)
```bash
# Install WSL if not already available
wsl --install

# Follow Linux installation instructions
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install codeweaver
```

## Development Installation

### From Source (Latest Development)

```bash
# Clone repository
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp

# Install with uv (recommended)
uv sync
uv run codeweaver --version

# Or install with pip
pip install -e .
```

### For Contributing

```bash
# Clone and setup development environment
git clone https://github.com/knitli/codeweaver-mcp.git
cd codeweaver-mcp

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run linting
uv run ruff check
```

## Container Installation

### Docker

```dockerfile
# Dockerfile for CodeWeaver
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install CodeWeaver
RUN uv tool install codeweaver

# Set working directory
WORKDIR /workspace

# Expose port for MCP server
EXPOSE 8000

# Run CodeWeaver
CMD ["codeweaver"]
```

```bash
# Build and run
docker build -t codeweaver .
docker run -p 8000:8000 \
  -e CW_EMBEDDING_API_KEY=your-key \
  -e CW_VECTOR_BACKEND_URL=http://host.docker.internal:6333 \
  codeweaver
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  codeweaver:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CW_EMBEDDING_API_KEY=${CW_EMBEDDING_API_KEY}
      - CW_VECTOR_BACKEND_URL=http://qdrant:6333
    depends_on:
      - qdrant
    volumes:
      - ./codebase:/workspace/codebase

volumes:
  qdrant_storage:
```

```bash
# Run with Docker Compose
CW_EMBEDDING_API_KEY=your-key docker-compose up
```

## Cloud Platform Installation

### AWS Lambda

```python
# lambda_function.py
import json
from codeweaver import CodeWeaverServer

def lambda_handler(event, context):
    server = CodeWeaverServer()
    response = server.handle_request(event)
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
```

```bash
# Package for Lambda
pip install codeweaver -t ./package
cd package
zip -r ../codeweaver-lambda.zip .
cd ..
zip -g codeweaver-lambda.zip lambda_function.py
```

### Google Cloud Functions

```python
# main.py
from flask import Flask, request, jsonify
from codeweaver import CodeWeaverServer

app = Flask(__name__)
server = CodeWeaverServer()

@app.route('/', methods=['POST'])
def handle_request():
    response = server.handle_request(request.get_json())
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

```yaml
# requirements.txt
codeweaver
flask
```

### Azure Functions

```python
# __init__.py
import json
import azure.functions as func
from codeweaver import CodeWeaverServer

server = CodeWeaverServer()

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        response = server.handle_request(req_body)
        
        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )
```

## Verification and Testing

### Basic Functionality Test

```bash
# Test installation
codeweaver --version
codeweaver --help

# Test with minimal configuration
export CW_EMBEDDING_API_KEY=test-key
export CW_VECTOR_BACKEND_URL=http://localhost:6333

# Dry run test
codeweaver index --dry-run /path/to/test/directory
```

### Integration Test

```bash
# Start local vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Set environment variables
export CW_EMBEDDING_API_KEY=your-voyage-ai-key
export CW_VECTOR_BACKEND_URL=http://localhost:6333

# Test indexing a small directory
mkdir test-codebase
echo 'print("Hello, world!")' > test-codebase/hello.py
echo 'console.log("Hello, world!");' > test-codebase/hello.js

# Index test codebase
codeweaver index test-codebase

# Test search
codeweaver search "print hello world"
```

## Troubleshooting Installation

### Common Issues

#### "Command not found" after installation

**Solution 1: Check PATH**
```bash
# Add to shell profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# For uv installations
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Solution 2: Use full path**
```bash
# Find installation location
which codeweaver
python -m pip show codeweaver

# Use full path
/home/user/.local/bin/codeweaver --version
```

#### Permission errors

```bash
# Install for user only
pip install --user codeweaver

# Or use virtual environment
python -m venv ~/.codeweaver-env
source ~/.codeweaver-env/bin/activate
pip install codeweaver
```

#### Python version conflicts

```bash
# Check Python version
python --version  # Should be 3.11+

# Use specific Python version
python3.11 -m pip install codeweaver

# Or install compatible Python
pyenv install 3.11.8
pyenv global 3.11.8
```

#### Network/proxy issues

```bash
# Configure pip for proxy
pip install --proxy http://proxy.company.com:8080 codeweaver

# Or configure pip.conf
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
proxy = http://proxy.company.com:8080
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
EOF
```

### Platform-Specific Troubleshooting

#### macOS: SIP (System Integrity Protection) Issues

```bash
# Use user installation
pip3 install --user codeweaver

# Or use Homebrew Python
brew install python
/usr/local/bin/pip3 install codeweaver
```

#### Windows: PowerShell Execution Policy

```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux: Missing Development Tools

```bash
# Ubuntu/Debian
sudo apt install -y build-essential python3-dev

# RHEL/CentOS
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel

# Arch Linux
sudo pacman -S base-devel python
```

## Updating CodeWeaver

### Using uv

```bash
# Update to latest version
uv tool upgrade codeweaver

# Upgrade all tools
uv tool upgrade --all
```

### Using pip

```bash
# Update to latest version
pip install --upgrade codeweaver

# Force reinstall
pip install --force-reinstall codeweaver
```

### Using pipx

```bash
# Update CodeWeaver
pipx upgrade codeweaver

# Upgrade all pipx packages
pipx upgrade-all
```

## Uninstalling CodeWeaver

### Using uv

```bash
# Remove CodeWeaver
uv tool uninstall codeweaver
```

### Using pip

```bash
# Uninstall CodeWeaver
pip uninstall codeweaver
```

### Using pipx

```bash
# Remove CodeWeaver
pipx uninstall codeweaver
```

### Complete Removal

```bash
# Remove configuration and cache
rm -rf ~/.codeweaver/
rm -rf ~/.cache/codeweaver/

# Remove from shell profiles
# Edit ~/.bashrc, ~/.zshrc etc. to remove PATH additions
```

## Next Steps

After successful installation:

1. [**Configuration**](configuration.md) - Set up API keys and vector database
2. [**Quick Start**](quick-start.md) - Get CodeWeaver running in 5 minutes  
3. [**Claude Desktop Integration**](../user-guide/claude-desktop.md) - Connect with Claude Desktop
4. [**Troubleshooting**](troubleshooting.md) - Resolve any issues that arise