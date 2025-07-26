# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""CodeWeaver MCP Server.

An extensible Model Context Protocol (MCP) server providing semantic code search through
a plugin-based architecture. Built with factory patterns and protocol-based interfaces
to support multiple embedding providers, vector databases, and data sources.

Key Features:
- Extensible architecture with pluggable providers and backends
- Factory pattern for dynamic component creation and configuration
- Support for multiple embedding providers (Voyage AI, OpenAI, Cohere, HuggingFace)
- Multiple vector database backends (Qdrant, Pinecone, Weaviate, ChromaDB)
- Universal data source abstraction (filesystem, git, database, API, web)
- Structural search using ast-grep patterns for 20+ programming languages
- Configuration-driven initialization with comprehensive plugin system
- FastMCP middleware integration for cross-cutting concerns
"""

__version__ = "0.1.0"
