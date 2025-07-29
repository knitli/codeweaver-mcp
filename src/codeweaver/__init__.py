# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""CodeWeaver MCP Platform.

CodeWeaver is the first model context protocol (MCP) server built for its users -- AI assistants.
It exposes a single tool to AI assistants -- asking them to declare their intent. It resolves that
using simple and sophisticated techniques to provide the LLM exactly the context it needs.

It's also a powerful platform for human developers to extend, change, and configure. CodeWeaver
is built with factory patterns and protocol-based interfaces, allowing it to support multiple
data sources, embedding generators/rerankers, sparse and vector databases, general services of any kind,
and fully extend its intent layer with new orchestration strategies. 

Key Features:
- Single natural language interface for its LLM users
- Extensible architecture with pluggable providers and backends
- Factory pattern for dynamic component creation and configuration
- Support for multiple embedding providers (Voyage AI, OpenAI, Cohere, HuggingFace)
- Multiple vector database backends (Qdrant, Pinecone, Weaviate, ChromaDB)
- Universal data source abstraction (filesystem, git, database, API, web)
- Structural search using ast-grep patterns for 20+ programming languages
- Configuration-driven initialization with comprehensive plugin system
- FastMCP middleware integration for cross-cutting concerns

CodeWeaver is the context layer for AI-driven coding.
"""

__version__ = "0.1.0"
