<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# ToDo

In no particular order:
[] Decouple from qdrant (though that will be the default backend) and the binary VoyagerAI or OpenAI-compatible embeddings model to a more general middleware layer.
   - Research existing python solutions that can be used as a middleware-like layer. DocArray comes to mind, but I don't know what else is out there.
   - Ideally support (not necessarily now, but provide an interface for) multiple backends, supporting different embeddings models, vector databases, and search engines.
   - Consider FastMCP's existing middleware layer (https://gofastmcp.com/servers/middleware), which we may be able to plug into instead of building our own.

[] Integrate hybrid search with qdrant's sparse indexing abilities. Add full tree-sitter nodes to metadata, optionally.

Later:
- Add support for indexing and searching through code comments
