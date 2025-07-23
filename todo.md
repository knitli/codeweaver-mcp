<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# ToDo

In no particular order:

- integrate rignore for gitignore awareness. Default to this and allow folks to index other directories.
- add watchdog for automatic embeddings from file changes and remove the indexing MCP command -- that shouldn't be assistant driven
- replace `mcp` with `fastmcp` for less boilerplate, more speed, more ergonomic development
- Make sure we have exponential backoffs and cool-off periods
- Integrate hybrid search with qdrant's sparse indexing abilities. Add full tree-sitter nodes to metadata, optionally.
- Add optional support for any openAI-API compatible embeddings provider instead of Voyager (voyager still first)
- Make sure config file is comprehensive, use toml, and look for it in set places (i.e. workspace local (like .local.code-weaver.toml), repo -- .code-weaver.toml, user $HOME/.config/code-weaver/config.toml)
- Make proper use of ast-grep for file extension/walking

- break up main.py into smaller files
