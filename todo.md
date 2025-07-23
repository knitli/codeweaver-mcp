# ToDo

In no particular order:

- integrate rignore for gitignore awareness. Default to this and allow folks to index other directories.
- add watchdog for automatic embeddings from file changes and remove the indexing MCP command -- that shouldn't be assistant driven
- Make sure we have exponential backoffs and cool-off periods
- Integrate hybrid search with qdrant's sparse indexing abilities. Add full tree-sitter nodes to metadata, optionally.
- Add optional support for any openAI-API compatible embeddings provider instead of Voyager (voyager still first)
- Make sure config file is comprehensive, use toml, and look for it in set places (i.e. workspace local (like .local.code-weaver.toml), repo -- .code-weaver.toml, user $HOME/.config/code-weaver/config.toml)
