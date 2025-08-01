<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# ToDo

## CLI Improvements

- Simple MCP client for the 1 person not using vscode/cursor/windsurfer/claudecode/claudedesktop/copilot
  - Not worth it if it wasn't just a few lines of code thanks to fastmcp
- Start/stop services
- Start indexing
- health and status reporting and statistics
- Generate config files
- Insert mcp config into common MCP config files (claude code, claude desktop, etc)

## Consistency and Cleanup

- Properly setup all of the optional feature flags in pyproject.toml
- Ensure consistency of error usage across the codebase -- we have a lot of errors defined... are we using them?
- Ensure consistency in health_check implementation -- should return ServiceHealth object across providers, sources, services, etc
  - Add similar implementations across sources, providers, backends, factories
- propagate fastmcp request context to the intent layer
  - Are we using it effectively? At all?
- ensure create_service_context methods implemented in all service providers and enhanced service context propagates to the intent layer
- Add similar enhanced context to other components and ensure they propagate to the intent layer
- Ensure auto-indexing is using the file filter service
- swap out `Requests` usage with `httpx` in all places (already indirect dependency from fastmcp which uses starlette)

## Expand Baseline Functionality

- Add more NLP patterns for different coding languages
- Add semantic search patterns for all ast-grep supported languages
- Add support other languages (not English)
- Index and search code comments

## Performance and Scalability

- Add performance profiling and benchmarking
- Add performance profiles
- Publish docker version

## Docs

- All of the docs!
