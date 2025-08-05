<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Community Hub

Welcome to the CodeWeaver community! We're building the future of semantic code search through extensible Model Context Protocol (MCP) servers, and we'd love your help.

## 🚀 Quick Start for Contributors

Ready to contribute? Here's how to get started:

1. **[Read our Contributing Guide](contributing.md)** - Essential information for all contributors
2. **[Set up your development environment](development-workflow.md)** - Get coding quickly
3. **[Create your first extension](extension-guidelines.md)** - Build providers, backends, or services
4. **[Join our code review process](code-review.md)** - Help maintain code quality

## 🎯 Ways to Contribute

### Code Contributions
- **New Providers**: Add support for new embedding/reranking APIs (OpenAI, Anthropic, etc.)
- **Vector Backends**: Integrate new vector databases (Milvus, LanceDB, etc.)
- **Data Sources**: Support new content sources (GitHub, Slack, databases, etc.)
- **Core Features**: Improve search algorithms, performance, or user experience
- **Bug Fixes**: Help us squash bugs and improve reliability

### Non-Code Contributions
- **Documentation**: Improve guides, add examples, fix typos
- **Testing**: Add test cases, improve coverage, performance benchmarks
- **Community**: Answer questions, write tutorials, share examples
- **Design**: UI/UX improvements, better error messages, developer experience

## 🏗️ Architecture Overview

CodeWeaver is built on extensible patterns that make it easy to add new functionality:

```plaintext
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│  CodeWeaver      │───▶│    Your Code    │
│   (Claude)      │    │     Server       │    │   Extensions    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌────▼────┐ ┌──────▼───────┐
        │  Providers   │ │Backends │ │   Sources    │
        │              │ │         │ │              │
        │ • Voyage AI  │ │• Qdrant │ │• Filesystem  │
        │ • OpenAI     │ │• Pinecone│ │• Git Repos   │
        │ • Custom     │ │• Custom │ │• Custom      │
        └──────────────┘ └─────────┘ └──────────────┘
```

**Key Extension Points:**
- **Providers**: Embedding and reranking services
- **Backends**: Vector database integrations
- **Sources**: Content discovery and processing
- **Services**: Cross-cutting concerns (caching, filtering, etc.)

## 📋 Contributor License Agreement

By contributing to CodeWeaver, you agree to our [Contributor License Agreement](../../CONTRIBUTORS_LICENSE_AGREEMENT.md). Key points:

- ✅ **You retain rights** to your contributions
- ✅ **Open source stays open** - contributions remain under MIT/Apache 2.0
- ✅ **You get credit** - name in docs and commit history
- ✅ **Clear licensing** - dual MIT/Apache 2.0 licensing for maximum compatibility

Questions about the CLA? [Email Adam](mailto:adam@knit.li) or ask in [discussions](https://github.com/knitli/codeweaver-mcp/discussions).

## 🛠️ Development Resources

### Core Documentation
- **[Extension Development Guide](../extension-development/index.md)** - Comprehensive development documentation
- **[Development Patterns](development_patterns.md)** - Coding standards and patterns
- **[Architecture Overview](../architecture/index.md)** - System design and components

### API References
- **[Protocols Reference](../reference/protocols.md)** - Interface definitions
- **[Error Codes](../reference/error-codes.md)** - Error handling patterns
- **[Performance Benchmarks](../reference/performance-benchmarks.md)** - Performance expectations
- **[Full API Reference](../api/index.md)** - Complete API documentation

### Examples and Templates
- **[Provider Examples](../extension-development/providers.md)** - Embedding provider templates
- **[Backend Examples](../extension-development/backends.md)** - Vector database templates
- **[Service Examples](../extension-development/services.md)** - Service provider templates

## 🏆 Recognition

We believe in recognizing our contributors:

### Hall of Fame
Contributors who make significant impact get featured in our documentation and project README.

### Contributor Benefits
- **Direct access** to maintainers and core team
- **Early preview** of new features and roadmap
- **Community recognition** in changelogs and release notes
- **Learning opportunities** through code review and mentorship

## 📞 Getting Help

### Community Channels
- **[GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions)** - General questions and discussions
- **[GitHub Issues](https://github.com/knitli/codeweaver-mcp/issues)** - Bug reports and feature requests
- **[Email Support](mailto:adam@knit.li)** - Direct access to core maintainers

### Development Support
- **[Troubleshooting Guide](../getting-started/troubleshooting.md)** - Common development issues
- **[Testing Guide](../extension-development/testing.md)** - Testing patterns and best practices
- **[Performance Guide](../extension-development/performance.md)** - Optimization techniques

## 🎉 Current Contributors

Thank you to everyone who has contributed to CodeWeaver! Every contribution, no matter how small, makes a difference.

*[View all contributors on GitHub :material-arrow-right-circle:](https://github.com/knitli/codeweaver-mcp/graphs/contributors)*

## 🗺️ Roadmap & Priorities

### High Priority Areas
1. **New Backends** - Expand DocArray providers (Milvus, Redis, etc.)
2. **New Sources** - Add support for more data sources (git, github, web/Tavily, SQL, Context7?)
2. **Performance Optimization** - Faster indexing and search
3. **Better Intent Recognition** - Improve search relevance and intent understanding

### Medium Priority Areas
1. **Advanced Search Features** - facets, filtering, ranking improvements
2. **Enterprise Features** - Multi-tenancy, security, compliance
3. **Monitoring & Observability** - better Metrics, logging, health checks

### Community Suggestions
We regularly review and prioritize community suggestions. Share your ideas in [GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions)!

---

## Ready to Contribute?

1. **Start with our [Contributing Guide](contributing.md)** 📖
2. **Pick an issue** from our [good first issue](https://github.com/knitli/codeweaver-mcp/labels/good%20first%20issue) label 🎯
3. **Join the discussion** in our [community forum](https://github.com/knitli/codeweaver-mcp/discussions) 💬
4. **Build something awesome** and share it with the community! 🚀

Thanks for being part of the CodeWeaver community! 🎸✨
