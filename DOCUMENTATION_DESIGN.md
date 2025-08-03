<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Documentation Design
**Developer-Focused Architecture for MCP Server Integration**

## 📋 Summary

This document outlines the comprehensive Material for MkDocs documentation structure designed for CodeWeaver's semantic code search MCP server. The documentation prioritizes developer workflows, integration guidance, and extensibility patterns for the two primary developer audiences: those implementing CodeWeaver for productivity, and those extending its capabilities.

## 🎯 Design Objectives

### 1. **Developer-First Documentation**
- Optimized for developers integrating MCP servers with AI assistants
- Clear technical setup guides with immediate productivity focus
- Practical code examples and integration patterns
- Extension development guidance for custom providers and backends

### 2. **Progressive Technical Disclosure**
- 6-section information hierarchy focused on developer workflows
- User journey mapping for two primary developer personas
- Technical architecture documentation for extension developers
- Clear separation between usage and extension development

### 3. **Technical User Experience**
- Modern Material Design with developer-focused navigation
- Performance documentation for production deployment
- Accessibility compliance (WCAG 2.1 AA) with technical content optimization
- Sub-3-second load times with efficient content organization

## 📁 Documentation Structure

### 🔄 Information Architecture (6 Sections) - **DEVELOPER-FOCUSED HIERARCHY**

```
Section 1: Getting Started  → Quick setup, installation, configuration, troubleshooting
Section 2: User Guide       → Integration workflows, Claude Desktop, performance
Section 3: Extension Dev    → Architecture, plugin system, custom providers
Section 4: Configuration   → Environment, providers, backends, deployment
Section 5: API Reference   → MCP tools, core components, protocols
Section 6: Reference       → Usage patterns, language support, provider comparison
```

### 👥 User Persona Mapping - **DEVELOPER-FOCUSED AUDIENCES**

**🛠️ Developers Using AI Assistants (Primary)**
- Journey: Home → Quick Setup → User Guide → Configuration → Troubleshooting
- Goal: Get CodeWeaver working with Claude Desktop to enhance development workflow
- Content: Installation guides, Claude Desktop integration, practical query examples, performance optimization

**🧩 Extension Developers (Secondary)**
- Journey: Architecture → Extension Development → API Reference → Testing
- Goal: Build custom providers, backends, and data sources for CodeWeaver
- Content: Plugin architecture, protocol implementation, testing frameworks, performance guidelines

## 🎨 Material Theme Configuration

### Core Features Enabled
- **Navigation**: Instant loading, tabs, sections, breadcrumbs, back-to-top
- **Search**: Highlighting, suggestions, sharing, advanced tokenization
- **Content**: Code copying, annotations, tabs, tooltips, footnotes
- **Performance**: Minification, caching, lazy loading, prefetching

### Brand Identity
- **Primary Color**: CodeWeaver Blue (#1976d2)
- **Accent Color**: Amber (#ffc107)
- **Typography**: Inter (text) + JetBrains Mono (code)
- **Design System**: Material Design with custom CodeWeaver components

### Dark/Light Mode Support
- Automatic detection based on system preference
- Consistent branding across both themes
- Manual toggle with smooth transitions

## 🔌 Plugin Ecosystem

### Essential Plugins (Required)
- **mkdocstrings**: Automatic API documentation from docstrings
- **search**: Enhanced search with code-aware tokenization
- **minify**: Performance optimization for faster loading
- **git-revision-date**: Version tracking and last modified dates

### Enhancement Plugins
- **social**: Branded social media cards for rich previews
- **tags**: Content organization and topic-based discovery
- **rss**: RSS feeds for changelog distribution
- **awesome-pages**: Simplified navigation management

### Quality Assurance Plugins
- **htmlproofer**: Link validation and quality checks
- **mike**: Multi-version documentation management
- **privacy**: GDPR compliance and consent management

## 📝 Content Strategy - **DEVELOPER-FOCUSED TECHNICAL CONTENT**

### Content Type Definitions - **TECHNICAL DOCUMENTATION PATTERNS**

**Technical Setup Guides**
- Structure: Prerequisites → Installation → Configuration → Verification → Troubleshooting
- Purpose: Get developers productive with CodeWeaver quickly and reliably
- Examples: Claude Desktop integration, provider setup, environment configuration

**Architecture Documentation**
- Structure: Overview → Components → Patterns → Extension Points → Examples
- Purpose: Help extension developers understand system design and customization options
- Examples: Plugin architecture, protocol interfaces, factory patterns, service layer

**API Reference Documentation**
- Structure: Interface Definition → Parameters → Return Types → Examples → Error Handling
- Purpose: Complete technical reference for developers building integrations
- Examples: MCP tool signatures, protocol implementations, configuration schemas

**Practical Integration Guides**
- Structure: Use Case → Setup → Implementation → Testing → Optimization
- Purpose: Demonstrate real-world integration patterns and best practices
- Examples: Development workflow integration, performance optimization, production deployment

### Cross-Referencing Strategy
- **Contextual Links**: Prerequisites, related concepts, next steps, examples
- **Smart Tagging**: User level, content type, domain, technology
- **Automatic References**: API cross-references, code examples, configuration links

## 🚀 Performance Optimization

### Loading Performance
- **Instant Navigation**: SPA-like experience with prefetching
- **Asset Optimization**: Minified CSS/JS, optimized images, font loading
- **Caching Strategy**: Aggressive caching with cache invalidation
- **CDN Integration**: Global content delivery for faster access

### Search Performance
- **Pre-built Index**: Search index generated at build time
- **Smart Tokenization**: Code-aware search term separation
- **Result Ranking**: Relevance-based result ordering
- **Suggestion System**: Auto-complete for better discovery

### Mobile Optimization
- **Responsive Design**: Mobile-first approach with touch optimization
- **Performance Budgets**: Sub-3-second load times on 3G networks
- **Progressive Enhancement**: Core functionality works on all devices

## 🔍 SEO and Discoverability

### Search Engine Optimization
- **Semantic HTML**: Proper heading hierarchy and structure
- **Meta Data**: Compelling descriptions and Open Graph tags
- **Structured Data**: JSON-LD for enhanced search results
- **Site Maps**: Automatic sitemap generation

### Content Discovery
- **Tag-Based Navigation**: Browse by topic and technology
- **Related Content**: Algorithmic content suggestions
- **Popular Content**: Usage-based recommendations
- **Full-Text Search**: Comprehensive site search with ranking

## 📊 Analytics and Monitoring

### Usage Analytics
- **Google Analytics**: User behavior and popular content
- **Performance Monitoring**: Core Web Vitals and load times
- **Search Analytics**: What users are looking for
- **Feedback System**: Page helpfulness ratings

### Quality Monitoring
- **Link Validation**: Automated broken link detection
- **Accessibility Testing**: WCAG compliance validation
- **Performance Audits**: Regular performance assessments
- **Content Freshness**: Last modified tracking

## 🛠️ Development Workflow

### Content Creation
1. **Template-Based**: Consistent structure using content templates
2. **Review Process**: Technical and editorial review workflow
3. **Version Control**: Git-based content management
4. **Automated Publishing**: CI/CD pipeline for deployment

### Maintenance Strategy
- **Regular Updates**: Content freshness and accuracy checks
- **Plugin Management**: Version monitoring and compatibility testing
- **Performance Monitoring**: Continuous performance optimization
- **User Feedback**: Incorporation of user suggestions and issues

## 📈 Success Metrics

### User Experience Metrics
- **Page Load Speed**: <3 seconds on 3G networks
- **Search Success Rate**: >80% successful searches
- **Navigation Efficiency**: <3 clicks to find information
- **User Satisfaction**: >4.5/5 helpfulness rating

### Content Quality Metrics
- **Accuracy**: <1% broken links and outdated information
- **Completeness**: 100% API coverage with examples
- **Accessibility**: WCAG 2.1 AA compliance
- **SEO Performance**: Top 3 search results for key terms

## 🚀 Implementation Status - **DEVELOPER-FOCUSED ARCHITECTURE**

### ✅ Completed Implementation
- [x] **CRITICAL**: Updated navigation structure for developer-first approach
- [x] **CRITICAL**: Revised homepage to focus on technical value and developer workflows
- [x] **CRITICAL**: Updated getting started guide for immediate developer productivity
- [x] **CRITICAL**: Created usage patterns reference for practical query examples
- [x] **CRITICAL**: Streamlined service layer documentation for technical clarity
- [x] **CRITICAL**: Created developer-focused architecture overview

### ✅ Technical Infrastructure
- [x] MkDocs configuration with Material theme optimized for developers
- [x] Brand identity and custom CSS system for technical content
- [x] Core plugin configuration and performance optimization
- [x] Navigation structure prioritizing developer workflows

### 🔄 Ongoing Content Development
- [ ] **HIGH**: User guide content for development workflow integration
- [ ] **HIGH**: Extension development guides with practical examples
- [ ] **HIGH**: Configuration documentation for all providers and backends
- [ ] **MEDIUM**: Troubleshooting guides for common integration issues
- [ ] **MEDIUM**: Performance optimization guides for production deployment

### 📋 Future Enhancements
- [ ] Advanced extension examples and tutorials
- [ ] Provider comparison matrix with technical specifications
- [ ] Enterprise deployment patterns and best practices
- [ ] Community contribution guidelines for extension developers
- [ ] Performance benchmarking and optimization guides

## 📚 Files Created

### Core Configuration
- **`mkdocs.yml`**: Complete Material theme configuration with developer-focused navigation
- **`docs/index.md`**: Homepage targeting developers using AI assistants and extension developers
- **`docs/getting-started/quick-start.md`**: Streamlined setup guide for immediate productivity

### Developer-Focused Documentation
- **`docs/reference/usage-patterns.md`**: Practical query patterns and examples
- **`docs/architecture/index.md`**: Technical architecture overview for extension developers
- **`docs/services/index.md`**: Service layer documentation with technical focus

### Documentation Strategy
- **`docs/content-strategy.md`**: Comprehensive content organization strategy
- **`docs/material-theme-config.md`**: Theme configuration recommendations
- **`docs/plugin-recommendations.md`**: Plugin ecosystem and configuration

### Assets and Styling
- **`docs/assets/extra.css`**: Custom CSS with CodeWeaver branding
- **`DOCUMENTATION_DESIGN.md`**: This comprehensive design document

## 🎯 Next Steps - **DEVELOPER-FOCUSED CONTENT COMPLETION**

### **Phase 1: Core User Guide (Week 1-2)**
1. **User Guide Development**: How CodeWeaver integrates with development workflows
2. **Claude Desktop and Claude Code Integration**: Detailed integration guide with troubleshooting
3. **Development Workflows**: Practical examples of using semantic search in daily work (to be clear, codeweaver doesn't *just* use semantic search, but also uses traditional search and can integrate with any data source, and we don't expose search directly to developers right now, but this is more useful in the context of writing custom search patterns that the tool will then use (triggered by the LLM user)). Codeweaver supports adding ast-grep and other patterns to the config, and I'd also hammer home that we only have native patterns for a few languages and could use folks to contribute more.
4. **Performance Optimization**: Configuration tuning for production use

### **Phase 2: Extension Development (Week 3-4)**
1. **Extension Development Guides**: Complete guides for building custom providers
2. **Protocol Documentation**: Detailed interface documentation for plugin development
3. **Testing and Validation**: Framework and examples for testing custom extensions
4. **Performance Guidelines**: Best practices for high-performance extensions

### **Phase 3: Configuration and Reference (Week 5-6)**
1. **Configuration Documentation**: Complete coverage of all providers and backends (not including intent layer, which gets its own section)
2. **Provider Comparison**: Technical specifications and selection guidance
3. **Troubleshooting Guides**: Common issues and resolution patterns
4. **Reference Materials**: Language support, error codes, performance benchmarks

### **Phase 4: The Intent Layer (Week 7-8)**
1. **Intent Layer Overview**: Explanation of the intent layer and its role in CodeWeaver
2. **Configuration Patterns**: How to configure the intent layer for custom search patterns
3. **Custom Strategies**: Developing custom strategies for the intent layer

### **Phase 5: Community and Enhancement**
1. **Community Guidelines**: Contribution patterns for extension developers
2. **Enterprise Deployment**: Production deployment patterns and best practices
3. **User Feedback Integration**: Analytics-based improvements for developer workflows

**Critical Success Factor**: The documentation focuses on enabling developers to be productive with CodeWeaver as a semantic and traditional code search tool, while providing comprehensive extension guidance for those who want to customize and enhance its capabilities.
