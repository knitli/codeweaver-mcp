<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Content Organization Strategy

This document outlines the comprehensive content strategy for CodeWeaver's Material for MkDocs documentation, designed to create a world-class documentation experience.

## Information Architecture Principles

### 1. Progressive Disclosure
The documentation follows a layered approach where complexity increases as users dive deeper:

```plaintext
Level 1: Getting Started (High-level overview)
Level 2: User Guides (Task-oriented)
Level 3: Configuration (Deep configuration)
Level 4: Architecture (System understanding)
Level 5: Plugin Development (Advanced users)
Level 6: API Reference (Detailed reference)
Level 7: Tutorials & Examples (Learning by doing)
Level 8: Reference Materials (Quick lookups)
Level 9: Community & Support (Ecosystem)
```

### 2. User Journey Mapping

#### Primary User Personas:

**🎯 AI Assistant Users (Beginners)**
- **Goal**: Get CodeWeaver working with Claude Desktop quickly
- **Journey**: Home :material-arrow-right-circle: Why CodeWeaver :material-arrow-right-circle: Quick Start :material-arrow-right-circle: First Search :material-arrow-right-circle: User Guide
- **Content Needs**: Simple setup, immediate value, troubleshooting

**🛠️ Developer Teams (Intermediate)**
- **Goal**: Configure and deploy CodeWeaver for team/organization use
- **Journey**: User Guide :material-arrow-right-circle: Configuration :material-arrow-right-circle: Architecture :material-arrow-right-circle: Tutorials :material-arrow-right-circle: Performance
- **Content Needs**: Configuration options, best practices, deployment guides

**🧩 Plugin Developers (Advanced)**
- **Goal**: Extend CodeWeaver with custom components
- **Journey**: Architecture :material-arrow-right-circle: Plugin Development :material-arrow-right-circle: API Reference :material-arrow-right-circle: Examples
- **Content Needs**: Technical depth, code examples, protocols, testing

### 3. Task-Oriented Organization

Each section is organized around user tasks rather than technical structure:

#### User Guide Tasks:
- ✅ Index my codebase
- ✅ Search my code semantically
- ✅ Use structural search patterns
- ✅ Optimize performance
- ✅ Troubleshoot issues

#### Configuration Tasks:
- ✅ Set up providers
- ✅ Configure backends
- ✅ Tune performance
- ✅ Deploy securely

#### Plugin Development Tasks:
- ✅ Create custom provider
- ✅ Build vector backend
- ✅ Implement data source
- ✅ Test plugins

## Content Type Definitions

### 1. Conceptual Content
**Purpose**: Help users understand concepts and make decisions

**Structure**:
```markdown
# Concept Title

## What is X?
Brief definition and purpose

## Why use X?
Benefits and use cases

## How does X work?
High-level explanation

## When to use X?
Decision criteria

## Examples
Real-world scenarios
```

**Examples**: Architecture concepts, plugin types, search strategies

### 2. Task-Based Guides
**Purpose**: Help users accomplish specific goals

**Structure**:
```markdown
# Task Title

## Overview
What you'll accomplish

## Prerequisites
What you need before starting

## Steps
1. Numbered steps with code examples
2. Each step explains why and what

## Verification
How to confirm success

## Next Steps
What to do next

## Troubleshooting
Common issues and solutions
```

**Examples**: Indexing codebases, configuring providers, setting up CI/CD

### 3. Reference Documentation
**Purpose**: Provide detailed technical information for lookup

**Structure**:
```markdown
# API/Feature Reference

## Description
Brief description

## Parameters/Options
Detailed parameter documentation

## Examples
Code examples showing usage

## Related
Links to related concepts
```

**Examples**: API documentation, configuration schemas, error codes

### 4. Tutorials
**Purpose**: Teach through hands-on learning

**Structure**:
```markdown
# Tutorial: Accomplish X

## What you'll learn
Learning objectives

## Prerequisites
Required knowledge and setup

## Step-by-step guide
Detailed walkthrough with explanations

## Understanding the code
Why the solution works

## Next steps
How to expand on what you learned
```

**Examples**: React project setup, enterprise deployment, custom plugins

## Material for MkDocs Features Utilization

### 1. Navigation Features
- **Tabs**: Top-level categories (Getting Started, User Guide, etc.)
- **Sections**: Logical groupings within categories
- **Instant loading**: Fast navigation between pages
- **Navigation indexes**: Landing pages for each section

### 2. Content Features
- **Admonitions**: Tips, warnings, notes for important information
- **Code blocks**: Syntax highlighting for 20+ languages
- **Content tabs**: Multiple examples or options
- **Snippets**: Reusable content blocks

### 3. Search Optimization
- **Custom separators**: Better search tokenization
- **Social cards**: Rich preview cards
- **Tags**: Content categorization
- **Search suggestions**: Auto-complete for better discovery

### 4. Interactive Elements
- **Mermaid diagrams**: Architecture and flow diagrams
- **Math notation**: For algorithm explanations
- **Feedback widgets**: Page helpfulness ratings
- **Copy buttons**: Easy code copying

## Cross-Referencing Strategy

### 1. Contextual Links
Every page includes relevant links to:
- **Prerequisites**: What to read first
- **Related concepts**: Supporting information
- **Next steps**: Natural progression
- **Examples**: Practical applications

### 2. Smart Tagging System
Content is tagged by:
- **User level**: beginner, intermediate, advanced
- **Content type**: guide, reference, tutorial, concept
- **Domain**: configuration, api, plugin, deployment
- **Technology**: python, typescript, docker, kubernetes

### 3. Automatic References
- **API cross-references**: Automatic linking to API docs
- **Code examples**: Link to full examples in repository
- **Configuration references**: Link to schema documentation

## SEO and Discoverability

### 1. Search Engine Optimization
- **Semantic HTML**: Proper heading hierarchy and structure
- **Meta descriptions**: Compelling page descriptions
- **Open Graph tags**: Rich social media previews
- **JSON-LD structured data**: Enhanced search results

### 2. Internal Discovery
- **Related content**: Algorithmic suggestions
- **Tag-based navigation**: Browse by topic
- **Site search**: Full-text search with results ranking
- **Popular content**: Usage-based recommendations

### 3. External Discovery
- **RSS feeds**: Changelog and update notifications
- **Social sharing**: Easy sharing buttons
- **GitHub integration**: Link to source code
- **Community links**: Discord, discussions, support

## Content Templates

### Page Template Structure
```markdown
---
title: "Page Title"
description: "Brief page description for SEO"
tags: [tag1, tag2, tag3]
---

# Page Title

!!! info "Context Setting"
    Brief explanation of what this page covers and who it's for

## Overview
High-level summary

## Main Content
Primary content organized logically

## Examples
Practical examples with code

## Related Topics
- [Link to related concept](../path/to/page.md)
- [Link to tutorial](../tutorials/related.md)

## Next Steps
What to read or do next

---
**Need help?** :material-arrow-right-circle: [Support](../community/support.md)
```

### Code Example Template
```markdown
=== "Basic Example"
    ```python
    # Simple, common use case
    basic_example()
    ```plaintext

=== "Advanced Example"
    ```python
    # More complex scenario
    advanced_example_with_options()
    ```plaintext

=== "Full Example"
    ```python
    # Complete, production-ready example
    full_featured_example()
    ```
```

## Maintenance and Updates

### 1. Content Lifecycle
- **Creation**: Follow templates and style guide
- **Review**: Technical and editorial review process
- **Publication**: Automated deployment pipeline
- **Maintenance**: Regular updates and accuracy checks

### 2. Analytics and Improvement
- **Usage analytics**: Track popular and underperforming content
- **User feedback**: Page helpfulness ratings and comments
- **Search analytics**: Understand what users are looking for
- **Continuous improvement**: Regular content audits and updates

### 3. Version Management
- **Documentation versioning**: Sync with software releases
- **Migration guides**: Help users upgrade between versions
- **Changelog integration**: Automatic documentation updates
- **Deprecation notices**: Clear communication about changes

This content strategy ensures CodeWeaver's documentation provides a world-class experience that scales with user needs and maintains high quality over time.
