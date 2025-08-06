<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Overview

The Intent Layer is CodeWeaver's intelligent interface that transforms natural language requests from LLMs into precise, context-rich operations. Rather than exposing LLMs to multiple tools and complex APIs, the Intent Layer provides a single, intelligent entry point that resolves user intent into the exact context needed for successful task completion.

## What is the Intent Layer?

The Intent Layer acts as a sophisticated middleware between LLM users (like Claude, GPT-4, etc.) and CodeWeaver's core search and analysis capabilities. It eliminates context pollution by:

- **Intent Recognition**: Understanding what the user wants to accomplish
- **Context Refinement**: Providing only the relevant information needed
- **Task Orchestration**: Coordinating multiple operations seamlessly
- **Intelligent Routing**: Selecting optimal strategies for each request

## The Problem It Solves

Without the Intent Layer, LLMs face several challenges when working with code search tools:

### Context Pollution
```plaintext
❌ Traditional Approach:
User: "Find authentication functions"
System: Exposes 4 different search tools, various filters, backend options...
LLM: Must decide between semantic_search, ast_grep_search, structural patterns...
Result: Confusion, suboptimal choices, wasted context
```

### Tool Complexity
- Multiple overlapping search capabilities
- Complex parameter combinations
- Unclear which tool fits which use case
- Manual orchestration of multi-step workflows

### Information Overload
- Too many configuration options
- Unclear success/failure states
- No guidance on result interpretation

## How the Intent Layer Works

### Streamlined Interface
```plaintext
✅ Intent Layer Approach:
User: "Find authentication functions"
Intent Layer: Recognizes SEARCH intent :material-arrow-right-circle: PROJECT scope :material-arrow-right-circle: MODERATE complexity
System: Orchestrates semantic + structural search :material-arrow-right-circle: Returns refined results
LLM: Gets exactly what it needs with clear metadata
```

### Processing Pipeline

```mermaid
graph LR
    A[Natural Language Input] --> B[Intent Parsing]
    B --> C[Strategy Selection]
    C --> D[Context Assembly]
    D --> E[Task Execution]
    E --> F[Result Refinement]
    F --> G[Structured Output]
```

1. **Intent Parsing**: Extracts intent type, target, scope, and complexity
2. **Strategy Selection**: Chooses optimal approach based on parsed intent
3. **Context Assembly**: Gathers necessary services and configuration
4. **Task Execution**: Runs selected strategy with assembled context
5. **Result Refinement**: Filters and structures output for clarity
6. **Structured Output**: Returns focused, actionable results

## Core Components

### Intent Types
- **SEARCH**: Find specific code elements, functions, or patterns
- **UNDERSTAND**: Comprehend architecture, organization, or relationships
- **ANALYZE**: Examine code for issues, patterns, or characteristics

### Scope Resolution
- **FILE**: Single file operations
- **MODULE**: Directory or module-level analysis
- **PROJECT**: Entire project examination
- **SYSTEM**: Cross-project or system-wide analysis

### Complexity Assessment
- **SIMPLE**: Direct, single-step operations
- **MODERATE**: Multi-step operations with some coordination
- **COMPLEX**: Advanced processing requiring multiple strategies
- **ADAPTIVE**: Context-dependent complexity requiring dynamic approach

## Built-in Strategies

### Simple Search Strategy
Perfect for direct queries with clear targets:
```plaintext
Input: "Find the login function"
Strategy: Direct semantic search with high confidence
Output: Specific function locations with relevant context
```

### Analysis Workflow Strategy
Handles complex analytical tasks:
```plaintext
Input: "Analyze security issues in authentication system"
Strategy: Multi-step workflow (search :material-arrow-right-circle: analyze :material-arrow-right-circle: report)
Output: Comprehensive analysis with findings and recommendations
```

### Adaptive Strategy
Universal fallback for unclear or complex requests:
```plaintext
Input: "Help me understand the codebase structure"
Strategy: Dynamic approach based on project characteristics
Output: Tailored exploration based on codebase patterns
```

## Key Benefits

### For LLM Users
- **Single Interface**: One natural language entry point
- **Reduced Complexity**: No need to understand multiple tools
- **Better Results**: Intelligent strategy selection and execution
- **Clear Output**: Structured, focused responses

### For Developers
- **Extensible**: Easy to add custom strategies and patterns
- **Configurable**: Flexible configuration for different use cases
- **Performant**: Optimized execution with caching and monitoring
- **Observable**: Comprehensive metrics and debugging support

### For Organizations
- **Consistent**: Standardized approach across different LLMs
- **Scalable**: Handles increasing complexity gracefully
- **Maintainable**: Clear separation of concerns and interfaces
- **Secure**: Controlled access to underlying systems

## Real-World Examples

### Code Discovery
```plaintext
Input: "Where is the user authentication handled?"
Intent Layer Processing:
  - Intent: SEARCH
  - Target: "user authentication"
  - Scope: PROJECT
  - Strategy: SimpleSearchStrategy

Output:
  - auth/login.py:45-67 (login_user function)
  - middleware/auth.py:23-89 (AuthMiddleware class)
  - utils/tokens.py:12-34 (generate_token function)
  - High confidence locations with relevant context
```

### Architecture Understanding
```plaintext
Input: "How does the payment system work?"
Intent Layer Processing:
  - Intent: UNDERSTAND
  - Target: "payment system"
  - Scope: SYSTEM
  - Strategy: AnalysisWorkflowStrategy

Output:
  - Payment flow diagram with key components
  - Integration points and dependencies
  - Database schemas and API contracts
  - Security and compliance considerations
```

### Security Analysis
```plaintext
Input: "Are there any SQL injection vulnerabilities?"
Intent Layer Processing:
  - Intent: ANALYZE
  - Target: "SQL injection vulnerabilities"
  - Scope: PROJECT
  - Strategy: AnalysisWorkflowStrategy

Output:
  - Vulnerability scan results with severity ratings
  - Specific code locations requiring attention
  - Remediation recommendations with examples
  - Compliance status and risk assessment
```

## Getting Started

The Intent Layer is automatically enabled when using CodeWeaver with compatible LLM integrations. For basic usage, simply provide natural language requests - the Intent Layer handles the rest.

For advanced configuration and custom strategies, see:

- [Configuration Guide](configuration.md) - Setup and customization options
- [Custom Strategies](custom-strategies.md) - Building your own intent handlers
- [API Reference](api-reference.md) - Technical interface documentation
- [Examples](examples.md) - Practical use cases and patterns

## Next Steps

1. **[Configuration](configuration.md)**: Learn how to configure the Intent Layer for your specific needs
2. **[Custom Strategies](custom-strategies.md)**: Develop custom strategies for specialized use cases
3. **[Architecture](architecture.md)**: Understand the technical architecture for extension development
4. **[Examples](examples.md)**: Explore practical examples and common patterns
