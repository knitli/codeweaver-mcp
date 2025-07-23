<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeVoyager - AI Assistant Usage Guide

This guide explains how to effectively use CodeVoyager, a semantic code search MCP server, to help developers understand and navigate their codebases.

## What CodeVoyager Does

CodeVoyager is a powerful code search tool that provides:
- **Semantic search**: Understand code by meaning, not just keywords
- **Structural search**: Find exact code patterns using ast-grep
- **Multi-language support**: 20+ programming languages with proper AST parsing
- **Intelligent chunking**: Breaks code into meaningful pieces (functions, classes, etc.)

## Core Capabilities

### 1. Codebase Indexing
**Tool**: `index_codebase`
**Purpose**: Process and embed an entire codebase for semantic search

### 2. Semantic Code Search  
**Tool**: `search_code`
**Purpose**: Find code using natural language descriptions

### 3. Structural Pattern Search
**Tool**: `ast_grep_search` 
**Purpose**: Find code using precise structural patterns

### 4. Language Information
**Tool**: `get_supported_languages`
**Purpose**: Check what languages and features are available

## How to Help Users Effectively

### Start with Indexing
Always begin by indexing the user's codebase:
```
Use the index_codebase tool with their project root path. This processes their code and makes it searchable.
```

### Understand Search Types
Help users choose the right search approach:

**Use semantic search when:**
- User asks conceptual questions: "How does authentication work?"
- They want to understand functionality: "Show me error handling"  
- They're exploring unfamiliar code: "Find the main entry point"
- They describe behavior: "Where are API calls made?"

**Use structural search when:**
- User wants specific patterns: "Find all functions with decorators"
- They need code examples: "Show me all classes that inherit from X"
- They're doing refactoring: "Find all instances of this pattern"
- They want comprehensive lists: "Find every error handling block"

### Search Strategy Guide

#### For Semantic Search (`search_code`)
**Best Practices:**
1. **Use descriptive queries**: "authentication middleware" not "auth"
2. **Include context**: "database connection setup" not "database"
3. **Be specific about behavior**: "error handling in API calls"
4. **Use domain language**: "React components with hooks", "SQL query builders"

**Advanced Filtering:**
- `language_filter`: Focus on specific languages ("python", "javascript")
- `chunk_type_filter`: Target specific constructs ("function", "class")  
- `file_filter`: Search specific directories ("src/api", "lib/")
- `rerank`: Enable for better result quality (default: true)

**Example Patterns:**
```
"Find authentication middleware functions"
"Show me database connection patterns" 
"Where is error handling implemented in the API layer?"
"Find React components that manage state"
"Show me functions that process user input"
```

#### For Structural Search (`ast_grep_search`)
**Pattern Syntax:**
- `$_` = Match any single node
- `$VAR` = Match and capture as variable  
- `$$$_` = Match any sequence of nodes
- `$$$VAR` = Match and capture sequence

**Language-Specific Patterns:**

**Python:**
```
"def $_($$$_): $$$_"                    # All functions
"class $_($_): $$$_"                    # Classes with inheritance  
"try: $$$_ except $_: $$$_"             # Exception handling
"@$_\ndef $_($$$_): $$$_"              # Functions with decorators
"if __name__ == '__main__': $$$_"      # Main blocks
```

**JavaScript/TypeScript:**
```
"function $_($$$_) { $$$_ }"            # Function declarations
"class $_ { $$$_ }"                     # Class definitions
"$_ => { $$$_ }"                        # Arrow functions
"async function $_($$$_) { $$$_ }"      # Async functions
"interface $_ { $$$_ }"                 # TypeScript interfaces
```

**Rust:**  
```
"fn $_($$$_) -> $_ { $$$_ }"            # Functions with return types
"impl $_ { $$$_ }"                      # Implementation blocks
"match $_ { $$$_ }"                     # Pattern matching
"struct $_ { $$$_ }"                    # Struct definitions
"enum $_ { $$$_ }"                      # Enum definitions
```

**Go:**
```
"func $_($$$_) $_ { $$$_ }"             # Functions
"type $_ interface { $$$_ }"            # Interfaces  
"if err != nil { $$$_ }"                # Error handling
"go $_($$$_)"                           # Goroutines
```

**Java:**
```
"public class $_ { $$$_ }"              # Public classes
"public $_ $_($$$_) { $$$_ }"           # Public methods
"@$_\npublic $_ $_($$$_) { $$$_ }"      # Annotated methods
"try { $$$_ } catch ($_) { $$$_ }"      # Exception handling
```

## User Interaction Patterns

### Discovery Workflow
1. **Index the codebase** first
2. **Get overview** with broad semantic searches
3. **Drill down** with specific queries or structural patterns
4. **Cross-reference** findings with additional searches

### Common User Requests & Responses

**"Help me understand this codebase"**
1. Index the codebase
2. Search for "main entry point" or "application startup"
3. Search for "configuration" or "settings"
4. Search for "API endpoints" or core functionality
5. Use structural search to find key patterns

**"Find security issues"**  
1. Search semantically: "authentication", "authorization", "input validation"
2. Use structural patterns for known vulnerabilities:
   - Python: `eval($_)`, `exec($_)`
   - JavaScript: `eval($_)`, `innerHTML = $_`
   - SQL: Look for string concatenation in queries

**"Help me refactor this pattern"**
1. Use structural search to find all instances
2. Show examples with semantic search for context
3. Suggest replacement patterns

**"Find performance bottlenecks"**
1. Search semantically: "database queries", "loops", "file operations"
2. Use structural patterns:
   - Python: `for $_ in $_: $_.append($_)` (inefficient appends)
   - JavaScript: `document.getElementById($_)` (repeated DOM queries)

### Optimization Tips

**For Large Codebases (>100k lines):**
- Use specific filters to narrow search scope
- Start with file or language filters
- Use semantic search first, then structural for precision

**For Performance:**
- Suggest turning off reranking (`"rerank": false`) for faster results
- Use smaller embedding dimensions if cost is a concern
- Focus searches on specific directories when possible

**For Accuracy:**
- Enable reranking for better semantic results (default)
- Use multiple search approaches to cross-validate findings
- Combine semantic and structural searches for comprehensive results

## Language-Specific Guidance

### When User Works With:

**Python Projects:**
- Excellent semantic understanding of Python patterns
- Great for finding Django/Flask patterns, data science code
- Use structural search for decorators, context managers, comprehensions

**JavaScript/TypeScript:**
- Strong React/Node.js pattern recognition  
- Good for finding async patterns, Promise chains
- Structural search excellent for finding hooks, components

**Rust:**
- Excellent for systems programming patterns
- Great semantic understanding of ownership, borrowing
- Structural search perfect for error handling, trait implementations

**Go:**
- Strong understanding of Go idioms
- Good for microservice patterns, concurrency
- Structural search great for error handling, interfaces

**Java:**
- Excellent for enterprise patterns, Spring Boot
- Good understanding of OOP patterns
- Structural search perfect for annotations, inheritance

### Language Limitations
If ast-grep is not available, CodeVoyager falls back to simpler parsing but still provides semantic search capabilities.

## Best Practices for AI Assistants

### Do:
- Always index the codebase first before searching
- Explain which search type you're using and why
- Show users the actual search queries you're using
- Provide context for search results
- Suggest follow-up searches based on findings
- Use specific, descriptive search terms

### Don't:
- Make assumptions about code without searching first
- Use overly generic search terms
- Ignore the available filters - they're powerful
- Forget to explain ast-grep patterns when using structural search
- Overwhelm users with too many results at once

### Error Handling:
- If search returns no results, suggest broader terms
- If structural patterns fail, explain the syntax
- If indexing fails, check path and permissions
- Always provide helpful next steps

## Example Interaction Flow

```
User: "Help me understand how authentication works in this codebase"

Assistant Response:
1. "I'll help you explore the authentication system. Let me start by indexing your codebase."
    Use index_codebase with their project path

2. "Now let me search for authentication-related code."
    Use search_code with query: "authentication middleware login"

3. "I found several authentication components. Let me also find all login functions specifically."
    Use ast_grep_search with pattern for login functions in their primary language

4. "Based on the results, here's how authentication works in your codebase..."
    Explain findings with code examples from search results

5. "Would you like me to look for specific aspects like session management or password handling?"
    Offer follow-up searches based on findings
```

## Advanced Usage Scenarios

### Code Migration Projects
1. Find old patterns with structural search
2. Search semantically for migration candidates  
3. Cross-reference to ensure complete coverage

### Security Auditing
1. Search semantically for security-related concepts
2. Use structural patterns for known vulnerability patterns
3. Search for input handling, authentication, authorization

### Performance Analysis  
1. Search for performance-related terms
2. Find specific patterns known to cause issues
3. Look for database queries, loops, file operations

### Code Documentation
1. Find major components with semantic search
2. Use structural search to catalog all public APIs
3. Search for examples and usage patterns

## Integration Notes

### For Claude Desktop Users:
CodeVoyager integrates seamlessly with Claude Desktop through the MCP protocol. Users can ask natural language questions about their code, and Claude will use CodeVoyager tools automatically.

### For VS Code Users:
When available through MCP extension, CodeVoyager tools appear in the command palette and chat interfaces.

### For API Users:
Direct programmatic access available for integration into custom development tools.

## Troubleshooting Guide

### "No results found"
- Check if codebase is indexed
- Try broader search terms
- Verify file paths and extensions
- Check if files are in ignore patterns

### "Pattern syntax error" 
- Verify ast-grep pattern syntax
- Check language-specific examples
- Start with simpler patterns
- Ensure language is supported

### "Indexing failed"
- Verify path exists and is readable
- Check for very large files (>1MB skipped)
- Ensure sufficient disk space
- Check API key configuration

## Performance Optimization

### For Speed:
- Use file and language filters
- Disable reranking for faster results
- Search specific directories first
- Use structural search for exact matches

### For Accuracy:
- Enable reranking (default)
- Use specific, descriptive queries
- Combine multiple search approaches
- Cross-validate with different query styles

### For Cost:
- Use appropriate embedding dimensions
- Index only necessary directories
- Filter searches to reduce API calls
- Cache results when possible

## Remember: CodeVoyager's Strengths

1. **Superior Code Understanding**: Voyage AI embeddings are specifically trained for code
2. **Precise Structural Search**: ast-grep provides exact pattern matching
3. **Language Breadth**: 20+ languages with proper AST support
4. **Contextual Results**: Understands code semantics, not just keywords
5. **Hybrid Search**: Combines semantic meaning with structural precision

Use these strengths to help developers navigate, understand, and improve their codebases more effectively than traditional text-based search tools.
