<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# process_intent Tool API Reference

!!! tip "Primary Interface"
    `process_intent` is CodeWeaver's primary tool for natural language codebase interaction. This single tool replaces all traditional MCP tools with intelligent intent processing.

The `process_intent` tool provides natural language interface for all codebase analysis operations. Instead of orchestrating multiple tools manually, simply express your intent in natural language and let CodeWeaver handle the implementation details.

## Function Signature

```python
async def process_intent(
    ctx: Context,
    intent: str,
    context: dict[str, Any] | None = None
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ctx` | `Context` | Yes | MCP context object (automatically provided) |
| `intent` | `str` | Yes | Natural language description of what you want to accomplish |
| `context` | `dict[str, Any] \| None` | No | Additional context for intent processing |

### Return Value

```python
{
    "intent_type": str,           # Classified intent type (SEARCH, UNDERSTAND, ANALYZE)
    "confidence": float,          # Confidence score (0.0-1.0)
    "results": list[dict],        # Primary results based on intent
    "summary": str,               # Human-readable summary of findings
    "metadata": dict,             # Processing metadata and statistics
    "related_suggestions": list[str]  # Suggestions for follow-up queries
}
```

## Intent Types

### **SEARCH Intent**

Find specific code, patterns, or functionality within the codebase.

**Example Queries:**
```python
# Basic search
await process_intent("Find all authentication functions")

# Scoped search
await process_intent("Find React components that use useState in /src/components")

# Technology-specific search
await process_intent("Search for Express.js middleware in the API server")

# Pattern-based search
await process_intent("Find all classes that inherit from BaseModel")
```

**Response Structure:**
```python
{
    "intent_type": "SEARCH",
    "confidence": 0.95,
    "results": [
        {
            "file_path": "/src/auth/middleware.js",
            "chunk_content": "function authenticateUser(req, res, next) { ... }",
            "line_start": 15,
            "line_end": 32,
            "relevance_score": 0.92,
            "chunk_type": "function"
        },
        # ... more results
    ],
    "summary": "Found 8 authentication functions across 4 files in the authentication module.",
    "metadata": {
        "files_searched": 127,
        "total_chunks": 1543,
        "processing_time_ms": 234,
        "search_strategy": "semantic_with_structural_fallback"
    },
    "related_suggestions": [
        "Analyze the security implementation of these authentication functions",
        "Find all places where these authentication functions are used"
    ]
}
```

### **UNDERSTAND Intent**

Gain comprehensive understanding of systems, architectures, and code relationships.

**Example Queries:**
```python
# System understanding
await process_intent("Explain how the authentication system works")

# Architecture analysis
await process_intent("Help me understand the data flow from API to database")

# Component relationships
await process_intent("What is the relationship between User and Profile models?")

# Technology integration
await process_intent("How does the caching layer integrate with the application?")
```

**Response Structure:**
```python
{
    "intent_type": "UNDERSTAND",
    "confidence": 0.88,
    "results": [
        {
            "component": "Authentication System",
            "description": "Multi-layer authentication with JWT tokens and session management",
            "key_files": ["/src/auth/jwt.js", "/src/auth/session.js"],
            "relationships": [
                {
                    "type": "depends_on",
                    "target": "User Management System",
                    "description": "Validates user credentials"
                }
            ],
            "flow_steps": [
                "1. Client sends credentials to /auth/login",
                "2. Server validates against user database",
                "3. JWT token generated and returned",
                "4. Subsequent requests include token in headers"
            ]
        }
    ],
    "summary": "The authentication system uses JWT tokens with Redis session storage, implementing a multi-layer security approach with rate limiting and credential validation.",
    "metadata": {
        "components_analyzed": 5,
        "relationships_mapped": 12,
        "processing_time_ms": 1240,
        "analysis_depth": "comprehensive"
    },
    "related_suggestions": [
        "Analyze the security vulnerabilities in this authentication flow",
        "Find all API endpoints that require authentication"
    ]
}
```

### **ANALYZE Intent**

Perform deep analysis for code quality, performance, security, and patterns.

**Example Queries:**
```python
# Security analysis
await process_intent("Analyze potential security vulnerabilities in the authentication code")

# Performance analysis
await process_intent("What are the performance bottlenecks in the API layer?")

# Quality analysis
await process_intent("Review the code quality and identify refactoring opportunities")

# Pattern analysis
await process_intent("Analyze the error handling patterns used throughout the codebase")
```

**Response Structure:**
```python
{
    "intent_type": "ANALYZE",
    "confidence": 0.91,
    "results": [
        {
            "analysis_type": "security_vulnerability",
            "severity": "high",
            "title": "SQL Injection Risk in User Query",
            "description": "Direct string concatenation in SQL query without parameterization",
            "location": {
                "file_path": "/src/users/queries.js",
                "line_start": 45,
                "line_end": 47
            },
            "evidence": "const query = `SELECT * FROM users WHERE id = ${userId}`;",
            "recommendation": "Use parameterized queries or ORM methods to prevent SQL injection",
            "impact": "Potential unauthorized data access or database manipulation"
        },
        {
            "analysis_type": "performance_bottleneck",
            "severity": "medium",
            "title": "N+1 Query Pattern in User Profile Loading",
            "description": "Individual database queries for each user profile",
            "location": {
                "file_path": "/src/profiles/loader.js",
                "line_start": 23,
                "line_end": 31
            },
            "evidence": "for (const user of users) { profile = await getProfile(user.id); }",
            "recommendation": "Implement batch loading or use joins to reduce database round trips",
            "impact": "Increased latency and database load with user count scaling"
        }
    ],
    "summary": "Analysis identified 3 high-severity security issues, 5 performance bottlenecks, and 12 code quality improvements across 23 files.",
    "metadata": {
        "files_analyzed": 23,
        "issues_found": 20,
        "processing_time_ms": 3420,
        "analysis_rules_applied": ["security", "performance", "maintainability"]
    },
    "related_suggestions": [
        "Get specific recommendations for fixing the SQL injection vulnerabilities",
        "Analyze the database query performance patterns"
    ]
}
```

## Advanced Usage

### **Context Parameter**

Provide additional context to refine intent processing:

```python
# Specify scope or constraints
await process_intent(
    intent="Find authentication functions",
    context={
        "scope": "/src/backend",
        "languages": ["javascript", "typescript"],
        "exclude_patterns": ["*.test.js", "*.spec.js"]
    }
)

# Provide previous query context
await process_intent(
    intent="Now analyze these functions for security issues",
    context={
        "previous_query": "Find authentication functions",
        "focus": "security_analysis"
    }
)

# Specify analysis depth
await process_intent(
    intent="Explain the user registration flow",
    context={
        "analysis_depth": "detailed",
        "include_dependencies": True,
        "format": "step_by_step"
    }
)
```

### **Multi-Intent Queries**

Combine multiple intent types in a single query:

```python
# Search + Analyze
await process_intent("Find all payment processing functions and analyze them for PCI compliance")

# Understand + Search
await process_intent("Explain how the caching system works and find all cache invalidation points")

# Sequential analysis
await process_intent("Find the user authentication middleware, explain how it works, and identify any security vulnerabilities")
```

### **Comparative Analysis**

Compare different components or implementations:

```python
# Compare implementations
await process_intent("Compare the error handling patterns between the REST API and GraphQL API")

# Version comparison
await process_intent("Analyze the differences between the old and new payment processing implementations")

# Technology comparison
await process_intent("Compare the performance characteristics of the Redis and Memcached caching implementations")
```

## Error Handling

### **Common Errors**

| Error Type | Description | Resolution |
|------------|-------------|------------|
| `InvalidIntentError` | Intent could not be classified | Provide more specific description |
| `PathNotFoundError` | Specified codebase path doesn't exist | Verify the path exists and is accessible |
| `IndexingError` | Background indexing failed | Check file permissions and disk space |
| `ServiceUnavailableError` | Required service is down | Check service health and retry |
| `InsufficientContextError` | Not enough context to process intent | Provide more specific query or context |

### **Error Response Format**

```python
{
    "error": {
        "type": "InvalidIntentError",
        "message": "Could not classify intent from query",
        "details": "Query too vague: 'find stuff'",
        "suggestions": [
            "Be more specific about what you're looking for",
            "Specify the type of code or functionality",
            "Include the file path or directory if known"
        ]
    },
    "intent_type": None,
    "confidence": 0.0,
    "results": [],
    "metadata": {
        "processing_time_ms": 45,
        "error_code": "INTENT_001"
    }
}
```

## Performance Considerations

### **Background Indexing**

- **First Query**: May take longer as CodeWeaver indexes the codebase
- **Subsequent Queries**: Fast response due to intelligent caching
- **Large Codebases**: Indexing happens progressively in background

### **Caching Strategy**

- **Query-level caching**: Similar intent queries return cached results
- **Component-level caching**: File analysis results are cached and reused
- **Context-aware invalidation**: Cache updates when code changes detected

### **Optimization Tips**

```python
# Efficient: Specific scope reduces processing time
await process_intent("Find authentication functions in /src/auth")

# Less efficient: Broad scope requires more processing
await process_intent("Find authentication functions in the entire codebase")

# Efficient: Use context to refine queries
await process_intent(
    "Find security issues",
    context={"focus": "authentication", "severity": "high"}
)
```

## Integration Examples

### **Claude Desktop Integration**

```typescript
// Example Claude Desktop usage
const result = await mcp.callTool("process_intent", {
    intent: "Find all React components that manage user state"
});

console.log(`Found ${result.results.length} components`);
console.log(result.summary);
```

### **Custom Client Integration**

```python
import asyncio
from mcp_client import MCPClient

async def analyze_codebase():
    client = MCPClient("codeweaver")
    
    # Sequential analysis workflow
    auth_functions = await client.call_tool("process_intent", {
        "intent": "Find all authentication functions"
    })
    
    security_analysis = await client.call_tool("process_intent", {
        "intent": "Analyze these authentication functions for security vulnerabilities",
        "context": {"previous_results": auth_functions["results"]}
    })
    
    return security_analysis

# Run analysis
result = asyncio.run(analyze_codebase())
```

### **Batch Processing**

```python
# Process multiple intents efficiently
intents = [
    "Find all API endpoints",
    "Analyze error handling patterns",
    "Review database query performance",
    "Check for security vulnerabilities"
]

results = []
for intent in intents:
    result = await client.call_tool("process_intent", {"intent": intent})
    results.append(result)

# Combine results for comprehensive analysis
comprehensive_report = {
    "endpoints": results[0]["results"],
    "error_patterns": results[1]["results"],
    "db_performance": results[2]["results"],
    "security_issues": results[3]["results"]
}
```

## Best Practices

### **Query Construction**

✅ **Good Practices:**
```python
# Specific and actionable
"Find all Express.js middleware functions that handle authentication"

# Includes context and scope
"Analyze the payment processing security in /src/billing for PCI compliance"

# Clear intent with constraints
"Explain how the React state management works, focusing on Redux patterns"
```

❌ **Avoid:**
```python
# Too vague
"Find code"

# Multiple unrelated requests
"Find functions and also explain databases and check security"

# Unclear intent
"Look at the thing in the files"
```

### **Context Usage**

```python
# Effective context usage
await process_intent(
    intent="Find performance bottlenecks",
    context={
        "scope": "/src/api",
        "focus": "database_queries",
        "severity_threshold": "medium",
        "exclude_test_files": True
    }
)
```

### **Progressive Refinement**

```python
# Start broad, then narrow
broad_results = await process_intent("Find all authentication code")

# Refine based on initial results
specific_analysis = await process_intent(
    "Analyze the JWT token handling for security vulnerabilities",
    context={"focus_area": "jwt_processing"}
)
```

## Rate Limits and Quotas

| Metric | Limit | Notes |
|--------|--------|-------|
| Requests per minute | 60 | Per MCP client connection |
| Concurrent requests | 5 | Per client session |
| Query complexity | High | Automatically managed by service layer |
| Response size | 10MB | Large responses are paginated |

## Next Steps

<div class="grid cards" markdown>

-   :material-brain: **[Intent Patterns Library](../../intent-guide/patterns.md)**

    Comprehensive examples of effective intent queries

-   :material-api: **[get_intent_capabilities](get-intent-capabilities.md)**

    Discover available intent types and system capabilities

-   :material-school: **[Intent Tutorials](../../tutorials/intent-workflows/codebase-exploration.md)**

    Step-by-step guides for complex intent workflows

-   :material-cog: **[Service Configuration](../../configuration/services.md)**

    Configure intent processing performance and behavior

</div>

---

**Ready to explore intent patterns?** → [Intent Patterns Library](../../intent-guide/patterns.md)