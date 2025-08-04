<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Error Codes Reference

**Complete reference for CodeWeaver error codes, causes, and solutions**

This comprehensive guide catalogs all error codes, warning messages, and status indicators used throughout CodeWeaver, with detailed explanations and resolution steps.

## Error Code Format

CodeWeaver uses structured error codes for systematic troubleshooting:

```plaintext
CW-<COMPONENT>-<SEVERITY>-<CODE>
```

- **COMPONENT**: System component (AUTH, EMBED, VECTOR, INDEX, SEARCH, CONFIG)
- **SEVERITY**: Error level (E=Error, W=Warning, I=Info)
- **CODE**: Unique 3-digit identifier

## Authentication Errors (CW-AUTH-E-xxx)

### CW-AUTH-E-001: Invalid API Key
```plaintext
Error: Invalid API key for embedding provider
Authentication failed with status 401
```

**Causes:**
- Incorrect API key format
- Expired or revoked API key
- Wrong provider selected for key type

**Solutions:**
1. Verify API key format matches provider requirements
2. Check API key validity with provider dashboard
3. Regenerate API key if expired
4. Ensure correct provider configuration

**Related Environment Variables:**
- `CW_EMBEDDING_API_KEY`
- `CW_VOYAGE_API_KEY`
- `CW_OPENAI_API_KEY`

### CW-AUTH-E-002: API Key Not Found
```plaintext
Error: API key is required but not provided
No valid authentication credentials found
```

**Causes:**
- Missing environment variable
- Empty API key value
- Configuration file not loaded

**Solutions:**
1. Set required environment variable: `export CW_EMBEDDING_API_KEY="your-key"`
2. Verify environment variable is accessible: `echo $CW_EMBEDDING_API_KEY`
3. Check Claude Desktop configuration includes API key

### CW-AUTH-E-003: Authentication Timeout
```plaintext
Error: Authentication request timed out
Failed to validate API key within 30 seconds
```

**Causes:**
- Network connectivity issues
- Provider service outage
- Firewall blocking requests

**Solutions:**
1. Check internet connectivity
2. Verify provider service status
3. Increase timeout: `export CW_AUTH_TIMEOUT=60`
4. Check firewall/proxy settings

### CW-AUTH-E-004: Rate Limited Authentication
```plaintext
Error: Too many authentication attempts
Rate limit exceeded for API key validation
```

**Causes:**
- Multiple rapid authentication attempts
- Shared API key across multiple instances
- Provider rate limiting

**Solutions:**
1. Wait before retrying (typically 60 seconds)
2. Use separate API keys for different instances
3. Implement exponential backoff
4. Check provider rate limit documentation

## Embedding Provider Errors (CW-EMBED-E-xxx)

### CW-EMBED-E-001: Provider Unavailable
```plaintext
Error: Embedding provider 'voyage-ai' is not available
Provider not found in registry
```

**Causes:**
- Provider not installed
- Invalid provider name
- Import error in provider module

**Solutions:**
1. Install provider: `uv add voyageai`
2. Check spelling: use `voyage` not `voyage-ai`
3. Verify installation: `python -c "import voyageai"`
4. Check supported providers list

### CW-EMBED-E-002: Model Not Found
```plaintext
Error: Model 'unknown-model' not supported
Available models: ['voyage-code-3', 'voyage-context-3']
```

**Causes:**
- Typo in model name
- Model not available in selected provider
- Model deprecated or removed

**Solutions:**
1. Check model name spelling
2. Use `get_supported_models()` to list available models
3. Update to latest supported model
4. Verify model is available in your API tier

### CW-EMBED-E-003: Request Too Large
```plaintext
Error: Input text exceeds maximum length
Max length: 8000 characters, received: 12000 characters
```

**Causes:**
- Text chunk too large for model
- Model input limits exceeded
- No text truncation configured

**Solutions:**
1. Reduce chunk size: `export CW_CHUNK_SIZE=1000`
2. Enable truncation: `export CW_TRUNCATE_INPUT=true`
3. Use model with larger context window
4. Implement text preprocessing

### CW-EMBED-E-004: Rate Limit Exceeded
```plaintext
Error: Rate limit exceeded for embedding provider
Requests per minute limit reached: 100/100
```

**Causes:**
- API rate limits exceeded
- Concurrent requests too high
- Insufficient rate limiting configuration

**Solutions:**
1. Reduce batch size: `export CW_BATCH_SIZE=4`
2. Add request delays: `export CW_REQUEST_DELAY=1.0`
3. Upgrade API tier if available
4. Implement exponential backoff

### CW-EMBED-E-005: Service Unavailable
```plaintext
Error: Embedding service temporarily unavailable
HTTP 503: Service Unavailable
```

**Causes:**
- Provider service outage
- Maintenance window
- Regional service issues

**Solutions:**
1. Check provider status page
2. Wait and retry (typically 5-15 minutes)
3. Switch to fallback provider temporarily
4. Enable provider failover configuration

## Vector Database Errors (CW-VECTOR-E-xxx)

### CW-VECTOR-E-001: Connection Failed
```plaintext
Error: Failed to connect to vector database
Connection refused: http://localhost:6333
```

**Causes:**
- Vector database not running
- Wrong URL or port
- Network connectivity issues
- Firewall blocking connection

**Solutions:**
1. Start vector database: `docker run -p 6333:6333 qdrant/qdrant`
2. Verify URL: `curl http://localhost:6333/health`
3. Check port availability: `netstat -tuln | grep 6333`
4. Configure firewall to allow connections

### CW-VECTOR-E-002: Collection Not Found
```plaintext
Error: Collection 'my-project' does not exist
HTTP 404: Collection not found
```

**Causes:**
- Collection name typo
- Collection not created
- Collection deleted
- Wrong database instance

**Solutions:**
1. Verify collection name spelling
2. Enable auto-creation: `export CW_AUTO_CREATE_COLLECTION=true`
3. Create collection manually through API
4. Check database instance and credentials

### CW-VECTOR-E-003: Dimension Mismatch
```plaintext
Error: Vector dimension mismatch
Expected: 1024, received: 1536
```

**Causes:**
- Changed embedding model without recreating collection
- Multiple embedding models in same collection
- Configuration inconsistency

**Solutions:**
1. Recreate collection with correct dimensions
2. Use consistent embedding model
3. Migrate existing vectors to new collection
4. Update collection configuration

### CW-VECTOR-E-004: Storage Full
```plaintext
Error: Vector database storage limit exceeded
Available space: 0 bytes
```

**Causes:**
- Disk space exhausted
- Storage quota exceeded
- Too many vectors stored

**Solutions:**
1. Free up disk space
2. Increase storage allocation
3. Clean up old collections
4. Implement vector pruning strategy

### CW-VECTOR-E-005: Query Timeout
```plaintext
Error: Vector search query timed out
Query exceeded 30 second timeout
```

**Causes:**
- Large dataset search
- Inefficient query parameters
- Database performance issues
- Resource contention

**Solutions:**
1. Increase query timeout: `export CW_QUERY_TIMEOUT=60`
2. Optimize query parameters (reduce top_k)
3. Enable query result caching
4. Scale database resources

## Indexing Errors (CW-INDEX-E-xxx)

### CW-INDEX-E-001: No Files Found
```plaintext
Error: No files found for indexing
Path: /path/to/project contains 0 eligible files
```

**Causes:**
- Empty directory
- All files filtered out
- Incorrect path
- Permission issues

**Solutions:**
1. Verify directory exists and has files
2. Check file filtering settings
3. Use absolute path instead of relative
4. Verify read permissions: `ls -la /path/to/project`

### CW-INDEX-E-002: File Too Large
```plaintext
Warning: File exceeds size limit, skipping
File: large_file.py (5.2MB > 1MB limit)
```

**Causes:**
- File exceeds configured size limit
- Generated or minified files
- Data files mixed with code

**Solutions:**
1. Increase limit: `export CW_MAX_FILE_SIZE=5242880`
2. Exclude large file directories
3. Use .gitignore patterns to skip generated files
4. Filter by file types only

### CW-INDEX-E-003: Parse Error
```plaintext
Error: Failed to parse file with AST
File: broken_syntax.py - Syntax error at line 45
```

**Causes:**
- Syntax errors in source code
- Unsupported language features
- File encoding issues
- Corrupted files

**Solutions:**
1. Fix syntax errors in source code
2. Enable fallback parsing: `export CW_FALLBACK_PARSING=true`
3. Check file encoding (should be UTF-8)
4. Skip problematic files with exclusion patterns

### CW-INDEX-E-004: Memory Exhausted
```plaintext
Error: Out of memory during indexing
Failed to allocate memory for batch processing
```

**Causes:**
- Large files or batches
- Insufficient system memory
- Memory leaks
- Too many concurrent workers

**Solutions:**
1. Reduce batch size: `export CW_BATCH_SIZE=4`
2. Limit workers: `export CW_INDEXING_WORKERS=2`
3. Enable streaming: `export CW_STREAM_PROCESSING=true`
4. Increase system memory or add swap

### CW-INDEX-E-005: Permission Denied
```plaintext
Error: Permission denied accessing file
File: /protected/file.py - Read permission required
```

**Causes:**
- Insufficient file permissions
- Protected system directories
- SELinux or AppArmor restrictions

**Solutions:**
1. Fix file permissions: `chmod 644 file.py`
2. Run with appropriate user permissions
3. Check SELinux context if applicable
4. Exclude protected directories

## Search Errors (CW-SEARCH-E-xxx)

### CW-SEARCH-E-001: Empty Query
```plaintext
Error: Search query cannot be empty
Received empty or whitespace-only query
```

**Causes:**
- No search terms provided
- Query contains only whitespace
- Query preprocessing removed all terms

**Solutions:**
1. Provide meaningful search terms
2. Check query preprocessing logic
3. Ensure minimum query length requirements
4. Validate input before processing

### CW-SEARCH-E-002: Query Too Long
```plaintext
Error: Search query exceeds maximum length
Query length: 2000 characters, limit: 1000
```

**Causes:**
- Very long search query
- Model input limits
- API constraints

**Solutions:**
1. Shorten search query
2. Break into multiple queries
3. Use query summarization
4. Increase provider limits if possible

### CW-SEARCH-E-003: No Results Found
```plaintext
Warning: No results found for query
Query returned 0 matches from 10,000 vectors
```

**Causes:**
- No relevant content indexed
- Search threshold too high
- Query terms not in corpus
- Index corruption

**Solutions:**
1. Lower search threshold: `export CW_SEARCH_THRESHOLD=0.5`
2. Try broader or different search terms
3. Verify content is properly indexed
4. Check index integrity

### CW-SEARCH-E-004: Search Timeout
```plaintext
Error: Search request timed out
Query exceeded 10 second timeout limit
```

**Causes:**
- Large vector database
- Complex query processing
- System resource constraints
- Network latency

**Solutions:**
1. Increase timeout: `export CW_SEARCH_TIMEOUT=30`
2. Optimize vector database configuration
3. Reduce search scope (use filters)
4. Enable result caching

## Configuration Errors (CW-CONFIG-E-xxx)

### CW-CONFIG-E-001: Invalid Configuration
```plaintext
Error: Configuration validation failed
Invalid value for 'chunk_size': must be between 50 and 5000
```

**Causes:**
- Configuration values out of range
- Wrong data types
- Missing required settings
- Conflicting options

**Solutions:**
1. Check configuration documentation
2. Validate against schema
3. Use default values for testing
4. Review all related settings

### CW-CONFIG-E-002: File Not Found
```plaintext
Error: Configuration file not found
Path: /path/to/config.toml does not exist
```

**Causes:**
- Wrong file path
- File deleted or moved
- Permission issues
- Typo in filename

**Solutions:**
1. Verify file path and name
2. Create configuration file if needed
3. Check file permissions
4. Use absolute path

### CW-CONFIG-E-003: Parse Error
```plaintext
Error: Failed to parse configuration file
TOML syntax error at line 15: Invalid escape sequence
```

**Causes:**
- Invalid TOML syntax
- Encoding issues
- Special characters in values

**Solutions:**
1. Validate TOML syntax with online parser
2. Check file encoding (use UTF-8)
3. Escape special characters properly
4. Use configuration generator tool

### CW-CONFIG-E-004: Environment Override
```plaintext
Warning: Environment variable overrides config file
CW_BATCH_SIZE=16 overrides config file value of 8
```

**Causes:**
- Environment variables taking precedence
- Multiple configuration sources
- Unclear configuration hierarchy

**Solutions:**
1. This is expected behavior (info only)
2. Remove environment variable if unwanted
3. Document configuration precedence
4. Use consistent configuration method

## Warning Messages (CW-*-W-xxx)

### CW-EMBED-W-001: Model Deprecated
```plaintext
Warning: Embedding model 'text-embedding-ada-002' is deprecated
Consider upgrading to 'text-embedding-3-small'
```

**Solutions:**
1. Update to recommended model
2. Test new model performance
3. Plan migration timeline
4. Monitor for breaking changes

### CW-VECTOR-W-001: Performance Degraded
```plaintext
Warning: Vector database performance below optimal
Query latency: 2.5s (normal: <200ms)
```

**Solutions:**
1. Check database resource usage
2. Optimize query parameters
3. Consider scaling database
4. Review index configuration

### CW-INDEX-W-001: Large Batch Size
```plaintext
Warning: Large batch size may cause memory issues
Current: 128, recommended maximum: 32
```

**Solutions:**
1. Reduce batch size for stability
2. Monitor memory usage
3. Test performance impact
4. Consider incremental processing

## Status Codes

### HTTP Status Codes

| Code | Meaning | Common Causes | Solutions |
|------|---------|---------------|-----------|
| **400** | Bad Request | Invalid parameters | Check request format |
| **401** | Unauthorized | Invalid API key | Verify credentials |
| **403** | Forbidden | Insufficient permissions | Check access rights |
| **404** | Not Found | Resource doesn't exist | Verify resource path |
| **429** | Rate Limited | Too many requests | Implement rate limiting |
| **500** | Internal Error | Server-side issue | Retry, check logs |
| **503** | Service Unavailable | Temporary outage | Wait and retry |

### Custom Status Codes

| Code | Component | Meaning | Action |
|------|-----------|---------|--------|
| **2001** | Indexing | Completed successfully | None |
| **2002** | Search | Results found | Process results |
| **3001** | Embedding | Rate limit warning | Slow down requests |
| **3002** | Vector DB | Performance warning | Optimize queries |
| **4001** | Config | Invalid setting | Fix configuration |
| **5001** | System | Resource exhausted | Scale resources |

## Diagnostic Tools

### Error Code Lookup

```bash
# Get detailed error information
codeweaver error-info CW-AUTH-E-001

# Check error patterns in logs
grep "CW-.*-E-" /var/log/codeweaver.log

# Validate configuration
codeweaver config validate --verbose
```

### Health Check Commands

```bash
# Comprehensive system check
codeweaver health-check --detailed

# Component-specific checks
codeweaver check embedding-provider
codeweaver check vector-database
codeweaver check configuration
```

### Log Analysis

```bash
# Enable debug logging for detailed errors
export CW_LOG_LEVEL=DEBUG
export CW_ERROR_DETAILS=true

# Search for specific error patterns
grep -E "CW-[A-Z]+-[EWI]-[0-9]{3}" codeweaver.log

# Count error types
grep -oE "CW-[A-Z]+-E-[0-9]{3}" codeweaver.log | sort | uniq -c
```

## Prevention and Best Practices

### Error Prevention

1. **Configuration Validation**
   - Always validate configuration before deployment
   - Use configuration schemas and type checking
   - Test with minimal viable configuration first

2. **Resource Monitoring**
   - Monitor memory, CPU, and disk usage
   - Set up alerts for resource thresholds
   - Implement graceful degradation strategies

3. **API Management**
   - Implement proper rate limiting
   - Use exponential backoff for retries
   - Monitor API quota and usage patterns

4. **Testing Strategies**
   - Test with various codebase sizes
   - Validate different file types and languages
   - Perform load testing before production

### Error Handling Best Practices

1. **Graceful Degradation**
   - Provide fallback options when possible
   - Maintain core functionality during partial failures
   - Clear user communication about limitations

2. **Retry Logic**
   - Implement exponential backoff
   - Limit retry attempts
   - Handle different error types appropriately

3. **User Communication**
   - Provide clear, actionable error messages
   - Include relevant context and suggestions
   - Avoid technical jargon when possible

4. **Monitoring and Alerting**
   - Track error rates and patterns
   - Set up automated alerts for critical errors
   - Maintain error rate SLAs

## Next Steps

- **[Troubleshooting Guide →](../getting-started/troubleshooting.md)**: Step-by-step problem resolution
- **[Configuration Reference →](../getting-started/configuration.md)**: Prevent configuration errors
- **[Performance Optimization →](./performance-benchmarks.md)**: Optimize for error-free operation
- **[Monitoring Setup →](../services/monitoring.md)**: Implement error tracking