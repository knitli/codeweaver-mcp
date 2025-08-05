<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Provider Comparison Matrix

**Complete technical comparison of all CodeWeaver providers for informed selection**

This comprehensive guide compares embedding providers, backend providers, and service providers with detailed technical specifications, performance characteristics, and selection criteria.

## Quick Selection Guide

### For Code Embeddings
| Provider | Best For | Quality | Cost | Setup |
|----------|----------|---------|------|-------|
| **Voyage AI** | Code-focused projects | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **OpenAI** | General purpose, established | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cohere** | Multilingual, enterprise | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **HuggingFace** | Custom models, research | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Local Models** | Privacy, offline use | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### For Vector Backends
| Backend | Best For | Performance | Scalability | Setup |
|---------|----------|-------------|-------------|-------|
| **Qdrant** | Production, hybrid search | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Pinecone** | Managed service, enterprise | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ChromaDB** | Development, prototyping | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Weaviate** | GraphQL, complex queries | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **PgVector** | Existing PostgreSQL | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Embedding Providers Detailed Comparison

### Technical Specifications

| Provider | Max Input | Dimensions | Batch Size | Rate Limits | Languages |
|----------|-----------|------------|------------|-------------|-----------|
| **Voyage AI** | 32K tokens | 1024-1536 | 128 docs | 100 req/min | 100+ |
| **OpenAI** | 8K tokens | 1536-3072 | 2048 docs | 500 req/min | 100+ |
| **Cohere** | 512 tokens | 1024-4096 | 96 docs | 1000 req/min | 100+ |
| **HuggingFace** | Variable | Variable | Variable | Model dependent | Variable |
| **Sentence Transformers** | Variable | Variable | No limit | No limit | Variable |

### Performance Characteristics

#### Voyage AI (Recommended for Code)
```yaml
Models:
  embedding: ["voyage-code-3", "voyage-context-3", "voyage-3-large"]
  reranking: ["voyage-rerank-2", "voyage-rerank-25"]
Strengths:
  - Code-optimized embeddings
  - Best-in-class quality for technical content
  - Combined embedding + reranking
  - Generous free tier (1M tokens/month)
Limitations:
  - Newer provider (less ecosystem)
  - Moderate rate limits
Performance:
  latency: "~200ms per batch"
  throughput: "32K tokens/request"
  accuracy: "95%+ on code similarity tasks"
```

#### OpenAI (General Purpose Leader)
```yaml
Models:
  embedding: ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
Strengths:
  - Mature ecosystem and tooling
  - High rate limits
  - Excellent general-purpose quality
  - Configurable dimensions (v3 models)
Limitations:
  - Higher cost than alternatives
  - Not code-optimized
  - No built-in reranking
Performance:
  latency: "~150ms per batch"
  throughput: "8K tokens/request"
  accuracy: "90%+ on general tasks"
```

#### Cohere (Enterprise & Multilingual)
```yaml
Models:
  embedding: ["embed-english-v3.0", "embed-multilingual-v3.0"]
  reranking: ["rerank-english-v3.0", "rerank-multilingual-v3.0"]
Strengths:
  - Strong multilingual support
  - Combined embedding + reranking
  - Enterprise-grade features
  - High rate limits
Limitations:
  - Higher cost
  - Shorter input length (512 tokens)
  - Less code-focused
Performance:
  latency: "~180ms per batch"
  throughput: "512 tokens/request"
  accuracy: "92%+ on multilingual tasks"
```

#### HuggingFace (Flexible & Open)
```yaml
Models:
  popular: ["microsoft/codebert-base", "sentence-transformers/all-mpnet-base-v2"]
Strengths:
  - Thousands of available models
  - Open source and transparent
  - No API costs for inference
  - Research-grade models
Limitations:
  - Complex setup and deployment
  - Variable model quality
  - Resource intensive
  - Limited commercial support
Performance:
  latency: "Variable (50ms-2s)"
  throughput: "Model dependent"
  accuracy: "Variable (60%-95%)"
```

#### Local Models (Privacy & Control)
```yaml
Options:
  - Sentence Transformers
  - ONNX Runtime models
  - Custom fine-tuned models
Strengths:
  - Complete privacy and control
  - No API costs or limits
  - Customizable for specific domains
  - Offline capability
Limitations:
  - Requires significant compute resources
  - Complex setup and maintenance
  - Lower quality than hosted models
  - No technical support
Performance:
  latency: "Hardware dependent (100ms-5s)"
  throughput: "Hardware dependent"
  accuracy: "Variable (70%-90%)"
```

### Cost Analysis

#### Monthly Cost Estimates (1M Tokens)

| Provider | Free Tier | Pay-per-use | Enterprise |
|----------|-----------|-------------|------------|
| **Voyage AI** | 1M tokens | $0.10/1M tokens | Custom |
| **OpenAI** | None | $0.13-0.30/1M tokens | Volume discounts |
| **Cohere** | 100 calls | $1.00/1M tokens | Custom |
| **HuggingFace** | Free inference | Infrastructure costs | Custom |
| **Local** | Free | Hardware/electricity | Hardware costs |

#### Total Cost of Ownership (Annual)

| Provider | Small Project | Medium Project | Large Project |
|----------|---------------|----------------|---------------|
| **Voyage AI** | $0-50 | $50-200 | $200-1000 |
| **OpenAI** | $50-150 | $150-500 | $500-2500 |
| **Cohere** | $100-300 | $300-1000 | $1000-5000 |
| **HuggingFace** | $0-100 | $100-500 | $500-2000 |
| **Local** | $500-2000 | $1000-5000 | $5000-20000 |

*Note: Costs include compute, storage, and operational overhead*

## Vector Backend Providers Comparison

### Technical Specifications

| Backend | Vector Types | Max Dimensions | Index Types | Query Types |
|---------|--------------|----------------|-------------|-------------|
| **Qdrant** | Dense, Sparse | 65536 | HNSW, IVF | Vector, Hybrid, Filter |
| **Pinecone** | Dense | 20000 | Proprietary | Vector, Metadata |
| **ChromaDB** | Dense | Unlimited | HNSW | Vector, Metadata |
| **Weaviate** | Dense | Unlimited | HNSW | Vector, GraphQL, Hybrid |
| **PgVector** | Dense | 16000 | IVF, HNSW | Vector, SQL |

### Performance Characteristics

#### Qdrant (Production Ready)
```yaml
Deployment:
  options: ["Cloud", "Self-hosted", "Docker"]
  architectures: ["Single-node", "Cluster", "Distributed"]
Strengths:
  - Native hybrid search (vector + keyword)
  - Excellent performance and scalability
  - Comprehensive filtering capabilities
  - Active development and community
  - Rust-based (memory efficient)
Limitations:
  - Newer ecosystem compared to Pinecone
  - Self-hosting requires more expertise
Performance:
  throughput: "10K+ QPS"
  latency: "<10ms for typical queries"
  capacity: "100M+ vectors per node"
```

#### Pinecone (Managed Simplicity)
```yaml
Deployment:
  options: ["Managed cloud only"]
  regions: ["Multiple AWS/GCP regions"]
Strengths:
  - Fully managed service
  - Proven at scale
  - Excellent documentation
  - Enterprise support
  - Simple API
Limitations:
  - Vendor lock-in
  - Higher costs
  - Limited customization
  - No hybrid search
Performance:
  throughput: "5K+ QPS"
  latency: "<100ms typical"
  capacity: "Unlimited (with cost)"
```

#### ChromaDB (Development Friendly)
```yaml
Deployment:
  options: ["Local", "Docker", "Cloud (beta)"]
  use_cases: ["Development", "Prototyping", "Small datasets"]
Strengths:
  - Simple setup and use
  - Good for development
  - Open source
  - Python-native
  - Built-in document processing
Limitations:
  - Not production-ready for scale
  - Limited query capabilities
  - Single-node only
  - Performance limitations
Performance:
  throughput: "1K QPS"
  latency: "<50ms for small datasets"
  capacity: "1M vectors recommended"
```

#### Weaviate (Knowledge Graphs)
```yaml
Deployment:
  options: ["Cloud", "Self-hosted", "Docker"]
  integrations: ["ML models", "GraphQL", "RESTful"]
Strengths:
  - GraphQL query interface
  - Built-in ML model integration
  - Knowledge graph capabilities
  - Hybrid search support
  - Schema-based approach
Limitations:
  - Complex setup and configuration
  - Steeper learning curve
  - Resource intensive
  - Smaller community
Performance:
  throughput: "3K+ QPS"
  latency: "<20ms typical"
  capacity: "10M+ vectors"
```

#### PgVector (PostgreSQL Integration)
```yaml
Deployment:
  options: ["PostgreSQL extension"]
  compatibility: ["PostgreSQL 11+"]
Strengths:
  - Integrates with existing PostgreSQL
  - ACID compliance
  - Familiar SQL interface
  - Battle-tested reliability
  - Rich ecosystem
Limitations:
  - Performance limitations vs specialized DBs
  - Manual index optimization
  - Limited vector-specific features
  - Scaling challenges
Performance:
  throughput: "500-2K QPS"
  latency: "<100ms with proper indexing"
  capacity: "1M+ vectors (with tuning)"
```

### Scalability and Deployment

#### Single Node Performance

| Backend | Vectors | QPS | Memory (GB) | Storage (GB) |
|---------|---------|-----|-------------|--------------|
| **Qdrant** | 10M | 5K | 32 | 100 |
| **Pinecone** | 10M | 3K | Managed | Managed |
| **ChromaDB** | 1M | 1K | 16 | 50 |
| **Weaviate** | 10M | 2K | 64 | 200 |
| **PgVector** | 1M | 500 | 32 | 100 |

#### Multi-Node Scaling

| Backend | Horizontal Scaling | Replication | Sharding |
|---------|-------------------|-------------|----------|
| **Qdrant** | ✅ Cluster mode | ✅ Built-in | ✅ Auto-sharding |
| **Pinecone** | ✅ Managed | ✅ Built-in | ✅ Transparent |
| **ChromaDB** | ❌ Single-node | ❌ No | ❌ No |
| **Weaviate** | ✅ Cluster mode | ✅ Built-in | ✅ Manual |
| **PgVector** | ✅ PostgreSQL tools | ✅ PostgreSQL | ⚠️ Manual |

## Service Providers Comparison

### Core Services

| Service | Provider | Purpose | Performance | Customizable |
|---------|----------|---------|-------------|--------------|
| **Chunking** | FastMCP | AST-aware code segmentation | ⭐⭐⭐⭐⭐ | ✅ |
| **Filtering** | FastMCP | File discovery & filtering | ⭐⭐⭐⭐⭐ | ✅ |
| **Caching** | Redis/Memory | Performance optimization | ⭐⭐⭐⭐ | ✅ |
| **Rate Limiting** | Token Bucket | API protection | ⭐⭐⭐⭐⭐ | ✅ |
| **Telemetry** | PostHog | Usage analytics | ⭐⭐⭐⭐ | ✅ |

### FastMCP Middleware

#### Chunking Service
```yaml
Features:
  - AST-aware code parsing using ast-grep
  - Language-specific intelligent boundaries
  - Fallback to text-based chunking
  - Streaming support for large files
Supported Languages: 20+
Performance:
  throughput: "1K files/sec"
  memory: "Low overhead streaming"
Customization:
  - Custom AST patterns
  - Configurable chunk sizes
  - Language-specific rules
```

#### Filtering Service
```yaml
Features:
  - Gitignore integration
  - Parallel directory scanning
  - File type detection
  - Custom pattern matching
Performance:
  throughput: "10K files/sec scan"
  memory: "Minimal footprint"
Customization:
  - Include/exclude patterns
  - File size limits
  - Language-specific filtering
```

## Selection Decision Matrix

### Use Case-Based Recommendations

#### Small Development Team (< 10 developers)
```yaml
Embedding: Voyage AI (free tier)
Backend: Qdrant (local Docker)
Services: All default FastMCP services
Total Monthly Cost: $0-50
Setup Time: 2-4 hours
```

#### Medium Enterprise Team (10-100 developers)
```yaml
Embedding: Voyage AI or OpenAI
Backend: Qdrant Cloud or Pinecone
Services: All services + monitoring
Total Monthly Cost: $200-1000
Setup Time: 1-2 days
```

#### Large Enterprise (100+ developers)
```yaml
Embedding: Multi-provider setup (Voyage + OpenAI)
Backend: Qdrant cluster or Pinecone
Services: Full service stack + custom
Total Monthly Cost: $1000-5000
Setup Time: 1-2 weeks
```

#### Research/Academic Use
```yaml
Embedding: HuggingFace or Local models
Backend: Qdrant (self-hosted) or PgVector
Services: Core services only
Total Monthly Cost: $0-200
Setup Time: 3-7 days
```

#### Privacy-Critical Applications
```yaml
Embedding: Local Sentence Transformers
Backend: Self-hosted Qdrant or PgVector
Services: On-premise deployment
Total Monthly Cost: $500-2000 (infrastructure)
Setup Time: 1-4 weeks
```

### Technical Requirements Matrix

| Requirement | Recommended Providers |
|-------------|----------------------|
| **High Accuracy** | Voyage AI + Qdrant |
| **Low Latency** | OpenAI + Pinecone |
| **Cost Optimization** | HuggingFace + ChromaDB |
| **Scalability** | Multi-provider + Qdrant Cluster |
| **Privacy** | Local Models + Self-hosted Qdrant |
| **Multilingual** | Cohere + Weaviate |
| **Enterprise** | Voyage/OpenAI + Pinecone |
| **Research** | HuggingFace + PgVector |

## Migration Considerations

### Provider Migration Paths

#### From OpenAI to Voyage AI
```yaml
Complexity: Low
Steps:
  1. Update API key configuration
  2. Change model names
  3. Test embedding compatibility
  4. Monitor quality metrics
Downtime: Minimal
Data Migration: Not required
```

#### Between Vector Backends
```yaml
Complexity: High
Steps:
  1. Export existing vectors
  2. Set up new backend
  3. Re-index data
  4. Update application config
  5. Performance testing
Downtime: Several hours
Data Migration: Full re-indexing required
```

### Best Practices for Provider Selection

1. **Start Simple**: Begin with recommended defaults (Voyage AI + Qdrant)
2. **Measure Performance**: Benchmark with your actual data
3. **Plan for Scale**: Consider growth requirements
4. **Budget Appropriately**: Include operational costs
5. **Test Thoroughly**: Validate quality with domain-specific queries
6. **Monitor Continuously**: Track performance and costs
7. **Document Decisions**: Record rationale for future reference

## Next Steps

- **[Complete Configuration :material-arrow-right-circle:](../getting-started/configuration.md)**: Set up your chosen providers
- **[Troubleshooting Guide :material-arrow-right-circle:](../getting-started/troubleshooting.md)**: Resolve common provider issues
- **[Performance Optimization :material-arrow-right-circle:](../user-guide/performance.md)**: Fine-tune your setup
- **[Language Support :material-arrow-right-circle:](./language-support.md)**: Understand language-specific features
