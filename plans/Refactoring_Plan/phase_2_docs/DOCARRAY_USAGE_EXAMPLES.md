# DocArray Usage Examples

<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

**Version**: 1.0.0  
**Date**: 2025-01-26  
**Purpose**: Practical usage examples for DocArray integration

## Overview

This document provides comprehensive usage examples for CodeWeaver's DocArray integration, demonstrating real-world scenarios, best practices, and advanced features.

## Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [Code Search Examples](#code-search-examples)
3. [Hybrid Search Examples](#hybrid-search-examples)
4. [Advanced Configuration Examples](#advanced-configuration-examples)
5. [Performance Optimization Examples](#performance-optimization-examples)
6. [Migration Examples](#migration-examples)
7. [Production Deployment Examples](#production-deployment-examples)

## Basic Usage Examples

### Example 1: Simple Qdrant Setup

```python
"""Basic DocArray Qdrant backend setup for code search."""

import asyncio
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import QdrantDocArrayConfig

async def basic_qdrant_example():
    # Create factory with DocArray support
    factory = CodeWeaverFactory(enable_docarray=True)
    
    # Configure Qdrant DocArray backend
    config = QdrantDocArrayConfig(
        url="http://localhost:6333",
        api_key="your-api-key",
        collection_name="my_codebase",
        schema_config={
            "embedding_dimension": 512,
            "schema_template": "code_search",
            "include_sparse_vectors": True,
        }
    )
    
    # Create backend
    backend = factory.create_backend(config)
    
    # Create collection
    await backend.create_collection(
        name="my_codebase",
        dimension=512,
        distance_metric="cosine"
    )
    
    # Index some code vectors
    vectors = [
        {
            "id": "func_1",
            "vector": [0.1] * 512,  # Your embedding here
            "payload": {
                "content": "def hello_world(): print('Hello')",
                "file_path": "src/main.py",
                "language": "python",
                "function_name": "hello_world",
            }
        },
        {
            "id": "func_2", 
            "vector": [0.2] * 512,
            "payload": {
                "content": "function greetUser(name) { console.log(`Hello ${name}`); }",
                "file_path": "src/app.js",
                "language": "javascript",
                "function_name": "greetUser",
            }
        }
    ]
    
    await backend.upsert_vectors("my_codebase", vectors)
    
    # Search for similar code
    query_vector = [0.15] * 512  # Your query embedding
    results = await backend.search_vectors(
        collection_name="my_codebase",
        query_vector=query_vector,
        limit=5
    )
    
    for result in results:
        print(f"ID: {result.id}, Score: {result.score:.3f}")
        print(f"Content: {result.payload['content']}")
        print(f"File: {result.payload['file_path']}")
        print("---")

# Run the example
asyncio.run(basic_qdrant_example())
```

### Example 2: Pinecone Document Search

```python
"""Document search using Pinecone DocArray backend."""

import asyncio
import os
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import PineconeDocArrayConfig

async def pinecone_document_search():
    factory = CodeWeaverFactory(enable_docarray=True)
    
    config = PineconeDocArrayConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment="us-west1-gcp",
        index_name="document-search",
        schema_config={
            "embedding_dimension": 1536,  # OpenAI ada-002
            "schema_template": "semantic_search",
            "metadata_fields": {
                "document_type": "str",
                "category": "str",
                "priority": "int",
            }
        }
    )
    
    backend = factory.create_backend(config)
    
    # Create collection
    await backend.create_collection(
        name="documents", 
        dimension=1536
    )
    
    # Index documents
    documents = [
        {
            "id": "doc_1",
            "vector": [0.1] * 1536,  # OpenAI embedding
            "payload": {
                "content": "Machine learning algorithms for data analysis",
                "title": "ML Guide",
                "author": "Data Scientist",
                "document_type": "guide",
                "category": "machine learning",
                "priority": 1,
            }
        },
        {
            "id": "doc_2",
            "vector": [0.2] * 1536,
            "payload": {
                "content": "Vector database optimization techniques",
                "title": "Vector DB Optimization",
                "author": "Database Engineer", 
                "document_type": "tutorial",
                "category": "databases",
                "priority": 2,
            }
        }
    ]
    
    await backend.upsert_vectors("documents", documents)
    
    # Search with filtering
    query_vector = [0.15] * 1536
    results = await backend.search_vectors(
        collection_name="documents",
        query_vector=query_vector,
        limit=10,
        search_filter={
            "must": [
                {"key": "category", "match": {"value": "machine learning"}}
            ]
        }
    )
    
    for result in results:
        print(f"Document: {result.payload['title']}")
        print(f"Author: {result.payload['author']}")
        print(f"Score: {result.score:.3f}")
        print("---")

asyncio.run(pinecone_document_search())
```

## Code Search Examples

### Example 3: Advanced Code Search with Metadata

```python
"""Advanced code search with structured metadata and filtering."""

import asyncio
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import QdrantDocArrayConfig

class CodeSearchExample:
    def __init__(self):
        self.factory = CodeWeaverFactory(enable_docarray=True)
        self.backend = None
    
    async def setup(self):
        """Setup the code search backend."""
        config = QdrantDocArrayConfig(
            url="http://localhost:6333",
            collection_name="advanced_codebase",
            schema_config={
                "embedding_dimension": 768,
                "include_sparse_vectors": True,
                "metadata_fields": {
                    "file_path": "str",
                    "language": "str", 
                    "function_name": "str",
                    "class_name": "str",
                    "line_number": "int",
                    "complexity_score": "float",
                    "test_coverage": "float",
                    "is_public": "bool",
                },
                "custom_fields": {
                    "ast_hash": ("str", "AST hash for change detection"),
                    "dependencies": ("list[str]", "Function dependencies"),
                    "git_hash": ("str | None", "Git commit hash"),
                }
            }
        )
        
        self.backend = self.factory.create_backend(config)
        await self.backend.create_collection(
            name="advanced_codebase",
            dimension=768
        )
    
    async def index_codebase(self, code_files):
        """Index a codebase with rich metadata."""
        vectors = []
        
        for file_info in code_files:
            for func_info in file_info["functions"]:
                vector = {
                    "id": f"{file_info['path']}::{func_info['name']}",
                    "vector": func_info["embedding"],
                    "payload": {
                        "content": func_info["code"],
                        "file_path": file_info["path"],
                        "language": file_info["language"],
                        "function_name": func_info["name"],
                        "class_name": func_info.get("class_name"),
                        "line_number": func_info["line_number"],
                        "complexity_score": func_info["complexity"],
                        "test_coverage": func_info.get("coverage", 0.0),
                        "is_public": func_info["is_public"],
                        "ast_hash": func_info["ast_hash"],
                        "dependencies": func_info["dependencies"],
                        "git_hash": file_info.get("git_hash"),
                    },
                    "sparse_vector": func_info.get("sparse_embedding", {})
                }
                vectors.append(vector)
        
        # Batch index for performance
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            await self.backend.upsert_vectors("advanced_codebase", batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
    
    async def search_by_functionality(self, query_embedding, language=None, min_coverage=None):
        """Search for functions by functionality."""
        filters = []
        
        if language:
            filters.append({"key": "language", "match": {"value": language}})
        
        if min_coverage:
            filters.append({"key": "test_coverage", "range": {"gte": min_coverage}})
        
        # Only search public functions
        filters.append({"key": "is_public", "match": {"value": True}})
        
        search_filter = {"must": filters} if filters else None
        
        results = await self.backend.search_vectors(
            collection_name="advanced_codebase",
            query_vector=query_embedding,
            limit=10,
            search_filter=search_filter,
            score_threshold=0.7
        )
        
        return results
    
    async def find_similar_functions(self, function_id, limit=5):
        """Find functions similar to a given function."""
        # First, get the target function's embedding
        # In practice, you'd query by ID or maintain a mapping
        target_embedding = [0.1] * 768  # Placeholder
        
        results = await self.backend.search_vectors(
            collection_name="advanced_codebase",
            query_vector=target_embedding,
            limit=limit + 1  # +1 to exclude the function itself
        )
        
        # Filter out the original function
        similar_functions = [r for r in results if r.id != function_id][:limit]
        return similar_functions
    
    async def search_high_quality_code(self, query_embedding):
        """Search for high-quality, well-tested code."""
        results = await self.backend.search_vectors(
            collection_name="advanced_codebase",
            query_vector=query_embedding,
            limit=20,
            search_filter={
                "must": [
                    {"key": "test_coverage", "range": {"gte": 0.8}},
                    {"key": "complexity_score", "range": {"lte": 10.0}},
                    {"key": "is_public", "match": {"value": True}}
                ]
            }
        )
        
        return results

# Example usage
async def run_code_search_example():
    searcher = CodeSearchExample()
    await searcher.setup()
    
    # Mock codebase data
    codebase = [
        {
            "path": "src/utils.py",
            "language": "python",
            "git_hash": "abc123",
            "functions": [
                {
                    "name": "calculate_similarity",
                    "code": "def calculate_similarity(a, b): return dot(a, b) / (norm(a) * norm(b))",
                    "embedding": [0.1] * 768,
                    "sparse_embedding": {"calculate": 0.5, "similarity": 0.8, "cosine": 0.3},
                    "line_number": 15,
                    "complexity": 3.2,
                    "coverage": 0.95,
                    "is_public": True,
                    "ast_hash": "hash123",
                    "dependencies": ["numpy", "math"],
                    "class_name": None,
                }
            ]
        }
    ]
    
    # Index the codebase
    await searcher.index_codebase(codebase)
    
    # Search for similarity calculation functions in Python
    query_embedding = [0.1] * 768
    results = await searcher.search_by_functionality(
        query_embedding, 
        language="python",
        min_coverage=0.8
    )
    
    print("High-quality Python functions for similarity:")
    for result in results:
        print(f"Function: {result.payload['function_name']}")
        print(f"File: {result.payload['file_path']}")
        print(f"Coverage: {result.payload['test_coverage']:.1%}")
        print(f"Score: {result.score:.3f}")
        print("---")

asyncio.run(run_code_search_example())
```

## Hybrid Search Examples

### Example 4: Hybrid Code Search with Sparse Vectors

```python
"""Hybrid search combining semantic embeddings with keyword matching."""

import asyncio
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import QdrantDocArrayConfig
from codeweaver.backends.base import HybridSearchBackend

class HybridCodeSearch:
    def __init__(self):
        self.factory = CodeWeaverFactory(enable_docarray=True)
        self.backend = None
    
    async def setup(self):
        """Setup hybrid search backend."""
        config = QdrantDocArrayConfig(
            url="http://localhost:6333",
            collection_name="hybrid_codebase",
            enable_hybrid_search=True,
            schema_config={
                "embedding_dimension": 512,
                "include_sparse_vectors": True,
                "schema_template": "code_search",
            }
        )
        
        self.backend = self.factory.create_backend(config)
        
        # Ensure backend supports hybrid search
        if not isinstance(self.backend, HybridSearchBackend):
            raise ValueError("Backend doesn't support hybrid search")
        
        await self.backend.create_collection(
            name="hybrid_codebase",
            dimension=512
        )
        
        # Create sparse index for keywords
        await self.backend.create_sparse_index(
            collection_name="hybrid_codebase",
            fields=["content", "keywords"],
            index_type="bm25"
        )
    
    async def index_code_with_keywords(self, code_snippets):
        """Index code with both dense and sparse vectors."""
        vectors = []
        
        for snippet in code_snippets:
            # Extract keywords from code
            keywords = self._extract_keywords(snippet["code"])
            sparse_vector = self._create_sparse_vector(keywords)
            
            vector = {
                "id": snippet["id"],
                "vector": snippet["dense_embedding"],
                "payload": {
                    "content": snippet["code"],
                    "file_path": snippet["file_path"],
                    "language": snippet["language"],
                    "function_name": snippet.get("function_name"),
                    "keywords": keywords,
                },
                "sparse_vector": sparse_vector
            }
            vectors.append(vector)
        
        await self.backend.upsert_vectors("hybrid_codebase", vectors)
    
    async def hybrid_search(self, query_text, dense_embedding, alpha=0.5):
        """Perform hybrid search with both semantic and keyword matching."""
        results = await self.backend.hybrid_search(
            collection_name="hybrid_codebase",
            dense_vector=dense_embedding,
            sparse_query=query_text,
            limit=10,
            hybrid_strategy="rrf",  # Reciprocal Rank Fusion
            alpha=alpha  # 0.5 = equal weight to dense and sparse
        )
        
        return results
    
    async def semantic_search(self, dense_embedding):
        """Pure semantic search using dense vectors only."""
        results = await self.backend.search_vectors(
            collection_name="hybrid_codebase",
            query_vector=dense_embedding,
            limit=10
        )
        
        return results
    
    async def keyword_search(self, keywords):
        """Keyword-based search using sparse vectors."""
        sparse_query = self._create_sparse_vector(keywords)
        
        # Use hybrid search with alpha=0 (sparse only)
        results = await self.backend.hybrid_search(
            collection_name="hybrid_codebase",
            dense_vector=[0.0] * 512,  # Dummy dense vector
            sparse_query=sparse_query,
            limit=10,
            alpha=0.0  # Only sparse search
        )
        
        return results
    
    def _extract_keywords(self, code):
        """Extract meaningful keywords from code."""
        import re
        
        # Simple keyword extraction (in practice, use proper AST parsing)
        keywords = set()
        
        # Function names
        func_matches = re.findall(r'def\s+(\w+)', code)
        keywords.update(func_matches)
        
        # Class names  
        class_matches = re.findall(r'class\s+(\w+)', code)
        keywords.update(class_matches)
        
        # Import statements
        import_matches = re.findall(r'import\s+(\w+)', code)
        keywords.update(import_matches)
        
        # Variable assignments
        var_matches = re.findall(r'(\w+)\s*=', code)
        keywords.update(var_matches)
        
        # API calls
        api_matches = re.findall(r'\.(\w+)\(', code)
        keywords.update(api_matches)
        
        return list(keywords)
    
    def _create_sparse_vector(self, keywords):
        """Create sparse vector from keywords using TF-IDF-like scoring."""
        if not keywords:
            return {}
        
        # Simple TF scoring (in practice, use proper TF-IDF)
        term_freq = {}
        for keyword in keywords:
            term_freq[keyword] = term_freq.get(keyword, 0) + 1
        
        # Normalize
        max_freq = max(term_freq.values())
        sparse_vector = {
            word: freq / max_freq 
            for word, freq in term_freq.items()
        }
        
        return sparse_vector

# Example usage
async def run_hybrid_search_example():
    searcher = HybridCodeSearch()
    await searcher.setup()
    
    # Mock code data
    code_snippets = [
        {
            "id": "snippet_1",
            "code": """
def calculate_cosine_similarity(vector_a, vector_b):
    import numpy as np
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)
""",
            "dense_embedding": [0.1] * 512,
            "file_path": "similarity.py",
            "language": "python",
            "function_name": "calculate_cosine_similarity"
        },
        {
            "id": "snippet_2", 
            "code": """
class VectorDatabase:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
    
    def add_vector(self, vector_id, embedding):
        self.vectors.append({"id": vector_id, "embedding": embedding})
""",
            "dense_embedding": [0.2] * 512,
            "file_path": "database.py", 
            "language": "python",
            "function_name": None
        }
    ]
    
    # Index code snippets
    await searcher.index_code_with_keywords(code_snippets)
    
    # Test different search modes
    query_embedding = [0.15] * 512
    
    print("=== Semantic Search ===")
    semantic_results = await searcher.semantic_search(query_embedding)
    for result in semantic_results:
        print(f"Function: {result.payload.get('function_name', 'N/A')}")
        print(f"Score: {result.score:.3f}")
        print("---")
    
    print("\n=== Keyword Search ===")
    keyword_results = await searcher.keyword_search(["similarity", "cosine", "vector"])
    for result in keyword_results:
        print(f"Function: {result.payload.get('function_name', 'N/A')}")
        print(f"Keywords: {result.payload.get('keywords', [])}")
        print("---")
    
    print("\n=== Hybrid Search ===")
    hybrid_results = await searcher.hybrid_search(
        query_text="cosine similarity calculation",
        dense_embedding=query_embedding,
        alpha=0.5
    )
    for result in hybrid_results:
        print(f"Function: {result.payload.get('function_name', 'N/A')}")
        print(f"Score: {result.score:.3f}")
        print(f"File: {result.payload['file_path']}")
        print("---")

asyncio.run(run_hybrid_search_example())
```

## Advanced Configuration Examples

### Example 5: Multi-Backend Setup with Fallbacks

```python
"""Advanced setup with multiple backends and automatic fallbacks."""

import asyncio
import logging
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import (
    QdrantDocArrayConfig, 
    PineconeDocArrayConfig,
    DocArrayBackendConfig
)

class MultiBackendManager:
    def __init__(self):
        self.factory = CodeWeaverFactory(enable_docarray=True)
        self.primary_backend = None
        self.fallback_backend = None
        
    async def setup(self):
        """Setup primary and fallback backends."""
        # Primary: Qdrant with full features
        primary_config = QdrantDocArrayConfig(
            url="http://localhost:6333",
            collection_name="primary_collection",
            enable_hybrid_search=True,
            schema_config={
                "embedding_dimension": 768,
                "include_sparse_vectors": True,
                "schema_template": "code_search",
            },
            retry_attempts=3,
            connection_timeout=30.0,
        )
        
        # Fallback: In-memory for development/testing
        fallback_config = DocArrayBackendConfig(
            provider="docarray_inmemory",
            schema_config={
                "embedding_dimension": 768,
                "include_sparse_vectors": False,
            }
        )
        
        try:
            self.primary_backend = self.factory.create_backend(primary_config)
            await self.primary_backend.create_collection("primary_collection", 768)
            print("Primary backend (Qdrant) initialized successfully")
        except Exception as e:
            print(f"Primary backend failed: {e}")
            self.primary_backend = None
        
        try:
            self.fallback_backend = self.factory.create_backend(fallback_config)
            await self.fallback_backend.create_collection("fallback_collection", 768)
            print("Fallback backend initialized successfully")
        except Exception as e:
            print(f"Fallback backend failed: {e}")
            raise RuntimeError("No usable backend available")
    
    async def get_active_backend(self):
        """Get the currently active backend with health checking."""
        if self.primary_backend:
            try:
                # Health check
                await self.primary_backend.get_collection_info("primary_collection")
                return self.primary_backend, "primary_collection"
            except Exception as e:
                print(f"Primary backend health check failed: {e}")
                
        if self.fallback_backend:
            return self.fallback_backend, "fallback_collection"
            
        raise RuntimeError("No healthy backend available")
    
    async def search_with_fallback(self, query_vector, **kwargs):
        """Search with automatic fallback on failure."""
        backend, collection = await self.get_active_backend()
        
        try:
            results = await backend.search_vectors(
                collection_name=collection,
                query_vector=query_vector,
                **kwargs
            )
            return results, "primary" if backend == self.primary_backend else "fallback"
            
        except Exception as e:
            print(f"Search failed on active backend: {e}")
            
            # Try fallback if we were using primary
            if backend == self.primary_backend and self.fallback_backend:
                try:
                    results = await self.fallback_backend.search_vectors(
                        collection_name="fallback_collection",
                        query_vector=query_vector,
                        **kwargs
                    )
                    return results, "fallback"
                except Exception as fallback_error:
                    print(f"Fallback search also failed: {fallback_error}")
            
            raise RuntimeError("All backends failed")

# Example usage
async def run_multi_backend_example():
    manager = MultiBackendManager()
    await manager.setup()
    
    # Test search with fallback
    query_vector = [0.1] * 768
    
    try:
        results, backend_used = await manager.search_with_fallback(
            query_vector=query_vector,
            limit=5
        )
        print(f"Search completed using {backend_used} backend")
        print(f"Found {len(results)} results")
        
    except Exception as e:
        print(f"Search failed completely: {e}")

asyncio.run(run_multi_backend_example())
```

## Performance Optimization Examples

### Example 6: Batch Processing and Performance Monitoring

```python
"""Performance optimization with batch processing and monitoring."""

import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any
from codeweaver import CodeWeaverFactory
from codeweaver.backends.docarray.config import QdrantDocArrayConfig

@dataclass
class PerformanceMetrics:
    operation: str
    duration: float
    batch_size: int
    throughput: float
    memory_usage: float = 0.0

class PerformanceOptimizedSearch:
    def __init__(self):
        self.factory = CodeWeaverFactory(enable_docarray=True)
        self.backend = None
        self.metrics: List[PerformanceMetrics] = []
    
    async def setup(self, batch_size=200):
        """Setup optimized backend configuration."""
        config = QdrantDocArrayConfig(
            url="http://localhost:6333",
            collection_name="performance_test",
            
            # Performance optimizations
            batch_size=batch_size,
            enable_async=True,
            enable_compression=True,
            enable_caching=True,
            
            schema_config={
                "embedding_dimension": 512,
                "include_sparse_vectors": False,  # Disable for better performance
                "enable_validation": False,  # Disable for bulk operations
            },
            
            # Connection optimizations
            connection_timeout=60.0,
            retry_attempts=1,  # Fast failure for performance testing
            
            # Database optimizations
            db_config={
                "timeout": 30.0,
                "prefer_grpc": True,  # Better performance than HTTP
            },
            
            # Runtime optimizations
            runtime_config={
                "batch_size": batch_size,
                "scroll_size": batch_size * 2,
            }
        )
        
        self.backend = self.factory.create_backend(config)
        await self.backend.create_collection("performance_test", 512)
    
    async def benchmark_bulk_indexing(self, vectors: List[Dict[str, Any]], batch_size: int = 100):
        """Benchmark bulk indexing performance."""
        total_vectors = len(vectors)
        batches = [vectors[i:i + batch_size] for i in range(0, total_vectors, batch_size)]
        
        print(f"Indexing {total_vectors} vectors in {len(batches)} batches of {batch_size}")
        
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(batches):
            batch_start = time.time()
            
            await self.backend.upsert_vectors("performance_test", batch)
            
            batch_duration = time.time() - batch_start
            batch_times.append(batch_duration)
            
            # Progress reporting
            if (i + 1) % 10 == 0:
                avg_batch_time = statistics.mean(batch_times[-10:])
                throughput = batch_size / avg_batch_time
                print(f"Batch {i+1}/{len(batches)}: {throughput:.1f} vectors/sec")
        
        total_duration = time.time() - start_time
        total_throughput = total_vectors / total_duration
        
        metrics = PerformanceMetrics(
            operation="bulk_indexing",
            duration=total_duration,
            batch_size=batch_size,
            throughput=total_throughput
        )
        self.metrics.append(metrics)
        
        print(f"\nIndexing Complete:")
        print(f"Total time: {total_duration:.2f} seconds")
        print(f"Average throughput: {total_throughput:.1f} vectors/sec")
        print(f"Batch time std dev: {statistics.stdev(batch_times):.3f} seconds")
        
        return metrics
    
    async def benchmark_search_performance(self, query_vectors: List[List[float]], batch_search=False):
        """Benchmark search performance."""
        if batch_search and hasattr(self.backend, 'batch_search'):
            # Batch search if supported
            start_time = time.time()
            all_results = await self.backend.batch_search(
                collection_name="performance_test",
                query_vectors=query_vectors,
                limit=10
            )
            duration = time.time() - start_time
            
            total_queries = len(query_vectors)
            
        else:
            # Individual searches
            start_time = time.time()
            all_results = []
            
            for query_vector in query_vectors:
                results = await self.backend.search_vectors(
                    collection_name="performance_test",
                    query_vector=query_vector,
                    limit=10
                )
                all_results.append(results)
            
            duration = time.time() - start_time
            total_queries = len(query_vectors)
        
        throughput = total_queries / duration
        
        metrics = PerformanceMetrics(
            operation="search",
            duration=duration,
            batch_size=total_queries,
            throughput=throughput
        )
        self.metrics.append(metrics)
        
        print(f"\nSearch Performance:")
        print(f"Queries: {total_queries}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Throughput: {throughput:.1f} queries/sec")
        print(f"Average latency: {(duration/total_queries)*1000:.1f} ms")
        
        return metrics, all_results
    
    async def benchmark_concurrent_searches(self, query_vectors: List[List[float]], concurrency=10):
        """Benchmark concurrent search performance."""
        async def search_batch(batch_queries):
            results = []
            for query in batch_queries:
                result = await self.backend.search_vectors(
                    collection_name="performance_test",
                    query_vector=query,
                    limit=10
                )
                results.append(result)
            return results
        
        # Split queries into concurrent batches
        batch_size = len(query_vectors) // concurrency
        batches = [
            query_vectors[i:i + batch_size] 
            for i in range(0, len(query_vectors), batch_size)
        ]
        
        start_time = time.time()
        
        # Run batches concurrently
        tasks = [search_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        total_queries = len(query_vectors)
        throughput = total_queries / duration
        
        metrics = PerformanceMetrics(
            operation="concurrent_search",
            duration=duration,
            batch_size=total_queries,
            throughput=throughput
        )
        self.metrics.append(metrics)
        
        print(f"\nConcurrent Search Performance (concurrency={concurrency}):")
        print(f"Queries: {total_queries}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Throughput: {throughput:.1f} queries/sec")
        
        return metrics
    
    def print_performance_summary(self):
        """Print comprehensive performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        for metric in self.metrics:
            print(f"\n{metric.operation.upper()}:")
            print(f"  Duration: {metric.duration:.2f} seconds")
            print(f"  Batch size: {metric.batch_size}")
            print(f"  Throughput: {metric.throughput:.1f} ops/sec")
        
        # Compare indexing vs search performance
        indexing_metrics = [m for m in self.metrics if m.operation == "bulk_indexing"]
        search_metrics = [m for m in self.metrics if "search" in m.operation]
        
        if indexing_metrics and search_metrics:
            avg_indexing = statistics.mean([m.throughput for m in indexing_metrics])
            avg_search = statistics.mean([m.throughput for m in search_metrics])
            
            print(f"\nAVERAGE PERFORMANCE:")
            print(f"  Indexing: {avg_indexing:.1f} vectors/sec")
            print(f"  Search: {avg_search:.1f} queries/sec")

# Example usage
async def run_performance_benchmark():
    optimizer = PerformanceOptimizedSearch()
    
    # Test different batch sizes
    for batch_size in [50, 100, 200]:
        print(f"\n{'='*20} BATCH SIZE {batch_size} {'='*20}")
        
        await optimizer.setup(batch_size=batch_size)
        
        # Generate test data
        test_vectors = [
            {
                "id": f"vec_{i}",
                "vector": [i * 0.001] * 512,
                "payload": {"content": f"Test content {i}"}
            }
            for i in range(1000)  # 1k vectors for testing
        ]
        
        # Benchmark indexing
        await optimizer.benchmark_bulk_indexing(test_vectors, batch_size)
        
        # Generate query vectors
        query_vectors = [[i * 0.001] * 512 for i in range(100)]
        
        # Benchmark search
        await optimizer.benchmark_search_performance(query_vectors)
        
        # Benchmark concurrent search
        await optimizer.benchmark_concurrent_searches(query_vectors, concurrency=5)
    
    # Print final summary
    optimizer.print_performance_summary()

asyncio.run(run_performance_benchmark())
```

This comprehensive usage guide provides practical examples for every major use case of the DocArray integration, from basic setup to advanced performance optimization. Each example includes complete, runnable code with proper error handling and best practices.