# Performance Guidelines for Extensions

This guide covers performance optimization strategies for CodeWeaver extensions, including benchmarking, profiling, and best practices for high-performance implementations.

## 🎯 Overview

Performance is critical for CodeWeaver extensions, especially when processing large codebases or handling high-volume requests. This guide provides:

- **Performance Requirements**: SLA targets and benchmarks
- **Optimization Strategies**: Proven techniques for each extension type
- **Profiling Tools**: Monitoring and measurement approaches
- **Best Practices**: Common patterns for high-performance extensions

## 📊 Performance Requirements

### Provider Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Embedding Latency** | < 2 seconds | Single document embedding |
| **Batch Throughput** | > 10 docs/sec | Batch embedding processing |
| **Memory Usage** | < 500MB | Peak memory during processing |
| **API Utilization** | > 80% | Efficient use of provider API limits |

### Backend Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Search Latency** | < 100ms | Vector similarity search |
| **Upsert Throughput** | > 1000 vectors/sec | Batch vector insertion |
| **Index Build Time** | < 10min/1M vectors | Large collection indexing |
| **Memory Efficiency** | < 2x vector data | Index overhead ratio |

### Service Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Processing Latency** | < 50ms | Service operation latency |
| **Throughput** | > 100 ops/sec | Service operation throughput |
| **Resource Usage** | < 100MB | Service memory footprint |
| **Health Check** | < 5ms | Service health validation |

## 🚀 Provider Optimization

### Embedding Provider Optimization

#### Batch Processing Strategy
```python
class OptimizedEmbeddingProvider(EmbeddingProviderBase):
    """High-performance embedding provider with optimizations."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.session_pool: aiohttp.ClientSession | None = None
        self._batch_cache: dict[str, list[float]] = {}
        self._rate_limiter = AsyncRateLimiter(
            max_calls=config.rate_limit,
            time_window=60
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create connection session with connection pooling."""
        if not self.session_pool:
            connector = aiohttp.TCPConnector(
                limit=50,  # Total connection pool size
                limit_per_host=20,  # Connections per host
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=10,
                sock_read=self.config.timeout
            )
            
            self.session_pool = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_headers()
            )
        
        return self.session_pool
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Optimized batch embedding with caching and rate limiting."""
        # Check cache first
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._batch_cache:
                cached_results[i] = self._batch_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            new_embeddings = await self._embed_batch_optimized(uncached_texts)
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._batch_cache[cache_key] = embedding
        else:
            new_embeddings = []
        
        # Combine cached and new results
        results = [None] * len(texts)
        for i, embedding in cached_results.items():
            results[i] = embedding
        
        for i, embedding in zip(uncached_indices, new_embeddings):
            results[i] = embedding
        
        return results
    
    async def _embed_batch_optimized(self, texts: list[str]) -> list[list[float]]:
        """Optimized batch processing with parallel requests."""
        optimal_batch_size = self.config.max_batch_size or 100
        
        # Process in optimal batches
        tasks = []
        for i in range(0, len(texts), optimal_batch_size):
            batch = texts[i:i + optimal_batch_size]
            task = self._process_batch_with_rate_limit(batch)
            tasks.append(task)
        
        # Execute batches concurrently
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _process_batch_with_rate_limit(self, batch: list[str]) -> list[list[float]]:
        """Process batch with rate limiting and retries."""
        await self._rate_limiter.acquire()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._embed_batch(batch)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise
                
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.random()
                await asyncio.sleep(delay)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session_pool:
            await self.session_pool.close()
            self.session_pool = None
```

#### Memory Optimization
```python
class MemoryOptimizedProvider(EmbeddingProviderBase):
    """Provider optimized for memory efficiency."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._memory_monitor = MemoryMonitor(threshold_mb=400)
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Memory-efficient embedding with streaming."""
        # Use streaming for large datasets
        if len(texts) > 1000:
            return await self._embed_streaming(texts)
        else:
            return await self._embed_batch(texts)
    
    async def _embed_streaming(self, texts: list[str]) -> list[list[float]]:
        """Stream processing for memory efficiency."""
        results = []
        batch_size = self._calculate_optimal_batch_size()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Monitor memory usage
            if self._memory_monitor.should_gc():
                import gc
                gc.collect()
            
            batch_results = await self._embed_batch(batch)
            results.extend(batch_results)
            
            # Optional: write to temporary file for very large datasets
            if self._memory_monitor.is_critical():
                results = await self._offload_to_disk(results)
        
        return results
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        available_memory = self._memory_monitor.get_available_memory_mb()
        
        # Estimate memory per vector (dimension * 4 bytes + overhead)
        estimated_vector_size = (self.dimension * 4 + 100) / 1024 / 1024  # MB
        
        # Use 50% of available memory for batch processing
        safe_memory = available_memory * 0.5
        optimal_batch_size = int(safe_memory / estimated_vector_size)
        
        # Clamp to reasonable bounds
        return max(10, min(optimal_batch_size, self.config.max_batch_size or 100))
```

### Reranking Provider Optimization

```python
class OptimizedRerankProvider(RerankProviderBase):
    """High-performance reranking provider."""
    
    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[RerankResult]:
        """Optimized reranking with early termination."""
        if not documents:
            return []
        
        # Early termination if top_k is small
        if top_k and top_k < len(documents) * 0.1:
            return await self._rerank_with_early_termination(query, documents, top_k)
        
        # Full reranking for larger result sets
        return await self._rerank_full(query, documents, top_k)
    
    async def _rerank_with_early_termination(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int
    ) -> list[RerankResult]:
        """Rerank with early termination optimization."""
        # Process in smaller batches and terminate early
        batch_size = min(100, len(documents))
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self._rerank_batch(query, batch)
            
            # Add batch results with adjusted indices
            for result in batch_results:
                result.index += i
                results.append(result)
            
            # Sort and keep only top results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = results[:top_k * 2]  # Keep extra for final selection
            
            # Early termination if we have enough high-confidence results
            if len(results) >= top_k and results[top_k - 1].relevance_score > 0.8:
                break
        
        # Final sort and limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
```

## 🔧 Backend Optimization

### Vector Backend Optimization

#### Connection Management
```python
class OptimizedVectorBackend:
    """High-performance vector backend with optimizations."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self._connection_pool: ConnectionPool | None = None
        self._query_cache: QueryCache = QueryCache(max_size=1000, ttl=300)
    
    async def _get_connection_pool(self) -> ConnectionPool:
        """Get or create optimized connection pool."""
        if not self._connection_pool:
            self._connection_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                pool_size=20,  # Concurrent connections
                max_retries=3,
                retry_delay=1.0,
                keepalive_timeout=60,
                connection_timeout=10
            )
            await self._connection_pool.initialize()
        
        return self._connection_pool
    
    async def search_vectors(
        self, 
        collection_name: str, 
        query_vector: list[float], 
        limit: int = 10, 
        search_filter: SearchFilter | None = None, 
        score_threshold: float | None = None, 
        **kwargs: Any
    ) -> list[SearchResult]:
        """Optimized vector search with caching."""
        # Generate cache key
        cache_key = self._generate_search_cache_key(
            collection_name, query_vector, limit, search_filter, score_threshold
        )
        
        # Check cache first
        cached_results = self._query_cache.get(cache_key)
        if cached_results:
            return cached_results
        
        # Perform search
        pool = await self._get_connection_pool()
        
        # Optimize search parameters
        search_params = self._optimize_search_params(
            limit, score_threshold, **kwargs
        )
        
        async with pool.get_connection() as conn:
            results = await conn.search(
                collection=collection_name,
                vector=query_vector,
                filter=search_filter,
                **search_params
            )
        
        # Process and cache results
        processed_results = self._process_search_results(results)
        self._query_cache.set(cache_key, processed_results)
        
        return processed_results
    
    async def upsert_vectors(
        self, 
        collection_name: str, 
        vectors: list[VectorPoint]
    ) -> None:
        """Optimized batch upsert with parallel processing."""
        if not vectors:
            return
        
        pool = await self._get_connection_pool()
        
        # Calculate optimal batch size based on vector dimension and data size
        optimal_batch_size = self._calculate_upsert_batch_size(vectors)
        
        # Process in parallel batches
        semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
        
        async def process_batch(batch: list[VectorPoint]):
            async with semaphore:
                async with pool.get_connection() as conn:
                    await conn.upsert(collection_name, batch)
        
        tasks = []
        for i in range(0, len(vectors), optimal_batch_size):
            batch = vectors[i:i + optimal_batch_size]
            task = process_batch(batch)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def _calculate_upsert_batch_size(self, vectors: list[VectorPoint]) -> int:
        """Calculate optimal batch size for upsert operations."""
        if not vectors:
            return 100
        
        # Estimate data size per vector
        sample_vector = vectors[0]
        vector_size = len(sample_vector.vector) * 8  # 8 bytes per float
        payload_size = len(str(sample_vector.payload or {}))
        estimated_size_per_vector = vector_size + payload_size + 100  # overhead
        
        # Target 1MB per batch
        target_batch_size_bytes = 1024 * 1024
        optimal_batch_size = target_batch_size_bytes // estimated_size_per_vector
        
        # Clamp to reasonable bounds
        return max(10, min(optimal_batch_size, 1000))
    
    def _optimize_search_params(
        self, 
        limit: int, 
        score_threshold: float | None, 
        **kwargs: Any
    ) -> dict[str, Any]:
        """Optimize search parameters for better performance."""
        params = kwargs.copy()
        
        # Optimize ef parameter for HNSW index
        if 'ef' not in params:
            # Use higher ef for better recall, lower for speed
            params['ef'] = min(limit * 4, 200)
        
        # Optimize exact search threshold
        if 'exact' not in params and limit < 10:
            params['exact'] = False  # Use approximate search for small results
        
        # Add score threshold for early termination
        if score_threshold:
            params['score_threshold'] = score_threshold
        
        return params
```

#### Index Optimization
```python
class IndexOptimizedBackend(OptimizedVectorBackend):
    """Backend with index optimization strategies."""
    
    async def create_collection(
        self, 
        name: str, 
        dimension: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE, 
        **kwargs: Any
    ) -> None:
        """Create collection with optimized index parameters."""
        # Optimize index parameters based on use case
        index_config = self._optimize_index_config(dimension, **kwargs)
        
        await super().create_collection(
            name, dimension, distance_metric, **index_config
        )
    
    def _optimize_index_config(
        self, 
        dimension: int, 
        **kwargs: Any
    ) -> dict[str, Any]:
        """Optimize index configuration for performance."""
        config = kwargs.copy()
        
        # HNSW optimization
        if 'hnsw_config' not in config:
            config['hnsw_config'] = {
                'M': self._calculate_optimal_m(dimension),
                'ef_construct': self._calculate_optimal_ef_construct(dimension),
                'max_m': 16,
                'max_m0': 32
            }
        
        # Quantization for memory efficiency
        if dimension > 768 and 'quantization_config' not in config:
            config['quantization_config'] = {
                'scalar': {
                    'type': 'int8',
                    'always_ram': True
                }
            }
        
        # On-disk storage for large collections
        if 'on_disk_payload' not in config:
            config['on_disk_payload'] = True
        
        return config
    
    def _calculate_optimal_m(self, dimension: int) -> int:
        """Calculate optimal M parameter for HNSW index."""
        # Higher M for better recall, lower for memory efficiency
        if dimension <= 384:
            return 16
        elif dimension <= 768:
            return 32
        else:
            return 48
    
    def _calculate_optimal_ef_construct(self, dimension: int) -> int:
        """Calculate optimal ef_construct parameter."""
        # Higher ef_construct for better index quality
        base_ef = 200
        return min(base_ef + (dimension // 100) * 50, 800)
```

## 🔍 Service Optimization

### Chunking Service Optimization

```python
class OptimizedChunkingService(BaseServiceProvider, ChunkingService):
    """High-performance chunking service."""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(ServiceType.CHUNKING, config)
        self._parser_cache: dict[str, Any] = {}
        self._chunk_cache: LRUCache = LRUCache(max_size=1000)
    
    async def chunk_content(
        self, 
        content: str, 
        file_path: str | None = None
    ) -> list[str]:
        """Optimized content chunking with caching."""
        # Generate cache key
        cache_key = self._generate_chunk_cache_key(content, file_path)
        
        # Check cache first
        cached_chunks = self._chunk_cache.get(cache_key)
        if cached_chunks:
            return cached_chunks
        
        # Determine optimal chunking strategy
        chunks = await self._chunk_optimized(content, file_path)
        
        # Cache results
        self._chunk_cache.set(cache_key, chunks)
        
        return chunks
    
    async def _chunk_optimized(
        self, 
        content: str, 
        file_path: str | None = None
    ) -> list[str]:
        """Optimized chunking based on content type and size."""
        content_size = len(content)
        
        # Fast path for small content
        if content_size < self.config.min_chunk_size * 2:
            return [content]
        
        # Choose chunking strategy based on content size and type
        if file_path and self._is_code_file(file_path):
            return await self._chunk_code_optimized(content, file_path)
        elif content_size > 100000:  # Large content
            return await self._chunk_streaming(content)
        else:
            return await self._chunk_standard(content)
    
    async def _chunk_code_optimized(
        self, 
        content: str, 
        file_path: str
    ) -> list[str]:
        """Optimized code chunking with AST parsing."""
        language = self._detect_language(file_path)
        
        # Get or create parser
        parser = self._get_cached_parser(language)
        if not parser:
            return await self._chunk_fallback(content)
        
        try:
            # Parse with timeout to prevent hanging
            tree = await asyncio.wait_for(
                parser.parse_async(content.encode()),
                timeout=5.0
            )
            
            return self._extract_semantic_chunks(tree, content)
            
        except asyncio.TimeoutError:
            # Fallback to simple chunking
            return await self._chunk_fallback(content)
    
    def _get_cached_parser(self, language: str):
        """Get cached parser for language."""
        if language not in self._parser_cache:
            try:
                import tree_sitter
                parser = tree_sitter.Parser()
                # Load language grammar (implementation specific)
                self._parser_cache[language] = parser
            except ImportError:
                self._parser_cache[language] = None
        
        return self._parser_cache[language]
    
    async def _chunk_streaming(self, content: str) -> list[str]:
        """Stream-based chunking for large content."""
        chunks = []
        current_chunk = ""
        
        # Process content in lines to maintain semantic boundaries
        for line in content.splitlines(keepends=True):
            if len(current_chunk) + len(line) > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # Line is too long, force split
                    chunks.append(line[:self.config.max_chunk_size])
                    current_chunk = line[self.config.max_chunk_size:]
            else:
                current_chunk += line
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
```

## 📊 Monitoring and Profiling

### Performance Monitoring

```python
class PerformanceMonitor:
    """Performance monitoring for extensions."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)
    
    @contextmanager
    def measure_time(self, operation: str):
        """Context manager to measure operation time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.metrics[f"{operation}_duration"].append(duration)
            self.counters[f"{operation}_count"] += 1
    
    def measure_memory(self, operation: str):
        """Measure memory usage for operation."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics[f"{operation}_memory_mb"].append(memory_mb)
    
    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report."""
        report = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                report[metric_name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
        
        report["counters"] = dict(self.counters)
        return report
    
    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

# Usage in extension
class MonitoredProvider(EmbeddingProviderBase):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.monitor = PerformanceMonitor()
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self.monitor.measure_time("embed_documents"):
            self.monitor.measure_memory("embed_documents")
            return await super().embed_documents(texts)
    
    def get_performance_metrics(self) -> dict[str, Any]:
        return self.monitor.get_performance_report()
```

### Profiling Tools

```python
import cProfile
import pstats
from memory_profiler import profile

class ExtensionProfiler:
    """Profiling utilities for extensions."""
    
    @staticmethod
    def profile_cpu(func):
        """Decorator for CPU profiling."""
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                pr.disable()
                stats = pstats.Stats(pr)
                stats.sort_stats('cumulative')
                stats.print_stats(20)  # Top 20 functions
        
        return wrapper
    
    @staticmethod
    def profile_memory(func):
        """Decorator for memory profiling."""
        return profile(func)
    
    @staticmethod
    async def benchmark_async_function(
        func, 
        *args, 
        iterations: int = 100, 
        **kwargs
    ) -> dict[str, float]:
        """Benchmark async function performance."""
        durations = []
        
        # Warmup
        await func(*args, **kwargs)
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            durations.append(end_time - start_time)
        
        return {
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "p95": sorted(durations)[int(len(durations) * 0.95)],
            "std": statistics.stdev(durations)
        }
```

## 🎯 Best Practices Summary

### General Optimization Principles

1. **Measure First**: Always profile before optimizing
2. **Cache Strategically**: Cache expensive operations with appropriate TTL
3. **Batch Operations**: Process multiple items together when possible
4. **Use Connection Pooling**: Reuse connections for external services
5. **Implement Rate Limiting**: Respect API limits and prevent overload
6. **Monitor Memory**: Track memory usage and implement cleanup
7. **Handle Errors Gracefully**: Implement retries with exponential backoff
8. **Optimize for Common Cases**: Fast paths for typical scenarios

### Provider-Specific Best Practices

- **Batch embedding requests** to maximize API efficiency
- **Implement caching** for frequently requested embeddings
- **Use connection pooling** for HTTP clients
- **Monitor rate limits** and implement backoff strategies
- **Optimize payload sizes** to reduce network overhead

### Backend-Specific Best Practices

- **Optimize index parameters** for your use case
- **Use appropriate batch sizes** for bulk operations
- **Implement query caching** for frequent searches
- **Monitor index build performance** for large collections
- **Use quantization** for memory efficiency with large vectors

### Service-Specific Best Practices

- **Cache parsing results** for repeated content
- **Use streaming** for large datasets
- **Implement timeout handling** for external parsers
- **Monitor resource usage** during processing
- **Provide fast paths** for simple cases

## 🚀 Next Steps

- **[Protocol Reference →](../reference/protocols.md)**: Complete protocol documentation
- **[Examples →](../examples/)**: Working extension examples with performance optimizations
- **[Testing Framework →](./testing.md)**: Performance testing strategies