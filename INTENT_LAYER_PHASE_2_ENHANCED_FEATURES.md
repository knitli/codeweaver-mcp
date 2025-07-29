<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Phase 2: Enhanced Features Implementation Plan

## 🎯 Overview

**Phase Duration**: 3-4 weeks  
**Priority**: SHOULD HAVE for production readiness  
**Prerequisite**: Phase 1 complete with all services operational

This phase enhances the intent layer with advanced natural language processing, semantic caching, performance optimization, and improved error recovery while maintaining full architectural compliance.

## 📋 Phase Summary

### Previous Phase Accomplishments (Phase 1)
✅ **Service-Compliant Infrastructure**: Intent orchestrator and auto-indexing services operational  
✅ **Pattern-Based Parsing**: 85%+ intent recognition accuracy using regex patterns  
✅ **Strategy System**: Three core strategies registered with ExtensibilityManager  
✅ **Workflow Engine**: Multi-step workflows with existing service integration  
✅ **MCP Tools**: `process_intent` and `get_capabilities` tools functional  
✅ **Background Indexing**: Transparent auto-indexing without LLM exposure  

### Phase 2 Enhancements
🚀 **NLP-Enhanced Parsing**: Advanced natural language understanding with spaCy  
🚀 **Semantic Caching**: Vector-based caching using existing backends  
🚀 **Performance Optimization**: Response time improvements and resource optimization  
🚀 **Advanced Error Recovery**: Context preservation and intelligent fallbacks  
🚀 **Enhanced Monitoring**: Detailed metrics and performance tracking  

## 📊 Weekly Breakdown

### Week 1: NLP-Enhanced Parsing System

#### Deliverables

**1. Enhanced Intent Parser** (`src/codeweaver/intent/parsing/nlp_parser.py`)
```python
import spacy
from codeweaver.intent.parsing.pattern_matcher import PatternBasedParser
from codeweaver.types import ParsedIntent, IntentType

class NLPEnhancedParser:
    """NLP-enhanced parser with spaCy integration."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pattern_parser = PatternBasedParser()  # Fallback
        self.nlp = None
        self.domain_models = {}
        self._initialize_nlp()
    
    def _initialize_nlp(self) -> None:
        """Initialize spaCy pipeline with domain-specific models."""
        try:
            # Load base English model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom components for code domain
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                self._add_code_patterns(ruler)
            
            # Load domain-specific models if available
            self._load_domain_models()
            
        except OSError:
            logger.warning("spaCy model not available, falling back to pattern matching")
            self.nlp = None
    
    async def parse(self, intent_text: str) -> ParsedIntent:
        """Parse intent with NLP enhancement and pattern fallback."""
        # Try NLP parsing first
        if self.nlp:
            try:
                nlp_result = await self._nlp_parse(intent_text)
                if nlp_result.confidence > 0.7:  # High confidence threshold
                    return nlp_result
            except Exception as e:
                logger.warning(f"NLP parsing failed, using fallback: {e}")
        
        # Fallback to pattern matching
        pattern_result = await self.pattern_parser.parse(intent_text)
        
        # Enhance pattern result with NLP insights if available
        if self.nlp:
            enhanced_result = await self._enhance_with_nlp(pattern_result, intent_text)
            return enhanced_result
        
        return pattern_result
    
    async def _nlp_parse(self, intent_text: str) -> ParsedIntent:
        """Parse using NLP techniques."""
        doc = self.nlp(intent_text)
        
        # Extract intent type using classification
        intent_type = self._classify_intent_type(doc)
        
        # Extract primary target using NER and dependency parsing
        primary_target = self._extract_primary_target(doc)
        
        # Assess scope and complexity using semantic analysis
        scope = self._assess_scope_semantic(doc)
        complexity = self._assess_complexity_semantic(doc)
        
        # Calculate confidence using multiple factors
        confidence = self._calculate_nlp_confidence(doc, intent_type, primary_target)
        
        return ParsedIntent(
            intent_type=intent_type,
            primary_target=primary_target,
            scope=scope,
            complexity=complexity,
            confidence=confidence,
            filters=self._extract_semantic_filters(doc),
            metadata={
                "parser": "nlp_enhanced",
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "pos_tags": [(token.text, token.pos_) for token in doc],
                "dependencies": [(token.text, token.dep_, token.head.text) for token in doc]
            }
        )
```

**2. Domain-Specific Entity Recognition** (`src/codeweaver/intent/parsing/entities.py`)
```python
class CodeDomainEntityRecognizer:
    """Domain-specific entity recognition for code-related queries."""
    
    def __init__(self):
        self.code_patterns = self._load_code_patterns()
        self.technology_patterns = self._load_technology_patterns()
    
    def _load_code_patterns(self) -> list[dict]:
        """Load patterns for code-related entities."""
        return [
            {"label": "CODE_ELEMENT", "pattern": [{"LOWER": {"IN": ["function", "class", "method", "variable"]}}]},
            {"label": "LANGUAGE", "pattern": [{"LOWER": {"IN": ["python", "javascript", "typescript", "java", "go"]}}]},
            {"label": "FRAMEWORK", "pattern": [{"LOWER": {"IN": ["react", "django", "flask", "express", "spring"]}}]},
            {"label": "DATABASE", "pattern": [{"LOWER": {"IN": ["mysql", "postgresql", "mongodb", "redis"]}}]},
            {"label": "OPERATION", "pattern": [{"LOWER": {"IN": ["authentication", "authorization", "validation", "logging"]}}]},
        ]
    
    def add_patterns_to_ruler(self, ruler) -> None:
        """Add code domain patterns to spaCy ruler."""
        ruler.add_patterns(self.code_patterns)
        ruler.add_patterns(self.technology_patterns)
```

**3. Enhanced Confidence Scoring** (`src/codeweaver/intent/parsing/confidence_scorer.py` enhanced)
```python
class EnhancedConfidenceScorer:
    """Advanced confidence scoring combining multiple factors."""
    
    def __init__(self):
        self.pattern_scorer = BasicConfidenceScorer()
        self.semantic_weights = {
            "entity_match": 0.3,
            "dependency_clarity": 0.2,
            "domain_specificity": 0.2,
            "linguistic_complexity": 0.1,
            "pattern_match": 0.2
        }
    
    async def score_nlp_enhanced(
        self,
        intent_text: str,
        doc: spacy.Doc,
        intent_type: IntentType,
        primary_target: str
    ) -> float:
        """Calculate confidence using NLP features."""
        scores = {}
        
        # Entity matching score
        scores["entity_match"] = self._score_entity_match(doc)
        
        # Dependency parsing clarity
        scores["dependency_clarity"] = self._score_dependency_clarity(doc)
        
        # Domain specificity
        scores["domain_specificity"] = self._score_domain_specificity(doc)
        
        # Linguistic complexity
        scores["linguistic_complexity"] = self._score_linguistic_complexity(doc)
        
        # Pattern matching fallback
        scores["pattern_match"] = await self.pattern_scorer.score(
            intent_text, intent_type, primary_target
        )
        
        # Weighted combination
        final_score = sum(
            score * self.semantic_weights[key] 
            for key, score in scores.items()
        )
        
        return min(1.0, max(0.0, final_score))
```

#### Success Criteria - Week 1
- [ ] NLP parser achieves >92% intent recognition accuracy
- [ ] Semantic entity extraction identifies domain-specific terms
- [ ] Enhanced confidence scoring provides better accuracy assessment
- [ ] Graceful fallback to pattern matching when NLP unavailable

### Week 2: Semantic Caching System

#### Deliverables

**1. Vector-Based Cache** (`src/codeweaver/intent/caching/semantic_cache.py`)
```python
from codeweaver.types import CacheService, VectorBackend
from codeweaver.intent.caching.intent_cache import IntentCacheManager

class SemanticIntentCache(IntentCacheManager):
    """Semantic caching using vector similarity for intent results."""
    
    def __init__(
        self,
        cache_service: CacheService | None,
        vector_backend: VectorBackend | None,
        embedding_provider: EmbeddingProvider | None
    ):
        super().__init__(cache_service)
        self.vector_backend = vector_backend
        self.embedding_provider = embedding_provider
        self.similarity_threshold = 0.85
        self.semantic_cache_enabled = (
            vector_backend is not None and embedding_provider is not None
        )
    
    async def get_cached_result(self, intent_text: str) -> IntentResult | None:
        """Get cached result using semantic similarity."""
        # Try exact match first (fast path)
        exact_result = await super().get_cached_result(intent_text)
        if exact_result:
            return exact_result
        
        # Try semantic similarity if enabled
        if self.semantic_cache_enabled:
            return await self._get_semantic_cached_result(intent_text)
        
        return None
    
    async def _get_semantic_cached_result(self, intent_text: str) -> IntentResult | None:
        """Find semantically similar cached results."""
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_provider.embed_documents(
                [intent_text]
            )
            
            # Search for similar intents
            search_results = await self.vector_backend.search(
                query_embedding[0],
                limit=5,
                metadata_filter={"type": "intent_cache"}
            )
            
            # Check if any results meet similarity threshold
            for result in search_results:
                if result.score >= self.similarity_threshold:
                    # Retrieve cached result
                    cache_key = result.metadata.get("cache_key")
                    if cache_key:
                        cached_result = await self.cache_service.get(cache_key)
                        if cached_result:
                            # Update metadata to indicate semantic match
                            cached_result.metadata["semantic_cache_hit"] = True
                            cached_result.metadata["similarity_score"] = result.score
                            return cached_result
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")
            return None
    
    async def cache_result(
        self,
        intent_text: str,
        result: IntentResult,
        ttl: int = 3600
    ) -> None:
        """Cache result with semantic indexing."""
        # Standard caching
        await super().cache_result(intent_text, result, ttl)
        
        # Semantic indexing if enabled
        if self.semantic_cache_enabled and result.success:
            await self._index_semantic_cache(intent_text, result)
    
    async def _index_semantic_cache(
        self,
        intent_text: str,
        result: IntentResult
    ) -> None:
        """Index intent for semantic similarity search."""
        try:
            # Generate embedding
            embedding = await self.embedding_provider.embed_documents([intent_text])
            
            # Store in vector backend
            cache_key = self._generate_cache_key(intent_text)
            await self.vector_backend.upsert_points([{
                "id": f"intent_cache_{hash(intent_text)}",
                "vector": embedding[0],
                "metadata": {
                    "type": "intent_cache",
                    "cache_key": cache_key,
                    "intent_text": intent_text,
                    "intent_type": result.metadata.get("intent_type"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }])
            
        except Exception as e:
            logger.warning(f"Semantic cache indexing failed: {e}")
```

**2. Cache Performance Monitoring** (`src/codeweaver/intent/caching/cache_metrics.py`)
```python
from dataclasses import dataclass

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_requests: int
    cache_hits: int
    semantic_hits: int
    cache_misses: int
    average_response_time: float
    semantic_similarity_scores: list[float]

class CacheMetricsCollector:
    """Collect and analyze cache performance metrics."""
    
    def __init__(self):
        self.metrics = CacheMetrics(0, 0, 0, 0, 0.0, [])
        self.response_times = []
    
    def record_cache_hit(self, response_time: float, semantic: bool = False) -> None:
        """Record cache hit with timing."""
        self.metrics.total_requests += 1
        self.metrics.cache_hits += 1
        if semantic:
            self.metrics.semantic_hits += 1
        self.response_times.append(response_time)
        self._update_average_response_time()
    
    def record_cache_miss(self, response_time: float) -> None:
        """Record cache miss with timing."""
        self.metrics.total_requests += 1
        self.metrics.cache_misses += 1
        self.response_times.append(response_time)
        self._update_average_response_time()
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.metrics.total_requests == 0:
            return 0.0
        return self.metrics.cache_hits / self.metrics.total_requests
    
    def get_semantic_hit_rate(self) -> float:
        """Calculate semantic cache hit rate."""
        if self.metrics.cache_hits == 0:
            return 0.0
        return self.metrics.semantic_hits / self.metrics.cache_hits
```

#### Success Criteria - Week 2
- [ ] Semantic cache achieves >85% hit rate for similar queries
- [ ] Vector similarity search response time <200ms
- [ ] Cache performance monitoring provides detailed metrics
- [ ] Integration with existing vector backends successful

### Week 3: Performance Optimization

#### Deliverables

**1. Response Time Optimization** (`src/codeweaver/intent/optimization/performance.py`)
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class IntentPerformanceOptimizer:
    """Performance optimization for intent processing."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("max_workers", 4)
        )
        self.strategy_cache = {}
        self.parser_cache = {}
    
    async def optimize_parsing(self, parser, intent_text: str) -> ParsedIntent:
        """Optimize parsing with caching and parallelization."""
        # Check parser cache
        cache_key = f"parse_{hash(intent_text)}"
        if cache_key in self.parser_cache:
            return self.parser_cache[cache_key]
        
        # Parse with timeout
        try:
            result = await asyncio.wait_for(
                parser.parse(intent_text),
                timeout=self.config.get("parse_timeout", 5.0)
            )
            
            # Cache successful parse
            self.parser_cache[cache_key] = result
            return result
            
        except asyncio.TimeoutError:
            # Fallback to fast pattern matching
            logger.warning(f"Parsing timeout for: {intent_text}")
            return await self._fast_parse_fallback(intent_text)
    
    async def optimize_strategy_execution(
        self,
        strategy,
        parsed_intent: ParsedIntent,
        context: dict[str, Any]
    ) -> IntentResult:
        """Optimize strategy execution with parallel processing."""
        
        # Check strategy cache
        strategy_key = self._generate_strategy_cache_key(parsed_intent)
        if strategy_key in self.strategy_cache:
            cached_result = self.strategy_cache[strategy_key]
            if self._is_cache_valid(cached_result):
                return cached_result
        
        # Execute with optimization
        if hasattr(strategy, 'supports_parallel') and strategy.supports_parallel:
            result = await self._execute_parallel_strategy(strategy, parsed_intent, context)
        else:
            result = await strategy.execute(parsed_intent, context)
        
        # Cache successful result
        if result.success:
            self.strategy_cache[strategy_key] = result
        
        return result
    
    async def _execute_parallel_strategy(
        self,
        strategy,
        parsed_intent: ParsedIntent,
        context: dict[str, Any]
    ) -> IntentResult:
        """Execute strategy with parallel optimization."""
        
        # Break down complex operations into parallel tasks
        if parsed_intent.complexity == Complexity.COMPLEX:
            # Execute search and analysis in parallel
            search_task = asyncio.create_task(
                strategy._execute_search_component(parsed_intent, context)
            )
            analysis_task = asyncio.create_task(
                strategy._execute_analysis_component(parsed_intent, context)
            )
            
            # Wait for both with timeout
            search_result, analysis_result = await asyncio.gather(
                search_task, analysis_task, return_exceptions=True
            )
            
            # Combine results
            return strategy._combine_parallel_results(search_result, analysis_result)
        else:
            # Standard execution
            return await strategy.execute(parsed_intent, context)
```

**2. Memory Usage Optimization** (`src/codeweaver/intent/optimization/memory.py`)
```python
import gc
import weakref
from typing import WeakValueDictionary

class MemoryOptimizer:
    """Memory usage optimization for intent processing."""
    
    def __init__(self):
        self.weak_cache: WeakValueDictionary = weakref.WeakValueDictionary()
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
        self.cleanup_interval = 300  # 5 minutes
        self._start_cleanup_task()
    
    def optimize_result_storage(self, result: IntentResult) -> IntentResult:
        """Optimize result storage to reduce memory usage."""
        # Compress large data payloads
        if hasattr(result.data, '__len__') and len(str(result.data)) > 10000:
            result.data = self._compress_large_data(result.data)
        
        # Store in weak cache for deduplication
        result_key = self._generate_result_key(result)
        existing_result = self.weak_cache.get(result_key)
        if existing_result:
            return existing_result
        
        self.weak_cache[result_key] = result
        return result
    
    async def periodic_cleanup(self) -> None:
        """Periodic memory cleanup."""
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        memory_usage = self._get_memory_usage()
        if memory_usage > self.memory_threshold:
            await self._aggressive_cleanup()
    
    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self.periodic_cleanup()
```

**3. Resource Monitoring** (`src/codeweaver/intent/monitoring/resource_monitor.py`)
```python
import psutil
from dataclasses import dataclass

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    response_time_p95: float
    active_requests: int

class ResourceMonitor:
    """Monitor resource usage for intent processing."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_p95": 3000.0  # 3 seconds
        }
    
    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        process = psutil.Process()
        
        return ResourceMetrics(
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            response_time_p95=self._calculate_p95_response_time(),
            active_requests=self._get_active_requests()
        )
    
    def check_alerts(self, metrics: ResourceMetrics) -> list[str]:
        """Check for resource usage alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time_p95 > self.alert_thresholds["response_time_p95"]:
            alerts.append(f"High response time: {metrics.response_time_p95:.0f}ms")
        
        return alerts
```

#### Success Criteria - Week 3
- [ ] Response time P95 improved to <3s for complex queries
- [ ] Memory usage optimized with <50MB baseline usage
- [ ] Resource monitoring provides real-time alerts
- [ ] Parallel processing improves throughput by >30%

### Week 4: Advanced Error Recovery

#### Deliverables

**1. Context-Preserving Error Recovery** (`src/codeweaver/intent/recovery/context_recovery.py`)
```python
from codeweaver.intent.recovery.fallback_handler import IntentErrorHandler

class ContextPreservingErrorHandler(IntentErrorHandler):
    """Advanced error handler with context preservation."""
    
    def __init__(self, services_manager: ServicesManager):
        super().__init__(services_manager)
        self.context_store = {}
        self.recovery_strategies = self._build_recovery_strategies()
    
    async def handle_error_with_context(
        self,
        error: Exception,
        context: dict[str, Any],
        parsed_intent: ParsedIntent,
        execution_history: list[dict[str, Any]]
    ) -> IntentResult:
        """Handle error while preserving execution context."""
        
        # Store context for recovery
        context_id = self._store_context(context, parsed_intent, execution_history)
        
        # Analyze error for recovery strategy
        recovery_strategy = self._select_recovery_strategy(error, execution_history)
        
        try:
            # Attempt recovery with preserved context
            result = await recovery_strategy.recover(
                error, context, parsed_intent, execution_history
            )
            
            # Enhance result with recovery information
            result.metadata["recovery_applied"] = recovery_strategy.name
            result.metadata["context_preserved"] = True
            result.metadata["context_id"] = context_id
            
            return result
            
        except Exception as recovery_error:
            # Final fallback with context information
            return await self._final_fallback_with_context(
                recovery_error, context_id, parsed_intent
            )
    
    def _build_recovery_strategies(self) -> dict[str, Any]:
        """Build context-aware recovery strategies."""
        return {
            "partial_execution": PartialExecutionRecovery(),
            "service_degradation": ServiceDegradationRecovery(),
            "alternative_strategy": AlternativeStrategyRecovery(),
            "context_reconstruction": ContextReconstructionRecovery()
        }
```

**2. Intelligent Fallback System** (`src/codeweaver/intent/recovery/intelligent_fallback.py`)
```python
class IntelligentFallbackSystem:
    """Intelligent fallback with learning capabilities."""
    
    def __init__(self):
        self.fallback_success_rates = {}
        self.context_patterns = {}
        self.learning_enabled = True
    
    async def execute_fallback(
        self,
        parsed_intent: ParsedIntent,
        context: dict[str, Any],
        original_error: Exception
    ) -> IntentResult:
        """Execute intelligent fallback with learning."""
        
        # Analyze context for patterns
        context_pattern = self._analyze_context_pattern(context)
        
        # Select best fallback based on historical success
        fallback_strategy = self._select_optimal_fallback(
            parsed_intent, context_pattern, original_error
        )
        
        try:
            # Execute fallback
            result = await fallback_strategy.execute(parsed_intent, context)
            
            # Learn from success
            if self.learning_enabled and result.success:
                self._record_fallback_success(fallback_strategy, context_pattern)
            
            return result
            
        except Exception as fallback_error:
            # Learn from failure
            if self.learning_enabled:
                self._record_fallback_failure(fallback_strategy, context_pattern)
            
            # Try next best fallback
            return await self._try_next_fallback(
                parsed_intent, context, original_error, fallback_strategy
            )
    
    def _record_fallback_success(
        self,
        strategy: Any,
        context_pattern: str
    ) -> None:
        """Record successful fallback for learning."""
        key = f"{strategy.name}_{context_pattern}"
        
        if key not in self.fallback_success_rates:
            self.fallback_success_rates[key] = {"successes": 0, "attempts": 0}
        
        self.fallback_success_rates[key]["successes"] += 1
        self.fallback_success_rates[key]["attempts"] += 1
```

**3. Health-Based Recovery** (`src/codeweaver/intent/recovery/health_recovery.py`)
```python
from codeweaver.types import ServiceHealth, HealthStatus

class HealthBasedRecovery:
    """Recovery strategies based on service health status."""
    
    def __init__(self, services_manager: ServicesManager):
        self.services_manager = services_manager
    
    async def recover_based_on_health(
        self,
        parsed_intent: ParsedIntent,
        context: dict[str, Any]
    ) -> IntentResult:
        """Recover based on current service health."""
        
        # Get comprehensive health report
        health_report = await self.services_manager.get_health_report()
        
        # Determine available services
        healthy_services = [
            service_type for service_type, health in health_report.services.items()
            if health.status == HealthStatus.HEALTHY
        ]
        
        degraded_services = [
            service_type for service_type, health in health_report.services.items()
            if health.status == HealthStatus.DEGRADED
        ]
        
        # Adapt strategy based on available services
        if ServiceType.CHUNKING in healthy_services and ServiceType.FILTERING in healthy_services:
            # Full functionality available
            return await self._execute_full_strategy(parsed_intent, context)
        
        elif ServiceType.CHUNKING in degraded_services or ServiceType.FILTERING in degraded_services:
            # Limited functionality
            return await self._execute_limited_strategy(parsed_intent, context)
        
        else:
            # Minimal functionality
            return await self._execute_minimal_strategy(parsed_intent, context)
    
    async def _execute_minimal_strategy(
        self,
        parsed_intent: ParsedIntent,
        context: dict[str, Any]
    ) -> IntentResult:
        """Execute with minimal service dependencies."""
        # Direct tool routing without advanced processing
        if parsed_intent.intent_type == IntentType.SEARCH:
            # Route directly to search tool with basic parameters
            from codeweaver.server import search_code_handler
            
            result = await search_code_handler(
                query=parsed_intent.primary_target,
                context=context
            )
            
            return IntentResult(
                success=True,
                data=result,
                metadata={
                    "strategy": "minimal_fallback",
                    "services_available": "none",
                    "degraded_mode": True
                }
            )
        
        # Provide helpful message for other intent types
        return IntentResult(
            success=False,
            data=None,
            error_message="Services temporarily unavailable",
            suggestions=[
                f"Try a simpler search: 'find {parsed_intent.primary_target}'",
                "Check service health and try again later"
            ]
        )
```

#### Success Criteria - Week 4
- [ ] Context preservation maintains 90%+ of execution state during recovery
- [ ] Intelligent fallback learning improves success rates over time
- [ ] Health-based recovery adapts to service availability
- [ ] Error recovery provides meaningful suggestions to users

## 📊 Success Metrics - Phase 2

### Enhanced Performance Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Intent Recognition Accuracy** | >92% | NLP-enhanced parsing accuracy |
| **Response Time P95** | <3s | Optimized execution time |
| **Cache Hit Rate** | >85% | Semantic cache effectiveness |
| **Semantic Cache Hit Rate** | >60% | Vector similarity matching |
| **Memory Usage** | <150MB | Optimized memory footprint |
| **Service Recovery Rate** | >90% | Error recovery through health-based strategies |

### Advanced Features Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| **NLP Parser Availability** | >95% | spaCy model availability and performance |
| **Parallel Processing Improvement** | >30% | Throughput increase for complex queries |
| **Context Preservation Rate** | >90% | Successful context recovery during errors |
| **Fallback Learning Accuracy** | >80% | Intelligent fallback selection improvement |

## 🚀 Phase 2 Completion Criteria

✅ **NLP Enhancement**: >92% intent recognition with semantic understanding  
✅ **Semantic Caching**: >85% cache hit rate with vector similarity  
✅ **Performance Optimization**: <3s response time with <150MB memory usage  
✅ **Advanced Error Recovery**: >90% recovery rate with context preservation  
✅ **Resource Monitoring**: Real-time alerts and performance tracking  
✅ **Intelligent Fallbacks**: Learning-based fallback selection with >80% accuracy  

**Ready for Phase 3**: Advanced features including user learning and debugging tools

---

*This phase significantly enhances the intent layer's intelligence and performance while maintaining full architectural compliance, delivering a production-ready natural language interface for CodeWeaver.*