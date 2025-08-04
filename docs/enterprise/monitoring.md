# Production Monitoring and Observability

This guide covers comprehensive monitoring, observability, and alerting strategies for CodeWeaver in enterprise environments. It includes metrics collection, logging, alerting, and business intelligence for production deployments.

## Observability Architecture Overview

CodeWeaver implements a comprehensive observability stack with multiple layers of monitoring:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Observability Stack                           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Metrics   │  │   Logging   │  │   Tracing   │            │
│  │             │  │             │  │             │            │
│  │• Prometheus │  │• ELK Stack  │  │• Jaeger     │            │
│  │• Grafana    │  │• Fluentd    │  │• OpenTel    │            │
│  │• AlertMgr   │  │• Kibana     │  │• Zipkin     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Application Layer                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │CodeWvr-1│  │CodeWvr-2│  │CodeWvr-3│  │Vector DB│    │   │
│  │  │ +Metrics│  │ +Metrics│  │ +Metrics│  │ +Metrics│    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Infrastructure Layer                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │K8s Node │  │Load Bal │  │Database │  │Network  │    │   │
│  │  │Exporter │  │ Metrics │  │ Metrics │  │ Metrics │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics Collection and Analysis

### Prometheus Configuration

#### Core Prometheus Setup
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'codeweaver-prod'
    region: 'us-east-1'

rule_files:
  - "codeweaver-rules.yml"
  - "infrastructure-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # CodeWeaver application metrics
  - job_name: 'codeweaver'
    static_configs:
      - targets: ['codeweaver-1:9090', 'codeweaver-2:9090', 'codeweaver-3:9090']
    scrape_interval: 10s
    metrics_path: /metrics
    params:
      format: ['prometheus']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_pod_label_tenant]
        target_label: tenant_id

  # Vector database metrics
  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant-1:6333', 'qdrant-2:6333', 'qdrant-3:6333']
    scrape_interval: 30s
    metrics_path: /metrics

  # Kubernetes cluster metrics
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - source_labels: [__address__]
        regex: '(.*):10250'
        replacement: '${1}:9100'
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  # Kubernetes pods
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  # NGINX metrics
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Node exporter for system metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

# Storage configuration
storage:
  tsdb:
    path: /prometheus/data
    retention.time: 30d
    retention.size: 100GB
    wal-compression: true
```

#### CodeWeaver-Specific Alert Rules
```yaml
# codeweaver-rules.yml
groups:
- name: codeweaver.application
  rules:
  # High error rate
  - alert: CodeWeaverHighErrorRate
    expr: rate(codeweaver_requests_total{status=~"5.."}[5m]) / rate(codeweaver_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
      component: application
    annotations:
      summary: "CodeWeaver high error rate"
      description: "CodeWeaver error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

  # High response time
  - alert: CodeWeaverHighLatency
    expr: histogram_quantile(0.95, rate(codeweaver_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
      component: application
    annotations:
      summary: "CodeWeaver high latency"
      description: "95th percentile latency is {{ $value }}s for {{ $labels.instance }}"

  # Search performance degradation
  - alert: CodeWeaverSearchLatency
    expr: histogram_quantile(0.95, rate(codeweaver_search_latency_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
      component: search
    annotations:
      summary: "CodeWeaver search latency high"
      description: "Search latency 95th percentile is {{ $value }}s for tenant {{ $labels.tenant_id }}"

  # Vector database connection issues
  - alert: CodeWeaverVectorDBDown
    expr: codeweaver_vector_operations_total{status="error"} > 0
    for: 1m
    labels:
      severity: critical
      component: database
    annotations:
      summary: "CodeWeaver vector database issues"
      description: "Vector database operations failing for {{ $labels.instance }}"

  # Cache hit rate degradation
  - alert: CodeWeaverLowCacheHitRate
    expr: rate(codeweaver_cache_hits_total[5m]) / (rate(codeweaver_cache_hits_total[5m]) + rate(codeweaver_cache_misses_total[5m])) < 0.7
    for: 10m
    labels:
      severity: warning
      component: cache
    annotations:
      summary: "CodeWeaver low cache hit rate"
      description: "Cache hit rate is {{ $value | humanizePercentage }} for {{ $labels.cache_type }}"

  # Memory usage
  - alert: CodeWeaverHighMemoryUsage
    expr: codeweaver_memory_usage_bytes / (1024*1024*1024) > 6
    for: 5m
    labels:
      severity: warning
      component: resource
    annotations:
      summary: "CodeWeaver high memory usage"
      description: "Memory usage is {{ $value }}GB for {{ $labels.component }}"

  # Active connections
  - alert: CodeWeaverHighConnectionCount
    expr: codeweaver_active_connections > 1000
    for: 5m
    labels:
      severity: warning
      component: network
    annotations:
      summary: "CodeWeaver high connection count"
      description: "Active connections is {{ $value }} for tenant {{ $labels.tenant_id }}"

- name: codeweaver.business
  rules:
  # Search volume anomaly
  - alert: CodeWeaverSearchVolumeAnomaly
    expr: rate(codeweaver_requests_total{endpoint=~".*search.*"}[5m]) > 100
    for: 5m
    labels:
      severity: info
      component: business
    annotations:
      summary: "Unusual search volume detected"
      description: "Search rate is {{ $value }} requests/second"

  # Tenant usage spike
  - alert: CodeWeaverTenantUsageSpike
    expr: rate(codeweaver_requests_total[5m]) > 50
    for: 5m
    labels:
      severity: info
      component: business
    annotations:
      summary: "High usage for tenant"
      description: "Tenant {{ $labels.tenant_id }} request rate is {{ $value }} requests/second"
```

### Application Metrics Implementation

#### Comprehensive Metrics Collection
```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, REGISTRY
from prometheus_client import start_http_server, generate_latest
import time
import psutil
import asyncio
from typing import Dict, Any, Optional
from functools import wraps

class CodeWeaverMetrics:
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # Request metrics
        self.request_count = Counter(
            'codeweaver_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status', 'tenant_id'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'codeweaver_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'tenant_id'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Search metrics
        self.search_latency = Histogram(
            'codeweaver_search_latency_seconds',
            'Search operation latency',
            ['tenant_id', 'query_type', 'result_count_bucket'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.search_results = Histogram(
            'codeweaver_search_results_count',
            'Number of search results returned',
            ['tenant_id', 'query_type'],
            buckets=[0, 1, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry
        )
        
        self.search_cache_hit_rate = Gauge(
            'codeweaver_search_cache_hit_rate',
            'Search cache hit rate',
            ['tenant_id'],
            registry=self.registry
        )
        
        # Vector database metrics
        self.vector_operations = Counter(
            'codeweaver_vector_operations_total',
            'Vector database operations',
            ['operation', 'tenant_id', 'status', 'collection'],
            registry=self.registry
        )
        
        self.vector_operation_duration = Histogram(
            'codeweaver_vector_operation_duration_seconds',
            'Vector operation duration',
            ['operation', 'tenant_id'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.vector_collection_size = Gauge(
            'codeweaver_vector_collection_size',
            'Number of vectors in collection',
            ['tenant_id', 'collection'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'codeweaver_cache_operations_total',
            'Cache operations',
            ['operation', 'cache_type', 'tenant_id'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'codeweaver_cache_hit_rate',
            'Cache hit rate by type',
            ['cache_type', 'tenant_id'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'codeweaver_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type', 'tenant_id'],
            registry=self.registry
        )
        
        # System resource metrics
        self.memory_usage = Gauge(
            'codeweaver_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'codeweaver_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'codeweaver_active_connections',
            'Number of active connections',
            ['tenant_id'],
            registry=self.registry
        )
        
        # Business metrics
        self.tenant_usage = Counter(
            'codeweaver_tenant_usage_total',
            'Tenant usage statistics',
            ['tenant_id', 'operation_type'],
            registry=self.registry
        )
        
        self.codebase_stats = Gauge(
            'codeweaver_codebase_statistics',
            'Codebase statistics',
            ['tenant_id', 'metric_type'],
            registry=self.registry
        )
        
        # Error tracking
        self.error_count = Counter(
            'codeweaver_errors_total',
            'Total number of errors',
            ['error_type', 'component', 'tenant_id'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'codeweaver_service_info',
            'Service information',
            registry=self.registry
        )
        
        # Health metrics
        self.health_check = Gauge(
            'codeweaver_health_check',
            'Health check status (1=healthy, 0=unhealthy)',
            ['component', 'check_type'],
            registry=self.registry
        )
        
        # Start system metrics collection
        self.start_system_metrics_collection()
    
    def track_request(self, method: str, endpoint: str, tenant_id: str = "default"):
        """Decorator to track request metrics."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.error_count.labels(
                        error_type=type(e).__name__,
                        component="api",
                        tenant_id=tenant_id
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_count.labels(
                        method=method,
                        endpoint=endpoint,
                        status=status,
                        tenant_id=tenant_id
                    ).inc()
                    
                    self.request_duration.labels(
                        method=method,
                        endpoint=endpoint,
                        tenant_id=tenant_id
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def track_search_operation(self, tenant_id: str, query_type: str = "semantic"):
        """Track search operation metrics."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    results = await func(*args, **kwargs)
                    
                    # Calculate metrics
                    duration = time.time() - start_time
                    result_count = len(results) if results else 0
                    
                    # Bucket result count for latency analysis
                    if result_count == 0:
                        result_bucket = "empty"
                    elif result_count <= 10:
                        result_bucket = "small"
                    elif result_count <= 50:
                        result_bucket = "medium"
                    else:
                        result_bucket = "large"
                    
                    # Record metrics
                    self.search_latency.labels(
                        tenant_id=tenant_id,
                        query_type=query_type,
                        result_count_bucket=result_bucket
                    ).observe(duration)
                    
                    self.search_results.labels(
                        tenant_id=tenant_id,
                        query_type=query_type
                    ).observe(result_count)
                    
                    # Track tenant usage
                    self.tenant_usage.labels(
                        tenant_id=tenant_id,
                        operation_type="search"
                    ).inc()
                    
                    return results
                    
                except Exception as e:
                    self.error_count.labels(
                        error_type=type(e).__name__,
                        component="search",
                        tenant_id=tenant_id
                    ).inc()
                    raise
            
            return wrapper
        return decorator
    
    def track_vector_operation(self, operation: str, tenant_id: str, collection: str):
        """Track vector database operations."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    
                    self.vector_operations.labels(
                        operation=operation,
                        tenant_id=tenant_id,
                        status=status,
                        collection=collection
                    ).inc()
                    
                    self.vector_operation_duration.labels(
                        operation=operation,
                        tenant_id=tenant_id
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def update_cache_metrics(self, cache_type: str, tenant_id: str, 
                           hits: int, misses: int, size_bytes: int):
        """Update cache metrics."""
        total_requests = hits + misses
        hit_rate = hits / total_requests if total_requests > 0 else 0
        
        self.cache_hit_rate.labels(
            cache_type=cache_type,
            tenant_id=tenant_id
        ).set(hit_rate)
        
        self.cache_size.labels(
            cache_type=cache_type,
            tenant_id=tenant_id
        ).set(size_bytes)
    
    def update_collection_size(self, tenant_id: str, collection: str, size: int):
        """Update vector collection size."""
        self.vector_collection_size.labels(
            tenant_id=tenant_id,
            collection=collection
        ).set(size)
    
    def update_health_check(self, component: str, check_type: str, healthy: bool):
        """Update health check status."""
        self.health_check.labels(
            component=component,
            check_type=check_type
        ).set(1 if healthy else 0)
    
    def start_system_metrics_collection(self):
        """Start collecting system metrics."""
        async def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.labels(component="system").set(memory.used)
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.labels(component="system").set(cpu_percent)
                    
                    await asyncio.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(collect_system_metrics())
    
    def set_service_info(self, version: str, commit: str, build_time: str):
        """Set service information."""
        self.service_info.info({
            'version': version,
            'commit': commit,
            'build_time': build_time,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        })

# Global metrics instance
metrics = CodeWeaverMetrics()
```

### Business Intelligence Metrics

#### Custom Business Metrics Dashboard
```python
# business_metrics.py
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

@dataclass
class TenantMetrics:
    tenant_id: str
    active_users: int
    search_volume: int
    codebase_count: int
    storage_usage_gb: float
    monthly_cost: float
    
@dataclass
class UsageAnalytics:
    total_searches: int
    unique_users: int
    avg_response_time: float
    cache_hit_rate: float
    error_rate: float
    popular_queries: List[str]

class BusinessMetricsCollector:
    def __init__(self, prometheus_client, database_client):
        self.prometheus = prometheus_client
        self.database = database_client
    
    async def get_tenant_metrics(self, tenant_id: str, 
                               period_days: int = 30) -> TenantMetrics:
        """Get comprehensive metrics for a tenant."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=period_days)
        
        # Query Prometheus for usage metrics
        search_volume = await self.get_search_volume(tenant_id, start_time, end_time)
        active_users = await self.get_active_users(tenant_id, start_time, end_time)
        
        # Query database for tenant data
        codebase_count = await self.get_codebase_count(tenant_id)
        storage_usage = await self.get_storage_usage(tenant_id)
        
        # Calculate costs
        monthly_cost = await self.calculate_tenant_cost(tenant_id, period_days)
        
        return TenantMetrics(
            tenant_id=tenant_id,
            active_users=active_users,
            search_volume=search_volume,
            codebase_count=codebase_count,
            storage_usage_gb=storage_usage,
            monthly_cost=monthly_cost
        )
    
    async def get_usage_analytics(self, tenant_id: Optional[str] = None,
                                 period_days: int = 30) -> UsageAnalytics:
        """Get usage analytics for tenant or system-wide."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=period_days)
        
        # Build Prometheus queries
        tenant_filter = f'{{tenant_id="{tenant_id}"}}' if tenant_id else ''
        
        queries = {
            'total_searches': f'sum(increase(codeweaver_requests_total{{endpoint=~".*search.*"}}{tenant_filter}[{period_days}d]))',
            'unique_users': f'count by (user_id) (codeweaver_requests_total{tenant_filter})',
            'avg_response_time': f'avg(rate(codeweaver_request_duration_seconds_sum{tenant_filter}[1h]) / rate(codeweaver_request_duration_seconds_count{tenant_filter}[1h]))',
            'cache_hit_rate': f'avg(codeweaver_cache_hit_rate{tenant_filter})',
            'error_rate': f'rate(codeweaver_requests_total{{status=~"5.."}}{tenant_filter}[1h]) / rate(codeweaver_requests_total{tenant_filter}[1h])'
        }
        
        # Execute queries
        results = {}
        for metric, query in queries.items():
            result = await self.prometheus.query(query)
            results[metric] = self.extract_metric_value(result)
        
        # Get popular queries from logs
        popular_queries = await self.get_popular_queries(tenant_id, start_time, end_time)
        
        return UsageAnalytics(
            total_searches=int(results.get('total_searches', 0)),
            unique_users=int(results.get('unique_users', 0)),
            avg_response_time=float(results.get('avg_response_time', 0)),
            cache_hit_rate=float(results.get('cache_hit_rate', 0)),
            error_rate=float(results.get('error_rate', 0)),
            popular_queries=popular_queries
        )
    
    async def generate_usage_report(self, tenant_id: str) -> Dict[str, Any]:
        """Generate comprehensive usage report for tenant."""
        tenant_metrics = await self.get_tenant_metrics(tenant_id)
        usage_analytics = await self.get_usage_analytics(tenant_id)
        
        # Performance metrics
        performance_data = await self.get_performance_metrics(tenant_id)
        
        # Growth trends
        growth_trends = await self.calculate_growth_trends(tenant_id)
        
        return {
            'tenant_id': tenant_id,
            'report_date': datetime.utcnow().isoformat(),
            'metrics': {
                'usage': {
                    'active_users': tenant_metrics.active_users,
                    'search_volume': tenant_metrics.search_volume,
                    'codebase_count': tenant_metrics.codebase_count,
                    'storage_usage_gb': tenant_metrics.storage_usage_gb
                },
                'performance': {
                    'avg_response_time': usage_analytics.avg_response_time,
                    'cache_hit_rate': usage_analytics.cache_hit_rate,
                    'error_rate': usage_analytics.error_rate,
                    'availability': performance_data.get('availability', 0.999)
                },
                'costs': {
                    'monthly_cost': tenant_metrics.monthly_cost,
                    'cost_per_search': tenant_metrics.monthly_cost / max(tenant_metrics.search_volume, 1),
                    'cost_per_user': tenant_metrics.monthly_cost / max(tenant_metrics.active_users, 1)
                }
            },
            'trends': growth_trends,
            'recommendations': await self.generate_recommendations(tenant_metrics, usage_analytics)
        }
    
    async def generate_recommendations(self, metrics: TenantMetrics, 
                                     analytics: UsageAnalytics) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if analytics.avg_response_time > 1.0:
            recommendations.append("Consider enabling more aggressive caching to improve response times")
        
        if analytics.cache_hit_rate < 0.7:
            recommendations.append("Cache hit rate is low - review cache warming strategies")
        
        if analytics.error_rate > 0.01:
            recommendations.append("Error rate is elevated - investigate error patterns")
        
        # Cost optimization
        cost_per_search = metrics.monthly_cost / max(metrics.search_volume, 1)
        if cost_per_search > 0.10:  # $0.10 per search
            recommendations.append("High cost per search - consider resource optimization")
        
        # Usage optimization
        if metrics.search_volume / max(metrics.active_users, 1) < 10:
            recommendations.append("Low search volume per user - consider user training or UX improvements")
        
        return recommendations
```

## Logging and Log Analysis

### Centralized Logging Architecture

#### ELK Stack Configuration
```yaml
# elasticsearch.yml
cluster.name: codeweaver-logs
node.name: codeweaver-es-01
network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

# Cluster settings
discovery.seed_hosts: ["es-01", "es-02", "es-03"]
cluster.initial_master_nodes: ["codeweaver-es-01", "codeweaver-es-02", "codeweaver-es-03"]

# Index settings
indices.memory.index_buffer_size: 30%
indices.memory.min_index_buffer_size: 48mb

# Performance settings
thread_pool.write.queue_size: 1000
thread_pool.search.queue_size: 1000

# Security settings
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.http.ssl.enabled: true

---
# logstash.conf
input {
  beats {
    port => 5044
  }
  
  http {
    port => 8080
    codec => json
  }
}

filter {
  # Parse CodeWeaver application logs
  if [fields][service] == "codeweaver" {
    json {
      source => "message"
    }
    
    # Add log level numeric value for sorting
    if [level] == "DEBUG" { mutate { add_field => { "level_num" => 10 } } }
    if [level] == "INFO" { mutate { add_field => { "level_num" => 20 } } }
    if [level] == "WARNING" { mutate { add_field => { "level_num" => 30 } } }
    if [level] == "ERROR" { mutate { add_field => { "level_num" => 40 } } }
    if [level] == "CRITICAL" { mutate { add_field => { "level_num" => 50 } } }
    
    # Parse search queries for analysis
    if [event_type] == "search" {
      mutate {
        add_tag => ["search"]
      }
      
      # Extract query terms
      if [query] {
        mutate {
          add_field => { "query_length" => "%{[query]}" }
        }
        ruby {
          code => "
            query = event.get('query')
            event.set('query_length', query.length)
            event.set('query_word_count', query.split.length)
          "
        }
      }
    }
    
    # Parse performance data
    if [response_time] {
      mutate {
        convert => { "response_time" => "float" }
      }
      
      # Categorize response times
      if [response_time] < 0.1 {
        mutate { add_field => { "performance_category" => "fast" } }
      } else if [response_time] < 1.0 {
        mutate { add_field => { "performance_category" => "normal" } }
      } else if [response_time] < 5.0 {
        mutate { add_field => { "performance_category" => "slow" } }
      } else {
        mutate { add_field => { "performance_category" => "very_slow" } }
      }
    }
    
    # Parse error information
    if [level] in ["ERROR", "CRITICAL"] {
      mutate {
        add_tag => ["error"]
      }
      
      # Extract stack trace information
      if [traceback] {
        ruby {
          code => "
            traceback = event.get('traceback')
            lines = traceback.split(\"\n\")
            event.set('error_file', lines[1].match(/File \"([^\"]+)\"/)[1] rescue nil)
            event.set('error_line', lines[1].match(/line (\d+)/)[1] rescue nil)
          "
        }
      }
    }
    
    # GeoIP lookup for client IPs
    if [client_ip] {
      geoip {
        source => "client_ip"
        target => "geoip"
      }
    }
  }
  
  # Parse NGINX access logs
  if [fields][service] == "nginx" {
    grok {
      match => { 
        "message" => "%{COMBINEDAPACHELOG} %{GREEDYDATA:nginx_extras}"
      }
    }
    
    mutate {
      convert => { "response" => "integer" }
      convert => { "bytes" => "integer" }
    }
    
    # Parse response time from nginx_extras
    grok {
      match => { 
        "nginx_extras" => "rt=%{NUMBER:request_time:float}"
      }
    }
  }
  
  # Add common fields
  mutate {
    add_field => { "[@metadata][index_prefix]" => "codeweaver" }
  }
  
  # Date parsing
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["https://elasticsearch:9200"]
    index => "%{[@metadata][index_prefix]}-%{+YYYY.MM.dd}"
    user => "logstash_writer"
    password => "${LOGSTASH_PASSWORD}"
    ssl => true
    ssl_certificate_verification => true
  }
  
  # Output errors to separate index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["https://elasticsearch:9200"]
      index => "codeweaver-errors-%{+YYYY.MM.dd}"
      user => "logstash_writer"
      password => "${LOGSTASH_PASSWORD}"
      ssl => true
    }
  }
}

---
# kibana.yml
server.host: "0.0.0.0"
server.port: 5601
elasticsearch.hosts: ["https://elasticsearch:9200"]
elasticsearch.username: "kibana_system"
elasticsearch.password: "${KIBANA_PASSWORD}"
elasticsearch.ssl.certificateAuthorities: ["/usr/share/kibana/config/certs/ca.crt"]

# Monitoring
monitoring.ui.ccs.enabled: false
monitoring.kibana.collection.enabled: false

# Security
server.ssl.enabled: true
server.ssl.certificate: "/usr/share/kibana/config/certs/kibana.crt"
server.ssl.key: "/usr/share/kibana/config/certs/kibana.key"
xpack.security.enabled: true

# Custom settings
xpack.canvas.enabled: true
xpack.infra.enabled: true
xpack.apm.enabled: false
```

#### Structured Logging Implementation
```python
# structured_logger.py
import json
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from functools import wraps

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
tenant_id_var: ContextVar[str] = ContextVar('tenant_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context information
        request_id = request_id_var.get('')
        if request_id:
            log_entry['request_id'] = request_id
        
        tenant_id = tenant_id_var.get('')
        if tenant_id:
            log_entry['tenant_id'] = tenant_id
        
        user_id = user_id_var.get('')
        if user_id:
            log_entry['user_id'] = user_id
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add performance data if available
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        
        if hasattr(record, 'response_time'):
            log_entry['response_time'] = record.response_time
        
        return json.dumps(log_entry)

class CodeWeaverLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add structured handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def info(self, message: str, **extra):
        """Log info message with extra fields."""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, **extra):
        """Log warning message with extra fields."""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, **extra):
        """Log error message with extra fields."""
        self._log(logging.ERROR, message, extra)
    
    def critical(self, message: str, **extra):
        """Log critical message with extra fields."""
        self._log(logging.CRITICAL, message, extra)
    
    def debug(self, message: str, **extra):
        """Log debug message with extra fields."""
        self._log(logging.DEBUG, message, extra)
    
    def log_request(self, method: str, endpoint: str, status_code: int, 
                   response_time: float, **extra):
        """Log HTTP request with standardized fields."""
        log_data = {
            'event_type': 'http_request',
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time': response_time,
            **extra
        }
        self._log(logging.INFO, f"{method} {endpoint} - {status_code}", log_data)
    
    def log_search(self, query: str, result_count: int, duration: float, 
                  query_type: str = "semantic", **extra):
        """Log search operation with standardized fields."""
        log_data = {
            'event_type': 'search',
            'query': query,
            'query_type': query_type,
            'result_count': result_count,
            'duration': duration,
            'query_length': len(query),
            'query_word_count': len(query.split()),
            **extra
        }
        self._log(logging.INFO, f"Search query executed", log_data)
    
    def log_vector_operation(self, operation: str, collection: str, 
                           duration: float, success: bool, **extra):
        """Log vector database operation."""
        log_data = {
            'event_type': 'vector_operation',
            'operation': operation,
            'collection': collection,
            'duration': duration,
            'success': success,
            **extra
        }
        level = logging.INFO if success else logging.ERROR
        message = f"Vector {operation} on {collection} {'succeeded' if success else 'failed'}"
        self._log(level, message, log_data)
    
    def log_cache_operation(self, operation: str, cache_type: str, 
                          hit: bool, **extra):
        """Log cache operation."""
        log_data = {
            'event_type': 'cache',
            'operation': operation,
            'cache_type': cache_type,
            'hit': hit,
            **extra
        }
        self._log(logging.DEBUG, f"Cache {operation} - {'hit' if hit else 'miss'}", log_data)
    
    def log_performance_alert(self, metric: str, value: float, 
                            threshold: float, severity: str, **extra):
        """Log performance alert."""
        log_data = {
            'event_type': 'performance_alert',
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'severity': severity,
            **extra
        }
        level = logging.WARNING if severity == 'warning' else logging.ERROR
        self._log(level, f"Performance alert: {metric} = {value} (threshold: {threshold})", log_data)
    
    def _log(self, level: int, message: str, extra: Dict[str, Any]):
        """Internal logging method."""
        record = self.logger.makeRecord(
            self.logger.name, level, '', 0, message, (), None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def log_with_context(self, request_id: str, tenant_id: str = '', user_id: str = ''):
        """Context manager for logging with request context."""
        class LogContext:
            def __enter__(context_self):
                request_id_var.set(request_id)
                if tenant_id:
                    tenant_id_var.set(tenant_id)
                if user_id:
                    user_id_var.set(user_id)
                return self
            
            def __exit__(context_self, exc_type, exc_val, exc_tb):
                request_id_var.set('')
                tenant_id_var.set('')
                user_id_var.set('')
        
        return LogContext()

# Usage decorator for automatic request logging
def log_requests(logger: CodeWeaverLogger):
    """Decorator for automatic request logging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = kwargs.get('request_id', 'unknown')
            tenant_id = kwargs.get('tenant_id', 'default')
            
            with logger.log_with_context(request_id, tenant_id):
                try:
                    result = await func(*args, **kwargs)
                    response_time = time.time() - start_time
                    
                    logger.log_request(
                        method=kwargs.get('method', 'UNKNOWN'),
                        endpoint=kwargs.get('endpoint', func.__name__),
                        status_code=200,
                        response_time=response_time
                    )
                    
                    return result
                except Exception as e:
                    response_time = time.time() - start_time
                    
                    logger.log_request(
                        method=kwargs.get('method', 'UNKNOWN'),
                        endpoint=kwargs.get('endpoint', func.__name__),
                        status_code=500,
                        response_time=response_time,
                        error=str(e)
                    )
                    
                    logger.error(f"Request failed: {str(e)}", 
                               error_type=type(e).__name__)
                    raise
        
        return wrapper
    return decorator

# Global logger instance
logger = CodeWeaverLogger('codeweaver')
```

## Alerting and Incident Management

### AlertManager Configuration

#### Alert Routing and Notifications
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'mail.company.com:587'
  smtp_from: 'alerts@company.com'
  smtp_auth_username: 'alerts@company.com'
  smtp_auth_password: 'password'

# Inhibition rules to reduce noise
inhibit_rules:
  - source_matchers:
      - alertname = CodeWeaverDown
    target_matchers:
      - alertname =~ "CodeWeaver.*"
    equal: ['instance']

  - source_matchers:
      - severity = critical
    target_matchers:
      - severity = warning
    equal: ['alertname', 'instance']

# Alert routing
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  
  routes:
  # Critical alerts to on-call team
  - matchers:
      - severity = critical
    receiver: 'critical-alerts'
    group_wait: 5s
    repeat_interval: 15m
    
  # Application-specific alerts
  - matchers:
      - component = application
    receiver: 'app-team'
    group_by: ['alertname', 'tenant_id']
    
  # Infrastructure alerts
  - matchers:
      - component = infrastructure
    receiver: 'infra-team'
    
  # Business alerts
  - matchers:
      - component = business
    receiver: 'business-team'
    repeat_interval: 4h

# Notification receivers
receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://webhook.company.com/alerts'
    send_resolved: true

- name: 'critical-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#critical-alerts'
    title: 'CRITICAL: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    color: 'danger'
    send_resolved: true
  
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    severity: 'critical'
    
  email_configs:
  - to: 'oncall@company.com'
    subject: 'CRITICAL Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    body: |
      Alert Details:
      {{ range .Alerts }}
      Summary: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Instance: {{ .Labels.instance }}
      Severity: {{ .Labels.severity }}
      {{ end }}

- name: 'app-team'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#codeweaver-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    color: '{{ if eq .Status "firing" }}warning{{ else }}good{{ end }}'

- name: 'infra-team'
  email_configs:
  - to: 'infrastructure@company.com'
    subject: 'Infrastructure Alert: {{ .GroupLabels.alertname }}'

- name: 'business-team'
  email_configs:
  - to: 'business-intelligence@company.com'
    subject: 'Business Metric Alert: {{ .GroupLabels.alertname }}'
```

### Incident Response Automation

#### Automated Incident Response
```python
# incident_response.py
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IncidentStatus(Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class Incident:
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assignee: Optional[str] = None
    labels: Dict[str, str] = None
    
class IncidentManager:
    def __init__(self, alertmanager_client, slack_client, prometheus_client):
        self.alertmanager = alertmanager_client
        self.slack = slack_client
        self.prometheus = prometheus_client
        self.incidents = {}
        self.auto_response_enabled = True
        
        # Response playbooks
        self.playbooks = {
            'CodeWeaverHighErrorRate': self.handle_high_error_rate,
            'CodeWeaverDown': self.handle_service_down,
            'CodeWeaverHighLatency': self.handle_high_latency,
            'CodeWeaverVectorDBDown': self.handle_vector_db_down,
            'CodeWeaverHighMemoryUsage': self.handle_high_memory,
        }
    
    async def handle_alert(self, alert_data: Dict[str, Any]):
        """Handle incoming alert from AlertManager."""
        alert_name = alert_data.get('labels', {}).get('alertname')
        severity = alert_data.get('labels', {}).get('severity', 'medium')
        status = alert_data.get('status', 'firing')
        
        if status == 'firing':
            incident = await self.create_incident(alert_data)
            
            # Execute automatic response if enabled
            if self.auto_response_enabled and alert_name in self.playbooks:
                await self.playbooks[alert_name](incident, alert_data)
        
        elif status == 'resolved':
            # Update existing incident
            incident_id = self.get_incident_id_from_alert(alert_data)
            if incident_id in self.incidents:
                await self.resolve_incident(incident_id)
    
    async def create_incident(self, alert_data: Dict[str, Any]) -> Incident:
        """Create new incident from alert."""
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        labels = alert_data.get('labels', {})
        annotations = alert_data.get('annotations', {})
        
        incident = Incident(
            id=incident_id,
            title=annotations.get('summary', 'Unknown alert'),
            description=annotations.get('description', ''),
            severity=IncidentSeverity(labels.get('severity', 'medium')),
            status=IncidentStatus.OPEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            labels=labels
        )
        
        self.incidents[incident_id] = incident
        
        # Notify stakeholders
        await self.notify_incident_created(incident)
        
        return incident
    
    async def handle_high_error_rate(self, incident: Incident, alert_data: Dict[str, Any]):
        """Automated response for high error rate."""
        instance = alert_data.get('labels', {}).get('instance')
        
        # Step 1: Gather diagnostic information
        diagnostics = await self.gather_diagnostics(instance)
        
        # Step 2: Check if instance is healthy
        health_status = await self.check_instance_health(instance)
        
        # Step 3: Automatic remediation
        if not health_status['healthy']:
            await self.restart_instance(instance)
            
            # Update incident
            incident.status = IncidentStatus.INVESTIGATING
            incident.description += f"\n\nAutomatic remediation: Restarted unhealthy instance {instance}"
            
            await self.notify_incident_update(incident, "Automatic restart initiated")
        
        # Step 4: Scale up if needed
        error_rate = diagnostics.get('error_rate', 0)
        if error_rate > 0.1:  # More than 10% error rate
            await self.scale_up_service()
            incident.description += "\n\nAutomatic scaling: Increased replica count"
            
        await self.update_incident(incident)
    
    async def handle_service_down(self, incident: Incident, alert_data: Dict[str, Any]):
        """Automated response for service down."""
        instance = alert_data.get('labels', {}).get('instance')
        
        # Step 1: Verify the service is actually down
        is_down = await self.verify_service_down(instance)
        
        if is_down:
            # Step 2: Attempt automatic restart
            restart_success = await self.restart_instance(instance)
            
            if restart_success:
                incident.status = IncidentStatus.INVESTIGATING
                incident.description += f"\n\nAutomatic restart successful for {instance}"
                await self.notify_incident_update(incident, "Service automatically restarted")
            else:
                # Step 3: Escalate if restart fails
                incident.severity = IncidentSeverity.CRITICAL
                incident.assignee = "oncall-engineer"
                await self.escalate_incident(incident)
        
        await self.update_incident(incident)
    
    async def handle_high_latency(self, incident: Incident, alert_data: Dict[str, Any]):
        """Automated response for high latency."""
        # Step 1: Check cache performance
        cache_metrics = await self.get_cache_metrics()
        
        if cache_metrics['hit_rate'] < 0.5:
            # Warm cache with popular queries
            await self.warm_cache()
            incident.description += "\n\nAutomatic remediation: Cache warming initiated"
        
        # Step 2: Check if scaling is needed
        current_load = await self.get_current_load()
        if current_load > 0.8:
            await self.scale_up_service()
            incident.description += "\n\nAutomatic scaling: Increased capacity due to high load"
        
        # Step 3: Enable circuit breaker if needed
        if current_load > 0.9:
            await self.enable_circuit_breaker()
            incident.description += "\n\nCircuit breaker enabled to protect system"
        
        await self.update_incident(incident)
    
    async def handle_vector_db_down(self, incident: Incident, alert_data: Dict[str, Any]):
        """Automated response for vector database issues."""
        # Step 1: Check connection to backup instances
        backup_available = await self.check_vector_db_backups()
        
        if backup_available:
            # Failover to backup
            await self.failover_vector_db()
            incident.description += "\n\nAutomatic failover to backup vector database"
            incident.status = IncidentStatus.INVESTIGATING
        else:
            # Escalate immediately
            incident.severity = IncidentSeverity.CRITICAL
            incident.assignee = "database-team"
            await self.escalate_incident(incident)
        
        await self.update_incident(incident)
    
    async def handle_high_memory(self, incident: Incident, alert_data: Dict[str, Any]):
        """Automated response for high memory usage."""
        instance = alert_data.get('labels', {}).get('instance')
        
        # Step 1: Clear caches to free memory
        await self.clear_caches(instance)
        incident.description += f"\n\nCleared caches on {instance} to free memory"
        
        # Step 2: Check if memory usage is still high
        await asyncio.sleep(60)  # Wait for cache clearing to take effect
        
        current_memory = await self.get_memory_usage(instance)
        if current_memory > 0.85:  # Still high
            # Scale horizontally
            await self.scale_up_service()
            incident.description += "\n\nScaled up service due to persistent high memory usage"
        
        await self.update_incident(incident)
    
    async def gather_diagnostics(self, instance: str) -> Dict[str, Any]:
        """Gather diagnostic information for instance."""
        return {
            'error_rate': await self.get_error_rate(instance),
            'response_time': await self.get_response_time(instance),
            'memory_usage': await self.get_memory_usage(instance),
            'cpu_usage': await self.get_cpu_usage(instance),
            'active_connections': await self.get_active_connections(instance)
        }
    
    async def notify_incident_created(self, incident: Incident):
        """Notify stakeholders of new incident."""
        message = f"""
🚨 **New Incident Created**
**ID:** {incident.id}
**Title:** {incident.title}
**Severity:** {incident.severity.value.upper()}
**Description:** {incident.description}
**Created:** {incident.created_at.isoformat()}
        """
        
        channel = '#critical-alerts' if incident.severity == IncidentSeverity.CRITICAL else '#codeweaver-alerts'
        await self.slack.send_message(channel, message)
    
    async def notify_incident_update(self, incident: Incident, update: str):
        """Notify stakeholders of incident update."""
        message = f"""
📝 **Incident Update**
**ID:** {incident.id}
**Status:** {incident.status.value.upper()}
**Update:** {update}
**Updated:** {datetime.utcnow().isoformat()}
        """
        
        channel = '#critical-alerts' if incident.severity == IncidentSeverity.CRITICAL else '#codeweaver-alerts'
        await self.slack.send_message(channel, message)
    
    async def escalate_incident(self, incident: Incident):
        """Escalate incident to higher priority."""
        # Send to PagerDuty for critical incidents
        if incident.severity == IncidentSeverity.CRITICAL:
            await self.trigger_pagerduty(incident)
        
        # Notify management for critical incidents
        await self.notify_management(incident)
    
    async def resolve_incident(self, incident_id: str):
        """Mark incident as resolved."""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.updated_at = datetime.utcnow()
            
            await self.notify_incident_update(incident, "Incident automatically resolved")
```

## Grafana Dashboards and Visualization

### Main CodeWeaver Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "CodeWeaver Production Dashboard",
    "tags": ["codeweaver", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"codeweaver\"}",
            "legendFormat": "Instances Up"
          },
          {
            "expr": "rate(codeweaver_requests_total[5m])",
            "legendFormat": "Requests/sec"
          },
          {
            "expr": "histogram_quantile(0.95, rate(codeweaver_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th Percentile Latency"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 2}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(codeweaver_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests per second"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(codeweaver_request_duration_seconds_bucket[5m])",
            "format": "heatmap",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate by Endpoint",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(codeweaver_requests_total{status=~\"5..\"}[5m])) by (endpoint) / sum(rate(codeweaver_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate",
            "max": 1,
            "min": 0
          }
        ]
      },
      {
        "id": 5,
        "title": "Search Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(codeweaver_search_latency_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(codeweaver_search_latency_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(codeweaver_search_latency_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "id": 6,
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(codeweaver_cache_hit_rate)",
            "legendFormat": "Overall Hit Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "unit": "percentunit"
          }
        }
      },
      {
        "id": 7,
        "title": "Vector Database Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(codeweaver_vector_operations_total[5m])) by (operation)",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "id": 8,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "codeweaver_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "{{component}}"
          }
        ],
        "yAxes": [
          {
            "label": "Memory (GB)"
          }
        ]
      },
      {
        "id": 9,
        "title": "Active Connections by Tenant",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(codeweaver_active_connections) by (tenant_id)",
            "legendFormat": "{{tenant_id}}"
          }
        ]
      },
      {
        "id": 10,
        "title": "Top Search Queries",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, sum(rate(codeweaver_requests_total{endpoint=~\".*search.*\"}[1h])) by (query))",
            "format": "table"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Performance and SLA Monitoring

### SLA Definition and Tracking
```python
# sla_monitor.py
from typing import Dict, List, Any, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class SLAStatus(Enum):
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    BREACHED = "breached"

@dataclass
class SLATarget:
    name: str
    description: str
    target_value: float
    measurement_window: timedelta
    error_budget_percent: float

@dataclass
class SLAMeasurement:
    timestamp: datetime
    actual_value: float
    target_value: float
    status: SLAStatus
    error_budget_remaining: float

class SLAMonitor:
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
        
        # Define SLA targets
        self.sla_targets = {
            'availability': SLATarget(
                name='Availability',
                description='System uptime percentage',
                target_value=0.999,  # 99.9%
                measurement_window=timedelta(days=30),
                error_budget_percent=0.1  # 0.1% error budget
            ),
            'response_time': SLATarget(
                name='Response Time',
                description='95th percentile response time',
                target_value=1.0,  # 1 second
                measurement_window=timedelta(hours=1),
                error_budget_percent=5.0  # 5% of requests can exceed target
            ),
            'search_latency': SLATarget(
                name='Search Latency',
                description='95th percentile search latency',
                target_value=0.5,  # 500ms
                measurement_window=timedelta(hours=1),
                error_budget_percent=5.0
            ),
            'error_rate': SLATarget(
                name='Error Rate',
                description='Percentage of failed requests',
                target_value=0.01,  # 1% maximum error rate
                measurement_window=timedelta(hours=1),
                error_budget_percent=1.0
            )
        }
    
    async def measure_sla(self, sla_name: str, tenant_id: str = None) -> SLAMeasurement:
        """Measure current SLA status."""
        target = self.sla_targets[sla_name]
        
        # Build tenant filter
        tenant_filter = f'{{tenant_id="{tenant_id}"}}' if tenant_id else ''
        
        # Get current measurement based on SLA type
        if sla_name == 'availability':
            actual_value = await self.measure_availability(tenant_filter, target.measurement_window)
        elif sla_name == 'response_time':
            actual_value = await self.measure_response_time(tenant_filter)
        elif sla_name == 'search_latency':
            actual_value = await self.measure_search_latency(tenant_filter)
        elif sla_name == 'error_rate':
            actual_value = await self.measure_error_rate(tenant_filter)
        else:
            raise ValueError(f"Unknown SLA: {sla_name}")
        
        # Calculate status and error budget
        status, error_budget_remaining = self.calculate_sla_status(
            actual_value, target
        )
        
        return SLAMeasurement(
            timestamp=datetime.utcnow(),
            actual_value=actual_value,
            target_value=target.target_value,
            status=status,
            error_budget_remaining=error_budget_remaining
        )
    
    async def measure_availability(self, tenant_filter: str, window: timedelta) -> float:
        """Measure system availability."""
        query = f'avg_over_time(up{{job="codeweaver"}}{tenant_filter}[{int(window.total_seconds())}s])'
        result = await self.prometheus.query(query)
        return float(result.get('value', [0, 0])[1])
    
    async def measure_response_time(self, tenant_filter: str) -> float:
        """Measure 95th percentile response time."""
        query = f'histogram_quantile(0.95, rate(codeweaver_request_duration_seconds_bucket{tenant_filter}[5m]))'
        result = await self.prometheus.query(query)
        return float(result.get('value', [0, 0])[1])
    
    async def measure_search_latency(self, tenant_filter: str) -> float:
        """Measure 95th percentile search latency."""
        query = f'histogram_quantile(0.95, rate(codeweaver_search_latency_seconds_bucket{tenant_filter}[5m]))'
        result = await self.prometheus.query(query)
        return float(result.get('value', [0, 0])[1])
    
    async def measure_error_rate(self, tenant_filter: str) -> float:
        """Measure error rate."""
        error_query = f'sum(rate(codeweaver_requests_total{{status=~"5.."}}{tenant_filter}[5m]))'
        total_query = f'sum(rate(codeweaver_requests_total{tenant_filter}[5m]))'
        
        error_result = await self.prometheus.query(error_query)
        total_result = await self.prometheus.query(total_query)
        
        errors = float(error_result.get('value', [0, 0])[1])
        total = float(total_result.get('value', [0, 1])[1])  # Avoid division by zero
        
        return errors / total if total > 0 else 0
    
    def calculate_sla_status(self, actual: float, target: SLATarget) -> tuple:
        """Calculate SLA status and error budget remaining."""
        if target.name in ['availability']:
            # For availability, higher is better
            performance_ratio = actual / target.target_value
            error_budget_used = max(0, (target.target_value - actual) / (target.error_budget_percent / 100))
        else:
            # For latency/error rate, lower is better
            if actual <= target.target_value:
                performance_ratio = 1.0
                error_budget_used = 0
            else:
                performance_ratio = target.target_value / actual
                error_budget_used = (actual - target.target_value) / target.target_value
        
        error_budget_remaining = max(0, 1 - error_budget_used)
        
        # Determine status
        if error_budget_remaining > 0.5:
            status = SLAStatus.HEALTHY
        elif error_budget_remaining > 0.1:
            status = SLAStatus.AT_RISK
        else:
            status = SLAStatus.BREACHED
        
        return status, error_budget_remaining
    
    async def generate_sla_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive SLA report."""
        report = {
            'period_days': period_days,
            'generated_at': datetime.utcnow().isoformat(),
            'sla_measurements': {},
            'overall_status': SLAStatus.HEALTHY,
            'error_budget_burn_rate': {},
            'recommendations': []
        }
        
        worst_status = SLAStatus.HEALTHY
        
        for sla_name, target in self.sla_targets.items():
            measurement = await self.measure_sla(sla_name)
            report['sla_measurements'][sla_name] = {
                'current_value': measurement.actual_value,
                'target_value': measurement.target_value,
                'status': measurement.status.value,
                'error_budget_remaining': measurement.error_budget_remaining,
                'description': target.description
            }
            
            # Track worst status
            if measurement.status.value == 'breached':
                worst_status = SLAStatus.BREACHED
            elif measurement.status.value == 'at_risk' and worst_status == SLAStatus.HEALTHY:
                worst_status = SLAStatus.AT_RISK
        
        report['overall_status'] = worst_status.value
        
        # Generate recommendations
        report['recommendations'] = await self.generate_sla_recommendations(
            report['sla_measurements']
        )
        
        return report
    
    async def generate_sla_recommendations(self, measurements: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on SLA performance."""
        recommendations = []
        
        # Check availability
        if measurements['availability']['status'] in ['at_risk', 'breached']:
            recommendations.append(
                "Availability is below target. Consider implementing redundancy and "
                "reviewing infrastructure health checks."
            )
        
        # Check response time
        if measurements['response_time']['status'] in ['at_risk', 'breached']:
            recommendations.append(
                "Response time is above target. Consider scaling up resources, "
                "implementing caching, or optimizing database queries."
            )
        
        # Check search latency
        if measurements['search_latency']['status'] in ['at_risk', 'breached']:
            recommendations.append(
                "Search latency is high. Consider optimizing vector database "
                "configuration, implementing result caching, or scaling search infrastructure."
            )
        
        # Check error rate
        if measurements['error_rate']['status'] in ['at_risk', 'breached']:
            recommendations.append(
                "Error rate is elevated. Review recent deployments, check logs for "
                "error patterns, and consider implementing circuit breakers."
            )
        
        return recommendations
```

## Next Steps

1. **Deploy Monitoring Stack**: Set up Prometheus, Grafana, and AlertManager
2. **Configure Dashboards**: Import and customize monitoring dashboards
3. **Set Up Alerting**: Configure alert rules and notification channels
4. **Implement SLA Monitoring**: Define and track service level agreements
5. **Test Incident Response**: Validate automated response procedures

For advanced monitoring setups and custom metrics development, consider working with observability engineers and site reliability engineers to ensure comprehensive coverage of your production environment.