# Scalability and Performance

This guide covers scaling CodeWeaver for large organizations, multiple teams, and massive codebases. It includes horizontal and vertical scaling strategies, multi-tenant architectures, and performance optimization techniques.

## Scalability Architecture Overview

CodeWeaver's scalable architecture supports growth from small teams to enterprise-scale deployments:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enterprise Scale Architecture                 │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Client    │    │    CDN/     │    │Load Balancer│        │
│  │   Tier      │────│   Proxy     │────│    Tier     │        │
│  │             │    │   Layer     │    │             │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                │                │
│  ┌─────────────────────────────────────────────┼─────────────┐  │
│  │                Application Tier             │             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │CodeWvr-1│  │CodeWvr-2│  │CodeWvr-3│  │CodeWvr-N│     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                │                │
│  ┌─────────────────────────────────────────────┼─────────────┐  │
│  │                  Data Tier                  │             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │Vector-1 │  │Vector-2 │  │Vector-3 │  │Cache    │     │  │
│  │  │Cluster  │  │Cluster  │  │Cluster  │  │Cluster  │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Scaling Strategies

### Horizontal Scaling

#### Auto-Scaling Configuration
```yaml
# horizontal-pod-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: codeweaver-hpa
  namespace: codeweaver
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codeweaver
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: search_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

#### Cluster Auto-Scaling
```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/codeweaver-cluster
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
```

### Vertical Scaling

#### Vertical Pod Autoscaler
```yaml
# vertical-pod-autoscaler.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: codeweaver-vpa
  namespace: codeweaver
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codeweaver
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: codeweaver
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

### Multi-Region Deployment

#### Global Load Balancer Configuration
```yaml
# global-load-balancer.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: codeweaver-ssl-cert
spec:
  domains:
    - codeweaver.global.com
    - us.codeweaver.global.com
    - eu.codeweaver.global.com
    - asia.codeweaver.global.com

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: codeweaver-global-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "codeweaver-global-ip"
    networking.gke.io/managed-certificates: "codeweaver-ssl-cert"
    kubernetes.io/ingress.class: "gce"
    cloud.google.com/backend-config: '{"default": "codeweaver-backend-config"}'
spec:
  rules:
  - host: codeweaver.global.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: codeweaver-service
            port:
              number: 80

---
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: codeweaver-backend-config
spec:
  healthCheck:
    checkIntervalSec: 10
    timeoutSec: 5
    healthyThreshold: 1
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health
    port: 8080
  sessionAffinity:
    affinityType: "CLIENT_IP"
    affinityCookieTtlSec: 3600
  cdn:
    enabled: true
    cachePolicy:
      includeHost: true
      includeProtocol: true
      includeQueryString: false
  iap:
    enabled: false
  timeoutSec: 30
  connectionDraining:
    drainingTimeoutSec: 60
```

## Multi-Tenant Architecture

### Tenant Isolation Strategies

#### Namespace-based Isolation
```yaml
# tenant-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: codeweaver-tenant-acme
  labels:
    tenant: acme
    tier: enterprise
    region: us-east-1

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-isolation
  namespace: codeweaver-tenant-acme
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: codeweaver-shared
    - namespaceSelector:
        matchLabels:
          tenant: acme
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: codeweaver-shared
    - namespaceSelector:
        matchLabels:
          tenant: acme
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: UDP
      port: 53

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-quota
  namespace: codeweaver-tenant-acme
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    services: "5"
    secrets: "10"
    configmaps: "10"
```

#### Database-level Isolation
```python
# tenant_isolation.py
from typing import Dict, List, Optional
from codeweaver.db import DatabaseManager
from codeweaver.auth import TenantContext

class TenantManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tenant_configs = {}
    
    def create_tenant(self, tenant_id: str, config: Dict) -> bool:
        """Create isolated tenant environment."""
        try:
            # Create tenant-specific vector collection
            collection_name = f"codeweaver_tenant_{tenant_id}"
            self.db_manager.create_collection(
                collection_name,
                vector_size=config.get('vector_size', 1536),
                distance=config.get('distance', 'cosine'),
                shard_number=config.get('shards', 2),
                replication_factor=config.get('replicas', 1)
            )
            
            # Create tenant-specific indices
            indices = config.get('indices', ['content', 'metadata'])
            for index in indices:
                self.db_manager.create_index(collection_name, index)
            
            # Set up tenant configuration
            self.tenant_configs[tenant_id] = {
                'collection': collection_name,
                'created_at': datetime.utcnow(),
                'config': config,
                'status': 'active'
            }
            
            # Create tenant-specific service account
            self.create_tenant_service_account(tenant_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create tenant {tenant_id}: {e}")
            return False
    
    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Get tenant-specific context for operations."""
        if tenant_id not in self.tenant_configs:
            return None
        
        config = self.tenant_configs[tenant_id]
        return TenantContext(
            tenant_id=tenant_id,
            collection_name=config['collection'],
            config=config['config'],
            isolation_level='database'
        )
    
    def enforce_tenant_isolation(self, tenant_id: str, operation: str, resource: str) -> bool:
        """Enforce tenant isolation for operations."""
        tenant_context = self.get_tenant_context(tenant_id)
        if not tenant_context:
            return False
        
        # Ensure operations only access tenant-specific resources
        if operation in ['search', 'index', 'query']:
            return resource.startswith(f"codeweaver_tenant_{tenant_id}")
        
        return True
    
    def create_tenant_service_account(self, tenant_id: str):
        """Create Kubernetes service account for tenant."""
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": f"codeweaver-tenant-{tenant_id}",
                "namespace": f"codeweaver-tenant-{tenant_id}",
                "labels": {
                    "tenant": tenant_id,
                    "component": "codeweaver"
                }
            }
        }
        
        # Create RBAC for tenant
        role_binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": f"codeweaver-tenant-{tenant_id}-binding",
                "namespace": f"codeweaver-tenant-{tenant_id}"
            },
            "subjects": [{
                "kind": "ServiceAccount",
                "name": f"codeweaver-tenant-{tenant_id}",
                "namespace": f"codeweaver-tenant-{tenant_id}"
            }],
            "roleRef": {
                "kind": "Role",
                "name": "codeweaver-tenant-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
        
        # Apply to Kubernetes
        self.apply_kubernetes_resources([service_account, role_binding])
```

### Tenant Resource Management

#### Resource Allocation per Tenant
```yaml
# tenant-resource-allocation.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-limits
  namespace: codeweaver-tenant-acme
spec:
  limits:
  - default:
      cpu: "2"
      memory: "4Gi"
    defaultRequest:
      cpu: "500m"
      memory: "1Gi"
    max:
      cpu: "8"
      memory: "16Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    type: Container
  - max:
      cpu: "16"
      memory: "32Gi"
    type: Pod
  - max:
      storage: "100Gi"
    type: PersistentVolumeClaim

---
apiVersion: policy/v1beta1
kind: PodDisruptionBudget
metadata:
  name: tenant-pdb
  namespace: codeweaver-tenant-acme
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: codeweaver
      tenant: acme
```

## Performance Optimization

### Caching Strategies

#### Multi-Layer Caching Architecture
```python
# cache_manager.py
from typing import Any, Optional, List, Dict
import redis
import json
from codeweaver.config import CacheConfig

class CacheManager:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
            decode_responses=True
        )
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str, cache_level: str = 'all') -> Optional[Any]:
        """Get value from cache with multi-level strategy."""
        # L1: Local memory cache
        if cache_level in ['all', 'local'] and key in self.local_cache:
            self.cache_stats['hits'] += 1
            return self.local_cache[key]
        
        # L2: Redis distributed cache
        if cache_level in ['all', 'distributed']:
            value = self.redis_client.get(key)
            if value:
                parsed_value = json.loads(value)
                # Promote to L1 cache
                self.local_cache[key] = parsed_value
                self.cache_stats['hits'] += 1
                return parsed_value
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, cache_level: str = 'all'):
        """Set value in cache with TTL."""
        serialized_value = json.dumps(value)
        
        # L1: Local memory cache
        if cache_level in ['all', 'local']:
            self.local_cache[key] = value
            # Implement LRU eviction for local cache
            if len(self.local_cache) > self.config.local_cache_size:
                self.evict_lru()
        
        # L2: Redis distributed cache
        if cache_level in ['all', 'distributed']:
            self.redis_client.setex(key, ttl, serialized_value)
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # Invalidate local cache
        keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.local_cache[key]
        
        # Invalidate Redis cache
        keys = self.redis_client.keys(f"*{pattern}*")
        if keys:
            self.redis_client.delete(*keys)
    
    def evict_lru(self):
        """Evict least recently used items from local cache."""
        # Simple LRU implementation
        if self.local_cache:
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
            self.cache_stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size': len(self.local_cache),
            'evictions': self.cache_stats['evictions'],
            'redis_info': self.redis_client.info()
        }
```

#### Cache Warming Strategy
```python
# cache_warming.py
from typing import List, Dict
import asyncio
from codeweaver.search import SearchEngine
from codeweaver.cache import CacheManager

class CacheWarmer:
    def __init__(self, search_engine: SearchEngine, cache_manager: CacheManager):
        self.search_engine = search_engine
        self.cache_manager = cache_manager
        self.popular_queries = []
    
    async def warm_cache(self, tenant_id: str):
        """Warm cache with popular queries and common patterns."""
        # Get popular queries from analytics
        popular_queries = await self.get_popular_queries(tenant_id)
        
        # Warm search results cache
        tasks = []
        for query in popular_queries[:50]:  # Top 50 queries
            task = self.warm_search_cache(tenant_id, query)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Warm embedding cache for common patterns
        await self.warm_embedding_cache(tenant_id)
        
        # Warm metadata cache
        await self.warm_metadata_cache(tenant_id)
    
    async def warm_search_cache(self, tenant_id: str, query: str):
        """Warm cache for specific search query."""
        try:
            cache_key = f"search:{tenant_id}:{hash(query)}"
            
            # Check if already cached
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return
            
            # Execute search and cache result
            results = await self.search_engine.search(
                query=query,
                tenant_id=tenant_id,
                limit=20
            )
            
            self.cache_manager.set(cache_key, results, ttl=7200)  # 2 hours
            
        except Exception as e:
            logger.warning(f"Failed to warm cache for query {query}: {e}")
    
    async def warm_embedding_cache(self, tenant_id: str):
        """Warm embedding cache for common code patterns."""
        common_patterns = [
            "function definition",
            "class declaration",
            "import statement",
            "error handling",
            "database query",
            "API endpoint",
            "test case",
            "configuration"
        ]
        
        for pattern in common_patterns:
            try:
                cache_key = f"embedding:{tenant_id}:{hash(pattern)}"
                
                # Generate and cache embedding
                embedding = await self.search_engine.get_embedding(pattern)
                self.cache_manager.set(cache_key, embedding, ttl=86400)  # 24 hours
                
            except Exception as e:
                logger.warning(f"Failed to warm embedding cache for {pattern}: {e}")
```

### Database Optimization

#### Vector Database Tuning
```yaml
# qdrant-performance.yaml
service:
  http_port: 6333
  grpc_port: 6334

storage:
  # Optimize for large datasets
  mmap_threshold: 100000
  
  # Performance settings
  performance:
    max_indexing_threads: 32
    indexing_threshold: 20000
    max_search_threads: 16
    search_queue_size: 1000
  
  # Memory optimization
  wal:
    wal_capacity_mb: 256
    wal_segments_ahead: 4
  
  # Disk optimization
  optimizers:
    deleted_threshold: 0.1
    vacuum_min_vector_number: 10000
    default_segment_number: 32
    max_segment_size_kb: 131072  # 128MB
    memmap_threshold_kb: 4096
    indexing_threshold: 20000
    flush_interval_sec: 3
    max_optimization_threads: 8
  
  # Quantization for memory efficiency
  quantization:
    scalar:
      type: int8
      quantile: 0.99
      always_ram: true
    product:
      compression: x32
      always_ram: false

# Index configuration for optimal performance
collection_config:
  vectors:
    size: 1536
    distance: Cosine
  optimizers_config:
    deleted_threshold: 0.1
    vacuum_min_vector_number: 10000
    default_segment_number: 16
    max_segment_size_kb: 65536
    memmap_threshold_kb: 2048
    indexing_threshold: 20000
    flush_interval_sec: 5
    max_optimization_threads: 4
  wal_config:
    wal_capacity_mb: 128
    wal_segments_ahead: 2
  hnsw_config:
    m: 16
    ef_construct: 200
    full_scan_threshold: 10000
    max_indexing_threads: 16
  quantization_config:
    scalar:
      type: int8
      quantile: 0.99
      always_ram: true
```

#### Sharding Strategy
```python
# sharding_manager.py
from typing import Dict, List, Optional
import hashlib
from codeweaver.db import VectorDatabase

class ShardingManager:
    def __init__(self, databases: List[VectorDatabase]):
        self.databases = databases
        self.shard_count = len(databases)
        self.shard_map = {}
    
    def get_shard(self, key: str) -> VectorDatabase:
        """Get appropriate shard for a key using consistent hashing."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.shard_count
        return self.databases[shard_index]
    
    def get_shard_for_tenant(self, tenant_id: str) -> VectorDatabase:
        """Get dedicated shard for tenant (if configured)."""
        if tenant_id in self.shard_map:
            return self.databases[self.shard_map[tenant_id]]
        return self.get_shard(tenant_id)
    
    def create_tenant_shard(self, tenant_id: str, shard_index: int):
        """Assign dedicated shard to tenant."""
        if shard_index >= self.shard_count:
            raise ValueError(f"Shard index {shard_index} exceeds available shards")
        self.shard_map[tenant_id] = shard_index
    
    async def search_across_shards(self, query: str, limit: int = 20) -> List[Dict]:
        """Search across all shards and merge results."""
        tasks = []
        for db in self.databases:
            task = db.search(query, limit=limit//self.shard_count + 10)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Merge and sort results
        merged_results = []
        for shard_results in results:
            merged_results.extend(shard_results)
        
        # Sort by relevance score and return top results
        merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return merged_results[:limit]
```

## Load Balancing and Traffic Management

### Advanced Load Balancing

#### Intelligent Load Balancer Configuration
```nginx
# nginx-lb.conf
upstream codeweaver_backend {
    # Least connections with weighted distribution
    least_conn;
    
    # Production servers with health checks
    server codeweaver-1.internal:8080 weight=3 max_fails=3 fail_timeout=30s;
    server codeweaver-2.internal:8080 weight=3 max_fails=3 fail_timeout=30s;
    server codeweaver-3.internal:8080 weight=3 max_fails=3 fail_timeout=30s;
    
    # Backup servers (lower weight)
    server codeweaver-backup.internal:8080 weight=1 backup;
    
    # Keep alive connections
    keepalive 32;
}

upstream codeweaver_search {
    # Hash-based routing for cache locality
    hash $request_uri consistent;
    
    server codeweaver-search-1.internal:8080 weight=1;
    server codeweaver-search-2.internal:8080 weight=1;
    server codeweaver-search-3.internal:8080 weight=1;
    
    keepalive 16;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
limit_req_zone $http_x_tenant_id zone=tenant:10m rate=1000r/m;

server {
    listen 443 ssl http2;
    server_name codeweaver.example.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/codeweaver.crt;
    ssl_certificate_key /etc/ssl/private/codeweaver.key;
    
    # Rate limiting
    limit_req zone=api burst=20 nodelay;
    limit_req zone=tenant burst=100 nodelay;
    
    # Health check endpoint (no rate limiting)
    location /health {
        access_log off;
        proxy_pass http://codeweaver_backend;
        proxy_connect_timeout 1s;
        proxy_send_timeout 1s;
        proxy_read_timeout 1s;
    }
    
    # Search endpoints (use search-optimized upstream)
    location ~ ^/api/v1/(search|query) {
        proxy_pass http://codeweaver_search;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Cache search results
        proxy_cache search_cache;
        proxy_cache_valid 200 10m;
        proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
        proxy_cache_key "$scheme$request_method$host$request_uri$http_x_tenant_id";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Index endpoints (use general backend)
    location ~ ^/api/v1/index {
        proxy_pass http://codeweaver_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # No caching for index operations
        proxy_cache off;
        
        # Longer timeouts for indexing
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # All other API endpoints
    location /api/ {
        proxy_pass http://codeweaver_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Standard timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# Cache configuration
proxy_cache_path /var/cache/nginx/search levels=1:2 keys_zone=search_cache:10m 
                 max_size=1g inactive=60m use_temp_path=off;
```

#### Service Mesh with Istio
```yaml
# istio-traffic-management.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: codeweaver-vs
  namespace: codeweaver
spec:
  hosts:
  - codeweaver.example.com
  gateways:
  - codeweaver-gateway
  http:
  # Route search traffic to search-optimized service
  - match:
    - uri:
        prefix: /api/v1/search
    - uri:
        prefix: /api/v1/query
    route:
    - destination:
        host: codeweaver-search-service
        port:
          number: 80
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
  
  # Route index traffic to index-optimized service
  - match:
    - uri:
        prefix: /api/v1/index
    route:
    - destination:
        host: codeweaver-index-service
        port:
          number: 80
      weight: 100
    timeout: 300s
    retries:
      attempts: 2
      perTryTimeout: 150s
  
  # Default routing
  - route:
    - destination:
        host: codeweaver-service
        port:
          number: 80
      weight: 100
    timeout: 60s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: codeweaver-dr
  namespace: codeweaver
spec:
  host: codeweaver-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
        minHealthPercent: 30
    loadBalancer:
      simple: LEAST_CONN
    outlierDetection:
      consecutiveGatewayErrors: 3
      consecutive5xxErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
  - name: search
    labels:
      version: search-optimized
    trafficPolicy:
      loadBalancer:
        consistentHash:
          httpHeaderName: "x-search-key"
  - name: index
    labels:
      version: index-optimized
    trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 50
        http:
          http1MaxPendingRequests: 20
          maxRequestsPerConnection: 5
```

## Performance Monitoring and Optimization

### Application Performance Monitoring

#### Custom Metrics Collection
```python
# performance_metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps
from typing import Dict, Any, Callable

class PerformanceMonitor:
    def __init__(self):
        self.registry = CollectorRegistry()
        
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
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Search-specific metrics
        self.search_latency = Histogram(
            'codeweaver_search_latency_seconds',
            'Search operation latency',
            ['tenant_id', 'query_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.search_results_count = Histogram(
            'codeweaver_search_results_count',
            'Number of search results returned',
            ['tenant_id'],
            buckets=[1, 5, 10, 20, 50, 100, 200],
            registry=self.registry
        )
        
        # Vector database metrics
        self.vector_operations = Counter(
            'codeweaver_vector_operations_total',
            'Vector database operations',
            ['operation', 'tenant_id', 'status'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'codeweaver_cache_hits_total',
            'Cache hits',
            ['cache_type', 'tenant_id'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'codeweaver_cache_misses_total',
            'Cache misses',
            ['cache_type', 'tenant_id'],
            registry=self.registry
        )
        
        # System resource metrics
        self.active_connections = Gauge(
            'codeweaver_active_connections',
            'Number of active connections',
            ['tenant_id'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'codeweaver_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
    
    def track_request(self, method: str, endpoint: str, tenant_id: str = "default"):
        """Decorator to track request metrics."""
        def decorator(func: Callable) -> Callable:
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
        """Track search operation performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    results = await func(*args, **kwargs)
                    
                    # Track latency
                    duration = time.time() - start_time
                    self.search_latency.labels(
                        tenant_id=tenant_id,
                        query_type=query_type
                    ).observe(duration)
                    
                    # Track result count
                    result_count = len(results) if results else 0
                    self.search_results_count.labels(
                        tenant_id=tenant_id
                    ).observe(result_count)
                    
                    return results
                except Exception as e:
                    # Track failed searches
                    self.vector_operations.labels(
                        operation="search",
                        tenant_id=tenant_id,
                        status="error"
                    ).inc()
                    raise
            
            return wrapper
        return decorator
    
    def record_cache_hit(self, cache_type: str, tenant_id: str = "default"):
        """Record cache hit."""
        self.cache_hits.labels(
            cache_type=cache_type,
            tenant_id=tenant_id
        ).inc()
    
    def record_cache_miss(self, cache_type: str, tenant_id: str = "default"):
        """Record cache miss."""
        self.cache_misses.labels(
            cache_type=cache_type,
            tenant_id=tenant_id
        ).inc()
    
    def update_active_connections(self, tenant_id: str, count: int):
        """Update active connections gauge."""
        self.active_connections.labels(tenant_id=tenant_id).set(count)
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage gauge."""
        self.memory_usage.labels(component=component).set(bytes_used)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
```

### Performance Optimization Strategies

#### Query Optimization
```python
# query_optimizer.py
from typing import List, Dict, Any, Optional
import re
from codeweaver.search import SearchQuery

class QueryOptimizer:
    def __init__(self):
        self.common_patterns = {
            'function_search': r'\b(function|def|func)\s+(\w+)',
            'class_search': r'\b(class|struct|interface)\s+(\w+)',
            'variable_search': r'\b(var|let|const)\s+(\w+)',
            'import_search': r'\b(import|from|include|require)\s+([^\s]+)'
        }
        
        self.optimization_cache = {}
    
    def optimize_query(self, query: str, context: Dict[str, Any]) -> SearchQuery:
        """Optimize search query based on patterns and context."""
        # Check cache first
        cache_key = f"{query}:{hash(str(context))}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        optimized_query = SearchQuery(original_query=query)
        
        # Detect query patterns
        query_type = self.detect_query_type(query)
        optimized_query.query_type = query_type
        
        # Apply pattern-specific optimizations
        if query_type == 'function_search':
            optimized_query = self.optimize_function_search(query, context)
        elif query_type == 'class_search':
            optimized_query = self.optimize_class_search(query, context)
        elif query_type == 'semantic_search':
            optimized_query = self.optimize_semantic_search(query, context)
        
        # Apply general optimizations
        optimized_query = self.apply_general_optimizations(optimized_query, context)
        
        # Cache the optimized query
        self.optimization_cache[cache_key] = optimized_query
        
        return optimized_query
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of search query."""
        query_lower = query.lower()
        
        # Check for exact code patterns
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return pattern_name
        
        # Check for semantic indicators
        if any(word in query_lower for word in ['how', 'what', 'why', 'where', 'when']):
            return 'semantic_search'
        
        # Check for error/exception patterns
        if any(word in query_lower for word in ['error', 'exception', 'bug', 'fix']):
            return 'error_search'
        
        return 'general_search'
    
    def optimize_function_search(self, query: str, context: Dict[str, Any]) -> SearchQuery:
        """Optimize function search queries."""
        optimized = SearchQuery(original_query=query, query_type='function_search')
        
        # Extract function name if present
        match = re.search(self.common_patterns['function_search'], query, re.IGNORECASE)
        if match:
            function_name = match.group(2)
            optimized.filters = {
                'content_type': 'function',
                'function_name': function_name
            }
            
            # Boost relevance for exact matches
            optimized.boost_terms = [function_name]
        
        # Add language-specific filters if context provides language
        if 'language' in context:
            optimized.filters['language'] = context['language']
        
        return optimized
    
    def apply_general_optimizations(self, query: SearchQuery, context: Dict[str, Any]) -> SearchQuery:
        """Apply general optimizations to any query."""
        # Add tenant filtering
        if 'tenant_id' in context:
            if not query.filters:
                query.filters = {}
            query.filters['tenant_id'] = context['tenant_id']
        
        # Add recency boost for recent code
        if context.get('prefer_recent', True):
            query.recency_boost = True
        
        # Adjust result limit based on query complexity
        if not query.limit:
            if query.query_type in ['semantic_search', 'general_search']:
                query.limit = 20
            else:
                query.limit = 10
        
        # Add synonym expansion for better recall
        query.expand_synonyms = True
        
        return query

class SearchQuery:
    def __init__(self, original_query: str, query_type: str = 'general_search'):
        self.original_query = original_query
        self.query_type = query_type
        self.filters: Dict[str, Any] = {}
        self.boost_terms: List[str] = []
        self.limit: Optional[int] = None
        self.recency_boost: bool = False
        self.expand_synonyms: bool = False
```

## Capacity Planning and Resource Management

### Capacity Planning Framework

#### Resource Calculator
```python
# capacity_planner.py
from typing import Dict, Any, NamedTuple
import math

class ResourceRequirements(NamedTuple):
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    network_mbps: float
    vector_db_memory_gb: float

class CapacityPlanner:
    def __init__(self):
        # Base resource requirements per concurrent user
        self.base_user_resources = ResourceRequirements(
            cpu_cores=0.1,
            memory_gb=0.2,
            storage_gb=0.5,
            network_mbps=1.0,
            vector_db_memory_gb=0.1
        )
        
        # Resource multipliers for different operation types
        self.operation_multipliers = {
            'search': 1.0,
            'index': 3.0,
            'bulk_index': 5.0,
            'complex_query': 2.0
        }
        
        # Overhead factors
        self.system_overhead = 0.3  # 30% overhead for OS, monitoring, etc.
        self.redundancy_factor = 2.0  # 2x for HA deployment
    
    def calculate_requirements(self, 
                             concurrent_users: int,
                             codebases: int,
                             avg_codebase_size_mb: float,
                             search_qps: float,
                             index_operations_per_hour: int) -> Dict[str, ResourceRequirements]:
        """Calculate resource requirements for given load."""
        
        # Base user load
        base_cpu = concurrent_users * self.base_user_resources.cpu_cores
        base_memory = concurrent_users * self.base_user_resources.memory_gb
        base_storage = codebases * avg_codebase_size_mb / 1024  # Convert to GB
        base_network = concurrent_users * self.base_user_resources.network_mbps
        
        # Search load
        search_cpu_factor = search_qps / 10  # Assume 10 QPS per core
        search_cpu = search_cpu_factor * self.operation_multipliers['search']
        
        # Index load
        index_cpu_factor = index_operations_per_hour / 100  # Assume 100 ops per hour per core
        index_cpu = index_cpu_factor * self.operation_multipliers['index']
        
        # Vector database memory (based on embeddings)
        vector_count = codebases * avg_codebase_size_mb * 10  # ~10 vectors per MB of code
        vector_memory = (vector_count * 1536 * 4) / (1024**3)  # 1536-dim float32 vectors
        
        # Total requirements
        total_cpu = (base_cpu + search_cpu + index_cpu) * (1 + self.system_overhead)
        total_memory = (base_memory + vector_memory) * (1 + self.system_overhead)
        total_storage = base_storage * 3  # Code + embeddings + indices
        total_network = base_network * (1 + self.system_overhead)
        
        # Apply redundancy for HA
        ha_requirements = ResourceRequirements(
            cpu_cores=total_cpu * self.redundancy_factor,
            memory_gb=total_memory * self.redundancy_factor,
            storage_gb=total_storage * self.redundancy_factor,
            network_mbps=total_network,  # Network doesn't need 2x for HA
            vector_db_memory_gb=vector_memory * self.redundancy_factor
        )
        
        # Single node requirements (for comparison)
        single_node = ResourceRequirements(
            cpu_cores=total_cpu,
            memory_gb=total_memory,
            storage_gb=total_storage,
            network_mbps=total_network,
            vector_db_memory_gb=vector_memory
        )
        
        return {
            'single_node': single_node,
            'high_availability': ha_requirements,
            'recommended_nodes': math.ceil(ha_requirements.cpu_cores / 16),  # 16 cores per node
            'storage_iops': int(search_qps * 100),  # 100 IOPS per QPS
            'network_bandwidth_gbps': total_network / 1000
        }
    
    def generate_scaling_plan(self, current_load: Dict[str, Any], 
                            growth_rate: float, 
                            time_horizon_months: int) -> Dict[str, Any]:
        """Generate scaling plan based on growth projections."""
        scaling_plan = {'months': []}
        
        for month in range(1, time_horizon_months + 1):
            # Apply compound growth
            growth_factor = (1 + growth_rate) ** month
            
            projected_load = {
                'concurrent_users': int(current_load['concurrent_users'] * growth_factor),
                'codebases': int(current_load['codebases'] * growth_factor),
                'avg_codebase_size_mb': current_load['avg_codebase_size_mb'],
                'search_qps': current_load['search_qps'] * growth_factor,
                'index_operations_per_hour': int(current_load['index_operations_per_hour'] * growth_factor)
            }
            
            requirements = self.calculate_requirements(**projected_load)
            
            scaling_plan['months'].append({
                'month': month,
                'projected_load': projected_load,
                'requirements': requirements,
                'estimated_cost': self.estimate_cost(requirements['high_availability'])
            })
        
        return scaling_plan
    
    def estimate_cost(self, requirements: ResourceRequirements) -> Dict[str, float]:
        """Estimate monthly costs (USD) for given requirements."""
        # AWS pricing estimates (as of 2024)
        costs = {
            'compute': requirements.cpu_cores * 50,  # $50/month per core
            'memory': requirements.memory_gb * 5,    # $5/month per GB RAM
            'storage': requirements.storage_gb * 0.1, # $0.10/month per GB SSD
            'network': requirements.network_mbps * 10, # $10/month per Mbps
            'vector_db': requirements.vector_db_memory_gb * 10  # $10/month per GB vector DB
        }
        
        costs['total'] = sum(costs.values())
        return costs
```

### Auto-Scaling Policies

#### Predictive Scaling
```python
# predictive_scaling.py
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class PredictiveScaler:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.historical_data = {}
        self.prediction_window_hours = 2
        self.scale_up_threshold = 0.7
        self.scale_down_threshold = 0.3
    
    def collect_metrics(self, metrics: Dict[str, float], timestamp: datetime):
        """Collect historical metrics for prediction."""
        if 'historical' not in self.historical_data:
            self.historical_data['historical'] = []
        
        data_point = {'timestamp': timestamp, **metrics}
        self.historical_data['historical'].append(data_point)
        
        # Keep only last 30 days of data
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.historical_data['historical'] = [
            dp for dp in self.historical_data['historical'] 
            if dp['timestamp'] > cutoff
        ]
    
    def train_prediction_models(self):
        """Train prediction models on historical data."""
        if len(self.historical_data.get('historical', [])) < 48:  # Need at least 48 hours
            return False
        
        df = pd.DataFrame(self.historical_data['historical'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Features for prediction
        features = ['hour', 'day_of_week', 'day_of_month']
        
        # Metrics to predict
        metrics_to_predict = ['cpu_usage', 'memory_usage', 'request_rate', 'search_latency']
        
        for metric in metrics_to_predict:
            if metric not in df.columns:
                continue
            
            # Prepare data
            X = df[features].values
            y = df[metric].values
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[metric] = model
            self.scalers[metric] = scaler
        
        return True
    
    def predict_load(self, hours_ahead: int = 2) -> Dict[str, float]:
        """Predict load for specified hours ahead."""
        if not self.models:
            return {}
        
        future_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        # Create feature vector
        features = np.array([[
            future_time.hour,
            future_time.weekday(),
            future_time.day
        ]])
        
        predictions = {}
        for metric, model in self.models.items():
            if metric in self.scalers:
                features_scaled = self.scalers[metric].transform(features)
                prediction = model.predict(features_scaled)[0]
                predictions[metric] = max(0, prediction)  # Ensure non-negative
        
        return predictions
    
    def calculate_scaling_decision(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate scaling decision based on current and predicted metrics."""
        predictions = self.predict_load(self.prediction_window_hours)
        
        if not predictions:
            return {'action': 'none', 'reason': 'insufficient_data'}
        
        # Analyze key metrics
        key_metrics = ['cpu_usage', 'memory_usage', 'request_rate']
        current_max_usage = max([current_metrics.get(m, 0) for m in key_metrics])
        predicted_max_usage = max([predictions.get(m, 0) for m in key_metrics])
        
        # Decision logic
        if predicted_max_usage > self.scale_up_threshold:
            # Calculate required scale factor
            scale_factor = math.ceil(predicted_max_usage / self.scale_up_threshold)
            return {
                'action': 'scale_up',
                'factor': scale_factor,
                'reason': f'predicted_usage_{predicted_max_usage:.2f}',
                'predictions': predictions
            }
        elif current_max_usage < self.scale_down_threshold and predicted_max_usage < self.scale_down_threshold:
            return {
                'action': 'scale_down',
                'factor': 0.5,  # Scale down by half
                'reason': f'low_usage_current_{current_max_usage:.2f}_predicted_{predicted_max_usage:.2f}',
                'predictions': predictions
            }
        else:
            return {
                'action': 'none',
                'reason': 'within_thresholds',
                'current_usage': current_max_usage,
                'predicted_usage': predicted_max_usage
            }
    
    def apply_scaling_decision(self, decision: Dict[str, Any]) -> bool:
        """Apply scaling decision to the cluster."""
        if decision['action'] == 'none':
            return True
        
        try:
            if decision['action'] == 'scale_up':
                return self.scale_up(decision['factor'])
            elif decision['action'] == 'scale_down':
                return self.scale_down(decision['factor'])
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
        
        return True
    
    def scale_up(self, factor: float) -> bool:
        """Scale up the deployment."""
        # Implementation would interact with Kubernetes API or cloud provider
        logger.info(f"Scaling up by factor {factor}")
        # kubectl scale deployment codeweaver --replicas=new_count
        return True
    
    def scale_down(self, factor: float) -> bool:
        """Scale down the deployment."""
        # Implementation would interact with Kubernetes API or cloud provider
        logger.info(f"Scaling down by factor {factor}")
        # kubectl scale deployment codeweaver --replicas=new_count
        return True
```

## Cost Optimization

### Resource Optimization Strategies

#### Cost Analysis and Optimization
```python
# cost_optimizer.py
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ResourceCost:
    resource_type: str
    current_usage: float
    allocated: float
    cost_per_unit: float
    potential_savings: float

class CostOptimizer:
    def __init__(self):
        self.cost_per_resource = {
            'cpu_core_hour': 0.048,     # AWS m5.large equivalent
            'memory_gb_hour': 0.0096,   # $0.0096 per GB per hour
            'storage_gb_month': 0.10,   # SSD storage
            'network_gb': 0.09,         # Data transfer
            'vector_db_gb_hour': 0.012  # Managed vector DB cost
        }
        
        self.optimization_strategies = [
            'rightsizing',
            'reserved_instances',
            'spot_instances',
            'storage_optimization',
            'network_optimization'
        ]
    
    def analyze_costs(self, usage_data: Dict[str, Any], 
                     billing_period_days: int = 30) -> Dict[str, Any]:
        """Analyze current costs and identify optimization opportunities."""
        
        current_costs = self.calculate_current_costs(usage_data, billing_period_days)
        optimization_opportunities = self.identify_optimizations(usage_data)
        
        return {
            'current_costs': current_costs,
            'total_monthly_cost': sum(current_costs.values()),
            'optimization_opportunities': optimization_opportunities,
            'potential_monthly_savings': sum(op['potential_savings'] for op in optimization_opportunities),
            'recommendations': self.generate_recommendations(optimization_opportunities)
        }
    
    def calculate_current_costs(self, usage_data: Dict[str, Any], 
                               billing_period_days: int) -> Dict[str, float]:
        """Calculate current resource costs."""
        hours_in_period = billing_period_days * 24
        
        costs = {
            'compute': (
                usage_data.get('avg_cpu_cores', 0) * hours_in_period * 
                self.cost_per_resource['cpu_core_hour']
            ),
            'memory': (
                usage_data.get('avg_memory_gb', 0) * hours_in_period * 
                self.cost_per_resource['memory_gb_hour']
            ),
            'storage': (
                usage_data.get('storage_gb', 0) * 
                self.cost_per_resource['storage_gb_month'] * 
                (billing_period_days / 30)
            ),
            'network': (
                usage_data.get('network_gb_transferred', 0) * 
                self.cost_per_resource['network_gb']
            ),
            'vector_database': (
                usage_data.get('vector_db_memory_gb', 0) * hours_in_period * 
                self.cost_per_resource['vector_db_gb_hour']
            )
        }
        
        return costs
    
    def identify_optimizations(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Rightsizing opportunities
        cpu_utilization = usage_data.get('avg_cpu_utilization', 0)
        memory_utilization = usage_data.get('avg_memory_utilization', 0)
        
        if cpu_utilization < 0.3:  # Less than 30% CPU utilization
            rightsizing_savings = self.calculate_rightsizing_savings(
                usage_data.get('avg_cpu_cores', 0),
                cpu_utilization,
                'cpu'
            )
            opportunities.append({
                'type': 'rightsizing',
                'resource': 'cpu',
                'current_utilization': cpu_utilization,
                'recommended_reduction': 0.5,  # Reduce by 50%
                'potential_savings': rightsizing_savings,
                'confidence': 'high' if cpu_utilization < 0.2 else 'medium'
            })
        
        if memory_utilization < 0.4:  # Less than 40% memory utilization
            rightsizing_savings = self.calculate_rightsizing_savings(
                usage_data.get('avg_memory_gb', 0),
                memory_utilization,
                'memory'
            )
            opportunities.append({
                'type': 'rightsizing',
                'resource': 'memory',
                'current_utilization': memory_utilization,
                'recommended_reduction': 0.3,  # Reduce by 30%
                'potential_savings': rightsizing_savings,
                'confidence': 'high' if memory_utilization < 0.3 else 'medium'
            })
        
        # Reserved instance opportunities
        stable_usage_hours = usage_data.get('stable_usage_hours', 0)
        if stable_usage_hours > 24 * 30 * 0.7:  # More than 70% uptime
            ri_savings = self.calculate_reserved_instance_savings(usage_data)
            opportunities.append({
                'type': 'reserved_instances',
                'resource': 'compute',
                'stable_hours': stable_usage_hours,
                'potential_savings': ri_savings,
                'confidence': 'high'
            })
        
        # Spot instance opportunities
        fault_tolerant_workload = usage_data.get('fault_tolerant', False)
        if fault_tolerant_workload:
            spot_savings = self.calculate_spot_instance_savings(usage_data)
            opportunities.append({
                'type': 'spot_instances',
                'resource': 'compute',
                'workload_compatibility': 'high',
                'potential_savings': spot_savings,
                'confidence': 'medium'
            })
        
        # Storage optimization
        storage_utilization = usage_data.get('storage_utilization', 0)
        if storage_utilization < 0.6:  # Less than 60% storage utilization
            storage_savings = self.calculate_storage_optimization_savings(usage_data)
            opportunities.append({
                'type': 'storage_optimization',
                'resource': 'storage',
                'current_utilization': storage_utilization,
                'optimization_type': 'compression_and_cleanup',
                'potential_savings': storage_savings,
                'confidence': 'high'
            })
        
        return opportunities
    
    def calculate_rightsizing_savings(self, current_amount: float, 
                                    utilization: float, 
                                    resource_type: str) -> float:
        """Calculate potential savings from rightsizing."""
        if utilization >= 0.7:  # Already well-utilized
            return 0
        
        # Calculate optimal size based on utilization + buffer
        optimal_size = current_amount * (utilization + 0.2)  # 20% buffer
        reduction = current_amount - optimal_size
        
        cost_key = f"{resource_type}_{'core' if resource_type == 'cpu' else 'gb'}_hour"
        monthly_hours = 24 * 30
        
        return reduction * monthly_hours * self.cost_per_resource.get(cost_key, 0)
    
    def calculate_reserved_instance_savings(self, usage_data: Dict[str, Any]) -> float:
        """Calculate savings from reserved instances (30-50% discount)."""
        current_compute_cost = (
            usage_data.get('avg_cpu_cores', 0) * 24 * 30 * 
            self.cost_per_resource['cpu_core_hour']
        )
        return current_compute_cost * 0.4  # 40% savings
    
    def calculate_spot_instance_savings(self, usage_data: Dict[str, Any]) -> float:
        """Calculate savings from spot instances (60-90% discount)."""
        current_compute_cost = (
            usage_data.get('avg_cpu_cores', 0) * 24 * 30 * 
            self.cost_per_resource['cpu_core_hour']
        )
        return current_compute_cost * 0.7  # 70% savings
    
    def calculate_storage_optimization_savings(self, usage_data: Dict[str, Any]) -> float:
        """Calculate savings from storage optimization."""
        current_storage_cost = (
            usage_data.get('storage_gb', 0) * 
            self.cost_per_resource['storage_gb_month']
        )
        
        # Storage optimization can reduce costs by 20-40%
        optimization_factor = 0.3  # 30% reduction
        return current_storage_cost * optimization_factor
    
    def generate_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Sort opportunities by potential savings
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x['potential_savings'], 
            reverse=True
        )
        
        for opportunity in sorted_opportunities[:5]:  # Top 5 opportunities
            if opportunity['type'] == 'rightsizing':
                recommendations.append(
                    f"Reduce {opportunity['resource']} allocation by "
                    f"{opportunity['recommended_reduction']*100:.0f}% "
                    f"(Current utilization: {opportunity['current_utilization']*100:.0f}%) "
                    f"- Potential monthly savings: ${opportunity['potential_savings']:.2f}"
                )
            elif opportunity['type'] == 'reserved_instances':
                recommendations.append(
                    f"Purchase reserved instances for stable workloads "
                    f"- Potential monthly savings: ${opportunity['potential_savings']:.2f}"
                )
            elif opportunity['type'] == 'spot_instances':
                recommendations.append(
                    f"Use spot instances for fault-tolerant workloads "
                    f"- Potential monthly savings: ${opportunity['potential_savings']:.2f}"
                )
            elif opportunity['type'] == 'storage_optimization':
                recommendations.append(
                    f"Optimize storage usage through compression and cleanup "
                    f"(Current utilization: {opportunity['current_utilization']*100:.0f}%) "
                    f"- Potential monthly savings: ${opportunity['potential_savings']:.2f}"
                )
        
        return recommendations
```

## Next Steps

1. **Assessment**: Evaluate your current and projected scale requirements
2. **Architecture Planning**: Choose appropriate deployment patterns and scaling strategies
3. **Implementation**: Deploy auto-scaling, monitoring, and optimization systems
4. **Performance Testing**: Validate performance under load and optimize bottlenecks
5. **Cost Monitoring**: Implement cost tracking and optimization processes

For complex scaling implementations and performance optimization, consider engaging with cloud architects and performance engineering specialists to ensure optimal results for your specific use case.