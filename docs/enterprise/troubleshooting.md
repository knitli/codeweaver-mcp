# Enterprise Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues, debugging procedures, and support processes for CodeWeaver enterprise deployments. It includes diagnostic tools, resolution procedures, and escalation processes.

## Troubleshooting Framework

### Systematic Approach

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                 Troubleshooting Process                     │
│                                                             │
│  1. Issue Identification    :material-arrow-right-circle:    2. Information Gathering  │
│     • Symptoms Analysis           • Logs Collection        │
│     • Impact Assessment           • Metrics Review         │
│     • Scope Definition            • Environment Check      │
│                                                             │
│  3. Root Cause Analysis     :material-arrow-right-circle:    4. Solution Implementation│
│     • Hypothesis Testing          • Fix Application        │
│     • System Investigation        • Validation             │
│     • Pattern Recognition         • Monitoring            │
│                                                             │
│  5. Documentation          :material-arrow-right-circle:     6. Prevention             │
│     • Incident Recording          • Process Improvement    │
│     • Knowledge Base Update       • Monitoring Enhancement │
│     • Lessons Learned             • Training Updates       │
└─────────────────────────────────────────────────────────────┘
```

## Common Issues and Solutions

### Application-Level Issues

#### Search Performance Problems

**Symptoms:**
- Search queries taking longer than expected (>2 seconds)
- Timeouts during search operations
- High CPU usage during searches
- Users reporting slow responses

**Diagnostic Steps:**
```bash
# 1. Check search performance metrics
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, rate(codeweaver_search_latency_seconds_bucket[5m]))"

# 2. Check vector database status
curl -s "http://qdrant:6333/collections" | jq '.'

# 3. Check cache hit rates
curl -s "http://prometheus:9090/api/v1/query?query=codeweaver_cache_hit_rate"

# 4. Review recent search queries
kubectl logs -l app=codeweaver --tail=100 | grep "search"
```

**Common Causes and Solutions:**

1. **Vector Database Performance Issues**
   ```bash
   # Check Qdrant performance
   curl -s "http://qdrant:6333/metrics" | grep -E "(search_time|index_time)"

   # Solution: Optimize collection configuration
   curl -X PUT "http://qdrant:6333/collections/codeweaver/config" \
        -H "Content-Type: application/json" \
        -d '{
          "optimizers_config": {
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 1000,
            "default_segment_number": 8,
            "max_segment_size_kb": 32768
          }
        }'
   ```

2. **Cache Miss Rate Too High**
   ```python
   # Check cache statistics
   import redis
   r = redis.Redis(host='redis', port=6379)
   info = r.info()
   print(f"Cache hit rate: {info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses']):.2%}")

   # Solution: Implement cache warming
   # See cache warming implementation in scalability.md
   ```

3. **Embedding Generation Bottleneck**
   ```bash
   # Check embedding API latency
   curl -w "@curl-format.txt" -s -o /dev/null \
        -X POST "https://api.voyageai.com/v1/embeddings" \
        -H "Authorization: Bearer $VOYAGE_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"input": ["test query"], "model": "voyage-code-2"}'

   # Solution: Implement local embedding cache or use batch processing
   ```

#### High Memory Usage

**Symptoms:**
- Memory usage consistently above 80%
- Out of memory errors in logs
- Pod restarts due to memory limits
- System slowdown

**Diagnostic Steps:**
```bash
# 1. Check memory usage by component
curl -s "http://prometheus:9090/api/v1/query?query=codeweaver_memory_usage_bytes"

# 2. Check pod memory limits and usage
kubectl top pods -l app=codeweaver

# 3. Analyze memory usage patterns
kubectl logs -l app=codeweaver | grep -i "memory\|oom"

# 4. Check for memory leaks
curl -s "http://localhost:8080/debug/pprof/heap" > heap.prof
go tool pprof heap.prof
```

**Common Causes and Solutions:**

1. **Vector Database Memory Growth**
   ```bash
   # Check Qdrant memory usage
   curl -s "http://qdrant:6333/metrics" | grep memory

   # Solution: Configure memory limits and optimize collections
   curl -X PUT "http://qdrant:6333/collections/codeweaver/config" \
        -H "Content-Type: application/json" \
        -d '{
          "wal_config": {
            "wal_capacity_mb": 128,
            "wal_segments_ahead": 2
          },
          "optimizers_config": {
            "memmap_threshold_kb": 2048
          }
        }'
   ```

2. **Cache Memory Overflow**
   ```bash
   # Check Redis memory usage
   redis-cli info memory

   # Solution: Configure memory limits and eviction policy
   redis-cli config set maxmemory 2gb
   redis-cli config set maxmemory-policy allkeys-lru
   ```

3. **Memory Leaks in Application**
   ```python
   # Monitor memory growth over time
   import psutil
   import time

   process = psutil.Process()
   while True:
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Memory usage: {memory_mb:.2f} MB")
       time.sleep(60)
   ```

#### Authentication and Authorization Issues

**Symptoms:**
- Users unable to access system
- Authentication failures in logs
- Authorization denied errors
- Token validation failures

**Diagnostic Steps:**
```bash
# 1. Check authentication service status
curl -s "http://auth-service:8080/health"

# 2. Review authentication logs
kubectl logs -l app=codeweaver | grep -i "auth\|token\|login"

# 3. Test token validation
curl -H "Authorization: Bearer $TOKEN" "http://codeweaver:8080/api/v1/health"

# 4. Check user permissions
curl -H "Authorization: Bearer $TOKEN" "http://codeweaver:8080/api/v1/user/permissions"
```

**Common Causes and Solutions:**

1. **Token Expiration**
   ```bash
   # Check token expiration
   echo $TOKEN | base64 -d | jq '.exp'

   # Solution: Implement token refresh mechanism
   curl -X POST "http://auth-service:8080/refresh" \
        -H "Content-Type: application/json" \
        -d '{"refresh_token": "'$REFRESH_TOKEN'"}'
   ```

2. **RBAC Configuration Issues**
   ```bash
   # Check user roles and permissions
   kubectl get rolebindings -n codeweaver
   kubectl describe rolebinding codeweaver-users -n codeweaver

   # Solution: Update role bindings
   kubectl apply -f - <<EOF
   apiVersion: rbac.authorization.k8s.io/v1
   kind: RoleBinding
   metadata:
     name: codeweaver-users
     namespace: codeweaver
   subjects:
   - kind: User
     name: user@company.com
     apiGroup: rbac.authorization.k8s.io
   roleRef:
     kind: Role
     name: codeweaver-user
     apiGroup: rbac.authorization.k8s.io
   EOF
   ```

### Infrastructure-Level Issues

#### Database Connectivity Problems

**Symptoms:**
- Connection timeouts to vector database
- Database connection pool exhaustion
- Query execution failures
- Data inconsistency

**Diagnostic Steps:**
```bash
# 1. Test database connectivity
curl -f "http://qdrant:6333/health" || echo "Qdrant unreachable"

# 2. Check connection pool status
curl -s "http://codeweaver:8080/metrics" | grep connection_pool

# 3. Review database logs
kubectl logs -l app=qdrant --tail=100

# 4. Check network connectivity
kubectl exec -it codeweaver-pod -- nc -zv qdrant 6333
```

**Common Causes and Solutions:**

1. **Connection Pool Exhaustion**
   ```python
   # Monitor connection pool usage
   from qdrant_client import QdrantClient

   client = QdrantClient(url="http://qdrant:6333")
   # Configure connection limits
   client._client.timeout = 60
   client._client.limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
   ```

2. **Database Performance Issues**
   ```bash
   # Check Qdrant performance metrics
   curl -s "http://qdrant:6333/metrics" | grep -E "(duration|throughput)"

   # Solution: Optimize Qdrant configuration
   # See qdrant-performance.yaml in scalability.md
   ```

3. **Network Partitions**
   ```bash
   # Check network policies
   kubectl get networkpolicies -n codeweaver

   # Test inter-pod connectivity
   kubectl exec -it codeweaver-pod -- ping qdrant-service
   ```

#### Load Balancer Issues

**Symptoms:**
- Requests failing to reach application
- Uneven load distribution
- Health check failures
- SSL/TLS certificate errors

**Diagnostic Steps:**
```bash
# 1. Check load balancer status
kubectl get ingress -n codeweaver
kubectl describe ingress codeweaver-ingress -n codeweaver

# 2. Check service endpoints
kubectl get endpoints codeweaver-service -n codeweaver

# 3. Test health checks
curl -f "http://codeweaver:8080/health"

# 4. Check SSL certificate
openssl s_client -connect codeweaver.company.com:443 -servername codeweaver.company.com
```

**Common Causes and Solutions:**

1. **Unhealthy Backend Instances**
   ```bash
   # Check pod health
   kubectl get pods -l app=codeweaver -o wide

   # Fix unhealthy pods
   kubectl delete pod -l app=codeweaver --field-selector status.phase=Failed
   ```

2. **SSL Certificate Issues**
   ```bash
   # Check certificate expiration
   echo | openssl s_client -servername codeweaver.company.com -connect codeweaver.company.com:443 2>/dev/null | openssl x509 -noout -dates

   # Renew certificate with cert-manager
   kubectl delete certificate codeweaver-tls -n codeweaver
   ```

#### Kubernetes-Specific Issues

**Symptoms:**
- Pods stuck in pending state
- Resource quota exceeded
- Persistent volume mount failures
- Service discovery issues

**Diagnostic Steps:**
```bash
# 1. Check pod status
kubectl get pods -n codeweaver -o wide

# 2. Check resource usage
kubectl top nodes
kubectl top pods -n codeweaver

# 3. Check events
kubectl get events -n codeweaver --sort-by='.lastTimestamp'

# 4. Check resource quotas
kubectl describe resourcequota -n codeweaver
```

**Common Causes and Solutions:**

1. **Resource Constraints**
   ```bash
   # Check node capacity
   kubectl describe nodes | grep -E "(Name:|Allocatable:|Allocated resources:)" -A 5

   # Solution: Scale nodes or adjust resource requests
   kubectl scale deployment codeweaver --replicas=2 -n codeweaver
   ```

2. **Persistent Volume Issues**
   ```bash
   # Check PV status
   kubectl get pv,pvc -n codeweaver

   # Check storage class
   kubectl get storageclass

   # Solution: Fix PV binding issues
   kubectl patch pv pv-name -p '{"spec":{"claimRef":null}}'
   ```

## Diagnostic Tools and Scripts

### Automated Health Check Script

```bash
#!/bin/bash
# health-check.sh - Comprehensive health check for CodeWeaver

set -e

echo "=== CodeWeaver Health Check ==="
echo "Timestamp: $(date)"
echo

# Configuration
NAMESPACE="codeweaver"
PROMETHEUS_URL="http://prometheus:9090"
QDRANT_URL="http://qdrant:6333"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        return 1
    fi
}

warning_status() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
}

# 1. Kubernetes Health Checks
echo "1. Kubernetes Infrastructure"
echo "----------------------------"

# Check namespace
kubectl get namespace $NAMESPACE >/dev/null 2>&1
check_status $? "Namespace $NAMESPACE exists"

# Check pods
UNHEALTHY_PODS=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
if [ $UNHEALTHY_PODS -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}: All pods are running"
else
    echo -e "${RED}✗ FAIL${NC}: $UNHEALTHY_PODS unhealthy pods found"
    kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
fi

# Check services
kubectl get services -n $NAMESPACE >/dev/null 2>&1
check_status $? "Services are accessible"

echo

# 2. Application Health Checks
echo "2. Application Health"
echo "---------------------"

# Check CodeWeaver health endpoint
if curl -f -s "$CODEWEAVER_URL/health" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}: CodeWeaver health endpoint responding"
else
    echo -e "${RED}✗ FAIL${NC}: CodeWeaver health endpoint not responding"
fi

# Check authentication
if curl -f -s -H "Authorization: Bearer $TEST_TOKEN" "$CODEWEAVER_URL/api/v1/health" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}: Authentication working"
else
    echo -e "${RED}✗ FAIL${NC}: Authentication not working"
fi

echo

# 3. Database Health Checks
echo "3. Database Health"
echo "------------------"

# Check Qdrant health
if curl -f -s "$QDRANT_URL/health" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}: Qdrant health endpoint responding"

    # Check collections
    COLLECTIONS=$(curl -s "$QDRANT_URL/collections" | jq -r '.result.collections | length' 2>/dev/null)
    if [ "$COLLECTIONS" -gt 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $COLLECTIONS collections found"
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: No collections found"
    fi
else
    echo -e "${RED}✗ FAIL${NC}: Qdrant not responding"
fi

echo

# 4. Performance Checks
echo "4. Performance Metrics"
echo "----------------------"

# Check response times
RESPONSE_TIME=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=histogram_quantile(0.95, rate(codeweaver_request_duration_seconds_bucket[5m]))" | jq -r '.data.result[0].value[1]' 2>/dev/null)
if [ "$RESPONSE_TIME" != "null" ] && [ "$RESPONSE_TIME" != "" ]; then
    if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l) )); then
        echo -e "${GREEN}✓ PASS${NC}: 95th percentile response time: ${RESPONSE_TIME}s"
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: High response time: ${RESPONSE_TIME}s"
    fi
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Unable to get response time metrics"
fi

# Check error rate
ERROR_RATE=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(codeweaver_requests_total{status=~\"5..\"}[5m]) / rate(codeweaver_requests_total[5m])" | jq -r '.data.result[0].value[1]' 2>/dev/null)
if [ "$ERROR_RATE" != "null" ] && [ "$ERROR_RATE" != "" ]; then
    if (( $(echo "$ERROR_RATE < 0.01" | bc -l) )); then
        echo -e "${GREEN}✓ PASS${NC}: Error rate: $(echo "$ERROR_RATE * 100" | bc -l)%"
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: High error rate: $(echo "$ERROR_RATE * 100" | bc -l)%"
    fi
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Unable to get error rate metrics"
fi

# Check cache hit rate
CACHE_HIT_RATE=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=avg(codeweaver_cache_hit_rate)" | jq -r '.data.result[0].value[1]' 2>/dev/null)
if [ "$CACHE_HIT_RATE" != "null" ] && [ "$CACHE_HIT_RATE" != "" ]; then
    if (( $(echo "$CACHE_HIT_RATE > 0.7" | bc -l) )); then
        echo -e "${GREEN}✓ PASS${NC}: Cache hit rate: $(echo "$CACHE_HIT_RATE * 100" | bc -l)%"
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: Low cache hit rate: $(echo "$CACHE_HIT_RATE * 100" | bc -l)%"
    fi
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Unable to get cache hit rate metrics"
fi

echo

# 5. Resource Usage Checks
echo "5. Resource Usage"
echo "-----------------"

# Check memory usage
kubectl top pods -n $NAMESPACE --no-headers | while read pod cpu memory; do
    memory_mb=$(echo $memory | sed 's/Mi//')
    if [ "$memory_mb" -gt 4000 ]; then
        echo -e "${YELLOW}⚠ WARNING${NC}: High memory usage in $pod: $memory"
    else
        echo -e "${GREEN}✓ PASS${NC}: Memory usage in $pod: $memory"
    fi
done

echo

# 6. Security Checks
echo "6. Security Status"
echo "------------------"

# Check certificate expiration
if command -v openssl >/dev/null 2>&1; then
    CERT_EXPIRY=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -checkend 2592000 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: SSL certificate valid for at least 30 days"
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: SSL certificate expires within 30 days"
    fi
fi

# Check for security updates
SECURITY_UPDATES=$(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u | wc -l)
echo -e "${GREEN}✓ INFO${NC}: Using $SECURITY_UPDATES unique container images"

echo
echo "=== Health Check Complete ==="
```

### Performance Analysis Script

```python
#!/usr/bin/env python3
# performance-analysis.py - Analyze CodeWeaver performance metrics

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse

class PerformanceAnalyzer:
    def __init__(self, prometheus_url: str, time_range: str = "1h"):
        self.prometheus_url = prometheus_url
        self.time_range = time_range

    def query_prometheus(self, query: str) -> Dict[str, Any]:
        """Query Prometheus and return results."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return {}

    def query_prometheus_range(self, query: str, step: str = "1m") -> Dict[str, Any]:
        """Query Prometheus range data."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)  # Default 1 hour range

        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "step": step
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus range: {e}")
            return {}

    def analyze_response_times(self) -> Dict[str, float]:
        """Analyze response time metrics."""
        queries = {
            "p50": "histogram_quantile(0.50, rate(codeweaver_request_duration_seconds_bucket[5m]))",
            "p95": "histogram_quantile(0.95, rate(codeweaver_request_duration_seconds_bucket[5m]))",
            "p99": "histogram_quantile(0.99, rate(codeweaver_request_duration_seconds_bucket[5m]))",
            "avg": "rate(codeweaver_request_duration_seconds_sum[5m]) / rate(codeweaver_request_duration_seconds_count[5m])"
        }

        results = {}
        for percentile, query in queries.items():
            result = self.query_prometheus(query)
            if result.get("data", {}).get("result"):
                value = float(result["data"]["result"][0]["value"][1])
                results[percentile] = value
            else:
                results[percentile] = 0.0

        return results

    def analyze_search_performance(self) -> Dict[str, Any]:
        """Analyze search-specific performance."""
        search_latency = self.query_prometheus(
            "histogram_quantile(0.95, rate(codeweaver_search_latency_seconds_bucket[5m]))"
        )

        search_volume = self.query_prometheus(
            "sum(rate(codeweaver_requests_total{endpoint=~'.*search.*'}[5m]))"
        )

        result_counts = self.query_prometheus(
            "avg(codeweaver_search_results_count)"
        )

        return {
            "latency_p95": float(search_latency.get("data", {}).get("result", [{}])[0].get("value", [0, 0])[1]),
            "search_rate": float(search_volume.get("data", {}).get("result", [{}])[0].get("value", [0, 0])[1]),
            "avg_result_count": float(result_counts.get("data", {}).get("result", [{}])[0].get("value", [0, 0])[1])
        }

    def analyze_error_rates(self) -> Dict[str, float]:
        """Analyze error rates by endpoint."""
        total_requests = self.query_prometheus(
            "sum(rate(codeweaver_requests_total[5m])) by (endpoint)"
        )

        error_requests = self.query_prometheus(
            "sum(rate(codeweaver_requests_total{status=~'5..'}[5m])) by (endpoint)"
        )

        error_rates = {}

        if total_requests.get("data", {}).get("result"):
            for result in total_requests["data"]["result"]:
                endpoint = result["metric"]["endpoint"]
                total_rate = float(result["value"][1])

                # Find corresponding error rate
                error_rate = 0.0
                if error_requests.get("data", {}).get("result"):
                    for error_result in error_requests["data"]["result"]:
                        if error_result["metric"]["endpoint"] == endpoint:
                            error_rate = float(error_result["value"][1])
                            break

                error_rates[endpoint] = (error_rate / total_rate) if total_rate > 0 else 0.0

        return error_rates

    def analyze_cache_performance(self) -> Dict[str, float]:
        """Analyze cache performance metrics."""
        cache_hit_rate = self.query_prometheus(
            "avg(codeweaver_cache_hit_rate) by (cache_type)"
        )

        cache_size = self.query_prometheus(
            "sum(codeweaver_cache_size_bytes) by (cache_type)"
        )

        results = {"hit_rates": {}, "sizes": {}}

        if cache_hit_rate.get("data", {}).get("result"):
            for result in cache_hit_rate["data"]["result"]:
                cache_type = result["metric"]["cache_type"]
                hit_rate = float(result["value"][1])
                results["hit_rates"][cache_type] = hit_rate

        if cache_size.get("data", {}).get("result"):
            for result in cache_size["data"]["result"]:
                cache_type = result["metric"]["cache_type"]
                size_bytes = float(result["value"][1])
                results["sizes"][cache_type] = size_bytes / (1024 * 1024)  # Convert to MB

        return results

    def analyze_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Analyze resource usage patterns."""
        memory_query = "avg(codeweaver_memory_usage_bytes) by (component)"
        cpu_query = "avg(rate(codeweaver_cpu_usage_percent[5m])) by (component)"

        memory_result = self.query_prometheus(memory_query)
        cpu_result = self.query_prometheus(cpu_query)

        resources = {"memory": {}, "cpu": {}}

        if memory_result.get("data", {}).get("result"):
            for result in memory_result["data"]["result"]:
                component = result["metric"]["component"]
                memory_gb = float(result["value"][1]) / (1024 ** 3)
                resources["memory"][component] = memory_gb

        if cpu_result.get("data", {}).get("result"):
            for result in cpu_result["data"]["result"]:
                component = result["metric"]["component"]
                cpu_percent = float(result["value"][1])
                resources["cpu"][component] = cpu_percent

        return resources

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        print("Collecting performance metrics...")

        response_times = self.analyze_response_times()
        search_performance = self.analyze_search_performance()
        error_rates = self.analyze_error_rates()
        cache_performance = self.analyze_cache_performance()
        resource_usage = self.analyze_resource_usage()

        report = f"""
CodeWeaver Performance Analysis Report
=====================================
Generated: {datetime.utcnow().isoformat()}
Time Range: {self.time_range}

Response Time Analysis
---------------------
50th Percentile: {response_times['p50']:.3f}s
95th Percentile: {response_times['p95']:.3f}s
99th Percentile: {response_times['p99']:.3f}s
Average: {response_times['avg']:.3f}s

Search Performance
-----------------
95th Percentile Latency: {search_performance['latency_p95']:.3f}s
Search Rate: {search_performance['search_rate']:.2f} req/s
Average Result Count: {search_performance['avg_result_count']:.1f}

Error Rates by Endpoint
----------------------"""

        for endpoint, rate in error_rates.items():
            report += f"\n{endpoint}: {rate:.3%}"

        report += f"""

Cache Performance
----------------"""

        for cache_type, hit_rate in cache_performance['hit_rates'].items():
            size_mb = cache_performance['sizes'].get(cache_type, 0)
            report += f"\n{cache_type}: {hit_rate:.1%} hit rate, {size_mb:.1f} MB"

        report += f"""

Resource Usage
-------------"""

        for component, memory_gb in resource_usage['memory'].items():
            cpu_percent = resource_usage['cpu'].get(component, 0)
            report += f"\n{component}: {memory_gb:.2f} GB memory, {cpu_percent:.1f}% CPU"

        # Add recommendations
        recommendations = []

        if response_times['p95'] > 2.0:
            recommendations.append("• Response times are high - consider scaling or optimization")

        if search_performance['latency_p95'] > 1.0:
            recommendations.append("• Search latency is high - review vector database performance")

        for endpoint, rate in error_rates.items():
            if rate > 0.01:  # More than 1% error rate
                recommendations.append(f"• High error rate for {endpoint}: {rate:.1%}")

        for cache_type, hit_rate in cache_performance['hit_rates'].items():
            if hit_rate < 0.7:  # Less than 70% hit rate
                recommendations.append(f"• Low cache hit rate for {cache_type}: {hit_rate:.1%}")

        if recommendations:
            report += f"""

Recommendations
--------------"""
            for rec in recommendations:
                report += f"\n{rec}"
        else:
            report += f"""

Recommendations
--------------
• All metrics within acceptable ranges
• Continue monitoring for trends"""

        return report

def main():
    parser = argparse.ArgumentParser(description="Analyze CodeWeaver performance")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090",
                       help="Prometheus server URL")
    parser.add_argument("--time-range", default="1h",
                       help="Time range for analysis")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    analyzer = PerformanceAnalyzer(args.prometheus_url, args.time_range)
    report = analyzer.generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    main()
```

### Log Analysis Script

```bash
#!/bin/bash
# log-analysis.sh - Analyze CodeWeaver logs for issues

set -e

# Configuration
NAMESPACE="codeweaver"
TIME_RANGE="1h"
OUTPUT_DIR="/tmp/codeweaver-logs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== CodeWeaver Log Analysis ==="
echo "Time Range: $TIME_RANGE"
echo "Output Directory: $OUTPUT_DIR"
echo

# 1. Collect logs from all pods
echo "1. Collecting logs..."
kubectl get pods -n $NAMESPACE -o name | while read pod; do
    pod_name=$(echo $pod | cut -d'/' -f2)
    echo "  Collecting logs from $pod_name"
    kubectl logs $pod -n $NAMESPACE --since=$TIME_RANGE > "$OUTPUT_DIR/${pod_name}.log" 2>/dev/null || true
done

# 2. Analyze error patterns
echo
echo "2. Error Analysis"
echo "-----------------"

# Count errors by type
echo "Error counts by type:"
cat "$OUTPUT_DIR"/*.log | grep -i error | \
    sed 's/.*error[[:space:]]*://i' | \
    cut -d' ' -f1 | \
    sort | uniq -c | sort -nr | head -10

echo
echo "Critical errors:"
cat "$OUTPUT_DIR"/*.log | grep -i "critical\|fatal" | head -5

# 3. Performance issues
echo
echo "3. Performance Issues"
echo "--------------------"

# Slow requests
echo "Slow requests (>2s):"
cat "$OUTPUT_DIR"/*.log | grep -E "response_time.*[2-9]\.[0-9]|response_time.*[0-9]{2,}" | head -5

# Memory warnings
echo
echo "Memory warnings:"
cat "$OUTPUT_DIR"/*.log | grep -i "memory\|oom" | head -3

# 4. Search-specific issues
echo
echo "4. Search Analysis"
echo "------------------"

# Failed searches
echo "Failed searches:"
FAILED_SEARCHES=$(cat "$OUTPUT_DIR"/*.log | grep -c "search.*failed\|search.*error" 2>/dev/null || echo "0")
echo "  Failed searches: $FAILED_SEARCHES"

# Slow searches
echo "Slow searches (>1s):"
cat "$OUTPUT_DIR"/*.log | grep -E "search.*duration.*[1-9]\.[0-9]" | head -3

# 5. Authentication issues
echo
echo "5. Authentication Issues"
echo "-----------------------"

# Auth failures
AUTH_FAILURES=$(cat "$OUTPUT_DIR"/*.log | grep -c "auth.*fail\|unauthorized\|forbidden" 2>/dev/null || echo "0")
echo "Authentication failures: $AUTH_FAILURES"

if [ "$AUTH_FAILURES" -gt 0 ]; then
    echo "Recent auth failures:"
    cat "$OUTPUT_DIR"/*.log | grep -i "auth.*fail\|unauthorized\|forbidden" | tail -3
fi

# 6. Database connectivity
echo
echo "6. Database Issues"
echo "------------------"

# Database connection errors
DB_ERRORS=$(cat "$OUTPUT_DIR"/*.log | grep -c "database.*error\|connection.*failed\|qdrant.*error" 2>/dev/null || echo "0")
echo "Database errors: $DB_ERRORS"

if [ "$DB_ERRORS" -gt 0 ]; then
    echo "Recent database errors:"
    cat "$OUTPUT_DIR"/*.log | grep -i "database.*error\|connection.*failed\|qdrant.*error" | tail -3
fi

# 7. Generate summary report
echo
echo "7. Summary Report"
echo "----------------"

TOTAL_LOGS=$(cat "$OUTPUT_DIR"/*.log | wc -l)
TOTAL_ERRORS=$(cat "$OUTPUT_DIR"/*.log | grep -c -i error 2>/dev/null || echo "0")
TOTAL_WARNINGS=$(cat "$OUTPUT_DIR"/*.log | grep -c -i warning 2>/dev/null || echo "0")

cat > "$OUTPUT_DIR/summary.txt" << EOF
CodeWeaver Log Analysis Summary
==============================
Analysis Date: $(date)
Time Range: $TIME_RANGE
Total Log Lines: $TOTAL_LOGS
Total Errors: $TOTAL_ERRORS
Total Warnings: $TOTAL_WARNINGS
Error Rate: $(echo "scale=2; $TOTAL_ERRORS * 100 / $TOTAL_LOGS" | bc -l)%

Key Issues Found:
- Authentication failures: $AUTH_FAILURES
- Database errors: $DB_ERRORS
- Failed searches: $FAILED_SEARCHES

Top Error Types:
$(cat "$OUTPUT_DIR"/*.log | grep -i error | sed 's/.*error[[:space:]]*://i' | cut -d' ' -f1 | sort | uniq -c | sort -nr | head -5)

Recommendations:
$(if [ "$AUTH_FAILURES" -gt 10 ]; then echo "- Review authentication configuration"; fi)
$(if [ "$DB_ERRORS" -gt 5 ]; then echo "- Check database connectivity and performance"; fi)
$(if [ "$FAILED_SEARCHES" -gt 20 ]; then echo "- Investigate search performance issues"; fi)
$(if [ $(echo "$TOTAL_ERRORS * 100 / $TOTAL_LOGS > 5" | bc -l) -eq 1 ]; then echo "- High error rate requires immediate attention"; fi)
EOF

echo "Summary saved to $OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"
```

## Backup and Recovery Procedures

### Backup Strategy Implementation

```bash
#!/bin/bash
# backup-manager.sh - Comprehensive backup and recovery for CodeWeaver

set -e

# Configuration
BACKUP_DIR="/opt/backups/codeweaver"
S3_BUCKET="codeweaver-backups"
RETENTION_DAYS=30
NAMESPACE="codeweaver"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Timestamp for this backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"
mkdir -p "$BACKUP_PATH"

echo "=== CodeWeaver Backup Started ==="
echo "Timestamp: $TIMESTAMP"
echo "Backup Path: $BACKUP_PATH"
echo

backup_kubernetes_resources() {
    echo "1. Backing up Kubernetes resources..."

    # Backup deployments
    kubectl get deployments -n $NAMESPACE -o yaml > "$BACKUP_PATH/deployments.yaml"

    # Backup services
    kubectl get services -n $NAMESPACE -o yaml > "$BACKUP_PATH/services.yaml"

    # Backup configmaps
    kubectl get configmaps -n $NAMESPACE -o yaml > "$BACKUP_PATH/configmaps.yaml"

    # Backup secrets (be careful with sensitive data)
    kubectl get secrets -n $NAMESPACE -o yaml > "$BACKUP_PATH/secrets.yaml"

    # Backup ingress
    kubectl get ingress -n $NAMESPACE -o yaml > "$BACKUP_PATH/ingress.yaml"

    # Backup persistent volume claims
    kubectl get pvc -n $NAMESPACE -o yaml > "$BACKUP_PATH/pvc.yaml"

    echo "  ✓ Kubernetes resources backed up"
}

backup_vector_database() {
    echo "2. Backing up vector database..."

    # Get list of collections
    COLLECTIONS=$(curl -s "http://qdrant:6333/collections" | jq -r '.result.collections[].name')

    for collection in $COLLECTIONS; do
        echo "  Backing up collection: $collection"

        # Create snapshot
        SNAPSHOT_NAME="${collection}_${TIMESTAMP}"
        curl -X POST "http://qdrant:6333/collections/$collection/snapshots" \
             -H "Content-Type: application/json" \
             -d "{\"name\": \"$SNAPSHOT_NAME\"}"

        # Wait for snapshot creation
        sleep 5

        # Download snapshot
        curl -X GET "http://qdrant:6333/collections/$collection/snapshots/$SNAPSHOT_NAME/download" \
             -o "$BACKUP_PATH/${collection}.snapshot"

        # Verify snapshot
        if [ -f "$BACKUP_PATH/${collection}.snapshot" ]; then
            echo "    ✓ Collection $collection backed up"
        else
            echo "    ✗ Failed to backup collection $collection"
        fi
    done
}

backup_configuration() {
    echo "3. Backing up configuration files..."

    # Backup application configuration
    if [ -d "/opt/codeweaver/config" ]; then
        tar -czf "$BACKUP_PATH/app-config.tar.gz" /opt/codeweaver/config/
        echo "  ✓ Application configuration backed up"
    fi

    # Backup monitoring configuration
    if [ -d "/opt/monitoring/config" ]; then
        tar -czf "$BACKUP_PATH/monitoring-config.tar.gz" /opt/monitoring/config/
        echo "  ✓ Monitoring configuration backed up"
    fi

    # Backup SSL certificates
    if [ -d "/etc/ssl/codeweaver" ]; then
        tar -czf "$BACKUP_PATH/ssl-certs.tar.gz" /etc/ssl/codeweaver/
        echo "  ✓ SSL certificates backed up"
    fi
}

backup_metrics_data() {
    echo "4. Backing up metrics data..."

    # Export Prometheus data (last 7 days)
    END_TIME=$(date -u +%s)
    START_TIME=$((END_TIME - 604800))  # 7 days ago

    # Export key metrics
    METRICS=(
        "codeweaver_requests_total"
        "codeweaver_search_latency_seconds"
        "codeweaver_memory_usage_bytes"
        "codeweaver_cache_hit_rate"
    )

    for metric in "${METRICS[@]}"; do
        curl -s "http://prometheus:9090/api/v1/query_range" \
             -d "query=$metric" \
             -d "start=$START_TIME" \
             -d "end=$END_TIME" \
             -d "step=300" \
             --data-urlencode > "$BACKUP_PATH/${metric}.json"
    done

    echo "  ✓ Metrics data exported"
}

create_manifest() {
    echo "5. Creating backup manifest..."

    cat > "$BACKUP_PATH/manifest.json" << EOF
{
    "backup_id": "$TIMESTAMP",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$(kubectl get deployment codeweaver -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d':' -f2)",
    "components": {
        "kubernetes_resources": true,
        "vector_database": true,
        "configuration": true,
        "metrics_data": true
    },
    "collections": [$(echo "$COLLECTIONS" | tr '\n' ',' | sed 's/,$//' | sed 's/\([^,]*\)/"\1"/g')],
    "size_mb": $(du -sm "$BACKUP_PATH" | cut -f1)
}
EOF

    echo "  ✓ Backup manifest created"
}

upload_to_s3() {
    echo "6. Uploading to S3..."

    # Compress backup
    tar -czf "$BACKUP_DIR/codeweaver-backup-$TIMESTAMP.tar.gz" -C "$BACKUP_PATH" .

    # Upload to S3
    aws s3 cp "$BACKUP_DIR/codeweaver-backup-$TIMESTAMP.tar.gz" \
        "s3://$S3_BUCKET/backups/codeweaver-backup-$TIMESTAMP.tar.gz"

    # Upload manifest separately for easy listing
    aws s3 cp "$BACKUP_PATH/manifest.json" \
        "s3://$S3_BUCKET/manifests/manifest-$TIMESTAMP.json"

    echo "  ✓ Backup uploaded to S3"
}

cleanup_old_backups() {
    echo "7. Cleaning up old backups..."

    # Remove local backups older than retention period
    find "$BACKUP_DIR" -name "codeweaver-backup-*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} +

    # Remove old S3 backups
    aws s3 ls "s3://$S3_BUCKET/backups/" | while read -r line; do
        backup_date=$(echo $line | awk '{print $1}')
        backup_file=$(echo $line | awk '{print $4}')

        # Calculate age in days
        backup_epoch=$(date -d "$backup_date" +%s)
        current_epoch=$(date +%s)
        age_days=$(((current_epoch - backup_epoch) / 86400))

        if [ $age_days -gt $RETENTION_DAYS ]; then
            aws s3 rm "s3://$S3_BUCKET/backups/$backup_file"
            aws s3 rm "s3://$S3_BUCKET/manifests/manifest-${backup_file%.tar.gz}.json" 2>/dev/null || true
        fi
    done

    echo "  ✓ Old backups cleaned up"
}

# Execute backup steps
backup_kubernetes_resources
backup_vector_database
backup_configuration
backup_metrics_data
create_manifest
upload_to_s3
cleanup_old_backups

echo
echo "=== Backup Completed Successfully ==="
echo "Backup ID: $TIMESTAMP"
echo "Local Path: $BACKUP_PATH"
echo "S3 Location: s3://$S3_BUCKET/backups/codeweaver-backup-$TIMESTAMP.tar.gz"
echo
```

### Disaster Recovery Procedures

```bash
#!/bin/bash
# disaster-recovery.sh - Disaster recovery for CodeWeaver

set -e

# Configuration
BACKUP_ID=${1:-$(date +"%Y%m%d_%H%M%S")}
S3_BUCKET="codeweaver-backups"
RESTORE_DIR="/opt/restore/codeweaver"
NAMESPACE="codeweaver"

if [ -z "$BACKUP_ID" ]; then
    echo "Usage: $0 <backup_id>"
    echo "Available backups:"
    aws s3 ls "s3://$S3_BUCKET/backups/" | grep codeweaver-backup | tail -10
    exit 1
fi

echo "=== CodeWeaver Disaster Recovery ==="
echo "Backup ID: $BACKUP_ID"
echo "Restore Directory: $RESTORE_DIR"
echo

# Create restore directory
mkdir -p "$RESTORE_DIR"

download_backup() {
    echo "1. Downloading backup from S3..."

    # Download backup archive
    aws s3 cp "s3://$S3_BUCKET/backups/codeweaver-backup-$BACKUP_ID.tar.gz" \
        "$RESTORE_DIR/codeweaver-backup-$BACKUP_ID.tar.gz"

    # Extract backup
    tar -xzf "$RESTORE_DIR/codeweaver-backup-$BACKUP_ID.tar.gz" -C "$RESTORE_DIR"

    # Download and verify manifest
    aws s3 cp "s3://$S3_BUCKET/manifests/manifest-$BACKUP_ID.json" \
        "$RESTORE_DIR/manifest.json"

    echo "  ✓ Backup downloaded and extracted"
}

verify_backup() {
    echo "2. Verifying backup integrity..."

    # Check manifest
    if [ ! -f "$RESTORE_DIR/manifest.json" ]; then
        echo "  ✗ Manifest file missing"
        exit 1
    fi

    # Verify components
    COMPONENTS=$(jq -r '.components | keys[]' "$RESTORE_DIR/manifest.json")
    for component in $COMPONENTS; do
        case $component in
            "kubernetes_resources")
                if [ -f "$RESTORE_DIR/deployments.yaml" ]; then
                    echo "    ✓ Kubernetes resources present"
                else
                    echo "    ✗ Kubernetes resources missing"
                fi
                ;;
            "vector_database")
                if ls "$RESTORE_DIR"/*.snapshot >/dev/null 2>&1; then
                    echo "    ✓ Vector database snapshots present"
                else
                    echo "    ✗ Vector database snapshots missing"
                fi
                ;;
            "configuration")
                if [ -f "$RESTORE_DIR/app-config.tar.gz" ]; then
                    echo "    ✓ Configuration backup present"
                else
                    echo "    ✗ Configuration backup missing"
                fi
                ;;
            "metrics_data")
                if ls "$RESTORE_DIR"/*.json >/dev/null 2>&1; then
                    echo "    ✓ Metrics data present"
                else
                    echo "    ✗ Metrics data missing"
                fi
                ;;
        esac
    done
}

prepare_environment() {
    echo "3. Preparing recovery environment..."

    # Stop existing services
    kubectl scale deployment codeweaver --replicas=0 -n $NAMESPACE 2>/dev/null || true
    kubectl scale statefulset qdrant --replicas=0 -n $NAMESPACE 2>/dev/null || true

    # Wait for pods to terminate
    echo "  Waiting for pods to terminate..."
    kubectl wait --for=delete pod -l app=codeweaver -n $NAMESPACE --timeout=300s 2>/dev/null || true
    kubectl wait --for=delete pod -l app=qdrant -n $NAMESPACE --timeout=300s 2>/dev/null || true

    echo "  ✓ Environment prepared"
}

restore_configuration() {
    echo "4. Restoring configuration..."

    # Restore application configuration
    if [ -f "$RESTORE_DIR/app-config.tar.gz" ]; then
        tar -xzf "$RESTORE_DIR/app-config.tar.gz" -C /
        echo "    ✓ Application configuration restored"
    fi

    # Restore monitoring configuration
    if [ -f "$RESTORE_DIR/monitoring-config.tar.gz" ]; then
        tar -xzf "$RESTORE_DIR/monitoring-config.tar.gz" -C /
        echo "    ✓ Monitoring configuration restored"
    fi

    # Restore SSL certificates
    if [ -f "$RESTORE_DIR/ssl-certs.tar.gz" ]; then
        tar -xzf "$RESTORE_DIR/ssl-certs.tar.gz" -C /
        echo "    ✓ SSL certificates restored"
    fi
}

restore_kubernetes_resources() {
    echo "5. Restoring Kubernetes resources..."

    # Restore ConfigMaps first
    if [ -f "$RESTORE_DIR/configmaps.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/configmaps.yaml"
        echo "    ✓ ConfigMaps restored"
    fi

    # Restore Secrets
    if [ -f "$RESTORE_DIR/secrets.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/secrets.yaml"
        echo "    ✓ Secrets restored"
    fi

    # Restore PVCs
    if [ -f "$RESTORE_DIR/pvc.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/pvc.yaml"
        echo "    ✓ Persistent Volume Claims restored"
    fi

    # Wait for PVCs to be bound
    kubectl wait --for=condition=Bound pvc --all -n $NAMESPACE --timeout=300s

    # Restore Services
    if [ -f "$RESTORE_DIR/services.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/services.yaml"
        echo "    ✓ Services restored"
    fi

    # Restore Ingress
    if [ -f "$RESTORE_DIR/ingress.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/ingress.yaml"
        echo "    ✓ Ingress restored"
    fi
}

restore_vector_database() {
    echo "6. Restoring vector database..."

    # Start Qdrant first
    kubectl scale statefulset qdrant --replicas=3 -n $NAMESPACE
    kubectl wait --for=condition=Ready pod -l app=qdrant -n $NAMESPACE --timeout=300s

    # Wait for Qdrant to be healthy
    echo "  Waiting for Qdrant to be ready..."
    while ! curl -f -s "http://qdrant:6333/health" >/dev/null 2>&1; do
        sleep 5
    done

    # Restore collections from snapshots
    for snapshot in "$RESTORE_DIR"/*.snapshot; do
        if [ -f "$snapshot" ]; then
            collection_name=$(basename "$snapshot" .snapshot)
            echo "    Restoring collection: $collection_name"

            # Upload snapshot
            curl -X POST "http://qdrant:6333/collections/$collection_name/snapshots/upload" \
                 -F "snapshot=@$snapshot"

            # Recover from snapshot
            snapshot_filename=$(basename "$snapshot")
            curl -X PUT "http://qdrant:6333/collections/$collection_name/snapshots/recover" \
                 -H "Content-Type: application/json" \
                 -d "{\"location\": \"$snapshot_filename\", \"priority\": \"snapshot\"}"

            echo "      ✓ Collection $collection_name restored"
        fi
    done
}

restore_application() {
    echo "7. Restoring application..."

    # Restore Deployments
    if [ -f "$RESTORE_DIR/deployments.yaml" ]; then
        kubectl apply -f "$RESTORE_DIR/deployments.yaml"
        echo "    ✓ Deployments restored"
    fi

    # Wait for deployments to be ready
    kubectl wait --for=condition=Available deployment --all -n $NAMESPACE --timeout=600s

    echo "    ✓ Application restored and running"
}

verify_recovery() {
    echo "8. Verifying recovery..."

    # Check application health
    if curl -f -s "http://codeweaver:8080/health" >/dev/null 2>&1; then
        echo "    ✓ CodeWeaver health check passed"
    else
        echo "    ✗ CodeWeaver health check failed"
        return 1
    fi

    # Check vector database health
    if curl -f -s "http://qdrant:6333/health" >/dev/null 2>&1; then
        echo "    ✓ Qdrant health check passed"
    else
        echo "    ✗ Qdrant health check failed"
        return 1
    fi

    # Check collections
    COLLECTIONS=$(curl -s "http://qdrant:6333/collections" | jq -r '.result.collections[].name' | wc -l)
    echo "    ✓ $COLLECTIONS collections restored"

    # Test search functionality
    if curl -f -s -X POST "http://codeweaver:8080/api/v1/search" \
           -H "Content-Type: application/json" \
           -d '{"query": "test", "limit": 1}' >/dev/null 2>&1; then
        echo "    ✓ Search functionality verified"
    else
        echo "    ✗ Search functionality failed"
        return 1
    fi
}

generate_recovery_report() {
    echo "9. Generating recovery report..."

    RECOVERY_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    cat > "$RESTORE_DIR/recovery-report.json" << EOF
{
    "recovery_id": "REC-$(date +%Y%m%d%H%M%S)",
    "backup_id": "$BACKUP_ID",
    "recovery_time": "$RECOVERY_TIME",
    "status": "completed",
    "components_restored": {
        "kubernetes_resources": true,
        "vector_database": true,
        "configuration": true,
        "application": true
    },
    "collections_restored": $COLLECTIONS,
    "verification_results": {
        "health_checks": "passed",
        "search_functionality": "passed"
    }
}
EOF

    echo "  ✓ Recovery report generated"
}

# Execute recovery steps
download_backup
verify_backup
prepare_environment
restore_configuration
restore_kubernetes_resources
restore_vector_database
restore_application
verify_recovery
generate_recovery_report

echo
echo "=== Disaster Recovery Completed Successfully ==="
echo "Recovery Report: $RESTORE_DIR/recovery-report.json"
echo "CodeWeaver is now operational with data restored from backup $BACKUP_ID"
echo
```

## Support Escalation Procedures

### Support Tier Structure

#### Tier 1: First-Line Support
- **Scope**: Basic troubleshooting, known issues, user guidance
- **Response Time**: 4 hours during business hours
- **Escalation Criteria**: Technical complexity beyond basic troubleshooting

#### Tier 2: Technical Support
- **Scope**: Advanced troubleshooting, configuration issues, performance problems
- **Response Time**: 8 hours for normal issues, 2 hours for critical
- **Escalation Criteria**: Requires code changes or architectural decisions

#### Tier 3: Engineering Support
- **Scope**: Code-level issues, architectural problems, emergency response
- **Response Time**: 24 hours for normal issues, 1 hour for critical
- **Escalation Criteria**: Product bugs, security vulnerabilities, system design issues

### Escalation Decision Matrix

```python
# escalation_matrix.py
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass

class IssueCategory(Enum):
    USER_ERROR = "user_error"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUG = "bug"
    INFRASTRUCTURE = "infrastructure"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SupportTier(Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    EMERGENCY = "emergency"

@dataclass
class EscalationRule:
    category: IssueCategory
    severity: Severity
    keywords: List[str]
    tier: SupportTier
    response_time_hours: int

class EscalationMatrix:
    def __init__(self):
        self.rules = [
            # Critical issues - immediate escalation
            EscalationRule(
                category=IssueCategory.SECURITY,
                severity=Severity.CRITICAL,
                keywords=["security", "breach", "vulnerability", "unauthorized"],
                tier=SupportTier.EMERGENCY,
                response_time_hours=1
            ),

            # High severity performance issues
            EscalationRule(
                category=IssueCategory.PERFORMANCE,
                severity=Severity.HIGH,
                keywords=["timeout", "slow", "latency", "performance"],
                tier=SupportTier.TIER_2,
                response_time_hours=2
            ),

            # System bugs
            EscalationRule(
                category=IssueCategory.BUG,
                severity=Severity.HIGH,
                keywords=["crash", "error", "exception", "bug"],
                tier=SupportTier.TIER_3,
                response_time_hours=4
            ),

            # Configuration issues
            EscalationRule(
                category=IssueCategory.CONFIGURATION,
                severity=Severity.MEDIUM,
                keywords=["config", "setup", "deployment"],
                tier=SupportTier.TIER_2,
                response_time_hours=8
            ),

            # User errors
            EscalationRule(
                category=IssueCategory.USER_ERROR,
                severity=Severity.LOW,
                keywords=["how to", "usage", "question"],
                tier=SupportTier.TIER_1,
                response_time_hours=24
            )
        ]

    def determine_escalation(self, issue_description: str,
                           reported_severity: Severity) -> Dict[str, Any]:
        """Determine appropriate escalation tier and response time."""
        issue_lower = issue_description.lower()

        # Check for emergency keywords first
        emergency_keywords = ["down", "outage", "critical", "security breach"]
        if any(keyword in issue_lower for keyword in emergency_keywords):
            return {
                "tier": SupportTier.EMERGENCY,
                "response_time_hours": 1,
                "reason": "Emergency keywords detected"
            }

        # Find matching rules
        matching_rules = []
        for rule in self.rules:
            if any(keyword in issue_lower for keyword in rule.keywords):
                matching_rules.append(rule)

        if not matching_rules:
            # Default escalation based on reported severity
            if reported_severity == Severity.CRITICAL:
                return {
                    "tier": SupportTier.TIER_3,
                    "response_time_hours": 4,
                    "reason": "Critical severity with no specific category match"
                }
            elif reported_severity == Severity.HIGH:
                return {
                    "tier": SupportTier.TIER_2,
                    "response_time_hours": 8,
                    "reason": "High severity with no specific category match"
                }
            else:
                return {
                    "tier": SupportTier.TIER_1,
                    "response_time_hours": 24,
                    "reason": "Default tier assignment"
                }

        # Choose the highest priority matching rule
        highest_priority_rule = min(matching_rules,
                                  key=lambda r: (r.tier.value, r.response_time_hours))

        return {
            "tier": highest_priority_rule.tier,
            "response_time_hours": highest_priority_rule.response_time_hours,
            "category": highest_priority_rule.category,
            "reason": f"Matched rule for {highest_priority_rule.category.value}"
        }

# Usage example
matrix = EscalationMatrix()
escalation = matrix.determine_escalation(
    "CodeWeaver search is very slow and timing out",
    Severity.HIGH
)
print(f"Escalate to: {escalation['tier'].value}")
print(f"Response time: {escalation['response_time_hours']} hours")
```

## Knowledge Base Articles

### Common Issue Templates

#### Search Performance Issues
**Issue**: Search queries are slow or timing out

**Symptoms**:
- Search operations taking >2 seconds
- Timeout errors in logs
- High CPU usage during searches

**Root Cause Analysis**:
1. Check vector database performance metrics
2. Analyze cache hit rates
3. Review query complexity and patterns
4. Examine resource utilization

**Resolution Steps**:
1. Optimize vector database configuration
2. Implement or improve caching
3. Scale resources if needed
4. Review and optimize queries

**Prevention**:
- Monitor search performance continuously
- Implement cache warming strategies
- Set up alerting for performance degradation

#### Memory Issues
**Issue**: High memory usage causing performance problems or OOM kills

**Symptoms**:
- Memory usage consistently above 80%
- Pod restarts due to memory limits
- OOM killer messages in logs

**Root Cause Analysis**:
1. Identify memory consumption by component
2. Check for memory leaks
3. Analyze cache usage patterns
4. Review resource allocations

**Resolution Steps**:
1. Clear caches to free immediate memory
2. Optimize cache size limits
3. Scale horizontally if needed
4. Fix memory leaks if identified

**Prevention**:
- Set appropriate memory limits
- Monitor memory usage trends
- Implement cache eviction policies
- Regular memory leak testing

## Next Steps

1. **Implement Diagnostic Tools**: Deploy health check and analysis scripts
2. **Train Support Teams**: Ensure teams understand troubleshooting procedures
3. **Test Recovery Procedures**: Regularly test backup and recovery processes
4. **Update Documentation**: Keep troubleshooting guides current with system changes
5. **Monitor and Improve**: Continuously improve procedures based on incident learnings

This troubleshooting guide provides a comprehensive foundation for resolving issues in enterprise CodeWeaver deployments. Regular updates based on new issues and resolutions will keep it effective and current.
