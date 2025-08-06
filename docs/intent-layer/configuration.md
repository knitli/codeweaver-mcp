<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Configuration

The Intent Layer provides extensive configuration options for customizing how natural language requests are parsed, routed, and executed. This guide covers all configuration patterns, from basic setup to advanced customization.

## Basic Configuration

### Service Configuration

The Intent Layer is configured through the CodeWeaver services system. Add intent layer configuration to your `codeweaver.toml` or environment variables:

```toml
[services.intent]
provider = "intent_orchestrator"
default_strategy = "adaptive"
confidence_threshold = 0.6
max_execution_time = 30.0
cache_ttl = 3600

[services.intent.pattern_matching]
enabled = true
use_nlp_fallback = false
custom_patterns_file = "intent_patterns.toml"

[services.intent.performance]
circuit_breaker_enabled = true
circuit_breaker_threshold = 5
circuit_breaker_reset_time = 60.0
max_concurrent_intents = 10
```

### Environment Variables

```bash
# Core intent service
CW_INTENT_PROVIDER=intent_orchestrator
CW_INTENT_DEFAULT_STRATEGY=adaptive
CW_INTENT_CONFIDENCE_THRESHOLD=0.6

# Pattern matching
CW_INTENT_PATTERN_MATCHING=true
CW_INTENT_CUSTOM_PATTERNS_FILE=intent_patterns.toml

# Performance
CW_INTENT_MAX_CONCURRENT=10
CW_INTENT_CACHE_TTL=3600
```

## Pattern-Based Configuration

### Custom Search Patterns

Create custom patterns to recognize domain-specific terminology and improve intent recognition:

```toml
# intent_patterns.toml
[patterns.search]
# Authentication patterns
auth_patterns = [
    "auth(?:entication)?\\s+(.+)",
    "login\\s+(.+)",
    "credential\\s+(.+)",
    "(?:user|session)\\s+management\\s+(.+)"
]

# Security patterns  
security_patterns = [
    "vulnerability\\s+(?:in\\s+)?(.+)",
    "security\\s+(?:issue|flaw|hole)\\s+(?:in\\s+)?(.+)",
    "(?:sql\\s+injection|xss|csrf)\\s+(?:in\\s+)?(.+)",
    "permission\\s+(?:check|validation)\\s+(.+)"
]

# Performance patterns
performance_patterns = [
    "slow\\s+(.+)",
    "performance\\s+(?:issue|problem)\\s+(?:in\\s+)?(.+)",
    "optimize\\s+(.+)",
    "bottleneck\\s+(?:in\\s+)?(.+)"
]

[patterns.understand]
# Architecture patterns
architecture_patterns = [
    "how\\s+does\\s+(.+)\\s+work",
    "explain\\s+(?:the\\s+)?(.+)\\s+(?:system|architecture|design)",
    "understand\\s+(?:the\\s+)?(.+)\\s+(?:flow|process|workflow)"
]

# Relationship patterns
relationship_patterns = [
    "what\\s+calls\\s+(.+)",
    "dependencies\\s+(?:of\\s+)?(.+)",
    "(?:how\\s+)?(.+)\\s+(?:connects?\\s+to|integrates?\\s+with)\\s+(.+)"
]

[patterns.analyze]
# Code quality patterns
quality_patterns = [
    "analyze\\s+(?:the\\s+)?(.+)\\s+(?:quality|maintainability)",
    "technical\\s+debt\\s+(?:in\\s+)?(.+)",
    "code\\s+smell\\s+(?:in\\s+)?(.+)",
    "refactor(?:ing)?\\s+(?:opportunities\\s+(?:in\\s+)?)?(.+)"
]

# Bug patterns
bug_patterns = [
    "bug(?:s)?\\s+(?:in\\s+)?(.+)",
    "issue(?:s)?\\s+(?:with\\s+)?(.+)",
    "problem(?:s)?\\s+(?:in\\s+)?(.+)",
    "error(?:s)?\\s+(?:in\\s+)?(.+)"
]
```

### Pattern Priority and Confidence

Configure how patterns are weighted and scored:

```toml
[pattern_scoring]
# Base confidence scores for pattern types
search_base_confidence = 0.7
understand_base_confidence = 0.6
analyze_base_confidence = 0.8

# Pattern-specific multipliers
auth_multiplier = 1.2        # Authentication patterns get higher confidence
security_multiplier = 1.3    # Security patterns prioritized
performance_multiplier = 1.1 # Performance patterns slightly boosted

# Confidence thresholds
minimum_confidence = 0.4
excellent_confidence = 0.9
fallback_threshold = 0.3
```

## NLP Pattern Configuration

### Future NLP Integration

The Intent Layer is designed to support advanced NLP patterns. While currently using regex-based parsing, the architecture supports future NLP enhancements:

```toml
[services.intent.nlp]
# Future NLP configuration
enabled = false
model = "en_core_web_sm"
confidence_threshold = 0.8
fallback_to_patterns = true

# Entity recognition
enable_ner = false
custom_entities = ["FUNCTION_NAME", "CLASS_NAME", "MODULE_NAME"]

# Dependency parsing
enable_dependency_parsing = false
extract_relationships = false

# Semantic similarity
enable_semantic_matching = false
similarity_threshold = 0.7
```

### Hybrid Pattern Matching

Configure hybrid approaches that combine patterns with future NLP capabilities:

```toml
[pattern_combination]
# Strategy for combining pattern and NLP results
combination_strategy = "weighted_average"  # "max_confidence", "weighted_average", "ensemble"

# Weights for different approaches
pattern_weight = 0.7
nlp_weight = 0.3

# Conflict resolution
conflict_resolution = "higher_confidence"  # "pattern_priority", "nlp_priority", "higher_confidence"
```

## Strategy Configuration

### Strategy Selection

Configure how strategies are selected and prioritized:

```toml
[strategy_selection]
# Scoring algorithm
capability_weight = 0.7      # How well strategy handles intent type
performance_weight = 0.3     # Historical performance metrics

# Strategy-specific configuration
[strategies.simple_search]
enabled = true
priority = 0.8
max_results = 20
confidence_threshold = 0.7

[strategies.analysis_workflow]
enabled = true
priority = 0.9
max_steps = 5
step_timeout = 15.0
allow_partial_success = true

[strategies.adaptive]
enabled = true
priority = 0.1  # Fallback strategy
dynamic_approach = true
learning_enabled = true
```

### Custom Strategy Registration

Register custom strategies through configuration:

```toml
[custom_strategies]
# Security-focused strategy
[custom_strategies.security_analysis]
class_path = "myorg.strategies.SecurityAnalysisStrategy"
priority = 0.95
intent_types = ["ANALYZE"]
scope_types = ["PROJECT", "SYSTEM"]
specialization = "security"

# Performance analysis strategy  
[custom_strategies.performance_analysis]
class_path = "myorg.strategies.PerformanceAnalysisStrategy"
priority = 0.9
intent_types = ["ANALYZE", "UNDERSTAND"]
scope_types = ["MODULE", "PROJECT"]
specialization = "performance"
```

## Context Intelligence Configuration

### LLM Detection and Adaptation

Configure how the Intent Layer detects and adapts to different LLMs:

```toml
[services.context_intelligence]
provider = "context_intelligence"

# LLM identification
llm_identification_enabled = true
behavioral_fingerprinting = true

# Privacy settings
privacy_mode = "hash_identifiers"  # "strict", "hash_identifiers", "minimal"
session_timeout = 3600
max_concurrent_sessions = 100

# Context analysis
context_window_size = 4096
max_context_history = 50
enable_context_learning = true
context_similarity_threshold = 0.8
adaptive_context_sizing = true

# Model-specific optimizations
[context_intelligence.llm_profiles]
# Claude-specific configuration
claude = { context_preference = "detailed", format_preference = "structured" }
# GPT-4 specific configuration  
gpt4 = { context_preference = "concise", format_preference = "conversational" }
# Gemini specific configuration
gemini = { context_preference = "visual", format_preference = "hierarchical" }
```

### Implicit Learning Configuration

Configure behavioral learning and optimization:

```toml
[services.implicit_learning]
provider = "implicit_learning"

# Learning behavior
learning_enabled = true
pattern_recognition = true
success_tracking = true

# Pattern analysis
min_pattern_frequency = 3
pattern_confidence_threshold = 0.75
max_stored_patterns = 1000

# Success optimization
track_execution_time = true
track_result_quality = true
track_user_satisfaction = false  # Requires feedback mechanism

# Privacy and retention
anonymize_patterns = true
pattern_retention_days = 90
cleanup_interval_hours = 24
```

## Performance Configuration

### Caching Configuration

```toml
[services.intent.caching]
# Cache settings
cache_ttl = 3600              # 1 hour default TTL
max_cache_size = 10000        # Maximum cached items
cache_strategy = "lru"        # "lru", "lfu", "ttl"

# Cache keys
cache_by_input = true         # Cache by raw input
cache_by_parsed_intent = true # Cache by parsed intent structure
cache_by_context = false      # Don't cache by full context (too variable)

# Cache warming
enable_cache_warming = false
warmup_common_patterns = []
```

### Circuit Breaker Configuration

```toml
[services.intent.circuit_breaker]
enabled = true
failure_threshold = 5         # Failures before opening circuit
success_threshold = 3         # Successes needed to close circuit
timeout = 60.0               # Seconds before attempting reset
recovery_timeout = 300.0     # Seconds for full recovery

# Failure detection
timeout_as_failure = true
exception_as_failure = true
low_confidence_as_failure = false
```

### Concurrency Control

```toml
[services.intent.concurrency]
max_concurrent_intents = 10
max_concurrent_per_session = 3
queue_timeout = 30.0
rejection_policy = "drop_oldest"  # "drop_oldest", "drop_newest", "reject"

# Resource limits
max_memory_mb = 512
max_cpu_percent = 50
enable_resource_monitoring = true
```

## Monitoring and Telemetry

### Metrics Configuration

```toml
[services.intent.monitoring]
enable_performance_monitoring = true
enable_telemetry_tracking = true
enable_metrics_collection = true

# Metric collection
collect_execution_time = true
collect_confidence_scores = true
collect_strategy_usage = true
collect_error_rates = true

# Alerting thresholds
execution_time_warning = 10.0    # seconds
execution_time_critical = 30.0   # seconds
error_rate_warning = 0.05        # 5%
error_rate_critical = 0.15       # 15%
confidence_warning = 0.4         # Low confidence threshold
```

### Logging Configuration

```toml
[logging.intent]
level = "INFO"
format = "structured"

# What to log
log_parsed_intents = true
log_strategy_selection = true
log_execution_results = false    # Can be verbose
log_performance_metrics = true

# Privacy
anonymize_user_input = true
hash_session_ids = true
redact_sensitive_data = true
```

## Advanced Configuration

### Workflow Orchestration

```toml
[workflow_orchestration]
# Workflow settings
max_workflow_steps = 10
step_timeout = 30.0
workflow_timeout = 300.0
allow_partial_success = true

# Retry configuration
max_retries = 3
retry_backoff = "exponential"  # "linear", "exponential", "fixed"
retry_jitter = true

# Step dependencies
validate_dependencies = true
parallel_execution = true
max_parallel_steps = 3
```

### Extension Configuration

```toml
[extensions]
# Plugin discovery
scan_entry_points = true
scan_directories = ["plugins/", "extensions/"]
auto_register = true

# Plugin validation
validate_interfaces = true
require_metadata = true
sandbox_plugins = false

# Plugin lifecycle
auto_start = true
health_check_interval = 60.0
restart_on_failure = true
```

## Configuration Examples

### Development Environment

```toml
# Development-focused configuration
[services.intent]
provider = "intent_orchestrator"
default_strategy = "adaptive"
confidence_threshold = 0.4      # Lower threshold for experimentation
max_execution_time = 60.0       # Longer timeout for debugging

[services.intent.logging]
level = "DEBUG"
log_parsed_intents = true
log_strategy_selection = true
log_execution_results = true

[services.intent.performance]
circuit_breaker_enabled = false  # Disable for development
max_concurrent_intents = 5
```

### Production Environment

```toml
# Production-optimized configuration
[services.intent]
provider = "intent_orchestrator"
default_strategy = "simple_search"  # Prefer reliable strategies
confidence_threshold = 0.7          # Higher confidence required
max_execution_time = 15.0           # Strict timeout

[services.intent.performance]
circuit_breaker_enabled = true
circuit_breaker_threshold = 3       # Fail fast
max_concurrent_intents = 20
cache_ttl = 7200                   # Longer cache for stability

[services.intent.monitoring]
enable_performance_monitoring = true
enable_telemetry_tracking = true
collect_execution_time = true
collect_error_rates = true
```

### High-Security Environment

```toml
# Security-focused configuration
[services.intent]
provider = "intent_orchestrator"
confidence_threshold = 0.8    # High confidence required

[services.context_intelligence]
privacy_mode = "strict"       # Maximum privacy
session_timeout = 900         # 15 minute sessions
behavioral_fingerprinting = false

[services.implicit_learning]
learning_enabled = false      # Disable learning in secure environments
pattern_recognition = false

[logging.intent]
anonymize_user_input = true
hash_session_ids = true
redact_sensitive_data = true
```

## Next Steps

1. **[Custom Strategies](custom-strategies.md)**: Learn to build custom intent strategies
2. **[API Reference](api-reference.md)**: Detailed interface documentation  
3. **[Examples](examples.md)**: Practical configuration examples
4. **[Architecture](architecture.md)**: Understanding the technical implementation