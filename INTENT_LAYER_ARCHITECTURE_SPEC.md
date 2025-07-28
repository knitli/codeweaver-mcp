# Intent Layer Architecture Specification

## 🎯 Architecture Overview

**Purpose**: Transform complex CodeWeaver architecture into intent-based LLM interface while preserving developer extensibility.

**Core Strategy**: Strategy pattern with intelligent orchestration → 4 tools → 1-2 tools, intent-driven.

## 📋 Component Specifications

### 1. Intent Orchestrator (Core Engine)
```yaml
Location: src/codeweaver/intent/orchestrator.py
Responsibility: "Single entry point for intent processing"
Dependencies: [ServicesManager, ExtensibilityManager, StrategyEngine]
Protocols: [IntentOrchestrator, ServiceProvider]
```

**Key Methods**:
- `process_intent(intent_text, context) → IntentResult`
- `get_supported_intents() → list[IntentInfo]`
- `health_check() → ServiceHealth`

### 2. Intent Parser Engine
```yaml
Location: src/codeweaver/intent/parsing/
Components:
  - pattern_matcher.py    # Regex + heuristic patterns
  - ml_fallback.py       # Optional ML-based parsing
  - intent_classifier.py # Rule-based classification
  - confidence_scorer.py # Multi-factor confidence scoring
```

**ParsedIntent Data Structure**:
```python
@dataclass
class ParsedIntent:
    intent_type: IntentType  # SEARCH | UNDERSTAND | ANALYZE | INDEX
    primary_target: str      # Main focus ("authentication", "database")
    scope: Scope            # FILE | MODULE | PROJECT | SYSTEM
    complexity: Complexity  # SIMPLE | MODERATE | COMPLEX | ADAPTIVE
    confidence: float       # 0.0-1.0 confidence score
    filters: dict[str, Any] # Additional constraints
    metadata: dict[str, Any] # Parser metadata
```

### 3. Strategy Engine
```yaml
Location: src/codeweaver/intent/strategies/
Registry: StrategyRegistry (follows factory pattern)
Base: BaseIntentStrategy (protocol)
Strategies:
  - SimpleSearchStrategy     # Direct search_code mapping
  - AnalysisWorkflowStrategy # Multi-step orchestration
  - IndexingStrategy        # Smart codebase indexing
  - AdaptiveStrategy        # Escalation-based execution
```

**Strategy Selection Matrix**:
```yaml
SimpleSearch:    confidence >0.7 & complexity=SIMPLE & intent=SEARCH
AnalysisWorkflow: complexity=COMPLEX | scope=PROJECT | intent=UNDERSTAND
Indexing:        intent=INDEX | no_existing_index
Adaptive:        confidence <0.6 | complexity=ADAPTIVE | fallback
```

### 4. Workflow Engine
```yaml
Location: src/codeweaver/intent/workflows/
Components:
  - step_executor.py    # Individual workflow step execution
  - orchestration.py    # Multi-step coordination
  - result_aggregator.py # Result synthesis
  - error_recovery.py   # Fallback handling
```

**Workflow Steps**:
- `SearchStep`: Execute search operations
- `AnalysisStep`: Perform structural analysis
- `SummaryStep`: Generate comprehensive summaries
- `ValidationStep`: Validate results quality

## 🔗 Integration Architecture

### Layer Integration Stack
```
┌─────────────────────────────┐
│     Intent MCP Tools        │ ← New: 1-2 intent-based tools
├─────────────────────────────┤
│     Intent Layer           │ ← New: Orchestrator + Strategies
├─────────────────────────────┤
│     Existing Server Layer  │ ← Integrate: tool handlers
├─────────────────────────────┤
│     Services Layer         │ ← Leverage: dependency injection
├─────────────────────────────┤
│     Middleware Layer       │ ← Use: chunking, filtering
├─────────────────────────────┤
│     Factory Layer          │ ← Extend: strategy registration
├─────────────────────────────┤
│     Implementation Layer   │ ← Unchanged: providers, backends
└─────────────────────────────┘
```

### Service Integration Points
```python
# Intent Layer → Services Layer Integration
class IntentOrchestrator(BaseServiceProvider):
    def __init__(self, services_manager: ServicesManager):
        self.services = services_manager
        self.strategies = StrategyRegistry(services_manager)
    
    async def process_intent(self, intent_text: str, context: dict) -> IntentResult:
        # Service context injection
        enhanced_context = {
            **context,
            "chunking_service": self.services.get_chunking_service(),
            "filtering_service": self.services.get_filtering_service(),
            **self.services.create_service_context()
        }
        
        # Strategy execution with services
        strategy = await self.strategies.select_strategy(intent_text)
        return await strategy.execute(intent_text, enhanced_context)
```

### Factory Integration Pattern
```python
# Register intent strategies via factory system
from codeweaver.factories.extensibility_manager import ExtensibilityManager

class IntentExtensibilityManager:
    def __init__(self, base_manager: ExtensibilityManager):
        self.base = base_manager
        self.strategy_registry = StrategyRegistry()
    
    async def get_intent_orchestrator(self) -> IntentOrchestrator:
        services_manager = await self.base.get_services_manager()
        return IntentOrchestrator(services_manager, self.strategy_registry)
```

## ⚙️ Configuration Schema

### Intent Layer Configuration
```toml
[intent]
enabled = true
default_strategy = "adaptive"
confidence_threshold = 0.6
max_execution_time = 30.0

[intent.parsing]
use_ml_fallback = false
pattern_matching = true
confidence_scoring = "multi_factor"

[intent.strategies]
simple_search = { enabled = true, timeout = 5.0 }
analysis_workflow = { enabled = true, timeout = 15.0, max_steps = 5 }
indexing = { enabled = true, auto_index = true }
adaptive = { enabled = true, escalation_threshold = 0.5 }

[intent.performance]
cache_results = true
cache_ttl = 3600
strategy_learning = true
performance_tracking = true
```

### Strategy-Specific Configuration
```toml
[intent.strategies.analysis_workflow]
max_search_results = 50
enable_structural_search = true
summary_generation = true
result_ranking = true

[intent.strategies.adaptive]
escalation_steps = ["simple_search", "analysis_workflow", "full_analysis"]
confidence_boost_threshold = 0.3
max_escalations = 3
```

## 🛠️ Implementation Guidelines

### 1. Protocol Definitions
```python
# Core protocols following CodeWeaver patterns
class IntentStrategy(Protocol):
    strategy_name: str
    confidence_threshold: float
    
    async def can_handle(self, intent: ParsedIntent) -> float: ...
    async def execute(self, intent: ParsedIntent, context: dict) -> IntentResult: ...
    async def estimate_execution_time(self, intent: ParsedIntent) -> float: ...

class IntentParser(Protocol):
    async def parse(self, intent_text: str) -> ParsedIntent: ...
    async def get_confidence(self, intent_text: str) -> float: ...

class WorkflowStep(Protocol):
    step_name: str
    async def execute(self, context: dict) -> StepResult: ...
    async def can_execute(self, context: dict) -> bool: ...
```

### 2. Error Handling Pattern
```python
class IntentOrchestrator:
    async def process_intent(self, intent_text: str) -> IntentResult:
        try:
            # Parse intent
            parsed_intent = await self.parser.parse(intent_text)
            
            # Select strategy
            strategy = await self.strategy_engine.select(parsed_intent)
            
            # Execute with fallback chain
            return await self._execute_with_fallback(strategy, parsed_intent)
            
        except IntentParsingError as e:
            return IntentResult.error(f"Unable to understand intent: {e}")
        except StrategyExecutionError as e:
            return await self._fallback_execution(intent_text, e)
        except Exception as e:
            logger.exception("Unexpected intent processing error")
            return IntentResult.error("Internal error processing intent")
```

### 3. Testing Patterns
```python
class TestIntentOrchestrator:
    async def test_simple_intent_flow(self):
        """Test simple intent → direct tool mapping."""
        result = await self.orchestrator.process_intent("find auth functions")
        
        assert result.success
        assert result.strategy_used == "simple_search"
        assert len(result.data["results"]) > 0
        assert result.execution_time < 2.0
    
    async def test_complex_intent_flow(self):
        """Test complex intent → multi-step workflow."""
        result = await self.orchestrator.process_intent(
            "understand complete authentication architecture"
        )
        
        assert result.success
        assert result.strategy_used == "analysis_workflow"
        assert "summary" in result.data
        assert len(result.metadata["workflow_steps"]) > 1
```

## 📊 Performance Specifications

### Response Time Targets
```yaml
Simple Intents:    <2s   (95th percentile)
Complex Intents:   <5s   (95th percentile)
Adaptive Intents:  <10s  (99th percentile)
Strategy Selection: <100ms (mean)
Intent Parsing:    <50ms  (mean)
```

### Resource Management
```yaml
Memory Usage:      <50MB additional overhead
CPU Overhead:      <10% during idle
Cache Hit Rate:    >80% for repeated intents
Strategy Cache:    1000 entries max, 1h TTL
```

### Quality Metrics
```yaml
Intent Recognition: >90% accuracy
Strategy Selection: >95% appropriate choice
Fallback Success:   >85% recovery rate
User Satisfaction:  <2 clarification requests/session
```

## 🔄 Migration Strategy

### Phase 1: Core Implementation
1. Intent orchestrator + basic strategies
2. Integration with existing server
3. Parallel deployment (both old + new tools)

### Phase 2: Strategy Expansion  
1. Add complex workflow strategies
2. ML-based intent parsing (optional)
3. Performance optimization

### Phase 3: Full Deployment
1. Deprecate old direct tools
2. Intent-only interface
3. Developer strategy customization

### Backward Compatibility
```python
# Developer flag to expose original tools
[intent]
expose_debug_tools = true  # Exposes original 4 tools
developer_mode = true      # Additional debugging info
strategy_override = false  # Allow manual strategy selection
```

## ✅ Success Criteria

**User Experience**:
- ✅ Single tool handles 90%+ of use cases
- ✅ Natural language queries work intuitively  
- ✅ Response time <5s for complex queries
- ✅ Graceful error messages for unclear intents

**Developer Experience**:
- ✅ Custom strategies can be added via plugins
- ✅ Full configuration control over behavior
- ✅ Debug mode exposes internal operations
- ✅ Backward compatibility with existing tools

**System Integration**:
- ✅ Leverages existing services/middleware/factories
- ✅ Follows established CodeWeaver patterns
- ✅ No breaking changes to current architecture
- ✅ Performance overhead <10%