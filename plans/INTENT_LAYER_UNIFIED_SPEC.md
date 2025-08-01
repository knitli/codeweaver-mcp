<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Unified Implementation Specification

## 🎯 Executive Summary

**Purpose**: Transform CodeWeaver's LLM interface from 4 complex tools to 1-2 intuitive, natural language tools while preserving full developer extensibility.

**Strategy**: Incremental implementation starting with essential features (alpha-ready) and enhancing with advanced capabilities.

**Target**: Alpha release with 90%+ intent recognition accuracy and <3s response times.

## 🏗️ Core Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Interface                           │
│  OLD: 4 tools → NEW: process_intent + get_capabilities    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Intent Layer                              │
│                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ Intent          │ │ Strategy        │ │ Workflow        ││
│  │ Orchestrator    │ │ Engine          │ │ Engine          ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
│           │                   │                   │         │
│           └───────────────────┼───────────────────┘         │
└─────────────────────────────┼─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Existing CodeWeaver Architecture              │
│  [Server Layer → Services → Middleware → Factory → Impl]   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles
1. **Protocol-Based**: Runtime-checkable protocols for all interfaces
2. **Services Integration**: Leverages existing ServicesManager
3. **Factory Extension**: Extends ExtensibilityManager patterns
4. **Configuration-Driven**: TOML hierarchy with environment overrides
5. **Incremental Enhancement**: Essential features first, advanced features later

## 📋 Component Specifications

### 1. Intent Orchestrator (Core Entry Point)

**Location**: `src/codeweaver/intent/orchestrator.py`

```python
from typing import Protocol
from codeweaver.cw_types import ServiceProvider, IntentResult
from codeweaver.services.manager import ServicesManager

class IntentOrchestrator(BaseServiceProvider):
    """Main orchestrator for intent processing with service integration."""

    def __init__(self, services_manager: ServicesManager):
        self.services = services_manager
        self.parser = IntentParserFactory.create()
        self.strategy_engine = StrategyEngine(services_manager)
        self.cache = IntentCache()

    async def process_intent(
        self,
        intent_text: str,
        context: dict[str, Any]
    ) -> IntentResult:
        """Main entry point for intent processing."""
        try:
            # Check cache first
            cached_result = await self.cache.get(intent_text)
            if cached_result:
                return cached_result

            # Parse intent
            parsed_intent = await self.parser.parse(intent_text)

            # Select and execute strategy
            strategy = await self.strategy_engine.select_strategy(parsed_intent)
            result = await strategy.execute(parsed_intent, context)

            # Cache successful results
            await self.cache.set(intent_text, result)
            return result

        except IntentParsingError as e:
            return IntentResult.error(f"Unable to understand intent: {e}")
        except StrategyExecutionError as e:
            return await self._fallback_execution(intent_text, context, e)
        except Exception as e:
            logger.exception("Unexpected intent processing error")
            return IntentResult.error("Internal error processing intent")

    async def get_supported_intents(self) -> list[IntentInfo]:
        """Return list of supported intent types and examples."""
        return await self.strategy_engine.get_supported_intents()
```

**Protocols**:
```python
class IntentOrchestrator(Protocol):
    async def process_intent(self, intent_text: str, context: dict) -> IntentResult: ...
    async def get_supported_intents(self) -> list[IntentInfo]: ...
    async def health_check(self) -> ServiceHealth: ...
```

### 2. Intent Parser Engine

**Location**: `src/codeweaver/intent/parsing/`

#### Core Parser Implementation
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

class PatternBasedParser:
    """Essential: Regex-based pattern matching for alpha release."""

    def __init__(self):
        self.patterns = self._load_intent_patterns()
        self.confidence_scorer = BasicConfidenceScorer()

    async def parse(self, intent_text: str) -> ParsedIntent:
        """Parse intent using pattern matching."""
        # Intent type detection
        intent_type = self._detect_intent_type(intent_text)

        # Target extraction
        primary_target = self._extract_target(intent_text)

        # Scope and complexity assessment
        scope = self._assess_scope(intent_text)
        complexity = self._assess_complexity(intent_text, scope)

        # Confidence scoring
        confidence = await self.confidence_scorer.score(
            intent_text, intent_type, primary_target
        )

        return ParsedIntent(
            intent_type=intent_type,
            primary_target=primary_target,
            scope=scope,
            complexity=complexity,
            confidence=confidence,
            filters=self._extract_filters(intent_text),
            metadata={"parser": "pattern_based", "patterns_matched": []}
        )
```

#### Enhanced Parser (Future Enhancement)
```python
class NLPEnhancedParser:
    """Enhancement: spaCy + domain models for better accuracy."""

    def __init__(self):
        self.nlp_pipeline = spacy.load("en_core_web_trf")
        self.domain_classifier = self._load_domain_model()
        self.confidence_scorer = MultifactorConfidenceScorer()
        self.fallback_parser = PatternBasedParser()

    async def parse(self, intent_text: str) -> ParsedIntent:
        """Enhanced parsing with NLP and domain knowledge."""
        try:
            # NLP processing
            doc = self.nlp_pipeline(intent_text)

            # Domain classification
            domain_result = await self.domain_classifier.classify(intent_text)

            # Enhanced confidence scoring
            confidence = await self.confidence_scorer.score(
                intent_text, doc, domain_result
            )

            return ParsedIntent(
                intent_type=domain_result.intent_type,
                primary_target=domain_result.primary_target,
                scope=domain_result.scope,
                complexity=domain_result.complexity,
                confidence=confidence,
                filters=domain_result.filters,
                metadata={
                    "parser": "nlp_enhanced",
                    "entities": [ent.text for ent in doc.ents],
                    "pos_tags": [(token.text, token.pos_) for token in doc]
                }
            )
        except Exception as e:
            logger.warning("NLP parsing failed, falling back to patterns: %s", e)
            return await self.fallback_parser.parse(intent_text)
```

### 3. Strategy Engine

**Location**: `src/codeweaver/intent/strategies/`

#### Strategy Registry
```python
class StrategyRegistry:
    """Registry for intent strategies following factory pattern."""

    def __init__(self, services_manager: ServicesManager):
        self.services = services_manager
        self.strategies = {}
        self.performance_tracker = StrategyPerformanceTracker()
        self._register_default_strategies()

    def register_strategy(
        self,
        name: str,
        strategy_class: type[IntentStrategy]
    ) -> None:
        """Register a strategy class."""
        self.strategies[name] = strategy_class

    async def select_strategy(self, parsed_intent: ParsedIntent) -> IntentStrategy:
        """Select best strategy based on intent and performance data."""
        candidates = []

        for name, strategy_class in self.strategies.items():
            strategy = strategy_class(self.services)
            can_handle_score = await strategy.can_handle(parsed_intent)

            if can_handle_score > 0.1:  # Minimum threshold
                performance_score = self.performance_tracker.get_score(name)
                final_score = (can_handle_score * 0.7) + (performance_score * 0.3)
                candidates.append((final_score, name, strategy))

        if not candidates:
            # Fallback to adaptive strategy
            return AdaptiveStrategy(self.services)

        # Return highest scoring strategy
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]
```

#### Core Strategies

**SimpleSearchStrategy** (Essential):
```python
class SimpleSearchStrategy:
    """Direct mapping to search_code for simple queries."""

    strategy_name = "simple_search"
    confidence_threshold = 0.7

    async def can_handle(self, intent: ParsedIntent) -> float:
        """Determine if this strategy can handle the intent."""
        if (intent.intent_type == IntentType.SEARCH and
            intent.complexity == Complexity.SIMPLE and
            intent.confidence > 0.7):
            return 0.95
        return 0.1

    async def execute(self, intent: ParsedIntent, context: dict) -> IntentResult:
        """Execute simple search directly."""
        from codeweaver.server import search_code_handler

        result = await search_code_handler(
            query=intent.primary_target,
            limit=10,
            rerank=True,
            **context
        )

        return IntentResult(
            success=True,
            data=result,
            metadata={
                "strategy": self.strategy_name,
                "workflow": "direct_search",
                "execution_time": context.get("execution_time", 0)
            }
        )
```

**AnalysisWorkflowStrategy** (Essential):
```python
class AnalysisWorkflowStrategy:
    """Multi-step analysis for complex understanding queries."""

    strategy_name = "analysis_workflow"
    confidence_threshold = 0.6

    async def can_handle(self, intent: ParsedIntent) -> float:
        """Determine if this strategy can handle the intent."""
        if (intent.intent_type == IntentType.UNDERSTAND and
            intent.complexity in [Complexity.COMPLEX, Complexity.MODERATE] and
            intent.scope in [Scope.PROJECT, Scope.SYSTEM]):
            return 0.92
        return 0.2

    async def execute(self, intent: ParsedIntent, context: dict) -> IntentResult:
        """Execute multi-step analysis workflow."""
        workflow_results = {}

        # Step 1: Initial search
        initial_search = await self._initial_search(intent, context)
        workflow_results["initial_search"] = initial_search

        # Step 2: Structural analysis (if applicable)
        if initial_search["language_detected"]:
            structural_search = await self._structural_analysis(
                intent, initial_search, context
            )
            workflow_results["structural_search"] = structural_search

        # Step 3: Summary generation
        summary = await self._generate_summary(intent, workflow_results)
        workflow_results["summary"] = summary

        return IntentResult(
            success=True,
            data=workflow_results,
            metadata={
                "strategy": self.strategy_name,
                "workflow": "multi_step_analysis",
                "workflow_steps": list(workflow_results.keys())
            }
        )
```

**AdaptiveStrategy** (Essential):
```python
class AdaptiveStrategy:
    """Fallback strategy with escalation and learning."""

    strategy_name = "adaptive"

    async def can_handle(self, intent: ParsedIntent) -> float:
        """Always can handle as fallback."""
        return 0.7 if intent.confidence < 0.6 else 0.3

    async def execute(self, intent: ParsedIntent, context: dict) -> IntentResult:
        """Adaptive execution with escalation."""
        escalation_path = []

        # Phase 1: Try simple search
        try:
            simple_strategy = SimpleSearchStrategy(self.services)
            result = await simple_strategy.execute(intent, context)

            if self._is_result_sufficient(result):
                escalation_path.append("simple_search_success")
                return self._enhance_result(result, escalation_path)

            escalation_path.append("simple_search_insufficient")
        except Exception as e:
            escalation_path.append(f"simple_search_failed: {e}")

        # Phase 2: Escalate to analysis workflow
        try:
            analysis_strategy = AnalysisWorkflowStrategy(self.services)
            result = await analysis_strategy.execute(intent, context)

            escalation_path.append("analysis_workflow_success")
            return self._enhance_result(result, escalation_path)

        except Exception as e:
            escalation_path.append(f"analysis_workflow_failed: {e}")

        # Phase 3: Final fallback to original tools
        return await self._fallback_to_original_tools(intent, context, escalation_path)
```

### 4. Workflow Engine

**Location**: `src/codeweaver/intent/workflows/`

```python
@dataclass
class WorkflowStep:
    """Individual workflow step with execution context."""
    step_name: str
    executor: Callable
    dependencies: list[str]
    timeout: float = 10.0

class WorkflowOrchestrator:
    """Coordinates multi-step workflow execution."""

    def __init__(self, services_manager: ServicesManager):
        self.services = services_manager

    async def execute_workflow(
        self,
        steps: list[WorkflowStep],
        context: dict
    ) -> dict[str, Any]:
        """Execute workflow steps with dependency resolution."""
        results = {}

        for step in self._resolve_dependencies(steps):
            try:
                step_context = {**context, "previous_results": results}
                result = await asyncio.wait_for(
                    step.executor(step_context),
                    timeout=step.timeout
                )
                results[step.step_name] = result

            except asyncio.TimeoutError:
                logger.warning("Step %s timed out", step.step_name)
                results[step.step_name] = {"error": "timeout", "partial": True}
            except Exception as e:
                logger.exception("Step %s failed", step.step_name)
                results[step.step_name] = {"error": str(e), "failed": True}

        return results
```

## 🔧 Configuration Schema

### Intent Layer Configuration
```toml
[intent]
enabled = true
default_strategy = "adaptive"
confidence_threshold = 0.6
max_execution_time = 30.0
debug_mode = false  # Exposes original 4 tools

[intent.parsing]
use_nlp_fallback = false  # Set to true when NLP models available
pattern_matching = true
confidence_scoring = "multi_factor"

[intent.strategies]
simple_search = { enabled = true, timeout = 5.0 }
analysis_workflow = { enabled = true, timeout = 15.0, max_steps = 5 }
adaptive = { enabled = true, escalation_threshold = 0.3 }

[intent.caching]
enabled = true
cache_type = "basic"  # "basic" or "semantic"
cache_ttl = 3600
max_cache_size = 1000

[intent.performance]
strategy_learning = true
performance_tracking = true
fallback_timeout = 2.0
```

## 🚀 MCP Tools Interface

### Primary Tool: process_intent
```python
async def process_intent_tool(
    intent: str,
    context: Optional[dict] = None
) -> dict[str, Any]:
    """
    Process natural language intent and return appropriate results.

    Args:
        intent: Natural language description of what you want to do
        context: Optional context for the request

    Returns:
        Structured result with data, metadata, and execution info

    Examples:
        - "find authentication functions"
        - "understand the database connection architecture"
        - "analyze performance bottlenecks in the API layer"
        - "show me all error handling patterns"
    """
    orchestrator = await get_intent_orchestrator()
    result = await orchestrator.process_intent(intent, context or {})

    return {
        "success": result.success,
        "data": result.data,
        "error": result.error_message if not result.success else None,
        "metadata": result.metadata,
        "suggestions": result.suggestions if not result.success else None
    }
```

### Helper Tool: get_intent_capabilities
```python
async def get_intent_capabilities_tool() -> dict[str, Any]:
    """
    Get information about supported intent types and capabilities.

    Returns:
        Information about what types of requests can be processed
    """
    orchestrator = await get_intent_orchestrator()
    supported_intents = await orchestrator.get_supported_intents()

    return {
        "supported_intents": [
            {
                "type": intent.intent_type,
                "description": intent.description,
                "examples": intent.examples,
                "complexity_levels": intent.supported_complexity
            }
            for intent in supported_intents
        ],
        "strategies_available": len(supported_intents),
        "debug_mode": config.intent.debug_mode
    }
```

## 📊 Implementation Phases

### Phase 1: Essential Features (Alpha Release)
**Timeline**: 6-8 weeks
**Components**:
- Intent Orchestrator with basic service integration
- Pattern-based Intent Parser with confidence scoring
- 3 Core Strategies (Simple, Analysis, Adaptive)
- Basic caching and error recovery
- FastMCP middleware integration
- Configuration system

**Success Criteria**:
- 85%+ intent recognition accuracy for common patterns
- <5s response time for complex queries
- Graceful fallback to original tools
- No breaking changes to existing architecture

### Phase 2: Enhanced Features
**Timeline**: 3-4 weeks
**Components**:
- NLP-enhanced parser with spaCy integration
- Semantic caching with vector similarity
- Performance tracking and strategy optimization
- Advanced error recovery with context preservation

**Success Criteria**:
- 92%+ intent recognition accuracy
- <3s response time for complex queries
- 85%+ cache hit rate for similar queries

### Phase 3: Advanced Features
**Timeline**: 2-3 weeks
**Components**:
- User learning and feedback integration
- Multi-strategy composition
- Advanced debugging and profiling tools
- Developer customization framework

## 🛡️ Error Handling & Recovery

### Error Categories
```python
class IntentError(Exception):
    """Base class for intent layer errors."""

class IntentParsingError(IntentError):
    """Error in parsing user intent."""

class StrategyExecutionError(IntentError):
    """Error in strategy execution."""

class ServiceIntegrationError(IntentError):
    """Error in service layer integration."""
```

### Fallback Chain
1. **Strategy Fallback**: Failed strategy → Adaptive strategy
2. **Parser Fallback**: NLP parser failure → Pattern parser
3. **Service Fallback**: Service unavailable → Degraded mode
4. **Tool Fallback**: All strategies fail → Original tool routing

### Recovery Patterns
```python
async def _fallback_execution(
    self,
    intent_text: str,
    context: dict,
    error: Exception
) -> IntentResult:
    """Execute fallback strategy when primary execution fails."""
    try:
        # Try adaptive strategy
        adaptive_strategy = AdaptiveStrategy(self.services)
        result = await adaptive_strategy.execute(
            ParsedIntent.create_fallback(intent_text),
            context
        )

        result.metadata["fallback_used"] = True
        result.metadata["original_error"] = str(error)
        return result

    except Exception as fallback_error:
        logger.exception("Fallback execution also failed")

        # Final fallback: route to most appropriate original tool
        return await self._route_to_original_tool(intent_text, context)
```

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Service layer integration
3. **Intent Accuracy Tests**: Parser accuracy validation
4. **Performance Tests**: Response time and resource usage
5. **End-to-End Tests**: Complete workflow validation

### Key Test Scenarios
```python
class TestIntentAccuracy:
    """Test intent recognition accuracy."""

    async def test_common_intent_patterns(self):
        """Test recognition of common coding intents."""
        test_cases = [
            ("find authentication functions", IntentType.SEARCH, 0.9),
            ("understand database architecture", IntentType.UNDERSTAND, 0.85),
            ("analyze performance issues", IntentType.ANALYZE, 0.88),
            ("index this codebase", IntentType.INDEX, 0.95)
        ]

        for intent_text, expected_type, min_confidence in test_cases:
            result = await self.parser.parse(intent_text)
            assert result.intent_type == expected_type
            assert result.confidence >= min_confidence

class TestStrategySelection:
    """Test strategy selection accuracy."""

    async def test_strategy_routing(self):
        """Test correct strategy selection for different intents."""
        # Simple search intent → SimpleSearchStrategy
        simple_intent = ParsedIntent(
            intent_type=IntentType.SEARCH,
            complexity=Complexity.SIMPLE,
            confidence=0.9
        )
        strategy = await self.engine.select_strategy(simple_intent)
        assert isinstance(strategy, SimpleSearchStrategy)

        # Complex analysis intent → AnalysisWorkflowStrategy
        complex_intent = ParsedIntent(
            intent_type=IntentType.UNDERSTAND,
            complexity=Complexity.COMPLEX,
            confidence=0.8
        )
        strategy = await self.engine.select_strategy(complex_intent)
        assert isinstance(strategy, AnalysisWorkflowStrategy)
```

## 📈 Success Metrics

### Technical Metrics
- **Intent Recognition Accuracy**: >90% (target: 92%)
- **Strategy Selection Accuracy**: >95%
- **Response Time P95**: <3s for complex queries
- **Cache Hit Rate**: >80% for repeated/similar queries
- **Fallback Success Rate**: >85%

### User Experience Metrics
- **Single Tool Success Rate**: >90% of queries handled by process_intent
- **Clarification Requests**: <2 per session
- **Error Recovery Success**: >85%

### System Performance Metrics
- **Memory Overhead**: <30MB additional
- **CPU Overhead**: <5% during idle
- **Concurrent Request Handling**: >100 req/s

## 🔗 Integration Points

### FastMCP Middleware Integration
```python
# Middleware insertion point
class IntentMiddleware:
    """FastMCP middleware for intent processing."""

    async def __call__(self, request, call_next):
        """Process request through intent layer."""
        if request.method == "process_intent":
            # Route through intent layer
            orchestrator = await get_intent_orchestrator()
            return await orchestrator.process_intent(
                request.params["intent"],
                request.context
            )

        # Pass through to existing tools
        return await call_next(request)
```

### Service Context Integration
```python
# Enhanced service context for intent layer
async def create_intent_context(base_context: dict) -> dict:
    """Create enhanced context for intent processing."""
    services_manager = base_context.get("services_manager")

    return {
        **base_context,
        "chunking_service": services_manager.get_chunking_service(),
        "filtering_service": services_manager.get_filtering_service(),
        "caching_service": services_manager.get_caching_service(),
        "intent_metadata": {
            "session_id": generate_session_id(),
            "timestamp": datetime.utcnow(),
            "request_id": generate_request_id()
        }
    }
```

This unified specification provides a complete, actionable implementation plan that reconciles the original architecture vision with the critical enhancements identified in the gap analysis. The phased approach ensures alpha release readiness while enabling future sophistication.
