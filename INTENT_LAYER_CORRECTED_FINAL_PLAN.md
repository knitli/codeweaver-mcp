# Intent Layer: Corrected Implementation Plan

## 🎯 Executive Summary

**Project**: Transform CodeWeaver's LLM interface from 4 complex tools to 1-2 intuitive, natural language tools while maintaining full architectural compliance.

**Strategy**: Architecture-compliant implementation leveraging existing services layer, factory patterns, and configuration hierarchy.

**Outcome**: iPhone-like "just works" experience for LLMs with zero breaking changes to established patterns.

**Timeline**: 11-14 weeks total (6-8 weeks for alpha-ready implementation)

## 📊 Architectural Compliance Requirements

This corrected plan addresses critical gaps in the original design to ensure full integration with CodeWeaver's established architecture:

### ✅ **Existing Pattern Compliance**
- **Services Integration**: `IntentOrchestrator` extends `BaseServiceProvider` following established patterns
- **Factory Integration**: Strategy registration through existing `ExtensibilityManager` and `ServiceRegistry`
- **Configuration Hierarchy**: Extends `ServicesConfig` with `IntentServiceConfig` maintaining TOML hierarchy
- **FastMCP Integration**: Leverages existing `ServiceBridge` and middleware pipeline
- **Protocol-Based Design**: All components implement runtime-checkable protocols

### 🔧 **Critical Architecture Corrections**
- **No INDEX Intent**: Removed user-facing INDEX, replaced with `AutoIndexingService` background service
- **Service Provider Pattern**: All components extend `BaseServiceProvider` with proper lifecycle management
- **Error Handling**: Uses existing exception hierarchy and patterns from `_types.exceptions`
- **Testing Framework**: Integrates with existing testing utilities in `src/codeweaver/testing/`
- **Context Propagation**: Proper service context injection through `ServicesManager.create_service_context()`

### 🎯 **Background Indexing System** 
- **AutoIndexingService**: Background service integrated with `ServicesManager`
- **File Watching**: Uses existing `watchdog` patterns for real-time updates
- **Developer Control**: Start/stop exposed to framework developers only
- **Service Dependencies**: Leverages existing `chunking_service` and `filtering_service`
- **LLM Invisible**: No indexing tools exposed to LLM users

## 🏗️ Corrected Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Interface                           │
│  process_intent("find auth functions") → Structured Result │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Intent Layer (ARCHITECTURE COMPLIANT)        │
│  IntentOrchestrator(BaseServiceProvider) → Strategy →      │
│  Workflow → Result (via ServicesManager integration)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Existing CodeWeaver Architecture                │
│  [AutoIndexingService] ← ServicesManager ← FastMCP        │
│  [Service Context] ← ExtensibilityManager ← Factory        │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Corrected Implementation Phases

### Phase 1: Essential Features (Alpha Release) - CORRECTED
**Duration**: 6-8 weeks  
**Priority**: MUST HAVE for alpha release

#### Week 1-2: Service-Compliant Infrastructure
**Deliverables**:
- `IntentOrchestrator` extending `BaseServiceProvider`
- `AutoIndexingService` for background codebase monitoring
- Pattern-based intent parser with confidence scoring
- `IntentServiceConfig` extending existing configuration hierarchy
- FastMCP middleware integration via existing `ServiceBridge`

**Components (Corrected)**:
```
src/codeweaver/services/providers/
├── intent_orchestrator.py      # IntentOrchestrator(BaseServiceProvider)
├── auto_indexing.py           # AutoIndexingService(BaseServiceProvider)

src/codeweaver/intent/
├── parsing/pattern_matcher.py # Regex-based parsing
├── config/intent_config.py    # IntentServiceConfig extension
└── middleware/intent_bridge.py # ServiceBridge integration
```

**Success Criteria**:
- Intent orchestrator registered with `ServicesManager`
- Auto-indexing runs in background with health monitoring
- Service context properly injected through existing patterns
- Configuration extends existing hierarchy without conflicts

#### Week 3-4: Strategy System (Factory-Compliant)
**Deliverables**:
- Core strategies registered through `ExtensibilityManager`
- `StrategyRegistry` integrated with existing service discovery
- Strategy implementations following `BaseServiceProvider` patterns
- Performance tracking through existing health monitoring

**Components (Corrected)**:
```
src/codeweaver/intent/strategies/
├── base_strategy.py           # IntentStrategy(Protocol)
├── simple_search.py          # SimpleSearchStrategy(BaseServiceProvider)
├── analysis_workflow.py      # AnalysisWorkflowStrategy(BaseServiceProvider)
├── adaptive.py               # AdaptiveStrategy(BaseServiceProvider)
└── registry.py               # Integrates with ExtensibilityManager
```

**Success Criteria**:
- All strategies discoverable through existing plugin system
- Strategy selection uses existing performance tracking patterns
- Service context properly propagated to all strategy executions
- Health checks integrated with existing monitoring

#### Week 5-6: Workflow Engine & Error Handling (Pattern-Compliant)
**Deliverables**:
- Multi-step workflow orchestration using existing service patterns
- Error handling using existing exception hierarchy
- Basic result caching integrated with existing caching services
- Integration testing using existing testing framework

**Components (Corrected)**:
```
src/codeweaver/intent/workflows/
├── orchestrator.py           # WorkflowOrchestrator(BaseServiceProvider)
├── steps.py                  # Individual workflow steps

src/codeweaver/intent/recovery/
├── fallback_handler.py       # Uses existing exception patterns
└── error_categories.py       # Extends _types.exceptions hierarchy
```

**Success Criteria**:
- Multi-step workflows execute reliably with service dependencies
- Error recovery follows established patterns
- Caching integrates with existing cache services
- All testing uses existing testing utilities

#### Week 7-8: MCP Tools & Alpha Preparation
**Deliverables**:
- MCP tools implementation (`process_intent`, `get_capabilities`)
- Comprehensive test suite using existing testing framework
- Documentation aligned with existing patterns
- Performance optimization through existing monitoring

**Success Criteria**:
- Single `process_intent` tool handles 90%+ of queries
- No INDEX intent exposed to LLM users
- Background indexing operates transparently
- All alpha success criteria met with architectural compliance

### Phase 2: Enhanced Features (Performance & Intelligence) - UNCHANGED
**Duration**: 3-4 weeks  
**Priority**: SHOULD HAVE for production readiness

- Enhanced NLP parsing with spaCy integration
- Semantic caching using existing vector backends
- Advanced error recovery with context preservation
- Performance optimization through existing monitoring

### Phase 3: Advanced Features (Future Enhancement) - UNCHANGED
**Duration**: 2-3 weeks  
**Priority**: COULD HAVE for enhanced developer experience

- User feedback integration and learning
- Advanced debugging and profiling tools integrated with existing patterns
- Multi-strategy composition
- A/B testing framework using existing infrastructure

## 🔧 Corrected Technical Implementation

### 1. Intent Orchestrator (Service-Compliant)
```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver._types import ServiceHealth, ServiceStatus, IntentResult

class IntentOrchestrator(BaseServiceProvider):
    """Service-compliant orchestrator integrated with ServicesManager."""
    
    def __init__(self, config: IntentServiceConfig):
        super().__init__(config)
        self.parser = None
        self.strategy_registry = None
        self.cache_service = None
    
    async def _initialize_provider(self) -> None:
        """Initialize with existing service dependencies."""
        # Get services through existing dependency injection
        self.cache_service = await self.get_service_dependency("caching_service")
        self.parser = IntentParserFactory.create(self.config.parsing)
        self.strategy_registry = StrategyRegistry(self.services_manager)
    
    async def process_intent(
        self, 
        intent_text: str, 
        context: dict[str, Any]
    ) -> IntentResult:
        """Process intent with full service integration."""
        try:
            # Use existing service context patterns
            service_context = await self.services_manager.create_service_context()
            enhanced_context = {**context, **service_context}
            
            # Check cache using existing caching service
            if self.cache_service:
                cached_result = await self.cache_service.get(
                    self._generate_cache_key(intent_text)
                )
                if cached_result:
                    return cached_result
            
            # Parse intent
            parsed_intent = await self.parser.parse(intent_text)
            
            # Select and execute strategy with service context
            strategy = await self.strategy_registry.select_strategy(parsed_intent)
            result = await strategy.execute(parsed_intent, enhanced_context)
            
            # Cache using existing service
            if self.cache_service and result.success:
                await self.cache_service.set(
                    self._generate_cache_key(intent_text), 
                    result,
                    ttl=self.config.cache_ttl
                )
            
            return result
            
        except Exception as e:
            logger.exception("Intent processing failed")
            return await self._execute_fallback(intent_text, context, e)
    
    async def health_check(self) -> ServiceHealth:
        """Health check following existing patterns."""
        try:
            # Check dependent services
            if self.cache_service:
                cache_health = await self.cache_service.health_check()
                if cache_health.status == ServiceStatus.UNHEALTHY:
                    return ServiceHealth(
                        status=ServiceStatus.DEGRADED,
                        message=f"Cache service unhealthy: {cache_health.message}",
                        last_check=datetime.now(timezone.utc)
                    )
            
            # Test basic functionality
            test_intent = "test intent"
            await self.parser.parse(test_intent)
            
            return ServiceHealth(
                status=ServiceStatus.HEALTHY,
                message="Intent orchestrator operational",
                last_check=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Intent orchestrator error: {e}",
                last_check=datetime.now(timezone.utc)
            )
```

### 2. Auto-Indexing Service (Background)
```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoIndexingService(BaseServiceProvider):
    """Background indexing service - NOT exposed to LLM users."""
    
    def __init__(self, config: AutoIndexingConfig):
        super().__init__(config)
        self.observer = None
        self.watched_paths = set()
        self.chunking_service = None
        self.filtering_service = None
    
    async def _initialize_provider(self) -> None:
        """Initialize with existing service dependencies."""
        self.chunking_service = await self.get_service_dependency("chunking_service")
        self.filtering_service = await self.get_service_dependency("filtering_service")
        self.observer = Observer()
    
    async def start_monitoring(self, path: str) -> None:
        """Start background monitoring - exposed to framework developers only."""
        if path in self.watched_paths:
            logger.info("Path already being monitored: %s", path)
            return
        
        # Initial indexing
        await self._index_path(path)
        
        # Setup file watching
        event_handler = CodebaseChangeHandler(self)
        self.observer.schedule(event_handler, path, recursive=True)
        
        if not self.observer.is_alive():
            self.observer.start()
        
        self.watched_paths.add(path)
        logger.info("Started monitoring path: %s", path)
    
    async def stop_monitoring(self, path: str = None) -> None:
        """Stop monitoring - exposed to framework developers only."""
        if path:
            self.watched_paths.discard(path)
        else:
            self.watched_paths.clear()
        
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        logger.info("Stopped monitoring")
    
    async def _index_path(self, path: str) -> None:
        """Index path using existing services."""
        if not self.filtering_service or not self.chunking_service:
            logger.warning("Required services not available for indexing")
            return
        
        # Use existing filtering service
        files = await self.filtering_service.discover_files(path)
        
        for file_path in files:
            try:
                content = await self._read_file_content(file_path)
                
                # Use existing chunking service
                chunks = await self.chunking_service.chunk_content(
                    content, str(file_path)
                )
                
                # Index chunks using existing backend
                await self._store_chunks(file_path, chunks)
                
            except Exception as e:
                logger.warning("Failed to index file %s: %s", file_path, e)
    
    async def health_check(self) -> ServiceHealth:
        """Health check following existing patterns."""
        try:
            is_monitoring = bool(self.watched_paths and 
                               self.observer and 
                               self.observer.is_alive())
            
            status = ServiceStatus.HEALTHY if is_monitoring else ServiceStatus.DEGRADED
            message = f"Monitoring {len(self.watched_paths)} paths" if is_monitoring else "Not monitoring any paths"
            
            return ServiceHealth(
                status=status,
                message=message,
                last_check=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Auto-indexing error: {e}",
                last_check=datetime.now(timezone.utc)
            )
```

### 3. Configuration Integration (Hierarchy-Compliant)
```python
from codeweaver._types.config import BaseServiceConfig
from typing import Annotated
from pydantic import Field

class IntentServiceConfig(BaseServiceConfig):
    """Intent service configuration extending existing hierarchy."""
    
    enabled: Annotated[bool, Field(default=True, description="Enable intent processing")]
    default_strategy: Annotated[str, Field(default="adaptive", description="Default strategy")]
    confidence_threshold: Annotated[float, Field(default=0.6, description="Minimum confidence threshold")]
    max_execution_time: Annotated[float, Field(default=30.0, description="Maximum execution time")]
    debug_mode: Annotated[bool, Field(default=False, description="Enable debug mode")]
    cache_ttl: Annotated[int, Field(default=3600, description="Cache TTL in seconds")]

class AutoIndexingConfig(BaseServiceConfig):
    """Auto-indexing configuration extending existing hierarchy."""
    
    enabled: Annotated[bool, Field(default=True, description="Enable auto-indexing")]
    watch_patterns: Annotated[list[str], Field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts"], description="File patterns to watch")]
    ignore_patterns: Annotated[list[str], Field(default_factory=lambda: [".git", "node_modules", "__pycache__"], description="Patterns to ignore")]
    debounce_delay: Annotated[float, Field(default=1.0, description="Debounce delay for file changes")]

# Integration with existing services configuration
class ServicesConfig(BaseModel):
    """Extended services configuration."""
    
    # Existing services
    chunking: ChunkingServiceConfig = ChunkingServiceConfig()
    filtering: FilteringServiceConfig = FilteringServiceConfig()
    validation: ValidationServiceConfig = ValidationServiceConfig()
    
    # New intent-related services
    intent: IntentServiceConfig = IntentServiceConfig()
    auto_indexing: AutoIndexingConfig = AutoIndexingConfig()
```

### 4. FastMCP Integration (ServiceBridge Pattern)
```python
from codeweaver.services.middleware_bridge import ServiceBridge

class IntentServiceBridge(ServiceBridge):
    """Intent-specific service bridge extending existing patterns."""
    
    def __init__(self, services_manager: ServicesManager):
        super().__init__(services_manager)
        self.intent_orchestrator = None
    
    async def initialize(self) -> None:
        """Initialize intent orchestrator through service registry."""
        await super().initialize()
        self.intent_orchestrator = await self.services_manager.get_service("intent_orchestrator")
    
    async def create_intent_context(self, base_context: dict) -> dict:
        """Create intent-specific service context."""
        service_context = await self.create_service_context(base_context)
        
        return {
            **service_context,
            "intent_metadata": {
                "session_id": self._generate_session_id(),
                "timestamp": datetime.now(timezone.utc),
                "request_id": self._generate_request_id()
            }
        }

# FastMCP middleware integration
async def process_intent_tool(
    intent: str,
    context: Optional[dict] = None
) -> dict[str, Any]:
    """MCP tool integrated with existing service patterns."""
    # Get service bridge from FastMCP context - existing pattern
    service_bridge = context.fastmcp_context.get_state_value("service_bridge")
    
    if not service_bridge or not isinstance(service_bridge, IntentServiceBridge):
        raise ServiceUnavailableError("Intent service bridge not available")
    
    # Create enhanced context using existing patterns
    enhanced_context = await service_bridge.create_intent_context(context or {})
    
    # Process intent through service
    result = await service_bridge.intent_orchestrator.process_intent(
        intent, enhanced_context
    )
    
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error_message if not result.success else None,
        "metadata": result.metadata,
        "suggestions": result.suggestions if not result.success else None
    }
```

## 📊 Corrected Success Metrics

### Alpha Release Targets (Architecture-Compliant)
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Service Integration Compliance** | 100% | All components extend BaseServiceProvider |
| **Configuration Compliance** | 100% | Extends existing ServicesConfig hierarchy |
| **Factory Integration Compliance** | 100% | Uses ExtensibilityManager patterns |
| **Intent Recognition Accuracy** | >85% | Pattern matching on test queries (no INDEX intent) |
| **Strategy Selection Accuracy** | >95% | Correct strategy chosen for intent type |
| **Background Indexing** | Active | AutoIndexingService monitoring without user visibility |
| **Response Time P95** | <5s | Complex query execution time |
| **Service Health Monitoring** | Active | Integrated with existing health checks |

### Enhanced Release Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Intent Recognition Accuracy** | >92% | NLP-enhanced parsing accuracy |
| **Response Time P95** | <3s | Optimized execution time |
| **Cache Hit Rate** | >85% | Using existing caching services |
| **Service Recovery** | >90% | Error recovery through existing patterns |

## 🛡️ Risk Mitigation (Architecture-Focused)

### Implementation Risks & Solutions

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Architecture non-compliance** | Low | High | Comprehensive pattern compliance validation + integration tests |
| **Service integration complexity** | Medium | Medium | Leverage existing ServiceBridge patterns + gradual integration |
| **Background indexing performance** | Low | Medium | Use existing chunking/filtering services + health monitoring |
| **Configuration conflicts** | Low | High | Extend existing hierarchy + validation tests |

## 🔧 Testing Strategy (Framework-Compliant)

### Test Structure (Using Existing Patterns)
```python
# tests/unit/test_intent_service.py
from codeweaver.testing.mock_services import MockServicesManager
from codeweaver.services.providers.intent_orchestrator import IntentOrchestrator

class TestIntentOrchestrator:
    """Test intent orchestrator following existing testing patterns."""
    
    @pytest.fixture
    def services_manager(self):
        """Mock services manager using existing utilities."""
        return MockServicesManager()
    
    @pytest.fixture  
    def intent_orchestrator(self, services_manager):
        """Intent orchestrator with mocked dependencies."""
        config = IntentServiceConfig()
        return IntentOrchestrator(config, services_manager)
    
    async def test_service_compliance(self, intent_orchestrator):
        """Test service provider compliance."""
        assert isinstance(intent_orchestrator, BaseServiceProvider)
        
        health = await intent_orchestrator.health_check()
        assert isinstance(health, ServiceHealth)
        
    async def test_intent_processing_with_services(
        self, 
        intent_orchestrator, 
        services_manager
    ):
        """Test intent processing with service integration."""
        # Setup mock services using existing patterns
        services_manager.mock_service("caching_service")
        services_manager.mock_service("chunking_service")
        
        result = await intent_orchestrator.process_intent(
            "find authentication functions", 
            {}
        )
        
        assert result.success
        assert "search" in result.metadata.get("strategy", "")
```

## 📁 Corrected Directory Structure

```
src/codeweaver/services/providers/
├── intent_orchestrator.py           # IntentOrchestrator(BaseServiceProvider)
├── auto_indexing.py                # AutoIndexingService(BaseServiceProvider)

src/codeweaver/intent/
├── __init__.py                     # Public API
├── parsing/
│   ├── pattern_matcher.py          # Pattern-based parsing
│   ├── factory.py                  # Parser factory
│   └── confidence_scorer.py        # Confidence algorithms
├── strategies/
│   ├── base_strategy.py            # IntentStrategy(Protocol)
│   ├── simple_search.py           # SimpleSearchStrategy(BaseServiceProvider)
│   ├── analysis_workflow.py       # AnalysisWorkflowStrategy(BaseServiceProvider)  
│   ├── adaptive.py                # AdaptiveStrategy(BaseServiceProvider)
│   └── registry.py                # ExtensibilityManager integration
├── workflows/
│   ├── orchestrator.py            # WorkflowOrchestrator(BaseServiceProvider)
│   └── steps.py                   # Individual workflow steps
├── recovery/
│   ├── fallback_handler.py        # Uses existing exception patterns
│   └── error_categories.py        # Extends _types.exceptions
├── config/
│   └── intent_config.py           # IntentServiceConfig/AutoIndexingConfig
└── middleware/
    └── intent_bridge.py           # IntentServiceBridge extends ServiceBridge

tests/unit/intent/                  # Using existing testing framework
tests/integration/intent/           # Service integration tests
```

## 🎯 Conclusion

This corrected implementation plan ensures full architectural compliance with CodeWeaver's established patterns while delivering the transformative intent layer experience. Key corrections:

1. **No INDEX Intent**: Replaced with transparent `AutoIndexingService`
2. **Service Compliance**: All components extend `BaseServiceProvider`
3. **Factory Integration**: Leverages existing `ExtensibilityManager` patterns
4. **Configuration Hierarchy**: Extends `ServicesConfig` properly
5. **FastMCP Integration**: Uses existing `ServiceBridge` patterns
6. **Error Handling**: Follows established exception hierarchy
7. **Testing Framework**: Uses existing testing utilities

**Expected Outcome**: A production-ready intent layer that transforms the LLM user experience while preserving the architectural integrity that makes CodeWeaver powerful and extensible.