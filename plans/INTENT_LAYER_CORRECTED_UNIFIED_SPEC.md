<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer: Corrected Unified Implementation Specification

## 🎯 Executive Summary

**Purpose**: Transform CodeWeaver's LLM interface from 4 complex tools to 1-2 intuitive, natural language tools while maintaining full architectural compliance with existing patterns.

**Strategy**: Architecture-compliant implementation leveraging existing services layer, factory patterns, and configuration hierarchy with critical corrections for proper integration.

**Target**: Alpha release with 90%+ intent recognition accuracy, transparent background indexing, and zero breaking changes.

## 🏗️ Corrected Core Architecture

### System Overview (Architecture-Compliant)
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Interface                           │
│  NEW: process_intent + get_capabilities (NO INDEX)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Intent Layer (SERVICE COMPLIANT)             │
│                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ Intent          │ │ Strategy        │ │ Workflow        ││
│  │ Orchestrator    │ │ Engine          │ │ Engine          ││
│  │(BaseServiceProv)│ │(ExtensibilityMgr)│ │(ServiceBridge)  ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│       Existing CodeWeaver Architecture (ENHANCED)         │
│  [AutoIndexingService] ← ServicesManager ← FastMCP        │
│  [Enhanced Services] ← ExtensibilityManager ← Factory      │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles (Corrected)
1. **Service Provider Pattern**: All components extend `BaseServiceProvider`
2. **Factory Integration**: Uses existing `ExtensibilityManager` for registration
3. **Configuration Hierarchy**: Extends `ServicesConfig` without conflicts
4. **Background Indexing**: `AutoIndexingService` - NOT exposed to LLM users
5. **Context Propagation**: Uses existing `ServiceBridge` patterns

## 📋 Corrected Component Specifications

### 1. Intent Orchestrator (Service-Compliant)

**Location**: `src/codeweaver/services/providers/intent_orchestrator.py`

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import ServiceHealth, ServiceStatus, IntentResult

class IntentOrchestrator(BaseServiceProvider):
    """Service-compliant orchestrator following existing patterns."""

    def __init__(self, config: IntentServiceConfig):
        super().__init__(config)
        self.parser = None
        self.strategy_registry = None
        self.cache_service = None

    async def _initialize_provider(self) -> None:
        """Initialize using existing dependency injection patterns."""
        # Get services through existing ServicesManager patterns
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
            # Create service context using existing patterns
            service_context = await self.services_manager.create_service_context()
            enhanced_context = {**context, **service_context}

            # Check cache using existing caching service
            if self.cache_service:
                cache_key = self._generate_cache_key(intent_text)
                cached_result = await self.cache_service.get(cache_key)
                if cached_result:
                    return cached_result

            # Parse intent (NO INDEX intent type)
            parsed_intent = await self.parser.parse(intent_text)

            # Select and execute strategy with service context
            strategy = await self.strategy_registry.select_strategy(parsed_intent)
            result = await strategy.execute(parsed_intent, enhanced_context)

            # Cache using existing service patterns
            if self.cache_service and result.success:
                await self.cache_service.set(cache_key, result, ttl=self.config.cache_ttl)

            return result

        except Exception as e:
            logger.exception("Intent processing failed")
            return await self._execute_fallback(intent_text, context, e)

    async def health_check(self) -> ServiceHealth:
        """Health check following existing service patterns."""
        try:
            # Check dependent services using existing patterns
            if self.cache_service:
                cache_health = await self.cache_service.health_check()
                if cache_health.status == ServiceStatus.UNHEALTHY:
                    return ServiceHealth(
                        status=ServiceStatus.DEGRADED,
                        message=f"Cache service unhealthy: {cache_health.message}",
                        last_check=datetime.now(timezone.utc)
                    )

            # Test basic functionality
            test_intent = "test search intent"
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

### 2. Auto-Indexing Service (Background Only)

**Location**: `src/codeweaver/services/providers/auto_indexing.py`

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoIndexingService(BaseServiceProvider):
    """Background indexing service - NEVER exposed to LLM users."""

    def __init__(self, config: AutoIndexingConfig):
        super().__init__(config)
        self.observer = None
        self.watched_paths = set()
        self.chunking_service = None
        self.filtering_service = None
        self.backend_registry = None

    async def _initialize_provider(self) -> None:
        """Initialize with existing service dependencies."""
        # Use existing dependency injection patterns
        self.chunking_service = await self.get_service_dependency("chunking_service")
        self.filtering_service = await self.get_service_dependency("filtering_service")
        self.backend_registry = await self.get_service_dependency("backend_registry")
        self.observer = Observer()

    async def start_monitoring(self, path: str) -> None:
        """Start background monitoring - ONLY for framework developers."""
        if path in self.watched_paths:
            logger.info("Path already being monitored: %s", path)
            return

        # Initial indexing using existing services
        await self._index_path_initial(path)

        # Setup file watching
        event_handler = CodebaseChangeHandler(self)
        self.observer.schedule(event_handler, path, recursive=True)

        if not self.observer.is_alive():
            self.observer.start()

        self.watched_paths.add(path)
        logger.info("Started monitoring path: %s", path)

    async def stop_monitoring(self, path: str = None) -> None:
        """Stop monitoring - ONLY for framework developers."""
        if path:
            self.watched_paths.discard(path)
        else:
            self.watched_paths.clear()

        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        logger.info("Stopped monitoring")

    async def _index_path_initial(self, path: str) -> None:
        """Index path using existing service patterns."""
        if not all([self.filtering_service, self.chunking_service, self.backend_registry]):
            logger.warning("Required services not available for indexing")
            return

        # Use existing filtering service for file discovery
        files = await self.filtering_service.discover_files(path)

        for file_path in files:
            try:
                content = await self._read_file_content(file_path)

                # Use existing chunking service
                chunks = await self.chunking_service.chunk_content(
                    content, str(file_path)
                )

                # Store using existing backend patterns
                await self._store_chunks_via_backend(file_path, chunks)

            except Exception as e:
                logger.warning("Failed to index file %s: %s", file_path, e)

    async def health_check(self) -> ServiceHealth:
        """Health check following existing service patterns."""
        try:
            is_monitoring = bool(
                self.watched_paths and
                self.observer and
                self.observer.is_alive()
            )

            # Check dependent services
            service_health_checks = []
            if self.chunking_service:
                health = await self.chunking_service.health_check()
                service_health_checks.append(health.status)

            if self.filtering_service:
                health = await self.filtering_service.health_check()
                service_health_checks.append(health.status)

            # Overall health assessment
            if any(status == ServiceStatus.UNHEALTHY for status in service_health_checks):
                return ServiceHealth(
                    status=ServiceStatus.DEGRADED,
                    message=f"Dependent services unhealthy, monitoring {len(self.watched_paths)} paths",
                    last_check=datetime.now(timezone.utc)
                )

            status = ServiceStatus.HEALTHY if is_monitoring else ServiceStatus.DEGRADED
            message = (
                f"Monitoring {len(self.watched_paths)} paths"
                if is_monitoring
                else "Not monitoring any paths"
            )

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

### 3. Intent Parser (No INDEX Support)

**Location**: `src/codeweaver/intent/parsing/pattern_matcher.py`

```python
from enum import Enum
from pydantic.dataclasses import dataclass
from codeweaver.types import IntentType, Scope, Complexity

# NO INDEX INTENT TYPE - Only SEARCH, UNDERSTAND, ANALYZE
class IntentType(Enum):
    SEARCH = "search"
    UNDERSTAND = "understand"
    ANALYZE = "analyze"
    # INDEX REMOVED - handled by background service

@dataclass
class ParsedIntent:
    """Intent structure without INDEX support."""
    intent_type: IntentType  # SEARCH | UNDERSTAND | ANALYZE (no INDEX)
    primary_target: str      # Main focus ("authentication", "database")
    scope: Scope            # FILE | MODULE | PROJECT | SYSTEM
    complexity: Complexity  # SIMPLE | MODERATE | COMPLEX | ADAPTIVE
    confidence: float       # 0.0-1.0 confidence score
    filters: dict[str, Any] # Additional constraints
    metadata: dict[str, Any] # Parser metadata

class PatternBasedParser:
    """Pattern-based parser without INDEX intent support."""

    def __init__(self):
        self.patterns = self._load_intent_patterns_no_index()
        self.confidence_scorer = BasicConfidenceScorer()

    def _load_intent_patterns_no_index(self) -> dict:
        """Load patterns excluding INDEX intents."""
        return {
            "search_patterns": [
                r"find\s+(.+)",
                r"search\s+(.+)",
                r"locate\s+(.+)",
                r"show\s+me\s+(.+)"
            ],
            "understand_patterns": [
                r"understand\s+(.+)",
                r"explain\s+(.+)",
                r"how\s+does\s+(.+)\s+work",
                r"what\s+is\s+(.+)"
            ],
            "analyze_patterns": [
                r"analyze\s+(.+)",
                r"review\s+(.+)",
                r"investigate\s+(.+)",
                r"check\s+(.+)\s+for"
            ]
            # NO INDEX PATTERNS - indexing is background only
        }

    async def parse(self, intent_text: str) -> ParsedIntent:
        """Parse intent without INDEX support."""
        # Intent type detection (no INDEX)
        intent_type = self._detect_intent_type_no_index(intent_text)

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
            metadata={
                "parser": "pattern_based",
                "patterns_matched": [],
                "index_support": False  # Explicitly no INDEX
            }
        )

    def _detect_intent_type_no_index(self, text: str) -> IntentType:
        """Detect intent type without INDEX option."""
        text_lower = text.lower()

        # Search patterns
        for pattern in self.patterns["search_patterns"]:
            if re.search(pattern, text_lower):
                return IntentType.SEARCH

        # Understand patterns
        for pattern in self.patterns["understand_patterns"]:
            if re.search(pattern, text_lower):
                return IntentType.UNDERSTAND

        # Analyze patterns
        for pattern in self.patterns["analyze_patterns"]:
            if re.search(pattern, text_lower):
                return IntentType.ANALYZE

        # Default to search for simple queries
        return IntentType.SEARCH
```

### 4. Strategy System (ExtensibilityManager Integration)

**Location**: `src/codeweaver/intent/strategies/registry.py`

```python
from codeweaver.factories.extensibility_manager import ExtensibilityManager
from codeweaver.types import IntentStrategy

class StrategyRegistry:
    """Strategy registry integrated with existing ExtensibilityManager."""

    def __init__(self, services_manager: ServicesManager):
        self.services_manager = services_manager
        self.extensibility_manager = ExtensibilityManager()
        self.performance_tracker = StrategyPerformanceTracker()
        self._register_core_strategies()

    def _register_core_strategies(self) -> None:
        """Register core strategies through ExtensibilityManager."""
        # Register using existing factory patterns
        self.extensibility_manager.register_component(
            "simple_search_strategy",
            SimpleSearchStrategy,
            component_type="intent_strategy"
        )

        self.extensibility_manager.register_component(
            "analysis_workflow_strategy",
            AnalysisWorkflowStrategy,
            component_type="intent_strategy"
        )

        self.extensibility_manager.register_component(
            "adaptive_strategy",
            AdaptiveStrategy,
            component_type="intent_strategy"
        )

        # NO INDEX STRATEGY - indexing is background service

    async def select_strategy(self, parsed_intent: ParsedIntent) -> IntentStrategy:
        """Select strategy using existing extensibility patterns."""
        # Discover available strategies through ExtensibilityManager
        available_strategies = self.extensibility_manager.discover_components(
            component_type="intent_strategy"
        )

        candidates = []

        for strategy_info in available_strategies:
            try:
                # Create strategy instance with service dependency injection
                strategy = strategy_info.component_class(self.services_manager)

                # Check if strategy can handle the intent
                can_handle_score = await strategy.can_handle(parsed_intent)

                if can_handle_score > 0.1:  # Minimum threshold
                    performance_score = self.performance_tracker.get_score(
                        strategy_info.name
                    )
                    final_score = (can_handle_score * 0.7) + (performance_score * 0.3)
                    candidates.append((final_score, strategy_info.name, strategy))

            except Exception as e:
                logger.warning("Failed to evaluate strategy %s: %s", strategy_info.name, e)

        if not candidates:
            # Fallback to adaptive strategy
            return AdaptiveStrategy(self.services_manager)

        # Return highest scoring strategy
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_strategy = candidates[0][2]

        # Track selection for performance learning
        await self.performance_tracker.record_selection(
            candidates[0][1], parsed_intent
        )

        return selected_strategy
```

### 5. Configuration Integration (Hierarchy-Compliant)

**Location**: `src/codeweaver.types/config.py` (Enhanced)

```python
from codeweaver.types import BaseServiceConfig
from typing import Annotated
from pydantic import Field, BaseModel

class IntentServiceConfig(BaseServiceConfig):
    """Intent service configuration extending existing hierarchy."""

    enabled: Annotated[bool, Field(default=True, description="Enable intent processing")]
    default_strategy: Annotated[str, Field(default="adaptive", description="Default strategy")]
    confidence_threshold: Annotated[float, Field(default=0.6, description="Minimum confidence threshold")]
    max_execution_time: Annotated[float, Field(default=30.0, description="Maximum execution time")]
    debug_mode: Annotated[bool, Field(default=False, description="Enable debug mode - shows original tools")]
    cache_ttl: Annotated[int, Field(default=3600, description="Cache TTL in seconds")]

    # Parser configuration
    use_nlp_fallback: Annotated[bool, Field(default=False, description="Enable NLP fallback parser")]
    pattern_matching: Annotated[bool, Field(default=True, description="Enable pattern matching")]

class AutoIndexingConfig(BaseServiceConfig):
    """Auto-indexing configuration extending existing hierarchy."""

    enabled: Annotated[bool, Field(default=True, description="Enable background auto-indexing")]
    watch_patterns: Annotated[list[str], Field(
        default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
        description="File patterns to watch"
    )]
    ignore_patterns: Annotated[list[str], Field(
        default_factory=lambda: [".git", "node_modules", "__pycache__", "*.pyc", ".DS_Store"],
        description="Patterns to ignore"
    )]
    debounce_delay: Annotated[float, Field(default=1.0, description="Debounce delay for file changes")]
    max_file_size: Annotated[int, Field(default=1048576, description="Maximum file size to index (1MB)")]

# Enhanced ServicesConfig to include intent services
class ServicesConfig(BaseModel):
    """Enhanced services configuration with intent support."""
    model_config = ConfigDict(extra="allow")

    # Existing services
    chunking: ChunkingServiceConfig = ChunkingServiceConfig()
    filtering: FilteringServiceConfig = FilteringServiceConfig()
    validation: ValidationServiceConfig = ValidationServiceConfig()

    # New intent-related services
    intent: IntentServiceConfig = IntentServiceConfig()
    auto_indexing: AutoIndexingConfig = AutoIndexingConfig()
```

### 6. FastMCP Integration (ServiceBridge Pattern)

**Location**: `src/codeweaver/intent/middleware/intent_bridge.py`

```python
from codeweaver.services.middleware_bridge import ServiceBridge
from codeweaver.services.manager import ServicesManager

class IntentServiceBridge(ServiceBridge):
    """Intent-specific service bridge extending existing patterns."""

    def __init__(self, services_manager: ServicesManager):
        super().__init__(services_manager)
        self.intent_orchestrator = None
        self.auto_indexing_service = None

    async def initialize(self) -> None:
        """Initialize intent services through existing service registry."""
        await super().initialize()

        # Get services through existing ServicesManager
        self.intent_orchestrator = await self.services_manager.get_service("intent_orchestrator")
        self.auto_indexing_service = await self.services_manager.get_service("auto_indexing")

        # Start background indexing if configured
        if self.auto_indexing_service and self.auto_indexing_service.config.enabled:
            # This would be configured by framework developers, not LLM users
            await self._setup_background_indexing()

    async def create_intent_context(self, base_context: dict) -> dict:
        """Create intent-specific service context using existing patterns."""
        # Use existing service context creation
        service_context = await self.create_service_context(base_context)

        return {
            **service_context,
            "intent_metadata": {
                "session_id": self._generate_session_id(),
                "timestamp": datetime.now(timezone.utc),
                "request_id": self._generate_request_id()
            }
        }

    async def route_request(self, request, call_next):
        """Route requests with intent support."""
        if request.method == "process_intent":
            # Route through intent layer
            if not self.intent_orchestrator:
                raise ServiceUnavailableError("Intent orchestrator not available")

            context = await self.create_intent_context(request.context or {})
            return await self.intent_orchestrator.process_intent(
                request.params["intent"], context
            )

        elif request.method == "get_intent_capabilities":
            # Return capabilities (NO INDEX support)
            return await self._get_intent_capabilities()

        else:
            # Pass through to existing tools - NO CHANGES
            return await call_next(request)

    async def _get_intent_capabilities(self) -> dict:
        """Get intent capabilities without INDEX support."""
        return {
            "supported_intents": [
                {
                    "type": "SEARCH",
                    "description": "Find specific code elements, functions, or patterns",
                    "examples": [
                        "find authentication functions",
                        "search for error handling",
                        "locate database queries"
                    ]
                },
                {
                    "type": "UNDERSTAND",
                    "description": "Understand system architecture or code organization",
                    "examples": [
                        "understand the authentication system",
                        "explain the database architecture",
                        "how does the API routing work"
                    ]
                },
                {
                    "type": "ANALYZE",
                    "description": "Analyze code for issues, patterns, or improvements",
                    "examples": [
                        "analyze performance bottlenecks",
                        "review security vulnerabilities",
                        "check for code quality issues"
                    ]
                }
                # NO INDEX INTENT - background service handles indexing
            ],
            "indexing_mode": "automatic_background",
            "manual_indexing_required": False,
            "background_indexing_active": bool(
                self.auto_indexing_service and
                len(self.auto_indexing_service.watched_paths) > 0
            )
        }
```

## 🚀 Corrected MCP Tools Interface

### Primary Tool: process_intent (No INDEX)
```python
async def process_intent_tool(
    intent: str,
    context: Optional[dict] = None
) -> dict[str, Any]:
    """
    Process natural language intent and return appropriate results.

    Args:
        intent: Natural language description (SEARCH, UNDERSTAND, or ANALYZE)
        context: Optional context for the request

    Returns:
        Structured result with data, metadata, and execution info

    Examples:
        - "find authentication functions" (SEARCH)
        - "understand the database connection architecture" (UNDERSTAND)
        - "analyze performance bottlenecks in the API layer" (ANALYZE)

    Note: Indexing happens automatically in background - no INDEX intent needed
    """
    # Get service bridge from FastMCP context
    service_bridge = context.fastmcp_context.get_state_value("intent_service_bridge")

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
        "metadata": {
            **result.metadata,
            "background_indexing_active": bool(
                service_bridge.auto_indexing_service and
                len(service_bridge.auto_indexing_service.watched_paths) > 0
            ),
            "supported_intents": ["SEARCH", "UNDERSTAND", "ANALYZE"]  # NO INDEX
        },
        "suggestions": result.suggestions if not result.success else None
    }
```

### Helper Tool: get_intent_capabilities (No INDEX)
```python
async def get_intent_capabilities_tool() -> dict[str, Any]:
    """
    Get information about supported intent types and capabilities.

    Returns:
        Information about what types of requests can be processed
        (INDEX not included - handled automatically in background)
    """
    service_bridge = context.fastmcp_context.get_state_value("intent_service_bridge")

    if not service_bridge:
        return {
            "error": "Intent service not available",
            "supported_intents": [],
            "background_indexing": False
        }

    return await service_bridge._get_intent_capabilities()
```

## 📊 Corrected Implementation Phases

### Phase 1: Essential Features (Alpha Release) - ARCHITECTURE COMPLIANT
**Timeline**: 6-8 weeks
**Components**:
- `IntentOrchestrator(BaseServiceProvider)` with service integration
- `AutoIndexingService(BaseServiceProvider)` for background operation
- Pattern-based Intent Parser (NO INDEX support)
- 3 Core Strategies extending `BaseServiceProvider`
- Configuration extending `ServicesConfig` hierarchy
- FastMCP integration via `IntentServiceBridge(ServiceBridge)`

**Success Criteria**:
- 100% architectural compliance with existing patterns
- 85%+ intent recognition accuracy (SEARCH, UNDERSTAND, ANALYZE only)
- Background indexing active and transparent to LLM users
- <5s response time for complex queries
- Zero breaking changes to existing architecture

### Phase 2: Enhanced Features - SAME AS ORIGINAL
**Timeline**: 3-4 weeks
**Components**:
- NLP-enhanced parser with spaCy integration
- Semantic caching using existing vector backends
- Advanced error recovery with context preservation
- Performance optimization through existing monitoring

### Phase 3: Advanced Features - SAME AS ORIGINAL
**Timeline**: 2-3 weeks
**Components**:
- User learning and feedback integration
- Multi-strategy composition
- Advanced debugging tools
- Developer customization framework

## 🛡️ Error Handling & Recovery (Pattern-Compliant)

### Error Categories (Using Existing Hierarchy)
```python
from codeweaver.types import (
    ServiceError,
    ConfigurationError,
    ServiceUnavailableError
)

class IntentError(ServiceError):
    """Base class for intent layer errors extending existing hierarchy."""

class IntentParsingError(IntentError):
    """Error in parsing user intent (no INDEX parsing)."""

class StrategyExecutionError(IntentError):
    """Error in strategy execution."""

class ServiceIntegrationError(IntentError):
    """Error in service layer integration."""
```

### Fallback Chain (Service-Aware)
1. **Strategy Fallback**: Failed strategy → Adaptive strategy → Service recovery
2. **Parser Fallback**: Enhanced parser failure → Pattern parser → Basic search
3. **Service Fallback**: Service unavailable → Degraded mode → Health monitoring
4. **Tool Fallback**: All strategies fail → Route to appropriate original tool

## 🧪 Testing Strategy (Framework-Compliant)

### Test Structure Using Existing Patterns
```python
# tests/unit/test_intent_orchestrator.py
from codeweaver.testing.mock_services import MockServicesManager
from codeweaver.services.providers.intent_orchestrator import IntentOrchestrator

class TestIntentOrchestrator:
    """Test intent orchestrator using existing testing framework."""

    @pytest.fixture
    def services_manager(self):
        """Mock services manager using existing utilities."""
        return MockServicesManager()

    @pytest.fixture
    def intent_orchestrator(self, services_manager):
        """Intent orchestrator with mocked dependencies."""
        config = IntentServiceConfig()
        return IntentOrchestrator(config, services_manager)

    async def test_service_provider_compliance(self, intent_orchestrator):
        """Test compliance with BaseServiceProvider."""
        assert isinstance(intent_orchestrator, BaseServiceProvider)

        # Test health check
        health = await intent_orchestrator.health_check()
        assert isinstance(health, ServiceHealth)

        # Test service lifecycle
        await intent_orchestrator._initialize_provider()
        assert intent_orchestrator.parser is not None

    async def test_no_index_intent_support(self, intent_orchestrator):
        """Test that INDEX intents are not supported."""
        # INDEX intent should not be recognized
        result = await intent_orchestrator.process_intent("index this codebase", {})

        # Should be parsed as a different intent or handled gracefully
        assert result.metadata.get("intent_type") != "INDEX"
        assert "background" in result.metadata.get("indexing_note", "").lower()
```

## 📈 Corrected Success Metrics

### Technical Metrics (Architecture-Compliant)
- **Service Integration Compliance**: 100% (all extend BaseServiceProvider)
- **Configuration Compliance**: 100% (extends ServicesConfig properly)
- **Factory Integration Compliance**: 100% (uses ExtensibilityManager)
- **Intent Recognition Accuracy**: >90% (SEARCH, UNDERSTAND, ANALYZE only)
- **Background Indexing Success**: Active and transparent to LLM users
- **Response Time P95**: <3s for complex queries
- **Service Health Monitoring**: Integrated with existing health checks

### User Experience Metrics (No INDEX)
- **Single Tool Success Rate**: >90% (process_intent handles all queries)
- **INDEX Invisibility**: 100% (no INDEX tool exposed to LLMs)
- **Background Indexing**: Transparent and automatic
- **Framework Developer Control**: Full control over indexing service

## 🔗 Integration Points (Corrected)

### ServicesManager Integration
```python
# services_manager registration
services_manager.register_service("intent_orchestrator", IntentOrchestrator)
services_manager.register_service("auto_indexing", AutoIndexingService)

# Service dependency injection through existing patterns
intent_orchestrator = await services_manager.get_service("intent_orchestrator")
auto_indexing = await services_manager.get_service("auto_indexing")
```

### ExtensibilityManager Integration
```python
# Strategy registration through existing factory patterns
extensibility_manager.register_component(
    "simple_search_strategy",
    SimpleSearchStrategy,
    component_type="intent_strategy"
)
```

### Configuration Integration
```toml
# config.toml - Extends existing hierarchy
[services.intent]
enabled = true
default_strategy = "adaptive"
debug_mode = false  # When true, exposes original 4 tools

[services.auto_indexing]
enabled = true
watch_patterns = ["**/*.py", "**/*.js", "**/*.ts"]
# Background operation - not user-configurable for indexing behavior
```

## 🎯 Conclusion

This corrected unified specification ensures full architectural compliance while delivering the transformative intent layer experience. Critical corrections:

1. **No INDEX Intent**: Replaced with transparent `AutoIndexingService` background operation
2. **Service Compliance**: All components extend `BaseServiceProvider` with proper lifecycle
3. **Factory Integration**: Uses existing `ExtensibilityManager` for strategy registration
4. **Configuration Hierarchy**: Properly extends `ServicesConfig` without conflicts
5. **FastMCP Integration**: Uses existing `ServiceBridge` patterns with no breaking changes
6. **Error Handling**: Follows established exception hierarchy and recovery patterns (e.g. services exceptions in `codeweaver.types/services/exceptions`)
7. **Testing Framework**: Integrates with existing testing utilities and patterns

**Expected Outcome**: A production-ready intent layer that transforms the LLM user experience from 4 complex tools to 1-2 intuitive tools, with completely transparent background indexing, while preserving the architectural integrity and extensibility that makes CodeWeaver powerful for developers.
