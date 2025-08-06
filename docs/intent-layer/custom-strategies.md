<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Custom Strategies Guide

This guide shows you how to develop custom strategies for the Intent Layer, enabling specialized handling of domain-specific use cases and advanced processing workflows.

## Understanding Intent Strategies

Intent strategies are the core execution engines that transform parsed intents into concrete results. Each strategy implements the `IntentStrategy` protocol and specializes in handling specific types of requests.

### Strategy Architecture

```python
from codeweaver.cw_types import IntentStrategy, ParsedIntent, IntentResult

class CustomStrategy(IntentStrategy):
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """Return confidence score (0.0-1.0) for handling this intent"""
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """Execute the intent and return structured results"""
```

## Basic Strategy Development

### Step 1: Create Strategy Class

```python
from typing import Any
from datetime import datetime
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.cw_types import IntentStrategy, ParsedIntent, IntentResult, IntentType, Complexity

class SecurityAnalysisStrategy(BaseServiceProvider, IntentStrategy):
    """Specialized strategy for security-focused code analysis"""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.security_patterns = [
            "sql injection", "xss", "csrf", "authentication",
            "authorization", "vulnerability", "security"
        ]
        
    async def _initialize_provider(self) -> None:
        """Initialize any required resources"""
        self.logger.info("SecurityAnalysisStrategy initialized")
        
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """Evaluate if this strategy can handle the parsed intent"""
        score = 0.0
        
        # High confidence for security-related analysis
        if parsed_intent.intent_type == IntentType.ANALYZE:
            score += 0.3
            
        # Check for security keywords in target
        target_lower = parsed_intent.primary_target.lower()
        security_matches = sum(1 for pattern in self.security_patterns 
                             if pattern in target_lower)
        
        if security_matches > 0:
            score += min(security_matches * 0.2, 0.6)
            
        # Bonus for complex analysis requests
        if parsed_intent.complexity in [Complexity.COMPLEX, Complexity.ADAPTIVE]:
            score += 0.1
            
        return min(score, 1.0)
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """Execute security analysis strategy"""
        start_time = datetime.now()
        
        try:
            # Get required services from context
            search_service = context.get("search_service")
            if not search_service:
                raise ValueError("Search service not available in context")
                
            # Perform security-focused search
            security_results = await self._perform_security_search(
                parsed_intent.primary_target, search_service
            )
            
            # Analyze results for security patterns
            analysis_results = await self._analyze_security_patterns(security_results)
            
            # Generate security report
            report = await self._generate_security_report(analysis_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return IntentResult(
                success=True,
                data=report,
                metadata={
                    "strategy": "security_analysis",
                    "results_count": len(security_results),
                    "vulnerabilities_found": len(analysis_results.get("vulnerabilities", [])),
                    "confidence": parsed_intent.confidence
                },
                executed_at=datetime.now(),
                execution_time=execution_time,
                error_message=None,
                suggestions=self._generate_security_suggestions(analysis_results),
                strategy_used="SecurityAnalysisStrategy"
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.exception("Security analysis failed")
            
            return IntentResult(
                success=False,
                data=None,
                metadata={"strategy": "security_analysis", "error": str(e)},
                executed_at=datetime.now(),
                execution_time=execution_time,
                error_message=str(e),
                suggestions=["Try a simpler search query", "Check system logs for details"],
                strategy_used="SecurityAnalysisStrategy"
            )
    
    async def _perform_security_search(self, target: str, search_service) -> list[dict]:
        """Perform security-focused code search"""
        # Implementation details...
        pass
        
    async def _analyze_security_patterns(self, results: list[dict]) -> dict[str, Any]:
        """Analyze search results for security patterns"""
        # Implementation details...
        pass
        
    async def _generate_security_report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive security report"""
        # Implementation details...
        pass
        
    async def _generate_security_suggestions(self, analysis: dict[str, Any]) -> list[str]:
        """Generate actionable security suggestions"""
        # Implementation details...
        pass
```

### Step 2: Register Strategy

```python
# In your plugin or configuration
from codeweaver.factories.extensibility_manager import ExtensibilityManager

async def register_security_strategy(extensibility_manager: ExtensibilityManager):
    """Register the custom security strategy"""
    await extensibility_manager.register_component(
        name="security_analysis_strategy",
        component_class=SecurityAnalysisStrategy,
        component_type="intent_strategy",
        metadata={
            "intent_types": ["ANALYZE"],
            "specialization": "security",
            "priority": 0.95,
            "description": "Specialized security analysis for code vulnerabilities"
        }
    )
```

## Advanced Strategy Patterns

### Multi-Step Workflow Strategy

```python
from codeweaver.cw_types import WorkflowStep, WorkflowDefinition

class ComprehensiveAnalysisStrategy(BaseServiceProvider, IntentStrategy):
    """Multi-step analysis strategy with workflow orchestration"""
    
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        # Define analysis workflow
        workflow = WorkflowDefinition(
            name="comprehensive_analysis",
            steps=[
                WorkflowStep(
                    name="initial_search",
                    handler=self._initial_search_step,
                    timeout=15.0,
                    required=True
                ),
                WorkflowStep(
                    name="pattern_analysis", 
                    handler=self._pattern_analysis_step,
                    dependencies=["initial_search"],
                    timeout=20.0,
                    required=False
                ),
                WorkflowStep(
                    name="ast_analysis",
                    handler=self._ast_analysis_step,
                    dependencies=["initial_search"],
                    timeout=25.0,
                    required=False
                ),
                WorkflowStep(
                    name="generate_report",
                    handler=self._report_generation_step,
                    dependencies=["initial_search", "pattern_analysis", "ast_analysis"],
                    timeout=10.0,
                    required=True
                )
            ],
            allow_partial_success=True,
            max_parallel_steps=2
        )
        
        # Execute workflow
        workflow_orchestrator = context.get("workflow_orchestrator")
        workflow_result = await workflow_orchestrator.execute_workflow(workflow, context)
        
        # Transform workflow result to IntentResult
        return self._transform_workflow_result(workflow_result, parsed_intent)
        
    async def _initial_search_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """First step: perform initial semantic search"""
        # Implementation...
        pass
        
    async def _pattern_analysis_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Second step: analyze patterns in found files"""
        # Implementation...
        pass
        
    async def _ast_analysis_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Third step: perform AST-based structural analysis"""  
        # Implementation...
        pass
        
    async def _report_generation_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Final step: generate comprehensive report"""
        # Implementation...
        pass
```

### Adaptive Strategy with Learning

```python
from codeweaver.cw_types import LearningPattern

class AdaptiveLearningStrategy(BaseServiceProvider, IntentStrategy):
    """Strategy that adapts based on usage patterns and success rates"""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.success_history: dict[str, list[float]] = {}
        self.pattern_cache: dict[str, LearningPattern] = {}
        
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        # Base confidence
        base_score = 0.2  # Always available as fallback
        
        # Learning-based adjustment
        intent_key = self._get_intent_key(parsed_intent)
        if intent_key in self.success_history:
            recent_success = self.success_history[intent_key][-5:]  # Last 5 attempts
            success_rate = sum(recent_success) / len(recent_success)
            base_score += success_rate * 0.3
            
        return min(base_score, 1.0)
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        intent_key = self._get_intent_key(parsed_intent)
        
        # Check for learned patterns
        if intent_key in self.pattern_cache:
            pattern = self.pattern_cache[intent_key]
            result = await self._execute_learned_pattern(pattern, parsed_intent, context)
        else:
            result = await self._execute_adaptive_approach(parsed_intent, context)
            
        # Record success/failure for learning
        success_score = 1.0 if result.success else 0.0
        self._record_outcome(intent_key, success_score)
        
        # Update learned patterns if successful
        if result.success:
            await self._update_learned_patterns(intent_key, parsed_intent, result)
            
        return result
        
    def _get_intent_key(self, parsed_intent: ParsedIntent) -> str:
        """Generate cache key for intent pattern"""
        return f"{parsed_intent.intent_type.value}:{parsed_intent.scope.value}:{hash(parsed_intent.primary_target)}"
        
    def _record_outcome(self, intent_key: str, success_score: float) -> None:
        """Record execution outcome for learning"""
        if intent_key not in self.success_history:
            self.success_history[intent_key] = []
        self.success_history[intent_key].append(success_score)
        
        # Keep only recent history
        self.success_history[intent_key] = self.success_history[intent_key][-20:]
        
    async def _update_learned_patterns(self, intent_key: str, parsed_intent: ParsedIntent, result: IntentResult) -> None:
        """Update learned patterns based on successful execution"""
        # Implementation of pattern learning...
        pass
```

### Domain-Specific Strategy

```python
class DatabaseSchemaStrategy(BaseServiceProvider, IntentStrategy):
    """Specialized strategy for database schema analysis"""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.schema_keywords = ["table", "column", "index", "constraint", "migration", "schema"]
        self.db_file_patterns = [".sql", "migration", "schema", "models.py", "entity"]
        
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        score = 0.0
        target_lower = parsed_intent.primary_target.lower()
        
        # Check for database-related keywords
        db_matches = sum(1 for keyword in self.schema_keywords if keyword in target_lower)
        if db_matches > 0:
            score += min(db_matches * 0.25, 0.75)
            
        # Check intent type compatibility
        if parsed_intent.intent_type in [IntentType.UNDERSTAND, IntentType.ANALYZE]:
            score += 0.2
            
        return min(score, 1.0)
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        # Focus search on database-related files
        search_filters = {
            "file_patterns": self.db_file_patterns,
            "include_extensions": [".sql", ".py", ".js", ".ts"],
            "exclude_patterns": ["test", "mock", "fixture"]
        }
        
        # Perform database-focused search
        search_service = context["search_service"]
        db_results = await search_service.search(
            parsed_intent.primary_target,
            filters=search_filters,
            max_results=30
        )
        
        # Analyze database relationships
        relationships = await self._analyze_db_relationships(db_results)
        
        # Generate schema diagram
        schema_info = await self._extract_schema_info(db_results, relationships)
        
        return IntentResult(
            success=True,
            data={
                "schema_info": schema_info,
                "relationships": relationships,
                "files_analyzed": len(db_results),
                "tables_found": len(schema_info.get("tables", [])),
                "migrations_found": len([r for r in db_results if "migration" in r.get("file_path", "")])
            },
            metadata={
                "strategy": "database_schema",
                "specialization": "database",
                "analysis_type": "schema_extraction"
            },
            executed_at=datetime.now(),
            execution_time=0.0,  # Calculate actual time
            strategy_used="DatabaseSchemaStrategy"
        )
```

## Strategy Configuration

### Configuration-Driven Strategies

```python
class ConfigurableStrategy(BaseServiceProvider, IntentStrategy):
    """Strategy that adapts behavior based on configuration"""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        
        # Load configuration
        self.search_patterns = config.get("search_patterns", [])
        self.analysis_rules = config.get("analysis_rules", {})
        self.output_format = config.get("output_format", "detailed")
        self.custom_handlers = config.get("custom_handlers", {})
        
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        # Configuration-based capability assessment
        capability_rules = self.config.get("capability_rules", {})
        
        score = 0.0
        for rule_name, rule_config in capability_rules.items():
            if self._matches_rule(parsed_intent, rule_config):
                score += rule_config.get("confidence_boost", 0.1)
                
        return min(score, 1.0)
        
    def _matches_rule(self, parsed_intent: ParsedIntent, rule_config: dict) -> bool:
        """Check if intent matches configuration rule"""
        # Check intent type
        if "intent_types" in rule_config:
            if parsed_intent.intent_type.value not in rule_config["intent_types"]:
                return False
                
        # Check target patterns
        if "target_patterns" in rule_config:
            target_lower = parsed_intent.primary_target.lower()
            patterns = rule_config["target_patterns"]
            if not any(pattern in target_lower for pattern in patterns):
                return False
                
        # Check scope
        if "scopes" in rule_config:
            if parsed_intent.scope.value not in rule_config["scopes"]:
                return False
                
        return True
```

### Configuration Example

```toml
[custom_strategies.configurable_strategy]
class_path = "myorg.strategies.ConfigurableStrategy"
priority = 0.7

[custom_strategies.configurable_strategy.capability_rules]
# Security analysis rule
[custom_strategies.configurable_strategy.capability_rules.security]
intent_types = ["ANALYZE"]
target_patterns = ["security", "vulnerability", "auth", "permission"]
scopes = ["PROJECT", "SYSTEM"]
confidence_boost = 0.4

# Performance analysis rule
[custom_strategies.configurable_strategy.capability_rules.performance]
intent_types = ["ANALYZE", "UNDERSTAND"] 
target_patterns = ["performance", "slow", "bottleneck", "optimize"]
scopes = ["MODULE", "PROJECT"]
confidence_boost = 0.3

[custom_strategies.configurable_strategy.search_patterns]
security_patterns = ["auth", "login", "password", "token", "session"]
performance_patterns = ["cache", "query", "optimization", "benchmark"]

[custom_strategies.configurable_strategy.analysis_rules]
max_results = 50
include_context = true
generate_summary = true

[custom_strategies.configurable_strategy.custom_handlers]
security = "SecurityAnalysisHandler"
performance = "PerformanceAnalysisHandler"
```

## Testing Custom Strategies

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock
from codeweaver.cw_types import ParsedIntent, IntentType, Scope, Complexity

@pytest.mark.asyncio
async def test_security_strategy_can_handle():
    strategy = SecurityAnalysisStrategy()
    
    # Test high confidence for security-related analysis
    security_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="authentication vulnerabilities",
        scope=Scope.PROJECT,
        complexity=Complexity.COMPLEX,
        confidence=0.8,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    confidence = await strategy.can_handle(security_intent)
    assert confidence > 0.8
    
    # Test low confidence for non-security analysis
    general_intent = ParsedIntent(
        intent_type=IntentType.UNDERSTAND,
        primary_target="code structure",
        scope=Scope.MODULE,
        complexity=Complexity.SIMPLE,
        confidence=0.7,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    confidence = await strategy.can_handle(general_intent)
    assert confidence < 0.3

@pytest.mark.asyncio 
async def test_security_strategy_execution():
    strategy = SecurityAnalysisStrategy()
    
    # Mock search service
    mock_search_service = AsyncMock()
    mock_search_service.search.return_value = [
        {"file_path": "auth.py", "content": "def login(username, password):"},
        {"file_path": "models.py", "content": "class User: pass"}
    ]
    
    context = {"search_service": mock_search_service}
    
    security_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="authentication system",
        scope=Scope.PROJECT,
        complexity=Complexity.MODERATE,
        confidence=0.85,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    result = await strategy.execute(security_intent, context)
    
    assert result.success
    assert result.strategy_used == "SecurityAnalysisStrategy"
    assert "results_count" in result.metadata
    assert isinstance(result.data, dict)
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_strategy_integration():
    """Test strategy integration with intent layer"""
    
    # Setup test environment
    extensibility_manager = ExtensibilityManager()
    await register_security_strategy(extensibility_manager)
    
    strategy_registry = StrategyRegistry(extensibility_manager)
    await strategy_registry.initialize()
    
    # Test strategy selection
    security_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="sql injection vulnerabilities",
        scope=Scope.PROJECT,
        complexity=Complexity.COMPLEX,
        confidence=0.9,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    selected_strategy = await strategy_registry.select_strategy(security_intent)
    assert isinstance(selected_strategy, SecurityAnalysisStrategy)
    
    # Test execution through registry
    context = {"search_service": mock_search_service}
    result = await selected_strategy.execute(security_intent, context)
    
    assert result.success
    assert result.strategy_used == "SecurityAnalysisStrategy"
```

## Best Practices

### Strategy Design

1. **Single Responsibility**: Each strategy should handle one specific domain or approach
2. **Clear Confidence Scoring**: Provide accurate confidence scores for capability assessment
3. **Robust Error Handling**: Handle failures gracefully with meaningful error messages
4. **Performance Awareness**: Implement timeouts and resource limits
5. **Extensible Configuration**: Support configuration-driven behavior modification

### Implementation Guidelines

1. **Use Base Classes**: Extend `BaseServiceProvider` for lifecycle management
2. **Implement Protocols**: Ensure `IntentStrategy` protocol compliance  
3. **Async/Await**: Use async patterns for all I/O operations
4. **Structured Logging**: Use the provided logger for debugging and monitoring
5. **Metadata Rich**: Include comprehensive metadata in results

### Registration Patterns

1. **Plugin Entry Points**: Use setuptools entry points for automatic discovery
2. **Configuration Registration**: Support configuration-driven registration
3. **Runtime Registration**: Allow dynamic registration during application lifecycle
4. **Metadata Complete**: Provide comprehensive metadata for strategy discovery

## Next Steps

1. **[API Reference](api-reference.md)**: Detailed interface documentation
2. **[Architecture](architecture.md)**: Understanding the technical architecture  
3. **[Examples](examples.md)**: More practical examples and patterns
4. **[Configuration](configuration.md)**: Advanced configuration options