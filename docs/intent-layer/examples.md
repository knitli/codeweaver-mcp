<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Examples

This cookbook provides practical examples demonstrating how to use, configure, and extend the Intent Layer for various use cases.

## Basic Usage Examples

### Simple Code Search

```python
# Basic natural language search
user_input = "Find all authentication functions in the project"

# The Intent Layer processes this as:
# - Intent Type: SEARCH
# - Target: "authentication functions"
# - Scope: PROJECT
# - Complexity: MODERATE

# Expected result structure:
{
    "success": True,
    "data": {
        "results": [
            {
                "file_path": "auth/authentication.py",
                "function_name": "authenticate_user",
                "line_range": [45, 67],
                "relevance_score": 0.95
            },
            {
                "file_path": "middleware/auth_middleware.py", 
                "function_name": "verify_token",
                "line_range": [23, 34],
                "relevance_score": 0.87
            }
        ],
        "total_found": 12,
        "search_strategy": "semantic_with_structural"
    },
    "strategy_used": "SimpleSearchStrategy",
    "execution_time": 1.2
}
```

### Architecture Understanding

```python
# Understanding system architecture
user_input = "How does the payment processing system work?"

# Processed as:
# - Intent Type: UNDERSTAND
# - Target: "payment processing system" 
# - Scope: SYSTEM
# - Complexity: COMPLEX

# Expected result with workflow orchestration:
{
    "success": True,
    "data": {
        "architecture_overview": {
            "components": [
                "PaymentController",
                "PaymentService", 
                "PaymentGateway",
                "TransactionLogger"
            ],
            "flow_diagram": "...",
            "integration_points": [
                "Stripe API",
                "Database transactions",
                "Audit logging"
            ]
        },
        "key_files": [
            "payments/controller.py",
            "payments/service.py",
            "payments/models.py"
        ],
        "dependencies": {
            "external": ["stripe", "requests"],
            "internal": ["auth", "logging", "database"]
        }
    },
    "strategy_used": "AnalysisWorkflowStrategy",
    "execution_time": 8.4
}
```

### Security Analysis

```python
# Security-focused analysis
user_input = "Check for SQL injection vulnerabilities in user input handling"

# Processed as:
# - Intent Type: ANALYZE
# - Target: "SQL injection vulnerabilities user input handling"
# - Scope: PROJECT
# - Complexity: COMPLEX

# Multi-step analysis result:
{
    "success": True,
    "data": {
        "vulnerability_scan": {
            "high_risk": [
                {
                    "file": "api/user_controller.py",
                    "line": 156,
                    "issue": "Direct string interpolation in SQL query",
                    "severity": "HIGH",
                    "recommendation": "Use parameterized queries"
                }
            ],
            "medium_risk": [
                {
                    "file": "models/user.py",
                    "line": 89,
                    "issue": "Potential SQL injection in dynamic query building",
                    "severity": "MEDIUM", 
                    "recommendation": "Validate input parameters"
                }
            ],
            "low_risk": []
        },
        "security_score": 6.5,
        "files_analyzed": 45,
        "patterns_checked": ["sql_injection", "xss", "auth_bypass"]
    },
    "suggestions": [
        "Review high-risk findings immediately",
        "Consider using an ORM for database queries",
        "Implement input validation middleware"
    ],
    "strategy_used": "SecurityAnalysisStrategy",
    "execution_time": 15.7
}
```

## Configuration Examples

### Custom Pattern Configuration

```toml
# intent_patterns.toml
[patterns.search]
# E-commerce specific patterns
product_patterns = [
    "(?:find|search|locate)\\s+products?\\s+(.+)",
    "show\\s+(?:me\\s+)?(?:all\\s+)?products?\\s+(?:for|with|matching)\\s+(.+)",
    "products?\\s+(?:in|from)\\s+category\\s+(.+)"
]

order_patterns = [
    "(?:find|get|show)\\s+orders?\\s+(.+)",
    "orders?\\s+(?:for|by|from)\\s+(?:customer|user)\\s+(.+)",
    "(?:recent|latest|new)\\s+orders?"
]

# Custom confidence scoring
[pattern_scoring]
product_base_confidence = 0.8
order_base_confidence = 0.9
search_base_confidence = 0.7

# E-commerce patterns get priority
product_multiplier = 1.3
order_multiplier = 1.2
```

### Multi-Environment Configuration

```toml
# Development environment
[development.services.intent]
provider = "intent_orchestrator"
confidence_threshold = 0.4  # Lower for experimentation
max_execution_time = 60.0   # Longer for debugging
cache_ttl = 300            # Shorter cache

[development.services.intent.logging]
level = "DEBUG"
log_parsed_intents = true
log_strategy_selection = true

# Production environment  
[production.services.intent]
provider = "intent_orchestrator"
confidence_threshold = 0.7  # Higher confidence required
max_execution_time = 15.0   # Strict timeout
cache_ttl = 7200           # Longer cache

[production.services.intent.performance]
circuit_breaker_enabled = true
circuit_breaker_threshold = 3
max_concurrent_intents = 20

# Testing environment
[testing.services.intent]
provider = "intent_orchestrator"
confidence_threshold = 0.5
max_execution_time = 30.0

[testing.services.implicit_learning]
learning_enabled = false    # Disable learning in tests
pattern_recognition = false
```

## Custom Strategy Examples

### Domain-Specific Strategy

```python
# E-commerce specific strategy
class ECommerceAnalysisStrategy(BaseServiceProvider, IntentStrategy):
    """Strategy specialized for e-commerce domain analysis"""
    
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.ecommerce_keywords = [
            "product", "order", "customer", "inventory", "cart", 
            "checkout", "payment", "shipping", "catalog"
        ]
        self.business_metrics = [
            "conversion_rate", "cart_abandonment", "customer_lifetime_value"
        ]
        
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        score = 0.0
        target_lower = parsed_intent.primary_target.lower()
        
        # High confidence for e-commerce analysis
        if parsed_intent.intent_type == IntentType.ANALYZE:
            score += 0.2
            
        # Check for e-commerce keywords
        ecommerce_matches = sum(
            1 for keyword in self.ecommerce_keywords 
            if keyword in target_lower
        )
        
        if ecommerce_matches > 0:
            score += min(ecommerce_matches * 0.25, 0.7)
            
        # Bonus for business metrics
        metrics_matches = sum(
            1 for metric in self.business_metrics
            if metric.replace("_", " ") in target_lower
        )
        
        if metrics_matches > 0:
            score += 0.1
            
        return min(score, 1.0)
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        search_service = context["search_service"]
        
        # E-commerce specific search filters
        search_filters = {
            "file_patterns": ["*model*", "*service*", "*controller*"],
            "include_paths": ["ecommerce/", "shop/", "store/"],
            "exclude_patterns": ["test", "mock", "fixture"]
        }
        
        # Search for e-commerce components
        results = await search_service.search(
            parsed_intent.primary_target,
            filters=search_filters,
            max_results=50
        )
        
        # Analyze e-commerce patterns
        analysis = await self._analyze_ecommerce_patterns(results)
        
        # Generate business insights
        insights = await self._generate_business_insights(analysis)
        
        return IntentResult(
            success=True,
            data={
                "ecommerce_analysis": analysis,
                "business_insights": insights,
                "components_found": len(results),
                "patterns_analyzed": len(analysis.get("patterns", [])),
                "recommendations": self._generate_recommendations(analysis)
            },
            metadata={
                "strategy": "ecommerce_analysis",
                "domain": "ecommerce",
                "analysis_type": "business_focused"
            },
            executed_at=datetime.now(),
            execution_time=0.0,
            strategy_used="ECommerceAnalysisStrategy"
        )
        
    async def _analyze_ecommerce_patterns(self, results: list[dict]) -> dict[str, Any]:
        """Analyze e-commerce specific patterns"""
        patterns = {
            "product_management": [],
            "order_processing": [],
            "customer_management": [],
            "inventory_tracking": [],
            "payment_processing": []
        }
        
        for result in results:
            content = result.get("content", "").lower()
            
            if any(keyword in content for keyword in ["product", "catalog", "inventory"]):
                patterns["product_management"].append(result)
            elif any(keyword in content for keyword in ["order", "checkout", "cart"]):
                patterns["order_processing"].append(result)
            elif any(keyword in content for keyword in ["customer", "user", "account"]):
                patterns["customer_management"].append(result)
            elif any(keyword in content for keyword in ["payment", "billing", "stripe"]):
                patterns["payment_processing"].append(result)
                
        return patterns
        
    async def _generate_business_insights(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate business-focused insights"""
        insights = {
            "architecture_health": "good",
            "scalability_concerns": [],
            "performance_opportunities": [],
            "security_considerations": []
        }
        
        # Analyze component distribution
        component_counts = {k: len(v) for k, v in analysis.items()}
        
        if component_counts.get("payment_processing", 0) < 3:
            insights["scalability_concerns"].append(
                "Limited payment processing components may impact scalability"
            )
            
        if component_counts.get("product_management", 0) > 20:
            insights["performance_opportunities"].append(
                "Large number of product management files suggests potential for optimization"
            )
            
        return insights
        
    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        component_counts = {k: len(v) for k, v in analysis.items()}
        
        if component_counts.get("order_processing", 0) < 5:
            recommendations.append(
                "Consider consolidating order processing logic for better maintainability"
            )
            
        if component_counts.get("customer_management", 0) > 15:
            recommendations.append(
                "Customer management components could benefit from refactoring"
            )
            
        return recommendations
```

### Workflow-Based Strategy

```python
class ComprehensiveCodeReviewStrategy(BaseServiceProvider, IntentStrategy):
    """Multi-step workflow strategy for comprehensive code reviews"""
    
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        score = 0.0
        target_lower = parsed_intent.primary_target.lower()
        
        review_keywords = ["review", "audit", "quality", "comprehensive", "analyze"]
        
        if parsed_intent.intent_type == IntentType.ANALYZE:
            score += 0.3
            
        keyword_matches = sum(1 for keyword in review_keywords if keyword in target_lower)
        if keyword_matches > 0:
            score += min(keyword_matches * 0.2, 0.6)
            
        if parsed_intent.complexity in [Complexity.COMPLEX, Complexity.ADAPTIVE]:
            score += 0.1
            
        return min(score, 1.0)
        
    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        # Define comprehensive review workflow
        workflow = WorkflowDefinition(
            name="comprehensive_code_review",
            steps=[
                WorkflowStep(
                    name="initial_discovery",
                    handler=self._discovery_step,
                    timeout=20.0
                ),
                WorkflowStep(
                    name="security_analysis",
                    handler=self._security_analysis_step,
                    dependencies=["initial_discovery"],
                    timeout=30.0
                ),
                WorkflowStep(
                    name="quality_analysis", 
                    handler=self._quality_analysis_step,
                    dependencies=["initial_discovery"],
                    timeout=25.0
                ),
                WorkflowStep(
                    name="performance_analysis",
                    handler=self._performance_analysis_step,
                    dependencies=["initial_discovery"],
                    timeout=20.0
                ),
                WorkflowStep(
                    name="generate_report",
                    handler=self._report_generation_step,
                    dependencies=["security_analysis", "quality_analysis", "performance_analysis"],
                    timeout=15.0
                )
            ],
            allow_partial_success=True,
            max_parallel_steps=3
        )
        
        # Execute workflow
        workflow_orchestrator = context["workflow_orchestrator"]
        workflow_result = await workflow_orchestrator.execute_workflow(workflow, context)
        
        return self._transform_workflow_result(workflow_result, parsed_intent)
        
    async def _discovery_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Initial code discovery and categorization"""
        search_service = context["search_service"]
        
        # Broad search to understand codebase
        results = await search_service.search(
            context["parsed_intent"].primary_target,
            filters={"max_results": 100},
        )
        
        # Categorize files by type
        categories = {
            "controllers": [],
            "models": [],
            "services": [],
            "utilities": [],
            "tests": [],
            "configuration": []
        }
        
        for result in results:
            file_path = result.get("file_path", "").lower()
            
            if "controller" in file_path or "handler" in file_path:
                categories["controllers"].append(result)
            elif "model" in file_path or "entity" in file_path:
                categories["models"].append(result)
            elif "service" in file_path:
                categories["services"].append(result)
            elif "test" in file_path:
                categories["tests"].append(result)
            elif "config" in file_path or "setting" in file_path:
                categories["configuration"].append(result)
            else:
                categories["utilities"].append(result)
                
        return {
            "categories": categories,
            "total_files": len(results),
            "coverage_estimate": min(len(results) / 50, 1.0)  # Rough coverage estimate
        }
        
    async def _security_analysis_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Security-focused analysis"""
        discovery_data = context["step_results"]["initial_discovery"]
        
        security_issues = []
        security_patterns = [
            "password", "token", "secret", "auth", "permission",
            "sql", "query", "input", "validation"
        ]
        
        # Analyze each category for security concerns
        for category, files in discovery_data["categories"].items():
            if category == "tests":  # Skip test files
                continue
                
            for file_info in files[:10]:  # Limit analysis
                content = file_info.get("content", "").lower()
                
                for pattern in security_patterns:
                    if pattern in content:
                        security_issues.append({
                            "file": file_info.get("file_path"),
                            "category": category,
                            "pattern": pattern,
                            "severity": self._assess_security_severity(pattern, content),
                            "line_estimate": content.count('\n') + 1
                        })
                        
        return {
            "security_issues": security_issues,
            "high_priority_count": len([i for i in security_issues if i["severity"] == "HIGH"]),
            "categories_analyzed": len([c for c in discovery_data["categories"] if c != "tests"])
        }
        
    async def _quality_analysis_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Code quality analysis"""
        discovery_data = context["step_results"]["initial_discovery"]
        
        quality_metrics = {
            "large_files": [],
            "complexity_concerns": [],
            "documentation_gaps": [],
            "naming_issues": []
        }
        
        for category, files in discovery_data["categories"].items():
            for file_info in files[:10]:
                content = file_info.get("content", "")
                line_count = content.count('\n') + 1
                
                # Large file detection
                if line_count > 500:
                    quality_metrics["large_files"].append({
                        "file": file_info.get("file_path"),
                        "lines": line_count,
                        "category": category
                    })
                    
                # Documentation gap detection
                if line_count > 50 and not any(doc_marker in content.lower() 
                                             for doc_marker in ['"""', "'''", "# ", "/**"]):
                    quality_metrics["documentation_gaps"].append({
                        "file": file_info.get("file_path"),
                        "category": category
                    })
                    
        return quality_metrics
        
    async def _performance_analysis_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Performance analysis"""
        discovery_data = context["step_results"]["initial_discovery"]
        
        performance_concerns = []
        performance_patterns = [
            ("loop", "nested_loops"),
            ("query", "database_queries"),
            ("request", "http_requests"),
            ("cache", "caching_usage")
        ]
        
        for category, files in discovery_data["categories"].items():
            if category in ["controllers", "services"]:  # Focus on key areas
                for file_info in files[:10]:
                    content = file_info.get("content", "").lower()
                    
                    for pattern, concern_type in performance_patterns:
                        if content.count(pattern) > 5:  # Threshold for concern
                            performance_concerns.append({
                                "file": file_info.get("file_path"),
                                "concern_type": concern_type,
                                "pattern_count": content.count(pattern),
                                "category": category
                            })
                            
        return {
            "performance_concerns": performance_concerns,
            "high_impact_files": [c for c in performance_concerns if c["pattern_count"] > 10]
        }
        
    async def _report_generation_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive review report"""
        step_results = context["step_results"]
        
        # Combine all analysis results
        report = {
            "executive_summary": self._generate_executive_summary(step_results),
            "detailed_findings": {
                "security": step_results.get("security_analysis", {}),
                "quality": step_results.get("quality_analysis", {}), 
                "performance": step_results.get("performance_analysis", {})
            },
            "recommendations": self._generate_recommendations(step_results),
            "metrics": {
                "files_analyzed": step_results["initial_discovery"]["total_files"],
                "security_issues": len(step_results.get("security_analysis", {}).get("security_issues", [])),
                "quality_concerns": sum(len(v) for v in step_results.get("quality_analysis", {}).values() if isinstance(v, list)),
                "performance_concerns": len(step_results.get("performance_analysis", {}).get("performance_concerns", []))
            }
        }
        
        return report
        
    def _assess_security_severity(self, pattern: str, content: str) -> str:
        """Assess security issue severity"""
        high_risk_patterns = ["password", "secret", "token", "sql"]
        if pattern in high_risk_patterns:
            return "HIGH"
        return "MEDIUM"
        
    def _generate_executive_summary(self, step_results: dict[str, Any]) -> str:
        """Generate executive summary"""
        discovery = step_results["initial_discovery"]
        security = step_results.get("security_analysis", {})
        quality = step_results.get("quality_analysis", {})
        performance = step_results.get("performance_analysis", {})
        
        total_files = discovery["total_files"]
        security_issues = len(security.get("security_issues", []))
        high_priority_security = security.get("high_priority_count", 0)
        
        summary = f"""
Code Review Summary:
- Analyzed {total_files} files across multiple categories
- Found {security_issues} security-related items ({high_priority_security} high priority)
- Identified {len(quality.get("large_files", []))} large files that may benefit from refactoring
- Detected {len(performance.get("performance_concerns", []))} potential performance concerns

Overall Assessment: {"Good" if security_issues < 5 else "Needs Attention"}
        """.strip()
        
        return summary
        
    def _generate_recommendations(self, step_results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        security = step_results.get("security_analysis", {})
        quality = step_results.get("quality_analysis", {})
        performance = step_results.get("performance_analysis", {})
        
        if security.get("high_priority_count", 0) > 0:
            recommendations.append("Address high-priority security issues immediately")
            
        if len(quality.get("large_files", [])) > 5:
            recommendations.append("Consider refactoring large files for better maintainability")
            
        if len(quality.get("documentation_gaps", [])) > 10:
            recommendations.append("Improve code documentation coverage")
            
        if len(performance.get("high_impact_files", [])) > 0:
            recommendations.append("Review high-impact files for performance optimization")
            
        return recommendations
        
    def _transform_workflow_result(self, workflow_result, parsed_intent: ParsedIntent) -> IntentResult:
        """Transform workflow result to IntentResult"""
        if workflow_result.success:
            report_data = workflow_result.results.get("generate_report", {})
            
            return IntentResult(
                success=True,
                data=report_data,
                metadata={
                    "strategy": "comprehensive_code_review",
                    "workflow_steps": len(workflow_result.results),
                    "partial_success": len(workflow_result.results) < 5
                },
                executed_at=datetime.now(),
                execution_time=0.0,
                suggestions=[
                    "Review detailed findings for prioritization",
                    "Implement recommendations based on team capacity",
                    "Schedule follow-up review in 3-6 months"
                ],
                strategy_used="ComprehensiveCodeReviewStrategy"
            )
        else:
            return IntentResult(
                success=False,
                data=workflow_result.partial_results,
                metadata={"strategy": "comprehensive_code_review", "workflow_error": True},
                executed_at=datetime.now(),
                execution_time=0.0,
                error_message=workflow_result.error or "Workflow execution failed",
                suggestions=["Try a simpler analysis approach", "Check system resources"],
                strategy_used="ComprehensiveCodeReviewStrategy"
            )
```

## Integration Examples

### FastMCP Integration

```python
# Integrating Intent Layer with FastMCP application
from fastmcp import FastMCPApplication
from codeweaver.intent.middleware.intent_bridge import IntentServiceBridge

async def setup_intent_layer_integration(app: FastMCPApplication):
    """Setup Intent Layer with FastMCP application"""
    
    # Register intent services
    intent_service = IntentOrchestratorService(app.config.services.intent)
    context_intelligence = ContextIntelligenceService(app.config.services.context_intelligence)
    implicit_learning = ImplicitLearningService(app.config.services.implicit_learning)
    
    await app.register_service("intent", intent_service)
    await app.register_service("context_intelligence", context_intelligence)
    await app.register_service("implicit_learning", implicit_learning)
    
    # Register intent bridge middleware
    intent_bridge = IntentServiceBridge(
        intent_service=intent_service,
        context_intelligence=context_intelligence,
        implicit_learning=implicit_learning
    )
    app.add_middleware(IntentBridgeMiddleware(intent_bridge))
    
    # Register MCP tools that use Intent Layer
    @app.tool("smart_code_search")
    async def smart_code_search(
        query: str,
        ctx: FastMCPContext
    ) -> dict[str, Any]:
        """Intelligent code search using natural language"""
        intent_service = ctx.get_service("intent")
        result = await intent_service.process_intent(query, ctx)
        return result.to_dict()
        
    @app.tool("code_analysis")
    async def code_analysis(
        analysis_request: str,
        ctx: FastMCPContext
    ) -> dict[str, Any]:
        """Comprehensive code analysis using natural language"""
        intent_service = ctx.get_service("intent")
        result = await intent_service.process_intent(analysis_request, ctx)
        return result.to_dict()
        
    @app.tool("architecture_exploration")
    async def architecture_exploration(
        exploration_query: str,
        ctx: FastMCPContext
    ) -> dict[str, Any]:
        """Explore and understand code architecture"""
        intent_service = ctx.get_service("intent")
        result = await intent_service.process_intent(exploration_query, ctx)
        return result.to_dict()
```

### Plugin Registration Example

```python
# setup.py entry points for plugin registration
setup(
    name="my-codeweaver-plugin",
    packages=find_packages(),
    entry_points={
        "codeweaver.intent_strategies": [
            "ecommerce_analysis = myorg.strategies.ecommerce:ECommerceAnalysisStrategy",
            "security_audit = myorg.strategies.security:SecurityAuditStrategy",
            "performance_review = myorg.strategies.performance:PerformanceReviewStrategy",
        ],
        "codeweaver.intent_parsers": [
            "domain_aware_parser = myorg.parsers.domain:DomainAwareParser",
        ]
    }
)

# Alternative: Runtime registration
async def register_custom_strategies(extensibility_manager: ExtensibilityManager):
    """Register custom strategies at runtime"""
    
    # Register e-commerce strategy
    await extensibility_manager.register_component(
        name="ecommerce_analysis_strategy",
        component_class=ECommerceAnalysisStrategy,
        component_type="intent_strategy",
        metadata={
            "intent_types": ["ANALYZE", "UNDERSTAND"],
            "specialization": "ecommerce",
            "priority": 0.9,
            "description": "E-commerce domain analysis and insights"
        }
    )
    
    # Register comprehensive review strategy
    await extensibility_manager.register_component(
        name="comprehensive_review_strategy", 
        component_class=ComprehensiveCodeReviewStrategy,
        component_type="intent_strategy",
        metadata={
            "intent_types": ["ANALYZE"],
            "specialization": "code_review",
            "priority": 0.85,
            "description": "Multi-step comprehensive code review"
        }
    )
```

## Testing Examples

### Strategy Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

@pytest.mark.asyncio
async def test_ecommerce_strategy_identification():
    """Test e-commerce strategy can identify relevant intents"""
    strategy = ECommerceAnalysisStrategy()
    
    # Test high confidence for e-commerce analysis
    ecommerce_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="analyze product catalog performance",
        scope=Scope.PROJECT,
        complexity=Complexity.MODERATE,
        confidence=0.8,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    confidence = await strategy.can_handle(ecommerce_intent)
    assert confidence > 0.7, f"Expected high confidence, got {confidence}"
    
    # Test low confidence for non-e-commerce analysis
    general_intent = ParsedIntent(
        intent_type=IntentType.UNDERSTAND,
        primary_target="explain database schema",
        scope=Scope.MODULE,
        complexity=Complexity.SIMPLE,
        confidence=0.7,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    confidence = await strategy.can_handle(general_intent)
    assert confidence < 0.3, f"Expected low confidence, got {confidence}"

@pytest.mark.asyncio
async def test_ecommerce_strategy_execution():
    """Test e-commerce strategy execution"""
    strategy = ECommerceAnalysisStrategy()
    
    # Mock search service
    mock_search_service = AsyncMock()
    mock_search_service.search.return_value = [
        {
            "file_path": "ecommerce/models/product.py",
            "content": "class Product: def calculate_price(self): pass",
            "relevance_score": 0.9
        },
        {
            "file_path": "ecommerce/services/order_service.py", 
            "content": "class OrderService: def process_order(self): pass",
            "relevance_score": 0.85
        }
    ]
    
    context = {"search_service": mock_search_service}
    
    ecommerce_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="analyze product management system",
        scope=Scope.PROJECT,
        complexity=Complexity.MODERATE,
        confidence=0.8,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    result = await strategy.execute(ecommerce_intent, context)
    
    assert result.success
    assert result.strategy_used == "ECommerceAnalysisStrategy"
    assert "ecommerce_analysis" in result.data
    assert "business_insights" in result.data
    assert result.data["components_found"] == 2

@pytest.mark.asyncio
async def test_workflow_strategy_execution():
    """Test workflow-based strategy execution"""
    strategy = ComprehensiveCodeReviewStrategy()
    
    # Mock workflow orchestrator
    mock_orchestrator = AsyncMock()
    mock_workflow_result = MagicMock()
    mock_workflow_result.success = True
    mock_workflow_result.results = {
        "initial_discovery": {
            "categories": {"controllers": [], "models": []},
            "total_files": 25
        },
        "security_analysis": {
            "security_issues": [],
            "high_priority_count": 0
        },
        "quality_analysis": {
            "large_files": [],
            "documentation_gaps": []
        },
        "performance_analysis": {
            "performance_concerns": []
        },
        "generate_report": {
            "executive_summary": "Test summary",
            "recommendations": ["Test recommendation"]
        }
    }
    mock_orchestrator.execute_workflow.return_value = mock_workflow_result
    
    context = {
        "workflow_orchestrator": mock_orchestrator,
        "search_service": AsyncMock()
    }
    
    review_intent = ParsedIntent(
        intent_type=IntentType.ANALYZE,
        primary_target="comprehensive code review",
        scope=Scope.PROJECT,
        complexity=Complexity.COMPLEX,
        confidence=0.9,
        filters={},
        metadata={},
        parsed_at=datetime.now()
    )
    
    result = await strategy.execute(review_intent, context)
    
    assert result.success
    assert result.strategy_used == "ComprehensiveCodeReviewStrategy"
    assert "executive_summary" in result.data
    assert len(result.suggestions) > 0
```

### Configuration Testing

```python
@pytest.mark.asyncio
async def test_custom_pattern_configuration():
    """Test custom pattern configuration"""
    
    # Test configuration loading
    config = {
        "patterns": {
            "search": {
                "product_patterns": [
                    "find products (.+)",
                    "show products for (.+)"
                ]
            }
        },
        "scoring": {
            "product_base_confidence": 0.8,
            "product_multiplier": 1.2
        }
    }
    
    parser = PatternBasedParser(config)
    
    # Test pattern matching
    intent = await parser.parse("find products in electronics category")
    
    assert intent.intent_type == IntentType.SEARCH
    assert "electronics category" in intent.primary_target
    assert intent.confidence > 0.7

@pytest.mark.asyncio 
async def test_multi_environment_configuration():
    """Test configuration for different environments"""
    
    # Development configuration
    dev_config = IntentServiceConfig(
        confidence_threshold=0.4,
        max_execution_time=60.0,
        circuit_breaker_enabled=False
    )
    
    assert dev_config.confidence_threshold == 0.4
    assert dev_config.circuit_breaker_enabled is False
    
    # Production configuration
    prod_config = IntentServiceConfig(
        confidence_threshold=0.7,
        max_execution_time=15.0,
        circuit_breaker_enabled=True,
        circuit_breaker_threshold=3
    )
    
    assert prod_config.confidence_threshold == 0.7
    assert prod_config.circuit_breaker_enabled is True
    assert prod_config.circuit_breaker_threshold == 3
```

## Performance Examples

### Caching Strategy

```python
# Example of implementing custom caching strategy
class DomainAwareCacheService(BaseServiceProvider):
    """Cache service that considers domain context"""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.domain_ttls = {
            "ecommerce": 7200,  # 2 hours for e-commerce
            "security": 3600,   # 1 hour for security
            "general": 1800     # 30 minutes for general
        }
        
    async def get_cache_key(self, user_input: str, parsed_intent: ParsedIntent) -> str:
        """Generate domain-aware cache key"""
        domain = self._detect_domain(parsed_intent.primary_target)
        intent_hash = hash(f"{parsed_intent.intent_type.value}:{parsed_intent.primary_target}")
        
        return f"{domain}:{parsed_intent.scope.value}:{intent_hash}"
        
    async def get_cache_ttl(self, parsed_intent: ParsedIntent) -> int:
        """Get domain-specific cache TTL"""
        domain = self._detect_domain(parsed_intent.primary_target)
        return self.domain_ttls.get(domain, self.domain_ttls["general"])
        
    def _detect_domain(self, target: str) -> str:
        """Detect domain from intent target"""
        target_lower = target.lower()
        
        if any(keyword in target_lower for keyword in ["product", "order", "customer", "cart"]):
            return "ecommerce"
        elif any(keyword in target_lower for keyword in ["security", "auth", "vulnerability"]):
            return "security"
        else:
            return "general"
```

### Circuit Breaker Configuration

```python
# Example circuit breaker configuration for different strategies
strategy_circuit_breakers = {
    "security_analysis": {
        "failure_threshold": 3,  # Security analysis is critical
        "timeout": 45.0,         # Longer timeout for complex analysis
        "recovery_timeout": 300.0
    },
    "simple_search": {
        "failure_threshold": 10, # More tolerant for simple operations
        "timeout": 10.0,         # Shorter timeout
        "recovery_timeout": 60.0
    },
    "comprehensive_review": {
        "failure_threshold": 2,  # Low tolerance for complex workflows
        "timeout": 120.0,        # Long timeout for comprehensive analysis
        "recovery_timeout": 600.0
    }
}

# Usage in strategy registry
class StrategyRegistryWithCircuitBreakers(StrategyRegistry):
    async def execute_strategy_with_protection(
        self,
        strategy: IntentStrategy,
        parsed_intent: ParsedIntent,
        context: dict[str, Any]
    ) -> IntentResult:
        """Execute strategy with circuit breaker protection"""
        
        strategy_name = strategy.__class__.__name__
        cb_config = strategy_circuit_breakers.get(
            strategy_name.lower().replace("strategy", ""),
            strategy_circuit_breakers["simple_search"]
        )
        
        circuit_breaker = IntentCircuitBreaker(cb_config)
        
        try:
            result = await circuit_breaker.call(strategy, parsed_intent, context)
            return result
        except CircuitBreakerOpenError:
            # Fallback to adaptive strategy
            adaptive_strategy = self.strategies.get("adaptive")
            if adaptive_strategy:
                return await adaptive_strategy.execute(parsed_intent, context)
            else:
                raise
```

## Monitoring and Observability Examples

### Custom Metrics Collection

```python
class IntentMetricsCollector:
    """Collect custom metrics for intent processing"""
    
    def __init__(self):
        self.metrics = {
            "intent_types": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "execution_times": defaultdict(list),
            "confidence_scores": [],
            "success_rates": defaultdict(lambda: {"success": 0, "total": 0})
        }
        
    async def record_intent_processing(
        self,
        parsed_intent: ParsedIntent,
        result: IntentResult
    ) -> None:
        """Record metrics for intent processing"""
        
        # Count intent types
        self.metrics["intent_types"][parsed_intent.intent_type.value] += 1
        
        # Track strategy usage
        if result.strategy_used:
            self.metrics["strategy_usage"][result.strategy_used] += 1
            
        # Record execution times
        self.metrics["execution_times"][result.strategy_used or "unknown"].append(
            result.execution_time
        )
        
        # Track confidence scores
        self.metrics["confidence_scores"].append(parsed_intent.confidence)
        
        # Update success rates
        strategy_key = result.strategy_used or "unknown"
        self.metrics["success_rates"][strategy_key]["total"] += 1
        if result.success:
            self.metrics["success_rates"][strategy_key]["success"] += 1
            
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summarized metrics"""
        summary = {
            "intent_type_distribution": dict(self.metrics["intent_types"]),
            "strategy_usage_distribution": dict(self.metrics["strategy_usage"]),
            "average_confidence": statistics.mean(self.metrics["confidence_scores"]) if self.metrics["confidence_scores"] else 0,
            "strategy_performance": {}
        }
        
        # Calculate strategy performance
        for strategy, times in self.metrics["execution_times"].items():
            if times:
                success_data = self.metrics["success_rates"][strategy]
                success_rate = success_data["success"] / success_data["total"] if success_data["total"] > 0 else 0
                
                summary["strategy_performance"][strategy] = {
                    "avg_execution_time": statistics.mean(times),
                    "success_rate": success_rate,
                    "total_executions": len(times)
                }
                
        return summary
```

## Next Steps

1. **[Configuration](configuration.md)**: Learn advanced configuration options
2. **[Custom Strategies](custom-strategies.md)**: Develop your own intent strategies  
3. **[API Reference](api-reference.md)**: Detailed interface documentation
4. **[Architecture](architecture.md)**: Understanding the technical implementation