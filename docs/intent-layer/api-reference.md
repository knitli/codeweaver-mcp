# Intent Layer API Reference

This document provides comprehensive API documentation for all Intent Layer interfaces, protocols, and components.

## Core Protocols

### IntentStrategy Protocol

The primary interface for implementing custom intent processing strategies.

```python
@runtime_checkable
class IntentStrategy(Protocol):
    """Protocol defining the interface for intent processing strategies"""
    
    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """
        Evaluate capability to handle the given parsed intent.
        
        Args:
            parsed_intent: The structured intent to evaluate
            
        Returns:
            Confidence score from 0.0 (cannot handle) to 1.0 (perfect match)
            
        Example:
            >>> strategy = MyCustomStrategy()
            >>> intent = ParsedIntent(intent_type=IntentType.SEARCH, ...)
            >>> confidence = await strategy.can_handle(intent)
            >>> assert 0.0 <= confidence <= 1.0
        """
        ...
        
    async def execute(
        self, 
        parsed_intent: ParsedIntent, 
        context: dict[str, Any]
    ) -> IntentResult:
        """
        Execute the intent processing strategy.
        
        Args:
            parsed_intent: The structured intent to process
            context: Execution context with services and configuration
            
        Returns:
            IntentResult with processing outcome and data
            
        Raises:
            IntentProcessingError: When execution fails
            TimeoutError: When execution exceeds time limits
            
        Example:
            >>> context = {"search_service": search_service}
            >>> result = await strategy.execute(intent, context)
            >>> assert isinstance(result, IntentResult)
            >>> assert result.success in [True, False]
        """
        ...
        
    # Optional lifecycle methods
    async def initialize(self) -> None:
        """
        Initialize strategy resources.
        
        Called once during strategy registration.
        Use for setting up connections, loading models, etc.
        """
        ...
        
    async def cleanup(self) -> None:
        """
        Clean up strategy resources.
        
        Called during application shutdown.
        Use for closing connections, saving state, etc.
        """
        ...
        
    async def health_check(self) -> ServiceHealth:
        """
        Check strategy health status.
        
        Returns:
            ServiceHealth with current status and metrics
        """
        ...
```

### IntentParser Protocol

Interface for implementing custom intent parsing strategies.

```python
@runtime_checkable
class IntentParser(Protocol):
    """Protocol defining the interface for intent parsers"""
    
    async def parse(self, intent_text: str) -> ParsedIntent:
        """
        Parse natural language input into structured intent.
        
        Args:
            intent_text: Raw natural language input from user
            
        Returns:
            ParsedIntent with extracted intent components
            
        Raises:
            ParseError: When input cannot be parsed
            
        Example:
            >>> parser = PatternBasedParser()
            >>> intent = await parser.parse("find authentication functions")
            >>> assert intent.intent_type == IntentType.SEARCH
            >>> assert "authentication" in intent.primary_target
        """
        ...
        
    async def validate_intent(self, parsed_intent: ParsedIntent) -> bool:
        """
        Validate parsed intent structure and content.
        
        Args:
            parsed_intent: The intent to validate
            
        Returns:
            True if intent is valid, False otherwise
        """
        ...
        
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages for parsing.
        
        Returns:
            List of language codes (e.g., ['en', 'es', 'fr'])
        """
        ...
```

## Core Data Structures

### ParsedIntent

Structured representation of user intent extracted from natural language.

```python
@dataclass
class ParsedIntent:
    """Structured representation of parsed user intent"""
    
    intent_type: IntentType
    """Type of intent (SEARCH, UNDERSTAND, ANALYZE)"""
    
    primary_target: str
    """Main focus or target of the intent (e.g., 'authentication functions')"""
    
    scope: Scope
    """Scope of the operation (FILE, MODULE, PROJECT, SYSTEM)"""
    
    complexity: Complexity
    """Assessed complexity (SIMPLE, MODERATE, COMPLEX, ADAPTIVE)"""
    
    confidence: float
    """Parser confidence in the intent extraction (0.0-1.0)"""
    
    filters: dict[str, Any]
    """Additional filters and constraints extracted from input"""
    
    metadata: dict[str, Any]
    """Parser-specific metadata and processing information"""
    
    parsed_at: datetime
    """Timestamp when intent was parsed"""
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for this intent"""
        return f"{self.intent_type.value}:{self.scope.value}:{hash(self.primary_target)}"
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "intent_type": self.intent_type.value,
            "primary_target": self.primary_target,
            "scope": self.scope.value,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "filters": self.filters,
            "metadata": self.metadata,
            "parsed_at": self.parsed_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedIntent":
        """Create from dictionary representation"""
        return cls(
            intent_type=IntentType(data["intent_type"]),
            primary_target=data["primary_target"],
            scope=Scope(data["scope"]),
            complexity=Complexity(data["complexity"]),
            confidence=data["confidence"],
            filters=data["filters"],
            metadata=data["metadata"],
            parsed_at=datetime.fromisoformat(data["parsed_at"])
        )
```

### IntentResult

Structured result from intent processing execution.

```python
@dataclass
class IntentResult:
    """Structured result from intent processing"""
    
    success: bool
    """Whether processing completed successfully"""
    
    data: Any
    """Main result data (format varies by strategy)"""
    
    metadata: dict[str, Any]
    """Execution metadata and metrics"""
    
    executed_at: datetime
    """When execution completed"""
    
    execution_time: float
    """Execution duration in seconds"""
    
    error_message: str | None = None
    """Error details if execution failed"""
    
    suggestions: list[str] | None = None
    """Suggested next actions for the user"""
    
    strategy_used: str | None = None
    """Name of strategy that processed the intent"""
    
    @property
    def is_cached(self) -> bool:
        """Check if this result came from cache"""
        return self.metadata.get("from_cache", False)
        
    @property
    def cache_ttl(self) -> int | None:
        """Get cache TTL if applicable"""
        return self.metadata.get("cache_ttl")
        
    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion to the result"""
        if self.suggestions is None:
            self.suggestions = []
        self.suggestions.append(suggestion)
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "executed_at": self.executed_at.isoformat(),
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "suggestions": self.suggestions,
            "strategy_used": self.strategy_used
        }
        
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentResult":
        """Create from dictionary representation"""
        return cls(
            success=data["success"],
            data=data["data"],
            metadata=data["metadata"],
            executed_at=datetime.fromisoformat(data["executed_at"]),
            execution_time=data["execution_time"],
            error_message=data.get("error_message"),
            suggestions=data.get("suggestions"),
            strategy_used=data.get("strategy_used")
        )
```

### LLMProfile

Profile containing LLM behavioral characteristics and preferences.

```python
@dataclass
class LLMProfile:
    """Profile of LLM behavioral characteristics"""
    
    session_id: str
    """Hashed session identifier for tracking"""
    
    identified_model: str | None
    """Detected LLM model name (e.g., 'claude-3', 'gpt-4')"""
    
    confidence: float
    """Confidence in model identification (0.0-1.0)"""
    
    behavioral_features: dict[str, Any]
    """Behavioral fingerprint characteristics"""
    
    context_preferences: ContextPreferences
    """Preferred context format and detail level"""
    
    created_at: datetime
    """When profile was created"""
    
    last_updated: datetime | None = None
    """When profile was last updated"""
    
    request_count: int = 0
    """Number of requests in this session"""
    
    def update_preferences(self, preferences: ContextPreferences) -> None:
        """Update context preferences"""
        self.context_preferences = preferences
        self.last_updated = datetime.now()
        
    def increment_request_count(self) -> None:
        """Increment request counter"""
        self.request_count += 1
        self.last_updated = datetime.now()
```

## Enumeration Types

### IntentType

Core intent categories supported by the system.

```python
class IntentType(BaseEnum):
    """Types of user intents supported by the system"""
    
    SEARCH = "search"
    """Find specific code elements, functions, classes, or patterns"""
    
    UNDERSTAND = "understand"
    """Comprehend architecture, relationships, or system organization"""
    
    ANALYZE = "analyze"
    """Examine code for issues, patterns, quality, or characteristics"""
    
    @property
    def requires_context(self) -> bool:
        """Whether this intent type typically requires additional context"""
        return self in [IntentType.UNDERSTAND, IntentType.ANALYZE]
        
    @property
    def supports_caching(self) -> bool:
        """Whether results for this intent type can be cached"""
        return True  # All intent types support caching
        
    @classmethod
    def from_keywords(cls, keywords: list[str]) -> "IntentType | None":
        """Infer intent type from keywords"""
        search_keywords = {"find", "search", "locate", "where", "show"}
        understand_keywords = {"explain", "how", "what", "understand", "describe"}
        analyze_keywords = {"analyze", "check", "review", "examine", "audit"}
        
        keyword_set = set(word.lower() for word in keywords)
        
        if keyword_set & analyze_keywords:
            return cls.ANALYZE
        elif keyword_set & understand_keywords:
            return cls.UNDERSTAND
        elif keyword_set & search_keywords:
            return cls.SEARCH
            
        return None
```

### Scope

Defines the scope of intent processing operations.

```python
class Scope(BaseEnum):
    """Scope levels for intent processing"""
    
    FILE = "file"
    """Single file scope"""
    
    MODULE = "module"
    """Module or directory scope"""
    
    PROJECT = "project"
    """Entire project scope"""
    
    SYSTEM = "system"
    """System-wide or cross-project scope"""
    
    @property
    def estimated_complexity(self) -> float:
        """Estimated complexity multiplier for this scope"""
        complexity_map = {
            Scope.FILE: 1.0,
            Scope.MODULE: 2.5,
            Scope.PROJECT: 5.0,
            Scope.SYSTEM: 10.0
        }
        return complexity_map[self]
        
    @property
    def default_timeout(self) -> float:
        """Default timeout in seconds for this scope"""
        timeout_map = {
            Scope.FILE: 5.0,
            Scope.MODULE: 15.0,
            Scope.PROJECT: 30.0,
            Scope.SYSTEM: 60.0
        }
        return timeout_map[self]
```

### Complexity

Intent processing complexity levels.

```python
class Complexity(BaseEnum):
    """Complexity levels for intent processing"""
    
    SIMPLE = "simple"
    """Direct, single-step operations"""
    
    MODERATE = "moderate"
    """Multi-step operations with some coordination"""
    
    COMPLEX = "complex"
    """Advanced processing requiring multiple strategies"""
    
    ADAPTIVE = "adaptive"
    """Context-dependent complexity requiring dynamic approach"""
    
    @property
    def max_execution_time(self) -> float:
        """Maximum execution time in seconds"""
        time_map = {
            Complexity.SIMPLE: 10.0,
            Complexity.MODERATE: 30.0,
            Complexity.COMPLEX: 60.0,
            Complexity.ADAPTIVE: 120.0
        }
        return time_map[self]
        
    @property
    def requires_workflow(self) -> bool:
        """Whether this complexity typically requires workflow orchestration"""
        return self in [Complexity.COMPLEX, Complexity.ADAPTIVE]
```

## Service Interfaces

### IntentOrchestratorService

Main service interface for intent processing orchestration.

```python
class IntentOrchestratorService(BaseServiceProvider):
    """Main orchestration service for intent processing"""
    
    async def process_intent(
        self, 
        user_input: str, 
        context: FastMCPContext
    ) -> IntentResult:
        """
        Process natural language intent through complete pipeline.
        
        Args:
            user_input: Raw natural language input
            context: FastMCP request context
            
        Returns:
            IntentResult with processing outcome
            
        Raises:
            IntentProcessingError: When processing fails
            ValidationError: When input is invalid
        """
        ...
        
    async def parse_intent(self, user_input: str) -> ParsedIntent:
        """
        Parse natural language into structured intent.
        
        Args:
            user_input: Raw natural language input
            
        Returns:
            ParsedIntent structure
        """
        ...
        
    async def select_strategy(self, parsed_intent: ParsedIntent) -> IntentStrategy:
        """
        Select optimal strategy for parsed intent.
        
        Args:
            parsed_intent: Structured intent to process
            
        Returns:
            Selected strategy instance
        """
        ...
        
    async def get_available_strategies(self) -> list[str]:
        """
        Get list of available strategy names.
        
        Returns:
            List of registered strategy names
        """
        ...
        
    async def get_strategy_info(self, strategy_name: str) -> dict[str, Any]:
        """
        Get detailed information about a strategy.
        
        Args:
            strategy_name: Name of strategy to query
            
        Returns:
            Strategy metadata and capabilities
            
        Raises:
            StrategyNotFoundError: When strategy doesn't exist
        """
        ...
```

### ContextIntelligenceService

Service for LLM context analysis and adaptation.

```python
class ContextIntelligenceService(BaseServiceProvider):
    """Service for analyzing and adapting to LLM context"""
    
    async def extract_llm_profile(self, context: FastMCPContext) -> LLMProfile:
        """
        Extract LLM behavioral profile from request context.
        
        Args:
            context: FastMCP request context
            
        Returns:
            LLMProfile with behavioral characteristics
        """
        ...
        
    async def adapt_result_for_llm(
        self, 
        result: IntentResult, 
        llm_profile: LLMProfile
    ) -> IntentResult:
        """
        Adapt result format for specific LLM characteristics.
        
        Args:
            result: Original intent result
            llm_profile: Target LLM profile
            
        Returns:
            Adapted result optimized for LLM
        """
        ...
        
    async def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get session interaction history.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction records
        """
        ...
        
    async def update_model_preferences(
        self, 
        model_name: str, 
        preferences: ContextPreferences
    ) -> None:
        """
        Update preferences for specific LLM model.
        
        Args:
            model_name: LLM model identifier
            preferences: Updated preferences
        """
        ...
```

### ImplicitLearningService

Service for learning from usage patterns and optimizing performance.

```python
class ImplicitLearningService(BaseServiceProvider):
    """Service for implicit learning from usage patterns"""
    
    async def record_interaction(
        self, 
        parsed_intent: ParsedIntent, 
        result: IntentResult,
        llm_profile: LLMProfile
    ) -> None:
        """
        Record interaction for pattern learning.
        
        Args:
            parsed_intent: The processed intent
            result: Processing result
            llm_profile: LLM behavioral profile
        """
        ...
        
    async def get_optimization_suggestions(
        self, 
        parsed_intent: ParsedIntent
    ) -> list[OptimizationSuggestion]:
        """
        Get optimization suggestions based on learned patterns.
        
        Args:
            parsed_intent: Intent to optimize
            
        Returns:
            List of optimization suggestions
        """
        ...
        
    async def get_learned_patterns(
        self, 
        intent_type: IntentType | None = None
    ) -> list[LearnedPattern]:
        """
        Get learned patterns for intent types.
        
        Args:
            intent_type: Filter by intent type (None for all)
            
        Returns:
            List of learned patterns
        """
        ...
        
    async def clear_learned_patterns(
        self, 
        older_than: datetime | None = None
    ) -> int:
        """
        Clear learned patterns.
        
        Args:
            older_than: Clear patterns older than timestamp (None for all)
            
        Returns:
            Number of patterns cleared
        """
        ...
```

## Workflow Orchestration

### WorkflowDefinition

Defines multi-step workflows for complex intent processing.

```python
@dataclass
class WorkflowDefinition:
    """Definition of a multi-step workflow"""
    
    name: str
    """Workflow name"""
    
    steps: list[WorkflowStep]
    """Ordered list of workflow steps"""
    
    allow_partial_success: bool = False
    """Whether workflow can succeed with some step failures"""
    
    max_parallel_steps: int = 1
    """Maximum number of steps to execute in parallel"""
    
    timeout: float = 300.0
    """Total workflow timeout in seconds"""
    
    retry_policy: RetryPolicy | None = None
    """Retry policy for failed steps"""
    
    def validate(self) -> list[str]:
        """
        Validate workflow definition.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")
            
        # Validate step names are unique
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found")
            
        return errors
        
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow steps"""
        # Implementation of cycle detection
        ...
```

### WorkflowStep

Individual step within a workflow.

```python
@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    
    name: str
    """Step name (must be unique within workflow)"""
    
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    """Async function to execute this step"""
    
    dependencies: list[str] = field(default_factory=list)
    """List of step names this step depends on"""
    
    timeout: float = 30.0
    """Step timeout in seconds"""
    
    required: bool = True
    """Whether step failure should fail the entire workflow"""
    
    retry_count: int = 0
    """Number of times to retry failed step"""
    
    retry_backoff: float = 1.0
    """Backoff multiplier for retries"""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional step metadata"""
    
    @property
    def can_run_parallel(self) -> bool:
        """Whether this step can run in parallel with others"""
        return len(self.dependencies) == 0 or all(
            dep in self.metadata.get("completed_dependencies", [])
            for dep in self.dependencies
        )
```

## Exception Hierarchy

### Intent Layer Exceptions

```python
class IntentLayerError(Exception):
    """Base exception for Intent Layer errors"""
    pass

class IntentProcessingError(IntentLayerError):
    """Error during intent processing execution"""
    
    def __init__(self, message: str, intent: ParsedIntent | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.intent = intent
        self.cause = cause

class ParseError(IntentLayerError):
    """Error during intent parsing"""
    
    def __init__(self, message: str, input_text: str):
        super().__init__(message)
        self.input_text = input_text

class StrategyNotFoundError(IntentLayerError):
    """Error when requested strategy is not available"""
    
    def __init__(self, strategy_name: str):
        super().__init__(f"Strategy not found: {strategy_name}")
        self.strategy_name = strategy_name

class CircuitBreakerOpenError(IntentLayerError):
    """Error when circuit breaker prevents execution"""
    
    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)

class WorkflowExecutionError(IntentLayerError):
    """Error during workflow execution"""
    
    def __init__(self, message: str, workflow_name: str, failed_step: str | None = None):
        super().__init__(message)
        self.workflow_name = workflow_name
        self.failed_step = failed_step

class ValidationError(IntentLayerError):
    """Error during data validation"""
    
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field
```

## Configuration Types

### Service Configuration Classes

```python
class IntentServiceConfig(ServiceConfig):
    """Configuration for Intent Orchestrator Service"""
    
    # Core settings
    provider: str = "intent_orchestrator"
    default_strategy: str = "adaptive"
    confidence_threshold: float = 0.6
    max_execution_time: float = 30.0
    cache_ttl: int = 3600
    
    # Parser settings
    use_nlp_fallback: bool = False
    pattern_matching: bool = True
    custom_patterns_file: str | None = None
    
    # Performance settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_time: float = 60.0
    max_concurrent_intents: int = 10
    
    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_telemetry_tracking: bool = True
    enable_metrics_collection: bool = True

class ContextIntelligenceServiceConfig(ServiceConfig):
    """Configuration for Context Intelligence Service"""
    
    # LLM identification
    llm_identification_enabled: bool = True
    behavioral_fingerprinting: bool = True
    
    # Privacy settings
    privacy_mode: Literal["strict", "hash_identifiers", "minimal"] = "hash_identifiers"
    session_timeout: int = 3600
    max_concurrent_sessions: int = 100
    
    # Context analysis
    context_window_size: int = 4096
    max_context_history: int = 50
    enable_context_learning: bool = True
    context_similarity_threshold: float = 0.8
    adaptive_context_sizing: bool = True

class ImplicitLearningServiceConfig(ServiceConfig):
    """Configuration for Implicit Learning Service"""
    
    # Learning behavior
    learning_enabled: bool = True
    pattern_recognition: bool = True
    success_tracking: bool = True
    
    # Pattern analysis
    min_pattern_frequency: int = 3
    pattern_confidence_threshold: float = 0.75
    max_stored_patterns: int = 1000
    
    # Success optimization
    track_execution_time: bool = True
    track_result_quality: bool = True
    track_user_satisfaction: bool = False
    
    # Privacy and retention
    anonymize_patterns: bool = True
    pattern_retention_days: int = 90
    cleanup_interval_hours: int = 24
```

## Utility Functions

### Intent Processing Utilities

```python
def create_intent_result(
    success: bool,
    data: Any = None,
    error: Exception | None = None,
    strategy_name: str | None = None,
    execution_time: float = 0.0,
    suggestions: list[str] | None = None
) -> IntentResult:
    """
    Create an IntentResult with common patterns.
    
    Args:
        success: Whether processing succeeded
        data: Result data
        error: Exception if failed
        strategy_name: Name of strategy used
        execution_time: Execution duration
        suggestions: List of suggestions
        
    Returns:
        Constructed IntentResult
    """
    return IntentResult(
        success=success,
        data=data,
        metadata={
            "created_by": "utility_function",
            "has_error": error is not None
        },
        executed_at=datetime.now(),
        execution_time=execution_time,
        error_message=str(error) if error else None,
        suggestions=suggestions,
        strategy_used=strategy_name
    )

def validate_parsed_intent(parsed_intent: ParsedIntent) -> list[str]:
    """
    Validate a ParsedIntent structure.
    
    Args:
        parsed_intent: Intent to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not parsed_intent.primary_target.strip():
        errors.append("primary_target cannot be empty")
        
    if not (0.0 <= parsed_intent.confidence <= 1.0):
        errors.append("confidence must be between 0.0 and 1.0")
        
    if parsed_intent.intent_type not in IntentType:
        errors.append(f"invalid intent_type: {parsed_intent.intent_type}")
        
    return errors

def merge_intent_metadata(
    base_metadata: dict[str, Any],
    additional_metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge intent metadata dictionaries.
    
    Args:
        base_metadata: Base metadata
        additional_metadata: Additional metadata to merge
        
    Returns:
        Merged metadata dictionary
    """
    merged = base_metadata.copy()
    
    for key, value in additional_metadata.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_intent_metadata(merged[key], value)
        else:
            merged[key] = value
            
    return merged
```

## Type Definitions

### Context and Preference Types

```python
class ContextPreferences(TypedDict):
    """LLM context and format preferences"""
    context_preference: Literal["concise", "detailed", "balanced"]
    format_preference: Literal["structured", "conversational", "hierarchical"]
    complexity_handling: Literal["adaptive", "simplified", "comprehensive"]

class OptimizationSuggestion(TypedDict):
    """Suggestion for optimizing intent processing"""
    strategy_name: str
    confidence: float
    estimated_time: float
    reasoning: str

class LearnedPattern(TypedDict):
    """Pattern learned from usage history"""
    intent_signature: str
    optimal_strategy: str
    success_rate: float
    avg_execution_time: float
    sample_count: int
    last_updated: str  # ISO format datetime

class InteractionRecord(TypedDict):
    """Record of intent processing interaction"""
    intent_signature: str
    llm_model: str | None
    success: bool
    execution_time: float
    confidence: float
    strategy_used: str | None
    timestamp: str  # ISO format datetime
```

## Version Compatibility

### API Version Support

```python
class APIVersion:
    """Intent Layer API version information"""
    
    CURRENT = "1.0.0"
    SUPPORTED = ["1.0.0"]
    DEPRECATED = []
    
    @classmethod
    def is_supported(cls, version: str) -> bool:
        """Check if API version is supported"""
        return version in cls.SUPPORTED
        
    @classmethod
    def is_deprecated(cls, version: str) -> bool:
        """Check if API version is deprecated"""
        return version in cls.DEPRECATED
```

## Next Steps

1. **[Custom Strategies](custom-strategies.md)**: Learn to implement custom strategies
2. **[Configuration](configuration.md)**: Advanced configuration options
3. **[Examples](examples.md)**: Practical usage examples
4. **[Architecture](architecture.md)**: Understanding the technical implementation