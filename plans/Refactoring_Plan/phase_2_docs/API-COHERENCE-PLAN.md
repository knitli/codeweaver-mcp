<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

## Key Findings

**Module Maturity Assessment:**
- **Providers Module**: A+ (95%) - Exemplary architecture serving as the template
- **Backends Module**: B (70%) - Good foundation but needs legacy code cleanup
- **Sources Module**: C+ (60%) - Mixed implementation with protocol inconsistencies

## Critical Issues Identified

1. **TypedDict/Enum Conflicts**: Multiple `ProviderCapability` definitions causing runtime failures across 4 files
2. **Legacy Dual Systems**: Backend module maintaining both new abstraction and legacy QdrantClient initialization
3. **Protocol Mismatches**: Sources returning `SourceCapabilities` model while protocol expects `set[SourceCapability]`
4. **Hardcoded Capabilities**: Factory contains hardcoded provider dictionaries instead of using centralized `BackendCapabilities` model
5. **Manual Dict Construction**: Config migration bypassing Pydantic serialization with manual dictionary building
6. **Naming Inconsistencies**: Provider enums using `"voyage"` instead of `"voyage-ai"` and `"embeddings"` instead of `"embedding"`

## Comprehensive 3-Phase Implementation Plan

**Phase 1: Critical Foundation Fixes**
- Remove TypedDict/Enum conflicts (18 files affected)
- Eliminate dual backend system in server.py
- Fix sources protocol return type mismatch
- Replace hardcoded backend capabilities with centralized model

**Phase 2: Architectural Harmonization**
- Convert sources TypedDict configs to Pydantic models
- Fix provider enum naming inconsistencies
- Replace manual dict construction with Pydantic serialization
- Centralize backend capability definitions

**Phase 3: Factory Integration Preparation**
- Standardize capability access patterns across modules
- Consolidate environment variable handling
- Enhance custom component validation consistency
- Apply providers module SDK patterns to backends and sources

## Target Architecture

The plan transforms all modules to match the providers module's A+ architecture:
- **Unified Capability Queries**: All modules use identical `registry.get_capabilities()` patterns
- **Zero Legacy Code**: Complete elimination of dual systems and fallback mechanisms
- **Consistent Pydantic Usage**: All configuration uses Pydantic v2 with proper validation
- **Plugin SDK Patterns**: Uniform extensibility framework across all modules
- **Type Safety**: Enum-based systems replacing TypedDict conflicts

## Factory Integration Readiness

Upon completion, the modules will provide:
- Single capability query pattern across all components
- Unified registration APIs for seamless factory composition
- Consistent configuration serialization/deserialization
- Comprehensive validation frameworks
- Plugin extensibility matching providers module standards

This comprehensive plan addresses all priority levels from the analysis report and ensures the three modules achieve architectural coherence necessary for clean factory integration, with the providers module serving as the architectural template for backends and sources to follow.

## Implementation Details

Based on my comprehensive analysis of the codebase and the existing architecture analysis report, I'm now ready to present a detailed improvement plan. Let me create the comprehensive plan for complete module harmonization.

# 📋 CodeWeaver Module Harmonization Plan
**Comprehensive Improvement Strategy for Backends, Providers, and Sources**

## Executive Summary

This plan addresses all identified architectural inconsistencies across the three core modules (backends, providers, sources) to achieve complete harmonization and prepare for seamless factory integration. The analysis reveals that while the **providers module** serves as an excellent architectural template (A+ grade), the **backends** and **sources modules** require focused improvements to eliminate legacy code, resolve type conflicts, and standardize patterns.

## 🎯 Critical Findings Summary

### Module Maturity Assessment
- **Providers Module**: A+ (95%) - Exemplary architecture, minor enum conflicts
- **Backends Module**: B (70%) - Good foundation, legacy dual system issues
- **Sources Module**: C+ (60%) - Mixed implementation, protocol inconsistencies

### Key Issues Identified
1. **TypedDict/Enum Conflicts**: Multiple `ProviderCapability` definitions causing runtime failures
2. **Legacy Dual Systems**: Backends maintaining both old and new initialization paths
3. **Protocol Mismatches**: Sources returning different types than protocols expect
4. **Hardcoded Capabilities**: Factory still contains hardcoded provider dictionaries
5. **Manual Dict Construction**: Config migration bypassing Pydantic serialization
6. **Naming Inconsistencies**: Provider enum values not following established patterns

## 🏗️ Implementation Strategy

### Phase 1: Critical Foundation Fixes (Week 1-2)
**Objective**: Eliminate blocking issues for factory integration

### Phase 2: Architectural Harmonization (Week 3-4)
**Objective**: Align all modules to providers standard

### Phase 3: Factory Integration Preparation (Week 5-6)
**Objective**: Unified patterns and comprehensive validation

---

## 📍 PHASE 1: CRITICAL FOUNDATION FIXES

### 1.1 Eliminate TypedDict/Enum Conflicts 🚨 **CRITICAL**

**Problem**: [`ProviderCapability`](src/codeweaver/_types/provider_enums.py:16) exists as both TypedDict and enum

**Files to Modify**:
- `src/codeweaver/providers/base.py` (lines 24-53, 87-116)
- `src/codeweaver/_types/providers.py` (lines 16-45, 23-52)

**Action Plan**:
```python
# REMOVE all TypedDict versions:
# - src/codeweaver/providers/base.py:24-53
# - src/codeweaver/_types/providers.py:16-45

# KEEP ONLY the enum version from:
# - src/codeweaver/_types/provider_enums.py:16-29

# UPDATE all imports across codebase to use enum exclusively
```

**Files Requiring Import Updates**: 18 files identified using enum imports

### 1.2 Remove Legacy Dual Backend System 🚨 **CRITICAL**

**Problem**: [`server.py`](src/codeweaver/server.py:159) maintains both new backend abstraction and legacy QdrantClient

**Action Plan**:
```python
# REMOVE from server.py:
# - Line 159: "# Initialize legacy Qdrant client for backward compatibility"
# - QdrantClient initialization in _initialize_backend()
# - Legacy fallback mechanisms in backend creation

# REPLACE with single backend abstraction:
def _initialize_backend(self) -> VectorBackend | None:
    backend_config = self._create_backend_config()
    try:
        return BackendFactory.create_backend(backend_config)
    except Exception:
        logger.exception("Failed to initialize backend")
        raise  # No fallback - fail fast
```

### 1.3 Fix Sources Protocol Return Type Mismatch 🚨 **CRITICAL**

**Problem**: Base protocol expects `set[SourceCapability]` but [`FileSystemSource`](src/codeweaver/sources/filesystem.py:226) returns `SourceCapabilities` model

**Files to Modify**:
- `src/codeweaver/sources/base.py` (lines 323-325)
- `src/codeweaver/sources/filesystem.py` (lines 226-235)

**Action Plan**:
```python
# OPTION A: Update protocol to use SourceCapabilities model
@abstractmethod
def get_capabilities(self) -> SourceCapabilities:
    """Get capabilities supported by this source implementation."""

# OPTION B: Update FileSystemSource to return set[SourceCapability]
def get_capabilities(self) -> set[SourceCapability]:
    return {
        SourceCapability.CONTENT_DISCOVERY,
        SourceCapability.CONTENT_READING,
        # ... etc
    }
```

**Recommendation**: Choose Option A for consistency with providers architecture

### 1.4 Replace Hardcoded Backend Capabilities 🚨 **CRITICAL**

**Problem**: [`factory.py`](src/codeweaver/backends/factory.py:314-329) contains hardcoded `planned_providers` dictionary

**Action Plan**:
```python
# REMOVE hardcoded dictionary from factory.py:314-329
# REPLACE with centralized capability queries:

def get_backend_capabilities(provider_type: BackendProvider) -> BackendCapabilities:
    """Get capabilities from centralized BackendCapabilities model."""
    registry_entry = get_backend_registry_entry(provider_type)
    return registry_entry.capabilities

# UPDATE all capability checks to use centralized model
```

---

## 📍 PHASE 2: ARCHITECTURAL HARMONIZATION

### 2.1 Convert Sources TypedDict Configs to Pydantic 🔥 **HIGH PRIORITY**

**Problem**: [`BaseSourceConfig`](src/codeweaver/sources/base.py:118-143) and related configs use TypedDict

**Action Plan**:
```python
# CONVERT TypedDict to Pydantic BaseModel:

class BaseSourceConfig(BaseModel):
    """Base configuration for all data sources."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core settings
    enabled: bool = Field(default=True)
    priority: Annotated[int, Field(default=1, ge=1, le=10)]
    source_id: str = Field(..., min_length=1)

    # Content filtering
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    max_file_size_mb: Annotated[int, Field(default=10, ge=1, le=1000)]

    # Performance settings
    batch_size: Annotated[int, Field(default=10, ge=1, le=100)]
    max_concurrent_requests: Annotated[int, Field(default=5, ge=1, le=20)]
    request_timeout_seconds: Annotated[int, Field(default=30, ge=5, le=300)]
```

### 2.2 Fix Provider Enum Naming Inconsistencies 🔥 **HIGH PRIORITY**

**Problem**: [`provider_enums.py`](src/codeweaver/_types/provider_enums.py:47) has inconsistent naming

**Current Issues**:
```python
VOYAGE_AI = "voyage"  # Should be "voyage-ai"
EMBEDDINGS = "embeddings"  # Inconsistent with singular naming pattern
```

**Action Plan**:
```python
# UPDATE provider_enums.py:
class ProviderType(Enum):
    VOYAGE_AI = "voyage-ai"  # Fixed from "voyage"
    OPENAI = "openai"
    COHERE = "cohere"
    # ... rest unchanged

class ProviderCapability(Enum):
    EMBEDDING = "embedding"  # Fixed from "embeddings"
    RERANKING = "reranking"
    # ... rest unchanged
```

**Impact Analysis**: 47 files reference these enums - comprehensive update required

Recommendation: Consistently use singular forms, hyphenated names, lowercase values for all provider enums to match providers module standards.

Use a script to automate renaming across all files to avoid reading and updating each file manually.

### 2.3 Replace Manual Dict Construction with Pydantic 🔥 **HIGH PRIORITY**

**Problem**: [`config_migration.py`](src/codeweaver/config_migration.py:256-271) and [`config.py`](src/codeweaver/config.py:453-466) manually construct dictionaries

**Action Plan**:
```python
# REPLACE manual construction:
legacy_provider_dict = {
    "embedding_provider": embedding_config.get("provider", "voyage"),
    # ... manual field mapping
}

# WITH Pydantic serialization:
provider_config = ProviderConfig.model_validate(embedding_config)
legacy_provider_dict = provider_config.model_dump(exclude_none=True)
```

### 2.4 Centralize Backend Capability Definitions 🔥 **HIGH PRIORITY**

**Problem**: Capability logic scattered across multiple files

**Action Plan**:
```python
# CONSOLIDATE all capability logic in BackendCapabilities model
# REMOVE scattered hardcoded checks from:
# - config_migration.py:52-59 (BACKEND_EMBEDDING_COMPATIBILITY)
# - Multiple factory methods with hardcoded capability dictionaries

# CREATE centralized capability query pattern:
def get_backend_capabilities(provider: BackendProvider) -> BackendCapabilities:
    return BackendRegistry.get_capabilities(provider)
```

---

## 📍 PHASE 3: FACTORY INTEGRATION PREPARATION

### 3.1 Standardize Capability Access Patterns 📊 **MEDIUM PRIORITY**

**Objective**: Apply providers module patterns across all modules

**Target Pattern**:
```python
# Unified capability query across all modules:
def get_component_capabilities(component_type: str, provider: str) -> BaseCapabilities:
    registry = get_registry(component_type)  # backend|provider|source
    entry = registry.get_entry(provider)
    return entry.capabilities

# Usage examples:
backend_caps = get_component_capabilities("backend", "qdrant")
provider_caps = get_component_capabilities("provider", "voyage-ai")
source_caps = get_component_capabilities("source", "filesystem")
```

### 3.2 Consolidate Environment Variable Handling 📊 **MEDIUM PRIORITY**

**Problem**: Duplicate environment variable processing logic

**Action Plan**:
```python
# CREATE single environment processor:
class EnvironmentVariableProcessor:
    @staticmethod
    def process_component_env_vars(component_type: str) -> dict[str, Any]:
        """Single method for all environment variable processing."""
        processors = {
            "backend": BackendEnvProcessor,
            "provider": ProviderEnvProcessor,
            "source": SourceEnvProcessor
        }
        return processors[component_type].process()
```

### 3.3 Enhance Custom Component Validation 📊 **MEDIUM PRIORITY**

**Objective**: Apply providers validation patterns to backends and sources

**Implementation**:
```python
# EXTEND providers validation framework:
class ComponentValidator:
    def validate_backend(self, backend_class: type) -> ValidationResult:
        """Apply providers-style validation to backends."""

    def validate_source(self, source_class: type) -> ValidationResult:
        """Apply providers-style validation to sources."""

    def validate_capabilities(self, capabilities: BaseCapabilities) -> ValidationResult:
        """Unified capability validation."""
```

---

## 🎨 Unified Architecture Vision

### Target Factory Integration Pattern

```python
class CodeWeaverFactory:
    """Unified factory with harmonized component creation."""

    def __init__(self):
        self.backend_registry = BackendRegistry()
        self.provider_registry = ProviderRegistry()  # Already excellent
        self.source_registry = SourceRegistry()

    def create_server(self, config: CodeWeaverConfig) -> CodeEmbeddingsServer:
        """Unified capability-driven creation."""
        # All components follow identical patterns:
        backend = self._create_component("backend", config.backend_config)
        provider = self._create_component("provider", config.provider_config)
        source = self._create_component("source", config.source_config)

        return CodeEmbeddingsServer(backend, provider, source)

    def _create_component(self, component_type: str, config: BaseConfig) -> Any:
        """Unified component creation with consistent patterns."""
        registry = self._get_registry(component_type)
        capabilities = registry.get_capabilities(config.provider)

        if capabilities.validate_config(config):
            return registry.create(config)
        else:
            raise ConfigurationError(f"Invalid {component_type} configuration")
```

### Capability-Driven Design Pattern

```python
# All modules follow identical capability query pattern:
registry_entry = get_registry_entry(component_type, provider_name)
capabilities = registry_entry.capabilities

if capabilities.supports_feature:
    # Feature-specific logic
    component.use_feature()
```

---

## 📋 Implementation Checklist

### ✅ Critical Phase 1 Tasks
- [ ] Remove TypedDict `ProviderCapability` definitions (4 files)
- [ ] Eliminate dual backend system in `server.py`
- [ ] Fix sources protocol return type consistency
- [ ] Replace hardcoded backend capabilities in `factory.py`

### 🔥 High Priority Phase 2 Tasks
- [ ] Convert `BaseSourceConfig` and related TypedDict to Pydantic (6 files)
- [ ] Fix provider enum naming: `VOYAGE_AI = "voyage-ai"`, `EMBEDDING = "embedding"`
- [ ] Replace manual dict construction with Pydantic serialization (8 files)
- [ ] Centralize backend capability definitions

### 📊 Medium Priority Phase 3 Tasks
- [ ] Standardize capability access patterns across modules
- [ ] Consolidate environment variable handling
- [ ] Enhance custom component validation consistency
- [ ] Implement unified configuration orchestration
- [ ] Apply SDK patterns from providers to backends/sources

### 🧪 Validation Requirements
- [ ] Verify no legacy code remains in production paths
- [ ] Confirm consistent capability access patterns
- [ ] Validate plugin extensibility across all modules
- [ ] Test configuration serialization/deserialization
- [ ] Verify custom component registration works uniformly

---

## 🚀 Success Metrics

### Factory Integration Readiness
1. **Single Capability Query Pattern**: All modules use identical `registry.get_capabilities()` pattern
2. **Zero Legacy Code**: No dual systems or legacy fallbacks in production paths
3. **Consistent Pydantic Usage**: All configuration uses Pydantic v2 serialization
4. **Unified Registration APIs**: Plugin SDK patterns applied uniformly across modules
5. **Type Safety**: All TypedDict conflicts resolved, enum-based systems throughout

### Performance & Maintainability
- **Reduced Code Duplication**: Centralized patterns eliminate scattered implementations
- **Improved Type Safety**: Proper enum usage prevents runtime type errors
- **Enhanced Extensibility**: Consistent plugin architecture enables easy component addition
- **Simplified Configuration**: Pydantic validation replaces manual dictionary manipulation

---

## 🎯 Post-Implementation Architecture

Once complete, all three modules will demonstrate:

- **🏆 Providers-Level Excellence**: Backends and sources achieve A+ architectural maturity
- **🔄 Unified Patterns**: Identical capability queries, registration APIs, and validation frameworks
- **⚡ Factory-Ready**: Clean composition without legacy code or architectural inconsistencies
- **🛡️ Type Safety**: Enum-based systems with comprehensive Pydantic validation
- **🔌 Plugin Extensibility**: Consistent SDK patterns enable seamless custom component integration
