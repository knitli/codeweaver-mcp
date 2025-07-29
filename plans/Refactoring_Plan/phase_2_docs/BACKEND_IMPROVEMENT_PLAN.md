<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Backend System Improvement Plan

## Overview
This document outlines the comprehensive plan to improve the backends implementation in CodeWeaver, addressing attribute consolidation, type system improvements, interface flexibility, removal of backwards compatibility code, and adoption of Pydantic v2.

## Phase 1: Type System Consolidation

### 1.1 Create Centralized Enums
- **File**: `src/codeweaver/_types/enums.py`
- Replace all Literal types with proper enums
- Enums to create:
  - `BackendProvider` (all supported providers)
  - `SparseIndexType` (keyword, text, bm25)
  - `HybridFusionStrategy` (rrf, dbsf, linear, convex)
  - `StorageType` (disk, memory, cloud)
  - `IndexType` (flat, ivf_flat, hnsw, lsh)

### 1.2 Create BackendCapabilities Model
- **File**: `src/codeweaver/_types/capabilities.py`
- Pydantic model with all capability flags:
  ```python
  class BackendCapabilities(BaseModel):
      supports_hybrid_search: bool = False
      supports_sparse_vectors: bool = False
      supports_streaming: bool = False
      supports_transactions: bool = False
      supports_filtering: bool = True
      supports_updates: bool = True
      max_vector_dimensions: int | None = None
      max_collection_size: int | None = None
      supported_index_types: list[IndexType] = Field(default_factory=list)
      supported_sparse_types: list[SparseIndexType] = Field(default_factory=list)
  ```

### 1.3 Provider Registry Enhancement
- **File**: `src/codeweaver/_types/providers.py`
- Create a provider registry with full capability information:
  ```python
  @dataclass
  class ProviderInfo:
      backend_class: type[VectorBackend]
      capabilities: BackendCapabilities
      required_dependencies: list[str]
      optional_dependencies: list[str]
  ```

## Phase 2: Pydantic V2 Migration

### 2.1 Convert All Dataclasses
- Convert these dataclasses to Pydantic models:
  - `BackendConfig` → Pydantic BaseModel
  - `CollectionInfo` → Pydantic BaseModel
  - `BackendConfigExtended` → Merge into BackendConfig
  - `VectorSearchResult` → Pydantic BaseModel

### 2.2 Add Validation
- Add Pydantic validators for:
  - URL validation (backend URLs)
  - Dimension constraints (1-4096)
  - Batch size limits (1-1000)
  - Timeout ranges (0.1-300 seconds)

### 2.3 Serialization Support
- Add model_dump() for dict conversion
- Add model_dump_json() for JSON serialization
- Add model_dump_toml() using custom serializer
- Support for partial updates with exclude_unset

## Phase 3: Remove Backwards Compatibility

### 3.1 Remove Legacy Functions
- Delete `create_backend_config_from_legacy()`
- Remove all QDRANT_* environment variable fallbacks
- Remove migration helper functions

### 3.2 Simplify Factory
- Remove convenience functions added for compatibility
- Streamline creation methods
- Use only new-style configuration

### 3.3 Clean Environment Variables
- Support only new variable names:
  - CW_VECTOR_BACKEND_PROVIDER
  - CW_VECTOR_BACKEND_URL
  - CW_VECTOR_BACKEND_API_KEY
  - CW_VECTOR_BACKEND_COLLECTION

## Phase 4: Config System Centralization

### 4.1 Create Unified Config Module
- **File**: `src/codeweaver/config.py` (enhance existing)
- Single place for all configuration handling
- Use tomlkit for TOML preservation
- Support hierarchical config loading:
  1. Default values
  2. Config file (~/.codeweaver/config.toml)
  3. Environment variables
  4. Runtime overrides

### 4.2 Config Schema
```python
class CodeWeaverConfig(BaseModel):
    backend: BackendConfig
    providers: ProvidersConfig
    sources: SourcesConfig
    server: ServerConfig

    model_config = ConfigDict(
        extra="allow",  # Allow future extensions
        validate_assignment=True,
        use_enum_values=True
    )
```

### 4.3 TOML Handling
- Use tomlkit to preserve comments and formatting
- Generate default config with helpful comments
- Support config validation and migration

## Phase 5: Interface Documentation & Flexibility

### 5.1 Protocol Documentation
- Add comprehensive docstrings to all protocols
- Include implementation examples
- Document required vs optional methods
- Add type hints for all parameters

### 5.2 Simplify Interfaces
- Merge redundant protocols where possible
- Ensure protocols are minimal and focused
- Add default implementations where sensible

### 5.3 Custom Backend Support
- Create CustomBackend base class with sensible defaults
- Add registration validation
- Provide backend development guide
- Add example custom backend implementation

## Implementation Order

### Week 1: Type System & Capabilities
1. Create new enum types
2. Implement BackendCapabilities model
3. Update provider registry
4. Add capability tests

### Week 2: Pydantic Migration
1. Convert dataclasses to Pydantic models
2. Add validation rules
3. Implement serialization
4. Update tests

### Week 3: Remove Compatibility & Centralize Config
1. Remove backwards compatibility code
2. Create unified config system
3. Implement tomlkit integration

### Week 4: Documentation & Polish
1. Document all interfaces
2. Add custom backend examples
3. Final testing and validation

## File Changes Summary

### New Files
- `src/codeweaver/_types/enums.py`
- `src/codeweaver/_types/capabilities.py`
- `src/codeweaver/_types/providers.py`
- `src/codeweaver/config/schema.py`
- `src/codeweaver/config/loader.py`
- `examples/custom_backend.py`

### Modified Files
- `src/codeweaver/backends/base.py` (simplify protocols)
- `src/codeweaver/backends/factory.py` (enhance registration)
- `src/codeweaver/backends/config.py` (remove legacy code)
- `src/codeweaver/backends/qdrant.py` (use new types)
- `src/codeweaver/config.py` (centralize all config)

### Deleted Files
- Migration helper modules
- Backwards compatibility shims

## Testing Strategy

### Unit Tests
- Test each Pydantic model validation
- Test enum conversions
- Test capability detection
- Test custom backend registration

### Integration Tests
- Test config loading hierarchy
- Test TOML preservation
- Test backend creation with new system
- Test serialization round-trips

### Validation Tests
- Ensure no breaking changes in public API
- Validate custom backend support
- Check performance impact

## Success Criteria

1. **Single Source of Truth**: All backend capabilities defined in one place
2. **Type Safety**: All Literal types replaced with enums
3. **Clean API**: No backwards compatibility code
4. **Pydantic Everywhere**: All data models use Pydantic v2
5. **Centralized Config**: One config system with tomlkit
6. **Extensibility**: Easy to add custom backends
7. **Documentation**: All interfaces well-documented
8. **Testing**: >95% test coverage maintained

## Risk Mitigation

1. **Breaking Changes**: Document all API changes clearly
2. **Performance**: Benchmark Pydantic vs dataclasses
3. **Complexity**: Keep interfaces minimal and focused
4. **Migration**: Provide clear upgrade guide
5. **Dependencies**: Keep new dependencies minimal

## Dependencies to Add

```toml
[project.dependencies]
pydantic = "^2.5.0"
tomlkit = "^0.12.0"
```

## Next Steps

1. Review and approve this plan
2. Create feature branch
3. Implement Phase 1 (Type System)
4. Iterate based on feedback
5. Continue through all phases
