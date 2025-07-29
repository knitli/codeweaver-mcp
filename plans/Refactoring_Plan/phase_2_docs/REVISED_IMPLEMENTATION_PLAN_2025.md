<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Implementation Plan - January 2025 Revision

**Plan Date**: July 2025
**Status**: Clean Architecture Remediation Required
**Timeline**: 2-3 weeks to production-ready clean architecture
**Priority**: Critical - Remove legacy code violations before launch

## 🎯 Situation Assessment

### Current Reality
- ✅ **Technical Implementation**: 90% complete with enterprise-grade architecture
- ✅ **Plugin Ecosystem**: Advanced factory patterns and runtime discovery
- ✅ **Multi-Provider Support**: 6+ providers, 15+ backends, 5+ sources
- ❌ **Clean Architecture**: CRITICAL VIOLATION - extensive legacy code remains

### The Problem
Despite exceptional technical achievement, the codebase violates the core requirement:
> "absolutely no code in the codebase that is not part of the new architecture"

**Legacy Code Violations**:
- Dual-server implementation with compatibility layers
- 6+ legacy files actively used by new architecture
- Extensive fallback logic and "bubble gum" architecture
- Import dependencies from new to legacy components

---

## 📋 REVISED IMPLEMENTATION PLAN

### Strategic Approach: Complete Clean Architecture

**Goal**: Remove ALL legacy code while preserving advanced modular architecture
**Timeline**: 2-3 weeks to clean architecture completion
**Risk Level**: Medium - requires completing functionality gaps in new architecture

---

## Phase 1: Legacy System Analysis & Gap Assessment
**Duration**: 3 days
**Priority**: Critical
**Goal**: Understand exactly what functionality exists in legacy vs. new systems

### Tasks

#### 1.1 Legacy Server Functionality Mapping
- [ ] **Analyze `CodeEmbeddingsServer`** (legacy, lines 52-531 in server.py)
  - Map all methods and their functionality
  - Document API interfaces and behavior
  - Identify unique features not in new server
- [ ] **Analyze `ExtensibleCodeEmbeddingsServer`** (new, lines 533-890 in server.py)
  - Map implemented functionality
  - Identify missing features vs. legacy server
  - Document current factory integration points

#### 1.2 Legacy Component Analysis
- [ ] **`chunker.py` analysis**
  - Document AST-grep chunking functionality
  - Identify integration points with new architecture
  - Map to equivalent factory/source functionality
- [ ] **`models.py` analysis**
  - Map legacy data models to `_types/` equivalents
  - Identify any missing model definitions
  - Document serialization/compatibility requirements
- [ ] **`file_filter.py` analysis**
  - Document file filtering logic
  - Map to filesystem source filtering capabilities
  - Identify missing filtering features
- [ ] **`rate_limiter.py` analysis**
  - Document rate limiting implementation
  - Evaluate if new architecture needs rate limiting
  - Consider removal vs. integration
- [ ] **`search.py` analysis**
  - Document AST-grep search functionality
  - Map to source abstraction search capabilities
  - Identify missing search features
- [ ] **`task_search.py` analysis**
  - Document task coordination functionality
  - Evaluate if needed in new architecture
  - Consider removal vs. integration

#### 1.3 Functionality Gap Assessment
- [ ] **Create comprehensive gap analysis**
  - List all functionality present in legacy but missing in new
  - Prioritize gaps by importance and complexity
  - Estimate implementation effort for each gap
- [ ] **Dependency analysis**
  - Map all legacy imports in new architecture
  - Identify circular dependencies
  - Plan dependency elimination strategy

### Deliverables
- **Legacy Functionality Inventory** - Complete mapping of legacy features
- **Gap Analysis Report** - Missing functionality in new architecture
- **Implementation Effort Estimates** - Time required to close each gap

---

## Phase 2: Complete New Architecture Implementation
**Duration**: 1-2 weeks
**Priority**: Critical
**Goal**: Implement missing functionality in factories/sources to achieve feature parity

### Tasks

#### 2.1 Chunking Integration in Sources
- [ ] **Implement chunking directly in `FilesystemSource`**
  - Move AST-grep chunking logic from `chunker.py`
  - Integrate with source abstraction patterns
  - Maintain language detection and fallback logic
- [ ] **Update other sources with chunking**
  - Add chunking capabilities to Git source
  - Add chunking to API source
  - Ensure consistent chunking across all sources

#### 2.2 File Filtering Integration
- [ ] **Integrate filtering into `FilesystemSource`**
  - Move file filtering logic from `file_filter.py`
  - Use `rignore.walk()` for efficient file discovery
  - Maintain gitignore and custom filtering rules
- [ ] **Update source factory**
  - Ensure filtering is configurable via factory
  - Add filtering configuration to source configs

#### 2.3 Search Functionality Integration
- [ ] **Integrate AST-grep search into sources**
  - Move search logic from `search.py` into appropriate sources
  - Ensure search works with factory-created components
  - Maintain structural search capabilities
- [ ] **Update search coordination**
  - Integrate task search coordination if needed
  - Or remove if not necessary in new architecture

#### 2.4 Model Migration Completion
- [ ] **Complete transition to `_types/` models**
  - Ensure all legacy `models.py` functionality exists in `_types/`
  - Update all imports throughout codebase
  - Verify serialization compatibility

#### 2.5 Server Feature Parity
- [ ] **Complete `ExtensibleCodeEmbeddingsServer`**
  - Implement any missing methods from legacy server
  - Ensure all MCP tools work identically
  - Remove any fallback logic to legacy server
- [ ] **Rate limiting integration**
  - Implement rate limiting in new server if needed
  - Or remove if not necessary for launch

### Deliverables
- **Feature-complete new architecture** - All legacy functionality implemented
- **Factory integration** - All functionality accessible via factory system
- **Test validation** - All existing tests pass with new-only architecture

---

## Phase 3: Legacy Code Elimination
**Duration**: 1 week
**Priority**: Critical
**Goal**: Remove ALL legacy code and compatibility layers

### Tasks

#### 3.1 Server Cleanup
- [ ] **Remove legacy server implementation**
  - Delete `CodeEmbeddingsServer` class entirely (lines 52-531 in server.py)
  - Keep only `ExtensibleCodeEmbeddingsServer`
  - Update any server creation functions
- [ ] **Remove fallback logic**
  - Remove all "fallback to legacy" code paths
  - Remove compatibility layers
  - Clean up error handling for legacy fallbacks

#### 3.2 Legacy File Removal
- [ ] **Delete legacy files**
  ```bash
  rm src/codeweaver/chunker.py
  rm src/codeweaver/models.py
  rm src/codeweaver/file_filter.py
  rm src/codeweaver/rate_limiter.py
  rm src/codeweaver/search.py
  rm src/codeweaver/task_search.py
  ```
- [ ] **Update imports**
  - Remove all imports of deleted files
  - Update imports to use factory components
  - Clean up unused imports

#### 3.3 Import Dependency Cleanup
- [ ] **Clean new architecture imports**
  - Remove all legacy imports from new components
  - Update to use factory-created components only
  - Verify no circular dependencies

#### 3.4 Configuration Cleanup
- [ ] **Remove legacy configuration**
  - Remove any configuration specific to legacy components
  - Update configuration schema for new-only architecture
  - Update configuration documentation

### Deliverables
- **Clean codebase** - Zero legacy files or compatibility code
- **Pure new architecture** - All functionality via factory system
- **Updated documentation** - Reflects clean architecture only

---

## Phase 4: Validation & Launch Preparation
**Duration**: 2-3 days
**Priority**: Critical
**Goal**: Ensure new-only architecture works perfectly

### Tasks

#### 4.1 Comprehensive Testing
- [ ] **Test all MCP tools**
  - `index_codebase` - Verify chunking and filtering work
  - `search_code` - Verify search functionality
  - `ast_grep_search` - Verify structural search
  - `get_supported_languages` - Verify language support
- [ ] **Test factory system**
  - Verify all backends can be created and used
  - Verify all providers can be created and used
  - Verify all sources can be created and used
- [ ] **Integration testing**
  - Test complete indexing and search workflows
  - Test with multiple backends/providers
  - Test error handling and recovery

#### 4.2 Performance Validation
- [ ] **Benchmark new-only architecture**
  - Compare performance to previous hybrid system
  - Ensure no performance regression
  - Optimize if necessary

#### 4.3 Documentation Updates
- [ ] **Update all documentation**
  - Remove references to legacy components
  - Update architecture diagrams
  - Update configuration examples
- [ ] **Update examples and guides**
  - Ensure all examples use new architecture
  - Update migration guides to reflect clean architecture

### Deliverables
- **Fully tested clean architecture** - All functionality validated
- **Performance benchmarks** - New architecture performance verified
- **Updated documentation** - Reflects clean architecture reality

---

## 🎯 Success Criteria

### Technical Requirements
- [ ] **Single server implementation** - Only `ExtensibleCodeEmbeddingsServer` exists
- [ ] **Zero legacy files** - All legacy files completely removed
- [ ] **No compatibility logic** - No fallback or "bubble gum" code
- [ ] **Factory-only architecture** - All functionality via factory system
- [ ] **Feature parity** - All MCP tools work identically to before
- [ ] **Performance maintained** - No performance regression

### Clean Architecture Validation
- [ ] **No legacy imports** - New architecture imports nothing from deleted files
- [ ] **No dual implementations** - Single implementation of all functionality
- [ ] **No fallback logic** - Clean error handling without legacy fallbacks
- [ ] **Configuration consistency** - Single configuration approach
- [ ] **Documentation alignment** - Docs reflect clean architecture only

### Quality Gates
- [ ] **100% test coverage** - All functionality thoroughly tested
- [ ] **Integration test success** - Complete workflows work end-to-end
- [ ] **Performance benchmarks met** - No degradation in key metrics
- [ ] **Code quality maintained** - Type safety, error handling, documentation
- [ ] **Plugin system validated** - All factory components work correctly

---

## 🚨 Risk Assessment & Mitigation

### High-Risk Areas

#### 1. Functionality Gaps
**Risk**: Missing critical functionality when legacy code removed
**Mitigation**: Comprehensive gap analysis in Phase 1, thorough testing in Phase 4

#### 2. Performance Impact
**Risk**: New architecture may be slower than legacy implementation
**Mitigation**: Performance benchmarking, optimization if needed

#### 3. Configuration Complexity
**Risk**: Factory-based configuration may be more complex than legacy
**Mitigation**: Maintain configuration simplicity, provide migration guides

### Medium-Risk Areas

#### 4. Testing Coverage
**Risk**: Edge cases may not be covered in new architecture
**Mitigation**: Comprehensive test suite, integration testing

#### 5. Documentation Lag
**Risk**: Documentation may not reflect clean architecture changes
**Mitigation**: Update documentation in parallel with code changes

---

## 📊 Resource Requirements

### Development Team
- **Senior Developer** (1): Architecture refactoring and legacy elimination
- **Integration Specialist** (0.5): Testing and validation

### Timeline Breakdown
- **Phase 1** (Analysis): 3 days
- **Phase 2** (Implementation): 1-2 weeks
- **Phase 3** (Cleanup): 1 week
- **Phase 4** (Validation): 2-3 days

**Total Timeline**: **2-3 weeks**

### Dependencies
- Complete access to existing codebase and documentation
- Ability to run comprehensive test suites
- Performance benchmarking environment

---

## 🎉 Expected Outcomes

### Technical Achievements
- **Clean Architecture**: Zero legacy code, single implementation approach
- **Enhanced Maintainability**: Simplified codebase with consistent patterns
- **Improved Performance**: Elimination of dual-system overhead
- **Future-Proof Foundation**: Clean base for community contributions

### Business Benefits
- **Launch Ready**: Meets all clean architecture requirements
- **Community Ready**: Clean plugin SDK for ecosystem development
- **Enterprise Ready**: Professional-grade architecture and documentation
- **Competitive Advantage**: Industry-leading extensible code search platform

### Quality Improvements
- **Code Quality**: Single, well-architected implementation
- **Documentation**: Consistent, clean architecture documentation
- **Testing**: Comprehensive coverage of new-only architecture
- **Maintenance**: Simplified codebase with clear separation of concerns

---

## 🚀 Post-Implementation Vision

### Launch State
After completing this plan, CodeWeaver will have:
- **Pure plugin architecture** with zero legacy code
- **Enterprise-grade extensibility** with community SDK
- **Industry-leading performance** with optimized single implementation
- **Professional documentation** reflecting clean architecture
- **Comprehensive ecosystem** supporting 15+ backends, 6+ providers, 5+ sources

### Competitive Position
- **Most extensible** code search solution in the market
- **Clean architecture** serving as model for similar projects
- **Community ecosystem** enabling third-party contributions
- **Enterprise adoption** ready with professional-grade implementation

This revised plan addresses the critical clean architecture violations while preserving the exceptional technical achievements already implemented, delivering a world-class extensible code search platform ready for production launch and community adoption.
