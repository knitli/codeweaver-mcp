<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

CodeWeaver Consistency & Architecture Improvement Plan

📋 Architectural Analysis Summary

Based on comprehensive analysis of the CodeWeaver codebase patterns, I've identified critical consistency gaps across error handling, health monitoring, service context propagation, and auto-indexing integration.
The system shows sophisticated implementation in some areas (services layer, factory error handling) but significant inconsistencies in others.

🎯 Core Issues Identified

1. Error Usage Inconsistency ⚠️

- 7 unused exceptions defined but never raised
- 156 builtin exception usages vs 7 custom exceptions across providers
- Type mismatches in backend health checks (declares bool, returns ServiceHealth)
- Inconsistent patterns across component layers

- relevant files: `scripts/get_all_errors.py` (to find all exceptions and their usgage)
- tip: Devise a way to programmatically insert/replace exceptions to avoid reading whole files manually (i.e. ast-driven)

2. Health Check Implementation Gaps ❌

- 85% missing implementations across backends, providers, sources
- Return type inconsistencies between component layers
- Only services layer has comprehensive health monitoring

- relevant files: All implementations in component/providers: `src/codeweaver/providers/providers/*`, `src/codeweaver/sources/providers/*`, `src/codeweaver/backends/providers/*`

3. Service Context Propagation Issues 🔄

- 0/13 service providers implement create_service_context methods
- No centralized ServiceContext type definition
- Limited context intelligence and metadata enrichment

- relevant files: should connect with context services:
  - `src/codeweaver/services/providers/context_intelligence.py`
  - `src/codeweaver/services/providers/implicit_learning.py`
  - `src/codeweaver/services/providers/zero_shot_optimization.py`

4. Auto-Indexing Filter Integration 🔍

- Duplicated filtering logic bypassing FilteringService capabilities
- Inconsistent behavior between initial scans and live monitoring
- Missing dependency injection for filtering service

- relevant files: `src/codeweaver/services/providers/auto_indexing.py`, `src/codeweaver/services/providers/filtering_service.py`

Important context:
  - Service layer guide: `docs/SERVICES_LAYER_GUIDE.md` (may be dated in terms of structure but relevant for usage patterns)
  - Development Patterns: `docs/DEVELOPMENT_PATTERNS.md` (provides insights into expected patterns, also may not reflect current structure)

🏗️ Unified Architecture Design

Error Handling Standardization

# Unified error context across all components
class ComponentError(CodeWeaverError):
    def __init__(self, message: str, component_type: ComponentType,
                component_name: str, operation: str, original_error: Exception = None):
        super().__init__(message, component=f"{component_type.value}:{component_name}")
        self.component_type = component_type
        self.operation = operation

# Provider-specific pattern
class ProviderError(ComponentError):
    def __init__(self, message: str, provider_type: str, provider_name: str, **kwargs):
        super().__init__(message, ComponentType.PROVIDER, provider_name, **kwargs)
        self.provider_type = provider_type

Unified Health Check Architecture

# Base health check interfaces
class HealthCheckable(Protocol):
    async def health_check(self) -> bool  # Simple availability for components

class ServiceHealthCheckable(Protocol):
    async def health_check(self) -> ServiceHealth  # Detailed metrics for services

# Implementation pattern for backends/providers/sources
class BaseComponentHealth:
    async def health_check(self) -> bool:
        try:
            await self._perform_health_check()
            return True
        except Exception as e:
            self._logger.warning("Health check failed: %s", e)
            return False

Enhanced Service Context System

@dataclass
class ServiceContext:
    """Enhanced service context with intelligence and metadata."""
    services_manager: ServicesManager
    available_services: dict[ServiceType, Any]
    service_health: dict[ServiceType, ServiceHealth]
    service_capabilities: dict[ServiceType, ServiceCapabilities]
    context_metadata: dict[str, Any]
    session_id: str
    timestamp: datetime

# Standard service provider context method
async def create_service_context(self, base_context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        **(base_context or {}),
        f"{self.service_type.value}_service": self,
        f"{self.service_type.value}_capabilities": self.capabilities,
        f"{self.service_type.value}_health": await self.health_check(),
        f"{self.service_type.value}_metadata": self._get_metadata()
    }

Auto-Indexing Integration Pattern

# Unified filtering through FilteringService
def _should_process_file(self, file_path: Path) -> bool:
    if not self.filtering_service:
        return self._fallback_filtering(file_path)

    return self.filtering_service.should_include_file(
        file_path,
        include_patterns=self._config.watch_patterns,
        exclude_patterns=self._config.ignore_patterns
    )

# Enhanced event processing with metadata
async def _process_file_event(self, file_path: Path) -> None:
    metadata = await self.filtering_service.get_file_metadata(file_path)
    if metadata.is_binary or metadata.size > self._config.max_file_size:
        return
    # Continue with indexing...

📈 Implementation Roadmap

Phase 1: Error Cleanup (1-2 days)

1. Remove 7 unused exceptions → Import cleanup
2. Fix QdrantBackend return type mismatch → bool vs ServiceHealth
3. Standardize SpacyProvider health check → Consistent return type

Phase 2: Health Check Standardization (3-4 days)

1. Create base health interfaces → HealthCheckable protocols
2. Implement health checks in backends → (Backends may be OK already -- only current providers are `src/codeweaver/backends/providers/qdrant.py` and `src/codeweaver/backends/providers/docarray/qdrant.py`)
3. Implement health checks in providers → All embedding providers
4. Implement health checks in sources → Filesystem, Git, Database, Web, API/sc:

Phase 3: Service Context Enhancement (2-3 days)

1. Create ServiceContext type → Centralized definition in cw_types/services/
2. Implement context methods → All 13 service providers
3. Enhance context intelligence → Capabilities, health, metadata
4. Integrate with intent layer → Enhanced context propagation

Phase 4: Auto-Indexing Integration (2-3 days)

1. Fix dependency injection → Proper FilteringService access
2. Replace custom filtering logic → Use FilteringService methods
3. Unify configuration → Eliminate duplicated patterns
4. Add metadata integration → File size, binary detection, gitignore

Phase 5: Provider Error Standardization (3-4 days)

1. Create provider error hierarchy → ProviderError base classes
2. Replace 157 builtin exceptions → Structured custom exceptions
3. Add error context → Provider type, operation, recovery suggestions
4. Integrate advanced error handling → Factory error handler patterns

🔧 Development Priorities

High Priority (Immediate Impact)

- ✅ Error usage cleanup and unused exception removal
- ✅ Health check implementation across all component layers
- ✅ Service context method implementation

Medium Priority (Quality Improvements)

- ✅ Auto-indexing FilteringService integration
- ✅ Provider error standardization with structured context

Low Priority (Long-term Enhancement)

- Enhanced error recovery strategies system-wide
- Advanced context intelligence and adequacy assessment
- Performance monitoring integration across all health checks

📊 Expected Outcomes

- Consistency: Unified error handling and health monitoring across all components
- Maintainability: Centralized patterns reducing code duplication
- Observability: Comprehensive health monitoring and structured error context
- Performance: Efficient service context propagation and auto-indexing integration
- Reliability: Robust error handling with recovery strategies

This plan addresses all identified consistency issues while establishing sustainable architectural patterns for future development.
