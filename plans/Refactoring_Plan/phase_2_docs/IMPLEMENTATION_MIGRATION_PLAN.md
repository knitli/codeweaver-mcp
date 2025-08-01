<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Implementation Migration Plan

## Overview

This plan provides a step-by-step implementation strategy for integrating FastMCP middleware with CodeWeaver's services layer. The plan is designed to minimize disruption while ensuring thorough testing and validation at each step.

## Implementation Phases

### Phase 1: Foundation Setup (Estimated: 2-3 days)

#### Step 1.1: Create Type System Extensions
**Files to create/modify:**
- `src/codeweaver/_types/config.py` - Add middleware service types
- `src/codeweaver/_types/service_config.py` - Add middleware service configurations
- `src/codeweaver/_types/services.py` - Add middleware service protocols
- `src/codeweaver/_types/service_data.py` - Update health report structure

**Tasks:**
1. Add `ServiceType` enum extensions for middleware services
2. Create middleware service configuration classes
3. Define middleware service protocols (interfaces)
4. Update `ServicesHealthReport` to include middleware services
5. Update `ServicesConfig` to include middleware configurations

**Validation:**
- All type definitions compile without errors
- Pydantic model validation works correctly
- Type checking passes with mypy/pyright

#### Step 1.2: Create Base Middleware Provider
**Files to create:**
- `src/codeweaver/services/providers/middleware/` directory
- `src/codeweaver/services/providers/middleware/__init__.py`
- `src/codeweaver/services/providers/middleware/base_middleware_provider.py`

**Tasks:**
1. Create middleware provider directory structure
2. Implement `BaseMiddlewareProvider` abstract class
3. Define common middleware functionality (metrics, health checks)
4. Add package initialization file

**Validation:**
- Base provider can be imported successfully
- Abstract methods are properly defined
- Common functionality works as expected

### Phase 2: Middleware Service Providers (Estimated: 4-5 days)

#### Step 2.1: Implement Logging Service Provider
**Files to create:**
- `src/codeweaver/services/providers/middleware/logging_provider.py`

**Tasks:**
1. Implement `FastMCPLoggingProvider` class
2. Add log capture and metrics collection
3. Implement service methods (log_request, log_response, get_log_metrics)
4. Add configuration validation
5. Create unit tests

**Testing:**
- Unit tests for provider functionality
- Integration test with FastMCP LoggingMiddleware
- Configuration validation tests

#### Step 2.2: Implement Timing Service Provider
**Files to create:**
- `src/codeweaver/services/providers/middleware/timing_provider.py`

**Tasks:**
1. Implement `FastMCPTimingProvider` class
2. Add performance metrics collection
3. Implement service methods (start_timing, end_timing, get_performance_metrics)
4. Add operation statistics tracking
5. Create unit tests

**Testing:**
- Unit tests for timing functionality
- Performance impact assessment
- Metrics accuracy validation

#### Step 2.3: Implement Error Handling Service Provider
**Files to create:**
- `src/codeweaver/services/providers/middleware/error_handling_provider.py`

**Tasks:**
1. Implement `FastMCPErrorHandlingProvider` class
2. Add error aggregation and statistics
3. Implement service methods (handle_error, get_error_statistics, get_recent_errors)
4. Add optional error notification framework
5. Create unit tests

**Testing:**
- Unit tests for error handling
- Error aggregation accuracy
- Exception handling validation

#### Step 2.4: Implement Rate Limiting Service Provider
**Files to create:**
- `src/codeweaver/services/providers/middleware/rate_limiting_provider.py`

**Tasks:**
1. Implement `FastMCPRateLimitingProvider` class
2. Add rate limiting metrics and status tracking
3. Implement service methods (check_rate_limit, get_rate_limit_status, get_rate_limit_metrics)
4. Add client statistics tracking
5. Create unit tests

**Testing:**
- Unit tests for rate limiting functionality
- Rate limiting accuracy validation
- Performance impact assessment

### Phase 3: Services Manager Integration (Estimated: 3-4 days)

#### Step 3.1: Update ServicesManager
**Files to modify:**
- `src/codeweaver/services/manager.py`

**Tasks:**
1. Add FastMCP server dependency to constructor
2. Implement middleware service registration methods
3. Add middleware service creation and initialization
4. Update health monitoring to include middleware services
5. Update shutdown process for middleware services
6. Add middleware service access methods

**Testing:**
- Unit tests for ServicesManager changes
- Integration tests with middleware services
- Health monitoring validation
- Graceful shutdown testing

#### Step 3.2: Update ServiceBridge
**Files to modify:**
- `src/codeweaver/services/middleware_bridge.py`

**Tasks:**
1. Update service injection to include middleware services
2. Add middleware service availability checks
3. Update service mapping for tools
4. Add error handling for middleware service injection

**Testing:**
- Integration tests with middleware service injection
- Tool context validation
- Service availability testing

### Phase 4: Configuration Integration (Estimated: 2-3 days)

#### Step 4.1: Update Main Configuration
**Files to modify:**
- `src/codeweaver/config.py`

**Tasks:**
1. Add `ServicesConfig` to main configuration
2. Implement configuration migration support
3. Add configuration validation
4. Update environment variable support
5. Add configuration validation helpers

**Testing:**
- Configuration loading tests
- Migration testing with existing configs
- Environment variable override tests
- Validation testing

#### Step 4.2: Create Example Configurations
**Files to create:**
- `.codeweaver.example.toml`
- `docs/examples/middleware-config.toml`

**Tasks:**
1. Create comprehensive example TOML configuration
2. Add configuration documentation
3. Create migration examples
4. Add troubleshooting guide

**Validation:**
- Example configurations load successfully
- Documentation is clear and comprehensive

### Phase 5: Server Integration (Estimated: 2-3 days)

#### Step 5.1: Update Server Implementation
**Files to modify:**
- `src/codeweaver/server.py`

**Tasks:**
1. Remove hardcoded middleware setup from `_setup_middleware`
2. Add ServicesManager integration with FastMCP server reference
3. Update initialization order to create services before middleware registration
4. Keep domain-specific middleware (chunking, filtering) in server
5. Update tool handlers to use middleware services

**Testing:**
- Server initialization testing
- Middleware registration validation
- Tool functionality testing
- End-to-end integration testing

#### Step 5.2: Update Server Factory
**Files to modify:**
- `src/codeweaver/server.py` (factory function)

**Tasks:**
1. Update `create_server` function to handle services config
2. Add configuration validation
3. Update error handling for initialization failures

**Testing:**
- Server factory testing
- Configuration error handling
- Initialization failure recovery

### Phase 6: Testing and Validation (Estimated: 3-4 days)

#### Step 6.1: Comprehensive Unit Testing
**Files to create:**
- `tests/unit/test_middleware_providers.py`
- `tests/unit/test_services_manager_middleware.py`
- `tests/unit/test_middleware_config.py`

**Tasks:**
1. Create comprehensive unit test suite for all middleware providers
2. Test services manager middleware integration
3. Test configuration loading and validation
4. Test error scenarios and edge cases

#### Step 6.2: Integration Testing
**Files to create:**
- `tests/integration/test_middleware_integration.py`
- `tests/integration/test_server_with_middleware_services.py`

**Tasks:**
1. Create end-to-end integration tests
2. Test server initialization with middleware services
3. Test tool execution with middleware service injection
4. Test configuration migration scenarios

#### Step 6.3: Performance Testing
**Files to create:**
- `tests/performance/test_middleware_performance.py`

**Tasks:**
1. Benchmark middleware service overhead
2. Compare performance with direct middleware usage
3. Test memory usage and resource consumption
4. Validate that performance remains acceptable

### Phase 7: Documentation and Deployment (Estimated: 2-3 days)

#### Step 7.1: Update Documentation
**Files to create/modify:**
- `docs/services-architecture.md`
- `docs/middleware-configuration.md`
- `README.md` updates
- `CLAUDE.md` updates

**Tasks:**
1. Document new services architecture
2. Create middleware configuration guide
3. Update README with new configuration options
4. Update CLAUDE.md with new development patterns

#### Step 7.2: Migration Guide
**Files to create:**
- `docs/migration/middleware-services-migration.md`

**Tasks:**
1. Create step-by-step migration guide for existing users
2. Document breaking changes (if any)
3. Provide troubleshooting help
4. Create automated migration tools if needed

## Implementation Checklist

### Pre-Implementation
- [ ] Review current codebase architecture
- [ ] Identify potential conflicts or dependencies
- [ ] Set up development branch
- [ ] Plan testing strategy

### Phase 1: Foundation
- [ ] Create type system extensions
- [ ] Add middleware service types to ServiceType enum
- [ ] Create middleware service configuration classes
- [ ] Define middleware service protocols
- [ ] Update health report structures
- [ ] Create base middleware provider class
- [ ] Validate type definitions compile correctly

### Phase 2: Providers
- [ ] Implement FastMCPLoggingProvider
- [ ] Implement FastMCPTimingProvider
- [ ] Implement FastMCPErrorHandlingProvider
- [ ] Implement FastMCPRateLimitingProvider
- [ ] Create unit tests for all providers
- [ ] Validate middleware wrapper functionality

### Phase 3: Services Manager
- [ ] Update ServicesManager constructor
- [ ] Add middleware service registration
- [ ] Update health monitoring
- [ ] Update shutdown process
- [ ] Update ServiceBridge for middleware injection
- [ ] Create integration tests

### Phase 4: Configuration
- [ ] Add ServicesConfig to main configuration
- [ ] Implement configuration migration
- [ ] Add validation helpers
- [ ] Create example configurations
- [ ] Test configuration loading

### Phase 5: Server Integration
- [ ] Remove hardcoded middleware from server.py
- [ ] Integrate ServicesManager with server
- [ ] Update initialization order
- [ ] Update server factory
- [ ] Test end-to-end functionality

### Phase 6: Testing
- [ ] Create comprehensive unit test suite
- [ ] Create integration test suite
- [ ] Perform performance testing
- [ ] Validate backward compatibility
- [ ] Test migration scenarios

### Phase 7: Documentation
- [ ] Update architecture documentation
- [ ] Create configuration guide
- [ ] Create migration guide
- [ ] Update README and CLAUDE.md
- [ ] Review all documentation for accuracy

## Risk Mitigation

### Risk 1: Performance Degradation
**Mitigation:**
- Implement lightweight metrics collection
- Benchmark each phase
- Use efficient data structures
- Monitor memory usage

### Risk 2: Breaking Changes
**Mitigation:**
- Implement configuration migration
- Maintain backward compatibility where possible
- Provide clear migration documentation
- Test with existing configurations

### Risk 3: Complex Dependencies
**Mitigation:**
- Careful dependency injection design
- Robust error handling
- Graceful degradation when services unavailable
- Clear separation of concerns

### Risk 4: Testing Complexity
**Mitigation:**
- Modular testing approach
- Mock FastMCP components where needed
- Isolate service testing from middleware testing
- Comprehensive integration testing

## Success Criteria

### Functional Requirements
- [ ] All FastMCP middleware accessible as services
- [ ] Middleware functionality preserved
- [ ] Configuration fully TOML-based
- [ ] Health monitoring includes middleware services
- [ ] Service injection works in tool contexts

### Performance Requirements
- [ ] No measurable performance degradation (< 5%)
- [ ] Memory usage increase < 10%
- [ ] Initialization time increase < 2 seconds

### Quality Requirements
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass
- [ ] No regressions in existing functionality
- [ ] Documentation complete and accurate

### User Experience Requirements
- [ ] Existing configurations work with migration
- [ ] Clear error messages for configuration issues
- [ ] Smooth upgrade path documented
- [ ] New features are discoverable

## Timeline Estimate

**Total Estimated Time: 18-25 days**

- Phase 1: Foundation Setup (2-3 days)
- Phase 2: Middleware Service Providers (4-5 days)
- Phase 3: Services Manager Integration (3-4 days)
- Phase 4: Configuration Integration (2-3 days)
- Phase 5: Server Integration (2-3 days)
- Phase 6: Testing and Validation (3-4 days)
- Phase 7: Documentation and Deployment (2-3 days)

## Post-Implementation

### Monitoring
- Monitor performance metrics in production
- Track configuration migration success
- Monitor error rates and service health

### Future Enhancements
- Add more middleware service providers
- Enhance metrics collection and reporting
- Add service-to-service communication patterns
- Consider GraphQL or REST API for service status

### Maintenance
- Regular review of service configurations
- Update documentation as needed
- Monitor for FastMCP updates that may affect integration
- Continuous improvement based on user feedback
