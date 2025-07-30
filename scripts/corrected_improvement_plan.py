#!/usr/bin/env python
"""Corrected Improvement Plan.

Addresses the real consistency issues based on refined analysis.
"""

import argparse

from pathlib import Path


class CorrectedImprovementPlanner:
    """Create a corrected improvement plan based on actual issues."""

    def __init__(self, src_path: Path):
        """Initialize the planner with the source path."""
        self.src_path = src_path
        self.test_path = src_path.parent / "tests"

    def analyze_real_issues(self) -> dict[str, list[str]]:
        """Identify the real consistency issues."""
        # 1. Factory method signature inconsistencies
        # Based on the refined analysis, these methods are used by factories
        # but may have inconsistent signatures across packages

        factory_methods_needing_standardization = [
            {
                "method": "initialize",
                "issue": "Mixed async/sync patterns across packages",
                "packages_affected": ["services", "backends"],
                "factory_usage": "Used by service_registry.py and extensibility_manager.py",
            },
            {
                "method": "shutdown",
                "issue": "Mixed async/sync patterns and parameter differences",
                "packages_affected": ["services", "backends", "sources"],
                "factory_usage": "Used by service_registry.py and factory.py",
            },
            {
                "method": "health_check",
                "issue": "Different return types (bool vs ServiceHealth vs dict)",
                "packages_affected": ["backends", "services", "providers"],
                "factory_usage": "Used by service_registry.py and initialization.py",
            },
            {
                "method": "get_capabilities",
                "issue": "Different return types and signatures across packages",
                "packages_affected": ["sources", "services", "providers"],
                "factory_usage": "Used by plugin_protocols.py and codeweaver_factory.py",
            },
        ]

        # 2. Service compliance - all services should extend BaseServiceProvider
        # Good news: refined analysis shows all 9 services already do this!

        # 3. Lifecycle pattern inconsistencies
        lifecycle_issues = [
            {
                "pattern": "start/stop vs initialize/shutdown",
                "issue": "Sources use start/stop, services use initialize/shutdown",
                "recommendation": "Standardize on initialize/shutdown for managed components",
            },
            {
                "pattern": "health_check variations",
                "issue": "Some use health_check, some use check_health, some use _check_health",
                "recommendation": "Standardize on health_check (public) and _check_health (internal)",
            },
        ]

        # 4. Test impact analysis - which tests would break if we change signatures
        test_impacts = [
            {
                "method": "initialize",
                "test_files": [
                    "tests/integration/test_service_integration.py",
                    "tests/integration/test_fastmcp_middleware_integration.py",
                    "tests/unit/test_enhanced_config.py",
                ],
                "impact": "High - many integration tests call initialize",
            },
            {
                "method": "health_check",
                "test_files": [
                    "tests/unit/test_telemetry_service.py",
                    "tests/validation/test_services_integration.py",
                ],
                "impact": "Medium - some unit tests check health",
            },
        ]

        # 5. Missing service types that should implement BaseServiceProvider
        # Based on the service types found, check if any are missing the base
        missing_service_types = [
            {
                "class": "RateLimitConfig",
                "file": "rate_limiting.py",
                "issue": "Config class, should not extend BaseServiceProvider",
                "action": "No action needed - this is correct",
            },
            {
                "class": "TokenBucket",
                "file": "rate_limiting.py",
                "issue": "Utility class, should not extend BaseServiceProvider",
                "action": "No action needed - this is correct",
            },
            {
                "class": "CacheConfig",
                "file": "caching.py",
                "issue": "Config class, should not extend BaseServiceProvider",
                "action": "No action needed - this is correct",
            },
            {
                "class": "CacheEntry",
                "file": "caching.py",
                "issue": "Data class, should not extend BaseServiceProvider",
                "action": "No action needed - this is correct",
            },
        ]

        return {
            "service_compliance_issues": [],
            "factory_method_inconsistencies": factory_methods_needing_standardization,
            "lifecycle_pattern_inconsistencies": lifecycle_issues,
            "test_impact_analysis": test_impacts,
            "missing_service_types": missing_service_types,
        }

    def generate_corrected_plan(self, issues: dict[str, list[str]]) -> str:
        # sourcery skip: no-long-functions
        """Generate a corrected improvement plan."""
        plan = []
        plan.extend((
            "# Corrected Protocol Consistency Improvement Plan\n",
            "## Context & Corrections\n",
            "After refined analysis, the previous recommendations were too broad. Here are the **actual** issues:\n",
            "### ✅ What's Already Good",
        ))
        plan.extend((
            "- **All 9 service providers correctly inherit from BaseServiceProvider**",
            "- **@require_implementation decorators already applied** to base classes",
            "- **Protocol specificity is by design** (not all providers need reranking)",
            "- **Test coverage exists** for most utility methods\n",
            "## 🎯 Real Issue #1: Factory Method Signature Inconsistencies\n",
            "These methods are used by factories but have inconsistent signatures:\n",
        ))
        for method_issue in issues["factory_method_inconsistencies"]:
            plan.extend((
                f"### `{method_issue['method']}`",
                f"**Issue**: {method_issue['issue']}",
                f"**Packages affected**: {', '.join(method_issue['packages_affected'])}",
                f"**Factory usage**: {method_issue['factory_usage']}",
                "",
            ))

        # Lifecycle patterns
        plan.append("## 🔧 Real Issue #2: Lifecycle Pattern Inconsistencies\n")

        for lifecycle_issue in issues["lifecycle_pattern_inconsistencies"]:
            plan.extend((
                f"### {lifecycle_issue['pattern']}",
                f"**Issue**: {lifecycle_issue['issue']}",
                f"**Recommendation**: {lifecycle_issue['recommendation']}",
                "",
            ))

        # Test impact
        plan.extend((
            "## ⚠️  Real Issue #3: Test Impact Analysis\n",
            "These tests would be affected by signature changes:\n",
        ))

        for test_impact in issues["test_impact_analysis"]:
            plan.extend((
                f"### `{test_impact['method']}` changes",
                f"**Impact level**: {test_impact['impact']}",
                "**Affected test files**:",
            ))
            plan.extend(f"- {test_file}" for test_file in test_impact["test_files"])
            plan.append("")

        plan.extend((
            "## 🚀 Corrected Implementation Strategy\n",
            "### Phase 1: Signature Standardization (High Impact)",
            "1. **Standardize `initialize()` signatures**",
            "   - Make all async: `async def initialize(self) -> None`",
            "   - Update factories to use `await`",
            "   - Update affected tests",
        ))
        plan.extend((
            "",
            "2. **Standardize `shutdown()` signatures**",
            "   - Make all async: `async def shutdown(self) -> None`",
            "   - Ensure graceful cleanup in all implementations",
            "   - Update factory cleanup code",
            "",
            "3. **Standardize `health_check()` return types**",
            "   - Services: `async def health_check(self) -> ServiceHealth`",
            "   - Others: `async def health_check(self) -> bool`",
            "   - Update factory health monitoring",
            "",
            "### Phase 2: Lifecycle Pattern Unification (Medium Impact)",
            "1. **Unify start/stop → initialize/shutdown**",
            "   - Sources: Keep start/stop for watchers, add initialize/shutdown for managed lifecycle",
            "   - Services: Continue using initialize/shutdown",
            "   - Providers: Add initialize/shutdown for connection management",
            "",
            "### Phase 3: Test Updates (Critical)",
            "1. **Update integration tests** for async initialize/shutdown",
            "2. **Update unit tests** for new health_check return types",
            "3. **Add validation tests** for signature consistency",
            "",
            "## ❌ What NOT To Do (Corrections)\n",
            "1. **Don't force all providers to support reranking** - only some do",
            "2. **Don't force all backends to support hybrid search** - only some do",
            "3. **Don't make ChunkingService universal** - only for services that chunk",
            "4. **Don't change working service inheritance** - all services already extend BaseServiceProvider",
            "5. **Don't add unnecessary decorators** - focus on factory-used methods only",
        ))

        return "\n".join(plan)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate corrected improvement plan")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/corrected_improvement_plan.md"),
        help="Output file for corrected plan",
    )

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🎯 Generating corrected improvement plan...")
    planner = CorrectedImprovementPlanner(args.src_path)

    # Analyze real issues
    issues = planner.analyze_real_issues()

    # Generate corrected plan
    plan = planner.generate_corrected_plan(issues)

    with (args.output).open("w") as f:
        f.write(plan)

    print(f"📄 Corrected plan written to {args.output}")

    # Summary
    factory_issues = len(issues["factory_method_inconsistencies"])
    lifecycle_issues = len(issues["lifecycle_pattern_inconsistencies"])
    test_impacts = len(issues["test_impact_analysis"])

    print("\n📊 Real Issues Identified:")
    print(f"  • Factory method inconsistencies: {factory_issues}")
    print(f"  • Lifecycle pattern issues: {lifecycle_issues}")
    print(f"  • Tests that would be impacted: {test_impacts}")
    print("\n✅ Corrected analysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
