#!/usr/bin/env python
"""Protocol Consistency Improvement Script.

Applies systematic improvements to achieve consistency across packages.
"""

import argparse
import json

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConsistencyImprovement:
    """Represents a specific improvement to be applied."""

    category: str  # "signature", "protocol", "decorator", "naming"
    description: str
    file_path: str
    changes: list[dict[str, Any]]
    impact: str  # "low", "medium", "high"
    risk: str  # "low", "medium", "high"


class ProtocolConsistencyImprover:
    """Applies systematic consistency improvements."""

    def __init__(self, src_path: Path):
        """Initialize the protocol consistency improver."""
        self.src_path = src_path
        self.improvements: list[ConsistencyImprovement] = []

    def plan_improvements(self) -> list[ConsistencyImprovement]:
        """Plan all consistency improvements."""
        self._plan_signature_standardization()
        self._plan_protocol_enforcement()
        self._plan_decorator_application()
        self._plan_naming_consistency()

        return self.improvements

    def _plan_signature_standardization(self) -> None:
        """Plan standardization of common method signatures."""
        # Standard __init__ signatures by package
        standard_signatures = {
            "providers": {
                "__init__": ["config", "*", "logger", "api_key"],
                "health_check": [],
                "validate_api_key": ["api_key"],
                "get_capabilities": [],
                "_ensure_initialized": [],
            },
            "backends": {
                "__init__": ["config", "*", "logger", "client"],
                "health_check": [],
                "initialize": [],
                "shutdown": [],
                "get_capabilities": [],
            },
            "sources": {
                "__init__": ["source_id", "*", "config"],
                "health_check": [],
                "start": [],
                "stop": [],
                "get_capabilities": [],
                "check_availability": [],
            },
            "services": {
                "__init__": ["config", "*", "logger", "fastmcp_server"],
                "health_check": [],
                "_initialize_provider": [],
                "_shutdown_provider": [],
                "_check_health": [],
            },
        }

        for package, methods in standard_signatures.items():
            for method_name, expected_args in methods.items():
                improvement = ConsistencyImprovement(
                    category="signature",
                    description=f"Standardize {method_name} signature in {package} package",
                    file_path=f"src/codeweaver/{package}/",
                    changes=[
                        {
                            "method": method_name,
                            "expected_args": expected_args,
                            "action": "standardize_signature",
                        }
                    ],
                    impact="high",
                    risk="medium",
                )
                self.improvements.append(improvement)

    def _plan_protocol_enforcement(self) -> None:
        """Plan protocol compliance enforcement."""
        protocol_mappings = {
            "providers": ["EmbeddingProvider", "RerankProvider"],
            "backends": ["VectorBackend", "HybridSearchBackend"],
            "sources": ["DataSource"],
            "services": ["ServiceProvider", "ChunkingService", "FilteringService"],
        }

        for package, protocols in protocol_mappings.items():
            for protocol in protocols:
                improvement = ConsistencyImprovement(
                    category="protocol",
                    description=f"Ensure all {package} implement {protocol} protocol",
                    file_path=f"src/codeweaver/{package}/",
                    changes=[
                        {
                            "protocol": protocol,
                            "action": "validate_protocol_compliance",
                            "add_runtime_checks": True,
                        }
                    ],
                    impact="high",
                    risk="low",
                )
                self.improvements.append(improvement)

    def _plan_decorator_application(self) -> None:
        """Plan application of @require_implementation decorator."""
        # Methods that should be decorated in base classes
        required_methods = {
            "providers/base.py": {
                "EmbeddingProviderBase": ["embed_documents", "embed_query"],
                "RerankProviderBase": ["rerank_documents"],
            },
            "backends/base.py": {"VectorBackend": ["initialize", "search", "upsert"]},
            "sources/base.py": {"AbstractDataSource": ["discover_content", "read_content"]},
            "services/providers/base_provider.py": {
                "BaseServiceProvider": ["_initialize_provider", "_shutdown_provider"]
            },
        }

        for file_path, classes in required_methods.items():
            for class_name, methods in classes.items():
                improvement = ConsistencyImprovement(
                    category="decorator",
                    description=f"Apply @require_implementation to {class_name} methods",
                    file_path=f"src/codeweaver/{file_path}",
                    changes=[
                        {
                            "class": class_name,
                            "methods": methods,
                            "action": "add_require_implementation_decorator",
                        }
                    ],
                    impact="medium",
                    risk="low",
                )
                self.improvements.append(improvement)

    def _plan_naming_consistency(self) -> None:
        """Plan naming consistency improvements."""
        # Standard naming patterns
        naming_standards = [
            {
                "pattern": "health_check",
                "description": "Standardize health check method name",
                "alternatives": ["check_health", "_check_health", "is_healthy"],
            },
            {
                "pattern": "initialize",
                "description": "Standardize initialization method name",
                "alternatives": ["init", "_init", "setup", "_setup"],
            },
            {
                "pattern": "shutdown",
                "description": "Standardize shutdown method name",
                "alternatives": ["close", "_close", "cleanup", "_cleanup", "stop"],
            },
            {
                "pattern": "get_capabilities",
                "description": "Standardize capability query method name",
                "alternatives": ["capabilities", "get_capability", "list_capabilities"],
            },
        ]

        for standard in naming_standards:
            improvement = ConsistencyImprovement(
                category="naming",
                description=standard["description"],
                file_path="src/codeweaver/",
                changes=[
                    {
                        "standard_name": standard["pattern"],
                        "alternatives": standard["alternatives"],
                        "action": "standardize_method_names",
                    }
                ],
                impact="medium",
                risk="medium",
            )
            self.improvements.append(improvement)


def generate_improvement_plan(improvements: list[ConsistencyImprovement]) -> str:    # noqa: C901
    # sourcery skip: no-long-functions
    """Generate a comprehensive improvement plan."""
    # Summary
    total_improvements = len(improvements)
    high_impact = len([i for i in improvements if i.impact == "high"])
    high_risk = len([i for i in improvements if i.risk == "high"])

    report = [
        "# Protocol Consistency Improvement Plan\n",
        "## Executive Summary\n",
        f"📊 **Total Improvements**: {total_improvements}",
        f"🎯 **High Impact**: {high_impact}",
    ]
    report.extend((f"⚠️ **High Risk**: {high_risk}", ""))

    # Group by category
    by_category = defaultdict(list)
    for improvement in improvements:
        by_category[improvement.category].append(improvement)

    for category, items in by_category.items():
        report.append(f"## {category.title()} Improvements ({len(items)})\n")

        for i, improvement in enumerate(items, 1):
            report.extend((f"### {i}. {improvement.description}", f"- **File**: {improvement.file_path}", f"- **Impact**: {improvement.impact}", f"- **Risk**: {improvement.risk}"))

            if improvement.changes:
                report.append("- **Changes**:")
                for change in improvement.changes:
                    if change.get("action") == "standardize_signature":
                        report.append(
                            f"  - Standardize `{change['method']}({', '.join(change['expected_args'])})`"
                        )
                    elif change.get("action") == "validate_protocol_compliance":
                        report.append(f"  - Validate {change['protocol']} protocol compliance")
                    elif change.get("action") == "add_require_implementation_decorator":
                        report.append(
                            f"  - Add @require_implementation({', '.join(change['methods'])}) to {change['class']}"
                        )
                    elif change.get("action") == "standardize_method_names":
                        report.append(
                            f"  - Rename {', '.join(change['alternatives'])} → {change['standard_name']}"
                        )
            report.append("")

    # Implementation phases
    report.append("## Implementation Phases\n")

    phases = [
        {
            "name": "Phase 1: Low Risk Improvements",
            "description": "Apply decorators and protocol validation",
            "items": [i for i in improvements if i.risk == "low"],
        },
        {
            "name": "Phase 2: Medium Risk Improvements",
            "description": "Standardize signatures and naming",
            "items": [i for i in improvements if i.risk == "medium"],
        },
        {
            "name": "Phase 3: High Risk Improvements",
            "description": "Breaking changes requiring careful testing",
            "items": [i for i in improvements if i.risk == "high"],
        },
    ]

    for phase in phases:
        report.extend((f"### {phase['name']} ({len(phase['items'])} items)", f"{phase['description']}\n"))

        report.extend(f"- {improvement.description}" for improvement in phase["items"])
        report.extend((
            "",
            "## Enforcement Mechanisms\n",
            "### 1. 🔧 Enhanced Decorators",
            "- Apply `@require_implementation` to abstract methods",
            "- Use `@not_implemented` for placeholder classes",
            "- Add validation decorators for protocol compliance",
            "",
            "### 2. 📊 Runtime Validation",
            "- Protocol compliance checking at initialization",
            "- Method signature validation",
            "- Capability consistency verification",
            "",
            "### 3. 🧪 Testing Integration",
            "- Automated protocol compliance tests",
            "- Signature consistency validation",
            "- Cross-package integration testing",
            "",
            "### 4. 🔍 Static Analysis- Enhanced mypy config for protocol checking",
            "- Custom linting rules for consistency",
            "- Pre-commit hooks for validation",
            "",
            "## Success Metrics\n",
            "- ✅ Zero signature inconsistencies across packages",
            "- ✅ 100% protocol compliance for all implementations",
            "- ✅ All abstract methods decorated with @require_implementation",
            "- ✅ Consistent naming patterns across all packages",
            "- ✅ Runtime validation passing for all components",
        ))

    return "\n".join(report)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plan protocol consistency improvements")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/improvement_plan.md"),
        help="Output file for improvement plan",
    )
    parser.add_argument("--json", action="store_true", help="Output plan as JSON")

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🎯 Planning consistency improvements...")
    improver = ProtocolConsistencyImprover(args.src_path)
    improvements = improver.plan_improvements()

    if args.json:
        # Output as JSON
        json_data = {
            "total_improvements": len(improvements),
            "by_category": {},
            "by_risk": {},
            "improvements": [],
        }

        # Group by category and risk
        for improvement in improvements:
            if improvement.category not in json_data["by_category"]:
                json_data["by_category"][improvement.category] = 0
            json_data["by_category"][improvement.category] += 1

            if improvement.risk not in json_data["by_risk"]:
                json_data["by_risk"][improvement.risk] = 0
            json_data["by_risk"][improvement.risk] += 1

            json_data["improvements"].append({
                "category": improvement.category,
                "description": improvement.description,
                "file_path": improvement.file_path,
                "impact": improvement.impact,
                "risk": improvement.risk,
                "changes": improvement.changes,
            })

        with (args.output).open(args.output.with_suffix(".json"), "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"📄 JSON plan written to {args.output.with_suffix('.json')}")
    else:
        # Generate markdown plan
        plan = generate_improvement_plan(improvements)

        with (args.output).open("w") as f:
            f.write(plan)
        print(f"📄 Plan written to {args.output}")

    print(f"\n✅ Planning complete! {len(improvements)} improvements identified.")
    return 0


if __name__ == "__main__":
    exit(main())
