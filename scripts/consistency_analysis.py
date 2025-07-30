#!/usr/bin/env python
"""Refined Protocol Consistency Analysis.

More targeted analysis that considers the actual architecture and usage patterns.
"""

import argparse
import ast

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MethodUsage:
    """Track where methods are used and tested."""

    method_name: str
    implementations: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    factory_usage: list[str] = field(default_factory=list)


class RefinedConsistencyAnalyzer:
    """More targeted consistency analysis."""

    def __init__(self, src_path: Path):
        self.src_path = src_path
        self.test_path = src_path.parent / "tests"

    def analyze_factory_utility_methods(self) -> dict[str, MethodUsage]:
        """Analyze utility methods that should be consistent across factory-managed components."""
        # These are the methods that should be consistent across packages
        # because they're used by factories and service managers
        target_methods = {
            "initialize",
            "shutdown",
            "start",
            "stop",
            "health_check",
            "check_availability",
            "get_capabilities",
        }

        method_usage = {}
        for method in target_methods:
            method_usage[method] = MethodUsage(method_name=method)

        # Analyze each package for these methods
        packages = ["providers", "backends", "sources", "services"]

        for package in packages:
            package_path = self.src_path / "codeweaver" / package
            if package_path.exists():
                self._analyze_package_methods(package_path, method_usage, package)

        # Find test usage
        if self.test_path.exists():
            self._analyze_test_usage(method_usage)

        # Find factory usage
        self._analyze_factory_usage(method_usage)

        return method_usage

    def _analyze_package_methods(
        self, package_path: Path, method_usage: dict[str, MethodUsage], package: str
    ) -> None:
        """Analyze methods in a specific package."""
        python_files = list(package_path.rglob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                # Find class definitions and their methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                                method_name = item.name
                                if method_name in method_usage:
                                    impl_info = (
                                        f"{package}.{node.name}.{method_name} ({py_file.name})"
                                    )
                                    method_usage[method_name].implementations.append(impl_info)

            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

    def _analyze_test_usage(self, method_usage: dict[str, MethodUsage]) -> None:
        """Find where these methods are tested."""
        if not self.test_path.exists():
            return

        test_files = list(self.test_path.rglob("test_*.py"))

        for test_file in test_files:
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()

                for method_name in method_usage:
                    if method_name in content or f".{method_name}(" in content:
                        method_usage[method_name].test_files.append(
                            str(test_file.relative_to(self.src_path.parent))
                        )

            except Exception as e:
                print(f"⚠️  Error analyzing test {test_file}: {e}")

    def _analyze_factory_usage(self, method_usage: dict[str, MethodUsage]) -> None:
        """Find where these methods are used by factories."""
        factory_dirs = [
            self.src_path / "codeweaver" / "factories",
            self.src_path / "codeweaver" / "services" / "manager.py",
        ]

        factory_files = []
        for factory_dir in factory_dirs:
            if factory_dir.is_file():
                factory_files.append(factory_dir)
            elif factory_dir.is_dir():
                factory_files.extend(factory_dir.rglob("*.py"))

        for factory_file in factory_files:
            try:
                with open(factory_file, "r", encoding="utf-8") as f:
                    content = f.read()

                for method_name in method_usage:
                    if f".{method_name}(" in content or f"await {method_name}(" in content:
                        usage_info = f"{factory_file.relative_to(self.src_path)}"
                        method_usage[method_name].factory_usage.append(usage_info)

            except Exception as e:
                print(f"⚠️  Error analyzing factory {factory_file}: {e}")

    def analyze_service_implementations(self) -> dict[str, list[str]]:
        """Analyze which services implement BaseServiceProvider."""
        services_path = self.src_path / "codeweaver" / "services" / "providers"
        service_implementations = {"implements_base": [], "missing_base": [], "service_types": []}

        if not services_path.exists():
            return service_implementations

        service_files = [
            f
            for f in services_path.glob("*.py")
            if f.name not in ["__init__.py", "base_provider.py"]
        ]

        for service_file in service_files:
            try:
                with open(service_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it inherits from BaseServiceProvider
                        base_names = [self._extract_base_name(base) for base in node.bases]

                        if "BaseServiceProvider" in base_names:
                            service_implementations["implements_base"].append(
                                f"{node.name} ({service_file.name})"
                            )
                        else:
                            # Check if it's a service class (ends with Service or Provider)
                            if (
                                node.name.endswith(("Service", "Provider"))
                                and node.name != "BaseServiceProvider"
                            ):
                                service_implementations["missing_base"].append(
                                    f"{node.name} ({service_file.name})"
                                )

                        # Identify service types
                        service_implementations["service_types"].append(node.name)

            except Exception as e:
                print(f"⚠️  Error analyzing service {service_file}: {e}")

        return service_implementations

    def _extract_base_name(self, base: ast.expr) -> str:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        return ast.unparse(base)

    def analyze_protocol_specificity(self) -> dict[str, dict[str, list[str]]]:
        """Analyze which protocols are specific to certain implementations."""
        protocol_analysis = {
            "providers": {"embedding_only": [], "reranking_only": [], "combined": []},
            "backends": {
                "basic_vector": [],
                "hybrid_search": [],
                "streaming": [],
                "transactional": [],
            },
            "services": {
                "chunking_specific": [],
                "filtering_specific": [],
                "telemetry_specific": [],
                "caching_specific": [],
                "rate_limiting_specific": [],
            },
        }

        # Analyze providers
        providers_path = self.src_path / "codeweaver" / "providers"
        if providers_path.exists():
            for py_file in providers_path.glob("*.py"):
                if py_file.name in ["__init__.py", "base.py", "factory.py"]:
                    continue

                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    file_stem = py_file.stem

                    if "RerankProvider" in content and "EmbeddingProvider" in content:
                        protocol_analysis["providers"]["combined"].append(file_stem)
                    elif "RerankProvider" in content:
                        protocol_analysis["providers"]["reranking_only"].append(file_stem)
                    elif "EmbeddingProvider" in content:
                        protocol_analysis["providers"]["embedding_only"].append(file_stem)

                except Exception as e:
                    print(f"⚠️  Error analyzing provider {py_file}: {e}")

        # Analyze backends
        backends_path = self.src_path / "codeweaver" / "backends"
        if backends_path.exists():
            for py_file in backends_path.glob("*.py"):
                if py_file.name in ["__init__.py", "base.py", "factory.py"]:
                    continue

                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    file_stem = py_file.stem

                    if "HybridSearchBackend" in content:
                        protocol_analysis["backends"]["hybrid_search"].append(file_stem)
                    elif "StreamingBackend" in content:
                        protocol_analysis["backends"]["streaming"].append(file_stem)
                    elif "TransactionalBackend" in content:
                        protocol_analysis["backends"]["transactional"].append(file_stem)
                    else:
                        protocol_analysis["backends"]["basic_vector"].append(file_stem)

                except Exception as e:
                    print(f"⚠️  Error analyzing backend {py_file}: {e}")

        return protocol_analysis


def generate_refined_report(
    factory_methods: dict[str, MethodUsage],
    service_analysis: dict[str, list[str]],
    protocol_analysis: dict[str, dict[str, list[str]]],
) -> str:
    """Generate refined consistency report."""
    report = []
    report.append("# Refined Protocol Consistency Analysis\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("This analysis focuses on:")
    report.append("1. **Factory utility methods** that need consistency across packages")
    report.append("2. **Service implementations** and their BaseServiceProvider compliance")
    report.append("3. **Protocol specificity** - not all components need all protocols")
    report.append("4. **Test coverage** for methods that might change\n")

    # Factory utility methods analysis
    report.append("## Factory Utility Methods Analysis\n")
    report.append(
        "These methods need consistency because they're used by factories and service managers:\n"
    )

    for method_name, usage in factory_methods.items():
        report.append(f"### {method_name}")

        if usage.implementations:
            report.append("**Implementations found:**")
            for impl in usage.implementations:
                report.append(f"- {impl}")
        else:
            report.append("**⚠️  No implementations found**")

        if usage.factory_usage:
            report.append("**Used by factories:**")
            for factory in usage.factory_usage:
                report.append(f"- {factory}")
        else:
            report.append("**⚠️  Not used by factories**")

        if usage.test_files:
            report.append("**Test coverage:**")
            for test in usage.test_files:
                report.append(f"- {test}")
        else:
            report.append("**⚠️  No test coverage found**")

        report.append("")

    # Service implementations analysis
    report.append("## Service Implementations Analysis\n")

    report.append("### ✅ Services implementing BaseServiceProvider:")
    for service in service_analysis["implements_base"]:
        report.append(f"- {service}")

    if service_analysis["missing_base"]:
        report.append("\n### ⚠️  Services NOT implementing BaseServiceProvider:")
        for service in service_analysis["missing_base"]:
            report.append(f"- {service}")

    report.append("\n### All Service Types Found:")
    for service_type in service_analysis["service_types"]:
        report.append(f"- {service_type}")

    report.append("")

    # Protocol specificity analysis
    report.append("## Protocol Specificity Analysis\n")
    report.append("Not all implementations need all protocols - this is by design:\n")

    for category, protocols in protocol_analysis.items():
        report.append(f"### {category.title()}")
        for protocol_type, implementations in protocols.items():
            if implementations:
                report.append(f"**{protocol_type.replace('_', ' ').title()}:**")
                for impl in implementations:
                    report.append(f"- {impl}")
        report.append("")

    # Targeted recommendations
    report.append("## Targeted Recommendations\n")

    report.append("### 🎯 High Priority")
    report.append(
        "1. **Service BaseServiceProvider compliance**: Ensure all services inherit from BaseServiceProvider"
    )
    report.append(
        "2. **Factory utility method consistency**: Standardize signatures for methods used by factories"
    )
    report.append("3. **Test coverage**: Add tests for utility methods that lack coverage\n")

    report.append("### 🔧 Medium Priority")
    report.append(
        "1. **Protocol-specific validation**: Don't force all protocols on all implementations"
    )
    report.append(
        "2. **Factory integration**: Ensure factory methods actually use the utility methods"
    )
    report.append(
        "3. **Lifecycle management**: Standardize start/stop vs initialize/shutdown patterns\n"
    )

    report.append("### 💡 Low Priority")
    report.append(
        "1. **Service-specific protocols**: ChunkingService only for chunking services, etc."
    )
    report.append(
        "2. **Optional capabilities**: Some backends don't need hybrid search, some providers don't need reranking"
    )

    return "\n".join(report)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Refined protocol consistency analysis")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/refined_consistency_report.md"),
        help="Output file for refined report",
    )

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🎯 Running refined consistency analysis...")
    analyzer = RefinedConsistencyAnalyzer(args.src_path)

    # Run analyses
    factory_methods = analyzer.analyze_factory_utility_methods()
    service_analysis = analyzer.analyze_service_implementations()
    protocol_analysis = analyzer.analyze_protocol_specificity()

    # Generate report
    report = generate_refined_report(factory_methods, service_analysis, protocol_analysis)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"📄 Refined report written to {args.output}")
    print("\n✅ Refined analysis complete!")

    # Summary stats
    total_factory_methods = len([m for m in factory_methods.values() if m.implementations])
    missing_tests = len([m for m in factory_methods.values() if not m.test_files])
    services_with_base = len(service_analysis["implements_base"])
    services_without_base = len(service_analysis["missing_base"])

    print("\n📊 Quick Stats:")
    print(f"  • Factory utility methods found: {total_factory_methods}")
    print(f"  • Methods missing tests: {missing_tests}")
    print(f"  • Services with BaseServiceProvider: {services_with_base}")
    print(f"  • Services without BaseServiceProvider: {services_without_base}")

    return 0


if __name__ == "__main__":
    exit(main())
