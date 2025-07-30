#!/usr/bin/env python
"""Protocol Consistency Analysis Script.

Systematically analyzes method implementations across providers, backends, sources, and services
to identify consistency issues and improvement opportunities.
"""

import argparse
import ast
import json

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MethodSignature:
    """Represents a method signature with arguments and return type."""

    name: str
    args: list[str]
    defaults: list[Any] = field(default_factory=list)
    return_type: str | None = None
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class ClassInfo:
    """Information about a class implementation."""

    name: str
    module: str
    file_path: str
    methods: list[MethodSignature] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)
    is_protocol: bool = False
    is_abstract: bool = False


@dataclass
class PackageAnalysis:
    """Analysis results for a package."""

    package_name: str
    implementations: dict[str, ClassInfo] = field(default_factory=dict)
    protocols: dict[str, ClassInfo] = field(default_factory=dict)
    common_methods: set[str] = field(default_factory=set)
    inconsistencies: list[str] = field(default_factory=list)


class ProtocolAnalyzer:
    """Analyzes protocol consistency across packages."""

    def __init__(self, src_path: Path):
        """Initialize the analyzer with the source path."""
        self.src_path = src_path
        self.packages = ["providers", "backends", "sources", "services"]
        self.analyses: dict[str, PackageAnalysis] = {}

    def analyze_all_packages(self) -> dict[str, PackageAnalysis]:
        """Analyze all packages for protocol consistency."""
        for package in self.packages:
            package_path = self.src_path / "codeweaver" / package
            if package_path.exists():
                self.analyses[package] = self.analyze_package(package_path, package)

        return self.analyses

    def analyze_package(self, package_path: Path, package_name: str) -> PackageAnalysis:
        """Analyze a single package."""
        analysis = PackageAnalysis(package_name=package_name)

        # Find all Python files
        python_files = list(package_path.rglob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]

        for py_file in python_files:
            try:
                self.analyze_file(py_file, analysis)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

        # Identify common methods and inconsistencies
        self._analyze_consistency(analysis)

        return analysis

    def analyze_file(self, file_path: Path, analysis: PackageAnalysis) -> None:
        """Analyze a single Python file."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"⚠️  Could not read {file_path} (encoding issue)")
            return

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠️  Syntax error in {file_path}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, file_path)

                if class_info.is_protocol:
                    analysis.protocols[class_info.name] = class_info
                else:
                    analysis.implementations[class_info.name] = class_info

    def _extract_class_info(self, node: ast.ClassDef, file_path: Path) -> ClassInfo:
        """Extract information from a class definition."""
        class_info = ClassInfo(
            name=node.name, module=self._get_module_name(file_path), file_path=str(file_path)
        )

        # Check if it's a protocol or abstract class
        class_info.is_protocol = any(
            base.id == "Protocol" if isinstance(base, ast.Name) else False for base in node.bases
        )

        class_info.is_abstract = any(
            decorator.id == "abstractmethod" if isinstance(decorator, ast.Name) else False
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            for decorator in method.decorator_list
        )

        # Extract base classes
        class_info.base_classes = [self._extract_base_name(base) for base in node.bases]

        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                method_sig = self._extract_method_signature(item)
                class_info.methods.append(method_sig)

        return class_info

    def _extract_method_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> MethodSignature:
        """Extract method signature information."""
        # Extract arguments
        args = []
        args.extend(arg.arg for arg in node.args.args if arg.arg != "self")
        # Extract defaults
        defaults = []
        if node.args.defaults:
            defaults = [ast.unparse(default) for default in node.args.defaults]

        return_type = ast.unparse(node.returns) if node.returns else None
        # Extract decorators
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]

        # Extract docstring
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            docstring = node.body[0].value.value

        return MethodSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            return_type=return_type,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            docstring=docstring,
        )

    def _extract_base_name(self, base: ast.expr) -> str:
        """Extract base class name from AST node."""
        return base.id if isinstance(base, ast.Name) else ast.unparse(base)

    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        rel_path = file_path.relative_to(self.src_path)
        parts = list(rel_path.parts[:-1])  # Remove .py extension
        parts.append(rel_path.stem)
        return ".".join(parts)

    def _analyze_consistency(self, analysis: PackageAnalysis) -> None:
        """Analyze consistency within a package."""
        # Find common methods across implementations
        method_counts = defaultdict(int)
        for impl in analysis.implementations.values():
            for method in impl.methods:
                method_counts[method.name] += 1

        # Methods that appear in multiple implementations
        impl_count = len(analysis.implementations)
        if impl_count > 1:
            analysis.common_methods = {
                method
                for method, count in method_counts.items()
                if count >= max(2, impl_count // 2)  # At least half of implementations
            }

        # Check for signature inconsistencies
        self._check_signature_consistency(analysis)

    def _check_signature_consistency(self, analysis: PackageAnalysis) -> None:  # noqa: C901
        """Check for method signature inconsistencies."""
        method_signatures = defaultdict(list)

        # Collect all signatures for each method
        for impl in analysis.implementations.values():
            for method in impl.methods:
                if method.name in analysis.common_methods:
                    method_signatures[method.name].append((impl.name, method))

        # Check for inconsistencies
        for method_name, signatures in method_signatures.items():
            if len(signatures) < 2:
                continue

            # Compare signatures
            base_sig = signatures[0][1]
            for impl_name, sig in signatures[1:]:
                inconsistencies = []

                if sig.args != base_sig.args:
                    inconsistencies.append(f"args differ: {sig.args} vs {base_sig.args}")

                if sig.return_type != base_sig.return_type:
                    inconsistencies.append(
                        f"return type differs: {sig.return_type} vs {base_sig.return_type}"
                    )

                if sig.is_async != base_sig.is_async:
                    inconsistencies.append(f"async differs: {sig.is_async} vs {base_sig.is_async}")

                if inconsistencies:
                    analysis.inconsistencies.append(
                        f"Method '{method_name}' in {impl_name}: {'; '.join(inconsistencies)}"
                    )


def generate_report(analyses: dict[str, PackageAnalysis]) -> str:  # noqa: C901
    # sourcery skip: low-code-quality, no-long-functions
    """Generate a comprehensive consistency report."""
    total_implementations = sum(len(a.implementations) for a in analyses.values())
    total_protocols = sum(len(a.protocols) for a in analyses.values())
    total_inconsistencies = sum(len(a.inconsistencies) for a in analyses.values())

    report = [
        "# Protocol Consistency Analysis Report\n",
        "## Summary\n",
        f"📊 **Total Implementations**: {total_implementations}",
        f"🔧 **Total Protocols**: {total_protocols}",
        f"⚠️  **Total Inconsistencies**: {total_inconsistencies}\n",
    ]
    # Package-by-package analysis
    for package_name, analysis in analyses.items():
        report.append(f"## Package: {package_name}\n")

        # Implementations
        if analysis.implementations:
            report.append("### Implementations")
            for impl_name, impl in analysis.implementations.items():
                report.extend((
                    f"- **{impl_name}** ({impl.file_path})",
                    f"  - Base classes: {', '.join(impl.base_classes) or 'None'}",
                    f"  - Methods: {len(impl.methods)}",
                ))
            report.append("")

        # Protocols
        if analysis.protocols:
            report.append("### Protocols")
            for proto_name, proto in analysis.protocols.items():
                report.extend((
                    f"- **{proto_name}** ({proto.file_path})",
                    f"  - Methods: {len(proto.methods)}",
                ))
            report.append("")

        # Common methods
        if analysis.common_methods:
            report.append("### Common Methods")
            report.extend(f"- {method}" for method in sorted(analysis.common_methods))
            report.append("")

        # Inconsistencies
        if analysis.inconsistencies:
            report.append("### ⚠️  Inconsistencies")
            report.extend(f"- {inconsistency}" for inconsistency in analysis.inconsistencies)
            report.append("")

    # Cross-package analysis
    report.append("## Cross-Package Analysis\n")

    # Find common utility methods across packages
    all_methods = defaultdict(set)
    for package_name, analysis in analyses.items():
        for impl in analysis.implementations.values():
            for method in impl.methods:
                all_methods[method.name].add(package_name)

    if utility_methods := {
        method: packages
        for method, packages in all_methods.items()
        if len(packages) >= 2  # Present in at least 2 packages
    }:
        report.append("### Utility Methods Found Across Packages")
        report.extend(
            f"- **{method}**: {', '.join(sorted(packages))}"
            for method, packages in sorted(utility_methods.items())
        )
        report.append("")

    # Recommendations
    report.append("## Recommendations\n")

    if total_inconsistencies > 0:
        report.extend((
            "### 🎯 Priority Issues",
            "1. **Standardize method signatures** for common utility methods",
            "2. **Implement missing protocol methods** in implementations",
            "3. **Use @require_implementation decorator** for mandatory methods",
            "",
        ))

    report.extend((
        "### 🔧 Enforcement Improvements",
        "1. **Protocol validation**: Add runtime checks for protocol compliance",
        "2. **Base class standardization**: Create common base classes for utilities",
        "3. **Type checking**: Enhance static analysis with mypy protocols",
        "4. **Decorator usage**: Apply @require_implementation for abstract methods",
    ))

    return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze protocol consistency across packages")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument("--output", type=Path, help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output raw data as JSON")

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🔍 Analyzing protocol consistency...")
    analyzer = ProtocolAnalyzer(args.src_path)
    analyses = analyzer.analyze_all_packages()

    if args.json:
        json_data = {
            package_name: {
                "implementations": list(analysis.implementations.keys()),
                "protocols": list(analysis.protocols.keys()),
                "common_methods": list(analysis.common_methods),
                "inconsistencies": analysis.inconsistencies,
            }
            for package_name, analysis in analyses.items()
        }
        if args.output:
            with (args.output).open("w") as f:
                json.dump(json_data, f, indent=2)
        else:
            print(json.dumps(json_data, indent=2))
    else:
        # Generate markdown report
        report = generate_report(analyses)

        if args.output:
            with (args.output).open("w") as f:
                f.write(report)
            print(f"📄 Report written to {args.output}")
        else:
            print(report)

    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
