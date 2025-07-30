#!/usr/bin/env python
"""Focused Consistency Fixes.

Addresses the real issues identified in the corrected analysis.
"""

import argparse
import re

from pathlib import Path


class FocusedConsistencyFixer:
    """Apply focused fixes for the real consistency issues."""

    def __init__(self, src_path: Path):
        """Initialize the fixer with the source path."""
        self.src_path = src_path
        self.test_path = src_path.parent / "tests"
        self.fixes_applied = []

    def apply_focused_fixes(self, *, dry_run: bool = True) -> list[str]:
        """Apply the focused consistency fixes."""
        print("🎯 Applying focused consistency fixes...")

        # Phase 1: Factory method signature standardization
        self._standardize_factory_method_signatures(dry_run=dry_run)

        # Phase 2: Create validation for consistent signatures
        self._create_signature_validation(dry_run=dry_run)

        # Phase 3: Identify test impacts
        self._analyze_test_impacts()

        return self.fixes_applied

    def _standardize_factory_method_signatures(self, *, dry_run: bool = True) -> None:
        """Standardize signatures for factory-used methods."""
        print("\n📝 Standardizing factory method signatures...")

        # Target methods and their standard signatures
        standard_signatures = {
            "initialize": {
                "signature": "async def initialize(self) -> None:",
                "packages": ["services", "backends"],
                "description": "Async initialization with no parameters",
            },
            "shutdown": {
                "signature": "async def shutdown(self) -> None:",
                "packages": ["services", "backends", "sources"],
                "description": "Async shutdown with no parameters",
            },
            "health_check": {
                "signature_services": "async def health_check(self) -> ServiceHealth:",
                "signature_others": "async def health_check(self) -> bool:",
                "packages": ["services", "backends", "providers"],
                "description": "Async health check with appropriate return type",
            },
            "get_capabilities": {
                "signature_sources": "def get_capabilities(self) -> SourceCapabilities:",
                "signature_services": "def get_capabilities(self) -> ServiceCapabilities:",
                "signature_providers": "def get_capabilities(self) -> ProviderCapabilities:",
                "packages": ["sources", "services", "providers"],
                "description": "Sync capabilities with package-specific return type",
            },
        }

        for method_name, config in standard_signatures.items():
            for package in config["packages"]:
                package_path = self.src_path / "codeweaver" / package
                if package_path.exists():
                    files_needing_fix = self._find_files_needing_signature_fix(
                        package_path, method_name, config
                    )

                    for file_path, current_signature, target_signature in files_needing_fix:
                        if dry_run:
                            print(f"  📋 Would fix {method_name} in {file_path}")
                            print(f"      Current: {current_signature}")
                            print(f"      Target:  {target_signature}")
                        else:
                            self._apply_signature_fix(
                                file_path, current_signature, target_signature
                            )

                        self.fixes_applied.append(
                            f"Standardized {method_name} signature in {file_path.name}"
                        )

    def _find_files_needing_signature_fix(
        self, package_path: Path, method_name: str, config: dict
    ) -> list[tuple]:
        """Find files that need signature fixes."""
        files_needing_fix = []
        python_files = list(package_path.rglob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]

        for py_file in python_files:
            try:
                with py_file.open("r", encoding="utf-8") as f:
                    content = f.read()

                # Find method definitions
                method_pattern = rf"(async\s+)?def\s+{method_name}\s*\([^)]*\)\s*->\s*[^:]*:"
                if re.findall(method_pattern, content):
                    # Determine target signature based on package
                    if "signature_services" in config and "services" in str(package_path):
                        target_sig = config["signature_services"]
                    elif "signature_sources" in config and "sources" in str(package_path):
                        target_sig = config["signature_sources"]
                    elif "signature_providers" in config and "providers" in str(package_path):
                        target_sig = config["signature_providers"]
                    elif "signature_others" in config:
                        target_sig = config["signature_others"]
                    else:
                        target_sig = config["signature"]

                    if current_match := re.search(method_pattern, content):
                        current_sig = current_match[0]

                        # Check if it needs fixing
                        if current_sig.strip() != target_sig.strip():
                            files_needing_fix.append((py_file, current_sig, target_sig))

            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

        return files_needing_fix

    def _apply_signature_fix(
        self, file_path: Path, current_signature: str, target_signature: str
    ) -> None:
        """Apply a signature fix to a file."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()

            # Replace the signature
            updated_content = content.replace(current_signature, target_signature)

            with file_path.open("w", encoding="utf-8") as f:
                f.write(updated_content)

            print(f"  ✅ Fixed signature in {file_path.name}")

        except Exception as e:
            print(f"  ❌ Error fixing {file_path}: {e}")

    def _create_signature_validation(self, *, dry_run: bool = True) -> None:
        """Create validation for consistent signatures."""
        print("\n🔍 Creating signature validation...")

        validation_file = Path("scripts/validate_factory_signatures.py")

        if dry_run:
            print(f"  📋 Would create validation script: {validation_file}")
        else:
            validation_script = '''#!/usr/bin/env python
"""Signature Consistency Validation Script

Validates that factory-used methods have consistent signatures.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set


def validate_factory_method_signatures() -> bool:
    """Validate that factory-used methods have consistent signatures."""

    src_path = Path("src")
    success = True

    # Expected signatures for factory-used methods
    expected_signatures = {
        "initialize": {
            "services": "async def initialize(self) -> None:",
            "backends": "async def initialize(self) -> None:"
        },
        "shutdown": {
            "services": "async def shutdown(self) -> None:",
            "backends": "async def shutdown(self) -> None:",
            "sources": "async def shutdown(self) -> None:"
        },
        "health_check": {
            "services": "async def health_check(self) -> ServiceHealth:",
            "backends": "async def health_check(self) -> bool:",
            "providers": "async def health_check(self) -> bool:"
        },
        "get_capabilities": {
            "sources": "def get_capabilities(self) -> SourceCapabilities:",
            "services": "def get_capabilities(self) -> ServiceCapabilities:",
            "providers": "def get_capabilities(self) -> ProviderCapabilities:"
        }
    }

    print("🔍 Validating factory method signatures...")

    for method_name, package_expectations in expected_signatures.items():
        print(f"\\n📋 Checking {method_name}...")

        for package, expected_sig in package_expectations.items():
            package_path = src_path / "codeweaver" / package

            if package_path.exists():
                violations = check_package_signatures(package_path, method_name, expected_sig)

                if violations:
                    success = False
                    print(f"  ❌ {package} package violations:")
                    for violation in violations:
                        print(f"    • {violation}")
                else:
                    print(f"  ✅ {package} package: consistent")

    return success


def check_package_signatures(package_path: Path, method_name: str, expected_sig: str) -> List[str]:
    """Check signatures in a package."""
    violations = []

    python_files = list(package_path.rglob("*.py"))
    python_files = [f for f in python_files if f.name != "__init__.py"]

    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for method definitions
            import re
            method_pattern = rf'(async\\s+)?def\\s+{method_name}\\s*\\([^)]*\\)\\s*(?:->\\s*[^:]*)?:'
            matches = re.findall(method_pattern, content)

            if matches:
                # Extract the full signature
                match = re.search(method_pattern, content)
                if match:
                    actual_sig = match.group(0).strip()
                    if actual_sig != expected_sig.strip():
                        violations.append(f"{py_file.name}: {actual_sig}")

        except Exception as e:
            violations.append(f"{py_file.name}: Error reading file - {e}")

    return violations


if __name__ == "__main__":
    success = validate_factory_method_signatures()

    if success:
        print("\\n✅ All factory method signatures are consistent!")
        sys.exit(0)
    else:
        print("\\n❌ Factory method signature inconsistencies found!")
        sys.exit(1)
'''

            with validation_file.open("w") as f:
                f.write(validation_script)
            validation_file.chmod(0o755)
            print(f"  ✅ Created validation script: {validation_file}")

        self.fixes_applied.append("Created factory signature validation script")

    def _analyze_test_impacts(self) -> None:
        """Analyze which tests would be impacted by signature changes."""
        print("\\n⚠️  Analyzing test impacts...")

        if not self.test_path.exists():
            print("  📋 No test directory found")
            return

        # Methods that would change and their potential test impacts
        method_changes = {
            "initialize": {
                "change": "Made async across all packages",
                "test_files": [
                    "tests/integration/test_service_integration.py",
                    "tests/integration/test_fastmcp_middleware_integration.py",
                    "tests/unit/test_enhanced_config.py",
                ],
            },
            "health_check": {
                "change": "Standardized return types (ServiceHealth vs bool)",
                "test_files": [
                    "tests/unit/test_telemetry_service.py",
                    "tests/validation/test_services_integration.py",
                ],
            },
        }

        for method_name, impact_info in method_changes.items():
            print(f"  📋 {method_name} changes: {impact_info['change']}")

            for test_file in impact_info["test_files"]:
                test_path = Path(test_file)
                if test_path.exists():
                    print(f"    ⚠️  Impacted: {test_file}")
                else:
                    print(f"    📝 Would impact: {test_file}")

            self.fixes_applied.append(f"Analyzed test impact for {method_name} changes")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply focused consistency fixes")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be changed (default: true)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually apply the fixes")

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    # Determine if this is a dry run
    dry_run = not args.apply

    if dry_run:
        print("🔍 DRY RUN - Showing what would be changed")
        print("💡 Use --apply to actually make changes")
    else:
        print("🔧 APPLYING CHANGES")

    fixer = FocusedConsistencyFixer(args.src_path)
    fixes = fixer.apply_focused_fixes(dry_run=dry_run)

    print(f"\\n📊 Summary: {len(fixes)} fixes {'identified' if dry_run else 'applied'}")
    for fix in fixes:
        print(f"  • {fix}")

    if dry_run:
        print("\\n💡 Next steps:")
        print("  1. Review the proposed changes above")
        print("  2. Run with --apply to make changes")
        print("  3. Update affected tests")
        print("  4. Run validation: python scripts/validate_factory_signatures.py")
    else:
        print("\\n✅ Focused consistency fixes applied!")
        print("💡 Next steps:")
        print("  1. Update affected test files")
        print("  2. Run validation: python scripts/validate_factory_signatures.py")
        print("  3. Run full test suite to verify changes")

    return 0


if __name__ == "__main__":
    exit(main())
