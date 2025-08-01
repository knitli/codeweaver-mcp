#!/usr/bin/env python

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Signature Consistency Validation Script.

Validates that factory-used methods have consistent signatures.
"""

import sys

from pathlib import Path


def validate_factory_method_signatures() -> bool:
    """Validate that factory-used methods have consistent signatures."""
    src_path = Path("src")
    success = True

    # Expected signatures for factory-used methods
    expected_signatures = {
        "initialize": {
            "services": "async def initialize(self) -> None:",
            "backends": "async def initialize(self) -> None:",
        },
        "shutdown": {
            "services": "async def shutdown(self) -> None:",
            "backends": "async def shutdown(self) -> None:",
            "sources": "async def shutdown(self) -> None:",
        },
        "health_check": {
            "services": "async def health_check(self) -> ServiceHealth:",
            "backends": "async def health_check(self) -> bool:",
            "providers": "async def health_check(self) -> bool:",
        },
        "get_capabilities": {
            "sources": "def get_capabilities(self) -> SourceCapabilities:",
            "services": "def get_capabilities(self) -> ServiceCapabilities:",
            "providers": "def get_capabilities(self) -> ProviderCapabilities:",
        },
    }

    print("🔍 Validating factory method signatures...")

    for method_name, package_expectations in expected_signatures.items():
        print(f"\n📋 Checking {method_name}...")

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


def check_package_signatures(package_path: Path, method_name: str, expected_sig: str) -> list[str]:
    """Check signatures in a package."""
    violations = []

    python_files = list(package_path.rglob("*.py"))
    python_files = [f for f in python_files if f.name != "__init__.py"]

    for py_file in python_files:
        try:
            with py_file.open("r", encoding="utf-8") as f:
                content = f.read()

            # Look for method definitions
            import re

            method_pattern = rf"(async\s+)?def\s+{method_name}\s*\([^)]*\)\s*(?:->\s*[^:]*)?:"
            if matches := re.findall(method_pattern, content):
                if match := re.search(method_pattern, content):
                    actual_sig = match.group(0).strip()
                    if actual_sig != expected_sig.strip():
                        violations.append(f"{py_file.name}: {actual_sig}")

        except Exception as e:
            violations.append(f"{py_file.name}: Error reading file - {e}")

    return violations


if __name__ == "__main__":
    success = validate_factory_method_signatures()

    if success:
        print("\n✅ All factory method signatures are consistent!")
        sys.exit(0)
    else:
        print("\n❌ Factory method signature inconsistencies found!")
        sys.exit(1)
