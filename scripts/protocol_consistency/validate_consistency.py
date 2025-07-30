#!/usr/bin/env python
# sourcery skip: avoid-global-variables, lambdas-should-be-short
"""Validation script for consistency improvements."""

import sys

from pathlib import Path


def validate_consistency() -> bool:
    """Validate that consistency improvements have been applied."""
    src_path = Path("src")

    validation_checks = [
        {
            "name": "Provider base classes have @require_implementation",
            "check": lambda: "@require_implementation"
            in (src_path / "codeweaver/providers/base.py").read_text(),
        },
        {
            "name": "Source base classes have @require_implementation",
            "check": lambda: "@require_implementation"
            in (src_path / "codeweaver/sources/base.py").read_text(),
        },
        {
            "name": "Service base classes have @require_implementation",
            "check": lambda: "@require_implementation"
            in (src_path / "codeweaver/services/providers/base_provider.py").read_text(),
        },
    ]

    results = []
    for check in validation_checks:
        try:
            passed = check["check"]()
            results.append((check["name"], passed))
            print(f"{'✅' if passed else '❌'} {check['name']}")
        except Exception as e:
            results.append((check["name"], False))
            print(f"❌ {check['name']} - Error: {e}")

    passed_count = sum(bool(passed) for _, passed in results)
    total_count = len(results)

    print(f"\n📊 Validation Results: {passed_count}/{total_count} checks passed")

    return passed_count == total_count


if __name__ == "__main__":
    success = validate_consistency()
    sys.exit(0 if success else 1)
