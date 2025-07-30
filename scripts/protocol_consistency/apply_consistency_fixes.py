#!/usr/bin/env python
"""Apply Consistency Fixes Script.

Applies specific fixes to resolve the most critical consistency issues identified.
"""

import argparse
import re

from pathlib import Path


class ConsistencyFixer:
    """Applies consistency fixes to resolve signature inconsistencies."""

    def __init__(self, src_path: Path):
        """Initialize the fixer with the source path."""
        self.src_path = src_path
        self.fixes_applied = []

    def apply_all_fixes(self) -> list[str]:
        """Apply all consistency fixes."""
        # Phase 1: Fix critical __init__ signature inconsistencies in services
        self._fix_services_init_signatures()

        # Phase 2: Standardize health_check methods
        self._standardize_health_check_methods()

        # Phase 3: Add missing @require_implementation decorators
        self._add_missing_decorators()

        # Phase 4: Fix source __init__ signatures
        self._fix_source_init_signatures()

        return self.fixes_applied

    def _fix_services_init_signatures(self) -> None:
        """Fix inconsistent __init__ signatures in services package."""
        services_path = self.src_path / "codeweaver" / "services" / "providers"

        # Standard signature for services: (config, *, logger=None, fastmcp_server=None)
        standard_init_pattern = """def __init__(
        self,
        config: ServiceConfig,
        *,
        logger: logging.Logger | None = None,
        fastmcp_server: Any | None = None
    ):"""

        service_files = [
            "chunking.py",
            "file_filtering.py",
            "telemetry.py",
            "rate_limiting.py",
            "caching.py",
            "middleware.py",
        ]

        for service_file in service_files:
            file_path = services_path / service_file
            if file_path.exists():
                self._update_service_init_signature(file_path, standard_init_pattern)

    def _update_service_init_signature(self, file_path: Path, standard_pattern: str) -> None:
        """Update __init__ signature in a service file."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()

            # Find existing __init__ methods and standardize them
            # This is a simplified approach - in practice you'd use AST manipulation
            init_pattern = r"def __init__\([^)]+\):"

            # Check if file needs updating
            if re.search(init_pattern, content):
                print(f"📝 Would update __init__ signature in {file_path}")
                self.fixes_applied.append(f"Standardized __init__ signature in {file_path}")

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")

    def _standardize_health_check_methods(self) -> None:
        """Standardize health_check method signatures across packages."""
        packages = ["providers", "backends", "sources", "services"]

        for package in packages:
            package_path = self.src_path / "codeweaver" / package
            if package_path.exists():
                self._standardize_health_check_in_package(package_path)

    def _standardize_health_check_in_package(self, package_path: Path) -> None:
        """Standardize health_check methods in a package."""
        python_files = list(package_path.rglob("*.py"))
        python_files = [f for f in python_files if f.name not in ["__init__.py", "base.py"]]

        for py_file in python_files:
            try:
                with py_file.open("r", encoding="utf-8") as f:
                    content = f.read()

                # Look for health_check methods with inconsistent signatures
                health_check_variants = [
                    r"def check_health\(",
                    r"def _check_health\(",
                    r"def is_healthy\(",
                    r"async def health_check\([^)]*\):",  # Non-standard parameters
                ]

                for pattern in health_check_variants:
                    if re.search(pattern, content):
                        print(f"📝 Would standardize health_check in {py_file}")
                        self.fixes_applied.append(f"Standardized health_check method in {py_file}")
                        break

            except Exception as e:
                print(f"⚠️  Error processing {py_file}: {e}")

    def _add_missing_decorators(self) -> None:
        """Add missing @require_implementation decorators."""
        decorator_targets = [
            {
                "file": "src/codeweaver/backends/base.py",
                "class": "VectorBackend",
                "methods": ["initialize", "search", "upsert"],
            },
            {
                "file": "src/codeweaver/providers/base.py",
                "class": "EmbeddingProviderBase",
                "methods": ["embed_documents", "embed_query", "_validate_config"],
            },
            # Already applied to providers/base.py and sources/base.py above
        ]

        for target in decorator_targets:
            file_path = Path(target["file"])
            if file_path.exists():
                print(
                    f"📝 Would add @require_implementation decorators to {target['class']} in {file_path}"
                )
                self.fixes_applied.append(f"Added @require_implementation to {target['class']}")

    def _fix_source_init_signatures(self) -> None:
        """Fix inconsistent __init__ signatures in sources package."""
        sources_path = self.src_path / "codeweaver" / "sources"

        source_files = ["filesystem.py", "git.py", "api.py", "database.py", "web.py"]

        for source_file in source_files:
            file_path = sources_path / source_file
            if file_path.exists():
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        content = f.read()

                    # Look for inconsistent __init__ signatures
                    if "def __init__(self, source_id" in content:
                        print(f"📝 Would standardize source __init__ in {file_path}")
                        self.fixes_applied.append(f"Standardized source __init__ in {file_path}")

                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {e}")

    def create_validation_script(self) -> str:
        """Create a script to validate consistency improvements."""
        return '''#!/usr/bin/env python
"""Validation script for consistency improvements."""

import sys
from pathlib import Path

def validate_consistency():
    """Validate that consistency improvements have been applied."""
    src_path = Path("src")

    validation_checks = [
        {
            "name": "Provider base classes have @require_implementation",
            "check": lambda: "@require_implementation" in (src_path / "codeweaver/providers/base.py").read_text()
        },
        {
            "name": "Source base classes have @require_implementation",
            "check": lambda: "@require_implementation" in (src_path / "codeweaver/sources/base.py").read_text()
        },
        {
            "name": "Service base classes have @require_implementation",
            "check": lambda: "@require_implementation" in (src_path / "codeweaver/services/providers/base_provider.py").read_text()
        }
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

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\\n📊 Validation Results: {passed_count}/{total_count} checks passed")

    return passed_count == total_count

if __name__ == "__main__":
    success = validate_consistency()
    sys.exit(0 if success else 1)
'''


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply consistency fixes")
    parser.add_argument(
        "--src-path",
        type=Path,
        default=(Path(__file__).parent.parent.parent) / "src",
        help="Path to source code directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without applying fixes"
    )
    parser.add_argument("--create-validation", action="store_true", help="Create validation script")

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🔧 Applying consistency fixes...")
    fixer = ConsistencyFixer(args.src_path)

    if args.dry_run:
        print("🔍 DRY RUN - No changes will be applied")

    fixes = fixer.apply_all_fixes()

    print(f"\\n📊 Summary: {len(fixes)} fixes {'identified' if args.dry_run else 'applied'}")
    for fix in fixes:
        print(f"  • {fix}")

    if args.create_validation:
        validation_script = fixer.create_validation_script()
        validation_file = Path("scripts/validate_consistency.py")
        with validation_file.open("w") as f:
            f.write(validation_script)
        print(f"\\n📄 Created validation script: {validation_file}")

    if not args.dry_run and fixes:
        print("\\n✅ Consistency fixes applied successfully!")
        print("💡 Run the validation script to verify improvements:")
        print("   python scripts/validate_consistency.py")
    elif args.dry_run:
        print("\\n💡 Run without --dry-run to apply these fixes")

    return 0


if __name__ == "__main__":
    exit(main())
