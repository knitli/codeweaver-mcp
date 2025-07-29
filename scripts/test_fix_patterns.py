#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Comprehensive test script for fix-ruff-patterns.sh
Generates test files with various ruff violations and validates fixes.
"""

import shutil
import subprocess  # noqa: S404
import sys

from pathlib import Path


class RuffPatternTester:
    """Test the fix-ruff-patterns.sh script with generated problematic files."""

    def __init__(self, test_dir: str = "test_batch"):
        """Initialize the tester with a test directory."""
        self.test_dir = Path(test_dir)
        self.script_dir = Path(__file__).parent
        self.fix_script = self.script_dir / "fix-ruff-patterns.sh"

    def setup_test_environment(self) -> None:
        """Create test directory and clean up any existing files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        print(f"✅ Created test directory: {self.test_dir}")

    def generate_test_files(self) -> dict[str, str]:
        """Generate test files with various ruff violations."""
        test_files = {
            "g004_fstrings.py": """
import logging

logger = logging.getLogger(__name__)

def test_g004_violations():
    user_id = 123
    error_msg = "connection failed"
    value = 42

    # Simple f-string cases
    logger.info(f"Processing user {user_id}")
    logger.error(f"Error occurred: {error_msg}")
    logger.debug(f"Value is {value}")

    # Complex f-string cases
    logger.warning(f"User {user_id} has {len([1,2,3])} items")
    logger.critical(f"Failed to process {user_id} with error: {error_msg}")

    # Nested expressions
    data = {"key": "value"}
    logger.info(f"Data: {data.get('key', 'default')}")

    # Multiple variables
    x, y = 10, 20
    logger.debug(f"Coordinates: ({x}, {y})")
""",
            "try401_exceptions.py": """
import logging

logger = logging.getLogger(__name__)

def test_try401_violations():
    try:
        risky_operation()
    except Exception as e:
        # Basic redundant exception cases
        logger.exception("Failed: %s", e)
        logger.exception("Error occurred - %s", e)
        logger.exception("Connection timeout (%s)", e)
        logger.exception("Database error, %s", e)
        logger.exception("Processing failed %s.", e)

        # F-string redundant exceptions
        logger.exception(f"Failed with error: {e}")
        logger.exception(f"Database connection failed - {e}")
        logger.exception(f"Timeout occurred ({e})")

    try:
        another_operation()
    except ValueError as exc:
        logger.exception("Value error: %s", exc)
        logger.exception(f"Invalid value: {exc}")

    try:
        third_operation()
    except (TypeError, AttributeError) as exception:
        logger.exception("Type/Attribute error: %s", exception)
        logger.exception(f"Error details: {exception}")

def risky_operation():
    pass

def another_operation():
    pass

def third_operation():
    pass
""",
            "try300_returns.py": """
def test_try300_violations():
    # Simple try/return case
    try:
        result = calculate_simple()
        return result
    except ValueError:
        return None

    # Try/return with as clause
    try:
        data = fetch_data()
        return data
    except ConnectionError as e:
        log_error(e)
        return {}

    # Multiple statements before return
    try:
        x = process_input()
        y = validate(x)
        z = transform(y)
        return z
    except (ValueError, TypeError):
        return None

    # Multiple except blocks
    try:
        result = complex_operation()
        return result
    except ValueError as ve:
        handle_value_error(ve)
        return "value_error"
    except TypeError as te:
        handle_type_error(te)
        return "type_error"

    # Bare except
    try:
        dangerous_operation()
        return "success"
    except:
        return "failed"

def calculate_simple():
    return 42

def fetch_data():
    return {"data": "value"}

def process_input():
    return "processed"

def validate(x):
    return x

def transform(y):
    return y.upper()

def complex_operation():
    return "complex_result"

def log_error(e):
    pass

def handle_value_error(e):
    pass

def handle_type_error(e):
    pass

def dangerous_operation():
    pass
""",
            "mixed_violations.py": """
import logging

logger = logging.getLogger(__name__)

def mixed_violations_test():
    user_id = 456

    # G004 + TRY401 combination
    try:
        process_user(user_id)
        return "success"  # TRY300
    except Exception as e:
        logger.exception(f"Failed to process user {user_id}: {e}")  # G004 + TRY401
        return None

    # More complex mixed case
    try:
        data = fetch_user_data(user_id)
        logger.info(f"Fetched data for user {user_id}")  # G004
        result = process_data(data)
        return result  # TRY300
    except ValueError as ve:
        logger.exception("Value error occurred: %s", ve)  # TRY401
        return {}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")  # G004 + TRY401 (error, not exception)
        return None

def process_user(user_id):
    pass

def fetch_user_data(user_id):
    return {"id": user_id}

def process_data(data):
    return data
""",
            "edge_cases.py": """
import logging

logger = logging.getLogger(__name__)
log = logging.getLogger("test")

def edge_cases():
    # Different logger names
    try:
        operation()
    except Exception as e:
        log.exception("Failed: %s", e)  # Different logger variable

    # Nested try blocks
    try:
        try:
            inner_operation()
            return "inner_success"  # TRY300 in nested try
        except ValueError:
            return "inner_failed"
    except Exception as e:
        logger.exception(f"Outer exception: {e}")  # G004 + TRY401

    # Complex f-strings
    user = {"name": "John", "id": 123}
    try:
        process_user_complex(user)
    except Exception as e:
        logger.exception(f"Failed for user {user['name']} (ID: {user['id']}): {e}")

    # Multiple returns in try
    try:
        if condition_a():
            return "a"
        elif condition_b():
            return "b"
        else:
            return "c"
    except Exception:
        return "error"

def operation():
    pass

def inner_operation():
    pass

def process_user_complex(user):
    pass

def condition_a():
    return False

def condition_b():
    return True
""",
        }

        # Write all test files
        for filename, content in test_files.items():
            file_path = self.test_dir / filename
            file_path.write_text(content)
            print(f"✅ Generated {filename}")

        return test_files

    def run_ruff_check(self, target: str | None = None) -> tuple[bool, str]:
        """Run ruff check on target and return (success, output)."""
        target = target or str(self.test_dir)
        try:
            result = subprocess.run(  # noqa: S603
                [shutil.which("ruff"), "check", target, "--select=TRY401,G004,TRY300", "--no-fix"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, f"Ruff check failed: {e}"

    def run_fix_script(self, targets: list[str] | None = None) -> tuple[bool, str]:
        """Run the fix-ruff-patterns.sh script."""
        # Convert relative paths to absolute paths from the script directory
        if targets is None:
            targets = [str(self.script_dir / self.test_dir)]
        else:
            targets = [
                target if Path(target).is_absolute() else str(self.script_dir / target)
                for target in targets
            ]

        try:
            # Use subprocess.PIPE to capture output directly
            result = subprocess.run(  # noqa: S603
                [str(self.fix_script), *targets, "--debug"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                cwd=self.script_dir.parent,
            )

            success = result.returncode == 0
            output = result.stdout
            if not success:
                output += f"\n[DEBUG] Fix script returned exit code: {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "Fix script timed out"
        except Exception as e:
            return False, f"Fix script failed: {e}"
        else:
            return success, output

    def analyze_changes(self, test_files: dict[str, str]) -> dict[str, dict]:
        """Analyze what changes were made to each file."""
        changes = {}

        for filename, original_content in test_files.items():
            file_path = self.test_dir / filename
            if file_path.exists():
                current_content = file_path.read_text()
                changes[filename] = {
                    "modified": current_content != original_content,
                    "original_lines": len(original_content.splitlines()),
                    "current_lines": len(current_content.splitlines()),
                    "content": current_content,
                }
            else:
                changes[filename] = {"error": "File not found after processing"}

        return changes

    def validate_fixes(self, changes: dict[str, dict]) -> dict[str, list[str]]:
        """Validate that the fixes are correct."""
        validation_results = {}

        for filename, change_info in changes.items():
            issues = []

            if "error" in change_info:
                issues.append(change_info["error"])
                continue

            content = change_info["content"]

            # Check for remaining G004 violations
            if 'f"' in content or "f'" in content:
                # Check if it's in a logging call
                lines = content.splitlines()
                issues.extend(
                    f"Line {i}: Possible remaining G004 violation: {line.strip()}"
                    for i, line in enumerate(lines, 1)
                    if ("logger." in line or "logging." in line or "log." in line)
                    and ('f"' in line or "f'" in line)
                )
            # Check for remaining TRY401 violations
            lines = content.splitlines()
            issues.extend(
                f"Line {i}: Possible remaining TRY401 violation: {line.strip()}"
                for i, line in enumerate(lines, 1)
                if ".exception(" in line
                and (", e)" in line or ", exc)" in line or ", exception)" in line)
            )
            # Check for remaining TRY300 violations (basic check)
            in_try = False
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("try:"):
                    in_try = True
                elif stripped.startswith("except"):
                    in_try = False
                elif in_try and stripped.startswith("return "):
                    issues.append(f"Line {i}: Possible remaining TRY300 violation: {line.strip()}")

            validation_results[filename] = issues

        return validation_results

    def run_comprehensive_test(self) -> bool:  # noqa: C901
        # sourcery skip: no-long-functions
        """Run the complete test suite."""
        print("🚀 Starting comprehensive test of fix-ruff-patterns.sh")
        print("=" * 60)

        # Setup
        self.setup_test_environment()

        # Generate test files
        print("\n📝 Generating test files...")
        test_files = self.generate_test_files()

        # Check initial violations
        print("\n🔍 Checking initial ruff violations...")
        initial_success, initial_output = self.run_ruff_check()
        if initial_success:
            print(
                "⚠️  No initial violations found - this might indicate a problem with test generation"
            )
        else:
            violation_count = (
                initial_output.count("TRY401")
                + initial_output.count("G004")
                + initial_output.count("TRY300")
            )
            print(f"✅ Found {violation_count} violations as expected")
            print("Sample violations:")
            for line in initial_output.splitlines()[:10]:  # Show first 10 lines
                if any(code in line for code in ["TRY401", "G004", "TRY300"]):
                    print(f"  {line}")

        # Run fix script
        print("\n🔧 Running fix-ruff-patterns.sh...")
        fix_success, fix_output = self.run_fix_script()

        print("Fix script output:")
        print("-" * 40)
        print(fix_output)
        print("-" * 40)

        if not fix_success:
            print("❌ Fix script failed!")
            return False

        # Analyze changes
        print("\n📊 Analyzing changes...")
        changes = self.analyze_changes(test_files)

        modified_files = [f for f, info in changes.items() if info.get("modified", False)]
        print(f"✅ Modified {len(modified_files)} files: {', '.join(modified_files)}")

        # Validate fixes
        print("\n✅ Validating fixes...")
        validation_results = self.validate_fixes(changes)

        total_issues = sum(len(issues) for issues in validation_results.values())
        if total_issues == 0:
            print("🎉 All fixes appear correct!")
        else:
            print(f"⚠️  Found {total_issues} potential issues:")
            for filename, issues in validation_results.items():
                if issues:
                    print(f"  {filename}:")
                    for issue in issues:
                        print(f"    - {issue}")

        # Final ruff check
        print("\n🎯 Final ruff verification...")
        final_success, final_output = self.run_ruff_check()

        if final_success:
            print("🎉 Perfect! No remaining violations!")
        else:
            remaining_violations = (
                final_output.count("TRY401")
                + final_output.count("G004")
                + final_output.count("TRY300")
            )
            print(f"⚠️  {remaining_violations} violations remain:")
            for line in final_output.splitlines():
                if any(code in line for code in ["TRY401", "G004", "TRY300"]):
                    print(f"  {line}")

        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"Test files generated: {len(test_files)}")
        print(f"Files modified: {len(modified_files)}")
        print(f"Validation issues: {total_issues}")
        print(f"Final ruff check: {'PASSED' if final_success else 'FAILED'}")

        success = final_success and total_issues == 0
        print(f"Overall result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        return success

    def cleanup(self) -> None:
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"🧹 Cleaned up {self.test_dir}")


def main() -> None:
    """Main test runner."""
    tester = RuffPatternTester()

    try:
        success = tester.run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Optionally keep test files for manual inspection
        if "--keep-files" not in sys.argv:
            tester.cleanup()
        else:
            print(f"📁 Test files preserved in {tester.test_dir}")


if __name__ == "__main__":
    main()
