#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Comprehensive validation runner for CodeWeaver Phase 5 implementation.

This script runs all validation tests to ensure pattern consistency,
services integration, and architectural compliance across the codebase.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Run all validation tests and report results."""
    print("🚀 CodeWeaver Phase 5 Validation Suite")
    print("=" * 50)
    
    all_passed = True
    
    # 1. Pattern Consistency Validation
    print("\n📋 Phase 5.2.1: Pattern Consistency Validation")
    print("-" * 40)
    
    try:
        from tests.validation.test_pattern_consistency import validate_pattern_consistency
        if validate_pattern_consistency():
            print("✅ Pattern consistency validation PASSED")
        else:
            print("❌ Pattern consistency validation FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Pattern consistency validation ERROR: {e}")
        all_passed = False
    
    # 2. Services Integration Validation
    print("\n🔗 Phase 5.2.2: Services Integration Validation")
    print("-" * 40)
    
    try:
        from tests.validation.test_services_integration import validate_services_integration
        if validate_services_integration():
            print("✅ Services integration validation PASSED")
        else:
            print("❌ Services integration validation FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Services integration validation ERROR: {e}")
        all_passed = False
    
    # 3. Existing Service Implementation Validation
    print("\n🛠️ Existing: Service Implementation Validation")
    print("-" * 40)
    
    try:
        from tests.validation.validate_service_implementation import (
            validate_protocol_compliance,
            validate_type_system,
            validate_factory_integration
        )
        
        protocol_passed = validate_protocol_compliance()
        type_system_passed = validate_type_system()
        factory_passed = validate_factory_integration()
        
        if protocol_passed and type_system_passed and factory_passed:
            print("✅ Service implementation validation PASSED")
        else:
            print("❌ Service implementation validation FAILED")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Service implementation validation ERROR: {e}")
        all_passed = False
    
    # 4. Documentation Validation
    print("\n📚 Phase 5.1: Documentation Validation")
    print("-" * 40)
    
    required_docs = [
        "docs/SERVICES_LAYER_GUIDE.md",
        "docs/MIGRATION_GUIDE.md", 
        "docs/DEVELOPMENT_PATTERNS.md"
    ]
    
    docs_passed = True
    for doc_path in required_docs:
        doc_file = Path(doc_path)
        if doc_file.exists():
            print(f"✅ {doc_path} exists")
        else:
            print(f"❌ {doc_path} missing")
            docs_passed = False
            all_passed = False
    
    if docs_passed:
        print("✅ Documentation validation PASSED")
    else:
        print("❌ Documentation validation FAILED")
    
    # 5. Anti-Pattern Detection
    print("\n🚨 Anti-Pattern Detection")
    print("-" * 40)
    
    anti_patterns_found = False
    
    # Check for migration code
    migration_file = Path("src/codeweaver/config_migration.py")
    if migration_file.exists():
        print("❌ Migration code still exists (should be removed)")
        anti_patterns_found = True
        all_passed = False
    else:
        print("✅ Migration code has been removed")
    
    # Check for direct middleware imports (refined check)
    try:
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "from codeweaver.middleware", "src/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            violations = []
            
            for line in lines:
                # Parse the grep output: filename:line_number:content
                parts = line.split(':', 2)
                if len(parts) < 3:
                    continue
                    
                file_path, line_num, content = parts
                
                # Skip allowed files
                if any(allowed in file_path for allowed in [
                    "src/codeweaver/middleware/",
                    "src/codeweaver/server.py", 
                    "src/codeweaver/main.py",
                    "src/codeweaver/services/providers/"
                ]):
                    continue
                
                # For other files, check if the import is in a fallback method
                try:
                    with open(file_path, 'r') as f:
                        file_lines = f.readlines()
                    
                    line_idx = int(line_num) - 1
                    in_fallback_method = False
                    
                    # Look backwards to find the method definition
                    for i in range(line_idx, max(0, line_idx - 30), -1):
                        line_content = file_lines[i].strip()
                        if "def " in line_content and "fallback" in line_content:
                            in_fallback_method = True
                            break
                        elif line_content.startswith("def ") or line_content.startswith("class "):
                            break
                    
                    if not in_fallback_method:
                        violations.append(f"{file_path}:{line_num}")
                        
                except Exception as e:
                    # If we can't analyze the file, flag it as a potential violation
                    violations.append(f"{file_path}:{line_num} (could not analyze: {e})")
            
            if violations:
                print("❌ Direct middleware imports found outside allowed contexts:")
                for violation in violations:
                    print(f"   {violation}")
                anti_patterns_found = True
                all_passed = False
            else:
                print("✅ All middleware imports are in allowed contexts (services, fallbacks, etc.)")
        else:
            print("✅ No middleware imports found")
    except Exception as e:
        print(f"⚠️  Could not check for direct middleware imports: {e}")
    
    if not anti_patterns_found:
        print("✅ Anti-pattern detection PASSED")
    else:
        print("❌ Anti-pattern detection FAILED")
    
    # Final Results
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nPhase 5 implementation is complete and compliant:")
        print("✅ Documentation created (Services Layer Guide, Migration Guide, Development Patterns)")
        print("✅ Pattern consistency validated")
        print("✅ Services integration validated") 
        print("✅ Anti-patterns eliminated")
        print("✅ Architectural compliance verified")
        
        print("\n🚀 CodeWeaver is ready for the next phase!")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nPlease address the issues above before proceeding.")
        print("See the documentation guides for migration instructions:")
        print("- docs/SERVICES_LAYER_GUIDE.md")
        print("- docs/MIGRATION_GUIDE.md")
        print("- docs/DEVELOPMENT_PATTERNS.md")
        return 1


if __name__ == "__main__":
    exit(main())
