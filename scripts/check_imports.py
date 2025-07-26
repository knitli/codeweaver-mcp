#!/usr/bin/env python3
"""Simple import checker to identify missing dependencies."""

import sys


MODULES_TO_CHECK = (
    "codeweaver._types",
    "codeweaver.factories",
    "codeweaver.providers",
    "codeweaver.backends",
    "codeweaver.sources",
    "codeweaver.services",
    "codeweaver.middleware",
    "codeweaver.testing",
)

FAILED_IMPORTS = []
SUCCESSFUL_IMPORTS = []

for module in MODULES_TO_CHECK:
    try:
        __import__(module)
        SUCCESSFUL_IMPORTS.append(module)
        print(f"✅ {module}")
    except Exception as e:
        FAILED_IMPORTS.append((module, str(e)))
        print(f"❌ {module}: {e}")

print("\n📊 Summary:")
print(f"✅ Successful: {len(SUCCESSFUL_IMPORTS)}")
print(f"❌ Failed: {len(FAILED_IMPORTS)}")

if FAILED_IMPORTS:
    print("\n🔍 Failed imports:")
    for module, error in FAILED_IMPORTS:
        print(f"  - {module}: {error}")
    sys.exit(1)
else:
    print("\n🎉 All modules imported successfully!")
