#!/bin/bash
# fix-ruff-patterns.sh - Apply comprehensive fixes for ruff violations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_VERIFY=false
TARGETS=()

for arg in "$@"; do
    case $arg in
        --skip-verify)
            SKIP_VERIFY=true
            ;;
        *)
            TARGETS+=("$arg")
            ;;
    esac
done

# Default to current directory if no targets provided
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(".")
fi

echo -e "${BLUE}🔧 Fixing ruff patterns (TRY401, G004, TRY300)...${NC}"

# Track changes
CHANGES_MADE=0

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Use Python script for G004 (f-string conversion)
echo -e "${YELLOW}Step 1: Converting logging f-strings to % format...${NC}"
if python3 "$SCRIPT_DIR/f_string_converter.py" "${TARGETS[@]}" 2>/dev/null; then
    echo -e "${GREEN}✅ F-string conversion complete${NC}"
else
    echo -e "${YELLOW}⚠️  F-string converter not available, using ast-grep fallback${NC}"
fi

# Step 2: Use ast-grep for TRY401 patterns
echo -e "${YELLOW}Step 2: Removing redundant exception references...${NC}"

# TRY401: Fix redundant exception logging (% format)
if result=$(ast-grep scan -r "$SCRIPT_DIR/rules/fix-exception-logging-with-var.yml" --update-all "${TARGETS[@]}" 2>&1) && [[ $result =~ "Applied" ]]; then
    ((CHANGES_MADE++))
    echo "✅ Fixed redundant exception variables"
fi

# TRY401: Fix redundant exception logging with intelligent punctuation cleanup
echo -e "${YELLOW}   → Cleaning up trailing punctuation in exception messages...${NC}"
if python3 "$SCRIPT_DIR/punctuation_cleaner.py" "${TARGETS[@]}" 2>/dev/null | grep -q "Cleaned punctuation"; then
    ((CHANGES_MADE++))
    echo "✅ Fixed redundant exception with punctuation cleanup"
fi

# TRY401: Fix redundant exception logging (f-strings) - different variable names (fallback)
for rule in fix-exception-logging-fstring-simple fix-exception-logging-fstring-exc fix-exception-logging-fstring-exception; do
    if result=$(ast-grep scan -r "$SCRIPT_DIR/rules/$rule.yml" --update-all "${TARGETS[@]}" 2>&1) && [[ $result =~ "Applied" ]]; then
        ((CHANGES_MADE++))
        echo "✅ Fixed redundant exception in f-strings"
    fi
done

# Step 3: Use ast-grep for TRY300 patterns
echo -e "${YELLOW}Step 3: Moving return statements from try to else blocks...${NC}"

# TRY300: Fix try/return patterns - multiple variations
for rule in fix-try-return-simple fix-try-return-as fix-try-return-multiple fix-try-return-multiple-as fix-try-return-bare-except; do
    if result=$(ast-grep scan -r "$SCRIPT_DIR/rules/$rule.yml" --update-all "${TARGETS[@]}" 2>&1) && [[ $result =~ "Applied" ]]; then
        ((CHANGES_MADE++))
        echo "✅ Fixed try/return pattern"
    fi
done

# Summary
echo
if [ $CHANGES_MADE -gt 0 ]; then
    echo -e "${GREEN}🎉 Applied $CHANGES_MADE fix(es)! Run 'git diff' to review changes.${NC}"
else
    echo -e "${GREEN}✨ No fixes needed - your code already follows best practices!${NC}"
fi

# Optional: Run ruff to verify fixes
if [ "$SKIP_VERIFY" = true ]; then
    echo -e "${BLUE}🚀 Skipping ruff verification (--skip-verify flag used)${NC}"
elif command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Verifying fixes with ruff...${NC}"
    
    # Use timeout to prevent hangs and capture output
    if timeout 30s ruff check "${TARGETS[@]}" --select=TRY401,G004,TRY300 --quiet 2>/dev/null; then
        echo -e "${GREEN}🎯 Perfect! No remaining TRY401/G004/TRY300 violations!${NC}"
    else
        echo -e "${YELLOW}⚠️  Some violations may need manual review${NC}"
        echo "Run: ruff check ${TARGETS[*]} --select=TRY401,G004,TRY300 for details"
    fi
else
    echo -e "${BLUE}💡 Install ruff to verify fixes: pip install ruff${NC}"
fi
