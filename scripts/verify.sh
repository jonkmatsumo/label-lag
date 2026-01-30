#!/usr/bin/env bash
#
# verify.sh - Per-commit regression testing script
#
# Usage:
#   ./scripts/verify.sh          # Fast check (lint + unit tests)
#   ./scripts/verify.sh --full   # Full check (all tests including BFF)
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track failures
FAILED_STEPS=()

# Parse arguments
FULL_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full]"
            exit 1
            ;;
    esac
done

# Helper functions
section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    FAILED_STEPS+=("$1")
}

run_step() {
    local name="$1"
    shift
    local cmd="$@"

    echo -e "${YELLOW}Running: $cmd${NC}"
    if eval "$cmd"; then
        success "$name"
        return 0
    else
        fail "$name"
        return 1
    fi
}

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Report mode
if [ "$FULL_MODE" = true ]; then
    section "FULL VERIFICATION"
else
    section "FAST VERIFICATION"
fi

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""

# ============================================
# Python Linting
# ============================================
section "Python Linting (ruff)"

run_step "Ruff check" "uv run ruff check src tests" || true
run_step "Ruff format check" "uv run ruff format --check src tests" || true

# ============================================
# Python Tests
# ============================================
section "Python Tests (pytest)"

# Fast mode: exclude integration tests (they require running services)
# Full mode: include integration tests (will skip if services unavailable)
if [ "$FULL_MODE" = true ]; then
    run_step "Python tests (all)" "uv run pytest --cov=src/synthetic_pipeline --cov-report=term-missing -q" || true
else
    run_step "Python unit tests" "uv run pytest --ignore=tests/integration --cov=src/synthetic_pipeline --cov-report=term-missing -q" || true
fi

# ============================================
# BFF Checks (TypeScript)
# ============================================
if [ "$FULL_MODE" = true ]; then
    section "BFF Linting (eslint)"

    if [ -d "bff" ] && [ -f "bff/package.json" ]; then
        # Check if node_modules exists
        if [ -d "bff/node_modules" ]; then
            run_step "BFF lint" "cd bff && npm run lint" || true
        else
            echo -e "${YELLOW}Skipping BFF lint - node_modules not installed${NC}"
            echo "Run 'cd bff && npm install' to enable BFF checks"
        fi
    else
        echo -e "${YELLOW}Skipping BFF lint - bff directory not found${NC}"
    fi

    section "BFF Tests (vitest)"

    if [ -d "bff" ] && [ -f "bff/package.json" ]; then
        if [ -d "bff/node_modules" ]; then
            run_step "BFF unit tests" "cd bff && npm test" || true
        else
            echo -e "${YELLOW}Skipping BFF tests - node_modules not installed${NC}"
        fi
    fi

    section "Web Linting (eslint)"

    if [ -d "web" ] && [ -f "web/package.json" ]; then
        if [ -d "web/node_modules" ]; then
            run_step "Web lint" "cd web && npm run lint" || true
        else
            echo -e "${YELLOW}Skipping Web lint - node_modules not installed${NC}"
            echo "Run 'cd web && npm install' to enable Web checks"
        fi
    fi
fi

# ============================================
# Summary
# ============================================
section "SUMMARY"

if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Failed checks (${#FAILED_STEPS[@]}):${NC}"
    for step in "${FAILED_STEPS[@]}"; do
        echo -e "${RED}  - $step${NC}"
    done
    exit 1
fi
