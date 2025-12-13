#!/bin/bash
#
# TurDB Git Hooks Setup Script
#
# This script installs the pre-commit hooks using prek.
# Run this once after cloning the repository.
#
# Usage:
#   ./scripts/setup-hooks.sh
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if ! command -v prek &> /dev/null; then
    echo -e "${YELLOW}prek not found. Installing via brew...${NC}"
    if command -v brew &> /dev/null; then
        brew install prek
    else
        echo -e "${RED}Error: brew not found. Install prek manually:${NC}"
        echo "  brew install prek"
        echo "  # or"
        echo "  cargo install --locked prek"
        exit 1
    fi
fi

echo -e "${CYAN}Setting up TurDB git hooks with prek...${NC}"
echo ""

chmod +x "$SCRIPT_DIR/lint-unsafe-safety.sh"
chmod +x "$SCRIPT_DIR/lint-inline-comments.sh"
chmod +x "$SCRIPT_DIR/lint-unwrap-panic.sh"

prek install

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "The following checks will run before each commit:"
echo "  1. cargo fmt (formatting)"
echo "  2. cargo clippy (lints)"
echo "  3. SAFETY comments on unsafe blocks"
echo "  4. Inline comments in production code"
echo "  5. Unwrap/panic usage audit"
echo ""
echo "cargo test runs on pre-push."
echo ""
echo "Run hooks manually:"
echo "  prek run --all-files"
echo ""
echo "Run individual lint scripts:"
echo "  ./scripts/lint-unsafe-safety.sh"
echo "  ./scripts/lint-inline-comments.sh"
echo "  ./scripts/lint-unwrap-panic.sh"
echo ""
