#!/bin/bash
#
# Lint Script: Check for Missing SAFETY Comments on Unsafe Blocks
#
# This script ensures all `unsafe` blocks have a preceding SAFETY comment
# documenting why the unsafe code is sound. This is a critical requirement
# per the TurDB development guidelines.
#
# Exit codes:
#   0 - All unsafe blocks have SAFETY comments
#   1 - One or more unsafe blocks missing SAFETY comments
#
# Usage:
#   ./scripts/lint-unsafe-safety.sh [files...]
#   If no files specified, checks all staged .rs files or all src/*.rs files
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

resolve_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        echo "$path"
    else
        echo "$PROJECT_ROOT/$path"
    fi
}

get_files() {
    if [[ $# -gt 0 ]]; then
        for f in "$@"; do
            resolve_path "$f"
        done
    elif git rev-parse --git-dir > /dev/null 2>&1; then
        local staged
        staged=$(git diff --cached --name-only --diff-filter=ACM -- '*.rs' 2>/dev/null || true)
        if [[ -n "$staged" ]]; then
            echo "$staged" | while read -r f; do echo "$PROJECT_ROOT/$f"; done
        else
            find "$PROJECT_ROOT/src" -name "*.rs" -type f
        fi
    else
        find "$PROJECT_ROOT/src" -name "*.rs" -type f
    fi
}

check_file() {
    local file="$1"
    local errors=()
    local line_num=0
    local prev_lines=()

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++)) || true

        if [[ "$line" =~ ^[[:space:]]*unsafe[[:space:]]*\{ ]] || \
           [[ "$line" =~ ^[[:space:]]*unsafe[[:space:]]*impl ]] || \
           [[ "$line" =~ ^[[:space:]]*unsafe[[:space:]]*fn ]] || \
           [[ "$line" =~ ^[[:space:]]*unsafe[[:space:]]*trait ]]; then

            local has_safety=false
            for prev in "${prev_lines[@]}"; do
                # Match both regular comments (// SAFETY:) and doc comments (/// # Safety)
                if [[ "$prev" =~ SAFETY:|Safety:|'# Safety' ]]; then
                    has_safety=true
                    break
                fi
            done

            if [[ "$has_safety" == false ]]; then
                errors+=("$line_num")
            fi
        fi

        prev_lines+=("$line")
        if [[ ${#prev_lines[@]} -gt 10 ]]; then
            prev_lines=("${prev_lines[@]:1}")
        fi
    done < "$file"

    if [[ ${#errors[@]} -gt 0 ]]; then
        local rel_path="${file#$PROJECT_ROOT/}"
        for err_line in "${errors[@]}"; do
            echo -e "${RED}error${NC}: unsafe block missing SAFETY comment"
            echo -e "  ${YELLOW}-->${NC} $rel_path:$err_line"
            echo ""
        done
        return 1
    fi
    return 0
}

main() {
    echo -e "${GREEN}Checking for SAFETY comments on unsafe blocks...${NC}"
    echo ""

    local files
    files=$(get_files "$@")

    if [[ -z "$files" ]]; then
        echo -e "${YELLOW}No Rust files to check${NC}"
        exit 0
    fi

    local failed=0
    local checked=0

    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ ! -f "$file" ]] && continue

        ((checked++)) || true
        if ! check_file "$file"; then
            ((failed++)) || true
        fi
    done <<< "$files"

    echo ""
    if [[ $failed -gt 0 ]]; then
        echo -e "${RED}Found unsafe blocks without SAFETY comments in $failed file(s)${NC}"
        echo ""
        echo "Every unsafe block MUST have a preceding comment explaining why it's safe:"
        echo ""
        echo "    // SAFETY: <explanation of why this is sound>"
        echo "    unsafe {"
        echo "        // ..."
        echo "    }"
        echo ""
        exit 1
    else
        echo -e "${GREEN}All $checked file(s) passed unsafe SAFETY check${NC}"
        exit 0
    fi
}

main "$@"
