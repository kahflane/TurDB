#!/bin/bash
#
# Lint Script: Check for Inline Comments in Non-Test Code
#
# This script detects inline comments (// comments not at line start) in
# production code. Per TurDB guidelines, code should be self-documenting
# with block documentation at file tops, not inline comments.
#
# Exceptions:
#   - Test files (*_test.rs, tests.rs, tests/*.rs)
#   - Doc comments (///, //!)
#   - SAFETY comments (required for unsafe)
#   - TODO/FIXME/HACK/XXX comments (legitimate markers)
#   - URL comments (contain http:// or https://)
#   - License/copyright headers
#   - Comments that ARE at line start (full-line comments are allowed in moderation)
#
# Exit codes:
#   0 - No inline comments found
#   1 - Inline comments detected
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

is_test_file() {
    local file="$1"
    [[ "$file" == *"_test.rs" ]] && return 0
    [[ "$file" == *"/tests.rs" ]] && return 0
    [[ "$file" == *"/tests/"* ]] && return 0
    [[ "$file" == *"/test/"* ]] && return 0
    [[ "$file" == *"/benches/"* ]] && return 0
    return 1
}

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
    local in_string=false
    local in_block_comment=false

    if is_test_file "$file"; then
        return 0
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))

        [[ -z "$line" ]] && continue

        if [[ "$in_block_comment" == true ]]; then
            if [[ "$line" == *"*/"* ]]; then
                in_block_comment=false
            fi
            continue
        fi

        if [[ "$line" == *"/*"* ]] && [[ "$line" != *"*/"* ]]; then
            in_block_comment=true
            continue
        fi

        local stripped="${line#"${line%%[![:space:]]*}"}"

        [[ "$stripped" == "//"* ]] && continue

        if [[ "$line" =~ [^[:space:]].*// ]]; then
            [[ "$line" =~ \"[^\"]*//[^\"]*\" ]] && continue
            [[ "$line" =~ \'[^\']*//[^\']*\' ]] && continue

            [[ "$line" =~ https?:// ]] && continue

            [[ "$line" =~ //[[:space:]]*(SAFETY|Safety): ]] && continue
            [[ "$line" =~ //[[:space:]]*(TODO|FIXME|HACK|XXX|NOTE|WARN): ]] && continue

            [[ "$line" =~ ///|//! ]] && continue

            errors+=("$line_num:$line")
        fi
    done < "$file"

    if [[ ${#errors[@]} -gt 0 ]]; then
        local rel_path="${file#$PROJECT_ROOT/}"
        echo -e "${RED}error${NC}: inline comments in production code"
        echo -e "  ${YELLOW}-->${NC} $rel_path"
        echo ""
        for err in "${errors[@]}"; do
            local err_line="${err%%:*}"
            local err_content="${err#*:}"
            echo -e "  ${CYAN}$err_line${NC} | $err_content"
        done
        echo ""
        return 1
    fi
    return 0
}

main() {
    echo -e "${GREEN}Checking for inline comments in production code...${NC}"
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

        ((checked++))
        if ! check_file "$file"; then
            ((failed++))
        fi
    done <<< "$files"

    echo ""
    if [[ $failed -gt 0 ]]; then
        echo -e "${RED}Found inline comments in $failed file(s)${NC}"
        echo ""
        echo "Per TurDB guidelines, avoid inline comments. Instead:"
        echo "  - Use block documentation at file top (80-100 lines)"
        echo "  - Use doc comments (///) for public API"
        echo "  - Make code self-documenting with clear names"
        echo ""
        echo "Allowed exceptions:"
        echo "  - SAFETY: comments (required for unsafe)"
        echo "  - TODO/FIXME/HACK/XXX markers"
        echo "  - Full-line comments (// at line start)"
        echo ""
        exit 1
    else
        echo -e "${GREEN}All $checked file(s) passed inline comment check${NC}"
        exit 0
    fi
}

main "$@"
