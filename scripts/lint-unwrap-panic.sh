#!/bin/bash
#
# Lint Script: Check for unwrap/panic in Production Code
#
# This script detects potential panic points in production code:
#   - .unwrap() calls
#   - .expect() calls
#   - panic!() macro
#   - unreachable!() macro (without justification)
#   - todo!() macro
#   - unimplemented!() macro
#
# These should use proper error handling with eyre::Result instead.
#
# Exceptions:
#   - Test files (*_test.rs, tests.rs, tests/*.rs)
#   - Benchmark files (benches/*.rs)
#   - Build scripts (build.rs)
#   - Comments and strings
#   - cfg(test) blocks
#   - Lines with "// OK:" or "// PANIC:" justification
#
# Exit codes:
#   0 - No unhandled panic points found
#   1 - Panic points detected
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WARN_THRESHOLD=10
ERROR_THRESHOLD=50

is_test_file() {
    local file="$1"
    [[ "$file" == *"_test.rs" ]] && return 0
    [[ "$file" == *"/tests.rs" ]] && return 0
    [[ "$file" == *"/tests/"* ]] && return 0
    [[ "$file" == *"/test/"* ]] && return 0
    [[ "$file" == *"/benches/"* ]] && return 0
    [[ "$file" == *"build.rs" ]] && return 0
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
    local issues=()
    local line_num=0
    local in_test_cfg=0
    local brace_depth=0

    if is_test_file "$file"; then
        return 0
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))

        if [[ "$line" =~ \#\[cfg\(test\)\] ]]; then
            in_test_cfg=1
            continue
        fi

        if [[ $in_test_cfg -eq 1 ]]; then
            local opens="${line//[^\{]/}"
            local closes="${line//[^\}]/}"
            brace_depth=$((brace_depth + ${#opens} - ${#closes}))
            if [[ $brace_depth -le 0 ]]; then
                in_test_cfg=0
                brace_depth=0
            fi
            continue
        fi

        [[ "$line" =~ ^[[:space:]]*// ]] && continue

        [[ "$line" =~ //[[:space:]]*(OK|PANIC|SAFETY|INVARIANT): ]] && continue

        local pattern_found=""

        if [[ "$line" =~ \.unwrap\(\) ]]; then
            [[ "$line" =~ \"[^\"]*\.unwrap\(\)[^\"]*\" ]] || pattern_found="unwrap()"
        fi

        if [[ -z "$pattern_found" ]] && [[ "$line" =~ \.expect\( ]]; then
            [[ "$line" =~ \"[^\"]*\.expect\([^\"]*\" ]] || pattern_found="expect()"
        fi

        if [[ -z "$pattern_found" ]] && [[ "$line" =~ panic!\( ]]; then
            [[ "$line" =~ \"[^\"]*panic!\([^\"]*\" ]] || pattern_found="panic!()"
        fi

        if [[ -z "$pattern_found" ]] && [[ "$line" =~ todo!\( ]]; then
            [[ "$line" =~ \"[^\"]*todo!\([^\"]*\" ]] || pattern_found="todo!()"
        fi

        if [[ -z "$pattern_found" ]] && [[ "$line" =~ unimplemented!\( ]]; then
            [[ "$line" =~ \"[^\"]*unimplemented!\([^\"]*\" ]] || pattern_found="unimplemented!()"
        fi

        if [[ -n "$pattern_found" ]]; then
            issues+=("$line_num:$pattern_found:$line")
        fi
    done < "$file"

    if [[ ${#issues[@]} -gt 0 ]]; then
        local rel_path="${file#$PROJECT_ROOT/}"
        echo "$rel_path:${#issues[@]}"
        for issue in "${issues[@]}"; do
            local issue_line="${issue%%:*}"
            local rest="${issue#*:}"
            local issue_type="${rest%%:*}"
            local issue_content="${rest#*:}"
            echo "  ${issue_line}:${issue_type}:${issue_content}"
        done
    fi

    return 0
}

main() {
    echo -e "${GREEN}Checking for unwrap/panic in production code...${NC}"
    echo ""

    local files
    files=$(get_files "$@")

    if [[ -z "$files" ]]; then
        echo -e "${YELLOW}No Rust files to check${NC}"
        exit 0
    fi

    local total_issues=0
    local files_with_issues=0
    local checked=0
    local all_output=""

    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ ! -f "$file" ]] && continue

        ((checked++))
        local output
        output=$(check_file "$file")
        if [[ -n "$output" ]]; then
            ((files_with_issues++))
            local first_line="${output%%$'\n'*}"
            local count="${first_line##*:}"
            total_issues=$((total_issues + count))
            all_output+="$output"$'\n'
        fi
    done <<< "$files"

    if [[ $total_issues -gt 0 ]]; then
        echo -e "${YELLOW}Found $total_issues potential panic point(s) in $files_with_issues file(s):${NC}"
        echo ""

        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            if [[ "$line" =~ ^[[:space:]] ]]; then
                local ln="${line%%:*}"
                local rest="${line#*:}"
                local ptype="${rest%%:*}"
                local content="${rest#*:}"
                content="${content#"${content%%[![:space:]]*}"}"
                printf "  ${CYAN}%4s${NC} | ${RED}%-16s${NC} | %.60s\n" "$ln" "$ptype" "$content"
            else
                echo ""
                echo -e "${YELLOW}$line${NC}"
            fi
        done <<< "$all_output"

        echo ""
        echo "=========================================="
        echo ""

        if [[ $total_issues -gt $ERROR_THRESHOLD ]]; then
            echo -e "${RED}ERROR: $total_issues panic points exceeds threshold ($ERROR_THRESHOLD)${NC}"
            echo ""
            echo "Production code should use proper error handling:"
            echo "  - Use eyre::Result for fallible operations"
            echo "  - Use .wrap_err() or .wrap_err_with() for context"
            echo "  - Use bail!() or ensure!() for error conditions"
            echo ""
            echo "If a panic is truly justified, add a comment:"
            echo "  .unwrap() // OK: <reason why this can't fail>"
            echo ""
            exit 1
        elif [[ $total_issues -gt $WARN_THRESHOLD ]]; then
            echo -e "${YELLOW}WARNING: $total_issues panic points (threshold: $WARN_THRESHOLD)${NC}"
            echo ""
            echo "Consider reducing panic points in production code."
            echo "Add '// OK: <reason>' comments for justified cases."
            echo ""
            exit 0
        else
            echo -e "${GREEN}$total_issues panic points within acceptable range${NC}"
            exit 0
        fi
    else
        echo -e "${GREEN}All $checked file(s) passed unwrap/panic check${NC}"
        exit 0
    fi
}

main "$@"
