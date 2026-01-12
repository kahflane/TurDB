#!/bin/bash
# Test reminder hook - reminds Claude of TDD rules

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                            TURDB TDD ENFORCEMENT                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  BEFORE writing implementation code, answer these questions:                  ║
║                                                                               ║
║  1. "What REQUIREMENT does this test verify?"                                ║
║     (Not "what does the code do" - what SHOULD it do?)                       ║
║                                                                               ║
║  2. "What specific bug would this test catch?"                               ║
║     (Name it: "off-by-one", "null handling", etc.)                           ║
║                                                                               ║
║  3. "What's the expected value and HOW did I compute it?"                    ║
║     (Must be INDEPENDENT of the function being tested)                       ║
║                                                                               ║
║  4. "Did the test FAIL before I wrote the implementation?"                   ║
║     (If test passes before impl exists, it's worthless)                      ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FORBIDDEN:                                                                   ║
║  - assert_eq!(f(x), f(x))  ← tautology                                       ║
║  - assert!(result.is_ok()) ← what's in the Ok?                               ║
║  - let expected = encode(42); assert_eq!(encode(42), expected) ← circular   ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
