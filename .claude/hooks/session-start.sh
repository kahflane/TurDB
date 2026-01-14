#!/bin/bash
# Session start hook - displays rules at beginning of each session

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TURDB RULES REMINDER                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  CORE PRINCIPLES (Non-negotiable):                                           ║
║  • Zero-copy: &[u8] not Vec<u8>                                              ║
║  • Zero-alloc: No heap allocation in CRUD                                    ║
║  • TDD: Test fails FIRST, then implement                                     ║
║  • eyre only: No custom error types                                          ║
║                                                                               ║
║  BEFORE ANY TASK, check relevant rules:                                      ║
║  • Tests → .claude/rules/TESTING.md                                          ║
║  • Memory/Pages → .claude/rules/MEMORY.md                                    ║
║  • Errors → .claude/rules/ERRORS.md                                          ║
║  • Encoding → .claude/rules/ENCODING.md                                      ║
║  • Style → .claude/rules/STYLE.md                                            ║
║  • File formats → .claude/rules/FILE_FORMATS.md                              ║
║                                                                               ║
║  FORBIDDEN PATTERNS:                                                          ║
║  ✗ .to_vec() on page data                                                    ║
║  ✗ .clone() on page data                                                     ║
║  ✗ Custom error enums                                                        ║
║  ✗ Inline comments                                                           ║
║  ✗ Files > 1800 lines                                                         ║
║  ✗ assert!(result.is_ok()) without checking contents                         ║
║  ✗ Tests that use function output as expected value                          ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
