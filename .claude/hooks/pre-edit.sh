#!/bin/bash
# Pre-edit hook - reminds Claude of rules before editing files

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TURDB PRE-EDIT CHECKLIST                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ □ Zero-copy? Returning &[u8], not Vec<u8>                                    ║
║ □ Zero-alloc? No heap allocation in CRUD paths                               ║
║ □ eyre errors? Using bail!/ensure! with rich context                         ║
║ □ No inline comments? Documentation at file top only                         ║
║ □ TDD? Test written and FAILED before this implementation                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ FORBIDDEN: .to_vec(), .clone() on page data, custom error enums              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
