#!/bin/bash
# Pre-write hook - reminds Claude of rules before creating new files

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TURDB PRE-WRITE CHECKLIST                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ □ Is this file necessary? Prefer editing existing files                      ║
║ □ Will this file stay under 800 lines?                                       ║
║ □ Does it have 80-100 lines of module documentation at top?                  ║
║ □ Using pub(crate) not pub where possible?                                   ║
║ □ No inline comments in the code?                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Required file structure:                                                      ║
║   1. Module doc (//! ...) - 80-100 lines                                     ║
║   2. Imports                                                                  ║
║   3. Types/Structs                                                           ║
║   4. Implementations                                                          ║
║   5. Tests at bottom (#[cfg(test)])                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
