#!/bin/bash
# Compile NequIP-MP-L model for AOTInductor (one-time setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Compiling NequIP-MP-L model..."
nequip-compile nequip.net:mir-group/NequIP-MP-L:0.1 \
    "${SCRIPT_DIR}/NequIP-MP-L.nequip.pt2" \
    --mode aotinductor --device cuda --target ase
echo "Done: ${SCRIPT_DIR}/NequIP-MP-L.nequip.pt2"
