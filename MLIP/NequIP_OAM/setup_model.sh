#!/bin/bash
# Compile NequIP-OAM-L model for AOTInductor (one-time setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Compiling NequIP-OAM-L model..."
nequip-compile nequip.net:mir-group/NequIP-OAM-L:0.1 \
    "${SCRIPT_DIR}/NequIP-OAM-L.nequip.pt2" \
    --mode aotinductor --device cuda --target ase
echo "Done: ${SCRIPT_DIR}/NequIP-OAM-L.nequip.pt2"
