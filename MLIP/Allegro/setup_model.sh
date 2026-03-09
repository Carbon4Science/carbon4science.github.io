#!/bin/bash
# Compile Allegro-OAM-L model for AOTInductor (one-time setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Compiling Allegro-OAM-L model..."
nequip-compile nequip.net:mir-group/Allegro-OAM-L:0.1 \
    "${SCRIPT_DIR}/Allegro-OAM-L.nequip.pt2" \
    --mode aotinductor --device cuda --target ase
echo "Done: ${SCRIPT_DIR}/Allegro-OAM-L.nequip.pt2"
