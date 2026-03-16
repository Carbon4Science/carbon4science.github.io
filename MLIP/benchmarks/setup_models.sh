#!/bin/bash
#
# Download / pre-cache all MLIP model checkpoints (no GPU required).
# Run this on a login node before submitting benchmark jobs.
#
# For models requiring GPU compilation, use:
#   sbatch MLIP/benchmarks/slurm_setup_models.sh
#
# Usage:
#   ./setup_models.sh          # Download all checkpoints
#   ./setup_models.sh DPA3     # Download specific model
#   ./setup_models.sh --list   # Show status of all models
#

set -e

# Initialize conda
source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_DIR="$(dirname "$SCRIPT_DIR")"

# ==================== DPA3 ====================
# DPA-3.1-MPtrj — manual download from Figshare
setup_dpa3() {
    local ckpt="$MLIP_DIR/DPA3/dpa-3.1-mptrj.pth"
    if [ -f "$ckpt" ]; then
        echo "[OK] DPA3: checkpoint already exists"
        return
    fi
    echo "Downloading DPA-3.1-MPtrj from Figshare..."
    curl -L -o "$ckpt" "https://ndownloader.figshare.com/files/55141124"
    echo "[OK] DPA3: $ckpt"
}

# ==================== HuggingFace login ====================
# eSEN and eSEN_OAM require access to gated repo facebook/OMAT24.
# If not logged in, prompt the user; if they skip, eSEN models are skipped.
HF_LOGGED_IN=false

setup_hf_login() {
    conda activate esen || { echo "[SKIP] Cannot activate 'esen' env"; return; }

    # Check if already logged in via Python API
    if python -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null; then
        echo "[OK] HuggingFace: already logged in"
        HF_LOGGED_IN=true
    else
        echo ""
        echo "eSEN / eSEN_OAM require HuggingFace login (gated repo: facebook/OMAT24)."
        echo "You also need to request access at: https://huggingface.co/facebook/OMAT24"
        read -p "Do you want to log in now? [y/N] " reply
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            python -c "from huggingface_hub import login; login()"
            if python -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null; then
                HF_LOGGED_IN=true
            else
                echo "[SKIP] HuggingFace login failed. Skipping eSEN models."
            fi
        else
            echo "[SKIP] Skipping eSEN / eSEN_OAM (no HuggingFace token)."
        fi
    fi
    conda deactivate
}

# ==================== eSEN ====================
# eSEN-30M-MP — HuggingFace gated (facebook/OMAT24)
setup_esen() {
    if ! $HF_LOGGED_IN; then
        echo "[SKIP] eSEN: HuggingFace login required"
        return
    fi
    conda activate esen
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='facebook/OMAT24', filename='esen_30m_mptrj.pt')
print(f'[OK] eSEN: {path}')
"
    conda deactivate
}

# ==================== eSEN_OAM ====================
# eSEN-30M-OAM — HuggingFace gated (facebook/OMAT24)
setup_esen_oam() {
    if ! $HF_LOGGED_IN; then
        echo "[SKIP] eSEN_OAM: HuggingFace login required"
        return
    fi
    conda activate esen
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='facebook/OMAT24', filename='esen_30m_oam.pt')
print(f'[OK] eSEN_OAM: {path}')
"
    conda deactivate
}

# ==================== MACE ====================
# MACE-MP-0 medium — auto-downloads to ~/.cache/mace/
setup_mace() {
    conda activate mace
    python -c "
from mace.calculators import mace_mp
calc = mace_mp(model='medium', device='cpu', default_dtype='float64')
print('[OK] MACE: medium checkpoint cached')
"
    conda deactivate
}

# ==================== ORB ====================
# ORB v2 MPtrj — auto-downloads from S3
setup_orb() {
    conda activate orb
    python -c "
from orb_models.forcefield import pretrained
orbff = pretrained.orb_mptraj_only_v2(device='cpu')
print('[OK] ORB: orb_mptraj_only_v2 cached')
"
    conda deactivate
}

# ==================== Nequix ====================
# Nequix MP — auto-downloads to ~/.cache/nequix/
setup_nequix() {
    conda activate nequix
    # JAX needs nvidia pip-installed CUDA libs on LD_LIBRARY_PATH
    NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
    if [ -d "$NVIDIA_DIR" ]; then
        for d in "$NVIDIA_DIR"/*/lib; do
            [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
        done
    fi
    python -c "
from nequix.calculator import NequixCalculator
try:
    calc = NequixCalculator('nequix-mp-1', backend='jax')
except ImportError:
    calc = NequixCalculator('nequix-mp-1', backend='jax', use_kernel=False)
print('[OK] Nequix: nequix-mp-1 cached')
"
    conda deactivate
}

# ==================== PET ====================
# PET-OAM-XL — auto-downloads from HuggingFace (lab-cosmo/upet)
setup_pet() {
    conda activate pet-oam
    python -c "
from upet.calculator import UPETCalculator
calc = UPETCalculator(model='pet-oam-xl', version='1.0.0', device='cpu')
print('[OK] PET: pet-oam-xl v1.0.0 cached')
"
    conda deactivate
}

# ==================== CHGNet ====================
# CHGNet v0.3.0 — auto-downloads
setup_chgnet() {
    conda activate chgnet
    python -c "
from chgnet.model.dynamics import CHGNetCalculator
calc = CHGNetCalculator(use_device='cpu')
print('[OK] CHGNet: v0.3.0 cached')
"
    conda deactivate
}

# ==================== EquFlash ====================
# SevenNet 7net-mf-ompa — auto-downloads
setup_equflash() {
    conda activate equflash
    python -c "
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', device='cpu', enable_flash=True)
print('[OK] EquFlash: 7net-mf-ompa cached')
"
    conda deactivate
}

# ==================== SevenNet ====================
# SevenNet 7net-l3i5 — bundled in pip package, no download needed
setup_sevennet() {
    echo "[OK] SevenNet: 7net-l3i5 bundled in pip package"
}

# ==================== Status ====================
show_status() {
    echo "MLIP Model Checkpoint Status"
    echo "============================="
    echo ""

    echo "--- Manual download (setup_models.sh) ---"
    local dpa3="$MLIP_DIR/DPA3/dpa-3.1-mptrj.pth"
    echo "  DPA3         $([ -f "$dpa3" ] && echo '[OK]' || echo '[MISSING]')  dpa-3.1-mptrj.pth"
    echo ""

    echo "--- HuggingFace gated (setup_models.sh, requires huggingface-cli login) ---"
    echo "  eSEN         facebook/OMAT24 → esen_30m_mptrj.pt"
    echo "  eSEN_OAM     facebook/OMAT24 → esen_30m_oam.pt"
    echo ""

    echo "--- Auto-download / pre-cached (setup_models.sh) ---"
    echo "  MACE         ~/.cache/mace/ (medium)"
    echo "  ORB          S3 CDN (orb_mptraj_only_v2)"
    echo "  Nequix       ~/.cache/nequix/ (nequix-mp-1)"
    echo "  PET          HuggingFace (pet-oam-xl v1.0.0)"
    echo "  CHGNet       bundled/auto (v0.3.0)"
    echo "  EquFlash     ~/.sevenn/ (7net-mf-ompa)"
    echo "  SevenNet     bundled in pip (7net-l3i5)"
    echo ""

    echo "--- GPU compilation (sbatch slurm_setup_models.sh) ---"
    local nequip="$MLIP_DIR/NequIP/NequIP-MP-L.nequip.pt2"
    local nequip_oam="$MLIP_DIR/NequIP_OAM/NequIP-OAM-L.nequip.pt2"
    local allegro="$MLIP_DIR/Allegro/Allegro-OAM-L.nequip.pt2"
    echo "  NequIP       $([ -f "$nequip" ] && echo '[OK]' || echo '[MISSING] (fallback available)')  NequIP-MP-L.nequip.pt2"
    echo "  NequIP_OAM   $([ -f "$nequip_oam" ] && echo '[OK]' || echo '[MISSING] (fallback available)')  NequIP-OAM-L.nequip.pt2"
    echo "  Allegro      $([ -f "$allegro" ] && echo '[OK]' || echo '[MISSING] (fallback available)')  Allegro-OAM-L.nequip.pt2"
}

# ==================== Main ====================
if [[ $# -eq 0 ]]; then
    echo "Downloading / pre-caching all model checkpoints..."
    echo ""
    setup_dpa3;     echo ""
    setup_hf_login; echo ""
    setup_esen;     echo ""
    setup_esen_oam; echo ""
    setup_mace;     echo ""
    setup_orb;      echo ""
    setup_nequix;   echo ""
    setup_pet;      echo ""
    setup_chgnet;   echo ""
    setup_equflash; echo ""
    setup_sevennet; echo ""
    echo "Done. For GPU compilation, run:"
    echo "  sbatch MLIP/benchmarks/slurm_setup_models.sh"
    echo ""
    show_status
elif [[ "$1" == "--list" || "$1" == "--status" ]]; then
    show_status
else
    case $1 in
        DPA3|dpa3)             setup_dpa3 ;;
        eSEN|esen)             setup_esen ;;
        eSEN_OAM|esen_oam)    setup_esen_oam ;;
        MACE|mace)             setup_mace ;;
        ORB|orb)               setup_orb ;;
        Nequix|nequix)         setup_nequix ;;
        PET|pet)               setup_pet ;;
        CHGNet|chgnet)         setup_chgnet ;;
        EquFlash|equflash)     setup_equflash ;;
        SevenNet|sevennet)     setup_sevennet ;;
        *)
            echo "Unknown model: $1"
            echo "Available: DPA3, eSEN, eSEN_OAM, MACE, ORB, Nequix, PET, CHGNet, EquFlash, SevenNet"
            echo "For GPU compilation (NequIP, NequIP_OAM, Allegro): sbatch slurm_setup_models.sh"
            exit 1
            ;;
    esac
fi
