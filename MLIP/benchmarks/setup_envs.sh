#!/bin/bash
#
# Setup all conda environments for MLIP benchmarks
#
# All package versions are pinned to tested combinations.
# Last verified: 2026-03-16
#
# Usage:
#   ./setup_envs.sh          # Setup all environments
#   ./setup_envs.sh CHGNet   # Setup specific environment
#

set -e

# Initialize conda so 'conda activate' works in this script
source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh

# Helper: check if a conda env already exists
env_exists() {
    conda env list | grep -qw "^$1 "
}

setup_esen() {
    if env_exists esen; then
        echo "Skipping eSEN: 'esen' environment already exists."
        return
    fi
    echo "Setting up eSEN environment..."
    conda create -n esen python=3.11 -y
    conda activate esen
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch_geometric==2.7.0
    pip install torch_scatter==2.1.2+pt24cu121 torch_sparse==0.6.18+pt24cu121 \
        -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    pip install fairchem-core==1.10.0 "scipy==1.14.1" ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done. Run 'huggingface-cli login' to access gated checkpoint."
}

setup_nequip() {
    if env_exists nequip; then
        echo "Skipping NequIP: 'nequip' environment already exists."
        return
    fi
    echo "Setting up NequIP environment..."
    conda create -n nequip python=3.11 -y
    conda activate nequip
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install nequip==0.17.0 nequip-allegro==0.8.2 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done. Run 'MLIP/NequIP/setup_model.sh' to compile the model."
}

setup_nequix() {
    if env_exists nequix; then
        echo "Skipping Nequix: 'nequix' environment already exists."
        return
    fi
    echo "Setting up Nequix environment..."
    conda create -n nequix python=3.10 -y
    conda activate nequix
    pip install nequix==0.4.3 ase==3.27.0 codecarbon==3.2.3
    pip install nvidia-cusparse-cu12==12.5.10.65 nvidia-cusolver-cu12==11.7.5.82 \
        nvidia-cublas-cu12==12.9.1.4 nvidia-cufft-cu12==11.4.1.4 nvidia-curand-cu12==10.3.10.19
    conda deactivate
    echo "Done."
}

setup_deepmd() {
    if env_exists deepmd; then
        echo "Skipping DPA3: 'deepmd' environment already exists."
        return
    fi
    echo "Setting up DPA3 (deepmd) environment..."
    conda create -n deepmd python=3.10 -y
    conda activate deepmd
    pip install "deepmd-kit[torch]==3.1.2" ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done. Run 'MLIP/DPA3/download_model.sh' to download checkpoint."
}

setup_sevennet() {
    if env_exists sevennet; then
        echo "Skipping SevenNet: 'sevennet' environment already exists."
        return
    fi
    echo "Setting up SevenNet environment..."
    conda create -n sevennet python=3.10 -y
    conda activate sevennet
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch_geometric==2.7.0
    pip install torch_scatter==2.1.2+pt25cu121 torch_sparse==0.6.18+pt25cu121 \
        torch_cluster==1.6.3+pt25cu121 torch_spline_conv==1.2.2+pt25cu121 \
        -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    pip install sevenn==0.12.1 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done."
}

setup_mace() {
    if env_exists mace; then
        echo "Skipping MACE: 'mace' environment already exists."
        return
    fi
    echo "Setting up MACE environment..."
    conda create -n mace python=3.10 -y
    conda activate mace
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install "numpy<2" mace-torch==0.3.15 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done."
}

setup_orb() {
    if env_exists orb; then
        echo "Skipping ORB: 'orb' environment already exists."
        return
    fi
    echo "Setting up ORB environment..."
    conda create -n orb python=3.11 -y
    conda activate orb
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install orb-models==0.5.5 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done."
}

setup_chgnet() {
    if env_exists chgnet; then
        echo "Skipping CHGNet: 'chgnet' environment already exists."
        return
    fi
    echo "Setting up CHGNet environment..."
    conda create -n chgnet python=3.11 -y
    conda activate chgnet
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install chgnet==0.4.2 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done."
}

# --- New MLIP models ---

setup_pet_oam() {
    if env_exists pet-oam; then
        echo "Skipping PET: 'pet-oam' environment already exists."
        return
    fi
    echo "Setting up PET-OAM-XL environment..."
    conda create -n pet-oam python=3.11 -y
    conda activate pet-oam
    pip install upet==0.2.1 ase==3.27.0 codecarbon==3.2.3
    conda deactivate
    echo "Done."
}

setup_equflash() {
    if env_exists equflash; then
        echo "Skipping EquFlash: 'equflash' environment already exists."
        return
    fi
    echo "Setting up EquFlash environment..."
    conda create -n equflash python=3.11 -y
    conda activate equflash
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch_geometric==2.7.0
    pip install torch_scatter==2.1.2+pt25cu121 torch_sparse==0.6.18+pt25cu121 \
        torch_cluster==1.6.3+pt25cu121 torch_spline_conv==1.2.2+pt25cu121 \
        -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    pip install sevenn==0.12.1 ase==3.27.0 codecarbon==3.2.3
    # FlashTP requires nvcc to compile CUDA kernels
    module load cuda/12.6.3
    pip install git+https://github.com/SNU-ARC/flashTP@v0.1.0 --no-build-isolation || \
        echo "Warning: FlashTP installation failed. Will use standard SevenNet without flash acceleration."
    module unload cuda/12.6.3
    conda deactivate
    echo "Done."
}


setup_allegro() {
    if ! env_exists nequip; then
        echo "ERROR: 'nequip' environment not found. Run setup_nequip first."
        return 1
    fi
    echo "Setting up Allegro (in nequip env)..."
    conda activate nequip
    pip install nequip-allegro==0.8.2
    conda deactivate
    echo "Done. Run 'MLIP/Allegro/setup_model.sh' to compile the model."
}

# Main
if [[ $# -eq 0 ]]; then
    echo "Setting up all MLIP environments..."
    echo "This may take a while..."
    echo ""
    setup_esen
    setup_nequip
    setup_nequix
    setup_deepmd
    setup_sevennet
    setup_mace
    setup_orb
    setup_chgnet
    setup_pet_oam
    setup_equflash
    setup_allegro  # must come after setup_nequip (shares nequip env)
    echo ""
    echo "All MLIP environments ready!"
else
    case $1 in
        eSEN|esen) setup_esen ;;
        NequIP|nequip) setup_nequip ;;
        Nequix|nequix) setup_nequix ;;
        DPA3|dpa3|deepmd) setup_deepmd ;;
        SevenNet|sevennet) setup_sevennet ;;
        MACE|mace) setup_mace ;;
        ORB|orb) setup_orb ;;
        CHGNet|chgnet) setup_chgnet ;;
        PET|pet|pet-oam) setup_pet_oam ;;
        eSEN_OAM|esen_oam) setup_esen ;;
        EquFlash|equflash) setup_equflash ;;
        NequIP_OAM|nequip_oam) setup_nequip ;;
        Allegro|allegro) setup_allegro ;;
        *)
            echo "Unknown model: $1"
            echo "Available: eSEN, NequIP, Nequix, DPA3, SevenNet, MACE, ORB, CHGNet,"
            echo "           PET, eSEN_OAM, EquFlash, NequIP_OAM, Allegro"
            exit 1
            ;;
    esac
fi
