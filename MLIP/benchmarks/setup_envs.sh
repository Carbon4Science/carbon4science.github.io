#!/bin/bash
#
# Setup all conda environments for Carbon4Science benchmarks
#
# Usage:
#   ./setup_envs.sh          # Setup all environments
#   ./setup_envs.sh neuralsym # Setup specific environment
#

set -e

setup_neuralsym() {
    echo "Setting up neuralsym environment..."
    conda create -n neuralsym python=3.6 tqdm scipy pandas joblib -y
    conda activate neuralsym
    conda install pytorch=1.6.0 cudatoolkit=10.1 -c pytorch -y
    conda install rdkit -c rdkit -y
    pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
    conda deactivate
    echo "✓ neuralsym environment ready"
}

setup_localretro() {
    echo "Setting up LocalRetro environment..."
    conda create -c conda-forge -n rdenv python=3.7 -y
    conda activate rdenv
    conda install pytorch cudatoolkit=10.2 -c pytorch -y
    conda install -c conda-forge rdkit -y
    pip install dgl dgllife
    conda deactivate
    echo "✓ LocalRetro environment ready"
}

setup_retrobridge() {
    echo "Setting up RetroBridge environment..."
    conda create --name retrobridge python=3.9 rdkit=2023.09.5 -c conda-forge -y
    conda activate retrobridge
    pip install -r ../Retro/RetroBridge/requirements.txt
    conda deactivate
    echo "✓ RetroBridge environment ready"
}

setup_chemformer() {
    echo "Setting up Chemformer environment..."
    cd ../Retro/Chemformer
    conda env create -f env-dev.yml
    conda activate chemformer
    pip install poetry
    poetry install
    conda deactivate
    cd -
    echo "✓ Chemformer environment ready"
}

setup_rsgpt() {
    echo "Setting up RSGPT environment..."
    cd ../Retro/RSGPT
    conda env create -f environment.yml
    cd -
    echo "✓ RSGPT environment ready"
}

setup_rsmiles() {
    echo "Setting up R-SMILES environment..."
    cd ../Retro/RSMILES
    conda env create -f environment.yml
    cd -
    echo "✓ R-SMILES environment ready"
}

# --- MLIP models ---

setup_esen() {
    echo "Setting up eSEN environment..."
    conda create -n esen python=3.11 -y
    conda activate esen
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124
    pip install fairchem-core ase codecarbon
    conda deactivate
    echo "Done. Run 'huggingface-cli login' to access gated checkpoint."
}

setup_nequip() {
    echo "Setting up NequIP environment..."
    conda create -n nequip python=3.11 -y
    conda activate nequip
    pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124
    pip install nequip ase codecarbon
    conda deactivate
    echo "Done. Run 'MLIP/NequIP/setup_model.sh' to compile the model."
}

setup_nequix() {
    echo "Setting up Nequix environment..."
    conda create -n nequix python=3.10 -y
    conda activate nequix
    pip install nequix ase codecarbon
    conda deactivate
    echo "Done."
}

setup_deepmd() {
    echo "Setting up DPA3 (deepmd) environment..."
    conda create -n deepmd python=3.10 -y
    conda activate deepmd
    pip install torch torchvision torchaudio
    pip install "deepmd-kit[torch]" ase codecarbon
    conda deactivate
    echo "Done. Run 'MLIP/DPA3/download_model.sh' to download checkpoint."
}

setup_sevennet() {
    echo "Setting up SevenNet environment..."
    conda create -n sevennet python=3.10 -y
    conda activate sevennet
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    pip install sevenn ase codecarbon
    conda deactivate
    echo "Done."
}

setup_mace() {
    echo "Setting up MACE environment..."
    conda create -n mace python=3.10 -y
    conda activate mace
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    pip install mace-torch ase codecarbon
    conda deactivate
    echo "Done."
}

setup_orb() {
    echo "Setting up ORB environment..."
    conda create -n orb python=3.11 -y
    conda activate orb
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install orb-models ase codecarbon
    conda deactivate
    echo "Done."
}

setup_chgnet() {
    echo "Setting up CHGNet environment..."
    conda create -n chgnet python=3.10 -y
    conda activate chgnet
    pip install "torch>=2.4.1"
    pip install chgnet ase codecarbon
    conda deactivate
    echo "Done."
}

# --- New MLIP models ---

setup_pet_oam() {
    echo "Setting up PET-OAM-XL environment..."
    conda create -n pet-oam python=3.11 -y
    conda activate pet-oam
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install upet ase codecarbon
    conda deactivate
    echo "Done."
}

setup_equflash() {
    echo "Setting up EquFlash environment..."
    conda create -n equflash python=3.11 -y
    conda activate equflash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    pip install sevenn ase codecarbon
    # FlashTP (optional, requires CUDA toolkit + C compiler)
    pip install git+https://github.com/SNU-ARC/flashTP@v0.1.0 --no-build-isolation || \
        echo "Warning: FlashTP installation failed. Will use standard SevenNet without flash acceleration."
    conda deactivate
    echo "Done."
}


setup_allegro() {
    echo "Setting up Allegro (in nequip env)..."
    conda activate nequip
    pip install nequip-allegro
    conda deactivate
    echo "Done. Run 'MLIP/Allegro/setup_model.sh' to compile the model."
}

# Main
if [[ $# -eq 0 ]]; then
    echo "Setting up all environments..."
    echo "This may take a while..."
    echo ""
    setup_neuralsym
    setup_localretro
    setup_retrobridge
    setup_chemformer
    setup_rsgpt
    setup_rsmiles
    echo ""
    echo "All environments ready!"
else
    case $1 in
        neuralsym) setup_neuralsym ;;
        LocalRetro|localretro) setup_localretro ;;
        RetroBridge|retrobridge) setup_retrobridge ;;
        Chemformer|chemformer) setup_chemformer ;;
        RSGPT|rsgpt) setup_rsgpt ;;
        RSMILES*|rsmiles*) setup_rsmiles ;;
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
            echo "Available: neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT, RSMILES,"
            echo "           eSEN, NequIP, Nequix, DPA3, SevenNet, MACE, ORB, CHGNet,"
            echo "           PET, eSEN_OAM, EquFlash, NequIP_OAM, Allegro"
            exit 1
            ;;
    esac
fi
