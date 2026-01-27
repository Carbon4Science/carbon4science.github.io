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
    pip install -r ../Retrosynthesis/RetroBridge/requirements.txt
    conda deactivate
    echo "✓ RetroBridge environment ready"
}

setup_chemformer() {
    echo "Setting up Chemformer environment..."
    cd ../Retrosynthesis/Chemformer
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
    cd ../Retrosynthesis/RSGPT
    conda env create -f environment.yml
    cd -
    echo "✓ RSGPT environment ready"
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
    echo ""
    echo "All environments ready!"
else
    case $1 in
        neuralsym) setup_neuralsym ;;
        LocalRetro|localretro) setup_localretro ;;
        RetroBridge|retrobridge) setup_retrobridge ;;
        Chemformer|chemformer) setup_chemformer ;;
        RSGPT|rsgpt) setup_rsgpt ;;
        *)
            echo "Unknown model: $1"
            echo "Available: neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT"
            exit 1
            ;;
    esac
fi
