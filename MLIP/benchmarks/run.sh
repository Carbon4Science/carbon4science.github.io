#!/bin/bash
#
# Unified benchmark runner that handles conda environments automatically
#
# Usage:
#   ./run.sh --model neuralsym --device cuda:0 --smiles "CCO"
#   ./run.sh --model LocalRetro --input test.csv --track_carbon
#   ./run.sh --model all --input test.csv  # Run all models sequentially
#

set -e

# Model to conda environment mapping
declare -A MODEL_ENVS=(
    ["eSEN"]="esen"
    ["NequIP"]="nequip"
    ["Nequix"]="nequix"
    ["DPA3"]="deepmd"
    ["SevenNet"]="sevennet"
    ["MACE"]="mace"
    ["ORB"]="orb"
    ["CHGNet"]="chgnet"
    ["PET"]="pet-oam"
    ["eSEN_OAM"]="esen"
    ["EquFlash"]="equflash"
    ["NequIP_OAM"]="nequip"
    ["Allegro"]="nequip"
)

# Model to task mapping (default: Retro)
declare -A MODEL_TASK=(
    ["eSEN"]="MLIP"
    ["NequIP"]="MLIP"
    ["Nequix"]="MLIP"
    ["DPA3"]="MLIP"
    ["SevenNet"]="MLIP"
    ["MACE"]="MLIP"
    ["ORB"]="MLIP"
    ["CHGNet"]="MLIP"
    ["PET"]="MLIP"
    ["eSEN_OAM"]="MLIP"
    ["EquFlash"]="MLIP"
    ["NequIP_OAM"]="MLIP"
    ["Allegro"]="MLIP"
)

# Parse --model argument
MODEL=""
ALL_MODELS=false
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ "$2" == "all" ]]; then
                ALL_MODELS=true
            else
                MODEL="$2"
            fi
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Get script directory (MLIP/benchmarks/) and repo root (Carbon4Science/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Function to run benchmark for a single model
run_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}
    local task=${MODEL_TASK[$model]:-Retro}

    if [[ -z "$env" ]]; then
        echo "Error: Unknown model '$model'"
        echo "Available models: ${!MODEL_ENVS[@]}"
        exit 1
    fi

    echo "=========================================="
    echo "Running $model (env: $env, task: $task)"
    echo "=========================================="

    # Activate conda and run
    eval "$(conda shell.bash hook)"
    conda activate "$env"

    # Nequix (JAX) needs nvidia pip-installed CUDA libs on LD_LIBRARY_PATH
    if [ "$env" = "nequix" ]; then
        NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
        if [ -d "$NVIDIA_DIR" ]; then
            for d in "$NVIDIA_DIR"/*/lib; do
                [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
            done
        fi
    fi

    cd "$ROOT_DIR"
    python MLIP/benchmarks/run_benchmark.py --task "$task" --model "$model" "${ARGS[@]}"

    conda deactivate
    echo ""
}

# Run all models or single model
if $ALL_MODELS; then
    echo "Running all models sequentially..."
    echo ""
    for model in "${!MODEL_ENVS[@]}"; do
        run_model "$model" || echo "Warning: $model failed"
    done
    echo "Done!"
else
    if [[ -z "$MODEL" ]]; then
        echo "Usage: ./run.sh --model <model_name> [options]"
        echo ""
        echo "Models: ${!MODEL_ENVS[@]}"
        echo ""
        echo "Options are passed to run_benchmark.py:"
        echo "  --device cuda:0|cpu"
        echo "  --task inference|training"
        echo "  --metric top_1 top_5 top_10"
        echo "  --smiles \"SMILES\""
        echo "  --input file.csv"
        echo "  --track_carbon"
        exit 1
    fi
    run_model "$MODEL"
fi
