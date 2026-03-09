#!/bin/bash
#
# Production MD runner with conda environment switching.
#
# Usage:
#   ./MLIP/production/run_production.sh --model CHGNet --config MLIP/production/configs/LGPS_600K.json
#   ./MLIP/production/run_production.sh --model all --config MLIP/production/configs/LGPS_600K.json
#   ./MLIP/production/run_production.sh --model CHGNet --config MLIP/production/configs/LGPS_600K.json --skip_md
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

# Get script and root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Function to run production MD for a single model
run_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}

    if [[ -z "$env" ]]; then
        echo "Error: Unknown model '$model'"
        echo "Available models: ${!MODEL_ENVS[@]}"
        exit 1
    fi

    echo "=========================================="
    echo "Production MD: $model (env: $env)"
    echo "=========================================="

    # Model-specific environment variables
    if [[ "$model" == "ORB" ]]; then
        export TORCHDYNAMO_DISABLE=1
    fi

    # Activate conda and run
    eval "$(conda shell.bash hook)"
    conda activate "$env"

    cd "$ROOT_DIR"
    python MLIP/production/run_production_md.py --model "$model" "${ARGS[@]}"

    conda deactivate
    echo ""
}

# Run all models or single model
if $ALL_MODELS; then
    echo "Running all MLIP models sequentially..."
    echo ""
    for model in "${!MODEL_ENVS[@]}"; do
        run_model "$model" || echo "Warning: $model failed"
    done
    echo "Done!"
else
    if [[ -z "$MODEL" ]]; then
        echo "Usage: ./MLIP/production/run_production.sh --model <model_name> --config <config.json> [options]"
        echo ""
        echo "Models: ${!MODEL_ENVS[@]}"
        echo ""
        echo "Options:"
        echo "  --config <path>         JSON config file (required)"
        echo "  --structure_index <N>   Structure index in config (default: 0)"
        echo "  --skip_md               Skip MD, re-run analysis only"
        echo "  --skip_analysis         Run MD only, skip analysis"
        exit 1
    fi
    run_model "$MODEL"
fi
