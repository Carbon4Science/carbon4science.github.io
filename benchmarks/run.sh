#!/bin/bash
set -e

declare -A MODEL_ENVS=(
    ["neuralsym"]="neuralsym"
    ["MolecularTransformer"]="mol_transformer"
    ["MEGAN"]="megan"
    ["Graph2SMILES"]="mol_transformer"
    ["Chemformer"]="chemformer"
    ["RSMILES_20x"]="rsmiles"
    ["LlaSMol"]="llasmol"
    ["WLDN"]="wldn"
)

MODEL=""
ALL_MODELS=false
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ "$2" == "all" ]]; then ALL_MODELS=true; else MODEL="$2"; fi
            shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"

run_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}
    if [[ -z "$env" ]]; then
        echo "Error: Unknown model '$model'. Available: ${!MODEL_ENVS[@]}"
        exit 1
    fi
    echo "=========================================="
    echo "Running $model (env: $env)"
    echo "=========================================="
    eval "$(conda shell.bash hook)"
    conda activate "$env"
    cd "$TASK_DIR"
    python benchmarks/run_benchmark.py --model "$model" "${ARGS[@]}"
    conda deactivate
}

if $ALL_MODELS; then
    for model in "${!MODEL_ENVS[@]}"; do run_model "$model" || echo "Warning: $model failed"; done
else
    if [[ -z "$MODEL" ]]; then
        echo "Usage: ./benchmarks/run.sh --model <name> [options]"
        echo "Models: ${!MODEL_ENVS[@]}"
        exit 1
    fi
    run_model "$MODEL"
fi
