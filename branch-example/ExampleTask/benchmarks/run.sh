#!/bin/bash
#
# Unified benchmark runner that handles conda environments automatically.
#
# Usage:
#   ./run.sh --model ExampleModel --track_carbon
#   ./run.sh --model all --track_carbon
#
# When adapting for your task, update MODEL_ENVS with your models.

set -e

# ── Model to conda environment mapping ─────────────────────────────────────
# Add one entry per model: ["ModelName"]="conda_env_name"
declare -A MODEL_ENVS=(
    ["ExampleModel"]="example_env"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"

run_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}

    if [[ -z "$env" ]]; then
        echo "Error: Unknown model '$model'"
        echo "Available models: ${!MODEL_ENVS[@]}"
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
    echo ""
}

if $ALL_MODELS; then
    echo "Running all models sequentially..."
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
        echo "  --limit N          Limit test samples"
        echo "  --top_k N          Predictions per sample (default: 50)"
        echo "  --track_carbon     Track carbon emissions"
        echo "  --output FILE      Save results JSON"
        exit 1
    fi
    run_model "$MODEL"
fi
