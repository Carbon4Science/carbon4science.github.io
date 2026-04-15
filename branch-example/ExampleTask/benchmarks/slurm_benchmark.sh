#!/bin/bash
#SBATCH -J benchmark
#SBATCH -o benchmarks/logs/%x.o%j
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1
#
# Usage:
#   cd <Task>/
#   sbatch --job-name=ExampleModel benchmarks/slurm_benchmark.sh ExampleModel
#
# When adapting for your task, update MODEL_ENVS below.

MODEL=${1:?Usage: sbatch benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "MODEL                 = $MODEL"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# ── Model to conda environment mapping ─────────────────────────────────────
declare -A MODEL_ENVS=(
    ["ExampleModel"]="example_env"
)

ENV_NAME=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Unknown model '$MODEL'. Known models: ${!MODEL_ENVS[@]}"
    exit 1
fi

# Output paths
N="full"  # Replace with your test set size, e.g. 5007
OUTPUT="results/${MODEL,,}_${N}.json"
PREDICTIONS="results/${MODEL,,}_${N}_predictions.json"

# Activate conda (update path for your cluster)
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Starting Time is $(date)"
echo "Conda env: $ENV_NAME"
echo "Output: $OUTPUT"

mkdir -p benchmarks/logs results

PYTHONUNBUFFERED=1 python benchmarks/run_benchmark.py \
    --model "$MODEL" \
    --track_carbon \
    --output "$OUTPUT" \
    --save_predictions "$PREDICTIONS" \
    $EXTRA_ARGS

echo "Closing Time is $(date)"
