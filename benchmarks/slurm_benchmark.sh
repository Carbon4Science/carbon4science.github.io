#!/bin/bash
#SBATCH -J forward_benchmark
#SBATCH -o benchmarks/logs/%x.o%j
#SBATCH -p 5000_ada
#SBATCH -N 1 -n 4 --mem=16G --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1

MODEL=${1:?Usage: sbatch benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

echo "=============================================="
echo "DATE=$(date) JOBID=$SLURM_JOBID MODEL=$MODEL"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

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

ENV_NAME=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Unknown model '$MODEL'. Known: ${!MODEL_ENVS[@]}"
    exit 1
fi

N=40029
OUTPUT="results/${MODEL,,}_${N}.json"
PREDICTIONS="results/${MODEL,,}_${N}_predictions.json"

source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

if [ "$MODEL" = "Chemformer" ]; then
    export PYTHONPATH="$(pwd)/Chemformer:$PYTHONPATH"
fi

mkdir -p benchmarks/logs results

echo "Starting: $(date) env=$ENV_NAME output=$OUTPUT"

PYTHONUNBUFFERED=1 python benchmarks/run_benchmark.py \
    --model "$MODEL" --track_carbon \
    --output "$OUTPUT" --save_predictions "$PREDICTIONS" \
    $EXTRA_ARGS

echo "Done: $(date)"
