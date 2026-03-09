#!/bin/bash
#SBATCH -J benchmark            # job name (overridden by --job-name)
#SBATCH -o benchmarks/logs/%x.o%j  # output file (%x=job name, %j=jobID)
#SBATCH -p 5000_ada             # partition
#SBATCH -N 1                    # total number of nodes
#SBATCH -n 4                    # CPU cores
#SBATCH --mem=16G               # system RAM
#SBATCH --time=72:00:00         # max walltime
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1    # 1 GPU

# Usage:
#   sbatch --job-name=RSGPT benchmarks/slurm_benchmark.sh RSGPT
#   sbatch --job-name=RetroBridge benchmarks/slurm_benchmark.sh RetroBridge
#   sbatch --job-name=Chemformer benchmarks/slurm_benchmark.sh Chemformer --data Retro/data/uspto_50_chemforner.pickle
#   sbatch --job-name=RSMILES_20x benchmarks/slurm_benchmark.sh RSMILES_20x

MODEL=${1:?Usage: sbatch benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# Model-to-conda-env mapping
declare -A MODEL_ENVS=(
    ["neuralsym"]="neuralsym"
    ["LocalRetro"]="rdenv"
    ["RetroBridge"]="retrobridge"
    ["Chemformer"]="chemformer"
    ["RSGPT"]="gpt"
    ["RSMILES_1x"]="rsmiles"
    ["RSMILES_20x"]="rsmiles"
    ["eSEN"]="esen"
    ["NequIP"]="nequip"
    ["Nequix"]="nequix"
    ["DPA3"]="deepmd"
    ["SevenNet"]="sevennet"
    ["MACE"]="mace"
    ["ORB"]="orb"
    ["CHGNet"]="chgnet"
)

# Model-to-task mapping (default: Retro)
declare -A MODEL_TASK=(
    ["eSEN"]="MLIP"
    ["NequIP"]="MLIP"
    ["Nequix"]="MLIP"
    ["DPA3"]="MLIP"
    ["SevenNet"]="MLIP"
    ["MACE"]="MLIP"
    ["ORB"]="MLIP"
    ["CHGNet"]="MLIP"
)

ENV_NAME=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Unknown model '$MODEL'. Known models: ${!MODEL_ENVS[@]}"
    exit 1
fi

# Auto-detect task from model name
TASK=${MODEL_TASK[$MODEL]:-Retro}

# Determine output file names based on task and test set size
if [ "$TASK" = "MLIP" ]; then
    N=1
    OUTPUT="benchmarks/results/MLIP/${MODEL}_${N}.json"
    PREDICTIONS=""
elif echo "$EXTRA_ARGS" | grep -q "chemforner.pickle"; then
    N=5004
    OUTPUT="benchmarks/results/Retro/${MODEL,,}_${N}.json"
    PREDICTIONS="benchmarks/results/Retro/${MODEL,,}_${N}_predictions.json"
else
    N=5007
    OUTPUT="benchmarks/results/Retro/${MODEL,,}_${N}.json"
    PREDICTIONS="benchmarks/results/Retro/${MODEL,,}_${N}_predictions.json"
fi

# Activate conda
source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Set PYTHONPATH for Chemformer (needs molbart)
if [ "$MODEL" = "Chemformer" ]; then
    export PYTHONPATH="$(pwd)/Retro/Chemformer:$PYTHONPATH"
fi

echo "Starting Time is $(date)"
echo "Conda env: $ENV_NAME"
echo "Output: $OUTPUT"

PRED_ARGS=""
if [ -n "$PREDICTIONS" ]; then
    PRED_ARGS="--save_predictions $PREDICTIONS"
fi

PYTHONUNBUFFERED=1 python benchmarks/run_benchmark.py \
    --task "$TASK" \
    --model "$MODEL" \
    --track_carbon \
    --output "$OUTPUT" \
    $PRED_ARGS \
    $EXTRA_ARGS

echo "Closing Time is $(date)"
