#!/bin/bash
#SBATCH -J production            # job name (overridden by --job-name)
#SBATCH -o MLIP/production/logs/%x.o%j  # output file
#SBATCH -p 5000_ada              # partition
#SBATCH -N 1                     # total number of nodes
#SBATCH -n 4                     # CPU cores
#SBATCH --mem=16G                # system RAM
#SBATCH --time=72:00:00          # max walltime (72h for long production MD)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1     # 1 GPU

# Usage:
#   sbatch --job-name=CHGNet_prod MLIP/production/slurm_production.sh CHGNet
#   sbatch --job-name=MACE_prod MLIP/production/slurm_production.sh MACE --skip_md
#   sbatch --job-name=all_prod MLIP/production/slurm_production.sh all

MODEL=${1:?Usage: sbatch MLIP/production/slurm_production.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

# Default config
CONFIG="${CONFIG:-MLIP/production/configs/LGPS_600K.json}"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "CONFIG                = $CONFIG"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# Model-to-conda-env mapping
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

run_single_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}

    if [ -z "$env" ]; then
        echo "ERROR: Unknown model '$model'. Known models: ${!MODEL_ENVS[@]}"
        return 1
    fi

    echo "Starting $model at $(date)"
    echo "Conda env: $env"

    # Activate conda
    source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
    conda activate "$env"

    # Model-specific environment variables
    if [ "$model" = "ORB" ]; then
        export TORCHDYNAMO_DISABLE=1
    fi

    PYTHONUNBUFFERED=1 python MLIP/production/run_production_md.py \
        --model "$model" \
        --config "$CONFIG" \
        $EXTRA_ARGS

    echo "Finished $model at $(date)"
}

if [ "$MODEL" = "all" ]; then
    echo "Running all MLIP models sequentially..."
    for model in CHGNet MACE SevenNet DPA3 ORB NequIP Nequix eSEN; do
        run_single_model "$model" || echo "Warning: $model failed"
        echo ""
    done
else
    run_single_model "$MODEL"
fi

echo "Closing Time is $(date)"
