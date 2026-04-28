#!/bin/bash
#SBATCH -J mlip_relax
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:5000ada:1

# Usage:
#   # Unified protocol, single model
#   sbatch --job-name=CHGNet_relax MLIP/relaxation/slurm_relax.sh CHGNet
#
#   # Unified protocol, all 8 models (sequential)
#   sbatch --job-name=all_relax MLIP/relaxation/slurm_relax.sh all
#
#   # Model-specific protocol (NequIP, Nequix, eSEN only) — runs MLIP/<Model>/Relax.py
#   sbatch --job-name=NequIP_specific MLIP/relaxation/slurm_relax.sh NequIP --specific

MODEL=${1:?"Usage: sbatch slurm_relax.sh MODEL [--specific] [extra args...]"}
shift

MODE="unified"
EXTRA_ARGS=()
for a in "$@"; do
    if [ "$a" = "--specific" ]; then
        MODE="specific"
    else
        EXTRA_ARGS+=("$a")
    fi
done

LOGDIR="MLIP/relaxation/logs"
mkdir -p "$LOGDIR"

GPU_LOG="${LOGDIR}/gpu_monitor_${MODEL}_${MODE}_${SLURM_JOBID}.csv"
while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null
    sleep 10
done &
MON_PID=$!
trap "kill $MON_PID 2>/dev/null" EXIT

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "MODEL                 = $MODEL"
echo "MODE                  = $MODE"
echo "EXTRA_ARGS            = ${EXTRA_ARGS[*]}"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd "$SLURM_SUBMIT_DIR"

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

# Models with a non-default MBD protocol — only these have MLIP/<Model>/Relax.py
SPECIFIC_MODELS=("NequIP" "Nequix" "eSEN")

ALL_MODELS="CHGNet MACE SevenNet NequIP Nequix ORB eSEN DPA3"

FAILED=""

run_single_model() {
    local model=$1
    local mode=$2
    local env=${MODEL_ENVS[$model]}

    if [ -z "$env" ]; then
        echo "ERROR: Unknown model '$model'. Known: ${!MODEL_ENVS[@]}"
        return 1
    fi

    if [ "$mode" = "specific" ]; then
        local is_specific=0
        for m in "${SPECIFIC_MODELS[@]}"; do
            [ "$m" = "$model" ] && is_specific=1
        done
        if [ "$is_specific" = "0" ]; then
            echo "SKIP: $model has no model-specific protocol (uses MBD default that matches unified)"
            return 0
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Running $model ($mode, env: $env)"
    echo "=========================================="

    source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
    conda deactivate 2>/dev/null
    conda activate "$env"

    if [ "$model" = "ORB" ]; then
        export TORCHDYNAMO_DISABLE=1
    fi

    local script
    if [ "$mode" = "unified" ]; then
        script="MLIP/relaxation/run_relaxation.py --model $model"
    else
        script="MLIP/$model/Relax.py"
    fi

    local cmd="PYTHONUNBUFFERED=1 python $script ${EXTRA_ARGS[*]}"
    local logfile="${LOGDIR}/${model}_${mode}.log"
    echo "CMD: $cmd"
    echo "LOG: $logfile"
    eval $cmd 2>&1 | tee "$logfile"
    local status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        echo "WARNING: $model $mode FAILED (exit: $status)"
        FAILED="$FAILED $model/$mode"
    fi
    echo "Finished $model $mode at $(date) (exit: $status)"
    return $status
}

if [ "$MODEL" = "all" ]; then
    echo "Running all models ($MODE) sequentially..."
    for m in $ALL_MODELS; do
        run_single_model "$m" "$MODE" || true
    done
else
    run_single_model "$MODEL" "$MODE"
fi

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
fi
echo "=============================================="
