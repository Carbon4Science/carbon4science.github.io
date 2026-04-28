#!/bin/bash
#SBATCH -J jobName              # job name
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p 5000_ada             # partition name: 5000_ada, 6000_ada, cpu_only
#SBATCH -N 1                    # total number of nodes requested (DO NOT MODIFY)
#SBATCH -n 1                    # MPI-ranks (parallel processes) to be allocated
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00         # Max 72hrs. CPU-only jobs Max 48hrs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=shard:5000ada:100
#SBATCH --qos=qos_5000

# Usage:
#   # Single model (pretrained, default)
#   sbatch --job-name=CHGNet MLIP/benchmarks/slurm_benchmark.sh CHGNet
#
#   # Single model (finetuned)
#   sbatch --job-name=CHGNet_ft MLIP/benchmarks/slurm_benchmark.sh CHGNet finetuned
#
#   # All pretrained models
#   sbatch --job-name=all_pt MLIP/benchmarks/slurm_benchmark.sh all pretrained
#
#   # All finetuned models
#   sbatch --job-name=all_ft MLIP/benchmarks/slurm_benchmark.sh all finetuned

MODEL=${1:?"Usage: sbatch slurm_benchmark.sh MODEL [VARIANT] [extra args...]"}
VARIANT=${2:-pretrained}
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

LOGDIR="MLIP/production/logs"
mkdir -p "$LOGDIR"

# GPU utilization monitor (every 10s, background)
GPU_LOG="${LOGDIR}/gpu_monitor_${MODEL}_${VARIANT}_${SLURM_JOBID}.csv"
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
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "VARIANT               = $VARIANT"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "GPU_LOG               = $GPU_LOG"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# Model-to-conda-env mapping (synced with run.sh)
declare -A MODEL_ENVS=(
    ["eSEN"]="esen"
    ["eSEN_OAM"]="esen"
    ["NequIP"]="nequip"
    ["NequIP_OAM"]="nequip"
    ["Allegro"]="nequip"
    ["Nequix"]="nequix"
    ["DPA3"]="deepmd"
    ["SevenNet"]="sevennet"
    ["MACE"]="mace"
    ["MACE_pruned"]="mace"
    ["ORB"]="orb"
    ["CHGNet"]="chgnet"
    ["PET"]="pet-oam"
    ["EquFlash"]="equflash"
)

# Fine-tuned checkpoint paths (glob patterns resolved at runtime)
declare -A FT_CHECKPOINTS=(
    ["CHGNet"]="MLIP/finetuned/CHGNet/checkpoints/bestE.pth.tar"
    ["MACE"]="MLIP/finetuned/MACE/MACE_finetuned.model"
    ["SevenNet"]="MLIP/finetuned/SevenNet/checkpoint_best.pth"
    ["ORB"]="MLIP/finetuned/ORB/best_checkpoint.pt"
    ["PET"]="MLIP/finetuned/PET/model.pt"
    ["NequIP"]="MLIP/finetuned/NequIP/NequIP_finetuned.nequip.pt2"
    ["Allegro"]="MLIP/finetuned/Allegro/Allegro_finetuned.nequip.pt2"
    ["EquFlash"]="MLIP/finetuned/EquFlash/checkpoint_best.pth"
)

PRETRAINED_MODELS="eSEN eSEN_OAM NequIP NequIP_OAM Allegro Nequix DPA3 SevenNet MACE ORB CHGNet PET EquFlash"
FINETUNED_MODELS="CHGNet MACE SevenNet ORB PET NequIP Allegro EquFlash"

FAILED=""

run_single_model() {
    local model=$1
    local variant=$2
    local env=${MODEL_ENVS[$model]}

    if [ -z "$env" ]; then
        echo "ERROR: Unknown model '$model'. Known models: ${!MODEL_ENVS[@]}"
        return 1
    fi

    echo ""
    echo "=========================================="
    echo "Running $model ($variant, env: $env)"
    echo "=========================================="

    # Activate conda
    source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
    conda deactivate 2>/dev/null
    conda activate "$env"

    # Nequix (JAX) needs nvidia pip-installed CUDA libs on LD_LIBRARY_PATH
    # Commented out: switched Nequix to backend="torch" for fair framework comparison
    # if [ "$env" = "nequix" ]; then
    #     NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
    #     if [ -d "$NVIDIA_DIR" ]; then
    #         for d in "$NVIDIA_DIR"/*/lib; do
    #             [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
    #         done
    #     fi
    # fi

    # Model-specific env vars
    if [ "$model" = "ORB" ]; then
        export TORCHDYNAMO_DISABLE=1
    fi

    # Build command
    local cmd="PYTHONUNBUFFERED=1 python MLIP/benchmarks/run_benchmark.py"
    cmd="$cmd --model $model --variant $variant --track_carbon"

    # Add checkpoint for finetuned or pruned models
    if [ "$variant" = "finetuned" ]; then
        local ckpt=${FT_CHECKPOINTS[$model]}
        if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
            echo "WARNING: Checkpoint not found for $model: $ckpt — SKIPPING"
            FAILED="$FAILED $model/$variant(no_ckpt)"
            return 1
        fi
        cmd="$cmd --checkpoint $ckpt"
    elif [ "$model" = "MACE_pruned" ]; then
        cmd="$cmd --checkpoint MLIP/MACE/MACE_pretrained_pruned.model"
    fi

    if [ -n "$EXTRA_ARGS" ]; then
        cmd="$cmd $EXTRA_ARGS"
    fi

    local logfile="${LOGDIR}/${model}_${variant}.log"
    echo "CMD: $cmd"
    echo "LOG: $logfile"
    eval $cmd 2>&1 | tee "$logfile"
    local status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        echo "WARNING: $model $variant FAILED (exit: $status)"
        FAILED="$FAILED $model/$variant"
    fi
    echo "Finished $model $variant at $(date) (exit: $status)"
    return $status
}

# Run all models or a single model
if [ "$MODEL" = "all" ]; then
    if [ "$VARIANT" = "pretrained" ]; then
        models="$PRETRAINED_MODELS"
    elif [ "$VARIANT" = "finetuned" ]; then
        models="$FINETUNED_MODELS"
    else
        echo "ERROR: VARIANT must be 'pretrained' or 'finetuned' when MODEL=all"
        exit 1
    fi

    echo "Running all $VARIANT models sequentially..."
    for model in $models; do
        run_single_model "$model" "$VARIANT" || true
    done
else
    run_single_model "$MODEL" "$VARIANT"
fi

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
fi
echo "=============================================="
