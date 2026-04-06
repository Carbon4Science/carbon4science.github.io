#!/bin/bash
#SBATCH -J finetune
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:5000ada:1

# Usage:
#   sbatch --job-name=ft_CHGNet MLIP/benchmarks/slurm_finetune.sh CHGNet
#   sbatch --job-name=ft_CHGNet MLIP/benchmarks/slurm_finetune.sh CHGNet --clean
#   sbatch --job-name=ft_all MLIP/benchmarks/slurm_finetune.sh all --clean

MODEL=${1:?Usage: sbatch MLIP/benchmarks/slurm_finetune.sh MODEL_NAME [--clean] [extra args...]}
shift
EXTRA_ARGS="$@"

# Default dataset and epochs
DATASET="${DATASET:-MLIP/finetuning_data/dataset_LGPS_600K_n100.xyz}"
EPOCHS="${EPOCHS:-100}"

LOGDIR="MLIP/finetuned/logs"
mkdir -p "$LOGDIR"

# GPU utilization monitor (every 10s, background)
GPU_LOG="${LOGDIR}/gpu_monitor_finetune_${MODEL}_${SLURM_JOBID}.csv"
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
echo "DATASET               = $DATASET"
echo "EPOCHS                = $EPOCHS"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "GPU_LOG               = $GPU_LOG"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# CUDA_HOME is required for NequIP/Allegro AOTInductor compilation
module load cuda/12.6.3
export CUDA_HOME=/HL9/HCom/cuda/12.6.3

# Model-to-conda-env mapping
declare -A MODEL_ENVS=(
    ["CHGNet"]="chgnet"
    ["MACE"]="mace"
    ["SevenNet"]="sevennet"
    ["DPA3"]="deepmd"
    ["NequIP"]="nequip"
    ["ORB"]="orb"
    ["PET"]="pet-oam"
    ["Allegro"]="nequip"
    #["EquFlash"]="equflash"
)

FAILED=""

run_single_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}

    if [ -z "$env" ]; then
        echo "ERROR: Unknown model '$model'. Known models: ${!MODEL_ENVS[@]}"
        return 1
    fi

    local output_dir="MLIP/finetuned/${model}"
    local output_json="${output_dir}/finetune_results.json"
    local logfile="${LOGDIR}/${model}_finetune.log"

    echo ""
    echo "=========================================="
    echo "Fine-tuning $model (env: $env)"
    echo "Output: $output_dir"
    echo "Log:    $logfile"
    echo "=========================================="

    source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
    conda deactivate 2>/dev/null
    conda activate "$env"

    PYTHONUNBUFFERED=1 python MLIP/run_finetune.py \
        --model "$model" \
        --dataset "$DATASET" \
        --output_dir "$output_dir" \
        --epochs "$EPOCHS" \
        --track_carbon \
        --output_json "$output_json" \
        $EXTRA_ARGS 2>&1 | tee "$logfile"
    local status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        echo "WARNING: $model finetune FAILED (exit: $status)"
        FAILED="$FAILED $model"
    fi
    echo "Finished $model at $(date) (exit: $status)"
    return $status
}

if [ "$MODEL" = "all" ]; then
    echo "Fine-tuning all models sequentially..."
    for model in CHGNet MACE SevenNet NequIP ORB PET Allegro EquFlash; do
        run_single_model "$model" || true
    done
else
    run_single_model "$MODEL"
fi

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
fi
echo "=============================================="
