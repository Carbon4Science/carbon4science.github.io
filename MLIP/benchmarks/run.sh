#!/bin/bash
# Run a single MLIP production MD benchmark.
#
# Usage (inside salloc):
#   srun bash MLIP/benchmarks/run.sh <MODEL> <VARIANT> [--checkpoint PATH]
#
# Examples:
#   srun bash MLIP/benchmarks/run.sh CHGNet pretrained
#   srun bash MLIP/benchmarks/run.sh CHGNet finetuned --checkpoint MLIP/finetuned/CHGNet/checkpoints/bestE.pth.tar
#   srun bash MLIP/benchmarks/run.sh eSEN pretrained

set -e

MODEL=${1:?"Usage: run.sh MODEL VARIANT [--checkpoint PATH]"}
VARIANT=${2:?"Usage: run.sh MODEL VARIANT [--checkpoint PATH]"}
shift 2

CONFIG="MLIP/production/configs/LGPS_300K.json"
LOGDIR="MLIP/production/logs"
mkdir -p "$LOGDIR"

# Model-to-conda-env mapping
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
    ["ORB"]="orb"
    ["CHGNet"]="chgnet"
    ["PET"]="pet-oam"
    ["EquFlash"]="equflash"
)

ENV=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'"
    echo "Available: ${!MODEL_ENVS[@]}"
    exit 1
fi

# Activate conda
source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate "$ENV"

# Nequix (JAX) needs nvidia pip-installed CUDA libs on LD_LIBRARY_PATH
# Commented out: switched Nequix to backend="torch" for fair framework comparison
# if [ "$ENV" = "nequix" ]; then
#     NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
#     if [ -d "$NVIDIA_DIR" ]; then
#         for d in "$NVIDIA_DIR"/*/lib; do
#             [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
#         done
#     fi
# fi

# Model-specific env vars
if [ "$MODEL" = "ORB" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

# Build command
CMD="PYTHONUNBUFFERED=1 python MLIP/production/run_production_md.py"
CMD="$CMD --model $MODEL --config $CONFIG --variant $VARIANT --track_carbon"

# Pass through extra args (e.g. --checkpoint)
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

LOGFILE="${LOGDIR}/${MODEL}_${VARIANT}.log"

echo "=========================================="
echo "Model:   $MODEL"
echo "Variant: $VARIANT"
echo "Env:     $ENV"
echo "Log:     $LOGFILE"
echo "CMD:     $CMD"
echo "Started: $(date)"
echo "=========================================="

# GPU utilization monitor (every 10s, background)
GPU_LOG="${LOGDIR}/gpu_monitor_${MODEL}_${VARIANT}.csv"
while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null
    sleep 10
done &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

eval $CMD 2>&1 | tee "$LOGFILE"
STATUS=${PIPESTATUS[0]}

echo "Finished $MODEL $VARIANT at $(date) (exit: $STATUS)"
echo "GPU log: $GPU_LOG"
exit $STATUS
