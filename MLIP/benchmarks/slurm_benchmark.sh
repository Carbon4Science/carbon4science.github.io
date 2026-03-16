#!/bin/bash
#SBATCH -J jobName              # job name
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p 5000_ada             # partition name: 5000_ada, 6000_ada, cpu_only
#SBATCH -N 1                    # total number of nodes requested (DO NOT MODIFY)
#SBATCH -n 1                    # MPI-ranks (parallel processes) to be allocated
#SBATCH --ntasks-per-node=1    # Same as -n above unless using multiple GPUs and parallel runs. Make sure you know what you're doing.
#SBATCH --time=72:00:00         # Max 72hrs. CPU-only jobs Max 48hrs
#SBATCH --cpus-per-task=4       # CPU cores requested. Max 24 for GPU-included jobs (per GPU), Max 32 for CPU-only jobs. 2-8 should suffice for most jobs.
#SBATCH --mem=8G        # System RAM requested. Consider 2.5G/core as the upper limit and set accordingly.
#SBATCH --gres=gpu:5000ada:1  #GPU resources. Double-Hash if using cpu only. shard is Percentage of GPU resources required (base it on VRAM and Util% of your job) Dont use shard for 6000ada

# Usage:
#   sbatch --job-name=CHGNet MLIP/benchmarks/slurm_benchmark.sh CHGNet
#   sbatch --job-name=MACE MLIP/benchmarks/slurm_benchmark.sh MACE
#   sbatch --job-name=all MLIP/benchmarks/slurm_benchmark.sh all

MODEL=${1:?Usage: sbatch MLIP/benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

while true
do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    >> gpu_monitor_${MODEL}_${SLURM_JOBID}.csv
    sleep 10
done &

MON_PID=$!
trap "kill $MON_PID" EXIT

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

# Model-to-conda-env mapping (synced with run.sh)
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

# Model-to-task mapping (synced with run.sh)
declare -A MODEL_TASK=(
    ["eSEN"]="MLIP"
    ["NequIP"]="MLIP"
    ["Nequix"]="MLIP"
    ["DPA3"]="MLIP"
    ["SevenNet"]="MLIP"
    ["MACE"]="MLIP"
    ["ORB"]="MLIP"
    ["CHGNet"]="MLIP"
    ["PET"]="MLIP"
    ["eSEN_OAM"]="MLIP"
    ["EquFlash"]="MLIP"
    ["NequIP_OAM"]="MLIP"
    ["Allegro"]="MLIP"
)

# MLIP production mode args
MLIP_ARGS="--mlip_mode production --production_config MLIP/production/configs/LGPS_300K.json"

# Function to run a single model
run_single_model() {
    local model=$1
    local env=${MODEL_ENVS[$model]}

    if [ -z "$env" ]; then
        echo "ERROR: Unknown model '$model'. Known models: ${!MODEL_ENVS[@]}"
        return 1
    fi

    local task=${MODEL_TASK[$model]:-MLIP}
    local output="MLIP/results/outputs/${model}_1.json"

    echo ""
    echo "=========================================="
    echo "Running $model (env: $env, task: $task)"
    echo "Output: $output"
    echo "=========================================="

    # Activate conda
    source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate "$env"

    # Nequix (JAX) needs nvidia pip-installed CUDA libs on LD_LIBRARY_PATH
    if [ "$env" = "nequix" ]; then
        NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
        if [ -d "$NVIDIA_DIR" ]; then
            for d in "$NVIDIA_DIR"/*/lib; do
                [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
            done
        fi
    fi

    PYTHONUNBUFFERED=1 srun python MLIP/benchmarks/run_benchmark.py \
        --task "$task" \
        --model "$model" \
        --track_carbon \
        --output "$output" \
        $MLIP_ARGS \
        $EXTRA_ARGS

    echo "Finished $model at $(date)"
}

# Run all models sequentially or a single model
if [ "$MODEL" = "all" ]; then
    echo "Running all MLIP models sequentially..."
    for model in CHGNet MACE SevenNet DPA3 ORB NequIP Nequix eSEN PET eSEN_OAM EquFlash NequIP_OAM Allegro; do
        run_single_model "$model" || echo "Warning: $model failed"
    done
else
    run_single_model "$MODEL"
fi

echo "Closing Time is $(date)"
