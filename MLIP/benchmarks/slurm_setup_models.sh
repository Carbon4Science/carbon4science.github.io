#!/bin/bash
#SBATCH -J setup_models         # job name
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p 5000_ada             # partition name: 5000_ada, 6000_ada, cpu_only
#SBATCH -N 1                    # total number of nodes requested (DO NOT MODIFY)
#SBATCH -n 1                    # MPI-ranks (parallel processes) to be allocated
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00         # compilation is fast, 1 hour is plenty
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=shard:10  #GPU resources. Double-Hash if using cpu only. shard is Percentage of GPU resources required (base it on VRAM and Util% of your job) Dont use shard for 6000ada

#
# Compile NequIP-family model checkpoints (requires GPU).
# Run this BEFORE submitting benchmark jobs.
#
# Usage:
#   sbatch MLIP/benchmarks/slurm_setup_models.sh           # compile all
#   sbatch MLIP/benchmarks/slurm_setup_models.sh NequIP    # compile one
#
# Models compiled here:
#   NequIP     → NequIP-MP-L.nequip.pt2      (nequip.net:mir-group/NequIP-MP-L:0.1)
#   NequIP_OAM → NequIP-OAM-L.nequip.pt2     (nequip.net:mir-group/NequIP-OAM-L:0.1)
#   Allegro    → Allegro-OAM-L.nequip.pt2     (nequip.net:mir-group/Allegro-OAM-L:0.1)
#
# Skips models whose compiled checkpoint already exists.
#

MODEL=${1:-all}

cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "working directory     = $(pwd)"
echo "=============================================="

# Initialize conda and activate the nequip environment
source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
conda activate nequip

# CUDA_HOME is required for AOTInductor C++ compilation
module load cuda/12.6.3
export CUDA_HOME=/HL9/HCom/cuda/12.6.3

MLIP_DIR="MLIP"

# Model ID → output path mapping
declare -A CHECKPOINTS=(
    ["NequIP"]="$MLIP_DIR/NequIP/NequIP-MP-L.nequip.pt2"
    ["NequIP_OAM"]="$MLIP_DIR/NequIP_OAM/NequIP-OAM-L.nequip.pt2"
    ["Allegro"]="$MLIP_DIR/Allegro/Allegro-OAM-L.nequip.pt2"
)
declare -A MODEL_IDS=(
    ["NequIP"]="nequip.net:mir-group/NequIP-MP-L:0.1"
    ["NequIP_OAM"]="nequip.net:mir-group/NequIP-OAM-L:0.1"
    ["Allegro"]="nequip.net:mir-group/Allegro-OAM-L:0.1"
)

compile_model() {
    local model=$1
    local ckpt=${CHECKPOINTS[$model]}
    local model_id=${MODEL_IDS[$model]}

    if [ -z "$ckpt" ]; then
        echo "ERROR: Unknown model '$model'. Available: NequIP, NequIP_OAM, Allegro"
        return 1
    fi

    if [ -f "$ckpt" ]; then
        echo "Skipping $model: compiled checkpoint already exists ($ckpt)"
        return 0
    fi

    echo ""
    echo "Compiling $model ($model_id)..."
    echo "Output: $ckpt"
    nequip-compile "$model_id" "$ckpt" \
        --mode aotinductor --device cuda --target ase

    if [ -f "$ckpt" ]; then
        echo "Done: $ckpt"
    else
        echo "ERROR: Compilation failed for $model"
        return 1
    fi
}

if [ "$MODEL" = "all" ]; then
    echo "Compiling all NequIP-family models..."
    for model in NequIP NequIP_OAM Allegro; do
        compile_model "$model" || echo "Warning: $model compilation failed"
    done
else
    compile_model "$MODEL"
fi

conda deactivate
echo ""
echo "Finished at $(date)"
