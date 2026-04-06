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
#   # Pretrained models (default)
#   sbatch --job-name=CHGNet_prod MLIP/production/slurm_production.sh CHGNet
#   sbatch --job-name=all_prod MLIP/production/slurm_production.sh all
#
#   # Fine-tuned models
#   VARIANT=finetuned sbatch --job-name=ft_CHGNet_prod MLIP/production/slurm_production.sh CHGNet
#   VARIANT=finetuned sbatch --job-name=ft_all_prod MLIP/production/slurm_production.sh all
#
#   # Custom config
#   CONFIG=MLIP/production/configs/LGPS_300K.json sbatch ... MLIP/production/slurm_production.sh CHGNet

MODEL=${1:?Usage: sbatch MLIP/production/slurm_production.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

# Default config and variant
CONFIG="${CONFIG:-MLIP/production/configs/LGPS_300K.json}"
VARIANT="${VARIANT:-pretrained}"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "CONFIG                = $CONFIG"
echo "VARIANT               = $VARIANT"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# CUDA_HOME for NequIP/Allegro AOTInductor compilation
module load cuda/12.6.3
export CUDA_HOME=/HL9/HCom/cuda/12.6.3

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

# Fine-tuned model checkpoint paths
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

# Pretrained models list
PRETRAINED_MODELS="eSEN eSEN_OAM NequIP NequIP_OAM Allegro Nequix DPA3 SevenNet MACE ORB CHGNet PET EquFlash"
# Fine-tuned models list (no DPA3)
FINETUNED_MODELS="CHGNet MACE SevenNet ORB PET NequIP Allegro EquFlash"

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
    source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
    conda activate "$env"

    # Model-specific environment variables
    if [ "$model" = "ORB" ]; then
        export TORCHDYNAMO_DISABLE=1
    fi

    # Build command
    local cmd="PYTHONUNBUFFERED=1 python MLIP/production/run_production_md.py"
    cmd="$cmd --model $model --config $CONFIG --variant $VARIANT"

    # Add checkpoint for fine-tuned models
    if [ "$VARIANT" = "finetuned" ]; then
        local ckpt_pattern=${FT_CHECKPOINTS[$model]}
        local ckpt=$(ls -1 $ckpt_pattern 2>/dev/null | head -1)
        if [ -z "$ckpt" ]; then
            echo "ERROR: Checkpoint not found for $model: $ckpt_pattern"
            return 1
        fi
        echo "Checkpoint: $ckpt"
        cmd="$cmd --checkpoint $ckpt"
    fi

    cmd="$cmd $EXTRA_ARGS"
    echo "CMD: $cmd"
    eval $cmd

    echo "Finished $model at $(date)"
}

if [ "$MODEL" = "all" ]; then
    if [ "$VARIANT" = "finetuned" ]; then
        echo "Running all fine-tuned MLIP models sequentially..."
        for model in $FINETUNED_MODELS; do
            run_single_model "$model" || echo "Warning: $model failed"
            echo ""
        done
    else
        echo "Running all pretrained MLIP models sequentially..."
        for model in $PRETRAINED_MODELS; do
            run_single_model "$model" || echo "Warning: $model failed"
            echo ""
        done
    fi
else
    run_single_model "$MODEL"
fi

echo "Closing Time is $(date)"
