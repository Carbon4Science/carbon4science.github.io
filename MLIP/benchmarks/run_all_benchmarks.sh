#!/bin/bash
# Run all MLIP benchmarks: pretrained (13 models) + finetuned (8 models)
# Delegates each model to run.sh, which handles conda, GPU monitoring, and logging.
#
# Usage (after salloc):
#   srun --jobid=XXXX bash MLIP/benchmarks/run_all_benchmarks.sh              # both
#   VARIANT=pretrained srun --jobid=XXXX bash MLIP/benchmarks/run_all_benchmarks.sh
#   VARIANT=finetuned srun --jobid=XXXX bash MLIP/benchmarks/run_all_benchmarks.sh

VARIANT="${VARIANT:-both}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SH="${SCRIPT_DIR}/run.sh"

echo "=============================================="
echo "START       = $(date)"
echo "VARIANT     = $VARIANT"
echo "RUN_SH      = $RUN_SH"
echo "=============================================="

# Fine-tuned checkpoint paths
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

# ===== PRETRAINED =====
if [ "$VARIANT" = "pretrained" ] || [ "$VARIANT" = "both" ]; then
    echo ""
    echo "####################################################"
    echo "# PRETRAINED MODELS (13)                           #"
    echo "####################################################"
    for model in $PRETRAINED_MODELS; do
        echo ""
        echo ">>> $model pretrained at $(date)"
        bash "$RUN_SH" "$model" pretrained
        if [ $? -ne 0 ]; then
            echo "WARNING: $model pretrained FAILED"
            FAILED="$FAILED $model/pretrained"
        fi
    done
fi

# ===== FINETUNED =====
if [ "$VARIANT" = "finetuned" ] || [ "$VARIANT" = "both" ]; then
    echo ""
    echo "####################################################"
    echo "# FINETUNED MODELS (8)                             #"
    echo "####################################################"
    for model in $FINETUNED_MODELS; do
        local_ckpt=${FT_CHECKPOINTS[$model]}
        if [ -z "$local_ckpt" ] || [ ! -f "$local_ckpt" ]; then
            echo "WARNING: Checkpoint not found for $model: $local_ckpt — SKIPPING"
            FAILED="$FAILED $model/finetuned(no_ckpt)"
            continue
        fi
        echo ""
        echo ">>> $model finetuned at $(date)"
        bash "$RUN_SH" "$model" finetuned --checkpoint "$local_ckpt"
        if [ $? -ne 0 ]; then
            echo "WARNING: $model finetuned FAILED"
            FAILED="$FAILED $model/finetuned"
        fi
    done
fi

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
fi
echo "=============================================="
