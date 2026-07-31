#!/bin/bash
# Executed inside slurm_benchmark.sh after allocation.  Do not run this
# directly for the final benchmark unless equivalent Slurm resources are set.

set -euo pipefail
module load cuda/12.8.1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
MODEL="${1:?Usage: slurm_dynamat_benchmark.sh MODEL [extra arguments]}"
shift
# Be tolerant when a scheduler wrapper repeats the model argument.
if [[ "${1:-}" == "$MODEL" ]]; then
  shift
fi

declare -A MODEL_ENVS=(
  [eSEN]=esen [ORB]=orb [DPA4]=dpa4 [NequIP]=nequip
  [MACE]=mace [SevenNet]=sevennet [Nequix]=nequix [CHGNet]=chgnet
)
MODELS=(eSEN ORB DPA4 NequIP MACE SevenNet Nequix CHGNet)
DPA4_CHECKPOINT="${DPA4_MODEL:-MLIP/DPA4/dpa-4.0-pro-mptrj-21.88-32.10.pt}"

if [[ "$MODEL" == all && ! -f "$DPA4_CHECKPOINT" ]]; then
  echo "DPA4 checkpoint not found: $DPA4_CHECKPOINT" >&2
  echo "Prepare it before submission or set DPA4_MODEL." >&2
  exit 2
fi

run_one() {
  local model="$1"
  shift
  local env="${MODEL_ENVS[$model]}"
  source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
  conda deactivate 2>/dev/null || true
  conda activate "$env"
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/carbon4science-matplotlib"
  mkdir -p "$MPLCONFIGDIR"

  local -a command=(python MLIP/dynamat_benchmark.py --model "$model" --track-carbon)
  if [[ "$model" == DPA4 ]]; then
    if [[ ! -f "$DPA4_CHECKPOINT" ]]; then
      echo "DPA4 checkpoint not found: $DPA4_CHECKPOINT" >&2
      echo "Prepare it before submission or set DPA4_MODEL." >&2
      return 2
    fi
    command+=(--dpa4-checkpoint "$DPA4_CHECKPOINT")
  fi
  command+=("$@")
  echo "Running $model in $env"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  PYTHONUNBUFFERED=1 "${command[@]}"
}

if [[ "$MODEL" == all ]]; then
  for model in "${MODELS[@]}"; do
    run_one "$model" "$@"
  done
else
  [[ -n "${MODEL_ENVS[$MODEL]:-}" ]] || { echo "Unknown model: $MODEL" >&2; exit 2; }
  run_one "$MODEL" "$@"
fi
