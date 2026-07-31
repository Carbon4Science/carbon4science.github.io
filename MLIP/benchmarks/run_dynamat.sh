#!/bin/bash
# Launch the new dynamat benchmark in the model-specific conda environment.
# The legacy LGPS/RDF/MSD runners are intentionally not changed.

set -euo pipefail
module load cuda/12.8.1
MODEL="${1:?Usage: run_dynamat.sh MODEL [extra arguments]}"
shift

declare -A MODEL_ENVS=(
  [eSEN]=esen [ORB]=orb [DPA4]=dpa4 [NequIP]=nequip
  [MACE]=mace [SevenNet]=sevennet [Nequix]=nequix [CHGNet]=chgnet
)
ENV="${MODEL_ENVS[$MODEL]:-}"
if [[ -z "$ENV" ]]; then
  echo "Unknown model: $MODEL" >&2
  exit 2
fi

source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate "$ENV"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/carbon4science-matplotlib"
mkdir -p "$MPLCONFIGDIR"
exec python MLIP/dynamat_benchmark.py --model "$MODEL" --track-carbon "$@"
