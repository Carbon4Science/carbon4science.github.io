#!/bin/bash
# Download the exact DPA-4.0-Pro-MPtrj checkpoint once its official URL is
# supplied.  No alternative MatPES/OMat24 model is substituted implicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${DPA4_MODEL_PATH:-${SCRIPT_DIR}/dpa-4.0-pro-mptrj.pt}"
URL="${DPA4_MODEL_URL:-}"

if [[ -z "$URL" ]]; then
  echo "DPA4_MODEL_URL is not set." >&2
  echo "Set it to the official DPA-4.0-Pro-MPtrj checkpoint URL." >&2
  exit 2
fi

curl -L --fail --retry 3 -o "$TARGET" "$URL"
echo "Downloaded: $TARGET"
