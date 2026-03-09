#!/bin/bash
# Download DPA-3.1-MPtrj checkpoint from Figshare
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Downloading DPA-3.1-MPtrj checkpoint..."
curl -L -o "${SCRIPT_DIR}/dpa-3.1-mptrj.pth" "https://ndownloader.figshare.com/files/55141124"
echo "Done: ${SCRIPT_DIR}/dpa-3.1-mptrj.pth"
