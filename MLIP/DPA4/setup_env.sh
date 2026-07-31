#!/bin/bash
# Reproducible DPA4 environment setup.  This is intentionally separate from
# the existing `deepmd` environment used by the DPA3 benchmark.
set -euo pipefail
source /home/dgd03153/apps/anaconda3/etc/profile.d/conda.sh

# DPA4's PyTorch custom operators must be compiled against the cluster CUDA toolkit.
module load cuda/12.8.1

if ! conda env list | awk '{print $1}' | grep -qx dpa4; then
  conda create -n dpa4 python=3.11 -y
fi

conda run -n dpa4 python -m pip install --upgrade pip
conda run -n dpa4 python -m pip install \
  'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
conda run -n dpa4 env DP_ENABLE_PYTORCH=1 python -m pip install \
  --force-reinstall --no-deps \
  'deepmd-kit[torch] @ git+https://github.com/deepmodeling/deepmd-kit.git@v3.2.0b0' \
  ase==3.27.0 codecarbon==3.2.3 h5py
conda run -n dpa4 python -m pip install e3nn

echo "DPA4 environment is ready."
