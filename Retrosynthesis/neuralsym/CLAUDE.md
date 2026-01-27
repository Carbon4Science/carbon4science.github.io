# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuralSym is a neural-symbolic machine learning system for retrosynthesis prediction. Given a target product molecule (SMILES), it predicts reaction templates and precursor molecules needed for synthesis. This is a re-implementation of Segler et al.'s expansion network from "Planning chemical syntheses with deep neural networks and symbolic AI" (Nature 2018).

## Environment Setup

```bash
# Conda environment with Python 3.6
conda create -n neuralsym python=3.6 tqdm pathlib typing scipy pandas joblib -y
conda activate neuralsym

# PyTorch 1.6.0 with CUDA 10.1
conda install -y pytorch=1.6.0 torchvision cudatoolkit=10.1 torchtext -c pytorch
conda install -y rdkit -c rdkit

# RDChiral for template application
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```

For evaluation without full environment, a minimal venv with Python 3.9+ and PyTorch/pandas works for parameter counting and accuracy evaluation (see `venv_params/`).

## Commands

### Data Preparation
```bash
python prepare_data.py  # Extract templates, generate fingerprints (~15 min on 8-core)
```

### Training
```bash
bash -i train.sh  # ~5 minutes total on RTX2080, 8 sec/epoch
```

Or directly:
```bash
python train.py \
    --model Highway --do_train --do_test \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --depth 0 --hidden_size 300 --learning_rate 1e-3 \
    --early_stop --checkpoint
```

### Inference
```bash
# Batch inference on all datasets
bash infer_all.sh

# Single molecule inference (programmatic)
python -c "from infer_one import Proposer; p = Proposer(); print(p.propose(['CCO']))"
```

### Evaluation
```bash
python eval_accuracy.py   # Top-k exact match accuracy from pre-computed predictions
python count_params.py    # Model parameter count
```

## Architecture

### Data Flow
```
USPTO-50K reactions → prepare_data.py → ECFP4 fingerprints (1M dims)
                                      → variance threshold → 32,681 dims
                                      → sparse CSR matrices (.npz)
                                      → template labels (.npy)
```

### Model Pipeline
```
Product SMILES → fingerprint (32,681 dims) → TemplateNN_Highway → softmax
              → top-K template indices → RDChiral template application
              → predicted precursor SMILES
```

### Key Components
- **model.py**: `TemplateNN_Highway` (Highway-ELU blocks), `TemplateNN_FC` (simpler alternative)
- **dataset.py**: `FingerprintDataset` - PyTorch Dataset loading sparse CSR matrices
- **train.py**: Training loop with CrossEntropyLoss, Adam, ReduceLROnPlateau, early stopping
- **infer_one.py**: `Proposer` class for inference API
- **infer_all.py**: Batch predictions + exact match accuracy calculation (`calc_accs()`)

### Data Files (in `data/`)
- `50k_training_templates`: Reaction template patterns with counts
- `*_prod_fps_{train|valid|test}.npz`: Sparse fingerprint matrices
- `*_labels_{train|valid|test}.npy`: Template index labels
- `*_csv_{train|valid|test}.csv`: Reaction metadata

### Model Checkpoints (in `checkpoint/`)
- Format: `.pth.tar` containing `state_dict`, `optimizer`, accuracies, losses
- Pre-trained model available on Google Drive (see README.md)

## Default Configuration

- Input: 32,681-dim ECFP4 fingerprints (radius=2, log-transformed)
- Hidden size: 300, Highway depth: 0
- Output: ~10,198 template classes
- Expected accuracy: ~45.5% top-1, ~81.6% top-10, ~87.4% top-50 (exact match)
