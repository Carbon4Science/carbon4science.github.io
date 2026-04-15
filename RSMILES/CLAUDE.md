# CLAUDE.md

## Project Overview

R-SMILES (Root-aligned SMILES) is a retrosynthesis prediction method that uses root-aligned SMILES augmentation with an OpenNMT-py Transformer model. By choosing different root atoms for SMILES generation, it creates multiple aligned input representations that improve prediction accuracy when ensembled.

Reference: Zhong et al. "Root-aligned SMILES: A Tight Representation for Chemical Reaction Prediction" (Chemical Science, 2022)

## Environment

**Conda environment:** `rsmiles`

```bash
conda env create -f environment.yml
conda activate rsmiles
```

**Key Dependencies:**
- Python 3.7
- PyTorch >= 1.6.0
- OpenNMT-py 2.2.0
- RDKit 2020.09.1.0
- pandas 1.3.4
- textdistance 4.2.2

## Model Checkpoint

The pretrained checkpoint is an averaged OpenNMT-py model for USPTO-50K P2R (product-to-reactant) with 20x augmentation.

Download from: https://drive.google.com/drive/folders/1c15h6TNU6MSNXzqB6dQVMWOs2Aae8hs6
Place the checkpoint at: `Retro/RSMILES/model/average_model.pt`

## Inference

```python
from Retro.RSMILES.Inference import load_model, run

# 1x augmentation (fast, lower accuracy)
load_model("Retro/RSMILES/model/average_model.pt", augmentation_factor=1)
results = run("CCO", top_k=10)

# 20x augmentation (slower, higher accuracy)
load_model("Retro/RSMILES/model/average_model.pt", augmentation_factor=20)
results = run("CCO", top_k=10)
```

## Architecture

- **Model:** OpenNMT-py Transformer (seq2seq)
- **Input:** Tokenized product SMILES (space-separated)
- **Output:** Tokenized reactant SMILES
- **Augmentation:** Root atom enumeration via `Chem.MolToSmiles(mol, rootedAtAtom=root)`
- **Scoring:** Frequency-weighted rank aggregation across augmented predictions
- **Expected accuracy (USPTO-50K):** ~56% top-1 with 20x augmentation

## Benchmark Variants

- **RSMILES_1x:** No test-time augmentation (augmentation_factor=1)
- **RSMILES_20x:** Full test-time augmentation (augmentation_factor=20)

This demonstrates the CO2-accuracy tradeoff: 20x augmentation improves accuracy but uses ~20x more compute/energy.

## Source Repository

Cloned from: https://github.com/otori-bird/retrosynthesis (in `repo/` subdirectory)
