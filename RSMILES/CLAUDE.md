# Root-aligned SMILES (R-SMILES) - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> product)
- **Architecture**: Transformer (seq2seq with root-aligned SMILES)
- **Year**: 2022
- **Venue**: Chem. Sci.
- **Parameters**: ~30M
- **Conda Environment**: `rsmiles`
- **Dataset**: USPTO-480k (USPTO-MIT)

## Reference
Zhong et al., "Root-aligned SMILES: A Tight Representation for Chemical Reaction Prediction", Chem. Sci., 2022.
GitHub: https://github.com/otori-bird/retrosynthesis

## Setup
```bash
# Clone the repo
git clone https://github.com/otori-bird/retrosynthesis.git repo

# Download pretrained forward checkpoint from Google Drive
# Forward (RtoP) configs: pretrain_finetune/finetune/RtoP/

# Translate config example:
onmt_translate -config pretrain_finetune/finetune/RtoP/RtoP-MIT-separated-aug5-translate.yml
```

## Dependencies
- OpenNMT-py 2.2.0, PyTorch 1.6+, RDKit, pandas, textdistance

## Usage
```python
from Forward.RSMILES.Inference import load_model, run

# 1x augmentation (fast, lower accuracy)
load_model("Forward/RSMILES/checkpoint/model.pt", augmentation_factor=1)
results = run("CCO.CC(=O)Cl", top_k=5)

# 20x augmentation (slower, higher accuracy)
load_model("Forward/RSMILES/checkpoint/model.pt", augmentation_factor=20)
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Benchmark Variants

- **RSMILES_1x:** No test-time augmentation (augmentation_factor=1)
- **RSMILES_20x:** Full test-time augmentation (augmentation_factor=20)

This demonstrates the CO2-accuracy tradeoff: 20x augmentation improves accuracy but uses ~20x more compute/energy.

## Notes
- Uses root-aligned SMILES augmentation for better reaction prediction
- RtoP (Reactant-to-Product) mode for forward prediction
- Supports test-time augmentation (multiple SMILES roots)
- Built on OpenNMT-py 2.2
- Checkpoint: `checkpoint/model.pt`
