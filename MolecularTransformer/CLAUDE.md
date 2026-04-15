# Molecular Transformer - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> product)
- **Architecture**: Transformer (seq2seq)
- **Year**: 2019
- **Venue**: ACS Cent. Sci.
- **Parameters**: ~12M
- **Conda Environment**: `mol_transformer`
- **Dataset**: USPTO-480k (USPTO-MIT)

## Reference
Schwaller et al., "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction", ACS Cent. Sci., 2019.
GitHub: https://github.com/pschwllr/MolecularTransformer

## Setup
```bash
# Clone the repo into this directory
git clone https://github.com/pschwllr/MolecularTransformer.git repo

# Download pretrained checkpoint (USPTO-MIT)
# Available from IBM Box (see repo README)
```

## Dependencies
- Python 3.5+, PyTorch 0.4.1, OpenNMT-py 0.4.1 (old version)
- torchtext 0.3.1, RDKit, tqdm, pandas

## Usage
```python
from Forward.MolecularTransformer.Inference import run
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Notes
- Treats reaction prediction as SMILES-to-SMILES translation
- Uses beam search for top-k predictions
- Tokenizes SMILES into character-level tokens
- Checkpoint: `checkpoint/model.pt`
