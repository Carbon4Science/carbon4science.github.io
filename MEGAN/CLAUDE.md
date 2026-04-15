# MEGAN - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> product)
- **Architecture**: Graph Edit Network (GNN + attention)
- **Year**: 2021
- **Venue**: JCIM
- **Parameters**: ~10M
- **Conda Environment**: `megan`
- **Dataset**: USPTO-480k (USPTO-MIT)

## Reference
Sacha et al., "Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits", JCIM, 2021.
GitHub: https://github.com/molecule-one/megan

## Setup
```bash
# Clone the repo into this directory
git clone https://github.com/molecule-one/megan.git repo

# Download pretrained models from GitHub Release v1.1
# Forward prediction uses config: megan_for_8_dfs_cano

# Featurize data for forward prediction
python bin/featurize.py --config megan_for_8_dfs_cano
```

## Dependencies
- Python 3.6.10, PyTorch 1.3.1, RDKit 2020.03.2
- gin-config, argh, torchtext, scipy, numpy, pandas

## Usage
```python
from Forward.MEGAN.Inference import run
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Notes
- Models reactions as sequences of graph edits (bond breaking/forming)
- Forward mode uses `forward=True` flag in featurizer config
- Beam search over graph edit sequences
- Checkpoint: `checkpoint/model.pt`
