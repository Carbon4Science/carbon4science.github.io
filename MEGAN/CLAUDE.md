# MEGAN - Retrosynthesis Prediction

## Model Info
- **Task**: Retrosynthesis (product -> reactants)
- **Architecture**: Graph Edit Network (GNN + attention)
- **Year**: 2021
- **Venue**: JCIM
- **Parameters**: ~10M
- **Conda Environment**: `megan2`
- **Dataset**: USPTO-50K

## Reference
Sacha et al., "Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits", JCIM, 2021.
GitHub: https://github.com/molecule-one/megan

## Setup
```bash
# Download pretrained models from GitHub Release v1.0
wget https://github.com/molecule-one/megan/releases/download/v1.0/megan_data.zip
unzip megan_data.zip -d repo/
```

## Usage
```python
from Retro.MEGAN.Inference import run
results = run("CCOC(C)=O", top_k=10)
```

## Notes
- MEGAN was originally designed for retrosynthesis
- Pretrained on USPTO-50K (reaction type unknown variant)
- Beam search over graph edit sequences
- Checkpoint: `repo/models/uspto_50k/model_best.pt`
- Config: `repo/configs/uspto_50k.gin`
