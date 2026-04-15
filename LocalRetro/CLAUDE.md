# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalRetro is a deep learning model for retrosynthesis prediction that predicts chemical reactants from product molecules using local reaction templates and global attention mechanisms. Published in JACS Au 2021.

## Environment Setup

```bash
# Create and activate conda environment
conda create -c conda-forge -n rdenv python=3.7 -y
conda activate rdenv
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl dgllife
```

**Dependencies**: Python >=3.6, PyTorch >=1.0.0, RDKit >=2019, DGL >=0.5.2, DGLLife >=0.2.6, NumPy >=1.16.4

## Common Commands

```bash
# Activate environment
source /Users/admin/opt/anaconda3/etc/profile.d/conda.sh && conda activate rdenv

# Step 1: Extract local templates from training data
cd preprocessing
python Extract_from_train_data.py -d USPTO_50K

# Step 2: Assign templates to raw data
python Run_preprocessing.py -d USPTO_50K

# Step 3: Train model
cd ../scripts
python Train.py -d USPTO_50K -g cuda:0

# Step 4: Generate predictions on test set
python Test.py -d USPTO_50K -g cuda:0

# Step 5: Decode predictions to reactant SMILES
python Decode_predictions.py -d USPTO_50K
```

## Architecture

### Data Pipeline
```
Raw Reactions → Extract_from_train_data.py → Local Templates (CSV)
                                                    ↓
                                          Run_preprocessing.py
                                                    ↓
                                          Labeled Dataset → Train.py → Model (.pth)
                                                                           ↓
                                          Test Products → Test.py → Raw Predictions
                                                                           ↓
                                                    Decode_predictions.py → Reactant SMILES
```

### Key Modules

- **scripts/models.py**: LocalRetro neural network (MPNN backbone + global attention)
- **scripts/Train.py**: Training loop with early stopping
- **scripts/Test.py**: Batch inference, outputs raw predictions
- **scripts/Decode_predictions.py**: Converts predictions to reactant SMILES
- **scripts/dataset.py**: USPTODataset and USPTOTestDataset classes
- **LocalTemplate/**: Template extraction and decoding logic
- **Retrosynthesis.py**: User-facing API for single-molecule inference

### Model Architecture

- MPNN with 6 message passing steps
- Node features: 320 dims, Edge hidden: 64 dims
- Global attention: 8 heads, 1 layer, GELU activation
- Outputs: atom_logits (atom template classification) + bond_logits (bond template classification)
- Model size: ~8.65M parameters

### Configuration

Model hyperparameters in `data/configs/default_config.json`:
- `attention_heads`: 8
- `attention_layers`: 1
- `batch_size`: 16
- `node_out_feats`: 320
- `num_step_message_passing`: 6

### Data Files

Templates generated in `data/{DATASET}/`:
- `atom_templates.csv` - Atom modification templates
- `bond_templates.csv` - Bond modification templates
- `template_infos.csv` - Template metadata

Pre-trained model: `models/LocalRetro_USPTO_50K.pth`

## User API Example

```python
from Retro.LocalRetro.Retrosynthesis import LocalRetro

model = LocalRetro()
results = model.retrosnythesis("CCO", top_k=10)  # Returns DataFrame with predicted reactants
```
