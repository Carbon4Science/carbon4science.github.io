# NeuralSym - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> product)
- **Architecture**: MLP (template-based, Highway network)
- **Year**: 2017
- **Venue**: Chem. Eur. J.
- **Parameters**: ~32M
- **Conda Environment**: `neuralsym`

## Reference
Segler & Waller, "Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction", Chem. Eur. J., 2017.
GitHub (reimplementation): https://github.com/linminhtoo/neuralsym

## Key Differences from Retro Version

This is the **forward prediction** variant. Changes from `Retro/neuralsym/`:

1. **Input**: Reactant fingerprints (left side of `>>`) instead of product fingerprints
2. **Templates**: Forward direction `r_temp >> p_temp` instead of retro `p_temp >> r_temp`
3. **Inference**: Apply templates to reactants to generate products
4. **Data files**: Use `*_rct_fps_*.npz` instead of `*_prod_fps_*.npz`
5. **Template file**: `50k_forward_training_templates` instead of `50k_training_templates`

## Data Preprocessing

Must run before training. Uses USPTO-480k (USPTO-MIT) dataset:
```bash
# Prepare raw reaction SMILES as pickle files in Forward/neuralsym/data/
# Files: 480k_clean_rxnsmi_noreagent_allmapped_canon_{train|valid|test}.pickle

# Generate reactant fingerprints + forward templates
cd Forward/neuralsym
python prepare_data.py
```

This generates:
- `data/480k_1000000dim_2rad_rct_fps_{train|valid|test}.npz` - Raw reactant FPs
- `data/480k_1000000dim_2rad_to_32681_rct_fps_{train|valid|test}.npz` - Variance-reduced FPs
- `data/480k_forward_training_templates` - Forward templates with counts
- `data/480k_1000000dim_2rad_to_32681_labels_{train|valid|test}.npy` - Template labels
- `data/variance_indices.txt` - Feature indices for filtering

## Training

```bash
bash train.sh  # ~5 minutes on RTX 2080
```

## Inference

```python
from Forward.neuralsym.Inference import run
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Architecture

- Input: 32,681-dim ECFP4 fingerprints (radius=2, log-transformed, variance-reduced)
- Hidden: 300-dim Highway network (depth=0)
- Output: ~10K forward template class logits
- Checkpoint: `checkpoint/forward_depth0_dim300_lr1e3_stop2_fac30_pat1.pth.tar`
