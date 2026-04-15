# Graph2SMILES

## Model Information

- **Task**: Forward Reaction Prediction (reactants -> products)
- **Paper**: "Graph2SMILES: An End-to-End Approach for Reaction Prediction" (Tu & Coley, 2022, JCIM)
- **Year**: 2022
- **Venue**: JCIM (also ICML 2021 Workshop on Computational Biology)
- **Architecture**: Graph-to-sequence model with Directed Graph Convolutional Network (DGCN) encoder + Transformer decoder
- **Variant**: DGCN (as opposed to DGAT)
- **Dataset**: USPTO_480k (forward reaction prediction)
- **Repository**: https://github.com/coleygroup/Graph2SMILES

## Architecture Details

- **Encoder**: DGCN (Directed Graph Convolutional Network) for molecular graph encoding
- **Attention Encoder**: Transformer-style attention with relative positional encodings based on graph distances
- **Decoder**: Transformer decoder with beam search
- **Input**: Reactant SMILES converted to molecular graphs
- **Output**: Product SMILES generated token-by-token via beam search

## Environment

- **Conda env**: `mol_transformer` (shared with Molecular Transformer; has PyTorch 1.13.1, OpenNMT-py, RDKit)
- **Alternative**: Create dedicated `graph2smiles` env with Python 3.6+, PyTorch, OpenNMT-py==1.2.0, RDKit, networkx, selfies
- **GPU**: Required for reasonable inference speed; at least 8GB VRAM

## File Structure

```
Forward/Graph2SMILES/
├── CLAUDE.md                  # This file
├── Inference.py               # Uniform inference interface
└── repo/                      # Cloned Graph2SMILES repository
    ├── checkpoints/
    │   └── pretrained/
    │       └── USPTO_480k_dgcn.pt   # Pretrained checkpoint (~77MB)
    ├── data/
    │   └── USPTO_480k/              # Raw tokenized data (src/tgt files)
    ├── preprocessed/
    │   └── default_vocab_smiles.txt # Default SMILES vocabulary
    ├── models/                      # Model architecture code
    ├── utils/                       # Data processing, chemistry, training utils
    ├── predict.py                   # Original prediction script
    ├── preprocess.py                # Preprocessing pipeline
    └── scripts/                     # Shell scripts for train/predict/download
```

## Inference

The `Inference.py` implements on-the-fly preprocessing: SMILES are converted to molecular graphs, featurized, batched, and fed to the model without needing pre-saved `.npz` files. This enables single-molecule and arbitrary batch inference.

```python
from Forward.Graph2SMILES.Inference import run

# Single input
results = run("CCO.CC(=O)Cl", top_k=5)

# Batch input
results = run(["CCO.CC(=O)Cl", "c1ccccc1.Cl"], top_k=10)
```

## Key Implementation Notes

- The model expects **reactant SMILES** (multiple reactants separated by `.`) as input
- Graph features include atom features (sparse), bond features (dense), and graph distance matrices
- The chiral tag feature is masked to UNSPECIFIED during preprocessing (matching training behavior)
- For forward prediction, the graph distance computation uses `task="reaction_prediction"` which handles disconnected components
- Beam search is used for decoding; beam_size defaults to max(top_k, 10)
- Predictions are canonicalized and deduplicated before returning

## Preprocessing Pipeline

The original repo requires a two-step process:
1. **Tokenize**: Raw SMILES -> space-separated tokens
2. **Binarize**: Tokenized text -> `.npz` binary files with graph features + token IDs

The `Inference.py` bypasses this by doing graph featurization in-memory, directly from SMILES.

## Checkpoint

- **File**: `repo/checkpoints/pretrained/USPTO_480k_dgcn.pt` (~77 MB)
- **Downloaded via**: `python repo/scripts/download_checkpoints.py --data_name=USPTO_480k --mpn_type=dgcn`
- **Contains**: Model state_dict + training args (architecture config)
