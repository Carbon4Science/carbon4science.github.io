# WLDN (rexgen_direct) - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> products)
- **Architecture**: Weisfeiler-Lehman Difference Network (WLN), two-stage graph neural network
- **Year**: 2019
- **Venue**: Chem. Sci. (RSC)
- **Parameters**: ~2.6M total (~757K core finder + ~1,802K candidate ranker)
- **Conda Environment**: `wldn`

## Reference
Coley, Jin, Rogers & Jamison, "A graph-convolutional neural network model for the prediction of chemical reactivity", Chem. Sci., 2019.
GitHub: https://github.com/connorcoley/rexgen_direct

## Architecture

WLDN uses a **two-stage** approach:

1. **Core Finder** (`core_wln_global/directcorefinder.py`): Identifies which bonds in the reactants are likely to change. Uses a WLN with global attention to score all atom pairs for each of 5 bond order outcomes (no bond, single, double, triple, aromatic). Returns top-K candidate bond changes with scores.

2. **Candidate Ranker** (`rank_diff_wln/directcandranker.py`): Takes the candidate bond changes from Stage 1, enumerates possible product candidates by combining bond edits, and ranks them using a difference WLN that compares reactant vs. candidate product representations.

## Pretrained Weights
Pretrained model checkpoints are **included** in the cloned repo:
- Core finder: `repo/rexgen_direct/core_wln_global/model-300-3-direct/model.ckpt-140000`
- Candidate ranker: `repo/rexgen_direct/rank_diff_wln/model-core16-500-3-max150-direct-useScores/model.ckpt-2400000`

## Dependencies
- **Python**: 2.7 or 3.6 (originally trained with 2.7, deployment compatible with 3.6)
- **TensorFlow**: 1.x (trained with 1.3, deployment compatible with 1.6-1.15)
- **RDKit**: 2017.09+
- **NumPy**: 1.12+

### Environment Setup
```bash
source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
conda create -n wldn python=3.6 -y
conda activate wldn
pip install tensorflow==1.15.0
conda install -c conda-forge rdkit=2020.09 -y
pip install numpy
```

**Known issue**: TensorFlow 1.15 requires CUDA 10.x. The cluster has CUDA 12.x+, so GPU inference may not work. CPU-only fallback (`tensorflow==1.15.0` without GPU) is recommended. You may need `CUDA_VISIBLE_DEVICES=""` to force CPU mode.

## Atom Mapping Requirement

**Critical**: WLDN requires atom-mapped SMILES as input. The core finder uses `molAtomMapNumber` properties on atoms.

Our test data (`Forward/data/test.csv`) does **not** have atom mapping. The `DirectCoreFinder.predict()` method has built-in fallback logic: if atoms lack `molAtomMapNumber`, it assigns sequential atom map numbers automatically (lines 96-101 of `directcorefinder.py`). This means unmapped SMILES will work, but the arbitrary mapping may reduce accuracy compared to chemically meaningful atom mapping.

For better results, consider using **RXNMapper** to add atom mapping before inference:
```python
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()
# Map: reactants>>products, then extract mapped reactants
```

## Usage
```python
from Forward.WLDN.Inference import run
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Reported Performance (USPTO-50K, from authors)
- Top-1: 85.6% (strict), 86.2% (MOLVS-sanitized)
- Top-2: 90.5%
- Top-3: 92.1%
- Top-5: 93.4%

## Key Files
- `repo/rexgen_direct/core_wln_global/directcorefinder.py` - Stage 1: reaction center identification
- `repo/rexgen_direct/rank_diff_wln/directcandranker.py` - Stage 2: candidate ranking
- `repo/rexgen_direct/scripts/eval_by_smiles.py` - Contains `edit_mol()` for applying bond edits
- `Inference.py` - Uniform benchmark interface
