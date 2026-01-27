# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Carbon** is the official repository for the research paper *"The Carbon Cost of Generative AI for Science"* - a benchmarking framework that systematically evaluates both predictive performance AND carbon efficiency of generative AI models for scientific discovery.

**Research Hypothesis**: State-of-the-art performance often comes at disproportionate environmental cost, with order-of-magnitude increases in carbon emissions yielding only marginal accuracy improvements. Simpler, specialized models frequently match or approach SOTA while consuming 10-100× less compute.

**Three Core Tasks**:
1. **Retrosynthesis** (`Retrosynthesis/`) - Predicting reactants from product molecules (5 models implemented)
2. **Molecule Generation** (`MolGen/`) - Generating novel molecules (planned)
3. **Material Generation** (`MatGen/`) - Generating material structures (planned)

**Key Metrics**: All models are evaluated on both accuracy (task-specific) and carbon footprint (energy, emissions, compute time).

## Repository Structure

```
Carbon/
├── benchmarks/        # Carbon measurement infrastructure and results
│   ├── carbon_tracker.py   # Unified tracking wrapper
│   ├── configs/            # Hardware specs, benchmark configs
│   └── results/            # Benchmark outputs (CSV, JSON)
├── Retrosynthesis/
│   ├── neuralsym/     # Template-based, Highway-ELU network (Nature 2018)
│   ├── RetroBridge/   # Markov bridge generative model (ICLR 2024)
│   ├── LocalRetro/    # MPNN + global attention (JACS Au 2021)
│   ├── Chemformer/    # BART transformer pre-trained on SMILES (AstraZeneca)
│   └── RSGPT/         # GPT-based, 1.6B params, 10B synthetic reactions
├── MolGen/            # [Planned] Molecule generation models
└── MatGen/            # [Planned] Material generation models
```

Each module is self-contained with its own environment, dependencies, and CLAUDE.md file with module-specific details.

## Quick Inference (All Modules)

All modules provide a `run()` function in `Inference.py`:

```python
# neuralsym
from Retrosynthesis.neuralsym.Inference import run
results = run("CCO", topk=10)

# RetroBridge
from Retrosynthesis.RetroBridge.Inference import run
results = run("CCO", n_samples=10, n_steps=500)

# LocalRetro
from Retrosynthesis.LocalRetro.Inference import run
results = run("CCO", top_k=10)

# Chemformer
from Retrosynthesis.Chemformer.Inference import load_model, run
load_model(model_path="...", vocab_path="...")
results = run("CCO")

# RSGPT
from Retrosynthesis.RSGPT.inference import run
results = run("CCO")
```

## Environment Setup by Module

Each module requires its own conda environment due to different Python/PyTorch versions:

| Module | Python | PyTorch | Key Deps |
|--------|--------|---------|----------|
| neuralsym | 3.6 | 1.6.0 | rdkit, rdchiral |
| LocalRetro | 3.7 | 1.x | rdkit, dgl, dgllife |
| RetroBridge | 3.9 | 1.13.0 | rdkit, pytorch-lightning 2.2 |
| Chemformer | 3.7.11 | 1.8.1 | rdkit, pytorch-lightning 1.2.3, poetry |
| RSGPT | 3.9 | 2.1.0 | transformers, accelerate, deepspeed |

```bash
# neuralsym
conda create -n neuralsym python=3.6 pytorch=1.6.0 cudatoolkit=10.1 rdkit -c pytorch -c rdkit
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"

# LocalRetro
conda create -c conda-forge -n rdenv python=3.7
conda install pytorch cudatoolkit=10.2 rdkit -c pytorch -c conda-forge
pip install dgl dgllife

# RetroBridge
conda create --name retrobridge python=3.9 rdkit=2023.09.5 -c conda-forge
pip install -r requirements.txt

# Chemformer
conda env create -f env-dev.yml && poetry install

# RSGPT
conda env create -f environment.yml
```

## Common Commands

### neuralsym
```bash
python prepare_data.py                    # Extract templates, generate fingerprints
bash -i train.sh                          # Train (~5 min on RTX2080)
python infer_all.py                       # Batch inference
python eval_accuracy.py                   # Evaluation metrics
```

### LocalRetro
```bash
cd preprocessing
python Extract_from_train_data.py -d USPTO_50K    # Extract local templates
python Run_preprocessing.py -d USPTO_50K          # Preprocess data
cd ../scripts
python Train.py -d USPTO_50K -g cuda:0            # Train
python Test.py -d USPTO_50K -g cuda:0             # Inference
python Decode_predictions.py -d USPTO_50K         # Decode to SMILES
```

### RetroBridge
```bash
wget https://zenodo.org/record/10688201/files/retrobridge.ckpt -O models/retrobridge.ckpt
python predict.py --smiles "CCO" --checkpoint models/retrobridge.ckpt
python train.py --config configs/retrobridge.yaml --model RetroBridge
python sample.py --config configs/retrobridge.yaml --checkpoint models/retrobridge.ckpt
```

### Chemformer
```bash
python -m molbart.fine_tune data_path=data.csv model_path=model.ckpt task=backward_prediction
python -m molbart.inference_score data_path=data.csv
pytest tests/                             # Run tests
black .                                   # Format code
```

### RSGPT
```bash
# Multi-GPU training with DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file deepspeed.yaml train.py
python infer.py                           # Single molecule
python test.py                            # Dataset testing
```

## Key Dependencies

- **Chemistry**: `rdkit` (molecular toolkit), `rdchiral` (template application)
- **Deep Learning**: `torch`, `pytorch-lightning`, `transformers`, `dgl`
- **Configuration**: `hydra-core` (Chemformer, RetroBridge)
- **Distributed**: `accelerate`, `deepspeed` (RSGPT)

## Data

Primary dataset: **USPTO-50K** (39,713 train / 4,989 valid / 5,005 test reactions)

Pre-trained models available:
- neuralsym: Google Drive (see README.md)
- RetroBridge: https://zenodo.org/record/10688201
- RSGPT: https://sandbox.zenodo.org/records/203391

## Carbon Measurement

Use the unified `CarbonTracker` wrapper in `benchmarks/carbon_tracker.py`:

```python
from benchmarks.carbon_tracker import CarbonTracker

tracker = CarbonTracker(
    project_name="retrosynthesis_neuralsym",
    output_dir="benchmarks/results"
)

with tracker:
    # Your training or inference code
    model.train()

# Access metrics
print(tracker.get_metrics())  # energy_kwh, emissions_kg_co2, duration_seconds
```

**Standardized Protocol** (see `benchmarks/README.md`):
- Report hardware specs (GPU model, CPU, RAM)
- Use consistent batch sizes where possible
- Run inference benchmarks 3× and report mean ± std
- Track both training and inference costs separately

## Benchmarks (USPTO-50K)

| Model | Params | Top-1 | Top-10 | Training Energy | Inference Energy/1K |
|-------|--------|-------|--------|-----------------|---------------------|
| neuralsym | ~1M | 45.5% | 81.6% | TBD | TBD |
| LocalRetro | 8.65M | 52.6% | 90.2% | TBD | TBD |
| Chemformer | 45M | ~50% | ~85% | TBD | TBD |
| RetroBridge | - | SOTA | - | TBD | TBD |
| RSGPT | 1.6B | 63.4% | - | TBD | TBD |

## Contributing New Models

When adding a new model to any task:

1. Create a subdirectory under the appropriate task folder (e.g., `Retrosynthesis/NewModel/`)
2. Include a module-specific `CLAUDE.md` with environment setup and commands
3. Implement the standardized `Inference.py` with a `run()` function
4. Add carbon tracking to training and inference scripts using `CarbonTracker`
5. Run the standardized benchmark and submit results to `benchmarks/results/`

## Development Notes

- Each module maintains its own conda environment (incompatible PyTorch versions)
- Carbon tracking uses CodeCarbon backend with fallback to manual timing
- All accuracy metrics should be computed on USPTO-50K test set for comparability
- Report both training cost (one-time) and inference cost (per-sample) separately
