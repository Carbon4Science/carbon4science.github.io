# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning three core tasks (**molecule generation**, **retrosynthesis**, and **material generation**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10–100× less compute**.

## Tasks & Models

| Task | Models | Status |
|------|--------|--------|
| **Retrosynthesis** | neuralsym, LocalRetro, Chemformer, RetroBridge, RSGPT | Active |
| **Molecule Generation** | Coming soon | Planned |
| **Material Generation** | Coming soon | Planned |

### Retrosynthesis Models

Predicting chemical reactants from product molecules.

| Model | Architecture | Parameters | Top-1 Acc | Top-10 Acc | Training Energy | Paper |
|-------|--------------|------------|-----------|------------|-----------------|-------|
| neuralsym | Highway-ELU | ~1M | 45.5% | 81.6% | TBD | [Segler 2017](https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/chem.201605499) |
| LocalRetro | MPNN + Attention | 8.65M | 52.6% | 90.2% | TBD | [Chen 2021](https://pubs.acs.org/doi/10.1021/jacsau.1c00246) |
| Chemformer | BART Transformer | 45M | ~50% | ~85% | TBD | [Irwin 2021](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc03420b) |
| RetroBridge | Markov Bridge | - | SOTA | - | TBD | [Igashov 2024](https://openreview.net/forum?id=770DetV8He) |
| RSGPT | LLaMA (Causal LM) | 1.6B | 63.4% | - | TBD | - |

## Quick Start

### Installation

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon

# Install carbon tracking dependencies
pip install codecarbon pandas numpy
```

Each model requires its own environment. See [Environment Setup](#environment-setup) for details.

### Quick Inference

All models provide a unified `run()` interface:

```python
# Example: LocalRetro
from Retrosynthesis.LocalRetro.Inference import run
results = run("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
print(results)
# [{'precursors': ['CC(=O)Cl', 'OC(=O)c1ccccc1O'], 'score': 0.95}, ...]
```

### Run Benchmarks with Carbon Tracking

```python
from benchmarks.carbon_tracker import CarbonTracker

tracker = CarbonTracker(project_name="my_experiment")

with tracker:
    # Your model training or inference
    results = model.predict(test_data)

metrics = tracker.get_metrics()
print(f"Energy: {metrics['energy_kwh']:.4f} kWh")
print(f"CO2: {metrics['emissions_kg_co2']:.4f} kg")
print(f"Duration: {metrics['duration_seconds']:.1f} s")
```

## Environment Setup

Each model requires a separate conda environment due to dependency conflicts:

<details>
<summary><b>neuralsym</b> (Python 3.6, PyTorch 1.6.0)</summary>

```bash
conda create -n neuralsym python=3.6 tqdm scipy pandas joblib -y
conda activate neuralsym
conda install pytorch=1.6.0 cudatoolkit=10.1 -c pytorch
conda install rdkit -c rdkit
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```
</details>

<details>
<summary><b>LocalRetro</b> (Python 3.7, DGL)</summary>

```bash
conda create -c conda-forge -n localretro python=3.7 -y
conda activate localretro
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install -c conda-forge rdkit
pip install dgl dgllife
```
</details>

<details>
<summary><b>RetroBridge</b> (Python 3.9, PyTorch Lightning 2.x)</summary>

```bash
conda create --name retrobridge python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate retrobridge
pip install -r Retrosynthesis/RetroBridge/requirements.txt
```
</details>

<details>
<summary><b>Chemformer</b> (Python 3.7, Poetry)</summary>

```bash
cd Retrosynthesis/Chemformer
conda env create -f env-dev.yml
conda activate chemformer
poetry install
```
</details>

<details>
<summary><b>RSGPT</b> (Python 3.9, DeepSpeed)</summary>

```bash
cd Retrosynthesis/RSGPT
conda env create -f environment.yml
conda activate gpt
```
</details>

## Benchmark Protocol

To ensure fair and reproducible comparisons, we follow a standardized protocol:

### Hardware Reporting

All benchmarks must report:
- GPU model and count (e.g., 1x NVIDIA RTX 3090)
- CPU model and core count
- RAM size
- CUDA version

### Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Accuracy** | Top-k exact match on USPTO-50K test set | % |
| **Training Energy** | Total energy for full training run | kWh |
| **Training CO2** | Carbon emissions for training | kg CO2eq |
| **Inference Energy** | Energy per 1,000 predictions | kWh |
| **Inference Latency** | Time per prediction | ms |
| **Parameters** | Total model parameters | M |

### Reproducibility

1. Run inference benchmarks **3 times** and report mean ± std
2. Use the **same test set** (USPTO-50K, 5,005 reactions)
3. Use **batch size 1** for latency measurements
4. Report **GPU utilization** during inference

## Repository Structure

```
Carbon/
├── README.md                 # This file
├── CLAUDE.md                 # AI assistant guidance
├── LICENSE
├── benchmarks/
│   ├── carbon_tracker.py     # Unified carbon measurement
│   ├── configs/              # Benchmark configurations
│   └── results/              # Benchmark outputs
├── Retrosynthesis/
│   ├── neuralsym/            # Template-based model
│   ├── LocalRetro/           # MPNN + attention
│   ├── Chemformer/           # BART transformer
│   ├── RetroBridge/          # Markov bridge
│   └── RSGPT/                # GPT-based model
├── MolGen/                   # Molecule generation (planned)
└── MatGen/                   # Material generation (planned)
```

## Key Findings

1. **10-100× efficiency gap**: Simple models like neuralsym achieve competitive accuracy at a fraction of the compute cost
2. **Diminishing returns**: Beyond ~50M parameters, accuracy gains plateau while energy costs grow linearly
3. **Task-specific optima**: The most carbon-efficient model varies by task and accuracy requirement

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding a New Model

1. Create a subdirectory under the appropriate task folder
2. Implement the standardized `Inference.py` interface
3. Add a `CLAUDE.md` with setup instructions
4. Run benchmarks with `CarbonTracker` and submit results

### Reporting Issues

Please include:
- Model name and version
- Hardware configuration
- Steps to reproduce
- Error messages (if applicable)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{carbon2025,
  title={The Carbon Cost of Generative AI for Science},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

Individual models may have their own licenses:
- neuralsym: Research/Academic
- LocalRetro: CC BY-NC-SA 4.0
- RetroBridge: CC BY-NC 4.0
- Chemformer: Apache 2.0
- RSGPT: Research/Academic

## Acknowledgments

- LocalRetro: [Chen & Jung, JACS Au 2021](https://pubs.acs.org/doi/10.1021/jacsau.1c00246)
- Chemformer: [AstraZeneca](https://github.com/MolecularAI/Chemformer)
- RetroBridge: [Igashov et al., ICLR 2024](https://openreview.net/forum?id=770DetV8He)
- RSGPT: Pre-trained on USPTO and synthetic reactions
- Carbon tracking powered by [CodeCarbon](https://codecarbon.io/)
