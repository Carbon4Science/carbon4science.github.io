# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning four core tasks (**retrosynthesis**, **molecule generation**, **material generation**, and **machine learning interatomic potentials**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10-100x less compute**.

## Tasks

| Task | Directory | Leader | Status |
|------|-----------|--------|--------|
| Retrosynthesis | `Retrosynthesis/` | Shuan Chen | In Progress |
| Molecule Generation | `MolGen/` | Gunwook Nam | Planned |
| Material Generation | `MatGen/` | Junkil Park | Planned |
| ML Interatomic Potentials | `MLIP/` | Junyoung Choi | Planned |

---

## Guide for Task Leaders

Each task leader is responsible for benchmarking models in their domain. **Use Claude Code** to accelerate the process. The workflow is the same for every task:

### Step 1: Set Up Your Task Directory

Your task directory should follow this structure:

```
<Task>/
├── README.md           # Task description, metrics, models, results table
├── evaluate.py         # Evaluation module (metrics + test data loading)
├── data/               # Test datasets
├── <Model1>/
│   ├── Inference.py    # Uniform interface (must implement run())
│   ├── environment.yml # Conda environment
│   ├── CLAUDE.md       # Model-specific guidance for Claude Code
│   └── models/         # Checkpoints (gitignored)
├── <Model2>/
│   └── ...
└── ...
```

See `Retrosynthesis/` for a complete reference implementation.

### Step 2: Implement the Evaluation Module

Create `<Task>/evaluate.py` with:

```python
METRICS = ["metric_1", "metric_2", ...]  # Available metrics

def load_test_data(data_path=None, limit=None):
    """Load test dataset. Returns list of dicts."""
    ...

def evaluate(predictions, test_cases, metrics=None):
    """Compute metrics. Returns dict of metric_name -> score."""
    ...
```

### Step 3: Add Models with Uniform Interface

Each model must implement `Inference.py` with a `run()` function:

```python
def run(input_data, top_k=10) -> List[Dict]:
    """
    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
```

Each model needs its own conda environment (`environment.yml`) to avoid dependency conflicts.

### Step 4: Register in the Benchmark Runner

Update these files to include your task and models:

1. `benchmarks/run_benchmark.py` - Add task and model mappings
2. `benchmarks/run.sh` - Add conda environment mappings
3. `benchmarks/setup_envs.sh` - Add environment setup functions
4. `benchmarks/configs/models.yaml` - Add model configurations

### Step 5: Run Benchmarks with Carbon Tracking

```bash
# Run a single model
./benchmarks/run.sh --model <ModelName> --limit 1000 --track_carbon

# Run all models for your task
./benchmarks/run.sh --model all --limit 1000 --track_carbon
```

### Step 6: Report Results

Update your task's `README.md` with the results table:

```markdown
| Model | Params | Metric-1 | Metric-2 | Duration (s) | Energy (Wh) | CO2 (g) | Peak GPU (MB) |
```

---

## Getting Started

### Prerequisites

- Linux with NVIDIA GPU(s)
- Conda (Miniconda or Anaconda)
- Git

### Clone and Setup

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science

# Setup environments for a specific task
cd benchmarks
./setup_envs.sh            # All retrosynthesis models
./setup_envs.sh neuralsym  # Single model
```

### Run a Benchmark

```bash
cd benchmarks

# Single model with carbon tracking
./run.sh --model neuralsym --limit 100 --track_carbon

# All models
./run.sh --model all --limit 1000 --track_carbon --output results/benchmark.json
```

---

## Repository Structure

```
Carbon4Science/
├── README.md                 # This file
├── CLAUDE.md                 # Instructions for Claude Code
├── .claude/skills/           # Claude Code skills (add-model, benchmark, evaluate)
│
├── benchmarks/               # Shared benchmark infrastructure
│   ├── run.sh               # Unified runner (handles conda envs)
│   ├── run_benchmark.py     # Python benchmark runner
│   ├── carbon_tracker.py    # Carbon/energy measurement
│   ├── setup_envs.sh        # Environment setup
│   ├── configs/             # Model configs, hardware specs
│   └── results/             # Benchmark outputs (JSON)
│
├── Retrosynthesis/          # Retrosynthesis task (Shuan Chen)
│   ├── neuralsym/           # Template-based, Nature 2018
│   ├── LocalRetro/          # MPNN + attention, JACS Au 2021
│   ├── RetroBridge/         # Markov bridges, ICLR 2024
│   ├── Chemformer/          # BART transformer, ML:ST 2022
│   └── RSGPT/               # GPT 1.6B params, Nat. Comm. 2025
│
├── MolGen/                  # Molecule generation (Gunwook Nam)
├── MatGen/                  # Material generation (Junkil Park)
└── MLIP/                    # ML interatomic potentials (Junyoung Choi)
```

---

## Using Claude Code

This repository is designed to work with [Claude Code](https://claude.ai/code). Each task directory includes a `CLAUDE.md` file with model-specific instructions. Claude Code skills are available:

- `/add-model <Task> <ModelName>` - Step-by-step guide to add a new model
- `/benchmark <ModelName>` - Run a carbon-tracked benchmark
- `/evaluate <Task>` - Run evaluation on predictions

To get started with Claude Code on your task:

```bash
cd Carbon4Science
claude  # Launch Claude Code
```

Then tell Claude what you want to do, e.g.:
- "Add a new model called DiffSBDD to MolGen"
- "Run benchmark for all MatGen models with 1000 samples"
- "Set up the MLIP evaluation pipeline"

---

## Citation

```bibtex
@article{carbon2025,
  title={The Carbon Cost of Generative AI for Science},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
