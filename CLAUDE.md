# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Rules for Claude Code

### Rule 1: Uniform Inference Interface

When creating or modifying any `Inference.py` file, ALWAYS use this interface:

```python
from typing import List, Dict, Union

def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
```

**NEVER** deviate from this return format. The benchmark runner depends on it.

### Rule 2: Environment Isolation

Each model has its own conda environment. **NEVER** suggest installing packages globally or mixing environments.

**Retro (Retrosynthesis):**

| Model | Environment Name |
|-------|-----------------|
| neuralsym | `neuralsym` |
| LocalRetro | `rdenv` |
| RetroBridge | `retrobridge` |
| Chemformer | `chemformer` |
| RSGPT | `gpt` |
| RSMILES_1x | `rsmiles` |
| RSMILES_20x | `rsmiles` |

Other tasks (MolGen, MatGen, MLIP) follow the same pattern: one conda env per model.

When running models, use: `./benchmarks/run.sh --model <name>` (handles env switching automatically)

### Rule 3: Benchmark Runner

**ALWAYS** use the wrapper scripts instead of running Python directly:

```bash
# Correct
./benchmarks/run.sh --model neuralsym --limit 1000 --track_carbon

# Wrong - will fail due to environment issues
python benchmarks/run_benchmark.py --model neuralsym
```

### Rule 4: Adding New Models

When adding a new model, you MUST update ALL of these files:

1. `<Task>/<ModelName>/Inference.py` - Implement uniform interface
2. `<Task>/<ModelName>/environment.yml` - Dependencies
3. `<Task>/<ModelName>/CLAUDE.md` - Model-specific guidance
4. `benchmarks/run_benchmark.py` - Add to TASKS dict
5. `benchmarks/run.sh` - Add to MODEL_ENVS mapping
6. `benchmarks/setup_envs.sh` - Add setup function
7. `benchmarks/configs/models.yaml` - Add configuration

### Rule 5: Carbon Tracking

When running benchmarks, ALWAYS include `--track_carbon` flag:

```bash
./benchmarks/run.sh --model LocalRetro --limit 1000 --track_carbon
```

### Rule 6: Device Selection

Always let users specify device. Default to `cuda:0` but support CPU fallback.

### Rule 7: No Hardcoded Paths

Use relative paths from repository root. Never hardcode absolute paths in any script.

### Rule 8: Task-Specific Evaluation

Each task defines its own `evaluate.py` module with:
- `METRICS` - list of available metrics
- `load_test_data(data_path, limit)` - load test dataset
- `evaluate(predictions, test_cases, metrics)` - compute metrics

The benchmark runner dynamically loads the correct evaluator based on the `--task` flag.

### Rule 9: Benchmarking Workflow

When a task leader asks to benchmark their task end-to-end, follow these phases in order:

1. **Setup** — Create task directory structure (`<Task>/`, `<Task>/data/`, model subdirectories)
2. **Evaluate** — Implement `<Task>/evaluate.py` with `METRICS`, `load_test_data()`, and `evaluate()`
3. **Models** — For each model, implement `Inference.py` with the uniform `run()` interface, create `environment.yml`, and write `CLAUDE.md`
4. **Register** — Add all models to `benchmarks/run_benchmark.py` (TASKS dict), `benchmarks/run.sh` (MODEL_ENVS), `benchmarks/setup_envs.sh`, `benchmarks/configs/models.yaml`, and `benchmarks/plot_results.py` (MODEL_STYLES)
5. **Run** — Submit benchmarks via Slurm: `sbatch --job-name=<Model> benchmarks/slurm_benchmark.sh <Model>`. Run ALL models on the FULL test set with `--track_carbon`.
6. **Plot** — Generate normalized plots: `python benchmarks/plot_results.py --task <Task> --combined --norm <N>` where N is chosen by the task leader (e.g., 500 for Retro).
7. **Report** — Follow Rule 11 (Reporting Format) exactly:
   - Add three comparison tables to `benchmarks/README.md` (Model Specs, Accuracy, Carbon Efficiency)
   - Add a combined results table (per-N normalized) to the root `README.md`
   - Include the accuracy-vs-carbon plot in the root `README.md`
   - Write 3–5 key observations highlighting accuracy-efficiency tradeoffs

### Rule 10: Plot Generation

Use `benchmarks/plot_results.py` to generate accuracy vs cost plots.

**Commands:**
```bash
# All three plots (carbon, energy, speed) in combined view
python benchmarks/plot_results.py --task <Task> --combined

# Specific sample count
python benchmarks/plot_results.py --task <Task> --combined --samples 500

# Single x-axis metric with custom output
python benchmarks/plot_results.py --task <Task> --combined --xaxis emissions_g_co2 -o my_plot.png

# Per-metric panel view (one subplot per accuracy metric)
python benchmarks/plot_results.py --task <Task>
```

**Adding new models:** When adding a model to a new or existing task, add an entry to `MODEL_STYLES` in `plot_results.py`:
```python
MODEL_STYLES = {
    "MyModel": {"color": "#hex", "marker": "o", "params": "10M", "year": 2024, "venue": "NeurIPS"},
}
```

**Output:** Plots are saved to `benchmarks/figures/<Task>/`.

### Rule 11: Reporting Format

After running all benchmarks for a task, produce the following standardized deliverables. Use the `Retro/` results as the reference.

#### Three Comparison Tables in `benchmarks/README.md`

Add a section `## <Task> Model Comparison` with these three tables:

**Table 1 — Model Specifications:**

| Column | Description |
|--------|-------------|
| Model | Display name |
| Year | Publication year |
| Venue | Publication venue |
| Architecture | Brief architecture description |
| Parameters | Parameter count (e.g., "8.6M", "~1.6B") |
| Model Size | Checkpoint file size on disk |
| GPU Memory (MB) | `peak_gpu_memory_mb` from results JSON |

**Table 2 — Accuracy:** Task-specific metrics. Sort rows by primary metric (descending). Bold the best model.

**Table 3 — Carbon Efficiency:** Sort rows by Duration (ascending, fastest first).

| Column | Source / Formula |
|--------|-----------------|
| Duration (s) | `carbon.duration_seconds` |
| Speed (s/mol) | `duration_seconds / num_samples` |
| Energy (Wh) | `carbon.energy_wh` |
| CO2 (g) | `carbon.emissions_g_co2` |
| CO2 Intensity (g/s) | `emissions_g_co2 / duration_seconds` |

Include a **Key Observations** section (3–5 bullets) highlighting: best accuracy model, most efficient model, CO2 intensity range, and notable tradeoffs.

#### Combined Results Table in Root `README.md`

Add a section `## <Task> Results` with a single merged table. Costs normalized **per N samples** where N is chosen by the task leader (e.g., 500 for Retro):

| Model | Params | Primary Metric | ... | Time/N (s) | Energy/N (Wh) | CO2/N (g) | Peak GPU (MB) |

Formula: `raw_value / num_samples * N`

Include a plot reference: `![<Task>: Accuracy vs Carbon Cost](benchmarks/figures/<Task>/accuracy_vs_carbon_combined.png)`

#### Plots

Generate all plots with the task's chosen normalization:

```bash
python benchmarks/plot_results.py --task <Task> --combined --norm <N>
python benchmarks/plot_results.py --task <Task> --norm <N>
```

Where `<N>` is the per-sample normalization chosen by the task leader. Examples:
- Retro: `--norm 500` (500 molecules)
- MolGen: task leader decides (e.g., `--norm 1000`)
- MLIP: task leader decides (e.g., `--norm 100`)

This creates 6 files in `benchmarks/figures/<Task>/`:
- `accuracy_vs_{carbon,energy,speed}_combined.png`
- `accuracy_vs_{carbon,energy,speed}_panels.png`

### Rule 12: Slurm Job Submission

**ALWAYS** submit benchmark runs via Slurm instead of running them as background processes. Use `benchmarks/slurm_benchmark.sh`:

```bash
# Single model
sbatch --job-name=RSGPT benchmarks/slurm_benchmark.sh RSGPT

# Chemformer with proper test set (pickle)
sbatch --job-name=Chemformer benchmarks/slurm_benchmark.sh Chemformer --data Retro/data/uspto_50_chemforner.pickle

# R-SMILES variants
sbatch --job-name=RSMILES_20x benchmarks/slurm_benchmark.sh RSMILES_20x

# Check job status
squeue -u $USER
```

**Cluster details:**
- Partitions: `5000_ada` (GPU), `6000_ada` (GPU), `cpu_only`
- GPU resource: `--gres=gpu:5000ada:1`
- Max walltime: 72 hours (GPU), 48 hours (CPU)
- Override memory with `--mem=32G` for large models (e.g., RSGPT 1B)

Logs are saved to `benchmarks/logs/<jobname>.o<jobid>`.

**NEVER** run long benchmarks as background shell processes. Always use `sbatch`.

### Rule 13: Git Workflow for Contributors

All contributors must follow this branch-based workflow. **NEVER** commit directly to `main`.

#### Starting New Work

```bash
# 1. Always pull the latest main first
git checkout main
git pull origin main

# 2. Create a feature branch
git checkout -b <your-name>/<short-description>
# Examples:
#   git checkout -b gunwook/molgen-setup
#   git checkout -b junkil/add-cdvae
#   git checkout -b junyoung/mlip-mace-benchmark
```

#### Making Changes

```bash
# 3. Work on your branch — add models, run benchmarks, update READMEs
#    Follow Rules 4, 9, 11 for the full workflow

# 4. Commit your changes (small, focused commits)
git add <specific files>
git commit -m "Add CDVAE model to MatGen benchmarks"

# 5. Push your branch to remote
git push -u origin <your-branch-name>
```

#### Submitting for Review

```bash
# 6. Create a pull request to main
gh pr create --title "Add CDVAE to MatGen" --body "..."
# Or use the GitHub web UI
```

#### Before Starting New Work Again

```bash
# 7. Switch back to main and pull latest
git checkout main
git pull origin main

# 8. Create a new branch for the next task
git checkout -b <your-name>/<next-task>
```

**Key rules:**
- One branch per task/model — don't mix unrelated changes
- Pull `main` before creating each new branch to avoid merge conflicts
- Never force-push to `main`

### Results JSON Schema

Every benchmark run produces a JSON file following this structure:

```json
{
  "task": "Retro",
  "model": "LocalRetro",
  "num_samples": 1000,
  "top_k": 50,
  "model_params": 8645410,
  "metrics": ["top_1", "top_5", "top_10", "top_50"],
  "accuracy": {
    "top_1": 0.525,
    "top_5": 0.839,
    "top_10": 0.923,
    "top_50": 0.977
  },
  "correct": {
    "top_1": 525,
    "top_5": 839,
    "top_10": 923,
    "top_50": 977
  },
  "carbon": {
    "start_time": "2026-02-03T17:49:37.301875",
    "end_time": "2026-02-03T17:56:18.823578",
    "duration_seconds": 401.52,
    "energy_wh": 41.31,
    "emissions_g_co2": 16.52,
    "gpu_energy_wh": 19.06,
    "cpu_energy_wh": 22.20,
    "ram_energy_wh": 0.05,
    "peak_gpu_memory_mb": 154.2,
    "peak_cpu_memory_mb": 1236.7,
    "project_name": "LocalRetro_Retro_benchmark",
    "model_name": "LocalRetro",
    "task": "inference",
    "hardware": {
      "gpu_model": "NVIDIA RTX 5000 Ada Generation",
      "gpu_count": 8,
      "gpu_memory_gb": 31.99,
      "cpu_model": "INTEL(R) XEON(R) PLATINUM 8558",
      "cpu_cores": 192,
      "ram_gb": 503.04,
      "cuda_version": "Driver 580.82.09",
      "platform": "Linux-5.14.0-..."
    }
  }
}
```

Results are saved to `benchmarks/results/<Task>/<model>_<N>.json`.

---

## Project Overview

**Carbon4Science** benchmarks both predictive performance AND carbon efficiency of generative AI models for scientific discovery.

**Four Tasks:**
1. **Retrosynthesis** (`Retro/`) - Predict reactants from product molecules
2. **Molecule Generation** (`MolGen/`) - Generate novel molecules
3. **Material Generation** (`MatGen/`) - Generate crystal structures
4. **ML Interatomic Potentials** (`MLIP/`) - Predict atomic forces and energies

## Repository Structure

```
Carbon4Science/
├── benchmarks/        # Shared infrastructure (runner, carbon tracker, configs)
├── Retro/             # 7 models: neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT, RSMILES_1x, RSMILES_20x
├── MolGen/            # Molecule generation models (planned)
├── MatGen/            # Material generation models (planned)
└── MLIP/              # ML interatomic potential models (planned)
```

Each task directory is self-contained with its own evaluation module, test data, and model subdirectories. Each model subdirectory has its own conda environment and CLAUDE.md.

## Quick Inference (All Tasks)

All models provide a `run()` function in `Inference.py`:

```python
# Example: Retrosynthesis
from Retro.LocalRetro.Inference import run
results = run("CCO", top_k=10)

# Example: MolGen (when implemented)
from MolGen.SomeModel.Inference import run
results = run(num_samples=100)
```

## Common Commands

```bash
# Run benchmark for a single model
./benchmarks/run.sh --model <ModelName> --limit 1000 --track_carbon

# Run all models for a task
./benchmarks/run.sh --model all --limit 1000 --track_carbon

# Setup all environments
./benchmarks/setup_envs.sh

# Setup a specific model's environment
./benchmarks/setup_envs.sh <ModelName>
```

## Carbon Measurement

Use the unified `CarbonTracker` wrapper in `benchmarks/carbon_tracker.py`:

```python
from benchmarks.carbon_tracker import CarbonTracker

tracker = CarbonTracker(
    project_name="retrosynthesis_neuralsym",
    model_name="neuralsym",
    task="inference",
)

tracker.start()
# Your inference code here
tracker.stop()
print(tracker.get_metrics())  # energy_wh, emissions_g_co2, duration_seconds
```

## Standardized Protocol

- Report hardware specs (GPU model, CPU, RAM)
- Track both accuracy metrics and carbon footprint
- Use the same test dataset across all models in a task
- Run inference benchmarks and report mean results
