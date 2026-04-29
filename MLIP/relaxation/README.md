# MLIP Structure Relaxation Benchmark

Benchmarks the actual cost of structure relaxation for each MLIP, replacing the
previous proxy (MD cost-per-step × 1000). Accuracy is **not** computed here —
the Matbench Discovery CPS values in `MLIP/evaluate.py` are still used for the
score axis.

## Pipeline

```
prepare_data.py        →  data/wbm_subset_100.extxyz   (one-time, in `matbench` env)
run_relaxation.py      →  results/unified/<Model>/...   (8 models, one shared protocol)
<Model>/Relax.py       →  results/specific/<Model>/...  (NequIP, Nequix, eSEN only)
```

## Phase 1 — One-time data prep

Create a dedicated env that has access to `matbench-discovery`:

```bash
conda create -n matbench python=3.11 -y
conda activate matbench
pip install matbench-discovery ase pandas pymatgen
python MLIP/relaxation/prepare_data.py --n 100 --seed 42
```

This samples 100 WBM structures (`unique_prototype == True`) and writes:
- `data/wbm_subset_100.extxyz`
- `data/wbm_subset_100_ids.json`

Each model run reads the extxyz directly, so per-model envs don't need the
`matbench-discovery` package.

## Phase 2 — Unified-protocol benchmark (all 8 models)

Protocol: **FIRE + FrechetCellFilter + fmax=0.05 + max_steps=500**.

```bash
# Single model
sbatch --job-name=CHGNet_relax MLIP/relaxation/slurm_relax.sh CHGNet

# All 8 models sequentially
sbatch --job-name=all_relax MLIP/relaxation/slurm_relax.sh all
```

Slurm wrapper auto-switches conda env via `MODEL_ENVS`. Per-model output:

```
results/unified/<Model>/
├── relax_results.json       # aggregate metadata + per-structure summary + carbon
├── final/<material_id>.xyz  # final relaxed structure (extxyz)
└── traces/<material_id>.npz # per-step energies (n_steps,) and forces (n_steps, n_atoms, 3)
```

A single `CarbonTracker` wraps the entire 100-structure loop.

## Phase 3 — Model-specific protocol (NequIP, Nequix, eSEN)

These three models use non-default protocols upstream in MBD. To compare the
unified vs. their original protocol later, run their per-model scripts:

| Model | Optimizer | fmax | max_steps |
|-------|-----------|------|-----------|
| NequIP | GOQN | 0.005 | 500 |
| Nequix | FIRE | 0.02 | 500 |
| eSEN | FIRE | 0.02 | 500 |

```bash
sbatch --job-name=NequIP_specific MLIP/relaxation/slurm_relax.sh NequIP --specific
sbatch --job-name=Nequix_specific MLIP/relaxation/slurm_relax.sh Nequix --specific
sbatch --job-name=eSEN_specific   MLIP/relaxation/slurm_relax.sh eSEN   --specific
```

Output goes to `results/specific/<Model>/` so it can be diffed against
`results/unified/<Model>/`.

## Unified-protocol Results (100 WBM structures, FIRE + FrechetCellFilter, fmax=0.05, max_steps=500)

### Per-structure cost (averaged over 100 structures)

| Model | Conv. Rate | Mean Steps | Max Steps | Time/struct (s) | ms/step | Wh/struct | g CO₂/struct | Peak GPU (MB) |
|-------|-----------:|-----------:|----------:|----------------:|--------:|----------:|-------------:|--------------:|
| CHGNet | 96/100 | 29.5 | 236 | 0.84 | 28.3 | 0.0349 | 0.01504 | 398 |
| MACE | 97/100 | 28.8 | 98 | 2.03 | 70.7 | 0.0830 | 0.03574 | 307 |
| SevenNet | 97/100 | 30.2 | 112 | 1.55 | 51.4 | 0.0667 | 0.02874 | 268 |
| ORB | 97/100 | 18.1 | 51 | 0.33 | 18.0 | 0.0135 | 0.00582 | 113 |
| NequIP | 97/100 | 31.9 | 379 | 0.47 | 14.7 | 0.0250 | 0.01075 | 380 |
| DPA3 | 99/100 | 27.7 | 132 | 3.75 | 135.5 | 0.1558 | 0.06707 | 1377 |
| Nequix | 95/100 | 26.7 | 112 | 1.21 | 45.3 | 0.0514 | 0.02212 | 179 |
| eSEN | 100/100 | 30.3 | 321 | 7.58 | 250.1 | 0.3219 | 0.13862 | 1940 |

### Total over the 100-structure loop

| Model | Total Steps | Loop (s) | Energy (Wh) | CO₂ (g) |
|-------|------------:|---------:|------------:|--------:|
| CHGNet | 2,950 | 83.5 | 3.49 | 1.504 |
| MACE | 2,877 | 203.4 | 8.30 | 3.574 |
| SevenNet | 3,019 | 155.2 | 6.67 | 2.874 |
| ORB | 1,805 | 32.5 | 1.35 | 0.582 |
| NequIP | 3,189 | 46.8 | 2.50 | 1.075 |
| DPA3 | 2,767 | 375.0 | 15.58 | 6.707 |
| Nequix | 2,672 | 121.0 | 5.14 | 2.212 |
| eSEN | 3,031 | 758.0 | 32.19 | 13.862 |

### Notes

- Per-step inference cost spans **14.7 → 250 ms/step** (≈17×). End-to-end
  per-structure time spans **0.33 → 7.58 s** (≈23×). Mean step counts are all in
  the 18 – 32 range, so the total-time ranking is driven by per-step cost rather
  than convergence behaviour.
- Convergence rates are 95 – 100% across all models on this 100-structure
  subset; eSEN converges every structure.
- NequIP's max-step count is 379, much higher than the other models (98 – 321);
  a few structures hit oscillations that this protocol's tolerance accepts.
- Memory splits into a light group (ORB / Nequix / NequIP, ≲ 400 MB) and a
  heavy group (DPA3 1.4 GB, eSEN 1.9 GB).

## Phase 4 — Plotting

```bash
# CPS plotted against MD cost (existing default)
python MLIP/benchmarks/plot_results.py --combined

# CPS plotted against measured relax cost
python MLIP/benchmarks/plot_results.py --combined --cost_source relax

# Per-100-relax normalized
python MLIP/benchmarks/plot_results.py --combined --cost_source relax --relax_norm 100

# Use the model-specific results bucket instead of unified
python MLIP/benchmarks/plot_results.py --combined --cost_source relax --relax_bucket specific
```

Note: `--cost_source relax` only changes the **CPS** point; RDF and MSD always
use MD cost (those metrics come from the MD trajectory).
