# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning six tasks across four domains (**retrosynthesis**, **forward reaction prediction**, **molecule generation**, **material generation**, **structure optimization**, and **MD simulation**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10-100x less compute**.

## Tasks

| Task | Directory | Leader | Status |
|------|-----------|--------|--------|
| Retrosynthesis | `Retro/` | Shuan Chen | Complete |
| Forward Reaction Prediction | `Forward/` | Shuan Chen | Complete |
| Molecule Generation | `MolGen/` | Gunwook Nam | Complete |
| Material Generation | `MatGen/` | Junkil Park | Complete |
| Structure Optimization | `MLIP/` | Junyoung Choi | Complete |
| MD Simulation | `MLIP/` | Junyoung Choi | Complete |

---

## Benchmarking Methodology

All tasks follow the same standardized protocol to ensure fair, reproducible comparisons:

1. **Same dataset, same metrics, same hardware** — Every model in a task runs on the same test set, is evaluated with the same metrics, and uses the same GPU hardware (reported in results JSON).
2. **Uniform `Inference.py` interface** — Every model exposes a `run()` function with a standardized return format so the benchmark runner can orchestrate any model identically.
3. **Carbon tracking via CodeCarbon** — Energy consumption (Wh), CO2 emissions (g), and wall-clock time are recorded automatically using our `CarbonTracker` wrapper around [CodeCarbon](https://codecarbon.io/).
4. **Environment isolation** — Each model has its own conda environment to prevent dependency conflicts. The runner script (`run.sh`) activates the correct environment automatically.
5. **Normalized comparison** — Cost metrics are normalized to a fixed sample count (e.g., per 500 samples) so models evaluated on different subset sizes can be compared fairly.
6. **Structured JSON results** — Every benchmark run produces a JSON file with accuracy, carbon, hardware, and metadata fields, enabling automated analysis and plotting.
7. **Accuracy vs Cost visualization** — Results are plotted as accuracy (y-axis) vs cost metric (x-axis, log-scale) to reveal the efficiency frontier across models.

---

## Results Summary

All tasks benchmarked on standardized test sets with full carbon tracking on the same hardware.

**Hardware:** NVIDIA RTX 5000 Ada Generation (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM

### 1. Retrosynthesis (USPTO-50K, 5,007 reactions — metric: Top-50 accuracy)

![Retro: Accuracy vs Carbon Cost](Retro/results/figures/accuracy_vs_carbon_combined.png)

CO₂/call = per molecule; CO₂/task = per 500 molecules (typical retrosynthesis planning session).

| Task | Year | Venue | Model | Architecture | Params | Top-50 | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|--------|-------------|-------------|
| Retro | 2017 | Chem. Eur. J. | neuralsym | MLP | 32.5M | 74.8% | 0.0070 | 3.50 |
| Retro | 2021 | JCIM | MEGAN | GNN | 9.8M | 90.1% | 0.0103 | 5.15 |
| Retro | 2021 | JACS Au | LocalRetro | GNN | 8.6M | 95.6% | 0.0124 | 6.20 |
| Retro | 2022 | Chem. Sci. | RSMILES | LM | 44.6M | 93.0% | 0.2165 | 108.25 |
| Retro | 2022 | ML:ST | Chemformer | LM | 44.7M | 64.0% | 0.5133 | 256.65 |
| Retro | 2024 | COLM | LlaSMol | LLM | ~7.2B | 5.0% | 0.2767 | 138.35 |
| Retro | 2024 | ICLR | RetroBridge | Diffusion | 4.6M | 52.8% | 0.8069 | 403.45 |
| Retro | 2025 | Nat. Commun. | **RSGPT** | LLM | ~1.6B | **97.8%** | 0.5016 | 250.80 |

### 2. Forward Reaction Prediction (USPTO-MIT, 40,029 reactions — metric: Top-3 accuracy)

![Forward: Accuracy vs Carbon Cost](Forward/results/figures/accuracy_vs_carbon_combined.png)

CO₂/call = per molecule; CO₂/task = per 500 molecules (typical forward prediction session).

| Task | Year | Venue | Model | Architecture | Params | Top-3 | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|-------|-------------|-------------|
| Forward | 2017 | Chem. Eur. J. | neuralsym | MLP | 98.1M | 50.6% | 0.0011 | 0.55 |
| Forward | 2019 | ACS Cent. Sci. | MolecularTransformer | LM | 11.7M | 91.7% | 0.0090 | 4.50 |
| Forward | 2021 | JCIM | MEGAN | GNN | 9.9M | 86.4% | 0.0021 | 1.05 |
| Forward | 2022 | Nat. Mach. Intell. | LocalTransform | GNN | 9.1M | 92.1% | 0.0071 | 3.55 |
| Forward | 2022 | Chem. Sci. | **RSMILES** | LM | 44.6M | **94.7%** | 0.0154 | 7.70 |
| Forward | 2024 | COLM | LlaSMol | LLM | ~7.2B | 5.9% | 0.0353 | 17.65 |

### 3. Molecule Generation (ChEMBL 28, 10,000 molecules — metric: VUN%)

| V·U·N vs Carbon Cost | FCD vs Carbon Cost |
|:---------------------:|:------------------:|
| ![VUN vs Carbon](MolGen/results/figures/vun_vs_carbon.png) | ![FCD vs Carbon](MolGen/results/figures/fcd_vs_carbon.png) |

CO₂/call = per molecule; CO₂/task = per 10K molecules (typical generation campaign).

| Task | Year | Venue | Model | Architecture | Params | VUN (%) | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|---------|-------------|-------------|
| MolGen | 2017 | J. Cheminf. | REINVENT | LM | 4.4M | 91.12 | 0.000011 | 0.11 |
| MolGen | 2018 | ICML | JT-VAE | VAE | 7.1M | **99.48** | 0.0020 | 20.00 |
| MolGen | 2020 | ICML | HierVAE | VAE | 8.0M | 93.48 | 0.0014 | 14.00 |
| MolGen | 2021 | J. Chem. Inf. Model. | MolGPT | LM | 6.4M | 93.73 | 0.0002 | 2.00 |
| MolGen | 2023 | ICML | DiGress | Diffusion | 16.2M | 79.85 | 0.0392 | 392.00 |
| MolGen | 2024 | J. Cheminf. | **REINVENT4** | LM | 5.8M | 94.67 | **0.000009** | **0.09** |
| MolGen | 2024 | arXiv | SmileyLlama | LLM | 8.0B | 94.06 | 0.0023 | 23.00 |
| MolGen | 2024 | NeurIPS | DeFoG | Flow Matching | 16.3M | 97.47 | 0.0572 | 572.00 |

### 4. Material Generation (1,000 structures — metric: mSUN %)

<img src="MatGen/results/figures/msun_vs_carbon.png" alt="mSUN vs Carbon" width="60%">

CO₂/call = per structure; CO₂/job = per 1K structures (typical screening campaign).

| Task | Year | Venue | Model | Architecture | Params | mSUN (%) | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|----------|-------------|-------------|
| MatGen | 2022 | ICLR | CDVAE | Diffusion | 4.9M | 22.6 | 0.2704 | 270.40 |
| MatGen | 2023 | NeurIPS | DiffCSP | Diffusion | 12.4M | 29.0 | 0.0126 | 12.60 |
| MatGen | 2024 | Nat. Commun. | CrystaLLM | LM | 25.9M | 16.4 | 0.0192 | 19.20 |
| MatGen | 2024 | ICML | FlowMM | Flow Matching | 28.3M | 23.9 | 0.0128 | 12.80 |
| MatGen | 2024 | NeurIPS | **ChargeDIFF** | Diffusion | 59.5M | **33.5** | 0.1335 | 133.50 |
| MatGen | 2025 | Nature | MatterGen | Diffusion | 44.6M | 33.4 | 0.2481 | 248.10 |
| MatGen | 2025 | ICML | ADiT | Diffusion | 231.9M | 29.6 | 0.1125 | 112.50 |
| MatGen | 2025 | ICML | CrystalFlow | Flow Matching | 20.9M | 21.7 | **0.0015** | **1.50** |

### 5. Structure Optimization (LGPS, 75K steps — metric: CPS)

CO₂/call = per 1K MD steps; CO₂/task = per 1M steps (typical production run).

| Task | Year | Venue | Model | Architecture | Params | CPS | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|-----|-------------|-------------|
| StructOpt | 2023 | Nat. Mach. Intell. | CHGNet | GNN | 413K | 0.343 | 0.379 | 379 |
| StructOpt | 2023 | arXiv | MACE | GNN | 4.69M | 0.637 | 0.932 | 932 |
| StructOpt | 2024 | J. Chem. Theory Comput. | SevenNet | GNN | 1.17M | 0.714 | 0.648 | 648 |
| StructOpt | 2024 | arXiv | ORB | GNN | 25.2M | 0.470 | **0.155** | **155** |
| StructOpt | 2025 | arXiv | **eSEN** | GNN | 30.1M | **0.797** | 3.486 | 3,486 |
| StructOpt | 2025 | arXiv | NequIP | GNN | 9.6M | 0.733 | 0.454 | 454 |
| StructOpt | 2025 | arXiv | DPA3 | GNN | 4.81M | 0.718 | 1.538 | 1,538 |
| StructOpt | 2025 | arXiv | Nequix | GNN | 708K | 0.729 | 0.685 | 685 |

### 6. MD Simulation (LGPS, 75K steps — metric: MSD score)

CO₂/call = per 1K MD steps; CO₂/task = per 1M steps (typical production run).

| Task | Year | Venue | Model | Architecture | Params | MSD | CO₂ eq/call (g) | CO₂ eq/job (g) |
|------|------|-------|-------|-------------|--------|-----|-------------|-------------|
| MDSim | 2023 | Nat. Mach. Intell. | CHGNet | GNN | 413K | 0.047 | 0.379 | 379 |
| MDSim | 2023 | arXiv | MACE | GNN | 4.69M | 0.095 | 0.932 | 932 |
| MDSim | 2024 | J. Chem. Theory Comput. | SevenNet | GNN | 1.17M | 0.531 | 0.648 | 648 |
| MDSim | 2024 | arXiv | ORB | GNN | 25.2M | 0.385 | **0.155** | **155** |
| MDSim | 2025 | arXiv | **eSEN** | GNN | 30.1M | **0.720** | 3.486 | 3,486 |
| MDSim | 2025 | arXiv | NequIP | GNN | 9.6M | 0.361 | 0.454 | 454 |
| MDSim | 2025 | arXiv | DPA3 | GNN | 4.81M | 0.508 | 1.538 | 1,538 |
| MDSim | 2025 | arXiv | Nequix | GNN | 708K | 0.203 | 0.685 | 685 |

### Key Insights Across Tasks

- **MLIP is the most carbon-intensive per call**: A single 1M-step MD simulation costs 155–3,486 g CO₂ eq, orders of magnitude more than chemistry tasks where per-molecule costs are <1g
- **Architecture determines cost, not model size**: Diffusion models cost 10-100x more per call than LM or GNN models due to iterative sampling, regardless of parameter count
- **Larger models do not predict better performance**: Globally r=0.003; in chemistry tasks (Retro, Forward) the correlation is *negative*
- **50-75% of models per task are Pareto-dominated**: Another model exists that is both cheaper and better — the carbon was wasted

---

## For Task Leaders

Each task leader is responsible for benchmarking models in their domain.
Claude Code is the recommended way to do this — it reads the project's
`CLAUDE.md` files automatically and understands the benchmarking protocol,
directory structure, and conventions.

### Prerequisites

- Linux with NVIDIA GPU(s)
- Conda (Miniconda or Anaconda)
- Git
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
  (`npm install -g @anthropic-ai/claude-code`)

### Git Workflow

**Never commit directly to `main`.** Always use feature branches and pull requests.

```
  main (shared timeline)        Your branch (private workspace)
  ─────────────────────         ────────────────────────────────

       ● Retro complete
       │
       │  ① PULL ─ get the latest version
       │          "git pull origin main"
       │
       ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
       │                         │  ② BRANCH ─ create your own copy
       │                         │  "git checkout -b gunwook/molgen"
       │                         │
       │                         ● Add evaluate.py
       │                         │
       │                         ● Add VAE model
       │                         │
       │                         ● Run benchmarks
       │                         │
       │                         │  ③ PUSH ─ upload your branch to GitHub
       │                         │  "git push -u origin gunwook/molgen"
       │                         │
       │  ④ PULL REQUEST         │
       │     "Please review  ◄───┘  "gh pr create ..."
       │      and merge my
       │      changes"
       │
       ●◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ⑤ MERGE ─ your work joins main
       │  (MolGen added!)
       │
       │  ⑥ PULL again before next task
       │
       ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
       │                      │  New branch: junkil/matgen-cdvae
       │                      ● ...
       ▼                      ▼
```

**Key concepts:**
- **Pull** = download the latest changes from `main` to stay up to date
- **Branch** = a private copy where you work without affecting others
- **Push** = upload your branch to GitHub so others can see it
- **Pull Request (PR)** = ask the team to review and merge your branch into `main` (requires push first)
- **Merge** = your branch's changes are added to `main` for everyone

**Commands:**

```bash
# 1. Clone the repo (first time only)
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science

# 2. Pull latest main and create your branch
git checkout main
git pull origin main
git checkout -b <your-name>/<description>
# e.g., git checkout -b gunwook/molgen-setup

# 3. Do your work (add models, run benchmarks, update READMEs)
claude   # Claude Code guides you through the process

# 4. Commit and push
git add <files>
git commit -m "Add VAE model to MolGen benchmarks"
git push -u origin <your-branch-name>

# 5. Create a pull request
gh pr create --title "Add VAE to MolGen" --body "..."

# 6. Before starting new work, always pull latest main
git checkout main
git pull origin main
git checkout -b <your-name>/<next-task>
```

### Quick Start

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
git checkout -b <your-name>/<task>-setup
claude
```

Introduce yourself in the first message:

> I'm [name], the task leader for [task]. I need to set up the [task]
> benchmarking pipeline from scratch. Guide me through the process
> step by step.

Claude will walk you through the full workflow, using `Retro/` as the
reference implementation.

### What You'll Build

Claude Code will guide you through each of these steps:

1. **Task directory** — `<Task>/` with README, evaluate.py, data/, model subdirectories, and `results/{outputs,figures}/`
2. **Benchmark scripts** — `<Task>/benchmarks/` copied from `Retro/benchmarks/` and adapted for your models
3. **Evaluation module** — `<Task>/evaluate.py` with your task's metrics and test data loader
4. **Models** — `<Task>/<Model>/Inference.py` with the uniform `run()` interface, conda environment, and CLAUDE.md for each model
5. **Benchmark runs** — All models run with carbon tracking on the same test set and hardware
6. **Results** — JSON outputs in `<Task>/results/outputs/`, plots in `<Task>/results/figures/`, and a results table in your task README

### Reference

The `Retro/` task is the complete reference implementation. Key files to study:

- `Retro/evaluate.py` — evaluation module structure
- `Retro/LocalRetro/Inference.py` — uniform `run()` interface
- `Retro/LocalRetro/environment.yml` — conda environment spec
- `Retro/benchmarks/configs/models.yaml` — model registration format

### Skills

| Skill | Description |
|-------|-------------|
| `/add-model <Task> <ModelName>` | Add a new model to your task |
| `/benchmark <ModelName>` | Run a carbon-tracked benchmark |
| `/evaluate <Task>` | Evaluate model predictions |
| `/plot <Task>` | Generate accuracy vs cost plots |

### Example Prompts

- "Set up my task directory structure following the Retro template"
- "Write the evaluate.py for MolGen with FCD and validity metrics"
- "Create the Inference.py for CDVAE following the uniform interface"
- "Register my models in the benchmark runner"
- "Run benchmarks for all my models with 1000 samples and carbon tracking"
- "Generate plots and update my README with the results table"
- "What does the Retro/LocalRetro/Inference.py look like? I want to follow the same pattern."

---

## Repository Structure

```
Carbon4Science/
├── README.md                 # This file
├── CLAUDE.md                 # Instructions for Claude Code
├── .claude/skills/           # Claude Code skills (add-model, benchmark, evaluate)
│
├── Retro/                   # Retrosynthesis (Shuan Chen) — 9 models
├── Forward/                 # Forward reaction prediction (Shuan Chen) — 6 models
├── MolGen/                  # Molecule generation (Gunwook Nam) — 8 models
├── MatGen/                  # Material generation (Junkil Park) — 8 models
└── MLIP/                    # ML interatomic potentials (Junyoung Choi) — 8 models
    └── StructOpt + MDSim    # Two sub-tasks evaluated from same runs
```

---

## Citation

```bibtex
@article{carbon2026,
  title={The Carbon Cost of Generative AI for Science},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
