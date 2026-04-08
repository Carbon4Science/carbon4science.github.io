# MLIP (Machine Learning Interatomic Potentials)

**Task Leader:** Junyoung Choi

Machine learning interatomic potentials: Predict atomic energies, forces, and stresses for molecular dynamics simulations.

## Metrics

| Metric | Task | Description |
|--------|------|-------------|
| `CPS` | StructOpt | Crystal Phase Score (higher is better) |
| `MSD` | MDSim | Mean Square Displacement score (higher is better) |

## Test Dataset

- **LGPS** (Lithium Germanium Phosphorus Sulfide): mp-696128
- 3 trial runs per model (seeds 42, 43, 44)

## Models

| Model | Year | Params | Environment |
|-------|------|--------|-------------|
| CHGNet | 2023.2 | 413K | `chgnet` |
| MACE | 2023.12 | 4.69M | `mace` |
| SevenNet | 2024.2 | 1.17M | `sevennet` |
| ORB | 2024.10 | 25.2M | `orb` |
| eSEN | 2025.2 | 30.1M | `esen` |
| NequIP | 2025.4 | 9.6M | `nequip` |
| DPA3 | 2025.6 | 4.81M | `dpa3` |
| Nequix | 2025.8 | 708K | `nequix` |

## Results

All models run on the same hardware. Pretrained models evaluated on LGPS MD simulation.

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM*

### StructOpt — Structure Optimization (metric: CPS)

| Model | Params | CPS | Duration (s) | Energy (Wh) | CO₂ (g) |
|-------|--------|-----|--------------|-------------|---------|
| **eSEN** | 30.1M | **0.797** | 2,780 | 202.4 | 87.14 |
| NequIP | 9.6M | 0.733 | 316 | 26.3 | 11.34 |
| Nequix | 708K | 0.729 | 736 | 39.8 | 17.13 |
| DPA3 | 4.81M | 0.718 | 2,087 | 89.3 | 38.45 |
| SevenNet | 1.17M | 0.714 | 790 | 37.6 | 16.21 |
| MACE | 4.69M | 0.637 | 1,068 | 54.1 | 23.29 |
| ORB | 25.2M | 0.470 | 210 | 9.0 | 3.87 |
| CHGNet | 413K | 0.343 | 602 | 22.0 | 9.47 |

### MDSim — Molecular Dynamics Simulation (metric: MSD score)

| Model | Params | MSD | Duration (s) | Energy (Wh) | CO₂ (g) |
|-------|--------|-----|--------------|-------------|---------|
| **eSEN** | 30.1M | **0.720** | 2,780 | 202.4 | 87.14 |
| SevenNet | 1.17M | 0.531 | 790 | 37.6 | 16.21 |
| DPA3 | 4.81M | 0.508 | 2,087 | 89.3 | 38.45 |
| ORB | 25.2M | 0.385 | 210 | 9.0 | 3.87 |
| NequIP | 9.6M | 0.361 | 316 | 26.3 | 11.34 |
| Nequix | 708K | 0.203 | 736 | 39.8 | 17.13 |
| MACE | 4.69M | 0.095 | 1,068 | 54.1 | 23.29 |
| CHGNet | 413K | 0.047 | 602 | 22.0 | 9.47 |

### Carbon Efficiency per 1,000 Steps

| Model | Energy/1K steps (Wh) | CO₂/1K steps (g) |
|-------|----------------------|-------------------|
| ORB | 0.360 | 0.155 |
| CHGNet | 0.880 | 0.379 |
| NequIP | 1.053 | 0.454 |
| SevenNet | 1.506 | 0.648 |
| Nequix | 1.592 | 0.685 |
| MACE | 2.164 | 0.932 |
| DPA3 | 3.572 | 1.538 |
| eSEN | 8.096 | 3.486 |

### Key Observations

- **Best accuracy**: eSEN leads on both CPS (0.797) and MSD (0.720) — but at the highest carbon cost (87 g CO₂)
- **Best efficiency**: ORB is fastest (210 s) and cheapest (3.87 g CO₂) but moderate on CPS (0.470) and MSD (0.385)
- **Best tradeoff**: NequIP achieves strong CPS (0.733) at low cost (11.34 g CO₂) — 2nd best CPS at 7.7x less carbon than eSEN
- **CPS vs MSD divergence**: Models rank differently across tasks — SevenNet is mid-pack on CPS (0.714) but 2nd on MSD (0.531), while NequIP is 2nd on CPS but 5th on MSD

## Adding a New Model

See `/add-model MLIP <ModelName>` skill or `../.claude/skills/add-model.md`.
