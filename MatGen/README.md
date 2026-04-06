# MatGen (Material Generation)

**Task Leader:** Junkil Park

Material generation: Generate novel crystal structures and materials.

## Models

| Model | Paper | Environment | License |
|-------|-------|-------------|---------|
| CDVAE | [Crystal Diffusion Variational Autoencoder for Periodic Material Generation (ICLR 2022)](https://openreview.net/forum?id=03RLpj-tc2) | `cdvae` | MIT |
| DiffCSP | [Crystal Structure Prediction by Joint Equivariant Diffusion (NeurIPS 2023)](https://openreview.net/forum?id=oY3bBwDXGH) | `diffcsp` | MIT |
| CrystalFlow | [CrystalFlow: A Flow-Based Generative Model for Crystalline Materials (ICML 2025)](https://github.com/deepmodeling/CrystalFlow) | `crystalflow` | Apache 2.0 |
| CrystaLLM | [Crystal structure generation with autoregressive large language modeling (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-54639-7) | `crystallm` | MIT |
| FlowMM | [FlowMM: Generating Materials with Riemannian Flow Matching (ICML 2024)](https://proceedings.mlr.press/v235/miller24a.html) | `flowmm` | CC BY-NC 4.0 |
| ChargeDIFF | [ChargeDiff: A Charge-aware Diffusion Model for Crystal Structure Generation (NeurIPS 2024)](https://openreview.net/forum?id=chargediff) | `chargediff` | MIT |
| MatterGen | [MatterGen: a generative model for inorganic materials design (Nature 2025)](https://www.nature.com/articles/s41586-025-08628-5) | `mattergen` | MIT |
| ADiT | [All-Atom Diffusion Transformer for Unified Crystal and Molecule Generation (ICML 2025)](https://github.com/facebookresearch/all-atom-diffusion-transformer) | `adit` | CC BY-NC 4.0 |

## Results

### Benchmark (1,000 structures)

Stability evaluated with MatterSim (meta-stable: e_hull ≤ 0.1 eV/atom). **mSUN** = meta-Stable ∩ Unique ∩ Novel count. **ICER** = CO₂ (g) / mSUN (lower is better).

| Model | Params | mSUN | Duration (s) | Energy (Wh) | CO₂ (g) | ICER |
|-------|--------|------|--------------|-------------|---------|-------------------|
| CDVAE | 4.9M | 210 | 25,764 | 628.04 | 270.41 | - |
| DiffCSP | 12.4M | 273 | 381 | 29.39 | 12.65 | -1.44 |
| CrystaLLM | 25.9M | 94 | 942 | 44.72 | 19.25 | -1.08 |
| FlowMM | 28.3M | 221 | 547 | 29.64 | 12.76 | -1.35 |
| MatterGen | 44.6M | 334 | 8,079 | 576.19 | 248.09 | -0.21 |
| ADiT | 231.9M | 276 | 10,512 | 261.26 | 112.49 | -0.50 |
| CrystalFlow | 20.9M | 203 | 43 | 3.70 | 1.48 | -2.24 |
| ChargeDIFF | 59.5M | 335 | 2,994 | 333.76 | 133.50 | -0.48 |

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM.*


### mSUN vs Carbon Cost

<img src="./results/figures/msun_vs_carbon.png" alt="mSUN vs Carbon" width="50%">

### mSUN vs Energy

<img src="./results/figures/msun_vs_energy.png" alt="mSUN vs Energy" width="50%">

### mSUN vs Speed

<img src="./results/figures/msun_vs_time.png" alt="mSUN vs Speed" width="50%">

### ΔCO₂ vs Year

<img src="./results/figures/delta_co2_vs_year.png" alt="Delta CO2 vs Year" width="50%">

### ICER vs Year

<img src="./results/figures/icer_vs_year.png" alt="ICER vs Year" width="50%">


