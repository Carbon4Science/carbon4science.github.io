# The Carbon Cost of AI for Science

<p align="center">
  <img src="docs/noah.svg" alt="Carbon4Science logo" width="120" />
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" /></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/Paper-arXiv-red.svg" /></a>
  <a href="https://carbon4science.github.io/"><img src="https://img.shields.io/badge/Website-carbon4science-green.svg" /></a>
</p>


A benchmarking framework that jointly evaluates **predictive accuracy** and **carbon footprint** of generative AI models across six scientific discovery tasks.

**Key Finding:** Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10–100× less compute**.

---

## Contents

- [CO₂ Reference Points](#co-reference-points)
- [Results](#results)
  - [1. Retrosynthesis](#1-retrosynthesis)
  - [2. Forward Reaction Prediction](#2-forward-reaction-prediction)
  - [3. Molecule Generation](#3-molecule-generation)
  - [4. Material Generation](#4-material-generation)
  - [5. Structure Optimization](#5-structure-optimization)
  - [6. MD Simulation](#6-md-simulation)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [Citation](#citation)

---

## CO₂ Reference Points


| Category                     | Activity                                  | CO₂ Emission                                                                           |
| ---------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| **Everyday**                 | Smartphone charge (iPhone 16 Pro Max)     | ~9.7 g CO₂ eq/charge                                                                   |
|                              | Driving a car (EU average)                | ~170 g CO₂ eq/km                                                                       |
|                              | Commercial Aviation (Boeing 737)          | ~3.16 kg CO₂ eq/km                                                                       |
| **LLM inference**            | Text generation (Claude 3.7 Sonnet)       | ~2.12 g CO₂ eq/call                                                                    |
|                              | Image generation (Stable Diffusion)       | ~1.38 g CO₂ eq/image                                                                   |
| **Chemical simulation**      | Classical MD (force field)                | 10 g CO₂ eq/1M steps                                                                   |
|                              | Ab initio MD (PBE, 50 atoms)              | 140.96 kg CO₂ eq/1M steps                                                              |
| **Chemical synthesis**       | Organic synthesis (Letermovir)            | [369 kg CO₂ eq/kg](https://pubs.acs.org/doi/full/10.1021/jacs.5c14470)                 |
|                              | Material synthesis (UiO-66-NH₂)           | [43 kg CO₂ eq/kg](https://www.sciencedirect.com/science/article/pii/S2213343721001366) |
|                              | Battery synthesis (vanadium flow battery) | [37 kg CO₂ eq/MWh](https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.13328)        |


---

## Results

All tasks benchmarked on the same hardware with full carbon tracking.

**Hardware:** NVIDIA RTX 5000 Ada Generation (32 GB) · Intel Xeon Platinum 8558 (192 cores) · 503 GB RAM

**Column definitions:**

- **CO₂/exp** — total CO₂ for the full benchmark run (actual experiment)
- **CO₂/job** — normalized per a fixed workload (see per-task note)
- **Time/exp** — total wall-clock time for the full benchmark run

---

### 1. Retrosynthesis

**Dataset:** USPTO-50K · **N =** 5,007 reactions · **Metric:** Top-50 exact match · **CO₂/job:** per 500 reactions


| Year | Venue         | Model       | Architecture  | Params | Top-10    | Top-50    | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ------------- | ----------- | ------------- | ------ | --------- | --------- | ----------- | ----------- | ------------ | ------------ |
| 2017 | Chem. Eur. J. | neuralsym   | MLP           | 32.5M  | 72.8%     | 74.8%     | 35.0        | 3.50        | 1,282        | 128          |
| 2021 | JCIM          | MEGAN       | GNN           | 9.8M   | 87.0%     | 90.1%     | 51.7        | 5.15        | 2,951        | 295          |
| 2021 | JACS Au       | LocalRetro  | GNN           | 8.6M   | 91.5%     | 97.3%     | 62.1        | 6.20        | 2,313        | 231          |
| 2022 | Chem. Sci.    | RSMILES     | LM            | 44.6M  | 89.6%     | 93.0%     | 1,083       | 108.25      | 44,142       | 4,401        |
| 2022 | ML:ST         | Chemformer  | LM            | 44.7M  | 62.8%     | 64.0%     | 2,570       | 256.65      | 85,055       | 8,482        |
| 2024 | COLM          | LlaSMol     | LLM           | ~7.2B  | 5.0%      | 5.0%      | 1,385       | 138.35      | 39,119       | 3,905        |
| 2024 | ICLR          | RetroBridge | Flow Matching | 4.6M   | 44.9%     | 52.8%     | 4,040       | 403.45      | 157,820      | 15,740       |
| 2025 | Nat. Commun.  | **RSGPT**   | LLM           | ~1.6B  | **96.6%** | **97.8%** | 2,512       | 250.80      | 79,090       | 7,887        |


---

### 2. Forward Reaction Prediction

**Dataset:** USPTO-MIT · **N =** 40,029 reactions · **Metric:** Top-3 exact match · **CO₂/job:** per 500 reactions


| Year | Venue              | Model                | Architecture | Params | Top-1     | Top-3     | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ------------------ | -------------------- | ------------ | ------ | --------- | --------- | ----------- | ----------- | ------------ | ------------ |
| 2017 | Chem. Eur. J.      | neuralsym            | MLP          | 98.1M  | 49.5%     | 50.6%     | 43.9        | 0.55        | 2,732        | 34           |
| 2019 | ACS Cent. Sci.     | MolecularTransformer | LM           | 11.7M  | 86.8%     | 91.7%     | 360.0       | 4.50        | 12,317       | 154          |
| 2021 | JCIM               | MEGAN                | GNN          | 9.9M   | 80.1%     | 86.4%     | 85.3        | 1.07        | 6,657        | 83           |
| 2021 | JCIM               | Graph2SMILES         | LM           | 18M    | 88.5%     | 89.9%     | 287.8       | 3.60        | 7,940        | 99           |
| 2022 | Nat. Mach. Intell. | LocalTransform       | GNN          | 9.1M   | 90.4%     | 94.1%     | 141.4       | 1.77        | 8,799        | 110          |
| 2022 | ML:ST              | Chemformer           | LM           | 44.7M  | 89.0%     | 89.8%     | 580.0       | 7.25        | 45,288       | 566          |
| 2022 | Chem. Sci.         | **RSMILES**          | LM           | 44.6M  | **89.4%** | **94.7%** | 614.7       | 7.68        | 46,209       | 578          |
| 2024 | COLM               | LlaSMol              | LLM          | ~7.2B  | 3.8%      | 5.9%      | 1,413.8     | 17.67       | 104,960      | 1,312        |


---

### 3. Molecule Generation

**Dataset:** ChEMBL 28 · **N =** 10,000 molecules · **Metric:** VUN% · **CO₂/job:** per 10K molecules (= full exp)


| Year | Venue                | Model         | Architecture  | Params | VUN (%)   | SUN (%)   | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | -------------------- | ------------- | ------------- | ------ | --------- | --------- | ----------- | ----------- | ------------ | ------------ |
| 2017 | J. Cheminf.          | REINVENT      | LM            | 4.4M   | 87.90     | 74.40     | 0.11        | 0.18        | 10           | 14           |
| 2018 | ICML                 | JT-VAE        | VAE           | 7.1M   | 91.39     | 75.70     | 20.4        | 10.58       | 1,284        | 662          |
| 2020 | ICML                 | HierVAE       | VAE           | 8.0M   | 92.10     | **77.98** | 14.4        | 11.97       | 788          | 756          |
| 2021 | J. Chem. Inf. Model. | MolGPT        | LM            | 6.4M   | 77.15     | 65.00     | 1.85        | 1.07        | 60           | 37           |
| 2023 | ICML                 | DiGress       | Diffusion     | 16.2M  | 82.45     | 78.71     | 392.0       | 175.35      | 11,931       | 5,201        |
| 2024 | J. Cheminf.          | **REINVENT4** | LM            | 5.8M   | **94.16** | 75.65     | **0.09**    | **0.07**    | **10**       | **8**        |
| 2024 | arXiv                | SmileyLlama   | LLM           | 8.0B   | 94.26     | 77.75     | 22.7        | 21.79       | 645          | 638          |
| 2024 | NeurIPS              | DeFoG         | Flow Matching | 16.3M  | 82.27     | 75.90     | 355.2       | 355.24      | 9,874        | 9,874        |


---

### 4. Material Generation

**Dataset:** MP-20 · **N =** 1,000 structures · **Metric:** mSUN% · **CO₂/job:** per 1K structures (= full exp)


| Year | Venue        | Model          | Architecture  | Params | mSUN (%) | SUN (%) | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ------------ | -------------- | ------------- | ------ | -------- | ------- | ----------- | ----------- | ------------ | ------------ |
| 2022 | ICLR         | CDVAE          | Diffusion     | 4.9M   | 22.6     | 3.2     | 270.4       | 270.40      | 25,764       | 25,764       |
| 2023 | NeurIPS      | DiffCSP        | Diffusion     | 12.4M  | 29.0     | 4.3     | 12.7        | 12.60       | 381          | 381          |
| 2024 | Nat. Commun. | CrystaLLM      | LM            | 25.9M  | 16.4     | 3.5     | 19.3        | 19.20       | 942          | 942          |
| 2024 | ICML         | FlowMM         | Flow Matching | 28.3M  | 23.9     | 4.3     | 12.8        | 12.80       | 547          | 547          |
| 2025 | arXiv        | **ChargeDIFF** | Diffusion     | 59.5M  | **33.5** | 4.4     | 133.5       | 133.50      | 2,994        | 2,994        |
| 2025 | Nature       | MatterGen      | Diffusion     | 44.6M  | 33.4     | **5.2** | 248.1       | 248.10      | 8,079        | 8,079        |
| 2025 | ICML         | ADiT           | Diffusion     | 231.9M | 29.6     | **5.5** | 112.5       | 112.50      | 10,512       | 10,512       |
| 2025 | ICML         | CrystalFlow    | Flow Matching | 20.9M  | 21.7     | 3.0     | **1.5**     | **1.50**    | **43**       | **43**       |


---

### 5. Structure Optimization

**System:** LGPS · **N =** 75K steps × 3 seeds · **Metric:** CPS · **CO₂/job:** per 1M steps


| Year | Venue                   | Model    | Architecture | Params | CPS       | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ----------------------- | -------- | ------------ | ------ | --------- | ----------- | ----------- | ------------ | ------------ |
| 2023 | Nat. Mach. Intell.      | CHGNet   | GNN          | 413K   | 0.343     | 9.47        | 379         | 602          | 8,033        |
| 2023 | arXiv                   | MACE     | GNN          | 4.69M  | 0.637     | 23.29       | 932         | 1,068        | 14,241       |
| 2024 | J. Chem. Theory Comput. | SevenNet | GNN          | 1.17M  | 0.714     | 16.21       | 648         | 790          | 10,529       |
| 2024 | arXiv                   | ORB      | GNN          | 25.2M  | 0.470     | **3.87**    | **155**     | **210**      | **2,795**    |
| 2025 | arXiv                   | NequIP   | GNN          | 9.6M   | 0.733     | 11.34       | 454         | 316          | 4,219        |
| 2025 | arXiv                   | DPA3     | GNN          | 4.81M  | 0.718     | 38.45       | 1,538       | 2,087        | 27,829       |
| 2025 | arXiv                   | Nequix   | GNN          | 708K   | 0.729     | 17.13       | 685         | 736          | 9,809        |
| 2025 | arXiv                   | **eSEN** | GNN          | 30.1M  | **0.797** | 87.14       | 3,486       | 2,780        | 37,071       |


---

### 6. MD Simulation

**System:** LGPS · **N =** 75K steps × 3 seeds · **Metric:** MSD score · **CO₂/job:** per 1M steps


| Year | Venue                   | Model    | Architecture | Params | MSD       | CO₂/exp (g) | CO₂/job (g) | Time/exp (s) | Time/job (s) |
| ---- | ----------------------- | -------- | ------------ | ------ | --------- | ----------- | ----------- | ------------ | ------------ |
| 2023 | Nat. Mach. Intell.      | CHGNet   | GNN          | 413K   | 0.047     | 9.47        | 379         | 602          | 8,033        |
| 2023 | arXiv                   | MACE     | GNN          | 4.69M  | 0.095     | 23.29       | 932         | 1,068        | 14,241       |
| 2024 | J. Chem. Theory Comput. | SevenNet | GNN          | 1.17M  | 0.531     | 16.21       | 648         | 790          | 10,529       |
| 2024 | arXiv                   | ORB      | GNN          | 25.2M  | 0.385     | **3.87**    | **155**     | **210**      | **2,795**    |
| 2025 | arXiv                   | NequIP   | GNN          | 9.6M   | 0.361     | 11.34       | 454         | 316          | 4,219        |
| 2025 | arXiv                   | DPA3     | GNN          | 4.81M  | 0.508     | 38.45       | 1,538       | 2,087        | 27,829       |
| 2025 | arXiv                   | Nequix   | GNN          | 708K   | 0.203     | 17.13       | 685         | 736          | 9,809        |
| 2025 | arXiv                   | **eSEN** | GNN          | 30.1M  | **0.720** | 87.14       | 3,486       | 2,780        | 37,071       |


---

## Key Insights

- **Simpler models dominate the efficiency frontier:** In every task, at least one GNN or small LM achieves near-SOTA accuracy at 10–100× lower CO₂ than the best-performing model
- **Architecture drives cost more than parameter count:** Diffusion models cost 10–100× more per sample than LM or GNN models due to iterative sampling, regardless of size
- **LLMs underperform on narrow scientific tasks:** LlaSMol (7B) scores 3.8% top-1 on Forward prediction vs. 89.4% for RSMILES (45M) — at 2× the carbon cost
- **MLIP tasks are carbon-intensive by nature:** A single 75K-step MD run costs 4–87 g CO₂, orders of magnitude more than per-molecule chemistry tasks
- **50–75% of models per task are Pareto-dominated:** A cheaper and more accurate alternative always exists — the extra carbon was wasted

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide on adding new models and tasks.

**Quick start:**

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
git checkout main && git pull
git checkout -b <your-name>/<task>-<model>
cp -r Example/ <YourTask>/   # copy the template
claude                        # Claude Code guides you through the rest
```

Your PR to `main` only needs: `results/<Task>/<model>_<N>.json` + a new row in this README.

---

## Citation

```bibtex
@article{carbon2026,
  title={The Carbon Cost of AI for Science},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.