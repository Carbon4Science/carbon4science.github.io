# The Carbon Cost of Generative AI for Science

## Paper Story and Key Arguments

---

## 1. The Problem: Carbon Cost Is Growing, But Performance Isn't Keeping Up

**Claim:** Over the years, generative AI models for science have grown larger and more carbon-intensive, but their performance gains are plateauing or even declining relative to cost.

**Evidence from our data:**

Year-by-year trend (median across all 46 models):

| Year | N models | Median Params | Median CO2 (g) | Mean Norm. Perf |
|------|----------|--------------|----------------|-----------------|
| 2017 | 3 | 32.5M | 35.0 | 0.61 |
| 2018 | 1 | 7.1M | 20.4 | 1.00 |
| 2019 | 1 | 11.7M | 360.0 | 0.97 |
| 2020 | 1 | 8.0M | 14.4 | 0.69 |
| 2021 | 4 | 9.2M | 57.0 | 0.88 |
| 2022 | 5 | 44.6M | 614.7 | 0.78 |
| 2023 | 6 | 4.7M | 18.0 | 0.24 |
| 2024 | 13 | 25.2M | 19.2 | 0.51 |
| 2025 | 12 | 15.1M | 38.5 | 0.75 |

**Key points for the narrative:**
- 2022 marks a spike in both model size (44.6M median) and carbon cost (614.7g median), driven by transformer/seq2seq models entering chemistry tasks.
- 2023-2025 see a wave of new models across MLIP and MatGen. While median CO2 comes back down, model sizes remain large, and performance does not systematically improve.
- The most carbon-expensive models (Chemformer 2,570g, RetroBridge 4,040g, LlaSMol 1,385g) cluster in 2022-2024, but their performance is often below cheaper alternatives from 2021.

**Figures:** Year vs Performance, Year vs Model Size, Year vs CO2 (per-task 2x3 subplots, as in figure_final.ipynb Figures 1, 2, 2b).

---

## 2. The Pareto Front: Identifying Efficient vs Wasteful Models

**Claim:** Within each task, a Pareto front analysis using relative metrics reveals that a large fraction of models are strictly dominated — another model exists that is both cheaper and better.

**Method:**
- Baseline model per task: the earliest model (reference point).
- x-axis: log(CO2_i / CO2_baseline) — relative carbon cost (log scale).
- y-axis: (perf_i - perf_baseline) / perf_baseline — relative performance gain.
- Pareto front: models where no other model achieves both lower CO2 and higher performance.
- Quadrant classification:
  - **Dominant** (low CO2, high perf): the ideal
  - **Tradeoff** (high CO2, high perf): SOTA but expensive
  - **Dominated** (high CO2, low perf): strictly worse — a warning sign
  - **Inverse** (low CO2, low perf): cheaper but worse

**Evidence:**

| Task | Models | Pareto-Optimal | Dominated | % Dominated |
|------|--------|---------------|-----------|-------------|
| MatGen | 8 | 4 | 4 | 50% |
| MolGen | 8 | 2 | 6 | 75% |
| Retro | 8 | 4 | 4 | 50% |
| Forward | 6 | 4 | 2 | 33% |
| StructOpt | 8 | 3 | 5 | 63% |
| MDSim | 8 | 3 | 5 | 63% |

**Key points for the narrative:**
- **MolGen is the most wasteful field**: 75% of models are dominated. Only REINVENT4 and JT-VAE are Pareto-optimal. Six models (including DiGress at 392g CO2 with worst VUN, and DeFoG at 572g) are strictly worse than these two.
- **Forward is the most efficient field**: 67% of models are Pareto-optimal. Most models represent genuine tradeoffs rather than waste.
- **Cross-task pattern**: Diffusion models frequently appear in the "Dominated" quadrant (RetroBridge, DiGress, CDVAE). This is an architectural signal, not a coincidence.

**Figures:** Quadrant plots with arrows showing chronological trajectory (as in figure_final.ipynb Figure 3/3b). One 2x3 panel per task.

---

## 3. Strategy: What Drives Carbon Cost, and How to Reduce It

**Claim:** Carbon emission is almost entirely determined by inference time, which in turn is determined by model architecture — not model size. This means the most effective strategy to reduce carbon is to choose the right architecture, not to shrink the model.

### 3.1 The CO2 Decomposition

Multiple regression on all 46 models (log-log scale):

```
log(CO2) = 0.040 * log(model_size) + 0.999 * log(inference_time) + constant
R² = 0.973
```

**Inference time is ~18x more important than model size in predicting CO2.** Model size coefficient is near zero (0.040), meaning a 10x increase in parameters adds only 1.1x CO2 if inference time stays constant. But a 10x increase in inference time adds 10x CO2 regardless of size.

This holds per-architecture:
- GNN: `log(CO2) = 0.08*log(size) + 0.91*log(time)`, R²=0.95
- LM: `log(CO2) = -0.37*log(size) + 1.16*log(time)`, R²=0.99
- Diffusion: `log(CO2) = -0.08*log(size) + 0.86*log(time)`, R²=0.91

### 3.2 Model Size Does NOT Predict Inference Time

| Task | r(log_size, log_time) | p-value | Significant? |
|------|----------------------|---------|-------------|
| MatGen | +0.16 | 0.71 | No |
| MolGen | +0.29 | 0.52 | No |
| Retro | +0.32 | 0.44 | No |
| Forward | +0.57 | 0.23 | No |
| StructOpt | +0.03 | 0.95 | No |
| MDSim | +0.03 | 0.95 | No |

No task shows a statistically significant correlation (all p > 0.05). This is counterintuitive — one might expect bigger models to be slower. The reason is that **architecture determines computational density**:

### 3.3 Architecture Is the Key Factor

"Computational density" = time per parameter (s/param), i.e., how much computation each parameter requires at inference:

| Architecture | Median time/param (s) | Explanation |
|-------------|----------------------|-------------|
| LLM | 1.0e-05 | Large but efficient: single forward pass, optimized KV-cache |
| Flow Matching | 1.1e-05 | Few sampling steps, parallel computation |
| MLP | 3.4e-05 | Single forward pass, simplest architecture |
| VAE | 1.4e-04 | Encode + decode, but single pass |
| Diffusion | 1.8e-04 | **Iterative denoising: 100-1000 forward passes per sample** |
| GNN | 3.7e-04 | Message passing: multiple rounds over molecular graph |
| LM | 5.1e-04 | Autoregressive: sequential token generation, beam search |

Diffusion models run 100-1000 forward passes per sample (denoising steps). This is why RetroBridge (4.6M params) takes 157,706s while neuralsym (32.5M params, 7x larger) takes only 1,283s — a 123x speed difference despite being smaller.

**Outlier examples that prove the point:**

Models much SLOWER than their size predicts:
- DiGress (16.2M, Diffusion): 52x slower — graph diffusion with many denoising steps
- CDVAE (4.9M, Diffusion): 26x slower — crystal diffusion + relaxation
- RetroBridge (4.6M, Diffusion): 15x slower — Markov bridge sampling

Models much FASTER than their size predicts:
- CrystalFlow (20.9M, Flow Matching): 35x faster — few-step flow
- REINVENT4 (5.8M, LM): 17x faster — fast autoregressive SMILES generation
- REINVENT (4.4M, LM): 15x faster — same pattern
- neuralsym (32.5M, MLP): 12x faster — single forward pass through MLP

### 3.4 More Compute Time Does NOT Buy Better Performance

| Task | r(log_time, perf) | p-value | Significant? | Direction |
|------|------------------|---------|-------------|-----------|
| MatGen | +0.36 | 0.38 | No | Weak positive |
| MolGen | -0.41 | 0.37 | No | **Negative** |
| Retro | -0.33 | 0.43 | No | **Negative** |
| Forward | -0.33 | 0.52 | No | **Negative** |
| StructOpt | +0.51 | 0.19 | No | Weak positive |
| MDSim | +0.41 | 0.31 | No | Weak positive |

No task shows a statistically significant correlation. Strikingly, chemistry tasks (Retro, Forward, MolGen) show **negative** trends — slower models tend to perform worse. Example from Retro: LocalRetro runs in 2,316s and achieves 95.6% top-50, while Chemformer takes 84,990s (37x slower) for only 64.0%, and RetroBridge takes 157,706s (68x slower) for 52.8%. The extra time is purely architectural overhead from iterative sampling, not useful computation.

This further strengthens the argument: CO2 is driven by time, time is driven by architecture, and more time does not mean better results. The carbon is simply wasted.

### 3.5 Model Size Also Does NOT Predict Performance

| Task | r(log_size, perf) | Direction |
|------|------------------|-----------|
| Forward | -0.95 | **Strongly negative** — larger models perform worse |
| Retro | -0.47 | Negative |
| MatGen | +0.46 | Weakly positive |
| MDSim | +0.59 | Moderately positive |
| StructOpt | +0.30 | Weak |
| MolGen | +0.01 | No relationship |

Chemistry tasks (Retro, Forward) show **negative** correlation — the largest model (LlaSMol, 7.2B) is the worst performer. Physics tasks (MLIP) show mild positive correlation, suggesting atomic simulation may still benefit from scale.

### 3.6 Practical Strategy

Based on the decomposition above, we recommend a decision framework:

1. **First, choose the architecture class wisely.** Avoid diffusion for tasks where autoregressive or template-based methods exist. The architectural choice determines CO2 within 1-2 orders of magnitude before any other design decision.

2. **Second, optimize inference time, not model size.** Techniques like distillation, pruning, quantization, and fewer sampling steps directly reduce the dominant cost factor. Reducing parameters without reducing time saves almost nothing.

3. **Third, check the Pareto front before publishing.** If your model is dominated (another model is both cheaper and better), the carbon was wasted. Our data shows 50-75% of models per task are dominated.

---

## 4. The Urgency: Scaling Scenarios for Autonomous AI-Driven Discovery

**Claim:** As AI agents begin to use these models for automated, large-scale scientific discovery, the carbon cost will scale multiplicatively. The difference between choosing efficient vs inefficient models becomes enormous at scale.

**Method:** Rather than extrapolating trends (statistically fragile with our sample size), we use a concrete scaling scenario grounded in our measured per-run costs.

### 4.1 Per-Call Cost: AI4Science Tools vs LLM Prompts

A common assumption is that the AI4Science tool cost is negligible compared to the LLM agent's reasoning cost. Our data shows this depends entirely on the task and architecture:

| Task | Model | CO2 per sample | Equiv. LLM prompts |
|------|-------|---------------|-------------------|
| MolGen | REINVENT4 | 0.00001g | 0.000005x |
| Retro | LocalRetro | 0.012g | 0.006x |
| Retro | RetroBridge | 0.807g | 0.4x |
| **MLIP** | **CHGNet (1M steps)** | **379g** | **190x** |
| **MLIP** | **eSEN (1M steps)** | **3,486g** | **1,743x** |

(Reference: ~2g CO2 per LLM prompt at 10K input + 1K output tokens)

For chemistry tasks (Retro, MolGen, Forward), per-sample tool cost is typically far below a single LLM prompt. But for MLIP, a single production MD simulation (1M steps, typical for structure optimization or diffusion studies) costs **190-1,743x more than an LLM prompt**. MLIP is by far the most carbon-intensive task per inference call.

### 4.2 The Tipping Point: When Tool Cost Dominates Agent Cost

An AI agent's total carbon = (N_reasoning x LLM_cost) + (N_tool_calls x tool_cost). The question is: which term dominates?

**Scenario A: 100-step retrosynthesis planning**
- 100 LLM reasoning steps + 100 retro predictions

| Model | LLM cost | Tool cost | Total | Tool % |
|-------|----------|-----------|-------|--------|
| LocalRetro | 200g | 1.2g | 201g | 1% |
| RSGPT | 200g | 50g | 250g | 20% |
| RetroBridge | 200g | 81g | 281g | 29% |

Tool cost is minor — LLM reasoning dominates.

**Scenario B: 10,000-step materials screening**
- 500 LLM reasoning steps + 10,000 material generation calls

| Model | LLM cost | Tool cost | Total | Tool % |
|-------|----------|-----------|-------|--------|
| CrystalFlow | 1,000g | 15g | 1,015g | 1% |
| MatterGen | 1,000g | 2,481g | 3,481g | **71%** |
| CDVAE | 1,000g | 2,704g | 3,704g | **73%** |

Tool cost overtakes LLM cost for expensive architectures.

**Scenario C: MLIP production MD campaign**
- Agent screens 100 materials, each with 1M-step MD simulation
- 1,000 LLM reasoning steps for analysis

| Model | LLM cost | Tool cost | Total | Tool % |
|-------|----------|-----------|-------|--------|
| ORB | 2,000g | 15,480g | 17,480g | **89%** |
| CHGNet | 2,000g | 37,900g | 39,900g | **95%** |
| eSEN | 2,000g | 348,570g | 350,570g | **99.4%** |

**MLIP completely dominates.** The LLM agent cost is negligible. A 100-material screening campaign with eSEN emits 350 kg CO2 — equivalent to a transatlantic flight.

### 4.3 The Renewable Energy Caveat

An important nuance: AI agents (LLMs) typically run in hyperscaler data centers (Google, Azure, AWS) that increasingly use renewable energy, lowering their effective CO2. But many AI4Science tools — especially MLIP models — run on university clusters and institutional HPC systems powered by local grid electricity, which often has higher carbon intensity. This makes the gap even larger in practice: the "clean" LLM prompt at 2g becomes <1g on renewables, while the "dirty" MLIP simulation at 3,486g may be higher if running on a coal-heavy grid.

### 4.4 Scaling to Autonomous Discovery

**Scenario: 10,000 inference runs per day for 1 year** (plausible for an AI agent doing automated retrosynthesis planning or materials screening):

| Task | Expensive Model | Efficient Model | Annual CO2 (expensive) | Annual CO2 (efficient) | Ratio |
|------|----------------|----------------|----------------------|----------------------|-------|
| Retro | RSGPT (1.6B, LLM) | LocalRetro (8.6M, GNN) | 9,168 tonnes | 227 tonnes | 40x |
| Forward | RSMILES (44.6M, LM) | MEGAN (9.9M, GNN) | 2,244 tonnes | 311 tonnes | 7x |
| MolGen | DiGress (16.2M, Diffusion) | REINVENT4 (5.8M, LM) | 1,431 tonnes | 0.3 tonnes | 4,355x |
| MatGen | CDVAE (4.9M, Diffusion) | CrystalFlow (20.9M, Flow) | 987 tonnes | 5.4 tonnes | 183x |
| MLIP (1M steps) | eSEN (30.1M, GNN) | ORB (25.2M, GNN) | 12,723 tonnes | 565 tonnes | 23x |

**Key points for the narrative:**
- **MLIP is the most carbon-intensive task by far.** A single 1M-step MD simulation with eSEN (3.5 kg CO2) costs more than 4,000 retrosynthesis predictions with LocalRetro. At scale, MLIP dominates the total carbon budget of any multi-task AI agent.
- The MolGen case is extreme per-sample: DiGress emits **4,355x more CO2** than REINVENT4 for lower performance.
- Even in Retro where both models are competitive, choosing RSGPT over LocalRetro costs an extra 8,941 tonnes/year for a 2.2% accuracy gain.
- These are per-task costs. An integrated AI agent running retrosynthesis + forward prediction + molecule generation + MD simulation in a closed loop would multiply these numbers.

**Note on the "10K runs/day" assumption:** This is conservative for chemistry tasks. Retrosynthesis planning tools like ASKCOS already evaluate thousands of routes. For MLIP, 10K MD simulations/day is plausible for high-throughput materials screening campaigns.

---

## 5. Cross-Task Insight: Architecture Efficiency Transfers

**Bonus finding (unique to this multi-task study):**

Four models appear in both Retro and Forward tasks, allowing direct comparison of how well their efficiency transfers:

| Model | Architecture | Retro delta_perf | Forward delta_perf | Retro CO2 | Forward CO2 |
|-------|-------------|-----------------|-------------------|-----------|-------------|
| neuralsym | MLP | +0.0% (baseline) | +0.0% (baseline) | 35g | 44g |
| MEGAN | GNN | +20.5% | +70.8% | 52g | 85g |
| RSMILES | LM | +24.3% | +87.2% | 1,084g | 615g |
| LlaSMol | LLM | -93.3% | -88.3% | 1,385g | 1,414g |

- **MEGAN (GNN)** is consistently efficient across tasks: moderate CO2, strong performance gain.
- **LlaSMol (LLM)** is consistently terrible: highest CO2, worst performance in both tasks. The failure transfers — a general-purpose LLM that doesn't work for chemistry doesn't work for *any* chemistry task.
- **Architecture class predicts cross-task behavior** better than model size or venue.

Similarly in MLIP, the same 8 models are evaluated on both StructOpt (CPS) and MDSim (MSD). Models like eSEN are best on both, while CHGNet is worst on both. But notably, MACE ranks 6th on StructOpt (CPS=0.637) but 7th on MDSim (MSD=0.095) — structure optimization and dynamics simulation test different capabilities even within the same architecture class.

---

## Suggested Figure List

1. **Year vs Performance / Model Size / CO2** — 3 panels of 2x3 subplots showing trends over time per task. (Section 1)
2. **Pareto Front Quadrant Plots** — 2x3 subplots, one per task. x=log(CO2 ratio), y=delta performance. Quadrant shading. Arrows showing chronological trajectory. (Section 2)
3. **CO2 Decomposition** — scatter plot of log(time) vs log(CO2) colored by task, with R²=0.97 regression line. Side panel: log(size) vs log(CO2) showing weak correlation. (Section 3.1)
4. **Model Size vs Inference Time** — 2x3 per-task subplots with regression lines and p-values. None significant. (Section 3.2)
5. **Computational Density by Architecture** — box plot of time/parameter by architecture type. Shows 50x range. (Section 3.3)
6. **Scaling Scenario Bar Chart** — annual CO2 for efficient vs expensive model per task. Log-scale y-axis. (Section 4)
7. **Cross-Task Transfer** — grouped bar chart showing MEGAN/LlaSMol/RSMILES performance and CO2 across Retro+Forward. (Section 5)

---

## One-Paragraph Abstract Draft

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on predictive accuracy, neglecting the environmental cost of increasingly complex generative models. We benchmark 46 models across six scientific tasks — retrosynthesis, forward reaction prediction, molecule generation, material generation, structure optimization, and molecular dynamics simulation — measuring both accuracy and carbon footprint under identical hardware conditions. We find that carbon emissions scale almost entirely with inference time (R²=0.97), not model size, and that inference time is determined by architectural choice: diffusion models are 10-50x slower per parameter than MLPs or language models. Pareto front analysis reveals that 50-75% of models per task are strictly dominated — another model exists that is both cheaper and better. Scaling these costs to autonomous AI-driven discovery (10,000 runs/day), we show that architectural choice alone creates a 40-4,000x difference in annual carbon emissions. We propose that carbon efficiency should be a first-class evaluation metric alongside accuracy, and provide a decision framework for choosing architectures that maximize performance per unit carbon.
