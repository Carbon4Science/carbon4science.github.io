# Paper Story: The Carbon Cost of AI for Science

## One-sentence pitch
**Choosing the right model architecture — not shrinking model size — is the single most effective way to reduce the carbon cost of AI-driven scientific discovery.**

---

## Story flow

### Opening hook (Introduction)
AI agents (Coscientist, ChemCrow, AI Scientist) are automating science — calling retrosynthesis models, molecule generators, and MD simulators thousands of times per campaign. Everyone measures the cost of the LLM agent, but **nobody measures the cost of the scientific tools it calls**. Our data shows the tools are 89-99% of total agent emissions for compute-heavy tasks.

### The gap (Related Work)
Green AI has been studied in NLP (Strubell, Schwartz, Bender) but **never systematically applied to AI for science**. We fill this gap with the first cross-domain carbon benchmark.

### What we did (Methodology)
- 46 models across 6 tasks (MolGen, MatGen, Retro, Forward, StructOpt, MDSim)
- All on the same GPU, same protocol, same carbon tracker
- Two metrics per model: performance + CO₂ eq per job

### Key results (Section 4) — three acts:

**Act 1: The problem is real** (Fig. 1)
> Over the years, model sizes and carbon costs have grown, but performance has not scaled.

- Model sizes grew ~2x from 2017-2021 to 2023-2025
- CO₂/job grew ~185x (driven by new expensive tasks like MLIP)
- Mean normalized performance actually *decreased* (0.80 → 0.55)
- In Forward prediction, larger models are *significantly worse* (r=-0.95, p=0.004)

**Act 2: Most models are wasteful** (Fig. 2)
> Pareto analysis reveals 40-71% of models per task are strictly dominated.

- A "dominated" model = another model exists that is both cheaper AND better
- MolGen is worst: 71% dominated. Only REINVENT4 and JT-VAE on the frontier
- Diffusion models consistently land in high-cost regions
- LlaSMol (7.2B params) is dominated in both Retro AND Forward — general LLMs fail on specialized science

**Act 3: Architecture is the lever** (Fig. 3)
> CO₂ is determined by inference time (R²>0.91), and inference time is determined by architecture — not model size.

- Regression: log(CO₂) = 0.06·log(size) + 1.00·log(time). Time is 18x more important.
- Model size does NOT predict inference time (all p > 0.05 in every task)
- Why? Architecture determines "computational density" — diffusion needs 100-1000 forward passes, MLP/LM needs 1
- This means: shrinking a diffusion model barely helps. Switching to an LM architecture transforms the cost.

### The urgency (Discussion)
> As AI agents automate science, the tool cost — not the reasoning cost — dominates emissions.

- A single MLIP simulation (eSEN, 1M steps) = 3,486g CO₂ eq — more than 500 retrosynthesis predictions
- For AI-agent-driven materials screening, the tool is 89-99% of total carbon
- Unlike LLM prompts (run on Google/Azure renewables), science tools run on university clusters with dirty grid power
- Architecture choice alone creates 22-4,355x emission differences

### Recommendations (Discussion §5.2)
1. Choose architecture first (biggest lever)
2. Optimize wall-clock time, not parameter count
3. Check Pareto front before publishing
4. Report carbon cost as standard practice

---

## Figures

### Main text
- **Table 1**: All 46 models — architecture, params, performance, CO₂ eq/call, CO₂ eq/job. Grouped by 6 tasks. (Section 4.1)
- **Fig 1** (year trends): Left=performance vs year, Right=CO₂/job vs year. 6 task subplots each. Placed in Discussion §5.1. Message: performance plateaus while costs grow.
- **Fig 2** (Pareto): Top=all 46 models in one quadrant plot. Bottom=per-task Pareto frontiers with step-wise lines. (Section 4.2)
- **Fig 3** (CO₂ decomposition): Top=model size vs CO₂ (no correlation). Bottom=inference time vs CO₂ (R²>0.91). (Section 4.3)

### Appendix
- **Fig A1**: Year vs model size per task
- **Fig A2**: Model size vs inference time per task (no significant correlation)
- **Fig A3**: Cross-task CO₂ decomposition (time R²=0.97 vs size R²=0.12)

---

## Killer numbers for the abstract/intro
- 46 models, 6 tasks, 3 supertasks
- R² > 0.91 (time→CO₂ in every task)
- 40-71% of models per task are Pareto-dominated
- 22-4,355x emission differences from architecture choice alone
- MLIP: 155-3,486g per simulation vs chemistry: <1g per molecule
