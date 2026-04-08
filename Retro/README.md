# Retro (Retrosynthesis)

**Task Leader:** Shuan Chen

Retrosynthesis prediction: Given a target product molecule, predict the reactants needed to synthesize it.

## Metrics

| Metric | Description |
|--------|-------------|
| `top_1` | Exact match at rank 1 |
| `top_5` | Correct answer in top 5 predictions |
| `top_10` | Correct answer in top 10 predictions |
| `top_50` | Correct answer in top 50 predictions |

## Test Dataset

- **USPTO-50K**: 5,007 test reactions from US Patent data
- Location: `data/test_demapped.csv`

## Models

| Model | Year | Venue | Architecture | Parameters | Environment |
|-------|------|-------|-------------|-----------|-------------|
| neuralsym | 2017 | Chem. Eur. J. | MLP (template) | 32.5M | `neuralsym` |
| MEGAN | 2021 | JCIM | Graph Edit Network | 9.8M | `megan2` |
| LocalRetro | 2021 | JACS Au | MPNN + attention | 8.6M | `rdenv` |
| RSMILES_1x | 2022 | Chem. Sci. | Transformer (seq2seq) | 44.6M | `rsmiles` |
| RSMILES_20x | 2022 | Chem. Sci. | Transformer (seq2seq) | 44.6M | `rsmiles` |
| Chemformer | 2022 | ML:ST | BART Transformer | 44.7M | `chemformer` |
| LlaSMol | 2024 | COLM | LLM (Mistral-7B + LoRA) | ~7.2B | `gpt` |
| RetroBridge | 2024 | ICLR | Markov Bridge (diffusion) | 4.6M | `retrobridge` |
| RSGPT | 2025 | Nat. Commun. | LLaMA-1B (GPT) | ~1.6B | `gpt` |

## Results

Full USPTO-50K test set (~5,007 samples, top_k=50). All models run on the same hardware.

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM*

### Accuracy (Top-k Exact Match)

| Model | Top-1 | Top-5 | Top-10 | Top-50 |
|-------|-------|-------|--------|--------|
| **RSGPT** | **76.0%** | **94.5%** | **96.6%** | **97.8%** |
| MEGAN | 62.9% | 83.4% | 87.0% | 90.1% |
| RSMILES_20x | 55.3% | 84.8% | 89.6% | 93.0% |
| Chemformer | 53.6% | 62.0% | 62.8% | 64.0% |
| LocalRetro | 52.8% | 85.0% | 91.5% | 95.6% |
| RSMILES_1x | 49.3% | 77.8% | 83.5% | 83.5% |
| neuralsym | 43.0% | 67.7% | 72.8% | 74.8% |
| RetroBridge | 22.1% | 39.4% | 44.9% | 52.8% |
| LlaSMol | 2.1% | 4.2% | 5.0% | 5.0% |

### Carbon Efficiency

| Model | Duration (s) | Speed (s/mol) | Energy (Wh) | CO2 (g) | CO2 Intensity (g/s) |
|-------|-------------|---------------|-------------|---------|---------------------|
| neuralsym | 1,283 | 0.26 | 87.6 | 35.0 | 0.0273 |
| LocalRetro | 2,316 | 0.46 | 155.6 | 62.2 | 0.0269 |
| MEGAN | 2,951 | 0.59 | 129.3 | 51.7 | 0.0175 |
| RSMILES_1x | 3,197 | 0.64 | 349.3 | 139.7 | 0.0437 |
| LlaSMol | 39,119 | 7.81 | 3,463.5 | 1,385.4 | 0.0354 |
| RSMILES_20x | 44,092 | 8.81 | 2,709.6 | 1,083.8 | 0.0246 |
| RSGPT | 79,024 | 15.78 | 6,279.3 | 2,511.7 | 0.0318 |
| Chemformer | 84,990 | 16.99 | 6,424.7 | 2,569.9 | 0.0302 |
| RetroBridge | 157,706 | 31.50 | 9,383.1 | 4,040.1 | 0.0256 |

### Combined Results (per 500 samples)

| Model | Params | Top-1 | Top-5 | Top-10 | Top-50 | Time/500 (s) | Energy/500 (Wh) | CO2/500 (g) | Peak GPU (MB) |
|-------|--------|-------|-------|--------|--------|-------------|-----------------|-------------|---------------|
| RSGPT | ~1.6B | 76.0% | 94.5% | 96.6% | 97.8% | 7,892 | 627.3 | 250.9 | 6,950 |
| MEGAN | 9.8M | 62.9% | 83.4% | 87.0% | 90.1% | 295 | 12.9 | 5.2 | 152 |
| RSMILES_20x | 44.6M | 55.3% | 84.8% | 89.6% | 93.0% | 4,403 | 270.6 | 108.2 | 924 |
| Chemformer | 44.7M | 53.6% | 62.0% | 62.8% | 64.0% | 8,491 | 641.9 | 256.8 | 209 |
| LocalRetro | 8.6M | 52.8% | 85.0% | 91.5% | 95.6% | 231 | 15.5 | 6.2 | 172 |
| RSMILES_1x | 44.6M | 49.3% | 77.8% | 83.5% | 83.5% | 319 | 34.9 | 13.9 | 121 |
| neuralsym | 32.5M | 43.0% | 67.7% | 72.8% | 74.8% | 128 | 8.7 | 3.5 | 504 |
| RetroBridge | 4.6M | 22.1% | 39.4% | 44.9% | 52.8% | 15,745 | 936.8 | 403.3 | 601 |
| LlaSMol | ~7.2B | 2.1% | 4.2% | 5.0% | 5.0% | 3,906 | 345.8 | 138.3 | 17,525 |

### Accuracy vs Carbon Cost

![Retro: Accuracy vs Carbon Cost](results/figures/accuracy_vs_carbon_combined.png)

### Key Observations

- **Best accuracy**: RSGPT (76.0% top-1) — but at 2,512 g CO2 total
- **Best accuracy-efficiency tradeoff**: MEGAN (62.9% top-1 at only 52 g CO2) — 2nd highest accuracy with the lowest CO2 intensity (0.018 g/s)
- **Best efficiency**: neuralsym (43.0% top-1 at 35 g CO2, fastest at 0.26 s/mol)
- **LLM models are carbon-heavy**: LlaSMol (7B params) uses 1,385 g CO2 for only 2.1% top-1 — out-of-distribution on USPTO-50K. RSGPT (1.6B) is much better tuned at 76.0% but still costs 2,512 g
- **Test-time augmentation tradeoff**: RSMILES 1x→20x gains +6% top-1 at 7.7x carbon cost
- **Diffusion models are expensive**: RetroBridge uses the most carbon (4,040 g) for low accuracy (22.1%)

## Usage

```python
from Retro.evaluate import load_test_data, evaluate, METRICS

# Load test data
test_cases = load_test_data(limit=100)

# Run model inference
from Retro.LocalRetro.Inference import run
predictions = [run(tc['product'], top_k=50) for tc in test_cases]

# Evaluate
results = evaluate(predictions, test_cases, metrics=['top_10', 'top_50'])
print(f"Top-10: {results['top_10']*100:.2f}%")
print(f"Top-50: {results['top_50']*100:.2f}%")
```

## Adding a New Model

See `/add-model Retro <ModelName>` skill or `../.claude/skills/add-model.md`.
