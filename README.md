# Retro (Retrosynthesis)

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

| Model | Year | Venue | Architecture | Params | Environment |
|-------|------|-------|-------------|--------|-------------|
| neuralsym | 2017 | Chem. Eur. J. | MLP (template) | 32.5M | `neuralsym` |
| MEGAN | 2021 | JCIM | Graph Edit Network | 9.8M | `megan2` |
| LocalRetro | 2021 | JACS Au | MPNN + attention | 8.6M | `rdenv` |
| RSMILES | 2022 | Chem. Sci. | Transformer (seq2seq) | 44.6M | `rsmiles` |
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

## Adding a New Model

1. Create `<YourModel>/Inference.py` with the uniform `run()` interface
2. Create `<YourModel>/environment.yml` with your conda env spec
3. Open a PR to the `Retro` branch
