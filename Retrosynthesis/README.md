# Retrosynthesis

**Task Leader:** admin

Retrosynthesis prediction: Given a target product molecule, predict the reactants needed to synthesize it.

## Metrics

| Metric | Description |
|--------|-------------|
| `top_1` | Exact match at rank 1 |
| `top_5` | Correct answer in top 5 predictions |
| `top_10` | Correct answer in top 10 predictions |
| `top_50` | Correct answer in top 50 predictions |

## Test Dataset

- **USPTO-50K**: 5,005 test reactions from US Patent data
- Location: `data/USPTO_50K_test.pickle`

## Models

| Model | Paper | Environment |
|-------|-------|-------------|
| neuralsym | [Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction (Nature 2018)](https://www.nature.com/articles/nature25978) | `neuralsym` |
| LocalRetro | [Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention (JACS Au 2021)](https://pubs.acs.org/doi/10.1021/jacsau.1c00246) | `localRetro` |
| Chemformer | [Chemformer: A Pre-Trained Transformer for Computational Chemistry (Machine Learning: Science and Technology 2022)](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb) | `chemformer` |
| RetroBridge | [RetroBridge: Modeling Retrosynthesis with Markov Bridges (ICLR 2024)](https://openreview.net/forum?id=770DetV8He) | `retrobridge` |
| RSGPT | [Retrosynthesis prediction with an interpretable deep-learning framework based on molecular assembly tasks (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-62308-6) | `rsgpt` |

## Current Results

Results on USPTO-50K test set (500 samples):

| Model | Top-1 | Top-10 | Top-50 | Duration (s) | Energy (kWh) | CO2 (kg) |
|-------|-------|--------|--------|--------------|--------------|----------|
| LocalRetro | - | 89.6% | 94.0% | 302.5 | 0.0052 | 0.0022 |

*Hardware: Apple M4 Pro (12 cores), 24GB RAM, CPU-only inference*

## Usage

```python
from Retrosynthesis.evaluate import load_test_data, evaluate, METRICS

# Load test data
test_cases = load_test_data(limit=100)

# Run model inference
from Retrosynthesis.LocalRetro.Inference import run
predictions = [run(tc['product'], top_k=50) for tc in test_cases]

# Evaluate
results = evaluate(predictions, test_cases, metrics=['top_10', 'top_50'])
print(f"Top-10: {results['top_10']*100:.2f}%")
print(f"Top-50: {results['top_50']*100:.2f}%")
```

## Adding a New Model

See `/add-model Retrosynthesis <ModelName>` skill or `../.claude/skills/add-model.md`.
