# Forward (Forward Reaction Prediction)

Forward reaction prediction: Given reactant molecules, predict the product(s) of the reaction.

## Metrics

| Metric | Description |
|--------|-------------|
| `top_1` | Exact match at rank 1 |
| `top_2` | Correct answer in top 2 predictions |
| `top_3` | Correct answer in top 3 predictions |
| `top_5` | Correct answer in top 5 predictions |

## Test Dataset

- **USPTO-MIT** (USPTO-480K): 40,000 test reactions
- Location: `data/test.csv`

## Models

| Model | Year | Venue | Architecture | Params | Environment |
|-------|------|-------|-------------|--------|-------------|
| neuralsym | 2017 | Chem. Eur. J. | MLP (template) | 98.1M | `neuralsym` |
| MolecularTransformer | 2019 | ACS Cent. Sci. | Transformer (seq2seq) | 11.7M | `mol_transformer` |
| MEGAN | 2021 | JCIM | Graph Edit Network | 9.9M | `megan` |
| Graph2SMILES | 2022 | JCIM | DGCN + Transformer | 18.0M | `mol_transformer` |
| Chemformer | 2022 | ML:ST | BART Transformer | 44.7M | `chemformer` |
| RSMILES_20x | 2022 | Chem. Sci. | Transformer (seq2seq) | 44.6M | `rsmiles` |
| LlaSMol | 2024 | COLM | LLM (Mistral-7B + LoRA) | ~7.2B | `llasmol` |

## Results

Full USPTO-MIT test set (40,000 samples, top_k=5). All models run on the same hardware.

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM*

### Accuracy (Top-k Exact Match)

| Model | Top-1 | Top-2 | Top-3 | Top-5 |
|-------|-------|-------|-------|-------|
| **RSMILES_20x** | **89.4%** | **93.6%** | **94.7%** | **95.8%** |
| Graph2SMILES | 88.5% | 89.6% | 89.8% | 89.9% |
| MolecularTransformer | 86.8% | 90.5% | 91.7% | 92.4% |
| Chemformer | 84.9% | 90.4% | 91.9% | 93.0% |
| MEGAN | 80.1% | 85.0% | 86.4% | 87.1% |
| neuralsym | 49.5% | 50.6% | 50.6% | 50.6% |
| LlaSMol | 3.8% | 5.1% | 5.9% | 7.1% |

### Carbon Efficiency

| Model | Duration (s) | Speed (s/rxn) | Energy (Wh) | CO2 (g) | CO2 Intensity (g/s) |
|-------|-------------|---------------|-------------|---------|---------------------|
| neuralsym | 2,732 | 0.07 | 109.7 | 43.9 | 0.0161 |
| MEGAN | 6,657 | 0.17 | 213.3 | 85.3 | 0.0128 |
| Graph2SMILES | 7,940 | 0.20 | 628.2 | 287.8 | 0.0362 |
| MolecularTransformer | 12,317 | 0.31 | 899.9 | 360.0 | 0.0292 |
| Chemformer | 45,288 | 1.13 | 1,449.9 | 580.0 | 0.0128 |
| RSMILES_20x | 46,209 | 1.16 | 1,536.8 | 614.7 | 0.0133 |
| LlaSMol | 104,960 | 2.62 | 3,534.6 | 1,413.8 | 0.0135 |

### Key Observations

- **Best accuracy**: RSMILES_20x (89.4% top-1) — but at 615 g CO2 with 20x test-time augmentation
- **Best accuracy-efficiency tradeoff**: Graph2SMILES (88.5% top-1 at 288 g CO2) — near-SOTA at half the carbon of RSMILES
- **Template-based ceiling**: neuralsym is fastest (0.07 s/rxn) but limited by template coverage (49.5% top-1)
- **LLM out-of-distribution**: LlaSMol (7B params) achieves only 3.8% on USPTO-MIT — highest carbon cost for lowest accuracy

## Adding a New Model

1. Create `<YourModel>/Inference.py` with the uniform `run()` interface
2. Create `<YourModel>/environment.yml` with your conda env spec
3. Open a PR to the `Forward` branch
