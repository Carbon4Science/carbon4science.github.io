# ExampleTask

**Task description:** Brief description of what this task evaluates.

## Metrics

| Metric | Description |
|--------|-------------|
| `top_1` | Primary accuracy metric |
| `top_5` | Secondary accuracy metric |

## Test Dataset

- **Dataset name**: N test samples
- Location: `data/test.csv`

## Models

| Model | Year | Venue | Architecture | Params | Environment |
|-------|------|-------|-------------|--------|-------------|
| ExampleModel | 2024 | NeurIPS | Transformer | 10M | `example_env` |

## Results

Full test set (N samples). All models run on the same hardware.

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM*

### Accuracy

| Model | Top-1 | Top-5 |
|-------|-------|-------|
| ExampleModel | 85.0% | 95.0% |

### Carbon Efficiency

| Model | Duration (s) | Energy (Wh) | CO2 (g) |
|-------|-------------|-------------|---------|
| ExampleModel | 1,200 | 50.0 | 20.0 |
