# Benchmark Protocol

This directory contains the standardized benchmarking infrastructure for measuring both accuracy and carbon efficiency of generative AI models.

## Known Issues & Solutions

### Environment Conflicts

Each model requires a different conda environment due to incompatible dependencies:

| Model | Python | Key Dependencies |
|-------|--------|------------------|
| neuralsym | 3.6 | PyTorch 1.6, RDChiral |
| LocalRetro | 3.7 | DGL, DGLLife |
| RetroBridge | 3.9 | PyTorch Lightning 2.x |
| Chemformer | 3.7 | Poetry, PyTorch 1.8 |
| RSGPT | 3.9 | Transformers, DeepSpeed |

**Solution**: Use the wrapper script that handles environment switching:

```bash
# Setup all environments (one-time)
chmod +x setup_envs.sh
./setup_envs.sh

# Run benchmark with automatic environment switching
chmod +x run.sh
./run.sh --model neuralsym --smiles "CCO" --track_carbon
./run.sh --model LocalRetro --smiles "CCO" --track_carbon

# Run all models sequentially
./run.sh --model all --input test.csv --track_carbon
```

### Model Checkpoints

Pre-trained models must be downloaded separately. See `configs/models.yaml` for URLs:

```bash
# RetroBridge
wget https://zenodo.org/record/10688201/files/retrobridge.ckpt -O Retro/RetroBridge/models/retrobridge.ckpt

# RSGPT
# Download from https://sandbox.zenodo.org/records/203391
```

### GPU Memory Requirements

| Model | Approx. GPU Memory |
|-------|-------------------|
| neuralsym | 2 GB |
| LocalRetro | 4 GB |
| Chemformer | 6 GB |
| RetroBridge | 8 GB |
| RSGPT | 16 GB |

Use `--device cpu` if GPU memory is insufficient.

## Quick Start

### 1. Install Dependencies

```bash
pip install codecarbon pandas numpy
```

### 2. Run a Benchmark

```python
from carbon_tracker import CarbonTracker

# Initialize tracker
tracker = CarbonTracker(
    project_name="neuralsym_inference_run1",
    model_name="neuralsym",
    task="inference"
)

# Run your model with tracking
with tracker:
    predictions = model.predict(test_data)
    accuracy = evaluate(predictions, ground_truth)

# Add accuracy metrics
tracker.add_accuracy(top1=0.455, top10=0.816, num_samples=5005)

# View and save results
tracker.print_summary()
```

### 3. Aggregate Results

```python
from carbon_tracker import aggregate_results, create_comparison_table

results = aggregate_results("benchmarks/results")
print(create_comparison_table(results))
```

## Standardized Protocol

To ensure fair and reproducible comparisons, follow this protocol for all benchmarks.

### Hardware Requirements

Before running benchmarks, document your hardware using `configs/hardware_template.yaml`:

```yaml
hardware:
  gpu:
    model: "NVIDIA RTX 3090"
    count: 1
    memory_gb: 24
  cpu:
    model: "Intel Core i9-10900K"
    cores: 10
  ram_gb: 64
  cuda_version: "11.8"
```

### Training Benchmarks

For training benchmarks:

1. **Use the standard dataset split**
   - USPTO-50K: 39,713 train / 4,989 valid / 5,005 test

2. **Track the full training run**
   ```python
   tracker = CarbonTracker(
       project_name=f"{model_name}_training",
       model_name=model_name,
       task="training"
   )

   with tracker:
       train_model(train_data, valid_data)

   tracker.add_accuracy(top1=test_accuracy)
   ```

3. **Report**:
   - Total training energy (kWh)
   - Total training emissions (kg CO2)
   - Training duration (hours)
   - Final test accuracy

### Inference Benchmarks

For inference benchmarks:

1. **Use the standard test set**
   - USPTO-50K test: 5,005 reactions

2. **Run 3 independent trials**
   ```python
   for run in range(3):
       tracker = CarbonTracker(
           project_name=f"{model_name}_inference_run{run+1}",
           model_name=model_name,
           task="inference"
       )

       with tracker:
           predictions = model.predict(test_data)

       tracker.add_accuracy(top1=accuracy, num_samples=len(test_data))
   ```

3. **For latency measurements, use batch_size=1**:
   ```python
   tracker = CarbonTracker(project_name=f"{model_name}_latency")

   with tracker:
       for sample in test_data:
           model.predict([sample])

   tracker.add_accuracy(num_samples=len(test_data), batch_size=1)
   ```

4. **Report** (mean ± std across 3 runs):
   - Inference energy per 1,000 samples (kWh)
   - Inference time per sample (ms)
   - Top-k accuracy (k=1, 5, 10)

## Metrics Reference

### Energy Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `energy_kwh` | Total energy consumption | kWh |
| `gpu_energy_kwh` | GPU energy only | kWh |
| `cpu_energy_kwh` | CPU energy only | kWh |
| `emissions_kg_co2` | Carbon emissions | kg CO2eq |

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| `accuracy_top1` | Exact match at rank 1 |
| `accuracy_top5` | Correct in top 5 predictions |
| `accuracy_top10` | Correct in top 10 predictions |

### Timing Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `duration_seconds` | Total benchmark duration | seconds |
| Latency | Per-sample inference time | milliseconds |

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── carbon_tracker.py      # Unified tracking module
├── configs/
│   └── hardware_template.yaml
└── results/
    ├── neuralsym_inference_run1_20250127.json
    ├── neuralsym_inference_run2_20250127.json
    └── ...
```

## Result File Format

Each benchmark run saves a JSON file with this structure:

```json
{
  "start_time": "2025-01-27T10:30:00",
  "end_time": "2025-01-27T10:35:00",
  "duration_seconds": 300.5,
  "energy_kwh": 0.0234,
  "emissions_kg_co2": 0.0089,
  "gpu_energy_kwh": 0.0198,
  "cpu_energy_kwh": 0.0025,
  "ram_energy_kwh": 0.0011,
  "accuracy_top1": 0.455,
  "accuracy_top10": 0.816,
  "num_samples": 5005,
  "batch_size": 32,
  "project_name": "neuralsym_inference_run1",
  "model_name": "neuralsym",
  "task": "inference",
  "hardware": {
    "gpu_model": "NVIDIA RTX 3090",
    "gpu_count": 1,
    "gpu_memory_gb": 24.0,
    "cpu_model": "Intel Core i9-10900K",
    "cpu_cores": 10,
    "ram_gb": 64.0,
    "cuda_version": "Driver 535.104.05",
    "platform": "Linux-5.15.0-x86_64"
  }
}
```

## Carbon Intensity

CodeCarbon uses regional carbon intensity data. Default is USA average (~0.38 kg CO2/kWh).

To use a different region:
```python
tracker = CarbonTracker(
    project_name="my_experiment",
    country_iso_code="DEU"  # Germany
)
```

Common ISO codes:
- `USA` - United States
- `GBR` - United Kingdom
- `DEU` - Germany
- `FRA` - France (low carbon due to nuclear)
- `CHN` - China

## Troubleshooting

### CodeCarbon not detecting GPU

Ensure `nvidia-smi` is accessible:
```bash
nvidia-smi
```

### Permission errors on Linux

CodeCarbon may need access to Intel RAPL for CPU energy:
```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```

### Using without CodeCarbon

The tracker falls back to manual timing if CodeCarbon is unavailable:
```python
# Install only basic dependencies
pip install pandas numpy

# Tracker will work but only report duration
tracker = CarbonTracker(project_name="my_experiment")
```

## Contributing Results

To contribute benchmark results:

1. Run benchmarks following this protocol
2. Verify results are saved in `benchmarks/results/`
3. Include hardware configuration
4. Submit a pull request with your results

Please ensure your results are reproducible and include all required metadata.
