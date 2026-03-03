# Skill: Generate Plots

Generate accuracy vs cost plots from benchmark results.

## Usage
```
/plot [task] [--samples N] [--xaxis metric]
```

## Examples
```
/plot Retro
/plot MolGen --samples 500
/plot MatGen --xaxis energy_wh
```

## Instructions

When the user invokes this skill:

### Step 1: Check that results exist

Look for JSON files in `benchmarks/results/<Task>/`:
```bash
ls benchmarks/results/<Task>/*.json
```

If no results are found, tell the user to run benchmarks first (`/benchmark` or `./benchmarks/run.sh`).

### Step 2: Generate plots

Run `plot_results.py` to generate combined and panel plots. **Always normalize per N samples** where N is chosen by the task leader:

```bash
# Combined view, normalized (RECOMMENDED)
# N is task-specific: Retro uses 500, other tasks choose their own
python benchmarks/plot_results.py --task <Task> --combined --norm <N>

# Per-metric panel view, normalized
python benchmarks/plot_results.py --task <Task> --norm <N>

# Per-molecule normalization
python benchmarks/plot_results.py --task <Task> --combined --norm 1

# Raw (unnormalized) values
python benchmarks/plot_results.py --task <Task> --combined --no-normalize

# Filter to a specific sample count
python benchmarks/plot_results.py --task <Task> --combined --samples 500
```

### Step 3: Report output locations

Plots are saved to `benchmarks/figures/<Task>/`:
- `accuracy_vs_carbon_combined.png` — CO2 emissions (g) on x-axis
- `accuracy_vs_energy_combined.png` — Energy (Wh) on x-axis
- `accuracy_vs_speed_combined.png` — Time (s) on x-axis
- `accuracy_vs_carbon_panels.png` — CO2, panel per metric
- `accuracy_vs_energy_panels.png` — Energy, panel per metric
- `accuracy_vs_speed_panels.png` — Time, panel per metric

With `--samples N`, filenames include the sample count (e.g., `accuracy_vs_carbon_combined_500.png`).

## Available x-axis options

| `--xaxis` value | X-axis label | Description |
|----------------|-------------|-------------|
| `emissions_g_co2` | CO2 emissions (g) | Carbon footprint (default) |
| `energy_wh` | Energy (Wh) | Total energy consumption |
| `duration_seconds` | Time (s) | Wall-clock inference time |

## Adding MODEL_STYLES for new tasks

If a model appears as a gray "x" marker, it needs a style entry in `benchmarks/plot_results.py`:

```python
MODEL_STYLES = {
    "MyModel": {
        "color": "#2196F3",   # Hex color
        "marker": "o",        # matplotlib marker (o, s, D, ^, P, *, v, etc.)
        "params": "10M",      # Parameter count string
        "year": 2024,         # Publication year
        "venue": "NeurIPS",   # Publication venue
    },
}
```

Choose distinct colors and markers so models are visually distinguishable.

## Notes
- Plots use log-scale x-axis to handle the large range in cost across models
- **Always normalize** using `--norm N` where N is chosen by the task leader (e.g., Retro uses 500)
- `--norm N` normalizes cost to per-N samples (e.g., `--norm 1` for per molecule)
- `--no-normalize` shows raw (total) values
- `--samples N` filters to only results with exactly N samples (useful when you have multiple runs at different sizes)
