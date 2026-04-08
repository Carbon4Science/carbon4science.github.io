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

Look for JSON files in `<Task>/results/outputs/`:
```bash
ls <Task>/results/outputs/*.json
```

If no results are found, tell the user to run benchmarks first (`/benchmark` or `./Retro/benchmarks/run.sh`).

### Step 2: Generate plots

Run `plot_results.py` to generate combined and panel plots. **Always normalize per N samples** where N is chosen by the task leader:

```bash
# Combined view, normalized (RECOMMENDED)
# N is task-specific: Retro uses 500, other tasks choose their own
python Retro/benchmarks/plot_results.py --task <Task> --combined --norm <N>

# Per-metric panel view, normalized
python Retro/benchmarks/plot_results.py --task <Task> --norm <N>

# Per-molecule normalization
python Retro/benchmarks/plot_results.py --task <Task> --combined --norm 1

# Raw (unnormalized) values
python Retro/benchmarks/plot_results.py --task <Task> --combined --no-normalize

# Filter to a specific sample count
python Retro/benchmarks/plot_results.py --task <Task> --combined --samples 500
```

### Step 3: Report output locations

Plots are saved to `<Task>/results/figures/`:
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

If a model appears as a gray "x" marker, it needs a style entry in `Retro/benchmarks/plot_results.py`:

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

## Plot Style Rules

**CRITICAL**: All plots MUST follow these rules:

### Font sizes
- **Model name labels: fontsize >= 20.** This is non-negotiable. Labels must be clearly readable when the figure is scaled to a single paper column.
- Axis labels: fontsize >= 22
- Axis tick labels: fontsize >= 18
- Subplot titles: fontsize >= 26
- Legend: fontsize >= 16
- R² annotations: fontsize >= 22

Set these globally at the top of every plotting script:
```python
plt.rcParams.update({
    'font.size': 22, 'axes.titlesize': 26, 'axes.labelsize': 22,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 16,
})
```

### Label overlap prevention
- **Labels MUST NOT overlap with each other or with data points.** Use `adjustText` with strong repulsion forces:
```python
from adjustText import adjust_text
texts = [ax.text(x, y, name, fontsize=20) for ...]
adjust_text(texts, ax=ax,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            force_text=(2.0, 2.0), force_points=(2.5, 2.5),
            expand_text=(1.4, 1.6), expand_points=(1.6, 1.6),
            lim=300)
```
- Use large figure sizes (e.g., `figsize=(30, 18)` for 2x3 subplots) to give `adjust_text` room to work.
- For log-scale axes where `adjust_text` behaves badly, convert to log-transformed values on linear axes instead (e.g., plot `log10(size)` on a linear axis rather than `size` on a log axis).

### Layout
- Use 2 rows x 3 columns for 6-task subplots (paper-friendly width)
- Figure size: `(30, 18)` for 2x3 subplots, `(20, 8)` for 1x2 panels
- Always use `dpi=300` for publication quality
- Marker sizes should be large enough to distinguish shapes: minimum 80, scale with model size up to 600+

### Units
- Carbon: always use "g CO₂ eq / job" (not "g CO₂" or "g")
- MatGen metric: "mSUN (%)" (not "mSUN" or "SUN")
- Retro: "Top-50 Exact Match Acc. (%)"
- Forward: "Top-3 Exact Match Acc. (%)"
- For log-transformed axes, label as "log₁₀(...)"

## Notes
- Plots use log-scale x-axis to handle the large range in cost across models
- **Always normalize** using `--norm N` where N is chosen by the task leader (e.g., Retro uses 500)
- `--norm N` normalizes cost to per-N samples (e.g., `--norm 1` for per molecule)
- `--no-normalize` shows raw (total) values
- `--samples N` filters to only results with exactly N samples (useful when you have multiple runs at different sizes)
