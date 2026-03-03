# Skill: Run Benchmark

Run a carbon-tracked benchmark on a model.

## Usage
```
/benchmark [model] [--limit N] [--track_carbon]
```

## Examples
```
/benchmark LocalRetro --limit 100 --track_carbon
/benchmark neuralsym --limit 500
/benchmark MACE --limit 1000 --track_carbon
```

## Instructions

When the user invokes this skill:

1. **Identify the model and options:**
   - Model name (required)
   - `--limit N`: Number of test samples (default: full dataset)
   - `--track_carbon`: Enable carbon tracking (default: off)
   - `--metrics`: Metrics to compute (default: task defaults)
   - `--task`: Task name (default: auto-detect from model name)

2. **Activate the correct conda environment:**
   ```bash
   source /opt/conda/etc/profile.d/conda.sh
   conda activate <env_name>
   ```

   Known environment mappings (Retro):
   - neuralsym -> `neuralsym`
   - LocalRetro -> `rdenv`
   - RetroBridge -> `retrobridge`
   - Chemformer -> `chemformer`
   - RSGPT -> `gpt`

   For other tasks, check the model's `environment.yml` or `benchmarks/run.sh`.

3. **Run the benchmark:**
   ```bash
   python benchmarks/run_benchmark.py \
       --task <Task> \
       --model <model_name> \
       --limit <N> \
       --track_carbon \
       --output benchmarks/results/<Task>/<model>_<N>.json
   ```

4. **Report results:**
   - Show accuracy metrics (task-specific)
   - Show carbon metrics (duration, energy, CO2, peak GPU memory)
   - Note the output file location

5. **Generate plots (normalized):**
   After all models have been benchmarked, choose a normalization N (task leader decides, e.g., Retro uses 500):
   ```bash
   python benchmarks/plot_results.py --task <Task> --combined --norm <N>
   python benchmarks/plot_results.py --task <Task> --norm <N>
   ```

6. **Update READMEs (follow Rule 11 Reporting Format):**

   In `benchmarks/README.md`, add a `## <Task> Model Comparison` section with three tables:

   - **Model Specifications** — Year, Venue, Architecture, Parameters, Model Size, GPU Memory (MB)
   - **Accuracy** — Task-specific metrics, sorted by primary metric (descending), best model bolded
   - **Carbon Efficiency** — Duration (s), Speed (s/mol), Energy (Wh), CO2 (g), CO2 Intensity (g/s), sorted by Duration (ascending)
   - **Key Observations** — 3–5 bullets on accuracy-efficiency tradeoffs

   In root `README.md`, add a `## <Task> Results` section with:
   - A combined table with costs normalized per N samples
   - A reference to the accuracy-vs-carbon plot
   - Key insights (3–5 bullets)

   See the Retro sections as reference.

## Notes
- Always run from the repository root directory
- Results are saved to `benchmarks/results/<Task>/<model>_<N>.json`
- **Submit via Slurm** for full-dataset runs: `sbatch --job-name=<Model> benchmarks/slurm_benchmark.sh <Model>`
- Use `--limit` for quick smoke tests only
- For parallel GPU execution on different models, use `CUDA_VISIBLE_DEVICES=<N>` prefix to assign each model to a different GPU
