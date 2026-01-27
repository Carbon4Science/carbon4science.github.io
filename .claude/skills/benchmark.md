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
```

## Instructions

When the user invokes this skill:

1. **Identify the model and options:**
   - Model name (required): LocalRetro, neuralsym, RetroBridge, Chemformer, RSGPT
   - `--limit N`: Number of test samples (default: full dataset)
   - `--track_carbon`: Enable carbon tracking (default: off)
   - `--metrics`: Metrics to compute (default: task defaults)

2. **Activate the correct conda environment:**
   ```bash
   source /Users/admin/opt/anaconda3/etc/profile.d/conda.sh
   conda activate <env_name>
   ```

   Environment mapping:
   - neuralsym → `neuralsym`
   - LocalRetro → `rdenv`
   - RetroBridge → `retrobridge`
   - Chemformer → `chemformer`
   - RSGPT → `gpt`

3. **Run the benchmark:**
   ```bash
   python benchmarks/run_benchmark.py \
       --task Retrosynthesis \
       --model <model_name> \
       --limit <N> \
       --track_carbon \
       --output benchmarks/results/<model>_benchmark.json
   ```

4. **Report results:**
   - Show accuracy metrics (top_1, top_10, top_50)
   - Show carbon metrics (duration, energy, CO2)
   - Note the output file location

## Notes
- Always run from the repository root directory
- Results are saved to `benchmarks/results/`
- Use `--limit` for quick tests to avoid long runtimes
