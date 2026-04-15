# Branch Example

This directory shows the expected structure on a **task branch** (e.g. `Retro`, `Forward`, `MolGen`).

Each task branch is an **orphan branch** — it contains only the task's files, not main's files.

Contributors work on these branches. Only result JSONs and README leaderboard rows get merged into `main`.

## What goes where

```
main (maintained by Shaun)
├── results/<Task>/<model>_<N>.json   ← merged from branches
├── README.md                         ← leaderboard tables
├── analysis/                         ← figures and analysis
└── branch-example/                   ← this template

<Task> branch (contributors work here, orphan branch)
├── README.md                         ← task description + results
├── evaluate.py                       ← task-specific metrics
├── data/                             ← test dataset
├── benchmarks/                       ← benchmark infrastructure
│   ├── run_benchmark.py              ← Python benchmark runner
│   ├── run.sh                        ← conda env switching runner
│   ├── slurm_benchmark.sh            ← Slurm job template
│   ├── carbon_tracker.py             ← carbon/energy measurement
│   └── configs/models.yaml           ← model registry
├── results/                          ← JSON result files
│   └── <model>_<N>.json
└── <Model>/                          ← one per model (contributor adds this)
    ├── Inference.py                  ← uniform run() interface (REQUIRED)
    └── environment.yml               ← conda env spec (REQUIRED)
```

## Contributor workflow

1. Check out the task branch: `git checkout Retro`
2. Create `<YourModel>/Inference.py` with the uniform `run()` interface
3. Create `<YourModel>/environment.yml` with your conda env spec
4. Run the benchmark: `./benchmarks/run.sh --model <YourModel> --track_carbon --output results/<yourmodel>_<N>.json`
5. Open a PR **to the task branch** with your model folder + result JSON
6. Shaun reviews and merges the result JSON to `main`

## See also

- `ExampleTask/` — a complete example of the branch structure
- `ExampleTask/ExampleModel/` — a minimal example of a model contribution
- `ExampleTask/benchmarks/` — benchmark infrastructure template
