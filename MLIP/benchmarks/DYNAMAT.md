# Dynamat benchmark

The new benchmark is implemented in `MLIP/dynamat_benchmark.py`.  It is
separate from `MLIP/production/run_production_md.py`, which remains the
legacy LGPS RDF/MSD experiment and is not deleted.

The runner reads the first frame of each of the 17 groups in the checked-in
dynamat HDF5 file, runs the existing NVT Nose-Hoover protocol for 10 ps
equilibration and 50 ps production at 2 fs, and repeats each structure with
seeds 42, 43, and 44.  CarbonTracker measures the production segment, as in
the previous production runner.

For each model, raw output is stored as:

```text
MLIP/dynamat_results/{model}/{structure}/seed-{seed}.traj
MLIP/dynamat_results/{model}/{structure}/result.json
MLIP/dynamat_results/{model}/summary.json
```

The 17 initial frames are also saved once, independently of any model run:

```text
MLIP/dynamat_initial_structures/{structure}.extxyz
```

`summary.json` contains the per-structure results and
`carbon_mean_over_structures`.  No RDF/MSD analysis is run.  Static Matbench
Discovery CMDS values belong in `MLIP/dynamat_metrics.py`; edit the entries in
the following form when the values are supplied:

```python
"eSEN": {"CMDS": 0.123456},
```

Examples:

```bash
python MLIP/dynamat_benchmark.py --model CHGNet --track-carbon
python MLIP/dynamat_benchmark.py --model all --track-carbon
MLIP/benchmarks/run_dynamat.sh CHGNet
MLIP/benchmarks/run_dynamat.sh DPA4 --dpa4-checkpoint /path/to/dpa4-model.pt
```

The DPA4 model is intentionally not aliased to the old DPA3 checkpoint.
