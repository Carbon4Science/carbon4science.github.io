# MLIP Task — Claude Code Context

This file preserves all research findings and the implementation plan for the MLIP benchmarking pipeline. Claude Code reads this automatically when working in this directory.

---

## Current State

- `evaluate.py` — Skeleton with `METRICS`, `load_test_data()`, `evaluate()` (all stubs)
- `LGPS_mp-696128.cif` — Test structure (50 atoms, Li10GeP2S12)
- `md_chgnet040.py` — Example MD script for reference
- 8 model subdirectories created: `eSEN/`, `NequIP/`, `Nequix/`, `DPA3/`, `SevenNet/`, `MACE/`, `ORB/`, `CHGNet/` (each with `CLAUDE.md`)
- No `Inference.py` files, no `run_md.py`, nothing registered in benchmark infrastructure

## Goal

Run NVT MD simulations on the Li10GeP2S12 structure using 8 MLIP models and measure computational cost (time, energy, CO2). No accuracy metrics for now.

---

## Implementation Plan

### Architecture

MLIP flows through the **standard benchmark pipeline** (Rules 3, 11):

```
slurm_benchmark.sh → run_benchmark.py --task MLIP --model CHGNet → MLIP/CHGNet/Inference.py:run()
```

Key design decisions:
- Each model's `Inference.py` implements the standard `run(input_data, top_k)` interface (Rule 1)
- `run()` wraps calculator creation + MD execution internally
- `MLIP/run_md.py` is a **shared helper library** with the MD simulation logic
- `slurm_benchmark.sh` auto-detects MLIP task from model name and passes `--task MLIP`
- `evaluate.py` returns empty accuracy metrics (to be implemented later)
- Speed metrics (`steps_per_second`, `ns_per_day`) stored in `speed` field of results JSON

### Inference.py Interface (Rule 1)

Each model's `Inference.py` implements the **standard `run()` function**:

```python
from typing import List, Dict

def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Run NVT MD simulation on a structure using this model's calculator.

    Args:
        input_data: Path to CIF structure file
        top_k: Not used for MLIP (kept for interface compatibility)

    Returns:
        [{'input': 'path.cif', 'predictions': [{'output': 'md_completed', 'score': 1.0,
          'steps_per_second': ..., 'ns_per_day': ..., 'total_steps': ..., 'elapsed_seconds': ...}]}]
    """
    calc = _get_calculator()
    from MLIP.run_md import run_md
    md_results = run_md(input_data, calc)
    return [{'input': str(input_data), 'predictions': [
        {'output': 'md_completed', 'score': 1.0, **md_results}
    ]}]
```

Each model has an internal `_get_calculator(device=None)` helper. Device defaults to `cuda:0` with CPU fallback (Rule 6).

### run_md.py — Shared MD Helper

`MLIP/run_md.py` provides shared MD simulation logic:

```python
def run_md(structure_path, calculator, steps=5000, temperature_K=600,
           timestep_fs=2.0, ttime_fs=25.0, seed=42):
    """Run NVT MD simulation. Returns speed metrics dict."""
```

Essential MD logic (from `md_chgnet040.py` reference):
```python
atoms = read(structure_path)
atoms.calc = calculator
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
Stationary(atoms); ZeroRotation(atoms)
md = NoseHoover(atoms, timestep=timestep_fs*units.fs,
                temperature_K=temperature_K, ttime=ttime_fs*units.fs)
md.run(steps)
```

### Files to Create

1. **`MLIP/run_md.py`** — Shared MD helper library (not standalone runner)
2. **8x `MLIP/<Model>/Inference.py`** — Standard `run()` interface wrapping calculator + MD
3. **`MLIP/DPA3/download_model.sh`** — wget checkpoint from Figshare
4. **`MLIP/NequIP/setup_model.sh`** — nequip-compile step

Note: `environment.yml` per model will be added after environments are created and tested.

### Files to Modify

1. **`MLIP/evaluate.py`** — `load_test_data()` returns structures with `'product'` key for pipeline compatibility; `evaluate()` extracts speed metrics, returns empty accuracy
2. **`benchmarks/run_benchmark.py`** — Add 8 MLIP models to `TASKS["MLIP"]["models"]`; add `speed` and `structure` fields to results JSON
3. **`benchmarks/run.sh`** — Add 8 MLIP models to `MODEL_ENVS`; add `MODEL_TASK` dict for task detection
4. **`benchmarks/slurm_benchmark.sh`** — Add 8 MLIP models to `MODEL_ENVS`; auto-detect task from model name; pass `--task MLIP` to `run_benchmark.py`
5. **`benchmarks/setup_envs.sh`** — Add 8 `setup_*()` functions
6. **`benchmarks/configs/models.yaml`** — Add 8 MLIP model config blocks
7. **`benchmarks/plot_results.py`** — Add 8 `MODEL_STYLES` entries; add MLIP-specific plotting support

### Dummy Accuracy for Plotting

Since MLIP accuracy metrics (energy_mae, force_mae, etc.) are deferred, provide dummy values for plotting via a supplementary JSON:

**`benchmarks/configs/mlip_accuracy.json`**:
```json
{
  "CHGNet": {"energy_mae_meV": 30.0, "force_mae_meV_A": 75.0},
  "MACE": {"energy_mae_meV": 22.0, "force_mae_meV_A": 48.0}
}
```

`plot_results.py` merges these values into results when plotting MLIP task. Update this file manually with values from literature or separate evaluation runs.

### Results JSON Schema

```json
{
  "task": "MLIP",
  "model": "CHGNet",
  "num_samples": 1,
  "top_k": 10,
  "model_params": 412525,
  "metrics": [],
  "accuracy": {},
  "speed": {
    "steps_per_second": 125.3,
    "ns_per_day": 21.65,
    "total_steps": 5000,
    "elapsed_seconds": 39.9
  },
  "structure": {
    "formula": "Li20Ge2P4S24",
    "num_atoms": 50,
    "source_file": "LGPS_mp-696128.cif"
  },
  "md_params": {
    "ensemble": "NVT",
    "thermostat": "NoseHoover",
    "temperature_K": 600,
    "timestep_fs": 2.0,
    "ttime_fs": 25.0,
    "seed": 42
  },
  "carbon": { "..." }
}
```

### Implementation Order

1. Create directories: `MLIP/{eSEN,NequIP,Nequix,DPA3,SevenNet,MACE,ORB,CHGNet}/`, `MLIP/data/`, `benchmarks/results/MLIP/`
2. Write per-model `CLAUDE.md` files (8 files)
3. Update `MLIP/evaluate.py` — `load_test_data()` returns structures, `evaluate()` extracts speed
4. Write `MLIP/run_md.py` — shared MD helper
5. Write all 8 `Inference.py` files with standard `run()` interface
6. Write helper scripts (`DPA3/download_model.sh`, `NequIP/setup_model.sh`)
7. Update benchmark infrastructure:
   - `benchmarks/run_benchmark.py` — TASKS dict + speed/structure fields
   - `benchmarks/run.sh` — MODEL_ENVS + task detection
   - `benchmarks/slurm_benchmark.sh` — MODEL_ENVS + task auto-detection
   - `benchmarks/setup_envs.sh` — 8 setup functions
   - `benchmarks/configs/models.yaml` — 8 config blocks
   - `benchmarks/plot_results.py` — MODEL_STYLES + MLIP support
8. Verify with a quick test:
   ```bash
   sbatch --job-name=CHGNet benchmarks/slurm_benchmark.sh CHGNet
   # Check: benchmarks/results/MLIP/CHGNet_1.json
   ```
9. Run all MLIP benchmarks via slurm
10. Create `benchmarks/configs/mlip_accuracy.json` with dummy accuracy values
11. Generate plots: `python benchmarks/plot_results.py --task MLIP --combined`
12. Update `MLIP/README.md` with results table and figures

---

## Model Research — Complete Reference

### Model 1: eSEN-30M-MP

- **Directory:** `MLIP/eSEN/`
- **Conda env:** `esen`
- **Package:** `pip install fairchem-core` (Python >=3.10, torch ~=2.8)
- **Params:** ~30M
- **Checkpoint:** HuggingFace gated — requires access at `facebook/OMAT24`, then `huggingface-cli login`

**ASE Calculator:**
```python
from huggingface_hub import hf_hub_download
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

checkpoint_path = hf_hub_download(repo_id="facebook/OMAT24", filename="esen_30m_mptrj.pt")
predictor = load_predict_unit(path=checkpoint_path, device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms.calc = calc
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/A
stress = atoms.get_stress()            # Voigt, eV/A^3
```

**Environment setup:**
```bash
conda create -n esen python=3.11 -y
conda activate esen
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124
pip install fairchem-core ase codecarbon
# Then: huggingface-cli login
```

**Notes:**
- `task_name="omat"` for inorganic materials (MPtrj-trained)
- FAIRChem v2 API (v1 used `OCPCalculator`, now deprecated)
- Alternative: Meta's UMA models can be loaded by name without gated access

---

### Model 2: NequIP-MP-L

- **Directory:** `MLIP/NequIP/`
- **Conda env:** `nequip`
- **Package:** `pip install nequip` (Python >=3.10, torch >=2.6)
- **Checkpoint:** Auto-downloads from `nequip.net:mir-group/NequIP-MP-L:0.1`

**ASE Calculator (requires compile step first):**
```bash
# Step 1: Compile (one-time)
nequip-compile nequip.net:mir-group/NequIP-MP-L:0.1 NequIP-MP-L.nequip.pt2 \
  --mode aotinductor --device cuda --target ase
```
```python
# Step 2: Use in Python
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model(
    compile_path="NequIP-MP-L.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

**Alternative (skip compilation, internal API):**
```python
calc = NequIPCalculator._from_saved_model(
    model_path="nequip.net:mir-group/NequIP-MP-L:0.1",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
```

**Environment setup:**
```bash
conda create -n nequip python=3.11 -y
conda activate nequip
pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install nequip ase codecarbon
```

**Notes:**
- Key deps: e3nn >=0.5.9 <0.6.0, matscipy, lightning
- Models cached at `~/.nequip/model_cache`
- `--target ase` flag required for AOTInductor compilation

---

### Model 3: Nequix MP

- **Directory:** `MLIP/Nequix/`
- **Conda env:** `nequix`
- **Package:** `pip install nequix` (Python >=3.10, **JAX-based** not PyTorch)
- **Params:** ~708K
- **Checkpoint:** Auto-downloads to `~/.cache/nequix/models/`

**ASE Calculator:**
```python
from nequix.calculator import NequixCalculator
calc = NequixCalculator("nequix-mp-1", backend="jax")
# Also supports backend="torch"
```

**Environment setup:**
```bash
conda create -n nequix python=3.10 -y
conda activate nequix
pip install nequix ase codecarbon
# For GPU kernel acceleration (optional):
# pip install nequix[oeq]
```

**Notes:**
- JAX-based by default — device selection via `CUDA_VISIBLE_DEVICES`, not `device="cuda:0"`
- In `run_md.py`, set `os.environ["CUDA_VISIBLE_DEVICES"]` before importing nequix
- Also supports `backend="torch"` if PyTorch backend preferred
- Available models: `nequix-mp-1`, `nequix-omat-1`, `nequix-oam-1`
- Sets `jax.config.update("jax_default_matmul_precision", "highest")` for cross-GPU consistency

---

### Model 4: DPA-3.1-MPtrj

- **Directory:** `MLIP/DPA3/`
- **Conda env:** `deepmd`
- **Package:** `pip install deepmd-kit[torch]` (Python >=3.9)
- **Params:** ~4.81M
- **Checkpoint:** Download from Figshare

**ASE Calculator:**
```python
from deepmd.calculator import DP
calc = DP(model="dpa-3.1-mptrj.pth")
```

**Download checkpoint:**
```bash
wget -O MLIP/DPA3/dpa-3.1-mptrj.pth https://figshare.com/ndownloader/files/55141124
```

**Environment setup:**
```bash
conda create -n deepmd python=3.10 -y
conda activate deepmd
pip install torch torchvision torchaudio
pip install "deepmd-kit[torch]" ase codecarbon
```

**Notes:**
- `type_dict` is auto-inferred from model's type map
- No `head` parameter needed for DPA-3.1-MPtrj (single-task model)
- For the multi-task DPA-3.1-3M model, use `head="MP_traj_v024_alldata_mixu"`

---

### Model 5: SevenNet-l3i5

- **Directory:** `MLIP/SevenNet/`
- **Conda env:** `sevennet`
- **Package:** `pip install sevenn` (Python >=3.8, torch >=2.0)
- **Checkpoint:** Bundled in pip package (no download needed)

**ASE Calculator:**
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model="7net-l3i5", device="cuda")
```

**Environment setup:**
```bash
conda create -n sevennet python=3.10 -y
conda activate sevennet
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.x.x+cu121.html
pip install sevenn ase codecarbon
```

**Notes:**
- Requires `torch_geometric` extensions (scatter, sparse, cluster) — install AFTER PyTorch
- Keywords `"7net-l3i5"` or `"sevennet-l3i5"` both work
- No `modal` parameter needed (unlike multi-fidelity `7net-omni`)
- Architecture: l_max=3, 5 interaction layers
- Properties: energy, forces, stress, per-atom energies

---

### Model 6: MACE-MP-0

- **Directory:** `MLIP/MACE/`
- **Conda env:** `mace`
- **Package:** `pip install mace-torch` (NOT `mace` — that's a different package!)
- **Checkpoint:** Auto-downloads and caches at `~/.cache/mace/`

**ASE Calculator:**
```python
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
```

**Environment setup:**
```bash
conda create -n mace python=3.10 -y
conda activate mace
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install mace-torch ase codecarbon
```

**Notes:**
- CRITICAL: pins `e3nn==0.4.4` — will conflict with other models' e3nn versions
- Model sizes: `"small"`, `"medium"`, `"large"`
- `default_dtype="float64"` for better accuracy; `"float32"` for speed
- Also available: `mace_mp(model="medium-mpa-0")` (MACE-MPA-0, improved version)

---

### Model 7: ORB v2 MPtrj

- **Directory:** `MLIP/ORB/`
- **Conda env:** `orb`
- **Package:** `pip install orb-models` (Python >=3.10, torch >=2.6)
- **Checkpoint:** Auto-downloads from S3

**ASE Calculator:**
```python
import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
orbff = pretrained.orb_mptraj_only_v2(device=device)
calc = ORBCalculator(orbff, device=device)
```

**Environment setup:**
```bash
conda create -n orb python=3.11 -y
conda activate orb
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install orb-models ase codecarbon
```

**Notes:**
- `orb_mptraj_only_v2` trained on MPTraj only (for Matbench Discovery reproduction)
- For general use: `pretrained.orb_v2()` (MPTraj + Alexandria)
- Architecture: 256 latent dim, 15 message passing steps, 6.0 A cutoff
- Supports `torch.compile` (auto-detected)

---

### Model 8: CHGNet

- **Directory:** `MLIP/CHGNet/`
- **Conda env:** `chgnet`
- **Package:** `pip install chgnet` (Python >=3.10, torch >=2.4.1)
- **Params:** ~412K
- **Checkpoint:** Auto-downloads

**ASE Calculator:**
```python
from chgnet.model.dynamics import CHGNetCalculator
calc = CHGNetCalculator(use_device="cuda")
```

**Alternative (direct prediction via pymatgen Structure):**
```python
from chgnet.model import CHGNet
chgnet = CHGNet.load()
prediction = chgnet.predict_structure(structure)  # returns dict with e, f, s, m
```

**Environment setup:**
```bash
conda create -n chgnet python=3.10 -y
conda activate chgnet
pip install torch>=2.4.1
pip install chgnet ase codecarbon
```

**Notes:**
- Uses pymatgen `Structure` internally (converts from ASE Atoms in calculator)
- `CHGNetCalculator(use_device=...)` not `device=...`
- Model versions: `"0.3.0"` (default), `"0.2.0"`, `"r2scan"`
- Properties: energy, forces, stress, magnetic moments

---

## Key Environment Conflicts

These models CANNOT share a conda environment due to dependency conflicts:

| Dependency | eSEN | NequIP | MACE | SevenNet | Nequix |
|-----------|------|--------|------|----------|--------|
| e3nn | >=0.5 | >=0.5.9,<0.6 | ==0.4.4 | >0.5 | e3nn-jax |
| torch | ~=2.8 | >=2.6 | >=1.12 | >=2.0 | (JAX) |
| Framework | PyTorch | PyTorch | PyTorch | PyTorch | JAX |

This is why each model has its own conda environment (project Rule 2).

## Conda Environment Summary

| Model | Env Name | Key Package | Python |
|-------|----------|-------------|--------|
| eSEN | `esen` | `fairchem-core` | >=3.10 |
| NequIP | `nequip` | `nequip` | >=3.10 |
| Nequix | `nequix` | `nequix` | >=3.10 |
| DPA3 | `deepmd` | `deepmd-kit[torch]` | >=3.9 |
| SevenNet | `sevennet` | `sevenn` | >=3.8 |
| MACE | `mace` | `mace-torch` | >=3.7 |
| ORB | `orb` | `orb-models` | >=3.10 |
| CHGNet | `chgnet` | `chgnet` | >=3.10 |

---

## Benchmark Infrastructure Files to Update

### `benchmarks/run.sh` (lines 14-20)
Add to `MODEL_ENVS`:
```bash
["eSEN"]="esen"
["NequIP"]="nequip"
["Nequix"]="nequix"
["DPA3"]="deepmd"
["SevenNet"]="sevennet"
["MACE"]="mace"
["ORB"]="orb"
["CHGNet"]="chgnet"
```
Add `MODEL_TASK` dict for task auto-detection.

### `benchmarks/slurm_benchmark.sh`
Add 8 MLIP models to `MODEL_ENVS`. Add `MODEL_TASK` dict to auto-detect task from model name and pass `--task MLIP` to `run_benchmark.py`.

### `benchmarks/setup_envs.sh`
Add 8 `setup_*()` functions (see environment setup sections above, each appending `ase codecarbon`).

### `benchmarks/configs/models.yaml`
Add 8 MLIP model blocks with `env`, `task: MLIP`, `full_name`, `gpu_memory_mb`.

### `benchmarks/run_benchmark.py` (line 48-51)
Add models to `TASKS["MLIP"]["models"]` dict. Add `speed` and `structure` fields to results JSON output.

### `benchmarks/plot_results.py` (MODEL_STYLES)
```python
"eSEN":     {"color": "#E91E63", "marker": "o", "params": "30M",  "year": 2024, "venue": "Meta FAIR"},
"NequIP":   {"color": "#3F51B5", "marker": "s", "params": "~5M",  "year": 2022, "venue": "Nat. Commun."},
"Nequix":   {"color": "#009688", "marker": "D", "params": "708K", "year": 2024, "venue": "arXiv"},
"DPA3":     {"color": "#FF5722", "marker": "^", "params": "4.8M", "year": 2024, "venue": "arXiv"},
"SevenNet": {"color": "#795548", "marker": "P", "params": "~5M",  "year": 2024, "venue": "JCTC"},
"MACE":     {"color": "#607D8B", "marker": "h", "params": "~10M", "year": 2023, "venue": "NeurIPS"},
"ORB":      {"color": "#CDDC39", "marker": "v", "params": "~30M", "year": 2024, "venue": "arXiv"},
"CHGNet":   {"color": "#FF9800", "marker": "*", "params": "412K", "year": 2023, "venue": "Nat. Mach. Intell."},
```
