#!/usr/bin/env python3
"""One-time data preparation: sample N structures from the Matbench Discovery
WBM test set (unique-prototypes subset) and save as a portable extxyz file
so per-model conda environments don't need `matbench-discovery` installed.

Run once inside the `matbench` conda env:

    conda activate matbench
    python MLIP/relaxation/prepare_data.py --n 100 --seed 42

Outputs:
    MLIP/relaxation/data/wbm_subset_<N>.extxyz
    MLIP/relaxation/data/wbm_subset_<N>_ids.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample WBM subset for relaxation benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of structures to sample")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument(
        "--unique_prototypes_only",
        action="store_true",
        default=True,
        help="Restrict to the MBD unique_prototype subset (default: True)",
    )
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from ase.io import write as ase_write
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WBM summary from {DataFiles.wbm_summary.path}")
    df_wbm = pd.read_csv(DataFiles.wbm_summary.path)
    print(f"  Total WBM entries: {len(df_wbm):,}")

    if args.unique_prototypes_only and "unique_prototype" in df_wbm.columns:
        mask = df_wbm["unique_prototype"].astype(bool)
        df_pool = df_wbm[mask].reset_index(drop=True)
        print(f"  Filtered to unique_prototype==True: {len(df_pool):,}")
    else:
        df_pool = df_wbm

    if len(df_pool) < args.n:
        raise SystemExit(f"Pool has only {len(df_pool)} rows, cannot sample {args.n}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df_pool), size=args.n, replace=False)
    idx.sort()
    sampled_ids = df_pool.iloc[idx]["material_id"].tolist()
    id_set = set(sampled_ids)
    print(f"Sampled {len(sampled_ids)} material_ids (seed={args.seed})")

    print(f"Loading WBM initial atoms from {DataFiles.wbm_initial_atoms.path}")
    all_atoms = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path)
    print(f"  Total atoms loaded: {len(all_atoms):,}")

    kept = []
    for at in all_atoms:
        mid = at.info.get("material_id") or at.info.get("name") or at.info.get("id")
        if mid in id_set:
            at.info["material_id"] = mid
            kept.append(at)
            if len(kept) == len(id_set):
                break

    if len(kept) != len(sampled_ids):
        got = {a.info["material_id"] for a in kept}
        missing = id_set - got
        raise SystemExit(
            f"Missing {len(missing)} sampled ids in WBM atoms zip. "
            f"First few missing: {list(missing)[:5]}"
        )

    # Order `kept` to match `sampled_ids` order for reproducibility
    by_id = {a.info["material_id"]: a for a in kept}
    kept = [by_id[mid] for mid in sampled_ids]

    out_xyz = out_dir / f"wbm_subset_{args.n}.extxyz"
    ase_write(out_xyz, kept, format="extxyz")
    print(f"Wrote {out_xyz}")

    out_json = out_dir / f"wbm_subset_{args.n}_ids.json"
    meta = {
        "n": args.n,
        "seed": args.seed,
        "unique_prototypes_only": bool(args.unique_prototypes_only),
        "source": "matbench_discovery.DataFiles.wbm_initial_atoms",
        "wbm_summary_rows": int(len(df_wbm)),
        "pool_size": int(len(df_pool)),
        "material_ids": sampled_ids,
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
