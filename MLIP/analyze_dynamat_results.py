#!/usr/bin/env python3
"""Create a wide pandas/CSV summary from completed dynamat results.

The output has structures as columns and a two-level row index:

    metric                         model      structure-A  structure-B ...
    num atoms                      (blank)          ...
    co2_emission (g)               CHGNet           ...
    co2_emission (g/1 ns)          CHGNet           ...
    co2_emission (g/1 ns/1 atom) CHGNet           ...

CO2 values are the mean over the three seeds stored in each structure's
``carbon_mean_over_seeds`` result.  The normalization uses each result's
production time and the corresponding structure's atom count.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from ase.io import read


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "MLIP/dynamat_results"
DEFAULT_INITIAL_STRUCTURES = ROOT / "MLIP/dynamat_initial_structures"
DEFAULT_CSV = DEFAULT_RESULTS / "dynamat_carbon_summary.csv"
DEFAULT_PICKLE = DEFAULT_RESULTS / "dynamat_carbon_summary.pkl"


def _casefold_sorted(values):
    return sorted(values, key=lambda value: value.casefold())


def _load_num_atoms(structure, model_dir, initial_structures_dir):
    cif_path = Path(initial_structures_dir) / f"{structure}.cif"
    if cif_path.exists():
        return len(read(str(cif_path), format="cif"))

    traj_paths = sorted((model_dir / structure).glob("seed-*.traj"))
    if not traj_paths:
        raise FileNotFoundError(
            f"No CIF or trajectory found to determine atom count for {structure}"
        )
    return len(read(str(traj_paths[0]), index=0))


def build_dataframe(results_dir=DEFAULT_RESULTS,
                    initial_structures_dir=DEFAULT_INITIAL_STRUCTURES,
                    models=None):
    results_dir = Path(results_dir)
    model_dirs = {
        path.name: path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    }
    if models is not None:
        missing = set(models) - set(model_dirs)
        if missing:
            raise FileNotFoundError(f"Missing model summaries: {sorted(missing)}")
        model_dirs = {model: model_dirs[model] for model in models}

    model_names = _casefold_sorted(model_dirs)
    if not model_names:
        raise FileNotFoundError(f"No summary.json files found in {results_dir}")

    summaries = {}
    structure_names = set()
    for model in model_names:
        summary_path = model_dirs[model] / "summary.json"
        with summary_path.open() as handle:
            summary = json.load(handle)
        summaries[model] = summary
        structure_names.update(item["structure"] for item in summary["structures"])

    structures = _casefold_sorted(structure_names)
    atoms_by_structure = {}
    for structure in structures:
        # Any model directory has the same structure trajectories.
        atoms_by_structure[structure] = _load_num_atoms(
            structure, model_dirs[model_names[0]], initial_structures_dir
        )

    rows = [("num atoms", "")]
    rows.extend((metric, model)
                for metric in (
                    "co2_emission (g)",
                    "co2_emission (g/1 ns)",
                    "co2_emission (g/1 ns/1 atom)",
                )
                for model in model_names)
    data = {row: {structure: float("nan") for structure in structures}
            for row in rows}

    for structure in structures:
        data[("num atoms", "")][structure] = atoms_by_structure[structure]

    for model in model_names:
        summary = summaries[model]
        md = summary.get("md", {})
        production_ps = float(md.get("production_ps", 50.0))
        production_ns = production_ps / 1000.0
        result_by_structure = {
            item["structure"]: item for item in summary["structures"]
        }
        for structure in structures:
            item = result_by_structure.get(structure)
            if item is None:
                continue
            carbon = item.get("carbon_mean_over_seeds", {})
            emissions_g = carbon.get("emissions_g_co2")
            if emissions_g is None:
                continue
            emissions_g = float(emissions_g)
            emissions_per_ns = emissions_g / production_ns
            emissions_per_ns_atom = (
                emissions_per_ns * 1 / atoms_by_structure[structure]
            )
            data[("co2_emission (g)", model)][structure] = emissions_g
            data[("co2_emission (g/1 ns)", model)][structure] = emissions_per_ns
            data[("co2_emission (g/1 ns/1 atom)", model)][structure] = \
                emissions_per_ns_atom

    frame = pd.DataFrame.from_dict(data, orient="index", columns=structures)
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["metric", "model"])
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--initial-structures-dir", default=str(DEFAULT_INITIAL_STRUCTURES))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output-pickle", default=str(DEFAULT_PICKLE))
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    frame = build_dataframe(
        results_dir=args.results_dir,
        initial_structures_dir=args.initial_structures_dir,
        models=args.models,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv)
    frame.to_pickle(args.output_pickle)
    print(frame.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nCSV: {output_csv}")
    print(f"Pickle: {args.output_pickle}")


if __name__ == "__main__":
    main()
