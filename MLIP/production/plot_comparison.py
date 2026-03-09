#!/usr/bin/env python3
"""Plot MLIP vs AIMD comparison for RDF and MSD.

Supports two modes:
  - Combined overlay: all models on the same axes (default)
  - Per-model panels: 2x4 grid, each subplot shows one model vs AIMD

Usage:
    # All 4 figures (combined + panels for both RDF and MSD)
    python MLIP/production/plot_comparison.py

    # Per-model panels only
    python MLIP/production/plot_comparison.py --panels

    # Combined RDF only
    python MLIP/production/plot_comparison.py --property rdf

    # MSD panels only
    python MLIP/production/plot_comparison.py --property msd --panels

    # Specific models and pair
    python MLIP/production/plot_comparison.py --property rdf --pair Li-S \
        --models CHGNet MACE eSEN
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

# Consistent model colors (matches benchmarks/plot_results.py MODEL_STYLES)
MODEL_STYLES = {
    "eSEN": {"color": "#E91E63", "ls": "-"},
    "NequIP": {"color": "#3F51B5", "ls": "-"},
    "Nequix": {"color": "#009688", "ls": "-"},
    "DPA3": {"color": "#FF5722", "ls": "-"},
    "SevenNet": {"color": "#795548", "ls": "-"},
    "MACE": {"color": "#607D8B", "ls": "-"},
    "ORB": {"color": "#CDDC39", "ls": "-"},
    "CHGNet": {"color": "#FF9800", "ls": "-"},
    "PET": {"color": "#8E24AA", "ls": "-"},
    "eSEN_OAM": {"color": "#C2185B", "ls": "-"},
    "EquFlash": {"color": "#00695C", "ls": "-"},
    "NequIP_OAM": {"color": "#1565C0", "ls": "-"},
    "Allegro": {"color": "#4527A0", "ls": "-"},
}

AIMD_STYLE = {"color": "black", "ls": "-", "lw": 2.5, "label": "AIMD"}

# Canonical model order (by publication year)
MODEL_ORDER = [
    "NequIP", 
    "CHGNet", 
    "MACE", 
    "DPA3", 
    "Nequix", 
    "SevenNet", 
    "ORB", 
    "eSEN",
    "PET",
    "eSEN_OAM",
    "EquFlash",
    "NequIP_OAM",
    "Allegro",
    ]

# Line style per RDF pair (used in per-model panels)
PAIR_LINE_STYLES = {"Li-Li": "-", "Li-Ge": "--", "Li-P": ":", "Li-S": "-."}


def load_csv(path):
    """Load CSV file with header row."""
    return np.genfromtxt(path, delimiter=",", names=True)


def _order_models(models):
    """Sort models by MODEL_ORDER, then alphabetical for unknown models."""
    ordered = [m for m in MODEL_ORDER if m in models]
    ordered.extend(sorted(m for m in models if m not in ordered))
    return ordered


def _load_model_accuracy(label, model):
    """Load accuracy metrics from model's production results JSON."""
    results_path = os.path.join(
        ROOT, "MLIP", "production", "results", model, f"{label}_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f).get("accuracy", {})
    return {}


def _detect_rdf_pairs(label, models):
    """Auto-detect available RDF pairs from existing result CSV files."""
    result_base = os.path.join(ROOT, "MLIP", "production", "results")
    pairs = []
    for model in models:
        pattern = os.path.join(result_base, model, f"{label}_rdf_*.csv")
        for fpath in sorted(glob.glob(pattern)):
            fname = os.path.basename(fpath)
            suffix = fname.replace(f"{label}_rdf_", "").replace(".csv", "")
            parts = suffix.split("-")
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                if pair not in pairs:
                    pairs.append(pair)
        if pairs:
            break
    return pairs


def plot_rdf_comparison(config, models, structure_index=0, pair=None, output=None):
    """Overlay RDF g(r) for multiple models + AIMD ground truth."""
    struct_cfg = config["structures"][structure_index]
    label = struct_cfg["label"]

    if pair:
        pairs = [tuple(pair.split("-"))]
    else:
        pairs = _detect_rdf_pairs(label, models)
        if not pairs:
            print("No RDF result files found. Run analysis first.")
            return None

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)
    axes = axes[0]

    result_base = os.path.join(ROOT, "MLIP", "production", "results")
    gt_base = os.path.join(ROOT, "MLIP", "production", "ground_truth")

    for ax, (sp_i, sp_j) in zip(axes, pairs):
        fname = f"{label}_rdf_{sp_i}-{sp_j}.csv"

        # AIMD ground truth (black line)
        gt_path = os.path.join(gt_base, fname)
        if os.path.exists(gt_path):
            gt = load_csv(gt_path)
            ax.plot(gt["r_angstrom"], gt["g_r"], **AIMD_STYLE)

        # MLIP models
        for model in models:
            csv_path = os.path.join(result_base, model, fname)
            if not os.path.exists(csv_path):
                print(f"  Warning: {csv_path} not found, skipping")
                continue
            data = load_csv(csv_path)
            style = MODEL_STYLES.get(model, {"color": "gray", "ls": "--"})
            ax.plot(
                data["r_angstrom"],
                data["g_r"],
                color=style["color"],
                ls=style["ls"],
                lw=1.5,
                label=model,
            )

        ax.set_xlabel(r"r ($\mathrm{\AA}$)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"{sp_i}-{sp_j}")
        ax.legend(fontsize=9)
        ax.set_xlim(0, struct_cfg.get("rdf", {}).get("rmax", 8.0))
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"RDF Comparison: {label} ({struct_cfg['temperature_K']} K)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    if output is None:
        fig_dir = os.path.join(ROOT, "MLIP", "production", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        pair_tag = f"_{pair}" if pair else ""
        output = os.path.join(fig_dir, f"{label}_rdf{pair_tag}.png")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved: {output}")
    return fig


def plot_msd_comparison(config, models, structure_index=0, output=None):
    """Overlay MSD for multiple models + AIMD ground truth."""
    struct_cfg = config["structures"][structure_index]
    label = struct_cfg["label"]
    diff_species = struct_cfg["diffusing_species"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_total, ax_xyz = axes

    result_base = os.path.join(ROOT, "MLIP", "production", "results")
    gt_base = os.path.join(ROOT, "MLIP", "production", "ground_truth")
    fname = f"{label}_msd.csv"

    # AIMD ground truth
    gt_path = os.path.join(gt_base, fname)
    if os.path.exists(gt_path):
        gt = load_csv(gt_path)
        gt_t_ps = gt["t_fs"] / 1000.0
        ax_total.plot(gt_t_ps, gt["msd_angstrom2"], **AIMD_STYLE)
        for d, (comp, ls) in enumerate(
            zip(["msd_x", "msd_y", "msd_z"], ["-", "--", ":"])
        ):
            ax_xyz.plot(
                gt_t_ps,
                gt[comp],
                color="black",
                ls=ls,
                lw=2,
                label=f"AIMD {comp[-1]}",
            )

    # MLIP models
    for model in models:
        csv_path = os.path.join(result_base, model, fname)
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping")
            continue
        data = load_csv(csv_path)
        data_t_ps = data["t_fs"] / 1000.0
        style = MODEL_STYLES.get(model, {"color": "gray", "ls": "--"})

        ax_total.plot(
            data_t_ps,
            data["msd_angstrom2"],
            color=style["color"],
            ls=style["ls"],
            lw=1.5,
            label=model,
        )

        for d, (comp, ls) in enumerate(
            zip(["msd_x", "msd_y", "msd_z"], ["-", "--", ":"])
        ):
            ax_xyz.plot(
                data_t_ps,
                data[comp],
                color=style["color"],
                ls=ls,
                lw=1.2,
                label=f"{model} {comp[-1]}" if d == 0 else None,
            )

    ax_total.set_xlabel("t (ps)")
    ax_total.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
    ax_total.set_title(f"Total MSD ({diff_species})")
    ax_total.legend(fontsize=9)
    ax_total.grid(True, alpha=0.3)

    ax_xyz.set_xlabel("t (ps)")
    ax_xyz.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
    ax_xyz.set_title(f"Per-component MSD ({diff_species})")
    ax_xyz.legend(fontsize=8, ncol=3)
    ax_xyz.grid(True, alpha=0.3)

    fig.suptitle(
        f"MSD Comparison: {label} ({struct_cfg['temperature_K']} K)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    if output is None:
        fig_dir = os.path.join(ROOT, "MLIP", "production", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        output = os.path.join(fig_dir, f"{label}_msd.png")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved: {output}")
    return fig


def plot_rdf_panels(config, models, structure_index=0, output=None):
    """Per-model RDF panels: 2x4 grid, each subplot shows one model vs AIMD.

    All 4 RDF pairs are plotted per panel using pair-specific line styles.
    AIMD is black (thick), model is colored (thin).
    """
    struct_cfg = config["structures"][structure_index]
    label = struct_cfg["label"]
    rmax = struct_cfg.get("rdf", {}).get("rmax", 8.0)
    temp = struct_cfg["temperature_K"]

    pairs = _detect_rdf_pairs(label, models)
    if not pairs:
        print("No RDF result files found. Run analysis first.")
        return None

    ordered = _order_models(models)
    ncols = 4
    nrows = (len(ordered) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
    )

    result_base = os.path.join(ROOT, "MLIP", "production", "results")
    gt_base = os.path.join(ROOT, "MLIP", "production", "ground_truth")

    for idx, model in enumerate(ordered):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        style = MODEL_STYLES.get(model, {"color": "gray", "ls": "-"})

        # Load MAE for annotation
        acc = _load_model_accuracy(label, model)
        rdf_mae_avg = acc.get("rdf_mae", {}).get("average")
        rdf_score_avg = acc.get("rdf_score", {}).get("average")

        for sp_i, sp_j in pairs:
            pair_key = f"{sp_i}-{sp_j}"
            ls = PAIR_LINE_STYLES.get(pair_key, "-")
            fname = f"{label}_rdf_{pair_key}.csv"

            # AIMD ground truth
            gt_path = os.path.join(gt_base, fname)
            if os.path.exists(gt_path):
                gt = load_csv(gt_path)
                ax.plot(gt["r_angstrom"], gt["g_r"], color="black", ls=ls, lw=2.0)

            # Model
            csv_path = os.path.join(result_base, model, fname)
            if os.path.exists(csv_path):
                data = load_csv(csv_path)
                ax.plot(
                    data["r_angstrom"],
                    data["g_r"],
                    color=style["color"],
                    ls=ls,
                    lw=1.5,
                )

        title = model
        if rdf_score_avg is not None:
            title += f" (Score={rdf_score_avg:.3f})"
        ax.set_title(title, fontsize=12, color=style["color"], fontweight="bold")
        ax.set_xlim(0, rmax)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("g(r)")
        if row == nrows - 1:
            ax.set_xlabel(r"r ($\mathrm{\AA}$)")

    # Hide unused subplots
    for idx in range(len(ordered), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Figure legend: AIMD label + pair line styles
    handles = [
        plt.Line2D([0], [0], color="black", lw=2, ls="-", label="AIMD"),
    ]
    for pair_key, ls in PAIR_LINE_STYLES.items():
        handles.append(
            plt.Line2D([0], [0], color="gray", lw=1.5, ls=ls, label=pair_key)
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"RDF Per-Model Comparison: {label} ({temp} K)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output is None:
        fig_dir = os.path.join(ROOT, "MLIP", "production", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        output = os.path.join(fig_dir, f"{label}_rdf_panels.png")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved: {output}")
    return fig


def plot_msd_panels(config, models, structure_index=0, output=None):
    """Per-model MSD panels: 2x4 grid, each subplot shows one model vs AIMD.

    Each panel shows total MSD: AIMD in black, model in color.
    """
    struct_cfg = config["structures"][structure_index]
    label = struct_cfg["label"]
    diff_species = struct_cfg["diffusing_species"]
    temp = struct_cfg["temperature_K"]

    ordered = _order_models(models)
    ncols = 4
    nrows = (len(ordered) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
    )

    result_base = os.path.join(ROOT, "MLIP", "production", "results")
    gt_base = os.path.join(ROOT, "MLIP", "production", "ground_truth")
    fname = f"{label}_msd.csv"

    # Load AIMD ground truth once
    gt_path = os.path.join(gt_base, fname)
    gt = load_csv(gt_path) if os.path.exists(gt_path) else None

    for idx, model in enumerate(ordered):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        style = MODEL_STYLES.get(model, {"color": "gray", "ls": "-"})

        # Load MAE for annotation
        acc = _load_model_accuracy(label, model)
        msd_mae = acc.get("msd_mae")
        msd_score = acc.get("msd_score")

        # AIMD
        if gt is not None:
            gt_t_ps = gt["t_fs"] / 1000.0
            ax.plot(gt_t_ps, gt["msd_angstrom2"], color="black", lw=2.0,
                    label="AIMD")

        # Model
        csv_path = os.path.join(result_base, model, fname)
        if os.path.exists(csv_path):
            data = load_csv(csv_path)
            data_t_ps = data["t_fs"] / 1000.0
            ax.plot(
                data_t_ps,
                data["msd_angstrom2"],
                color=style["color"],
                lw=1.5,
                label=model,
            )

        title = model
        if msd_score is not None:
            title += f" (Score={msd_score:.3f})"
        ax.set_title(title, fontsize=12, color=style["color"], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
        if row == nrows - 1:
            ax.set_xlabel("t (ps)")

    # Hide unused subplots
    for idx in range(len(ordered), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Legend on first panel
    axes[0][0].legend(fontsize=9)

    fig.suptitle(
        f"MSD Per-Model Comparison ({diff_species}): {label} ({temp} K)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    if output is None:
        fig_dir = os.path.join(ROOT, "MLIP", "production", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        output = os.path.join(fig_dir, f"{label}_msd_panels.png")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved: {output}")
    return fig


def main():
    default_config = os.path.join(
        ROOT, "MLIP", "production", "configs", "LGPS_300K.json"
    )

    parser = argparse.ArgumentParser(
        description="Plot MLIP vs AIMD comparison",
        epilog=(
            "Examples:\n"
            "  python MLIP/production/plot_comparison.py                  # all 4 figures\n"
            "  python MLIP/production/plot_comparison.py --panels         # per-model panels only\n"
            "  python MLIP/production/plot_comparison.py --property rdf   # combined RDF only\n"
            "  python MLIP/production/plot_comparison.py --property msd --panels\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to JSON config (default: LGPS_300K.json)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names (default: all 8 models)",
    )
    parser.add_argument(
        "--property",
        choices=["rdf", "msd"],
        default=None,
        help="Property to plot (default: both rdf and msd)",
    )
    parser.add_argument(
        "--panels",
        action="store_true",
        help="Per-model panel view instead of combined overlay",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="RDF pair (e.g., Li-S). Only for combined RDF.",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    parser.add_argument("--structure_index", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    models = args.models or _order_models(list(MODEL_STYLES.keys()))
    props = [args.property] if args.property else ["rdf", "msd"]

    # When neither --property nor --panels is specified, generate all 4 figures
    if args.panels:
        modes = ["panels"]
    elif args.property:
        modes = ["combined"]
    else:
        modes = ["combined", "panels"]

    si = args.structure_index
    for prop in props:
        for mode in modes:
            if prop == "rdf" and mode == "combined":
                plot_rdf_comparison(
                    config, models, structure_index=si,
                    pair=args.pair, output=args.output,
                )
            elif prop == "rdf" and mode == "panels":
                plot_rdf_panels(
                    config, models, structure_index=si, output=args.output,
                )
            elif prop == "msd" and mode == "combined":
                plot_msd_comparison(
                    config, models, structure_index=si, output=args.output,
                )
            elif prop == "msd" and mode == "panels":
                plot_msd_panels(
                    config, models, structure_index=si, output=args.output,
                )


if __name__ == "__main__":
    main()
