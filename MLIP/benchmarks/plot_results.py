#!/usr/bin/env python3
"""
Plot accuracy vs cost (carbon, energy, or speed) for benchmark models.

Loads benchmark results from <Task>/results/outputs/ and creates scatter
plots showing top-k accuracy (y-axis) vs cost metric (x-axis).
Figures are saved to <Task>/results/figures/.

Usage:
    python MLIP/benchmarks/plot_results.py --task MLIP                  # all 3 plots
    python MLIP/benchmarks/plot_results.py --task MLIP --combined       # combined view
    python MLIP/benchmarks/plot_results.py --task MLIP --xaxis duration_seconds
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

BENCHMARKS_DIR = Path(__file__).resolve().parent
TASK_DIR = BENCHMARKS_DIR.parent          # e.g. Carbon4Science/MLIP/
REPO_ROOT = TASK_DIR.parent               # e.g. Carbon4Science/

# Model display settings: color, marker, params, publication venue
MODEL_STYLES = {
    "neuralsym":   {"color": "#2196F3", "marker": "o", "params": "32M",  "year": 2017, "venue": "Chem. Eur. J."},
    "LocalRetro":  {"color": "#4CAF50", "marker": "s", "params": "8.6M", "year": 2021, "venue": "JACS Au"},
    "RetroBridge": {"color": "#FF9800", "marker": "D", "params": "4.6M", "year": 2024, "venue": "ICLR"},
    "Chemformer":  {"color": "#9C27B0", "marker": "^", "params": "44.7M", "year": 2022, "venue": "ML:ST"},
    "RSGPT":       {"color": "#F44336", "marker": "P", "params": "~1.6B","year": 2025, "venue": "Nat. Commun."},
    "RSMILES_1x":  {"color": "#00BCD4", "marker": "v", "params": "~30M", "year": 2022, "venue": "Chem. Sci."},
    "RSMILES_20x": {"color": "#009688", "marker": "^", "params": "~30M", "year": 2022, "venue": "Chem. Sci."},
    # MLIP models
    "eSEN":     {"color": "#E91E63", "marker": "o", "params": "30.1M",  "year": 2025, "venue": "ICML"},
    "NequIP":   {"color": "#3F51B5", "marker": "s", "params": "9.6M",  "year": 2025, "venue": "arXiv"},
    "Nequix":   {"color": "#009688", "marker": "D", "params": "708K", "year": 2025, "venue": "arXiv"},
    "DPA3":     {"color": "#FF5722", "marker": "^", "params": "4.8M", "year": 2025, "venue": "arXiv"},
    "SevenNet": {"color": "#795548", "marker": "P", "params": "1.2M",  "year": 2024, "venue": "JCTC"},
    "MACE":     {"color": "#607D8B", "marker": "h", "params": "4.7M", "year": 2023, "venue": "NeurIPS"},
    "ORB":      {"color": "#CDDC39", "marker": "v", "params": "25.2M", "year": 2024, "venue": "arXiv"},
    "CHGNet":   {"color": "#FF9800", "marker": "*", "params": "413K", "year": 2023, "venue": "Nat. Mach. Intell."},
    # New MLIP models
    "PET":        {"color": "#8E24AA", "marker": "H", "params": "730M", "year": 2025, "venue": "Nat. Commun."},
    "eSEN_OAM":   {"color": "#C2185B", "marker": "d", "params": "30.2M",  "year": 2025, "venue": "ICML"},
    "EquFlash":   {"color": "#00695C", "marker": "X", "params": "28.7M",  "year": 2025, "venue": "ICML"},
    "NequIP_OAM": {"color": "#1565C0", "marker": ">", "params": "9.6M", "year": 2025, "venue": "arXiv"},
    "Allegro":    {"color": "#4527A0", "marker": "p", "params": "9.7M", "year": 2025, "venue": "arXiv"},
}


def _sorted_by_year(model_names):
    """Sort model names by publication year (earliest first)."""
    return sorted(model_names, key=lambda n: MODEL_STYLES.get(n, {}).get("year", 9999))


def _model_label(name):
    """Build display label like 'neuralsym (32M, 2018)'."""
    s = MODEL_STYLES.get(name, {})
    parts = [name]
    extra = []
    if "params" in s:
        extra.append(s["params"])
    if "year" in s:
        extra.append(str(s["year"]))
    if extra:
        parts.append(f"({', '.join(extra)})")
    return " ".join(parts)


def _model_annotation(name):
    """Build annotation like 'neuralsym\n(2021 JACS Au)'."""
    s = MODEL_STYLES.get(name, {})
    year = s.get("year", "")
    venue = s.get("venue", "")
    if year and venue:
        return f"{name}\n({year} {venue})"
    elif year:
        return f"{name}\n({year})"
    return name


# X-axis configuration for each cost metric
XAXIS_CONFIG = {
    "emissions_g_co2": {
        "label": "CO$_2$ emissions (g)",
        "title_word": "Carbon Cost",
        "file_suffix": "carbon",
    },
    "energy_wh": {
        "label": "Energy (Wh)",
        "title_word": "Energy Cost",
        "file_suffix": "energy",
    },
    "duration_seconds": {
        "label": "Time (s)",
        "title_word": "Inference Speed",
        "file_suffix": "speed",
    },
}


def _format_steps(n):
    """Format step count for axis labels (e.g., 100 -> '100', 1000000 -> '10^6')."""
    import math
    if n >= 1000 and math.log10(n) == int(math.log10(n)):
        exp = int(math.log10(n))
        return f"$10^{exp}$"
    return str(n)


# Per-metric normalization for MLIP panel plots
# CPS is a static property -> short MD suffices -> per 100 steps
# RDF/MSD require long production MD -> per 10^6 steps
MLIP_METRIC_NORMS = {
    "CPS": 100,
    "rdf_score": 1_000_000,
    "msd_score": 1_000_000,
}


def _xaxis_label(xaxis_key, norm_n=None, is_mlip=False):
    """Build x-axis label. If norm_n is set, append 'per N samples'."""
    base = XAXIS_CONFIG[xaxis_key]["label"]
    if norm_n == 1:
        return f"{base} per molecule"
    elif norm_n:
        if is_mlip:
            return f"{base} per {_format_steps(norm_n)} MD steps"
        return f"{base} per {norm_n} samples"
    return base


def load_results(task="Retro", samples=None):
    """Load benchmark result JSON files.

    Args:
        task: Task name to filter by.
        samples: If set, only load results with exactly this many samples.
                 If None, keeps the largest-N run per model.

    Searches <Task>/results/outputs/ under the repo root. Also checks the
    current task directory's results/outputs/ as fallback.
    """
    results = {}
    search_dirs = [
        REPO_ROOT / task / "results" / "outputs",   # <Task>/results/outputs/
        TASK_DIR / "results" / "outputs",            # current task's results/outputs/
    ]
    seen_files = set()

    for results_dir in search_dirs:
        if not results_dir.is_dir():
            continue
        for fpath in sorted(results_dir.glob("*.json")):
            if fpath.resolve() in seen_files:
                continue
            seen_files.add(fpath.resolve())
            if fpath.name == ".gitkeep":
                continue
            if "predictions" in fpath.name:
                continue
            try:
                data = json.loads(fpath.read_text())
            except (json.JSONDecodeError, KeyError):
                continue

            # Filter by task if the result has a task field
            if data.get("task") and data["task"] != task:
                continue

            model = data.get("model")
            n = data.get("num_samples", 0)
            if model is None:
                continue

            # Filter by exact sample count if specified
            if samples is not None and n != samples:
                continue

            # Keep the run with the most samples per model
            if model not in results or n > results[model]["num_samples"]:
                results[model] = data

    return results


def plot_accuracy_vs_cost(results, task="Retro", metrics=None,
                          xaxis_key="emissions_g_co2",
                          output=None, norm_n=None):
    """
    Create scatter plot(s) of accuracy vs a cost metric (CO2, energy, or time).

    Args:
        results: dict of model_name -> result dict
        task: task name (for title)
        metrics: list of metrics to plot
        xaxis_key: which cost metric for x-axis
        output: output file path (shows plot if None)
        norm_n: normalize cost to per-N-samples. None = raw values.
    """
    if metrics is None:
        metrics = _detect_metrics(results) or ["top_1", "top_5", "top_10", "top_50"]

    is_mlip = (task == "MLIP")
    is_score = is_mlip  # 0-1 scores, not percentages
    scale = 1 if is_score else 100

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        # Per-metric norm for MLIP; otherwise use the global norm_n
        metric_norm = MLIP_METRIC_NORMS.get(metric, norm_n) if is_mlip else norm_n
        x_label = _xaxis_label(xaxis_key, metric_norm, is_mlip=is_mlip)

        for model_name, data in sorted(results.items(), key=lambda x: MODEL_STYLES.get(x[0], {}).get("year", 9999)):
            acc = data.get("accuracy", {}).get(metric)
            cost = data.get("carbon", {}).get(xaxis_key, 0)
            n_samples = data.get("num_samples", 1)

            if acc is None or cost == 0:
                continue

            if metric_norm:
                cost = cost / n_samples * metric_norm

            style = MODEL_STYLES.get(model_name, {
                "color": "gray", "marker": "x"
            })

            ax.scatter(cost, acc * scale, s=120, zorder=5,
                       color=style["color"], marker=style["marker"],
                       edgecolors="white", linewidths=0.5)

            ax.annotate(
                _model_annotation(model_name),
                (cost, acc * scale),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
                color=style["color"],
                fontweight="bold",
            )

        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        if is_score:
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
        ax.set_title(_metric_display(metric))
        ax.grid(True, alpha=0.3, which="both")

    # No model legend needed — annotations identify each model

    fig.suptitle(f"{task}: Accuracy vs {XAXIS_CONFIG[xaxis_key]['title_word']}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


def plot_combined(results, task="Retro", xaxis_key="emissions_g_co2",
                  output=None, norm_n=None):
    """
    Single plot with all accuracy metrics connected by lines per model.
    X-axis: cost metric, Y-axis: accuracy. Each model shows all metrics as
    connected points (all share the same x since cost is per-run).
    """
    metrics = _detect_metrics(results) or ["top_1", "top_5", "top_10", "top_50"]
    is_score = (task == "MLIP")  # 0-1 scores, not percentages
    scale = 1 if is_score else 100

    marker_cycle = ["o", "s", "D", "*", "v", "^", "P", "h"]
    metric_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(metrics)}

    x_label = _xaxis_label(xaxis_key, norm_n, is_mlip=(task == "MLIP"))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for model_name, data in sorted(results.items(), key=lambda x: MODEL_STYLES.get(x[0], {}).get("year", 9999)):
        cost = data.get("carbon", {}).get(xaxis_key, 0)
        n_samples = data.get("num_samples", 1)
        if cost == 0:
            continue
        if norm_n:
            cost = cost / n_samples * norm_n

        style = MODEL_STYLES.get(model_name, {"color": "gray"})
        accs = []
        for k in metrics:
            acc = data.get("accuracy", {}).get(k)
            if acc is not None:
                accs.append(acc * scale)
                ax.scatter(cost, acc * scale, s=100, zorder=5,
                           color=style["color"], marker=metric_markers[k],
                           edgecolors="white", linewidths=0.5)

        # Vertical line connecting min to max
        if len(accs) >= 2:
            ax.vlines(cost, min(accs), max(accs), colors=style["color"],
                      linewidths=1.5, alpha=0.4, zorder=3)

        # Label near the top point with year
        if accs:
            ax.annotate(
                _model_annotation(model_name),
                (cost, max(accs)),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=style["color"],
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    if is_score:
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
    ax.set_title(f"{task}: Accuracy vs {XAXIS_CONFIG[xaxis_key]['title_word']}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # Metric legend only (model info is in annotations)
    metric_handles = [
        plt.Line2D([0], [0], marker=metric_markers[m], color="w",
                   markerfacecolor="gray",
                   markersize=8 if metric_markers[m] != "*" else 10,
                   label=_metric_display(m))
        for m in metrics
    ]

    ax.legend(handles=metric_handles, loc="lower left",
              framealpha=0.9, edgecolor="lightgray")

    fig.tight_layout()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


def _add_annotations(ax, texts_data):
    """Add model annotations with adjustText for clean label placement."""
    try:
        from adjustText import adjust_text
        texts = []
        for txt, x, y, color in texts_data:
            t = ax.text(x, y, txt, fontsize=9, color=color, fontweight="bold")
            texts.append(t)
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                    lw=0.5, alpha=0.5))
    except ImportError:
        for txt, x, y, color in texts_data:
            ax.annotate(txt, (x, y), textcoords="offset points",
                        xytext=(10, 6), fontsize=9, color=color,
                        fontweight="bold")


def _detect_metrics(results):
    """Detect metrics list from results (reads 'metrics' field from any result)."""
    for data in results.values():
        m = data.get("metrics")
        if m:
            return m
    return None


def _metric_display(metric):
    """Human-readable display name for a metric."""
    mapping = {
        "top_1": "Top-1", "top_5": "Top-5",
        "top_10": "Top-10", "top_50": "Top-50",
        "CPS": "CPS", "rdf_score": "RDF Score", "msd_score": "MSD Score",
    }
    return mapping.get(metric, metric)


def main():
    all_xaxis = list(XAXIS_CONFIG.keys())

    parser = argparse.ArgumentParser(description="Plot accuracy vs cost metrics")
    parser.add_argument("--task", type=str, default="Retro",
                        help="Task name (default: Retro)")
    parser.add_argument("--metric", nargs="+", default=None,
                        help="Metrics to plot (auto-detected from results if not given)")
    parser.add_argument("--xaxis", nargs="+", default=None,
                        choices=all_xaxis,
                        help="X-axis metric(s). Default: generate all three "
                             "(emissions_g_co2, energy_wh, duration_seconds)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (only when single --xaxis is given)")
    parser.add_argument("--combined", action="store_true",
                        help="Single combined plot instead of per-metric panels")
    parser.add_argument("--samples", type=int, default=None,
                        help="Only use results with exactly N samples (e.g., --samples 500)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Don't normalize to per-1000 samples")
    parser.add_argument("--norm", type=int, default=None,
                        help="Normalize cost to per-N samples (e.g., --norm 1 for per molecule)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Only plot these models (e.g., --models CHGNet MACE eSEN)")

    args = parser.parse_args()

    results = load_results(task=args.task, samples=args.samples)
    if args.models:
        results = {k: v for k, v in results.items() if k in args.models}
    if not results:
        print(f"No result files found for task '{args.task}'")
        return

    # Auto-detect metrics from results
    detected_metrics = _detect_metrics(results)
    plot_metrics = args.metric or detected_metrics or ["top_1", "top_5", "top_10", "top_50"]

    print(f"Task: {args.task}")
    print(f"Loaded results for: {', '.join(_sorted_by_year(results.keys()))}")
    for name, data in sorted(results.items(), key=lambda x: MODEL_STYLES.get(x[0], {}).get("year", 9999)):
        n = data["num_samples"]
        co2 = data.get("carbon", {}).get("emissions_g_co2", 0)
        dur = data.get("carbon", {}).get("duration_seconds", 0)
        if args.task == "MLIP":
            sps = data.get("speed", {}).get("steps_per_second", 0)
            nsd = data.get("speed", {}).get("ns_per_day", 0)
            acc_str = "  ".join(
                f"{k}={v:.3f}"
                for k, v in data.get("accuracy", {}).items()
            )
            print(f"  {name:15s}  {sps:6.1f} steps/s  {nsd:6.2f} ns/day  "
                  f"{acc_str}  CO2={co2:.2f}g  time={dur:.1f}s")
        else:
            top1 = data.get("accuracy", {}).get("top_1", 0)
            print(f"  {name:15s}  n={n:5d}  top-1={top1*100:5.1f}%  "
                  f"CO2={co2:.2f}g  time={dur:.1f}s")

    # Normalization: --norm N overrides all. Otherwise:
    # --no-normalize = raw values, --samples = per that count,
    # MLIP = None (costs already normalized per 1000 steps), default = per 1000.
    if args.norm is not None:
        norm_n = args.norm
    elif args.no_normalize:
        norm_n = None
    elif args.task == "MLIP":
        norm_n = 1000  # MLIP costs are per-step; default display is per 1000 MD steps
    elif args.samples:
        norm_n = args.samples
    else:
        norm_n = 1000

    xaxis_list = args.xaxis if args.xaxis else all_xaxis

    for xkey in xaxis_list:
        xcfg = XAXIS_CONFIG[xkey]
        # Determine output path
        if args.output and len(xaxis_list) == 1:
            output = args.output
        else:
            fig_dir = REPO_ROOT / args.task / "results" / "figures"
            suffix = "combined" if args.combined else "panels"
            n_tag = f"_{args.samples}" if args.samples else ""
            output = str(fig_dir / f"accuracy_vs_{xcfg['file_suffix']}_{suffix}{n_tag}.png")

        if args.combined:
            plot_combined(results, task=args.task, xaxis_key=xkey,
                          output=output, norm_n=norm_n)
        else:
            plot_accuracy_vs_cost(results, task=args.task, metrics=plot_metrics,
                                  xaxis_key=xkey, output=output,
                                  norm_n=norm_n)


if __name__ == "__main__":
    main()
