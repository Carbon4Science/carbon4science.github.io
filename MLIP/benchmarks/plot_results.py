#!/usr/bin/env python3
"""
Plot MLIP accuracy vs cost (carbon, energy, or speed).

Loads results from production/results/{variant}/{model}/ and creates scatter
plots: accuracy (y) vs cost (x) for each model.

Usage:
    python MLIP/benchmarks/plot_results.py                          # pretrained, all 3 x-axes
    python MLIP/benchmarks/plot_results.py --variant finetuned      # finetuned
    python MLIP/benchmarks/plot_results.py --combined               # single combined plot
    python MLIP/benchmarks/plot_results.py --errorbars              # with std error bars
    python MLIP/benchmarks/plot_results.py --metric msd_score --xaxis emissions_g_co2
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
TASK_DIR = BENCHMARKS_DIR.parent          # Carbon4Science/MLIP/
REPO_ROOT = TASK_DIR.parent               # Carbon4Science/

MODEL_STYLES = {
    "eSEN":       {"color": "#E91E63", "marker": "o", "params": "30.1M", "year": "2025. 2", "venue": "ICML"},
    "eSEN_OAM":   {"color": "#C2185B", "marker": "d", "params": "30.2M", "year": 2025, "venue": "ICML"},
    "NequIP":     {"color": "#3F51B5", "marker": "s", "params": "9.6M",  "year": "2025. 4", "venue": "arXiv"},
    "NequIP_OAM": {"color": "#1565C0", "marker": ">", "params": "9.6M",  "year": 2025, "venue": "arXiv"},
    "Allegro":    {"color": "#4527A0", "marker": "p", "params": "9.7M",  "year": 2025, "venue": "arXiv"},
    "Nequix":     {"color": "#009688", "marker": "D", "params": "708K",  "year": "2025. 8", "venue": "arXiv"},
    "DPA3":       {"color": "#FF5722", "marker": "^", "params": "4.8M",  "year": "2025. 6", "venue": "arXiv"},
    "SevenNet":   {"color": "#795548", "marker": "P", "params": "1.2M",  "year": "2024. 2", "venue": "JCTC"},
    "MACE":       {"color": "#607D8B", "marker": "h", "params": "4.7M",  "year": "2023. 12", "venue": "NeurIPS"},
    "MACE_pruned": {"color": "#90A4AE", "marker": "h", "params": "652K", "year": 2023, "venue": "NeurIPS"},
    "ORB":        {"color": "#CDDC39", "marker": "v", "params": "25.2M", "year": "2024. 10", "venue": "arXiv"},
    "CHGNet":     {"color": "#FF9800", "marker": "*", "params": "413K",  "year": "2023. 2", "venue": "Nat. Mach. Intell."},
    "PET":        {"color": "#8E24AA", "marker": "H", "params": "730M",  "year": 2025, "venue": "Nat. Commun."},
    "EquFlash":   {"color": "#00695C", "marker": "X", "params": "28.7M", "year": 2025, "venue": "ICML"},
}

XAXIS_CONFIG = {
    "emissions_g_co2": {"label": "CO$_2$ emissions (g)", "title_word": "Carbon Cost", "file_suffix": "carbon"},
    "energy_wh":       {"label": "Energy (Wh)",          "title_word": "Energy Cost", "file_suffix": "energy"},
    "duration_seconds": {"label": "Time (s)",            "title_word": "Speed",       "file_suffix": "speed"},
}

METRIC_DISPLAY = {
    "rdf_score.average": "RDF Score",
    "msd_score": "MSD Score",
    "CPS": "CPS",
}

# Per-metric cost normalization: steps to normalize to
# CPS is a fast metric -> per 1000 steps
# RDF/MSD require full production -> per 10^6 steps
METRIC_NORM_STEPS = {
    "CPS": 1000,
    "rdf_score.average": None,  # raw (full production run)
    "msd_score": 1_000_000,
}

DEFAULT_METRICS = ["CPS", "rdf_score.average", "msd_score"]


def _sorted_by_year(model_names):
    #try:
    return sorted(model_names, key=lambda n: (int(MODEL_STYLES.get(n, {}).get("year", 9999).split(".")[0]), int(MODEL_STYLES.get(n, {}).get("year", 9999).split(".")[1])))

    #return sorted(model_names, key=lambda n: MODEL_STYLES.get(n, {}).get("year", 9999))


def _model_annotation(name):
    s = MODEL_STYLES.get(name, {})
    year = s.get("year", "")
    venue = s.get("venue", "")
    if year and venue:
        #return f"{name}\n({year} {venue})"
        return f"{name} ({year})"
    return name


def load_results(variant="pretrained"):
    """Load results from production/results/{variant}/{model}/*_results.json."""
    results = {}
    prod_dir = TASK_DIR / "production" / "results" / variant
    if not prod_dir.is_dir():
        return results
    for model_dir in sorted(prod_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for fpath in model_dir.glob("*_results.json"):
            try:
                data = json.loads(fpath.read_text())
            except (json.JSONDecodeError, KeyError):
                continue
            model = data.get("model")
            if model:
                results[model] = data
    return results


def _get_metric(data, metric):
    """Get metric mean value from accuracy."""
    acc = data.get("accuracy", {})
    if metric in acc:
        return acc[metric]
    return acc.get("mean", {}).get(metric)


def _normalize_cost(cost, data, metric):
    """Normalize cost value based on per-metric normalization steps."""
    norm_steps = METRIC_NORM_STEPS.get(metric)
    if norm_steps is None:
        return cost
    total_steps = data.get("speed", {}).get("steps", 0)
    if total_steps <= 0:
        return cost
    return cost / total_steps * norm_steps


def _xaxis_label(xaxis_key, metric):
    """Build x-axis label with per-metric normalization."""
    base = XAXIS_CONFIG[xaxis_key]["label"]
    norm_steps = METRIC_NORM_STEPS.get(metric)
    if norm_steps is None:
        return base
    if norm_steps == 1000:
        return f"{base} per 1000 steps"
    elif norm_steps == 1_000_000:
        return f"{base} per $10^6$ steps"
    return f"{base} per {norm_steps} steps"


def _get_metric_std(data, metric):
    """Get metric std value from accuracy."""
    return data.get("accuracy", {}).get("std", {}).get(metric)


def plot_panels(results, metrics=None, xaxis_key="emissions_g_co2",
                output=None, errorbars=False, variant="pretrained"):
    """Per-metric panel plot: one subplot per accuracy metric."""
    metrics = metrics or DEFAULT_METRICS
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        for model_name in _sorted_by_year(results.keys()):
            data = results[model_name]
            acc = _get_metric(data, metric)
            raw_cost = data.get("carbon", {}).get(xaxis_key, 0)
            if acc is None or raw_cost == 0:
                continue

            cost = _normalize_cost(raw_cost, data, metric)
            style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "x"})

            if errorbars:
                std = _get_metric_std(data, metric)
                if std is not None:
                    ax.errorbar(cost, acc, yerr=std, fmt="none",
                                ecolor=style["color"], capsize=3, alpha=0.6, zorder=4)

            ax.scatter(cost, acc, s=120, zorder=5,
                       color=style["color"], marker=style["marker"],
                       edgecolors="white", linewidths=0.5,
                       label=_model_annotation(model_name)
                       )
            #ax.annotate(_model_annotation(model_name), (cost, acc),
            #            textcoords="offset points", xytext=(8, 6),
            #            fontsize=8, color=style["color"], fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel(_xaxis_label(xaxis_key, metric))
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_DISPLAY.get(metric, metric))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

    fig.suptitle(f"MLIP ({variant}): Accuracy vs {XAXIS_CONFIG[xaxis_key]['title_word']}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=200)
        print(f"Saved: {output}")
    else:
        plt.show()
    return fig


def plot_combined(results, metrics, xaxis_key="emissions_g_co2",
                  output=None, errorbars=False, variant="pretrained"):
    """Single combined plot with all metrics per model."""
    marker_cycle = ["o", "s", "D", "*", "v", "^", "P", "h"]
    metric_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(metrics)}

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for model_name in _sorted_by_year(results.keys()):
        data = results[model_name]
        cost = data.get("carbon", {}).get(xaxis_key, 0)
        if cost == 0:
            continue

        style = MODEL_STYLES.get(model_name, {"color": "gray"})
        accs = []
        for k in metrics:
            acc = _get_metric(data, k)
            if acc is None:
                continue
            accs.append(acc)

            if errorbars:
                std = _get_metric_std(data, k)
                if std is not None:
                    ax.errorbar(cost, acc, yerr=std, fmt="none",
                                ecolor=style["color"], capsize=3, alpha=0.6, zorder=4)

            ax.scatter(cost, acc, s=100, zorder=5,
                       color=style["color"], marker=metric_markers[k],
                       edgecolors="white", linewidths=0.5)

        if len(accs) >= 2:
            ax.vlines(cost, min(accs), max(accs), colors=style["color"],
                      linewidths=1.5, alpha=0.4, zorder=3)

        if accs:
            ax.annotate(_model_annotation(model_name), (cost, max(accs)),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=8, color=style["color"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel(XAXIS_CONFIG[xaxis_key]["label"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"MLIP ({variant}): Accuracy vs {XAXIS_CONFIG[xaxis_key]['title_word']}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    metric_handles = [
        plt.Line2D([0], [0], marker=metric_markers[m], color="w",
                   markerfacecolor="gray", markersize=8,
                   label=METRIC_DISPLAY.get(m, m))
        for m in metrics
    ]
    ax.legend(handles=metric_handles, loc="lower left",
              framealpha=0.9, edgecolor="lightgray")

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=200)
        print(f"Saved: {output}")
    else:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot MLIP accuracy vs cost")
    parser.add_argument("--variant", default="pretrained", choices=["pretrained", "finetuned"])
    parser.add_argument("--metric", nargs="+", default=None,
                        help=f"Metrics to plot (default: {DEFAULT_METRICS})")
    parser.add_argument("--xaxis", nargs="+", default=None,
                        choices=list(XAXIS_CONFIG.keys()),
                        help="X-axis metric(s). Default: all three.")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--combined", action="store_true",
                        help="Single combined plot instead of per-metric panels")
    parser.add_argument("--errorbars", action="store_true",
                        help="Show std error bars across seeds")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Only plot these models")

    args = parser.parse_args()

    results = load_results(variant=args.variant)
    if args.models:
        results = {k: v for k, v in results.items() if k in args.models}
    if not results:
        print(f"No results found for variant '{args.variant}'")
        return

    print(f"Variant: {args.variant}")
    print(f"Models:  {', '.join(_sorted_by_year(results.keys()))}")
    for name in _sorted_by_year(results.keys()):
        data = results[name]
        mean = data.get("accuracy", {}).get("mean", {})
        sps = data.get("speed", {}).get("steps_per_second", 0)
        steps = data.get("speed", {}).get("steps", 1)
        co2 = data.get("carbon", {}).get("emissions_g_co2", 0)
        eng = data.get("carbon", {}).get("energy_wh", 0)
        rdf = mean.get("rdf_score.average", 0)
        msd = mean.get("msd_score", 0)
        cps = _get_metric(data, "CPS") or 0
        co2_ps = co2 * 1000 / steps if steps > 0 else 0
        eng_ps = eng * 1000 / steps if steps > 0 else 0
        print(f"  {name:15s}  CPS={cps:.3f}  RDF={rdf:.3f}  MSD={msd:.3f}  "
              f"{sps:6.1f} steps/s  CO2={co2_ps:.4f} g/1k  Energy={eng_ps:.4f} Wh/1k")

    xaxis_list = args.xaxis or list(XAXIS_CONFIG.keys())
    fig_dir = TASK_DIR / "results"

    for xkey in xaxis_list:
        xcfg = XAXIS_CONFIG[xkey]
        if args.output and len(xaxis_list) == 1:
            output = args.output
        else:
            suffix = "combined" if args.combined else "panels"
            output = str(fig_dir / f"accuracy_vs_{xcfg['file_suffix']}_{suffix}_{args.variant}.png")

        if args.combined:
            plot_combined(results, metrics=args.metric, xaxis_key=xkey, output=output,
                          errorbars=args.errorbars, variant=args.variant)
        else:
            plot_panels(results, metrics=args.metric, xaxis_key=xkey,
                        output=output, errorbars=args.errorbars, variant=args.variant)


if __name__ == "__main__":
    main()
