"""
Generate all paper figures for Carbon4Science.
  Figure 1: Year trends (6 tasks × 3 panels: model size, CO2, performance)
  Figure 2: Pareto frontiers (Δ Performance % vs CO2 ratio)
  Figure 3: CO2 decomposition (6 tasks × 3 panels: size vs CO2, time vs CO2, size vs time)
  Figure 4: CO2 emission reference points

Usage:
    python analysis/plot_figures.py              # generate all
    python analysis/plot_figures.py --fig 1      # generate specific figure
    python analysis/plot_figures.py --fig 1 2 4  # generate multiple
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import re
import argparse
import os
from adjustText import adjust_text
from scipy.stats import spearmanr

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = 'analysis/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Keep text in SVG as editable <text> elements (not outlined paths) so labels
# can be repositioned/retyped in Inkscape. Harmless for PNG output.
plt.rcParams['svg.fonttype'] = 'none'

# Also emit an .svg alongside every PNG when True (enabled by --svg).
SAVE_SVG = False


def save_outputs(fig, out):
    """Save `fig` to `out` (PNG) and, when SAVE_SVG, an editable .svg sibling."""
    fig.savefig(out, dpi=300, bbox_inches='tight')
    if SAVE_SVG:
        svg = os.path.splitext(out)[0] + '.svg'
        fig.savefig(svg, bbox_inches='tight')  # vector; text stays editable
        print(f"  ↳ SVG: {svg}")

# StructOpt is kept in the leaderboard data (all_data.csv / data.js / README) but
# omitted from the task-grid figures (Figs 1-3); MDSim is the 5th task, Folding the 6th.
TASK_ORDER = ['MatGen', 'MolGen', 'Retro', 'Forward', 'MDSim', 'Folding']

TASK_COLORS = {
    'MatGen':    '#2ca02c',
    'MolGen':    '#1f77b4',
    'Retro':     '#17becf',
    'Forward':   '#ff7f0e',
    'StructOpt': '#d62728',
    'MDSim':     '#9467bd',
    'Folding':   '#8c564b',
}

# Display names for figures (data/color keys stay as-is; only the rendered text
# changes). The two MLIP-family tasks both render as "MLIPs": MDSim in the main
# figures and StructOpt in the alt-metric figures (they never co-occur in a panel
# layout; Fig 5 — the only figure with both — excludes StructOpt).
TASK_DISPLAY = {'MDSim': 'MLIPs', 'StructOpt': 'MLIPs'}

def _disp(task):
    """Display label for a task (falls back to the task key itself)."""
    return TASK_DISPLAY.get(task, task)

TASK_PERF_LABEL = {
    'MatGen': 'mSUN (%)',
    'MolGen': 'VUN (%)',
    'Retro': 'Top-50 Acc (%)',
    'Forward': 'Top-3 Acc (%)',
    'StructOpt': 'CPS',
    'MDSim': 'MSD',
    'Folding': 'GDT-TS (%)',
}

ARCH_MARKERS = {
    'MLP':           'o',
    'GNN':           'D',
    'LM':            '^',
    'LLM':           'v',
    'Diffusion':     'P',
    'Flow Matching': 'X',
}

# Architectures ordered cheap → expensive (used as a categorical axis), with a
# distinct qualitative palette (ColorBrewer Dark2) so they don't read as tasks.
ARCH_ORDER = ['MLP', 'GNN', 'LM', 'LLM', 'Flow Matching', 'Diffusion']
ARCH_COLORS = {
    'MLP':           '#1b9e77',
    'GNN':           '#d95f02',
    'LM':            '#7570b3',
    'LLM':           '#e7298a',
    'Flow Matching': '#66a61e',
    'Diffusion':     '#e6ab02',
    'VAE':           '#a6761d',
}
# Short x-tick labels (full names live in the color legend).
ARCH_ABBR = {'Flow Matching': 'Flow', 'Diffusion': 'Diff'}

# (category, main_label, task_key, fallback_desc, fallback_co2, unit)
# task_key=None for non-AI entries
ref_data_raw = [
    ('LLM inference',            'Image generation',         None,         'Stable Diffusion',            1.38,    'image'),
    ('LLM inference',            'Text generation',          None,         'Claude-3.7 Sonnet',           2.12,    '10k in & 1.5k out'),
    ('Everyday activities',      'Smartphone charge',        None,         'iPhone 16 Pro Max',            9.7,    'full charge'),
    ('Chemical simulation',      'Classical MD',             None,         'force field',                  10,     '1M steps'),
    ('AI Synthesis prediction',  'Reaction outcome pred.',   'Forward',    'LlaSMol',                     17.7,    '500 inputs'),
    ('Everyday activities',      'Driving a car',            None,         'EU average',                  170,     'km'),
    ('AI Chemical generation',   'Material generation',      'MatGen',     'MatterGen',                   248,     '1K structures'),
    ('AI Chemical generation',   'Molecule generation',      'MolGen',     'DeFoG',                       355.2,  '10K molecules'),
    ('AI Synthesis prediction',  'Synthesis Planning',       'Retro',      'RetroBridge',                 403,     '500 molecules'),
    ('AI MD simulation',         'MLIP MD',                  'MDSim',      'eSEN',                        3486,    '1M steps'),
    ('AI Protein folding',       'Protein folding',          'Folding',    'ColabFold',                   11.60,  '1 protein'),
    ('Everyday activities',      'Commercial aviation',      None,         'Boeing 737',                  15800,   'km'),
    ('Chemical synthesis',       'Battery synthesis',        None,         'Vanadium flow battery',       37000,   'MWh'),
    ('Chemical synthesis',       'Material synthesis',       None,         'UiO-66-NH₂ (aqueous-based)',  43000,   'kg'),
    ('Chemical simulation',      'Ab initio MD',             None,         'PBE',              140960,  '1M steps'),
    ('Chemical synthesis',       'Organic synthesis',        None,         'Letermovir (Merck)',           382000,  'kg'),
]
    
# Reference point categories (Figure 5)
# AI categories reuse task colors from Figs 1-3; non-AI get distinct colors
REF_CATEGORY_COLORS = {
    'Everyday activities':       '#78909C',   # blue-gray
    'LLM inference':             '#42A5F5',   # blue (same as LLM markers)
    'Chemical simulation':       '#FFB74D',   # amber
    'Chemical synthesis':        '#E53935',   # red
    'AI Chemical generation':    '#2ca02c',   # green (= MatGen)
    'AI Synthesis prediction':   '#ff7f0e',   # orange (= Forward)
    'AI Protein folding':        '#8c564b',   # brown (= Folding)
    'AI MD simulation':          '#9467bd',   # purple (= MDSim)
}

# Ordered legend for Figure 5
REF_LEGEND_ORDER = [
    'Everyday activities', 'LLM inference', 'Chemical simulation',
    'Chemical synthesis', 'AI Chemical generation', 'AI Synthesis prediction',
    'AI Protein folding', 'AI MD simulation',
]


# ── Helpers ────────────────────────────────────────────────────────────────
def parse_size(s):
    """Parse model size string like '4.4M' or '7.2B' to a number."""
    s = str(s).strip()
    m = re.match(r'~?([\d.]+)\s*([KMB]?)', s, re.I)
    if not m:
        return np.nan
    v, u = float(m.group(1)), m.group(2).upper()
    return v * {'K': 1e3, 'M': 1e6, 'B': 1e9, '': 1}[u]


# Star marker (*) appears smaller at the same s value, so scale it up
MARKER_SIZE_SCALE = {'*': 3.0, '^': 1.5, 'v': 1.5}

def marker_size(m, base=150):
    """Return scatter size, scaling up star markers."""
    return base * MARKER_SIZE_SCALE.get(m, 1.0)

def get_arch_legend_handles():
    return [mlines.Line2D([], [], color='gray', marker=m, linestyle='None',
            markersize=14 if m == '*' else 10, label=a) for a, m in ARCH_MARKERS.items()]


def best_badge_corner(lxs, lys, margin=0.16, bw=0.34, bh=0.16):
    """Pick the emptiest of the 4 corners for a ρ badge so it doesn't cover points.

    Normalizes points to axes-fraction coords (mirroring ax.margins) and models
    the badge as a rectangle (bw×bh) anchored in each corner. Returns
    (x, y, ha, va) for the corner whose badge rectangle contains the fewest
    points; ties broken by the largest clearance to the nearest point.
    """
    if len(lxs) < 2:
        return (0.03, 0.97, 'left', 'top')

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5 for _ in vals]
        span = hi - lo
        lo, hi = lo - margin * span, hi + margin * span
        return [(v - lo) / (hi - lo) for v in vals]

    return pick_badge_corner(list(zip(_norm(lxs), _norm(lys))), bw=bw, bh=bh)


def pick_badge_corner(pts, bw=0.34, bh=0.16):
    """Pick the emptiest corner for a badge given points already in axes-fraction
    coords (0-1). Models the badge as a bw×bh rectangle anchored in each corner;
    returns (x, y, ha, va) for the corner with the fewest points inside, ties
    broken by the largest clearance to the nearest point."""
    corners = [(0.03, 0.97, 'left', 'top'), (0.97, 0.97, 'right', 'top'),
               (0.03, 0.03, 'left', 'bottom'), (0.97, 0.03, 'right', 'bottom')]
    if not pts:
        return corners[0]
    best, best_key = corners[0], None
    for cx, cy, ha, va in corners:
        # Badge rectangle footprint anchored at this corner.
        x0 = cx if ha == 'left' else cx - bw
        y0 = cy if va == 'bottom' else cy - bh
        x1, y1 = x0 + bw, y0 + bh
        inside = sum(1 for px, py in pts if x0 <= px <= x1 and y0 <= py <= y1)
        # Clearance: distance from rectangle center to nearest point (bigger = better).
        rcx, rcy = x0 + bw / 2, y0 + bh / 2
        clear = min((px - rcx) ** 2 + (py - rcy) ** 2 for px, py in pts)
        key = (-inside, clear)  # fewest points inside first, then most clearance
        if best_key is None or key > best_key:
            best_key, best = key, (cx, cy, ha, va)
    return best


# ── Data Loading ───────────────────────────────────────────────────────────
def load_data():
    """Load all_data.csv."""
    df = pd.read_csv('analysis/all_data.csv')
    df['_size_num'] = df['model size'].apply(parse_size)

    baselines = (df[df['baseline?'] == True]
                 .set_index('task')[['major_metric']]
                 .rename(columns={'major_metric': 'base_perf'}))
    df = df.join(baselines, on='task')

    return df


# ── Figure 1: Year Trends ─────────────────────────────────────────────────
def _spearman_p(x, y):
    """Spearman ρ and its two-sided p-value (scipy). Returns (rho, p), or
    (None, None) for fewer than 3 pairs or zero variance (undefined ρ)."""
    x, y = pd.Series(list(x)), pd.Series(list(y))
    if len(x) < 3:
        return None, None
    rho, p = spearmanr(x, y)
    if pd.isna(rho):
        return None, None
    return float(rho), float(p)


def _year_spearman_p(grp, ycol):
    """Spearman ρ and p-value between year and ycol over plotted points (drops
    NaN / non-positive values to match the scatter; rank-based, so invariant to
    the log transform on size / CO2 panels). Returns (rho, p) or (None, None)
    for fewer than 3 valid points."""
    sub = grp[['year', ycol]].dropna()
    sub = sub[sub[ycol] > 0]
    if len(sub) < 3:
        return None, None
    return _spearman_p(sub['year'], sub[ycol])


def _rho_p_label(rho, p, signed=False):
    """Two-line badge text: 'ρ = X.XX' over 'p = 0.XX' (or 'p < 0.01' to avoid
    a misleading 'p = 0.00'). Both kept to 2 decimals so the badge stays compact."""
    rs = f'{rho:+.2f}' if signed else f'{rho:.2f}'
    ps = 'p < 0.01' if p < 0.01 else f'p = {p:.2f}'
    return f'ρ = {rs}\n{ps}'


def plot_fig1(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False):
    """6 rows (tasks) × 3 cols (model size, CO2, performance).

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0  # global font scale-up for Figure 1
    SHOW_ARCH = False  # architecture not distinguished here yet (single marker, no legend)
    fig, axes = plt.subplots(6, 3, figsize=(20, 32))

    # Consistent x-axis: use global min/max year across ALL tasks
    all_years = df['year'].dropna().unique()
    year_min, year_max = int(all_years.min()), int(all_years.max())
    year_ticks = list(range(year_min, year_max + 1, 2))  # e.g. 2017, 2019, ...
    x_pad = 0.5
    xlim = (year_min - x_pad, year_max + x_pad)

    # Every task shares the same y-quantity within a column, so each column's
    # y-label is drawn once (via fig.text below) instead of repeated on all 6
    # rows. The performance column reports a different metric per task, but all
    # are mapped to a common 0-1 float scale (percentages ÷ 100), so it shares a
    # single generic "Performance" label; the exact metric is defined in Methods.
    SHARED_YLABELS = {0: 'log₁₀(Model Size)', 1: co2_label, 2: 'Performance'}

    for i, task in enumerate(TASK_ORDER):
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        # Percentage metrics (label contains '%') are shown as fractions on 0-1.
        perf_is_pct = '%' in TASK_PERF_LABEL[task]
        col_specs = [
            ('year', '_size_num',    'log₁₀(Model Size)',       True),
            ('year', co2_col,          co2_label,                  True),
            ('year', 'major_metric',   'Performance',             False),
        ]
        for j, (xcol, ycol, ylabel, use_log10) in enumerate(col_specs):
            ax = axes[i, j]
            texts = []
            for _, row in grp.iterrows():
                yval = row[ycol]
                if pd.isna(yval) or yval <= 0:
                    continue
                if use_log10:
                    yval = np.log10(yval)
                elif j == 2 and perf_is_pct:
                    yval = yval / 100.0  # percentage → fraction (0-1)
                m = ARCH_MARKERS.get(row['model type'], 'o') if SHOW_ARCH else 'o'
                is_base = row.get('baseline?', False)
                ec = 'white'
                lw = 0.6
                ax.scatter(row['year'], yval, color=c, marker=m, s=marker_size(m),
                           edgecolors=ec, linewidths=lw, zorder=3)
                # Nudge overlapping labels (performance column, in 0-1 units)
                tx, ty = row['year'], yval
                if task == 'Forward' and j == 2 and row['model'] == 'RSMILES':
                    ty += 0.03
                elif task == 'Forward' and j == 2 and row['model'] == 'LocalTransform':
                    ty -= 0.03
                elif task == 'MolGen' and j == 2 and row['model'] == 'SmileyLlama':
                    ty += 0.03
                elif task == 'MolGen' and j == 2 and row['model'] == 'REINVENT4':
                    ty -= 0.03
                if not noname:
                    texts.append(ax.text(tx, ty, row['model'],
                                         fontsize=14 * FS, zorder=5))
            # Spearman year-trend annotation (top-left corner, legible over points)
            rho, pval = _year_spearman_p(grp, ycol)
            if rho is not None:
                ax.text(0.05, 0.95, _rho_p_label(rho, pval), transform=ax.transAxes,
                        fontsize=16 * FS, va='top', zorder=6,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            # Set limits BEFORE adjust_text so it adjusts within correct bounds
            ax.set_xlim(xlim)
            ax.set_xticks(year_ticks)
            ax.tick_params(labelsize=16 * FS)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            if i < 5:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Year', fontsize=20 * FS)
            if j == 0:
                ax.text(-0.58, 0.5, _disp(task), transform=ax.transAxes, fontsize=22 * FS,
                        fontweight='bold', color=c, va='center', ha='center', rotation=90)
            # adjust_text AFTER all axis setup
            if texts:
                adjust_text(texts, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    if SHOW_ARCH:
        fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=20 * FS,
                   loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.01),
                   framealpha=0.9, title_fontsize=20 * FS)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.06, hspace=0.35, wspace=0.4)

    # Shared per-column y-labels (size / CO2 / performance), centered vertically.
    for j, lbl in SHARED_YLABELS.items():
        top = axes[0, j].get_position().y1
        bot = axes[5, j].get_position().y0
        x0 = axes[0, j].get_position().x0
        fig.text(x0 - 0.082, (top + bot) / 2, lbl, rotation=90,
                 va='center', ha='center', fontsize=20 * FS)

    fname = '1_year_trends_combined_noname.png' if noname else '1_year_trends_combined.png'
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 1{' (noname)' if noname else ''} saved → {out}")


def plot_fig1_horizontal(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False):
    """3 rows (model size, CO2, performance) × 6 cols (tasks).

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0  # global font scale-up for Figure 1 (horizontal)
    SHOW_ARCH = False  # architecture not distinguished here yet (single marker, no legend)
    fig, axes = plt.subplots(3, 6, figsize=(42, 18))

    # Consistent x-axis: use global min/max year across ALL tasks
    all_years = df['year'].dropna().unique()
    year_min, year_max = int(all_years.min()), int(all_years.max())
    year_ticks = list(range(year_min, year_max + 1, 2))
    x_pad = 0.5
    xlim = (year_min - x_pad, year_max + x_pad)

    row_specs = [
        ('_size_num',    'log₁₀(Model Size)',       True),
        (co2_col,        co2_label,                  True),
        ('major_metric', 'Performance',              False),  # 0-1 float scale
    ]

    for j, task in enumerate(TASK_ORDER):
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        # Percentage metrics (label contains '%') are shown as fractions on 0-1.
        perf_is_pct = '%' in TASK_PERF_LABEL[task]
        for i, (ycol, ylabel, use_log10) in enumerate(row_specs):
            ax = axes[i, j]
            texts = []
            pxs, pys = [], []  # plotted points, for badge-corner placement
            for _, row in grp.iterrows():
                yval = row[ycol]
                if pd.isna(yval) or yval <= 0:
                    continue
                if use_log10:
                    yval = np.log10(yval)
                elif i == 2 and perf_is_pct:
                    yval = yval / 100.0  # percentage → fraction (0-1)
                m = ARCH_MARKERS.get(row['model type'], 'o') if SHOW_ARCH else 'o'
                is_base = row.get('baseline?', False)
                ec = 'white'
                lw = 0.6
                ax.scatter(row['year'], yval, color=c, marker=m, s=marker_size(m),
                           edgecolors=ec, linewidths=lw, zorder=3)
                pxs.append(row['year'])
                pys.append(yval)
                tx, ty = row['year'], yval
                if task == 'Forward' and i == 2 and row['model'] == 'RSMILES':
                    ty += 0.03
                elif task == 'Forward' and i == 2 and row['model'] == 'LocalTransform':
                    ty -= 0.03
                elif task == 'MolGen' and i == 2 and row['model'] == 'SmileyLlama':
                    ty += 0.03
                elif task == 'MolGen' and i == 2 and row['model'] == 'REINVENT4':
                    ty -= 0.03
                if not noname:
                    texts.append(ax.text(tx, ty, row['model'],
                                         fontsize=14 * FS, zorder=5))
            # Spearman year-trend annotation, placed in the emptiest corner so it
            # doesn't cover points. x uses the fixed global xlim; y uses the panel's
            # autoscaled range (matplotlib's default 0.05 margin).
            rho, pval = _year_spearman_p(grp, ycol)
            if rho is not None:
                if len(pys) >= 2 and max(pys) > min(pys):
                    yspan = max(pys) - min(pys)
                    ylo, yhi = min(pys) - 0.05 * yspan, max(pys) + 0.05 * yspan
                    xspan = xlim[1] - xlim[0]
                    pts = [((px - xlim[0]) / xspan, (py - ylo) / (yhi - ylo))
                           for px, py in zip(pxs, pys)]
                    bx, by, bha, bva = pick_badge_corner(pts, bh=0.22)
                else:
                    bx, by, bha, bva = 0.03, 0.97, 'left', 'top'
                ax.text(bx, by, _rho_p_label(rho, pval), transform=ax.transAxes,
                        fontsize=16 * FS, va=bva, ha=bha, zorder=6,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_xlim(xlim)
            ax.set_xticks(year_ticks)
            ax.tick_params(labelsize=16 * FS)
            # More y-ticks on the CO₂ row so it doesn't look too sparse.
            if i == 1:
                ax.locator_params(axis='y', nbins=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            # Column title (task name) on top row
            if i == 0:
                ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=20)
            # Each row shares one y-label, drawn once on the leftmost column. The
            # metric-type ("Model Size" / "CO₂ Emission" / "Performance") is conveyed
            # by the y-label itself, so no separate row title is drawn. Fix the label
            # x-position (axes fraction) so all three rows' y-labels align vertically
            # regardless of differing tick-label widths.
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=20 * FS)
                ax.yaxis.set_label_coords(-0.30, 0.5)
            # x-axis label only on bottom row
            if i < 2:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Year', fontsize=20 * FS)
                # Rotate year ticks so the larger labels don't overlap.
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(30)
                    lbl.set_ha('right')
                    lbl.set_rotation_mode('anchor')
            if texts:
                adjust_text(texts, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    if SHOW_ARCH:
        fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=20 * FS,
                   loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.03),
                   framealpha=0.9, title_fontsize=20 * FS)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.4)
    fname = ('1_year_trends_combined_horizontal_noname.png' if noname
             else '1_year_trends_combined_horizontal.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 1 horizontal{' (noname)' if noname else ''} saved → {out}")


# ── Figure 2: Pareto Frontiers ─────────────────────────────────────────────
def plot_fig2(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False, clean=False):
    """2×3 subplots, Δ Performance (%) vs log10(CO2 ratio), with Pareto front.

    If noname=True, omits per-point labels EXCEPT for models on the Pareto front,
    and saves to *_noname.png.
    If clean=True, omits ALL model labels and enlarges the markers (a template to
    annotate by hand); saves to *_clean.png.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, axes = plt.subplots(2, 3, figsize=(36, 20))  # large so labels don't overlap
    axes = axes.flatten()

    for ax, task in zip(axes, TASK_ORDER):
        grp = df[df['task'] == task].copy()
        base_row = grp[grp['baseline?'] == True].iloc[0]
        base_co2 = base_row[co2_col]
        base_perf = base_row['major_metric']

        grp['co2_ratio'] = grp[co2_col] / base_co2
        grp['log_co2_ratio'] = np.log10(grp['co2_ratio'])
        grp['delta_perf_pct'] = (grp['major_metric'] - base_perf) / abs(base_perf) * 100

        # Pareto front (upper-left dominant)
        grp_sorted = grp.sort_values('log_co2_ratio')
        pareto_x, pareto_y = [], []
        pareto_models = set()
        best_perf = -np.inf
        for _, row in grp_sorted.iterrows():
            if row['delta_perf_pct'] >= best_perf:
                pareto_x.append(row['log_co2_ratio'])
                pareto_y.append(row['delta_perf_pct'])
                pareto_models.add(row['model'])
                best_perf = row['delta_perf_pct']

        # axis limits (independent per task)
        all_x = grp['log_co2_ratio']
        all_y = grp['delta_perf_pct']
        xpad = (all_x.max() - all_x.min()) * 0.2 + 0.3
        xmin, xmax = all_x.min() - xpad, all_x.max() + xpad
        ypad = max(abs(all_y.min()), abs(all_y.max())) * 0.25
        ymin, ymax = all_y.min() - ypad, all_y.max() + ypad

        # shade quadrants (baseline at x=0, y=0)
        ax.fill_between([xmin, 0], [0, 0], [ymax, ymax], color='#d4edda', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [0, 0], [ymax, ymax], color='#fff3cd', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [ymin, ymin], [0, 0], color='#f8d7da', alpha=0.4, zorder=0)
        ax.fill_between([xmin, 0], [ymin, ymin], [0, 0], color='#e2e3e5', alpha=0.4, zorder=0)
        ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        ax.axvline(0, color='black', linewidth=0.8, zorder=1)

        # Pareto front step line
        if len(pareto_x) > 1:
            step_x, step_y = [pareto_x[0]], [pareto_y[0]]
            for k in range(1, len(pareto_x)):
                step_x.extend([pareto_x[k], pareto_x[k]])
                step_y.extend([pareto_y[k - 1], pareto_y[k]])
            ax.plot(step_x, step_y, color='black', linewidth=1.5, linestyle='--',
                    alpha=0.6, zorder=2)

        # Manual label offsets (dx, dy) from the model's own data point. These
        # labels are placed with annotate() — which draws the connector to the
        # FINAL position — and excluded from adjust_text. dx is in log₁₀(CO₂ ratio)
        # units, dy in Δ-perf %.
        MANUAL_NUDGES = {
            'MatGen':  {'CrystalFlow': (-0.5, 0), 'DiffCSP': (-0.5, 0)},
            'MolGen':  {'REINVENT': (-0.5, 0), 'REINVENT4': (-0.5, 0)},
            'Retro':   {'MEGAN': (-0.2, 0), 'neuralsym': (-0.2, 0), 'LocalRetro': (0, 10)},
            'Forward': {'neuralsym': (-0.3, 0), 'MEGAN': (-0.25, -5),
                        'LocalTransform': (-0.2, 12), 'RSMILES': (0, 18)},
            'MDSim':   {'CHGNet': (0.1, -10), 'ORB': (-0.2, 0), 'SevenNet': (-0.2, 0)},
            'Folding': {'OmegaFold': (-0.3, 0), 'ESMFold': (-0.3, 0),
                        'OpenFold': (-0.3, 3), 'ColabFold': (0.0, -1), 'AlphaFold2': (0.25, -3)},
        }
        nudges = MANUAL_NUDGES.get(task, {})
        label_fs = 14 * FS  # match Fig 1 label size (28pt); large but leaves room to de-overlap

        # plot points
        texts2 = []
        manual_objs = []  # manually-placed labels for adjust_text to avoid
        for _, row in grp.iterrows():
            # architecture encoded as marker shape (also in clean mode)
            m = ARCH_MARKERS.get(row['model type'], 'o')
            is_base = row.get('baseline?', False)
            sz = marker_size(m, base=440 if clean else 150)  # bigger points in clean mode
            ec = 'gray'
            lw = 0.8
            ax.scatter(row['log_co2_ratio'], row['delta_perf_pct'],
                       color=TASK_COLORS[task], marker=m, s=sz,
                       edgecolors=ec, linewidths=lw, zorder=4)
            if clean:
                continue  # clean template: markers only, no labels (add by hand)
            # In noname mode, only label models on the Pareto front (plus the baseline)
            if noname and row['model'] not in pareto_models and not is_base:
                continue
            label = f"{row['model']} ({row['year']})"
            dxy = nudges.get(row['model'])
            if dxy is not None:
                # Manually positioned: annotate draws the line to the final spot.
                ann = ax.annotate(label, xy=(row['log_co2_ratio'], row['delta_perf_pct']),
                                  xytext=(row['log_co2_ratio'] + dxy[0],
                                          row['delta_perf_pct'] + dxy[1]),
                                  textcoords='data', fontsize=label_fs, zorder=5,
                                  arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
                manual_objs.append(ann)
            else:
                texts2.append(ax.text(row['log_co2_ratio'], row['delta_perf_pct'], label,
                                      fontsize=label_fs, zorder=5))

        # Set limits and styling BEFORE adjust_text
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('log₁₀(CO₂ eq ratio)', fontsize=20 * FS)
        ax.set_ylabel('Δ Relative Performance (%)', fontsize=20 * FS)
        ax.set_title(_disp(task), fontsize=24 * FS, fontweight='bold', color=TASK_COLORS[task], pad=18)
        ax.tick_params(labelsize=16 * FS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Spearman correlation between CO₂ cost and performance gain (rank-based,
        # robust to outliers; matches Fig 6 style/placement). Omitted in clean mode.
        rho, pval = _spearman_p(grp['log_co2_ratio'], grp['delta_perf_pct'])
        if rho is not None and not clean:
            ax.text(0.96, 0.05, _rho_p_label(rho, pval), transform=ax.transAxes,
                    fontsize=16 * FS, va='bottom', ha='right', zorder=6,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        # adjust_text for the auto-placed labels; avoid the manually-placed ones.
        if texts2:
            adjust_text(texts2, ax=ax, objects=manual_objs or None,
                        expand=(1.5, 1.8), force_text=(2.0, 2.0),
                        force_points=(2.0, 2.0), iterations=200,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    # shared legend
    arch_handles = get_arch_legend_handles()
    quad_handles = [
        mpatches.Patch(color='#d4edda', alpha=0.7, label='Dominant'),
        mpatches.Patch(color='#fff3cd', alpha=0.7, label='Tradeoff'),
        mpatches.Patch(color='#f8d7da', alpha=0.7, label='Dominated'),
        mpatches.Patch(color='#e2e3e5', alpha=0.7, label='Inverse'),
    ]
    legend_handles = arch_handles + quad_handles
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=5, fontsize=16 * FS, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), title_fontsize=16 * FS)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.14, hspace=0.45, wspace=0.32)
    fname = ('2_pareto_delta_pct_clean.png' if clean else
             '2_pareto_delta_pct_noname.png' if noname else '2_pareto_delta_pct.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    tag = ' (clean, no labels)' if clean else (' (noname, Pareto-only labels)' if noname else '')
    print(f"Fig 2{tag} saved → {out}")


def plot_fig2_gradient(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)'):
    """Same layout as Fig 2, but marker color encodes release year (earlier=light, newer=dark).

    Each task subplot uses a sequential colormap derived from the task's base color.
    A single horizontal colorbar at the bottom maps year → shade.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    # Global year range for shared normalization across tasks
    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    norm = Normalize(vmin=year_min, vmax=year_max)

    def task_cmap(task):
        """Light-to-dark gradient ending at the task's base color."""
        base = to_rgb(TASK_COLORS[task])
        light = tuple(0.85 + 0.15 * c for c in base)  # very light tint
        dark = tuple(0.55 * c for c in base)          # darker than base
        return LinearSegmentedColormap.from_list(f'{task}_grad', [light, base, dark])

    for ax, task in zip(axes, TASK_ORDER):
        grp = df[df['task'] == task].copy()
        base_row = grp[grp['baseline?'] == True].iloc[0]
        base_co2 = base_row[co2_col]
        base_perf = base_row['major_metric']

        grp['co2_ratio'] = grp[co2_col] / base_co2
        grp['log_co2_ratio'] = np.log10(grp['co2_ratio'])
        grp['delta_perf_pct'] = (grp['major_metric'] - base_perf) / abs(base_perf) * 100

        grp_sorted = grp.sort_values('log_co2_ratio')
        pareto_x, pareto_y = [], []
        best_perf = -np.inf
        for _, row in grp_sorted.iterrows():
            if row['delta_perf_pct'] >= best_perf:
                pareto_x.append(row['log_co2_ratio'])
                pareto_y.append(row['delta_perf_pct'])
                best_perf = row['delta_perf_pct']

        all_x = grp['log_co2_ratio']
        all_y = grp['delta_perf_pct']
        xpad = (all_x.max() - all_x.min()) * 0.2 + 0.3
        xmin, xmax = all_x.min() - xpad, all_x.max() + xpad
        ypad = max(abs(all_y.min()), abs(all_y.max())) * 0.25
        ymin, ymax = all_y.min() - ypad, all_y.max() + ypad

        ax.fill_between([xmin, 0], [0, 0], [ymax, ymax], color='#d4edda', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [0, 0], [ymax, ymax], color='#fff3cd', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [ymin, ymin], [0, 0], color='#f8d7da', alpha=0.4, zorder=0)
        ax.fill_between([xmin, 0], [ymin, ymin], [0, 0], color='#e2e3e5', alpha=0.4, zorder=0)
        ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        ax.axvline(0, color='black', linewidth=0.8, zorder=1)

        if len(pareto_x) > 1:
            step_x, step_y = [pareto_x[0]], [pareto_y[0]]
            for k in range(1, len(pareto_x)):
                step_x.extend([pareto_x[k], pareto_x[k]])
                step_y.extend([pareto_y[k - 1], pareto_y[k]])
            ax.plot(step_x, step_y, color='black', linewidth=1.5, linestyle='--',
                    alpha=0.6, zorder=2)

        MANUAL_POSITIONS = {
            'Forward': {
                'MEGAN': (-0.2, 80),
                'RSMILES': (1.2, 95.0),
                'LocalTransform': (-0.1, 95.0),
                'MolecularTransformer': (0.3, 55.0),
                'Graph2SMILES': (0.1, 35.0),
                'Chemformer': (1.2, 70.0),
            },
        }
        positions = MANUAL_POSITIONS.get(task, {})
        label_fs = 14

        cmap = task_cmap(task)
        texts2 = []
        for _, row in grp.iterrows():
            m = ARCH_MARKERS.get(row['model type'], 'o')
            is_base = row.get('baseline?', False)
            sz = marker_size(m)
            ec = 'gray'
            lw = 0.8
            color = cmap(norm(row['year']))
            ax.scatter(row['log_co2_ratio'], row['delta_perf_pct'],
                       color=color, marker=m, s=sz,
                       edgecolors=ec, linewidths=lw, zorder=4)
            label = f"{row['model']} ({row['year']})"
            pos = positions.get(row['model'], None)
            if pos is not None:
                ax.annotate(label, xy=(row['log_co2_ratio'], row['delta_perf_pct']),
                            xytext=pos, fontsize=label_fs, zorder=5,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
            else:
                texts2.append(ax.text(row['log_co2_ratio'], row['delta_perf_pct'], label,
                                      fontsize=label_fs, zorder=5))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('log₁₀(CO₂ eq ratio)', fontsize=20)
        ax.set_ylabel('Δ Relative Performance (%)', fontsize=20)
        ax.set_title(_disp(task), fontsize=24, fontweight='bold', color=TASK_COLORS[task])
        ax.tick_params(labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if texts2:
            adjust_text(texts2, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                        force_points=(2.0, 2.0), iterations=200,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    arch_handles = get_arch_legend_handles()
    quad_handles = [
        mpatches.Patch(color='#d4edda', alpha=0.7, label='Dominant'),
        mpatches.Patch(color='#fff3cd', alpha=0.7, label='Tradeoff'),
        mpatches.Patch(color='#f8d7da', alpha=0.7, label='Dominated'),
        mpatches.Patch(color='#e2e3e5', alpha=0.7, label='Inverse'),
    ]
    fig.legend(handles=arch_handles + quad_handles,
               loc='lower center', ncol=5, fontsize=20, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), title_fontsize=20)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.14, hspace=0.35, wspace=0.3)
    out = os.path.join(OUT_DIR, '2_pareto_delta_pct_gradient.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 2 gradient saved → {out}")


# ── Figure 3: CO2 vs Compute Time (bubble = model size) ───────────────────
def plot_fig3(df, co2_col='CO2_per_exp', co2_label='log₁₀(CO₂/exp)', noname=False):
    """Per-task bubble chart: x = Inference Time, y = CO₂, bubble AREA = Model Size.

    Replaces the old 3-way pairwise grid (plot_fig3_pairwise), which repeated all
    pairings of the same three variables {Model Size, Inference Time, CO₂}. Here
    the third variable (size) is an encoding rather than its own panel, and the
    Inference-Time→CO₂ regression (the near-deterministic energy relationship) is
    kept with an R² box. 2×3 grid, one panel per task.

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    # Model size → bubble area. Params span ~4 orders of magnitude, so map on a
    # log scale (true area-proportional would make small models invisible).
    fig_df = df[df['task'].isin(TASK_ORDER)]
    sizes = fig_df['_size_num'].dropna()
    logmin, logmax = np.log10(sizes.min()), np.log10(sizes.max())

    def bubble_area(params):
        t = (np.log10(params) - logmin) / (logmax - logmin)
        return 150 + t * (3000 - 150)

    for k, task in enumerate(TASK_ORDER):
        ax = axes[k]
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        log_xs, log_ys = [], []
        texts = []
        for _, row in grp.iterrows():
            xv, yv, sv = row['inference_time_per_exp'], row[co2_col], row['_size_num']
            if pd.isna(xv) or pd.isna(yv) or pd.isna(sv) or xv <= 0 or yv <= 0:
                continue
            lx, ly = np.log10(xv), np.log10(yv)
            log_xs.append(lx)
            log_ys.append(ly)
            # color = task, size = model params (faithful, no per-shape boost so
            # large LLM stars don't dominate), shape = architecture
            m = ARCH_MARKERS.get(row['model type'], 'o')
            ax.scatter(lx, ly, s=bubble_area(sv),
                       marker=m, color=c, alpha=0.55,
                       edgecolors='white', linewidths=1.2, zorder=3)
            if not noname:
                texts.append(ax.text(lx, ly, row['model'], fontsize=12 * FS, zorder=5))
        # Inference-Time → CO₂ regression (the strong, near-deterministic relationship)
        if len(log_xs) > 1:
            xa, ya = np.array(log_xs), np.array(log_ys)
            coef = np.polyfit(xa, ya, 1)
            r2 = 1 - np.sum((ya - np.polyval(coef, xa))**2) / \
                     np.sum((ya - ya.mean())**2)
            xf = np.linspace(xa.min(), xa.max(), 50)
            ax.plot(xf, np.polyval(coef, xf), 'k--', lw=1.5, alpha=0.5, zorder=2)
            ax.text(0.05, 0.95, f'R²={r2:.2f}', transform=ax.transAxes,
                    fontsize=16 * FS, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=12)
        ax.tick_params(labelsize=16 * FS)
        ax.locator_params(axis='y', nbins=6)
        ax.margins(0.18)  # extra room so large bubbles aren't clipped at the edges
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, which='both')
        if k >= 3:
            ax.set_xlabel('log₁₀(Inf. Time/exp)', fontsize=20 * FS)
        if k % 3 == 0:
            ax.set_ylabel(co2_label, fontsize=20 * FS)
        if texts:
            adjust_text(texts, ax=ax, expand=(1.4, 1.6), force_text=(1.5, 1.5),
                        force_points=(1.5, 1.5), iterations=150,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    # Two legends: architecture (shape) and model size (bubble area).
    arch_handles = get_arch_legend_handles()
    fig.legend(handles=arch_handles, title='Architecture (shape)', fontsize=15 * FS,
               loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, 0.105),
               framealpha=0.9, title_fontsize=15 * FS)
    size_refs = [(1e6, '1M'), (1e8, '100M'), (1e9, '1B')]
    size_handles = [mlines.Line2D([], [], marker='o', linestyle='None',
                    markerfacecolor='gray', markeredgecolor='white', alpha=0.6,
                    markersize=np.sqrt(bubble_area(v)), label=lbl)
                    for v, lbl in size_refs]
    fig.legend(handles=size_handles, title='Model Size (bubble area)', fontsize=15 * FS,
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01),
               framealpha=0.9, title_fontsize=15 * FS, labelspacing=1.3,
               handletextpad=1.2, borderpad=0.8, columnspacing=3.0)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.30, hspace=0.32, wspace=0.28)
    fname = ('3_co2_decomposition_combined_noname.png' if noname
             else '3_co2_decomposition_combined.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 3{' (noname)' if noname else ''} saved → {out}")


def plot_fig3_pairwise(df, co2_col='CO2_per_exp', co2_label='log₁₀(CO₂/exp)', noname=False):
    """[Superseded by plot_fig3] 6 rows (tasks) × 3 cols (size vs CO2, time vs
    CO2, size vs time). Kept for reference / easy revert.

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, axes = plt.subplots(6, 3, figsize=(20, 32))

    # x-labels kept short so they don't overlap at the 2x font scale.
    panels = [
        ('_size_num',                co2_col,                   'log₁₀(Model Size)',     co2_label),
        ('inference_time_per_exp',   co2_col,                   'log₁₀(Inf. Time/exp)',  co2_label),
        ('_size_num',                'inference_time_per_exp',  'log₁₀(Model Size)',     'log₁₀(Inf. Time/exp)'),
    ]
    # Each column shares one y-quantity across all 6 task rows, so its y-label is
    # drawn once (via fig.text below) instead of repeated on every row (match Fig 1).
    SHARED_YLABELS = {0: co2_label, 1: co2_label, 2: 'log₁₀(Inf. Time/exp)'}

    # Compute shared x-limits per column across all tasks
    col_xlims = []
    for j, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
        all_logx = []
        for task in TASK_ORDER:
            grp = df[df['task'] == task]
            for _, row in grp.iterrows():
                xv = row[xcol]
                if pd.notna(xv) and xv > 0:
                    all_logx.append(np.log10(xv))
        pad = (max(all_logx) - min(all_logx)) * 0.1
        col_xlims.append((min(all_logx) - pad, max(all_logx) + pad))

    for i, task in enumerate(TASK_ORDER):
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        for j, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
            ax = axes[i, j]
            texts3 = []
            log_xs, log_ys = [], []
            for _, row in grp.iterrows():
                xv, yv = row[xcol], row[ycol]
                if pd.isna(xv) or pd.isna(yv) or xv <= 0 or yv <= 0:
                    continue
                log_xv = np.log10(xv)
                log_yv = np.log10(yv)
                log_xs.append(log_xv)
                log_ys.append(log_yv)
                m = ARCH_MARKERS.get(row['model type'], 'o')
                is_base = row.get('baseline?', False)
                sz = marker_size(m)
                ec = 'white'
                lw = 0.6
                ax.scatter(log_xv, log_yv, color=c, marker=m, s=sz,
                           edgecolors=ec, linewidths=lw, zorder=3)
                # In noname mode, still label NequIP/ORB in the Model Size vs
                # Inference Time column (j==2) for StructOpt/MDSim — worth discussing.
                highlight = (j == 2 and task in ('StructOpt', 'MDSim')
                             and row['model'] in ('NequIP', 'ORB'))
                if not noname or highlight:
                    label_fs = (20 if noname else 14) * FS  # fewer labels in noname → larger font
                    texts3.append(ax.text(log_xv, log_yv, row['model'], fontsize=label_fs, zorder=5))
            # Regression + R² for inference time vs CO₂ (column 1)
            if j == 1 and len(log_xs) > 1:
                log_xs_arr = np.array(log_xs)
                log_ys_arr = np.array(log_ys)
                coef = np.polyfit(log_xs_arr, log_ys_arr, 1)
                r2 = 1 - np.sum((log_ys_arr - np.polyval(coef, log_xs_arr))**2) / \
                         np.sum((log_ys_arr - log_ys_arr.mean())**2)
                xfit = np.linspace(log_xs_arr.min(), log_xs_arr.max(), 50)
                ax.plot(xfit, np.polyval(coef, xfit), 'k--', lw=1.2, alpha=0.5, zorder=2)
                ax.text(0.05, 0.95, f'R²={r2:.2f}', transform=ax.transAxes,
                        fontsize=16 * FS, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            # Shared x-limits per column
            ax.set_xlim(col_xlims[j])
            ax.tick_params(labelsize=16 * FS)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            # No column titles: each column's x-label + shared y-label already
            # state the "X vs Y" relationship (match Fig 1, avoids title overlap at 2x).
            if i < 5:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(xlabel, fontsize=20 * FS)
            if j == 0:
                ax.text(-0.58, 0.5, _disp(task), transform=ax.transAxes, fontsize=22 * FS,
                        fontweight='bold', color=c, va='center', ha='center', rotation=90)
            # adjust_text AFTER all axis setup
            if texts3:
                adjust_text(texts3, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=20 * FS,
               loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.035),
               framealpha=0.9, title_fontsize=20 * FS)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.10, hspace=0.35, wspace=0.4)

    # Shared per-column y-labels, centered vertically over each column (match Fig 1).
    for j, lbl in SHARED_YLABELS.items():
        top = axes[0, j].get_position().y1
        bot = axes[5, j].get_position().y0
        x0 = axes[0, j].get_position().x0
        fig.text(x0 - 0.082, (top + bot) / 2, lbl, rotation=90,
                 va='center', ha='center', fontsize=20 * FS)

    fname = ('3_co2_decomposition_pairwise_noname.png' if noname
             else '3_co2_decomposition_pairwise.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 3 pairwise{' (noname)' if noname else ''} saved → {out}")


def plot_fig3_horizontal(df, co2_col='CO2_per_exp', co2_label='log₁₀(CO₂/exp)', noname=False):
    """3 rows (panels) × 6 cols (tasks). Transposed version of fig3.

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, axes = plt.subplots(3, 6, figsize=(42, 18))

    # x-labels kept short so they don't overlap at the 2x font scale.
    panels = [
        ('_size_num',                co2_col,                   'log₁₀(Model Size)',     co2_label),
        ('inference_time_per_exp',   co2_col,                   'log₁₀(Inf. Time/exp)',  co2_label),
        ('_size_num',                'inference_time_per_exp',  'log₁₀(Model Size)',     'log₁₀(Inf. Time/exp)'),
    ]

    # Compute shared x-limits per row (panel) across all tasks
    row_xlims = []
    for i, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
        all_logx = []
        for task in TASK_ORDER:
            grp = df[df['task'] == task]
            for _, row in grp.iterrows():
                xv = row[xcol]
                if pd.notna(xv) and xv > 0:
                    all_logx.append(np.log10(xv))
        pad = (max(all_logx) - min(all_logx)) * 0.1
        row_xlims.append((min(all_logx) - pad, max(all_logx) + pad))

    for j, task in enumerate(TASK_ORDER):
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        for i, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
            ax = axes[i, j]
            texts3 = []
            log_xs, log_ys = [], []
            for _, row in grp.iterrows():
                xv, yv = row[xcol], row[ycol]
                if pd.isna(xv) or pd.isna(yv) or xv <= 0 or yv <= 0:
                    continue
                log_xv = np.log10(xv)
                log_yv = np.log10(yv)
                log_xs.append(log_xv)
                log_ys.append(log_yv)
                m = ARCH_MARKERS.get(row['model type'], 'o')
                is_base = row.get('baseline?', False)
                sz = marker_size(m)
                ec = 'white'
                lw = 0.6
                ax.scatter(log_xv, log_yv, color=c, marker=m, s=sz,
                           edgecolors=ec, linewidths=lw, zorder=3)
                if not noname:
                    texts3.append(ax.text(log_xv, log_yv, row['model'], fontsize=14 * FS, zorder=5))
            # Regression + R² for inference time vs CO₂ (row 1)
            if i == 1 and len(log_xs) > 1:
                log_xs_arr = np.array(log_xs)
                log_ys_arr = np.array(log_ys)
                coef = np.polyfit(log_xs_arr, log_ys_arr, 1)
                r2 = 1 - np.sum((log_ys_arr - np.polyval(coef, log_xs_arr))**2) / \
                         np.sum((log_ys_arr - log_ys_arr.mean())**2)
                xfit = np.linspace(log_xs_arr.min(), log_xs_arr.max(), 50)
                ax.plot(xfit, np.polyval(coef, xfit), 'k--', lw=1.2, alpha=0.5, zorder=2)
                ax.text(0.05, 0.95, f'R²={r2:.2f}', transform=ax.transAxes,
                        fontsize=16 * FS, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            # Shared x-limits per row
            ax.set_xlim(row_xlims[i])
            ax.tick_params(labelsize=16 * FS)
            # More y-ticks on the CO₂ rows so they don't look too sparse.
            if i in (0, 1):
                ax.locator_params(axis='y', nbins=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            # Column title (task name) on top row
            if i == 0:
                ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=20)
            # Each row shares one y-label, drawn once on the leftmost column (the
            # "X vs Y" relationship is conveyed by the x-label + y-label, so no row
            # title). Fixed label x-position so all rows' y-labels align vertically.
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=20 * FS)
                ax.yaxis.set_label_coords(-0.30, 0.5)
            # Each row has a different x variable — always show x-axis
            ax.set_xlabel(xlabel, fontsize=20 * FS)
            if texts3:
                adjust_text(texts3, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=20 * FS,
               loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9, title_fontsize=20 * FS)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.10, hspace=0.45, wspace=0.4)
    fname = ('3_co2_decomposition_combined_horizontal_noname.png' if noname
             else '3_co2_decomposition_combined_horizontal.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 3 horizontal{' (noname)' if noname else ''} saved → {out}")


# ── Figure: Architecture, speed & CO2 (within each domain) ─────────────────
def plot_fig_arch_speed_co2(df, co2_col='CO2_per_exp', noname=False):
    """Architecture vs prediction speed AND CO₂, within each domain.

    2×3 small multiples (one panel per domain): x = log₁₀(inference time/exp),
    y = log₁₀(CO₂/exp), marker SHAPE = architecture, color = domain. Both costs
    are only comparable within a domain, so every panel has its own scale. The
    architecture shapes line up along the speed→CO₂ trend, showing architecture
    sets where a model lands — fast & clean (MLP/GNN) vs slow & dirty
    (LLM/Diffusion). One figure covers both speed and carbon.

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for k, task in enumerate(TASK_ORDER):
        ax = axes[k]
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        texts = []
        lxs, lys = [], []
        for _, row in grp.iterrows():
            xv, yv = row['inference_time_per_exp'], row[co2_col]
            if pd.isna(xv) or pd.isna(yv) or xv <= 0 or yv <= 0:
                continue
            lx, ly = np.log10(xv), np.log10(yv)
            lxs.append(lx)
            lys.append(ly)
            m = ARCH_MARKERS.get(row['model type'], 'o')  # shape = architecture
            ax.scatter(lx, ly, marker=m, color=c,
                       s=320 * MARKER_SIZE_SCALE.get(m, 1.0), alpha=0.85,
                       edgecolors='white', linewidths=1.3, zorder=3)
            if not noname:
                texts.append(ax.text(lx, ly, row['model'], fontsize=10 * FS, zorder=5))
        # Spearman ρ between log(inference time) and log(CO₂) (one correlation
        # measure across all figures; rank-invariant to the log transform).
        badge = None
        rho, pval = _spearman_p(lxs, lys)
        if rho is not None:
            bx, by, bha, bva = best_badge_corner(lxs, lys, bh=0.22)
            badge = ax.text(bx, by, _rho_p_label(rho, pval), transform=ax.transAxes,
                    fontsize=16 * FS, va=bva, ha=bha, zorder=6,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        ax.margins(0.16)  # headroom so points don't crowd title/edges
        ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=18)
        ax.tick_params(labelsize=16 * FS)
        # x: cap the count so narrow ranges (MDSim) don't crowd, but keep ≥3.
        ax.locator_params(axis='x', nbins=4, min_n_ticks=3)
        ax.locator_params(axis='y', nbins=6)  # more than the sparse default (≤3)
        ax.grid(True, alpha=0.2, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k >= 3:
            ax.set_xlabel('log₁₀(Inf. Time/exp)', fontsize=20 * FS)
        if k % 3 == 0:
            ax.set_ylabel('log₁₀(CO₂/exp)', fontsize=20 * FS)
            # Fixed label x-position so both rows' y-labels align vertically.
            ax.yaxis.set_label_coords(-0.22, 0.5)
        if texts:
            adjust_text(texts, ax=ax, objects=[badge] if badge else None,
                        expand=(1.3, 1.5), force_text=(1.2, 1.2),
                        force_static=(0.8, 0.8), iterations=120,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(),
               ncol=len(ARCH_MARKERS), loc='lower center', bbox_to_anchor=(0.5, -0.02),
               fontsize=15 * FS, framealpha=0.9)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.20,
                        hspace=0.32, wspace=0.28)
    fname = ('arch_speed_vs_co2_noname.png' if noname else 'arch_speed_vs_co2.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig (arch speed vs CO2){' (noname)' if noname else ''} saved → {out}")


def plot_fig_arch_size_co2(df, co2_col='CO2_per_exp', noname=False):
    """Architecture vs model size AND CO₂, within each domain.

    Sibling of plot_fig_arch_speed_co2 but with x = log₁₀(model size, params).
    2×3 small multiples (one panel per domain): x = log₁₀(parameters),
    y = log₁₀(CO₂/exp), marker SHAPE = architecture, color = domain. CO₂ is
    only comparable within a domain, so every panel has its own scale. Shows
    whether bigger models cost more carbon — and where each architecture lands.

    If noname=True, omits per-point model labels and saves to *_noname.png.
    """
    FS = 2.0
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for k, task in enumerate(TASK_ORDER):
        ax = axes[k]
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        texts = []
        lxs, lys = [], []
        for _, row in grp.iterrows():
            xv, yv = parse_size(row['model size']), row[co2_col]
            if pd.isna(xv) or pd.isna(yv) or xv <= 0 or yv <= 0:
                continue
            lx, ly = np.log10(xv), np.log10(yv)
            lxs.append(lx)
            lys.append(ly)
            m = ARCH_MARKERS.get(row['model type'], 'o')  # shape = architecture
            ax.scatter(lx, ly, marker=m, color=c,
                       s=320 * MARKER_SIZE_SCALE.get(m, 1.0), alpha=0.85,
                       edgecolors='white', linewidths=1.3, zorder=3)
            if not noname:
                texts.append(ax.text(lx, ly, row['model'], fontsize=10 * FS, zorder=5))
        # Spearman ρ between log(model size) and log(CO₂) (one correlation
        # measure across all figures; rank-invariant to the log transform).
        badge = None
        rho, pval = _spearman_p(lxs, lys)
        if rho is not None:
            bx, by, bha, bva = best_badge_corner(lxs, lys, bh=0.22)
            badge = ax.text(bx, by, _rho_p_label(rho, pval), transform=ax.transAxes,
                    fontsize=16 * FS, va=bva, ha=bha, zorder=6,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        ax.margins(0.16)  # headroom so points don't crowd title/edges
        ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=18)
        ax.tick_params(labelsize=16 * FS)
        ax.locator_params(axis='x', nbins=4, min_n_ticks=3)
        ax.locator_params(axis='y', nbins=6)  # more than the sparse default (≤3)
        ax.grid(True, alpha=0.2, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k >= 3:
            ax.set_xlabel('log₁₀(Model Size)', fontsize=20 * FS)
        if k % 3 == 0:
            ax.set_ylabel('log₁₀(CO₂/exp)', fontsize=20 * FS)
            # Fixed label x-position so both rows' y-labels align vertically.
            ax.yaxis.set_label_coords(-0.22, 0.5)
        if texts:
            adjust_text(texts, ax=ax, objects=[badge] if badge else None,
                        expand=(1.3, 1.5), force_text=(1.2, 1.2),
                        force_static=(0.8, 0.8), iterations=120,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(),
               ncol=len(ARCH_MARKERS), loc='lower center', bbox_to_anchor=(0.5, -0.02),
               fontsize=15 * FS, framealpha=0.9)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.20,
                        hspace=0.32, wspace=0.28)
    fname = ('arch_size_vs_co2_noname.png' if noname else 'arch_size_vs_co2.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig (arch size vs CO2){' (noname)' if noname else ''} saved → {out}")


# ── Figure 4: CO2 Reference Points ────────────────────────────────────────
def _compute_pareto(grp):
    """Return set of model names that are Pareto-optimal (higher metric, lower CO2)."""
    pareto = set()
    rows = grp.to_dict('records')
    for r in rows:
        dominated = any(
            o['major_metric'] >= r['major_metric'] and o['CO2_per_job'] <= r['CO2_per_job'] and
            (o['major_metric'] > r['major_metric'] or o['CO2_per_job'] < r['CO2_per_job'])
            for o in rows if o is not r
        )
        if not dominated:
            pareto.add(r['model'])
    return pareto

# ── Figure 4: CO2 Reference Points ────────────────────────────────────────
def plot_fig4(df, highlight_ai=True):
    """Horizontal bar chart of CO2 reference points, ordered by magnitude.
    For AI task entries, the bar and description show the worst Pareto-optimal model.

    Parameters
    ----------
    highlight_ai : bool
        When True, use a 2-tone palette: bold accent for AI models,
        neutral gray for non-AI references.  Default False keeps the
        original 7-colour category scheme.
    """

    # Compute worst Pareto model per task (StructOpt and MDSim each plot independently)
    task_map = {
        'Forward':   'Forward',
        'Retro':     'Retro',
        'MolGen':    'MolGen',
        'MatGen':    'MatGen',
        'Folding':   'Folding',
        'MDSim':     'MDSim',
    }
    worst_pareto = {}  # task -> (model_name, co2_per_job)
    for task, key in task_map.items():
        grp = df[df['task'] == task].copy()
        if grp.empty:
            continue
        pareto_models = _compute_pareto(grp)
        pareto_grp = grp[grp['model'].isin(pareto_models)]
        worst = pareto_grp.loc[pareto_grp['CO2_per_job'].idxmax()]
        # Protein folding runs one protein at a time → report per-protein CO2
        worst_pareto[key] = (worst['model'], worst['CO2_per_exp' if key == 'Folding' else 'CO2_per_job'])

    # (category, main_label, task_key, fallback_desc, fallback_co2, unit)
    # task_key=None for non-AI entries

    # Resolve AI entries to worst Pareto model
    ref_data = []
    for (cat, label, task_key, fallback_desc, fallback_co2, unit) in ref_data_raw:
        if task_key and task_key in worst_pareto:
            model_name, co2 = worst_pareto[task_key]
            ref_data.append((cat, label, model_name, co2, unit))
        else:
            ref_data.append((cat, label, fallback_desc, fallback_co2, unit))

    ref_data.sort(key=lambda x: x[3])

    categories  = [d[0] for d in ref_data]
    main_labels = [d[1] for d in ref_data]
    descs       = [d[2] for d in ref_data]
    values      = [d[3] for d in ref_data]
    units       = [d[4] for d in ref_data]
    AI_CATEGORIES = {'AI Chemical generation', 'AI Synthesis prediction',
                     'AI Protein folding', 'AI MD simulation'}
    CHEM_CATEGORIES = {'Chemical simulation', 'Chemical synthesis'}

    if highlight_ai:
        ai_color   = '#1E88E5'   # bold blue for AI models
        chem_color = '#FFAB91'   # muted coral for conventional chemistry
        base_color = '#CFD8DC'   # light gray for everyday / LLM baselines

        def _pick(cat):
            if cat in AI_CATEGORIES:
                return ai_color, '#1565C0', 1.5, 0.72, '///'
            if cat in CHEM_CATEGORIES:
                return chem_color, '#E0E0E0', 0.5, 0.58, ''
            return base_color, '#E0E0E0', 0.5, 0.55, ''

        colors, edgecolors, linewidths, heights, hatches = zip(
            *[_pick(c) for c in categories])
    else:
        colors = [REF_CATEGORY_COLORS[c] for c in categories]
        edgecolors = ['white'] * len(categories)
        linewidths = [0.5] * len(categories)
        heights = [0.65] * len(categories)
        hatches = [''] * len(categories)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (val, c, ec, lw, h, hp) in enumerate(
            zip(values, colors, edgecolors, linewidths, heights, hatches)):
        ax.barh(i, val, color=c, edgecolor=ec, linewidth=lw, height=h,
                hatch=hp)

    for i, (val, unit) in enumerate(zip(values, units)):
        label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                 else f'{val:.1f} g CO₂ eq/{unit}')
        ax.text(val * 1.4, i, label, va='center', fontsize=8.5)

    ax.set_yticks(range(len(ref_data)))
    ax.set_yticklabels([''] * len(ref_data))
    for i, (main, desc) in enumerate(zip(main_labels, descs)):
        ax.text(-0.03, i, main, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=10.5, fontweight='bold', color='black')
        ax.text(-0.035, i - 0.42, f'{desc}', transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8.5, fontstyle='italic', color='#777777')

    ax.set_xscale('log')
    ax.set_xlabel('CO₂ Emission', fontsize=14)
    # ax.set_title('CO₂ Emission Reference Points', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, 5e8)

    if highlight_ai:
        leg = [
            mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                           linewidth=1.5, hatch='///',
                           label='AI models (this work)'),
            mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
            mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
        ]
    else:
        leg = [mpatches.Patch(facecolor=REF_CATEGORY_COLORS[cat], label=cat)
               for cat in REF_LEGEND_ORDER]
    ax.legend(handles=leg, loc='lower right', fontsize=10,
              framealpha=0.9, title='Category', title_fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.32)
    out = os.path.join(OUT_DIR, '4_co2_reference_points.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 4 saved → {out}")

def plot_fig4b(df):
    """Figure 4b: CO2 reference points with two bars per AI task —
    original worst-Pareto bar (lighter) and second-best Pareto model (bold).

    Second-best models (hardcoded):
        Forward → LocalTransform, Retro → LocalRetro, MolGen → REINVENT4,
        MatGen → DiffCSP, StructOpt → NequIP, MDSim → NequIP
    """
    # ── Compute worst Pareto model per task (same as fig4) ──
    task_map = {
        'Forward':   'Forward',
        'Retro':     'Retro',
        'MolGen':    'MolGen',
        'MatGen':    'MatGen',
        'Folding':   'Folding',
        'MDSim':     'MDSim',
    }
    worst_pareto = {}
    for task, key in task_map.items():
        grp = df[df['task'] == task].copy()
        if grp.empty:
            continue
        pareto_models = _compute_pareto(grp)
        pareto_grp = grp[grp['model'].isin(pareto_models)]
        worst = pareto_grp.loc[pareto_grp['CO2_per_job'].idxmax()]
        # Protein folding runs one protein at a time → report per-protein CO2
        worst_pareto[key] = (worst['model'], worst['CO2_per_exp' if key == 'Folding' else 'CO2_per_job'])

    # ── Second-best models (user-specified) ──
    second_best_names = {
        'Forward':   'LocalTransform',
        'Retro':     'LocalRetro',
        'MolGen':    'REINVENT4',
        'MatGen':    'DiffCSP',
        'Folding':   'OpenFold',
        'MDSim':     'SevenNet',
    }
    second_best = {}
    for task, model_name in second_best_names.items():
        row = df[(df['task'] == task) & (df['model'] == model_name)]
        if not row.empty:
            co2 = row.iloc[0]['CO2_per_exp' if task == 'Folding' else 'CO2_per_job']
            second_best[task] = (model_name, co2)

    # Resolve worst-Pareto entries
    ref_data = []
    for (cat, label, task_key, fallback_desc, fallback_co2, unit) in ref_data_raw:
        if task_key and task_key in worst_pareto:
            model_name, co2 = worst_pareto[task_key]
            ref_data.append((cat, label, task_key, model_name, co2, unit))
        else:
            ref_data.append((cat, label, None, fallback_desc, fallback_co2, unit))

    ref_data.sort(key=lambda x: x[4])

    categories  = [d[0] for d in ref_data]
    main_labels = [d[1] for d in ref_data]
    task_keys   = [d[2] for d in ref_data]
    descs       = [d[3] for d in ref_data]
    values      = [d[4] for d in ref_data]
    units       = [d[5] for d in ref_data]

    AI_CATEGORIES = {'AI Chemical generation', 'AI Synthesis prediction',
                     'AI Protein folding', 'AI MD simulation'}

    # ── Colour palette ──
    ai_color       = '#1E88E5'
    ai_color_light = '#D6EAFF'   # very light version for original bar
    chem_color     = '#FFAB91'
    base_color     = '#CFD8DC'

    def _pick_light(cat):
        """Lighter colours for the original (worst-Pareto) bars."""
        if cat in AI_CATEGORIES:
            return ai_color_light, '#D6EAFF', 0.5, ''
        if cat in {'Chemical simulation', 'Chemical synthesis'}:
            return chem_color, '#E0E0E0', 0.5, ''
        return base_color, '#E0E0E0', 0.5, ''

    def _pick_bold(cat):
        """Bold colours for the second-best bars."""
        return ai_color, '#1565C0', 1.5, '///'

    # ── Draw ──
    FS = 2.0  # global font scale-up (match other figures); taller canvas so the
    bar_height = 0.72            # two-line row labels don't overlap at 2x
    fig, ax = plt.subplots(figsize=(16, 18))

    # Pass 1: draw original bars (lighter)
    for i, (val, cat) in enumerate(zip(values, categories)):
        fc, ec, lw, hp = _pick_light(cat)
        h = bar_height if cat in AI_CATEGORIES else 0.58 if cat in {'Chemical simulation', 'Chemical synthesis'} else 0.55
        ax.barh(i, val, color=fc, edgecolor=ec, linewidth=lw, height=h, hatch=hp)

    # Pass 2: overlay second-best bars (bold) on AI rows
    for i, (cat, tk) in enumerate(zip(categories, task_keys)):
        if tk and tk in second_best:
            sb_name, sb_co2 = second_best[tk]
            fc, ec, lw, hp = _pick_bold(cat)
            ax.barh(i, sb_co2, color=fc, edgecolor=ec, linewidth=lw,
                    height=bar_height, hatch=hp)

    # ── CO2 value labels (only second-best for AI rows) ──
    for i, (val, unit, tk) in enumerate(zip(values, units, task_keys)):
        if tk and tk in second_best:
            sb_co2 = second_best[tk][1]
            label = (f'{sb_co2 / 1000:.1f} kg CO₂ eq/{unit}' if sb_co2 >= 1000
                     else f'{sb_co2:.1f} g CO₂ eq/{unit}')
            outer = max(val, sb_co2)
            ax.text(outer * 1.4, i, label,
                    va='center', fontsize=8.5 * FS, fontweight='bold', color='#1565C0')
        else:
            label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                     else f'{val:.1f} g CO₂ eq/{unit}')
            ax.text(val * 1.4, i, label, va='center', fontsize=8.5 * FS)

    # ── Y-axis labels (only second-best model name for AI rows) ──
    ax.set_yticks(range(len(ref_data)))
    ax.set_yticklabels([''] * len(ref_data))
    for i, (main, desc, tk) in enumerate(zip(main_labels, descs, task_keys)):
        ax.text(-0.03, i, main, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=10.5 * FS, fontweight='bold', color='black')
        if tk and tk in second_best:
            sb_name = second_best[tk][0]
            ax.text(-0.035, i - 0.42, sb_name, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8.5 * FS, fontweight='bold',
                    fontstyle='italic', color='#1565C0')
        else:
            ax.text(-0.035, i - 0.42, desc, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8.5 * FS, fontstyle='italic', color='#777777')

    ax.set_xscale('log')
    ax.set_xlabel('CO₂ Emission', fontsize=14 * FS)
    ax.tick_params(labelsize=12 * FS)
    ax.set_xlim(0.5, 5e8)

    # ── Legend ──
    leg = [
        mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                       linewidth=1.5, hatch='///',
                       label='AI models (2nd-best Pareto)'),
        mpatches.Patch(facecolor=ai_color_light, edgecolor='#D6EAFF',
                       label='AI models (worst Pareto)'),
        mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
        mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
    ]
    ax.legend(handles=leg, loc='lower right', fontsize=10 * FS,
              framealpha=0.9, title='Category', title_fontsize=11 * FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.36)
    out = os.path.join(OUT_DIR, '4b_co2_reference_second_pareto.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 4b saved → {out}")


# ── Figure 4 combined: (a) worst Pareto on top, (b) 2nd-best Pareto on bottom ─
def plot_fig4_combined(df):
    """Two-panel stacked Fig 4: (a) worst-Pareto bars, (b) worst + 2nd-best overlay.
    Shares the y-axis (same reference categories in same order)."""

    # ── Compute worst Pareto per task (same as fig4 / fig4b) ──
    task_map = {
        'Forward': 'Forward', 'Retro': 'Retro', 'MolGen': 'MolGen',
        'MatGen': 'MatGen', 'Folding': 'Folding', 'MDSim': 'MDSim',
    }
    worst_pareto = {}
    for task, key in task_map.items():
        grp = df[df['task'] == task].copy()
        if grp.empty:
            continue
        pareto_models = _compute_pareto(grp)
        pareto_grp = grp[grp['model'].isin(pareto_models)]
        worst = pareto_grp.loc[pareto_grp['CO2_per_job'].idxmax()]
        # Protein folding runs one protein at a time → report per-protein CO2
        worst_pareto[key] = (worst['model'], worst['CO2_per_exp' if key == 'Folding' else 'CO2_per_job'])

    # ── Second-best models (mirror fig4b) ──
    second_best_names = {
        'Forward': 'LocalTransform', 'Retro': 'LocalRetro',
        'MolGen': 'REINVENT4', 'MatGen': 'DiffCSP',
        'Folding': 'OpenFold', 'MDSim': 'SevenNet',
    }
    second_best = {}
    for task, model_name in second_best_names.items():
        row = df[(df['task'] == task) & (df['model'] == model_name)]
        if not row.empty:
            second_best[task] = (model_name, row.iloc[0]['CO2_per_exp' if task == 'Folding' else 'CO2_per_job'])

    # ── Resolve ref_data (carrying task_key for fig4b overlay) ──
    ref_data = []
    for (cat, label, task_key, fallback_desc, fallback_co2, unit) in ref_data_raw:
        if task_key and task_key in worst_pareto:
            model_name, co2 = worst_pareto[task_key]
            ref_data.append((cat, label, task_key, model_name, co2, unit))
        else:
            ref_data.append((cat, label, None, fallback_desc, fallback_co2, unit))
    ref_data.sort(key=lambda x: x[4])

    categories  = [d[0] for d in ref_data]
    main_labels = [d[1] for d in ref_data]
    task_keys   = [d[2] for d in ref_data]
    descs       = [d[3] for d in ref_data]
    values      = [d[4] for d in ref_data]
    units       = [d[5] for d in ref_data]

    AI_CATEGORIES = {'AI Chemical generation', 'AI Synthesis prediction',
                     'AI Protein folding', 'AI MD simulation'}
    CHEM_CATEGORIES = {'Chemical simulation', 'Chemical synthesis'}

    # ── Palette ──
    ai_color       = '#1E88E5'
    ai_color_light = '#D6EAFF'
    chem_color     = '#FFAB91'
    base_color     = '#CFD8DC'

    def _pick_top(cat):  # fig4 highlight_ai=True style
        if cat in AI_CATEGORIES:
            return ai_color, '#1565C0', 1.5, 0.72, '///'
        if cat in CHEM_CATEGORIES:
            return chem_color, '#E0E0E0', 0.5, 0.58, ''
        return base_color, '#E0E0E0', 0.5, 0.55, ''

    def _pick_light(cat):  # fig4b light pass
        if cat in AI_CATEGORIES:
            return ai_color_light, '#D6EAFF', 0.5, ''
        if cat in CHEM_CATEGORIES:
            return chem_color, '#E0E0E0', 0.5, ''
        return base_color, '#E0E0E0', 0.5, ''

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(12, 16), sharey=True)

    # ── Panel (a): worst-Pareto bars (fig4 style) ──
    for i, (val, cat) in enumerate(zip(values, categories)):
        fc, ec, lw, h, hp = _pick_top(cat)
        ax_a.barh(i, val, color=fc, edgecolor=ec, linewidth=lw, height=h, hatch=hp)
    for i, (val, unit) in enumerate(zip(values, units)):
        label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                 else f'{val:.1f} g CO₂ eq/{unit}')
        ax_a.text(val * 1.4, i, label, va='center', fontsize=8.5)

    # ── Panel (b): light worst-Pareto + bold 2nd-best overlay (fig4b style) ──
    bar_height = 0.72
    for i, (val, cat) in enumerate(zip(values, categories)):
        fc, ec, lw, hp = _pick_light(cat)
        h = bar_height if cat in AI_CATEGORIES else 0.58 if cat in CHEM_CATEGORIES else 0.55
        ax_b.barh(i, val, color=fc, edgecolor=ec, linewidth=lw, height=h, hatch=hp)
    for i, (cat, tk) in enumerate(zip(categories, task_keys)):
        if tk and tk in second_best:
            sb_co2 = second_best[tk][1]
            ax_b.barh(i, sb_co2, color=ai_color, edgecolor='#1565C0',
                      linewidth=1.5, height=bar_height, hatch='///')
    for i, (val, unit, tk) in enumerate(zip(values, units, task_keys)):
        if tk and tk in second_best:
            sb_co2 = second_best[tk][1]
            label = (f'{sb_co2 / 1000:.1f} kg CO₂ eq/{unit}' if sb_co2 >= 1000
                     else f'{sb_co2:.1f} g CO₂ eq/{unit}')
            outer = max(val, sb_co2)
            ax_b.text(outer * 1.4, i, label,
                      va='center', fontsize=8.5, fontweight='bold', color='#1565C0')
        else:
            label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                     else f'{val:.1f} g CO₂ eq/{unit}')
            ax_b.text(val * 1.4, i, label, va='center', fontsize=8.5)

    # ── Shared y-axis labels (only on left side, drawn once via ax_a) ──
    for ax in (ax_a, ax_b):
        ax.set_yticks(range(len(ref_data)))
        ax.set_yticklabels([''] * len(ref_data))
        ax.set_xscale('log')
        ax.set_xlim(0.5, 5e8)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel (a) y-labels: main label + worst-Pareto model name (italic gray)
    for i, (main, desc) in enumerate(zip(main_labels, descs)):
        ax_a.text(-0.03, i, main, transform=ax_a.get_yaxis_transform(),
                  ha='right', va='center', fontsize=10.5, fontweight='bold', color='black')
        ax_a.text(-0.035, i - 0.42, desc, transform=ax_a.get_yaxis_transform(),
                  ha='right', va='center', fontsize=8.5, fontstyle='italic', color='#777777')
    # Panel (b) y-labels: main label + 2nd-best model (blue) where applicable
    for i, (main, desc, tk) in enumerate(zip(main_labels, descs, task_keys)):
        ax_b.text(-0.03, i, main, transform=ax_b.get_yaxis_transform(),
                  ha='right', va='center', fontsize=10.5, fontweight='bold', color='black')
        if tk and tk in second_best:
            sb_name = second_best[tk][0]
            ax_b.text(-0.035, i - 0.42, sb_name, transform=ax_b.get_yaxis_transform(),
                      ha='right', va='center', fontsize=8.5, fontweight='bold',
                      fontstyle='italic', color='#1565C0')
        else:
            ax_b.text(-0.035, i - 0.42, desc, transform=ax_b.get_yaxis_transform(),
                      ha='right', va='center', fontsize=8.5, fontstyle='italic', color='#777777')

    ax_b.set_xlabel('CO₂ Emission', fontsize=14)

    # ── Panel labels (a) / (b) at top-left ──
    ax_a.text(-0.30, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=18, fontweight='bold', va='bottom', ha='left')
    ax_b.text(-0.30, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=18, fontweight='bold', va='bottom', ha='left')

    # ── Legends ──
    leg_a = [
        mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                       linewidth=1.5, hatch='///', label='AI models (this work)'),
        mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
        mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
    ]
    ax_a.legend(handles=leg_a, loc='lower right', fontsize=10,
                framealpha=0.9, title='Category', title_fontsize=11)

    leg_b = [
        mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                       linewidth=1.5, hatch='///', label='AI models (2nd-best Pareto)'),
        mpatches.Patch(facecolor=ai_color_light, edgecolor='#D6EAFF',
                       label='AI models (worst Pareto)'),
        mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
        mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
    ]
    ax_b.legend(handles=leg_b, loc='lower right', fontsize=10,
                framealpha=0.9, title='Category', title_fontsize=11)

    plt.subplots_adjust(left=0.32, hspace=0.18)
    out = os.path.join(OUT_DIR, '4ab_co2_reference_combined.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 4ab saved → {out}")


# ── Figure 4 horizontal: (a) worst Pareto on left, (b) 2nd-best Pareto on right ─
def plot_fig4_combined_horizontal(df):
    """Side-by-side Fig 4: (a) worst-Pareto bars (left), (b) worst + 2nd-best overlay (right).
    Each panel has its own y-axis labels since panel (b) shows 2nd-best model names."""

    # ── Compute worst Pareto per task (same as fig4 / fig4b) ──
    task_map = {
        'Forward': 'Forward', 'Retro': 'Retro', 'MolGen': 'MolGen',
        'MatGen': 'MatGen', 'Folding': 'Folding', 'MDSim': 'MDSim',
    }
    worst_pareto = {}
    for task, key in task_map.items():
        grp = df[df['task'] == task].copy()
        if grp.empty:
            continue
        pareto_models = _compute_pareto(grp)
        pareto_grp = grp[grp['model'].isin(pareto_models)]
        worst = pareto_grp.loc[pareto_grp['CO2_per_job'].idxmax()]
        # Protein folding runs one protein at a time → report per-protein CO2
        worst_pareto[key] = (worst['model'], worst['CO2_per_exp' if key == 'Folding' else 'CO2_per_job'])

    # ── Second-best models (mirror fig4b) ──
    second_best_names = {
        'Forward': 'LocalTransform', 'Retro': 'LocalRetro',
        'MolGen': 'REINVENT4', 'MatGen': 'DiffCSP',
        'Folding': 'OpenFold', 'MDSim': 'SevenNet',
    }
    second_best = {}
    for task, model_name in second_best_names.items():
        row = df[(df['task'] == task) & (df['model'] == model_name)]
        if not row.empty:
            second_best[task] = (model_name, row.iloc[0]['CO2_per_exp' if task == 'Folding' else 'CO2_per_job'])

    # ── Resolve ref_data (carrying task_key for fig4b overlay) ──
    ref_data = []
    for (cat, label, task_key, fallback_desc, fallback_co2, unit) in ref_data_raw:
        if task_key and task_key in worst_pareto:
            model_name, co2 = worst_pareto[task_key]
            ref_data.append((cat, label, task_key, model_name, co2, unit))
        else:
            ref_data.append((cat, label, None, fallback_desc, fallback_co2, unit))
    ref_data.sort(key=lambda x: x[4])

    categories  = [d[0] for d in ref_data]
    main_labels = [d[1] for d in ref_data]
    task_keys   = [d[2] for d in ref_data]
    descs       = [d[3] for d in ref_data]
    values      = [d[4] for d in ref_data]
    units       = [d[5] for d in ref_data]

    AI_CATEGORIES = {'AI Chemical generation', 'AI Synthesis prediction',
                     'AI Protein folding', 'AI MD simulation'}
    CHEM_CATEGORIES = {'Chemical simulation', 'Chemical synthesis'}

    # ── Palette ──
    ai_color       = '#1E88E5'
    ai_color_light = '#D6EAFF'
    chem_color     = '#FFAB91'
    base_color     = '#CFD8DC'

    def _pick_top(cat):  # fig4 highlight_ai=True style
        if cat in AI_CATEGORIES:
            return ai_color, '#1565C0', 1.5, 0.72, '///'
        if cat in CHEM_CATEGORIES:
            return chem_color, '#E0E0E0', 0.5, 0.58, ''
        return base_color, '#E0E0E0', 0.5, 0.55, ''

    def _pick_light(cat):  # fig4b light pass
        if cat in AI_CATEGORIES:
            return ai_color_light, '#D6EAFF', 0.5, ''
        if cat in CHEM_CATEGORIES:
            return chem_color, '#E0E0E0', 0.5, ''
        return base_color, '#E0E0E0', 0.5, ''

    # Side-by-side; do NOT share y-axis since each panel has its own descriptive labels
    FS = 2.0  # global font scale-up (match other figures); wide canvas so the
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(38, 18))  # in-panel legend has room

    # ── Panel (a): worst-Pareto bars (fig4 style) ──
    for i, (val, cat) in enumerate(zip(values, categories)):
        fc, ec, lw, h, hp = _pick_top(cat)
        ax_a.barh(i, val, color=fc, edgecolor=ec, linewidth=lw, height=h, hatch=hp)
    for i, (val, unit) in enumerate(zip(values, units)):
        label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                 else f'{val:.1f} g CO₂ eq/{unit}')
        ax_a.text(val * 1.4, i, label, va='center', fontsize=8.5 * FS)

    # ── Panel (b): light worst-Pareto + bold 2nd-best overlay (fig4b style) ──
    bar_height = 0.72
    for i, (val, cat) in enumerate(zip(values, categories)):
        fc, ec, lw, hp = _pick_light(cat)
        h = bar_height if cat in AI_CATEGORIES else 0.58 if cat in CHEM_CATEGORIES else 0.55
        ax_b.barh(i, val, color=fc, edgecolor=ec, linewidth=lw, height=h, hatch=hp)
    for i, (cat, tk) in enumerate(zip(categories, task_keys)):
        if tk and tk in second_best:
            sb_co2 = second_best[tk][1]
            ax_b.barh(i, sb_co2, color=ai_color, edgecolor='#1565C0',
                      linewidth=1.5, height=bar_height, hatch='///')
    for i, (val, unit, tk) in enumerate(zip(values, units, task_keys)):
        if tk and tk in second_best:
            sb_co2 = second_best[tk][1]
            label = (f'{sb_co2 / 1000:.1f} kg CO₂ eq/{unit}' if sb_co2 >= 1000
                     else f'{sb_co2:.1f} g CO₂ eq/{unit}')
            outer = max(val, sb_co2)
            ax_b.text(outer * 1.4, i, label,
                      va='center', fontsize=8.5 * FS, fontweight='bold', color='#1565C0')
        else:
            label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                     else f'{val:.1f} g CO₂ eq/{unit}')
            ax_b.text(val * 1.4, i, label, va='center', fontsize=8.5 * FS)

    # ── Shared axis settings (each panel has its own y-labels) ──
    for ax in (ax_a, ax_b):
        ax.set_yticks(range(len(ref_data)))
        ax.set_yticklabels([''] * len(ref_data))
        ax.set_xscale('log')
        ax.set_xlim(0.5, 5e8)
        ax.tick_params(labelsize=12 * FS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('CO₂ Emission', fontsize=14 * FS)

    # Panel (a) y-labels: main label + worst-Pareto model name (italic gray)
    for i, (main, desc) in enumerate(zip(main_labels, descs)):
        ax_a.text(-0.03, i, main, transform=ax_a.get_yaxis_transform(),
                  ha='right', va='center', fontsize=10.5 * FS, fontweight='bold', color='black')
        ax_a.text(-0.035, i - 0.42, desc, transform=ax_a.get_yaxis_transform(),
                  ha='right', va='center', fontsize=8.5 * FS, fontstyle='italic', color='#777777')
    # Panel (b) y-labels: main label + 2nd-best model (blue) where applicable
    for i, (main, desc, tk) in enumerate(zip(main_labels, descs, task_keys)):
        ax_b.text(-0.03, i, main, transform=ax_b.get_yaxis_transform(),
                  ha='right', va='center', fontsize=10.5 * FS, fontweight='bold', color='black')
        if tk and tk in second_best:
            sb_name = second_best[tk][0]
            ax_b.text(-0.035, i - 0.42, sb_name, transform=ax_b.get_yaxis_transform(),
                      ha='right', va='center', fontsize=8.5 * FS, fontweight='bold',
                      fontstyle='italic', color='#1565C0')
        else:
            ax_b.text(-0.035, i - 0.42, desc, transform=ax_b.get_yaxis_transform(),
                      ha='right', va='center', fontsize=8.5 * FS, fontstyle='italic', color='#777777')

    # ── Panel labels (a) / (b) at top-left ──
    ax_a.text(-0.30, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=18 * FS, fontweight='bold', va='bottom', ha='left')
    ax_b.text(-0.30, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=18 * FS, fontweight='bold', va='bottom', ha='left')

    # ── Legends ──
    leg_a = [
        mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                       linewidth=1.5, hatch='///', label='AI models (this work)'),
        mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
        mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
    ]
    ax_a.legend(handles=leg_a, loc='lower right', fontsize=10 * FS,
                framealpha=0.9, title='Category', title_fontsize=11 * FS)

    leg_b = [
        mpatches.Patch(facecolor=ai_color, edgecolor='#1565C0',
                       linewidth=1.5, hatch='///', label='AI models (2nd-best Pareto)'),
        mpatches.Patch(facecolor=ai_color_light, edgecolor='#D6EAFF',
                       label='AI models (worst Pareto)'),
        mpatches.Patch(facecolor=chem_color, label='Conventional chemistry'),
        mpatches.Patch(facecolor=base_color, label='Everyday activities / LLM'),
    ]
    ax_b.legend(handles=leg_b, loc='lower right', fontsize=10 * FS,
                framealpha=0.9, title='Category', title_fontsize=11 * FS)

    # Generous left/inter-panel space so the descriptive labels for each panel fit
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.62)
    out = os.path.join(OUT_DIR, '4ab_co2_reference_combined_horizontal.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 4ab horizontal saved → {out}")


# ── Figure 5: Cross-task CO2 decomposition (2 panels) ─────────────────────
def plot_fig5(df):
    """Two panels: log10(inference time) vs log10(CO2) and log10(model size) vs log10(CO2),
    all tasks combined with regression line and R²."""
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 11))

    for panel_ax, xcol, xlabel in [
        (ax_left,  'inference_time_per_exp', 'log₁₀(Inference Time/exp)'),
        (ax_right, '_size_num',              'log₁₀(Model Size)'),
    ]:
        all_logx, all_logy = [], []
        for task, grp in df.groupby('task'):
            if task == 'StructOpt':
                continue  # StructOpt and MDSim share models; show only MDSim here
            c = TASK_COLORS[task]
            for _, row in grp.iterrows():
                xv, yv = row[xcol], row['CO2_per_exp']
                if pd.isna(xv) or pd.isna(yv) or xv <= 0 or yv <= 0:
                    continue
                lx, ly = np.log10(xv), np.log10(yv)
                all_logx.append(lx)
                all_logy.append(ly)
                m = ARCH_MARKERS.get(row['model type'], 'o')
                panel_ax.scatter(lx, ly, color=c, marker=m, s=marker_size(m),
                                 edgecolors='white', linewidths=0.6, zorder=3)

        all_logx = np.array(all_logx)
        all_logy = np.array(all_logy)
        if len(all_logx) > 1:
            coef = np.polyfit(all_logx, all_logy, 1)
            rho, pval = _spearman_p(all_logx, all_logy)
            xfit = np.linspace(all_logx.min(), all_logx.max(), 50)
            panel_ax.plot(xfit, np.polyval(coef, xfit), 'r--', lw=1.5, alpha=0.7, zorder=2)
            if rho is not None:
                panel_ax.text(0.05, 0.95, _rho_p_label(rho, pval, signed=True),
                              transform=panel_ax.transAxes,
                              fontsize=18 * FS, va='top', ha='left',
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        panel_ax.set_xlabel(xlabel, fontsize=20 * FS)
        panel_ax.set_ylabel('log₁₀(CO₂/exp)', fontsize=20 * FS)
        panel_ax.tick_params(labelsize=16 * FS)
        panel_ax.spines['top'].set_visible(False)
        panel_ax.spines['right'].set_visible(False)
        panel_ax.grid(True, alpha=0.2)

    task_handles = [mpatches.Patch(color=c, label=_disp(t))
                    for t, c in TASK_COLORS.items() if t != 'StructOpt']
    arch_handles = get_arch_legend_handles()
    # Symmetric two-row legend: task colors on top, architecture markers below.
    # Pad the shorter row with invisible spacers so both rows span the same number
    # of columns → a clean rectangle.
    ncol = len(task_handles)
    def _spacer():
        return mlines.Line2D([], [], linestyle='none', marker='', label='')
    pad = ncol - len(arch_handles)  # 2
    left = pad // 2
    right = pad - left
    arch_row = [_spacer()] * left + arch_handles + [_spacer()] * right
    fig.legend(handles=task_handles + arch_row,
               loc='lower center', ncol=ncol, fontsize=15 * FS, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.24, wspace=0.3)
    out = os.path.join(OUT_DIR, '5_co2_decomposition_cross_task.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 5 saved → {out}")


# ── Figure 6: Pareto with alternative metrics ────────────────────────────
# Alternative metrics: MatGen(SUN), MolGen(SUN), Retro(Top-10), Forward(Top-1)
ALT_METRICS = {
    'MatGen':  {'CDVAE': 3.2, 'DiffCSP': 4.3, 'CrystaLLM': 3.5, 'FlowMM': 4.3,
                'ChargeDIFF': 4.4, 'MatterGen': 5.2, 'ADiT': 5.5, 'CrystalFlow': 3.0},
    'Retro':   {'neuralsym': 72.8, 'MEGAN': 87.0, 'LocalRetro': 91.5, 'RSMILES': 89.6,
                'Chemformer': 62.8, 'LlaSMol': 5.0, 'RetroBridge': 44.9, 'RSGPT': 96.6},
    'Forward': {'neuralsym': 49.5, 'MEGAN': 80.1, 'Graph2SMILES': 88.5, 'Chemformer': 89.0,
                'LocalTransform': 90.4, 'MolecularTransformer': 86.8, 'RSMILES': 89.4, 'LlaSMol': 3.8},
}
# MolGen VUNS is already in the pretrained data loaded in load_data()
MOLGEN_VUNS = {
    'REINVENT': 80.88, 'JT-VAE': 89.41, 'HierVAE': 88.89, 'MolGPT': 76.65,
    'DiGress': 81.18, 'REINVENT4': 85.44, 'SmileyLlama': 85.16, 'DeFoG': 81.73
}

# StructOpt is the MLIP-family alternative to MDSim (its CPS metric); Folding's
# alternative metric is lDDT-Cα (vs. its primary GDT-TS (%) in Figs 1-3).
ALT_TASK_ORDER = ['MatGen', 'MolGen', 'Retro', 'Forward', 'StructOpt', 'Folding']
ALT_LABELS = {
    'MatGen': 'SUN (%)', 'MolGen': 'VUNS (%)',
    'Retro': 'Top-10 Acc (%)', 'Forward': 'Top-1 Acc (%)',
    'StructOpt': 'CPS', 'Folding': 'lDDT-Cα',
}


def plot_fig6(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False, clean=False):
    """Pareto plots using alternative metrics, laid out like Fig 2 (2×3).

    Includes the MLIP-family StructOpt panel (CPS) and the Folding panel
    (TM-score). If noname=True, omits per-point labels EXCEPT for models on the
    Pareto front (plus the baseline), and saves to *_noname.png.
    If clean=True, omits ALL model labels and enlarges the markers (architecture
    still encoded as shape — a template to annotate by hand); saves to *_clean.png.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    fig, axes = plt.subplots(2, 3, figsize=(36, 20))  # large so labels don't overlap (match Fig 2)
    axes = axes.flatten()

    # Add alt_perf column
    df = df.copy()
    df['alt_perf'] = np.nan
    for idx, row in df.iterrows():
        task, model = row['task'], row['model']
        if task == 'MolGen' and model in MOLGEN_VUNS:
            df.at[idx, 'alt_perf'] = MOLGEN_VUNS[model]
        elif task in ALT_METRICS and model in ALT_METRICS[task]:
            df.at[idx, 'alt_perf'] = ALT_METRICS[task][model]
        elif task == 'StructOpt':
            df.at[idx, 'alt_perf'] = row['major_metric']   # CPS
        elif task == 'Folding':
            df.at[idx, 'alt_perf'] = row['minor_metric']   # lDDT-Cα

    for ax, task in zip(axes, ALT_TASK_ORDER):
        grp = df[(df['task'] == task) & df['alt_perf'].notna()].copy()
        base_row = grp[grp['baseline?'] == True].iloc[0]
        base_co2 = base_row[co2_col]
        base_alt = base_row['alt_perf']

        grp['log_co2_ratio'] = np.log10(grp[co2_col] / base_co2)
        grp['delta_alt_pct'] = (grp['alt_perf'] - base_alt) / abs(base_alt) * 100

        # Pareto front
        grp_sorted = grp.sort_values('log_co2_ratio')
        pareto_x, pareto_y = [], []
        pareto_models = set()
        best = -np.inf
        for _, row in grp_sorted.iterrows():
            if row['delta_alt_pct'] >= best:
                pareto_x.append(row['log_co2_ratio'])
                pareto_y.append(row['delta_alt_pct'])
                pareto_models.add(row['model'])
                best = row['delta_alt_pct']

        # Axis limits
        all_x, all_y = grp['log_co2_ratio'], grp['delta_alt_pct']
        xpad = (all_x.max() - all_x.min()) * 0.2 + 0.3
        xmin, xmax = all_x.min() - xpad, all_x.max() + xpad
        ypad = max(abs(all_y.min()), abs(all_y.max())) * 0.25
        ymin, ymax = all_y.min() - ypad, all_y.max() + ypad

        # Quadrants
        ax.fill_between([xmin, 0], [0, 0], [ymax, ymax], color='#d4edda', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [0, 0], [ymax, ymax], color='#fff3cd', alpha=0.4, zorder=0)
        ax.fill_between([0, xmax], [ymin, ymin], [0, 0], color='#f8d7da', alpha=0.4, zorder=0)
        ax.fill_between([xmin, 0], [ymin, ymin], [0, 0], color='#e2e3e5', alpha=0.4, zorder=0)
        ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        ax.axvline(0, color='black', linewidth=0.8, zorder=1)

        # Pareto step
        if len(pareto_x) > 1:
            sx, sy = [pareto_x[0]], [pareto_y[0]]
            for k in range(1, len(pareto_x)):
                sx.extend([pareto_x[k], pareto_x[k]])
                sy.extend([pareto_y[k - 1], pareto_y[k]])
            ax.plot(sx, sy, 'k--', lw=1.5, alpha=0.6, zorder=2)

        # Points
        texts6 = []
        for _, row in grp.iterrows():
            # architecture encoded as marker shape (also in clean mode)
            m = ARCH_MARKERS.get(row['model type'], 'o')
            is_base = row.get('baseline?', False)
            ec = 'gray'
            lw = 0.8
            sz = marker_size(m, base=440 if clean else 150)  # bigger points in clean mode
            ax.scatter(row['log_co2_ratio'], row['delta_alt_pct'],
                       color=TASK_COLORS[task], marker=m, s=sz,
                       edgecolors=ec, linewidths=lw, zorder=4)
            if clean:
                continue  # clean template: markers only, no labels (add by hand)
            # In noname mode, only label models on the Pareto front (plus the baseline)
            if noname and row['model'] not in pareto_models and not is_base:
                continue
            label = f"{row['model']} ({row['year']})"
            tx6, ty6 = row['log_co2_ratio'], row['delta_alt_pct']
            if task == 'Forward' and row['model'] == 'Graph2SMILES':
                ty6 += 5.0
            elif task == 'Forward' and row['model'] == 'LocalTransform':
                ty6 -= 5.0
            label_fs = 14 * FS  # match Fig 1/2 label size (28pt)
            texts6.append(ax.text(tx6, ty6, label,
                                  fontsize=label_fs, zorder=5))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('log₁₀(CO₂ eq ratio)', fontsize=20 * FS)
        ax.set_ylabel(f'Δ Relative {ALT_LABELS[task]}', fontsize=20 * FS)
        ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=TASK_COLORS[task], pad=18)
        ax.tick_params(labelsize=16 * FS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Spearman correlation (match Fig 2): rank-based, robust to outliers. Omitted in clean mode.
        rho, pval = _spearman_p(grp['log_co2_ratio'], grp['delta_alt_pct'])
        if rho is not None and not clean:
            ax.text(0.96, 0.05, _rho_p_label(rho, pval), transform=ax.transAxes,
                    fontsize=16 * FS, va='bottom', ha='right', zorder=6,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        if texts6:
            adjust_text(texts6, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                        force_points=(2.0, 2.0), iterations=200,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    arch_handles = get_arch_legend_handles()
    quad_handles = [
        mpatches.Patch(color='#d4edda', alpha=0.7, label='Dominant'),
        mpatches.Patch(color='#fff3cd', alpha=0.7, label='Tradeoff'),
        mpatches.Patch(color='#f8d7da', alpha=0.7, label='Dominated'),
        mpatches.Patch(color='#e2e3e5', alpha=0.7, label='Inverse'),
    ]
    legend_handles = arch_handles + quad_handles
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=5, fontsize=16 * FS, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.14, hspace=0.45, wspace=0.32)
    fname = ('6_pareto_alt_metrics_clean.png' if clean else
             '6_pareto_alt_metrics_noname.png' if noname else '6_pareto_alt_metrics.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    tag = ' (clean, no labels)' if clean else (' (noname, Pareto-only labels)' if noname else '')
    print(f"Fig 6{tag} saved → {out}")


def _alt_perf_column(df):
    """Return a copy of df with an 'alt_perf' column (the alternative metric per
    task, same mapping used by Fig 6)."""
    df = df.copy()
    df['alt_perf'] = np.nan
    for idx, row in df.iterrows():
        task, model = row['task'], row['model']
        if task == 'MolGen' and model in MOLGEN_VUNS:
            df.at[idx, 'alt_perf'] = MOLGEN_VUNS[model]
        elif task in ALT_METRICS and model in ALT_METRICS[task]:
            df.at[idx, 'alt_perf'] = ALT_METRICS[task][model]
        elif task == 'StructOpt':
            df.at[idx, 'alt_perf'] = row['major_metric']   # CPS
        elif task == 'Folding':
            df.at[idx, 'alt_perf'] = row['minor_metric']   # lDDT-Cα
    return df


def plot_fig1_alt(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False):
    """Alternative-metric counterpart of Figure 1 (Fig 6 is to Fig 2 as this is to Fig 1).

    6 rows (ALT_TASK_ORDER) × 3 cols (model size, CO2, alternative performance).
    The size and CO2 columns are identical in spirit to Fig 1; only the performance
    column swaps the primary metric for each task's alternative metric (SUN, VUNS,
    Top-10, Top-1, CPS, lDDT-Cα). If noname=True, omits per-point labels.
    """
    FS = 2.0  # global font scale-up (match Figure 1)
    SHOW_ARCH = False  # single marker, no architecture legend (match Fig 1)
    fig, axes = plt.subplots(6, 3, figsize=(20, 32))

    df = _alt_perf_column(df)

    all_years = df['year'].dropna().unique()
    year_min, year_max = int(all_years.min()), int(all_years.max())
    year_ticks = list(range(year_min, year_max + 1, 2))
    x_pad = 0.5
    xlim = (year_min - x_pad, year_max + x_pad)

    SHARED_YLABELS = {0: 'log₁₀(Model Size)', 1: co2_label, 2: 'Performance'}

    for i, task in enumerate(ALT_TASK_ORDER):
        grp = df[(df['task'] == task) & df['alt_perf'].notna()]
        c = TASK_COLORS[task]
        # Percentage alt-metrics (label contains '%') are shown as fractions on 0-1.
        perf_is_pct = '%' in ALT_LABELS[task]
        col_specs = [
            ('year', '_size_num',  'log₁₀(Model Size)', True),
            ('year', co2_col,      co2_label,           True),
            ('year', 'alt_perf',   'Performance',       False),
        ]
        for j, (xcol, ycol, ylabel, use_log10) in enumerate(col_specs):
            ax = axes[i, j]
            texts = []
            for _, row in grp.iterrows():
                yval = row[ycol]
                if pd.isna(yval) or yval <= 0:
                    continue
                if use_log10:
                    yval = np.log10(yval)
                elif j == 2 and perf_is_pct:
                    yval = yval / 100.0  # percentage → fraction (0-1)
                m = ARCH_MARKERS.get(row['model type'], 'o') if SHOW_ARCH else 'o'
                ax.scatter(row['year'], yval, color=c, marker=m, s=marker_size(m),
                           edgecolors='white', linewidths=0.6, zorder=3)
                if not noname:
                    texts.append(ax.text(row['year'], yval, row['model'],
                                         fontsize=14 * FS, zorder=5))
            # Spearman year-trend annotation (top-left corner)
            rho, pval = _year_spearman_p(grp, ycol)
            if rho is not None:
                ax.text(0.05, 0.95, _rho_p_label(rho, pval), transform=ax.transAxes,
                        fontsize=16 * FS, va='top', zorder=6,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_xlim(xlim)
            ax.set_xticks(year_ticks)
            ax.tick_params(labelsize=16 * FS)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            if i < 5:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Year', fontsize=20 * FS)
            if j == 0:
                ax.text(-0.58, 0.5, _disp(task), transform=ax.transAxes, fontsize=22 * FS,
                        fontweight='bold', color=c, va='center', ha='center', rotation=90)
            if texts:
                adjust_text(texts, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.06, hspace=0.35, wspace=0.4)

    for j, lbl in SHARED_YLABELS.items():
        top = axes[0, j].get_position().y1
        bot = axes[5, j].get_position().y0
        x0 = axes[0, j].get_position().x0
        fig.text(x0 - 0.082, (top + bot) / 2, lbl, rotation=90,
                 va='center', ha='center', fontsize=20 * FS)

    fname = ('1_year_trends_alt_combined_noname.png' if noname
             else '1_year_trends_alt_combined.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 1 alt{' (noname)' if noname else ''} saved → {out}")


def plot_fig1_alt_horizontal(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)', noname=False):
    """Horizontal alternative-metric Fig 1: 3 rows (size, CO2, alt performance) × 6 cols
    (ALT_TASK_ORDER). Mirrors plot_fig1_horizontal but the performance row uses each
    task's alternative metric (SUN, VUNS, Top-10, Top-1, CPS, lDDT-Cα)."""
    FS = 2.0  # global font scale-up (match Fig 1 horizontal)
    SHOW_ARCH = False  # single marker, no architecture legend (match Fig 1)
    fig, axes = plt.subplots(3, 6, figsize=(42, 18))

    df = _alt_perf_column(df)

    all_years = df['year'].dropna().unique()
    year_min, year_max = int(all_years.min()), int(all_years.max())
    year_ticks = list(range(year_min, year_max + 1, 2))
    x_pad = 0.5
    xlim = (year_min - x_pad, year_max + x_pad)

    row_specs = [
        ('_size_num', 'log₁₀(Model Size)', True),
        (co2_col,     co2_label,           True),
        ('alt_perf',  'Performance',       False),  # 0-1 float scale
    ]

    for j, task in enumerate(ALT_TASK_ORDER):
        grp = df[(df['task'] == task) & df['alt_perf'].notna()]
        c = TASK_COLORS[task]
        # Percentage alt-metrics (label contains '%') are shown as fractions on 0-1.
        perf_is_pct = '%' in ALT_LABELS[task]
        for i, (ycol, ylabel, use_log10) in enumerate(row_specs):
            ax = axes[i, j]
            texts = []
            pxs, pys = [], []  # plotted points, for badge-corner placement
            for _, row in grp.iterrows():
                yval = row[ycol]
                if pd.isna(yval) or yval <= 0:
                    continue
                if use_log10:
                    yval = np.log10(yval)
                elif i == 2 and perf_is_pct:
                    yval = yval / 100.0  # percentage → fraction (0-1)
                m = ARCH_MARKERS.get(row['model type'], 'o') if SHOW_ARCH else 'o'
                ax.scatter(row['year'], yval, color=c, marker=m, s=marker_size(m),
                           edgecolors='white', linewidths=0.6, zorder=3)
                pxs.append(row['year'])
                pys.append(yval)
                if not noname:
                    texts.append(ax.text(row['year'], yval, row['model'],
                                         fontsize=14 * FS, zorder=5))
            # Spearman year-trend annotation, placed in the emptiest corner so it
            # doesn't cover points (x via fixed xlim, y via the panel's range).
            rho, pval = _year_spearman_p(grp, ycol)
            if rho is not None:
                if len(pys) >= 2 and max(pys) > min(pys):
                    yspan = max(pys) - min(pys)
                    ylo, yhi = min(pys) - 0.05 * yspan, max(pys) + 0.05 * yspan
                    xspan = xlim[1] - xlim[0]
                    pts = [((px - xlim[0]) / xspan, (py - ylo) / (yhi - ylo))
                           for px, py in zip(pxs, pys)]
                    bx, by, bha, bva = pick_badge_corner(pts, bh=0.22)
                else:
                    bx, by, bha, bva = 0.03, 0.97, 'left', 'top'
                ax.text(bx, by, _rho_p_label(rho, pval), transform=ax.transAxes,
                        fontsize=16 * FS, va=bva, ha=bha, zorder=6,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_xlim(xlim)
            ax.set_xticks(year_ticks)
            ax.tick_params(labelsize=16 * FS)
            if i == 1:
                ax.locator_params(axis='y', nbins=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            if i == 0:
                ax.set_title(_disp(task), fontsize=22 * FS, fontweight='bold', color=c, pad=20)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=20 * FS)
                ax.yaxis.set_label_coords(-0.30, 0.5)
            if i < 2:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Year', fontsize=20 * FS)
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(30)
                    lbl.set_ha('right')
                    lbl.set_rotation_mode('anchor')
            if texts:
                adjust_text(texts, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                            force_points=(2.0, 2.0), iterations=200,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.4)
    fname = ('1_year_trends_alt_combined_horizontal_noname.png' if noname
             else '1_year_trends_alt_combined_horizontal.png')
    out = os.path.join(OUT_DIR, fname)
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 1 alt horizontal{' (noname)' if noname else ''} saved → {out}")


# ── Figure 2b: Pareto-tip diminishing returns (companion to Fig 2) ────────────
# Best vs. second-best Pareto model per domain. Curated task-lead results (NOT
# recomputed from all_data.csv, which yields slightly different carbon ratios).
# Best vs. second-best Pareto model per domain — (domain, best, second-best).
# All three reported metrics are derived from all_data.csv (see pareto_tip_metrics),
# so the figure stays reproducible and in sync with the source data.
PARETO_TIP = [
    ('MatGen',  'ChargeDIFF',  'ADiT'),
    ('MolGen',  'SmileyLlama', 'REINVENT4'),
    ('Retro',   'RSGPT',       'LocalRetro'),
    ('Forward', 'RSMILES',     'LocalTransform'),
    ('MDSim',   'eSEN',        'SevenNet'),
    ('Folding', 'ColabFold',   'OpenFold'),
]


def pareto_tip_metrics(df):
    """Derive the best-vs-second-best Pareto metrics for each domain from raw data.

    Returns a list of dicts with:
        perf    = (best − second)/second × 100   relative performance gain (%)
        ratio   = best CO₂ / second CO₂          scaled carbon-cost ratio (×)
        penalty = ratio / perf                   carbon penalty ΔC/ΔM (×/%)
    The carbon ratio is identical whether CO₂ is measured per-exp or per-job
    (the per-job/per-exp factor is constant within a domain), so CO₂/exp is used.
    """
    out = []
    for domain, best, second in PARETO_TIP:
        rb = df[(df['task'] == domain) & (df['model'] == best)].iloc[0]
        rs = df[(df['task'] == domain) & (df['model'] == second)].iloc[0]
        perf = (rb['major_metric'] - rs['major_metric']) / rs['major_metric'] * 100
        ratio = rb['CO2_per_exp'] / rs['CO2_per_exp']
        out.append({'domain': domain, 'best': best, 'second': second,
                    'perf': perf, 'ratio': ratio, 'penalty': ratio / perf})
    return out


def plot_fig2b_pareto_tip(df=None):
    """Companion to Figure 2: the diminishing returns at each domain's Pareto tip.

    For every domain, compares the best-performing Pareto model to the
    second-best Pareto model with ONE labelled point:
        x = relative performance improvement (%)  (best over second-best)
        y = scaled carbon-cost ratio (×)          (best CO₂ / second-best CO₂)
    Both axes log-scaled. The third tabulated metric — carbon-cost ratio per 1%
    gain — equals y/x, so a constant penalty plots as a diagonal: the y=x line
    separates "worthwhile" (below: gain outruns extra carbon) from "diminishing
    returns" (above: carbon multiplies for a sliver of performance).

    Values are derived from all_data.csv via pareto_tip_metrics(df).
    """
    if df is None:
        df = load_data()
    TIP = pareto_tip_metrics(df)
    FS = 2.0
    fig, ax = plt.subplots(figsize=(11, 9))

    # Axis ranges (log) with headroom around the data.
    xlo, xhi = 0.07, 70
    ylo, yhi = 0.8, 600

    # Diagonal y=x = "1× carbon per 1% gain"; shade above (diminishing returns)
    # red and below (worthwhile) green.
    diag = np.array([max(xlo, ylo), min(xhi, yhi)])
    ax.fill_between([xlo, xhi], [xlo, xhi], yhi, color='#f8d7da', alpha=0.35, zorder=0)
    ax.fill_between([xlo, xhi], ylo, [xlo, xhi], color='#d4edda', alpha=0.35, zorder=0)
    ax.plot(diag, diag, color='gray', linestyle='--', linewidth=1.5, zorder=1,
            label='Equal trade-off (penalty = 1×/%)')
    # Carbon parity: best costs the same as second-best.
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.3, zorder=1,
               label='Carbon parity (1×)')

    texts = []
    for t in TIP:
        domain, best, second = t['domain'], t['best'], t['second']
        gain, ratio, penalty = t['perf'], t['ratio'], t['penalty']
        c = TASK_COLORS[domain]
        ax.scatter(gain, ratio, s=700, color=c, alpha=0.9,
                   edgecolors='white', linewidths=2.2, zorder=4)
        label = f"{_disp(domain)}  ·  {penalty:.2g}×/%\n{best} ▸ {second}"
        texts.append(ax.text(gain, ratio, label, fontsize=8.5 * FS,
                             ha='center', va='center', zorder=5, color=c,
                             fontweight='bold'))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel('Relative performance improvement (%)', fontsize=20 * FS)
    ax.set_ylabel('Scaled carbon-cost ratio (×)', fontsize=20 * FS)
    ax.set_title('Best vs. second-best Pareto model', fontsize=22 * FS,
                 fontweight='bold', pad=16)
    ax.tick_params(labelsize=15 * FS)
    ax.grid(True, alpha=0.25, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Region meanings go in the legend (center, in empty space) so no floating
    # text collides with the data labels.
    legend_handles = [
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5,
                      label='Equal trade-off (1×/%)'),
        mlines.Line2D([], [], color='black', linestyle=':', linewidth=1.3,
                      label='Carbon parity'),
        mpatches.Patch(color='#f8d7da', alpha=0.6, label='Diminishing returns'),
        mpatches.Patch(color='#d4edda', alpha=0.6, label='Worthwhile'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10 * FS,
              framealpha=0.95, borderpad=0.5, handlelength=1.4)

    adjust_text(texts, ax=ax, expand=(1.4, 1.8), force_text=(1.4, 1.6),
                iterations=150,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7, alpha=0.6))

    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.11)
    out = os.path.join(OUT_DIR, '2b_pareto_tip_comparison.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 2b (Pareto-tip comparison) saved → {out}")


def plot_fig2b_pareto_tip_bar(df=None):
    """Bar-chart companion to Figure 2 (full width, no title) for placing below it.

    Per domain, three grouped bars on a log axis: relative performance
    improvement (%) of the best Pareto model over the second-best, the scaled
    carbon-cost ratio, and the carbon penalty (ratio per 1% gain). The mismatch
    is the story — a sliver of a performance bar next to a towering carbon bar
    = diminishing returns. Values derived from all_data.csv via pareto_tip_metrics.
    """
    if df is None:
        df = load_data()
    TIP = pareto_tip_metrics(df)
    FS = 2.0
    domains = [t['domain'] for t in TIP]
    perf = np.array([t['perf'] for t in TIP])
    carbon = np.array([t['ratio'] for t in TIP])
    penalty = np.array([t['penalty'] for t in TIP])

    fig, ax = plt.subplots(figsize=(36, 9))  # match Figure 2 width (36)
    x = np.arange(len(domains))
    width = 0.27
    PERF_C, CO2_C, PEN_C = '#2a9d8f', '#e76f51', '#6a4c93'

    b1 = ax.bar(x - width, perf, width, color=PERF_C, zorder=3,
                label='Relative performance improvement (%)')
    b2 = ax.bar(x, carbon, width, color=CO2_C, zorder=3,
                label='Scaled carbon-cost ratio')
    b3 = ax.bar(x + width, penalty, width, color=PEN_C, zorder=3,
                label='Carbon penalty')

    ax.set_yscale('log')
    ax.set_ylim(0.05, 9000)  # headroom above MolGen's penalty bar for value labels
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.4, zorder=1)

    # Value labels above each bar.
    for rect, v in zip(b1, perf):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.08, f'{v:.2f}%',
                ha='center', va='bottom', fontsize=11 * FS, color=PERF_C,
                fontweight='bold', zorder=4)
    for rect, v in zip(b2, carbon):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.08, f'{v:.2f}',
                ha='center', va='bottom', fontsize=11 * FS, color=CO2_C,
                fontweight='bold', zorder=4)
    for rect, v in zip(b3, penalty):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.08, f'{v:.2f}',
                ha='center', va='bottom', fontsize=11 * FS, color=PEN_C,
                fontweight='bold', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_disp(d) for d in domains], fontsize=20 * FS, fontweight='bold')
    for tick, dom in zip(ax.get_xticklabels(), domains):
        tick.set_color(TASK_COLORS[dom])
    ax.set_ylabel('Best vs. 2nd-best\nPareto model', fontsize=18 * FS)
    ax.tick_params(axis='y', labelsize=15 * FS)
    ax.grid(True, axis='y', alpha=0.25, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(x=0.02)

    ax.legend(loc='upper right', fontsize=15 * FS, framealpha=0.95, ncol=1)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.10)
    out = os.path.join(OUT_DIR, '2b_pareto_tip_bar.png')
    save_outputs(fig, out)
    plt.close(fig)
    print(f"Fig 2b (Pareto-tip bar) saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fig', nargs='*', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='Which figures to generate (default: all). 7=fig1 horizontal, 8=fig3 horizontal, 11=fig2 year gradient, 12=fig4ab horizontal')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--co2', choices=['per_exp', 'per_job'], default='per_exp',
                        help='CO2 metric: per_exp or per_job (default: per_job)')
    parser.add_argument('--noname', action='store_true',
                        help='For figs 1/2/3/6 (and horizontal variants of 1/3), omit per-point '
                             'model labels and save as *_noname.png. Figs 2 and 6 keep Pareto-front labels.')
    parser.add_argument('--svg', action='store_true',
                        help='Also save an editable .svg of each figure (text kept as text) '
                             'for adjusting label positions in Inkscape.')
    args = parser.parse_args()
    SAVE_SVG = args.svg

    # Set CO2 column and label based on argument
    if args.co2 == 'per_exp':
        co2_col = 'CO2_per_exp'
        co2_label = 'log₁₀(CO₂/exp)'
    else:
        co2_col = 'CO2_per_job'
        co2_label = 'log₁₀(CO₂/job)'

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20})

    df = None
    if any(f in args.fig for f in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
        df = load_data()
        print(f"Loaded {len(df)} data points across {df['task'].nunique()} tasks")
        print(f"Using CO2 metric: {args.co2} ({co2_col})")

    if 1 in args.fig:
        plot_fig1(df, co2_col, co2_label, noname=args.noname)
    if 2 in args.fig:
        plot_fig2(df, co2_col, co2_label, noname=args.noname)
    if 3 in args.fig:
        plot_fig3(df, noname=args.noname)
    if 4 in args.fig:
        plot_fig4(df)
    if 5 in args.fig:
        plot_fig5(df)
    if 6 in args.fig:
        plot_fig6(df, co2_col, co2_label, noname=args.noname)
    if 7 in args.fig:
        plot_fig1_horizontal(df, co2_col, co2_label, noname=args.noname)
    if 8 in args.fig:
        plot_fig3_horizontal(df, noname=args.noname)
    if 9 in args.fig:
        plot_fig4b(df)
    if 10 in args.fig:
        plot_fig4_combined(df)
    if 11 in args.fig:
        plot_fig2_gradient(df, co2_col, co2_label)
    if 12 in args.fig:
        plot_fig4_combined_horizontal(df)
    if 13 in args.fig:
        plot_fig_arch_speed_co2(df, co2_col, noname=args.noname)
    if 14 in args.fig:
        plot_fig2(df, co2_col, co2_label, clean=True)  # bigger markers, no labels
    if 15 in args.fig:
        plot_fig6(df, co2_col, co2_label, clean=True)  # bigger markers, no labels
    if 16 in args.fig:
        plot_fig1_alt(df, co2_col, co2_label, noname=args.noname)  # alt-metric Fig 1
    if 17 in args.fig:
        plot_fig1_alt_horizontal(df, co2_col, co2_label, noname=args.noname)  # alt-metric Fig 1 (horizontal)
    if 18 in args.fig:
        plot_fig_arch_size_co2(df, co2_col, noname=args.noname)  # model size vs CO2
    if 19 in args.fig:
        plot_fig2b_pareto_tip(df)  # best vs 2nd-best Pareto tip (companion to Fig 2)
    if 20 in args.fig:
        plot_fig2b_pareto_tip_bar(df)  # bar version, full width, below Fig 2

    print("Done!")
