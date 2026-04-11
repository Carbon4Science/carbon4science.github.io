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

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = 'analysis/figures'
os.makedirs(OUT_DIR, exist_ok=True)

TASK_ORDER = ['MatGen', 'MolGen', 'Retro', 'Forward', 'StructOpt', 'MDSim']

TASK_COLORS = {
    'MatGen':    '#2ca02c',
    'MolGen':    '#1f77b4',
    'Retro':     '#17becf',
    'Forward':   '#ff7f0e',
    'StructOpt': '#d62728',
    'MDSim':     '#9467bd',
}

TASK_PERF_LABEL = {
    'MatGen': 'mSUN (%)',
    'MolGen': 'VUN (%)',
    'Retro': 'Top-50 Acc (%)',
    'Forward': 'Top-3 Acc (%)',
    'StructOpt': 'CPS',
    'MDSim': 'MSD',
}

ARCH_MARKERS = {
    'MLP':           'o',
    'GNN':           'D',
    'LM':            '^',
    'LLM':           '*',
    'VAE':           's',
    'Diffusion':     'X',
    'Flow Matching': 'P',
}

# Reference point categories (Figure 5)
# AI categories reuse task colors from Figs 1-3; non-AI get distinct colors
REF_CATEGORY_COLORS = {
    'Everyday activities':       '#78909C',   # blue-gray
    'LLM inference':             '#42A5F5',   # blue (same as LLM markers)
    'Chemical simulation':       '#FFB74D',   # amber
    'Chemical synthesis':        '#E53935',   # red
    'AI Chemical generation':    '#2ca02c',   # green (= MatGen)
    'AI Synthesis prediction':   '#ff7f0e',   # orange (= Forward)
    'AI MD simulation':          '#9467bd',   # purple (= MDSim)
}
# Ordered legend for Figure 5
REF_LEGEND_ORDER = [
    'Everyday activities', 'LLM inference', 'Chemical simulation',
    'Chemical synthesis', 'AI Chemical generation', 'AI Synthesis prediction',
    'AI MD simulation',
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
MARKER_SIZE_SCALE = {'*': 3.0, '^': 1.5}

def marker_size(m, base=150):
    """Return scatter size, scaling up star markers."""
    return base * MARKER_SIZE_SCALE.get(m, 1.0)

def get_arch_legend_handles():
    return [mlines.Line2D([], [], color='gray', marker=m, linestyle='None',
            markersize=12 if m == '*' else 8, label=a) for a, m in ARCH_MARKERS.items()]


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
def plot_fig1(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)'):
    """6 rows (tasks) × 3 cols (model size, CO2, performance)."""
    fig, axes = plt.subplots(6, 3, figsize=(20, 32))

    # Consistent x-axis: use global min/max year across ALL tasks
    all_years = df['year'].dropna().unique()
    year_min, year_max = int(all_years.min()), int(all_years.max())
    year_ticks = list(range(year_min, year_max + 1, 2))  # e.g. 2017, 2019, ...
    x_pad = 0.5
    xlim = (year_min - x_pad, year_max + x_pad)

    for i, task in enumerate(TASK_ORDER):
        grp = df[df['task'] == task]
        c = TASK_COLORS[task]
        col_specs = [
            ('year', '_size_num',    'log₁₀(Model Size)',       True),
            ('year', co2_col,          co2_label,                  True),
            ('year', 'major_metric',   TASK_PERF_LABEL[task],     False),
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
                m = ARCH_MARKERS.get(row['model type'], 'o')
                is_base = row.get('baseline?', False)
                ec = 'black' if is_base else 'white'
                lw = 2.0 if is_base else 0.6
                ax.scatter(row['year'], yval, color=c, marker=m, s=marker_size(m),
                           edgecolors=ec, linewidths=lw, zorder=3)
                # Nudge overlapping labels: Forward performance column
                tx, ty = row['year'], yval
                if task == 'Forward' and j == 2 and row['model'] == 'RSMILES':
                    ty += 3.0
                elif task == 'Forward' and j == 2 and row['model'] == 'LocalTransform':
                    ty -= 3.0
                elif task == 'MolGen' and j == 2 and row['model'] == 'SmileyLlama':
                    ty += 3.0
                elif task == 'MolGen' and j == 2 and row['model'] == 'REINVENT4':
                    ty -= 3.0
                texts.append(ax.text(tx, ty, row['model'],
                                     fontsize=14, zorder=5))
            # Set limits BEFORE adjust_text so it adjusts within correct bounds
            ax.set_xlim(xlim)
            ax.set_xticks(year_ticks)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.tick_params(labelsize=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            if i == 0:
                titles = ['Model Size', 'CO₂ Emission', 'Performance']
                ax.set_title(titles[j], fontsize=24, fontweight='bold', pad=20)
            if i < 5:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Year', fontsize=20)
            if j == 0:
                ax.text(-0.3, 0.5, task, transform=ax.transAxes, fontsize=22,
                        fontweight='bold', color=c, va='center', ha='center', rotation=90)
            # adjust_text AFTER all axis setup
            adjust_text(texts, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                        force_points=(2.0, 2.0), iterations=200,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=16,
               loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.01),
               framealpha=0.9, title_fontsize=18)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.06, hspace=0.35, wspace=0.4)
    out = os.path.join(OUT_DIR, '1_year_trends_combined.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 1 saved → {out}")


# ── Figure 2: Pareto Frontiers ─────────────────────────────────────────────
def plot_fig2(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)'):
    """2×3 subplots, Δ Performance (%) vs log10(CO2 ratio), with Pareto front."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
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
        best_perf = -np.inf
        for _, row in grp_sorted.iterrows():
            if row['delta_perf_pct'] >= best_perf:
                pareto_x.append(row['log_co2_ratio'])
                pareto_y.append(row['delta_perf_pct'])
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

        # plot points
        texts2 = []
        for _, row in grp.iterrows():
            m = ARCH_MARKERS.get(row['model type'], 'o')
            is_base = row.get('baseline?', False)
            sz = marker_size(m)
            ec = 'black' if is_base else 'gray'
            lw = 2.5 if is_base else 0.8
            ax.scatter(row['log_co2_ratio'], row['delta_perf_pct'],
                       color=TASK_COLORS[task], marker=m, s=sz,
                       edgecolors=ec, linewidths=lw, zorder=4)
            label = f"{row['model']} ({row['year']})"
            fw = 'bold' if is_base else 'normal'
            texts2.append(ax.text(row['log_co2_ratio'], row['delta_perf_pct'], label,
                                  fontsize=14, fontweight=fw, zorder=5))

        # Set limits and styling BEFORE adjust_text
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('log₁₀(CO₂ eq ratio)', fontsize=20)
        ax.set_ylabel('Δ Relative Performance (%)', fontsize=20)
        ax.set_title(task, fontsize=24, fontweight='bold', color=TASK_COLORS[task])
        ax.tick_params(labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # adjust_text AFTER axis setup
        adjust_text(texts2, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
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
    pareto_line = mlines.Line2D([], [], color='black', linestyle='--',
                                linewidth=1.5, label='Pareto Front')
    fig.legend(handles=arch_handles + [pareto_line] + quad_handles,
               loc='lower center', ncol=6, fontsize=16, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02), title_fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1, hspace=0.35, wspace=0.3)
    out = os.path.join(OUT_DIR, '2_pareto_delta_pct.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 2 saved → {out}")


# ── Figure 3: CO2 Decomposition ───────────────────────────────────────────
def plot_fig3(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)'):
    """6 rows (tasks) × 3 cols (size vs CO2, time vs CO2, size vs time). No fit lines.
    Uses log10-transformed values on linear axes for clean tick labels."""
    fig, axes = plt.subplots(6, 3, figsize=(20, 32))

    panels = [
        ('_size_num',                co2_col,                   'log₁₀(Model Size)',    co2_label),
        ('inference_time_per_exp',   co2_col,                   'log₁₀(Inference Time)', co2_label),
        ('_size_num',                'inference_time_per_exp',  'log₁₀(Model Size)',    'log₁₀(Inference Time)'),
    ]

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
                ec = 'black' if is_base else 'white'
                lw = 2.0 if is_base else 0.6
                ax.scatter(log_xv, log_yv, color=c, marker=m, s=sz,
                           edgecolors=ec, linewidths=lw, zorder=3)
                texts3.append(ax.text(log_xv, log_yv, row['model'], fontsize=14, zorder=5))
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
                        fontsize=16, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            # Shared x-limits per column
            ax.set_xlim(col_xlims[j])
            ax.tick_params(labelsize=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, which='both')
            if i == 0:
                col_titles = ['Model Size vs CO₂', 'Inference Time vs CO₂',
                              'Model Size vs Inference Time']
                ax.set_title(col_titles[j], fontsize=24, fontweight='bold', pad=20)
            if i < 5:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            if j == 0:
                ax.text(-0.3, 0.5, task, transform=ax.transAxes, fontsize=22,
                        fontweight='bold', color=c, va='center', ha='center', rotation=90)
            # adjust_text AFTER all axis setup
            adjust_text(texts3, ax=ax, expand=(1.5, 1.8), force_text=(2.0, 2.0),
                        force_points=(2.0, 2.0), iterations=200,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

    fig.legend(handles=get_arch_legend_handles(), title='Architecture', fontsize=16,
               loc='lower center', ncol=len(ARCH_MARKERS), bbox_to_anchor=(0.5, -0.01),
               framealpha=0.9, title_fontsize=18)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.06, hspace=0.35, wspace=0.4)
    out = os.path.join(OUT_DIR, '3_co2_decomposition_combined.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 3 saved → {out}")


# ── Figure 4: CO2 Reference Points ────────────────────────────────────────
def plot_fig4():
    """Horizontal bar chart of CO2 reference points, ordered by magnitude."""
    # (category, main_label, description, co2_grams, unit)
    ref_data = [
        ('LLM inference',            'Image generation',         'Stable Diffusion',   1.38,    'image'),
        ('LLM inference',            'Text generation',          'Claude-3.7 Sonnet',  2.12,    '10k in & 1.5k out'),
        ('AI Synthesis prediction',  'Reaction outcome pred.',   'LlaSMol',            17.7,    '500 inputs'),
        ('Everyday activities',      'Smartphone charge',        'iPhone 16 Pro Max',   9.7,    'full charge'),
        ('Chemical simulation',      'Classical MD',             'force field',         10,     '1M steps'),
        ('AI Chemical generation',   'Material generation',      'MatterGen',         248,     '1K strcutures'),
        ('Everyday activities',      'Driving a car',            'EU average',         170,     'km'),
        ('AI Synthesis prediction',  'Synthesis Planning',       'RetroBridge',        403,     '500 molecules'),
        ('AI Chemical generation',   'Molecule generation',      'DeFoG',              355.2,  '10K molecules'),
        ('AI MD simulation',         'MLIP simulation',         'eSEN',              3486,     '1M steps'),
        ('Chemical synthesis',       'Battery synthesis',        'Vanadium flow battery', 37000, 'MWh'),
        ('Chemical synthesis',       'Material synthesis',       'UiO-66-NH₂ (aqueous-based) ',       43000,    'kg'),
        ('Chemical simulation',      'Ab initio MD',             'PBE (50 atoms)',   140960,    '1M steps'),
        ('Chemical synthesis',       'Organic synthesis',        'Letermovir (Merck)',  382000,    'kg'),
    ]
    ref_data.sort(key=lambda x: x[3])

    categories  = [d[0] for d in ref_data]
    main_labels = [d[1] for d in ref_data]
    descs       = [d[2] for d in ref_data]
    values      = [d[3] for d in ref_data]
    units       = [d[4] for d in ref_data]
    colors      = [REF_CATEGORY_COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(ref_data)), values, color=colors, edgecolor='white', height=0.65)

    for i, (val, unit) in enumerate(zip(values, units)):
        label = (f'{val / 1000:.1f} kg CO₂ eq/{unit}' if val >= 1000
                 else f'{val:.1f} g CO₂ eq/{unit}')
        ax.text(val * 1.4, i, label, va='center', fontsize=8.5)

    ax.set_yticks(range(len(ref_data)))
    ax.set_yticklabels([''] * len(ref_data))
    for i, (main, desc) in enumerate(zip(main_labels, descs)):
        ax.text(-0.03, i, main, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=10.5, fontweight='bold', color='black')
        ax.text(-0.035, i - 0.28, f'{desc}', transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8.5, fontstyle='italic', color='#777777')

    ax.set_xscale('log')
    ax.set_xlabel('CO₂ Emission', fontsize=14)
    # ax.set_title('CO₂ Emission Reference Points', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, 5e8)

    leg = [mpatches.Patch(facecolor=REF_CATEGORY_COLORS[cat], label=cat)
           for cat in REF_LEGEND_ORDER]
    ax.legend(handles=leg, loc='lower right', fontsize=9,
              framealpha=0.9, title='Category', title_fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.32)
    out = os.path.join(OUT_DIR, '4_co2_reference_points.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 4 saved → {out}")


# ── Figure 5: Cross-task CO2 decomposition (2 panels) ─────────────────────
def plot_fig5(df):
    """Two panels: log10(inference time) vs log10(CO2) and log10(model size) vs log10(CO2),
    all tasks combined with regression line and R²."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9))

    for panel_ax, xcol, xlabel in [
        (ax_left,  'inference_time_per_exp', 'log₁₀(Inference Time)'),
        (ax_right, '_size_num',              'log₁₀(Model Size)'),
    ]:
        all_logx, all_logy = [], []
        for task, grp in df.groupby('task'):
            c = TASK_COLORS[task]
            for _, row in grp.iterrows():
                xv, yv = row[xcol], row['CO2_per_job']
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
            r2 = 1 - np.sum((all_logy - np.polyval(coef, all_logx))**2) / \
                     np.sum((all_logy - all_logy.mean())**2)
            xfit = np.linspace(all_logx.min(), all_logx.max(), 50)
            panel_ax.plot(xfit, np.polyval(coef, xfit), 'r--', lw=1.5, alpha=0.7, zorder=2)
            panel_ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=panel_ax.transAxes,
                          fontsize=18, va='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        panel_ax.set_xlabel(xlabel, fontsize=20)
        panel_ax.set_ylabel('log₁₀(CO₂/job)', fontsize=20)
        panel_ax.tick_params(labelsize=16)
        panel_ax.spines['top'].set_visible(False)
        panel_ax.spines['right'].set_visible(False)
        panel_ax.grid(True, alpha=0.2)

    task_handles = [mpatches.Patch(color=c, label=t) for t, c in TASK_COLORS.items()]
    arch_handles = get_arch_legend_handles()
    fig.legend(handles=task_handles + arch_handles,
               loc='lower center', ncol=7, fontsize=14, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.15, wspace=0.3)
    out = os.path.join(OUT_DIR, '5_co2_decomposition_cross_task.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 5 saved → {out}")


# ── Figure 6: Pareto with alternative metrics ────────────────────────────
# Alternative metrics: MatGen(SUN), MolGen(SUN), Retro(Top-10), Forward(Top-1)
ALT_METRICS = {
    'MatGen':  {'CDVAE': 3.2, 'DiffCSP': 4.3, 'CrystaLLM': 3.5, 'FlowMM': 4.3,
                'ChargeDIFF': 4.4, 'MatterGen': 5.2, 'ADiT': 5.5, 'CrystalFlow': 3.0},
    'Retro':   {'neuralsym': 72.8, 'MEGAN': 87.0, 'LocalRetro': 90.4, 'RSMILES': 89.6,
                'Chemformer': 62.8, 'LlaSMol': 5.0, 'RetroBridge': 44.9, 'RSGPT': 96.6},
    'Forward': {'neuralsym': 49.5, 'MEGAN': 80.1, 'Graph2SMILES': 88.5, 'Chemformer': 89.0,
                'LocalTransform': 89.4, 'MolecularTransformer': 86.8, 'RSMILES': 89.4, 'LlaSMol': 3.8},
}
# MolGen VUNS is already in the pretrained data loaded in load_data()
MOLGEN_VUNS = {
    'REINVENT': 67.11, 'JT-VAE': 75.63, 'HierVAE': 74.18, 'MolGPT': 82.00,
    'DiGress': 56.14, 'REINVENT4': 74.27, 'SmileyLlama': 65.95, 'DeFoG': 64.75
}

ALT_TASK_ORDER = ['MatGen', 'MolGen', 'Retro', 'Forward']
ALT_LABELS = {
    'MatGen': 'SUN (%)', 'MolGen': 'VUNS (%)',
    'Retro': 'Top-10 Acc (%)', 'Forward': 'Top-1 Acc (%)',
}


def plot_fig6(df, co2_col='CO2_per_job', co2_label='log₁₀(CO₂/job)'):
    """Pareto plots using alternative metrics for 4 tasks."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # Add alt_perf column
    df = df.copy()
    df['alt_perf'] = np.nan
    for idx, row in df.iterrows():
        task, model = row['task'], row['model']
        if task == 'MolGen' and model in MOLGEN_VUNS:
            df.at[idx, 'alt_perf'] = MOLGEN_VUNS[model]
        elif task in ALT_METRICS and model in ALT_METRICS[task]:
            df.at[idx, 'alt_perf'] = ALT_METRICS[task][model]

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
        best = -np.inf
        for _, row in grp_sorted.iterrows():
            if row['delta_alt_pct'] >= best:
                pareto_x.append(row['log_co2_ratio'])
                pareto_y.append(row['delta_alt_pct'])
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
            m = ARCH_MARKERS.get(row['model type'], 'o')
            is_base = row.get('baseline?', False)
            ec = 'black' if is_base else 'gray'
            lw = 2.5 if is_base else 0.8
            ax.scatter(row['log_co2_ratio'], row['delta_alt_pct'],
                       color=TASK_COLORS[task], marker=m, s=marker_size(m),
                       edgecolors=ec, linewidths=lw, zorder=4)
            label = f"{row['model']} ({row['year']})"
            fw = 'bold' if is_base else 'normal'
            tx6, ty6 = row['log_co2_ratio'], row['delta_alt_pct']
            if task == 'Forward' and row['model'] == 'Graph2SMILES':
                ty6 += 5.0
            elif task == 'Forward' and row['model'] == 'LocalTransform':
                ty6 -= 5.0
            texts6.append(ax.text(tx6, ty6, label,
                                  fontsize=14, fontweight=fw, zorder=5))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('log₁₀(CO₂ eq ratio)', fontsize=20)
        ax.set_ylabel(f'Δ Relative {ALT_LABELS[task]}', fontsize=20)
        ax.set_title(task, fontsize=22, fontweight='bold', color=TASK_COLORS[task])
        ax.tick_params(labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
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
    pareto_line = mlines.Line2D([], [], color='black', linestyle='--',
                                linewidth=1.5, label='Pareto Front')
    fig.legend(handles=arch_handles + [pareto_line] + quad_handles,
               loc='lower center', ncol=6, fontsize=14, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.15, wspace=0.35)
    out = os.path.join(OUT_DIR, '6_pareto_alt_metrics.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Fig 6 saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fig', nargs='*', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='Which figures to generate (default: all)')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--co2', choices=['per_exp', 'per_job'], default='per_job',
                        help='CO2 metric: per_exp or per_job (default: per_job)')
    args = parser.parse_args()

    # Set CO2 column and label based on argument
    if args.co2 == 'per_exp':
        co2_col = 'CO2_per_exp'
        co2_label = 'log₁₀(CO₂/exp)'
    else:
        co2_col = 'CO2_per_job'
        co2_label = 'log₁₀(CO₂/job)'

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20})

    df = None
    if any(f in args.fig for f in [1, 2, 3, 5, 6]):
        df = load_data()
        print(f"Loaded {len(df)} data points across {df['task'].nunique()} tasks")
        print(f"Using CO2 metric: {args.co2} ({co2_col})")

    if 1 in args.fig:
        plot_fig1(df, co2_col, co2_label)
    if 2 in args.fig:
        plot_fig2(df, co2_col, co2_label)
    if 3 in args.fig:
        plot_fig3(df, co2_col, co2_label)
    if 4 in args.fig:
        plot_fig4()
    if 5 in args.fig:
        plot_fig5(df)
    if 6 in args.fig:
        plot_fig6(df, co2_col, co2_label)

    print("Done!")
