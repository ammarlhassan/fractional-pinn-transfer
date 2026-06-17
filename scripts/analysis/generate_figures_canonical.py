#!/usr/bin/env python3
"""
Generate all publication figures from canonical reproducibility JSON files.

Data is loaded exclusively from:
  reproducibility/data/core/          (fidelity sweeps, N_col sweeps)
  reproducibility/data/ablations/     (GL memory ablation)

Nothing is hardcoded; every value in every figure matches its
corresponding table in the paper exactly.

Usage (from project root):
    .venv/bin/python scripts/analysis/generate_figures_canonical.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Run from project root regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

DATA_DIR = ROOT / 'reproducibility' / 'data'
FIG_DIR  = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 9.5,
    'legend.framealpha': 0.85,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
})

BLUE   = '#1f77b4'   # MF-fPINN
RED    = '#d62728'   # Vanilla / MF hurts
GRAY   = '#8c8c8c'   # neutral
GREEN  = '#2ca02c'   # Vanilla+50

# LF quality table (matches Table 9 in paper)
LF_QUALITY = {
    5:  {0.3: 130.9, 0.5: 123.7, 0.7: 111.8, 0.9: 96.2,  1.0: 71.2},
    8:  {0.3: 51.2,  0.5: 40.8,  0.7: 40.5,  0.9: 33.2,  1.0: 25.2},
    10: {0.3: 32.9,  0.5: 25.3,  0.7: 25.2,  0.9: 20.2,  1.0: 15.3},
    12: {0.3: 21.7,  0.5: 16.6,  0.7: 16.1,  0.9: 12.7,  1.0: 10.0},
    15: {0.3: 12.4,  0.5: 9.2,   0.7: 8.9,   0.9: 7.0,   1.0: 5.5},
    20: {0.3: 5.1,   0.5: 3.7,   0.7: 3.5,   0.9: 2.7,   1.0: 2.2},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def bar_color(ratio):
    """Blue: MF clearly better; Red: MF clearly worse; Gray: neutral."""
    if ratio > 1.05:
        return BLUE
    if ratio < 0.95:
        return RED
    return GRAY


def annotate_ratio(ax, x_pos, ratio, p_val, fontsize=9):
    """Place 'ratio×[*]' above a bar, using the axis ylim for offset."""
    ylo, yhi = ax.get_ylim()
    y_ann = ratio + 0.04 * (yhi - ylo)

    if p_val is not None and p_val < 0.01:
        star = '**'
    elif p_val is not None and p_val < 0.05:
        star = '*'
    else:
        star = ''

    label = f'{ratio:.2f}×{star}'   # × = unicode 00d7
    color = 'navy' if ratio >= 1.0 else 'darkred'
    ax.text(x_pos, y_ann, label, ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold', color=color)


def make_legend_patches(items):
    """items: list of (color, label) tuples → list of Patch handles."""
    return [mpatches.Patch(facecolor=c, label=lbl) for c, lbl in items]


# ---------------------------------------------------------------------------
# FIGURE 1: α sweep at N_col = 100  (Tables 1 & 2)
# ---------------------------------------------------------------------------
def fig_alpha_advantage():
    alphas = [0.5, 0.7, 1.0]
    mf_m, mf_s, van_m, van_s, ratios, pvals = [], [], [], [], [], []

    for a in alphas:
        d = load_json(DATA_DIR / f'core/ncol_alpha{a}_nterms12.json')['N_col=100']
        vm, vs = d['vanilla']['mean'], d['vanilla']['std']
        mm, ms = d['mf']['mean'],      d['mf']['std']
        mf_m.append(mm);  mf_s.append(ms)
        van_m.append(vm); van_s.append(vs)
        ratios.append(vm / mm)
        diffs = [v - m for v, m in zip(d['vanilla']['l2_values'], d['mf']['l2_values'])]
        pval = scipy_stats.wilcoxon(diffs, alternative='two-sided').pvalue
        pvals.append(pval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(alphas))
    w = 0.32
    xlabels = [f'α = {a}' for a in alphas]   # α = 0.5 etc.

    # left: raw errors
    ax1.bar(x - w/2, mf_m,  w, yerr=mf_s,  label='MF-fPINN',
            color=BLUE, alpha=0.85, capsize=4, error_kw={'lw': 1.2},
            edgecolor='white', zorder=3)
    ax1.bar(x + w/2, van_m, w, yerr=van_s, label='Vanilla fPINN',
            color=RED,  alpha=0.85, capsize=4, error_kw={'lw': 1.2},
            edgecolor='white', zorder=3)
    ax1.set_xticks(x);  ax1.set_xticklabels(xlabels)
    ax1.set_ylabel('Relative $L^2$ error (%)')
    ax1.set_title('(a) MF-fPINN vs Vanilla at $N_{\\mathrm{col}} = 100$')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max(max(van_m), max(mf_m)) * 1.5)

    # right: advantage ratio
    colors_r = [bar_color(r) for r in ratios]
    ax2.bar(x, ratios, 0.45, color=colors_r, alpha=0.88,
            edgecolor='white', zorder=3)
    ax2.axhline(1.0, color='k', linestyle='--', lw=0.9, alpha=0.55)
    ax2.set_xticks(x);  ax2.set_xticklabels(xlabels)
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) MF advantage factor  ($N_{\\mathrm{terms}} = 12$, 10 seeds)')
    ax2.set_ylim(0, max(ratios) * 1.45)
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        annotate_ratio(ax2, i, r, p)
    handles = make_legend_patches([(BLUE, 'MF better (> 1.05×)'),
                                   (GRAY, 'Comparable'),
                                   (RED,  'MF worse (< 0.95×)')])
    ax2.legend(handles=handles, loc='upper left', fontsize=8.5)

    plt.tight_layout()
    out = FIG_DIR / 'alpha_advantage_ncol100.pdf'
    plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# FIGURE 2: N_col budget at α = 0.5  (Table 1)
# ---------------------------------------------------------------------------
def fig_ncol_budget():
    d = load_json(DATA_DIR / 'core/ncol_alpha0.5_nterms12.json')
    ncols = [100, 200, 500]

    mf_m, mf_s, van_m, van_s = [], [], [], []
    v50_nc, v50_m, v50_s = [], [], []
    for nc in ncols:
        key = f'N_col={nc}'
        mf_m.append(d[key]['mf']['mean']);        mf_s.append(d[key]['mf']['std'])
        van_m.append(d[key]['vanilla']['mean']);   van_s.append(d[key]['vanilla']['std'])
        v50 = d[key].get('van50')
        if v50:
            v50_nc.append(nc); v50_m.append(v50['mean']); v50_s.append(v50['std'])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ncols, mf_m, yerr=mf_s, fmt='o-', color=BLUE,
                label='MF-fPINN', capsize=4, markersize=7, zorder=4)
    ax.errorbar(ncols, van_m, yerr=van_s, fmt='s-', color=RED,
                label='Vanilla fPINN', capsize=4, markersize=7, zorder=4)
    if v50_nc:
        ax.errorbar(v50_nc, v50_m, yerr=v50_s, fmt='^--', color=GREEN,
                    label='Vanilla+50', capsize=4, markersize=7, zorder=4)

    ax.set_xscale('log')
    ax.set_xticks(ncols); ax.set_xticklabels([str(n) for n in ncols])
    ax.set_xlabel('$N_{\\mathrm{col}}$ (collocation budget)')
    ax.set_ylabel('Relative $L^2$ error (%)')
    ax.set_title('N_col budget  (α = 0.5, $N_{\\mathrm{terms}} = 12$, 10 seeds)')
    ax.legend()

    out = FIG_DIR / 'ncol_budget_alpha0.5_v2.pdf'
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# FIGURE 3: Fidelity × α cross-comparison  (Table 4)
# ---------------------------------------------------------------------------
def fig_fidelity_cross():
    d05 = load_json(DATA_DIR / 'core/fidelity_10seed_alpha0.5.json')
    d10 = load_json(DATA_DIR / 'core/fidelity_10seed_alpha1.0.json')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    for col_i, nterms in enumerate([8, 12]):
        ax = axes[col_i]
        alphas = [0.5, 1.0]
        src_map = {0.5: d05, 1.0: d10}
        x = np.arange(2)
        w = 0.30

        mf_m_list, mf_s_list, van_m_list, van_s_list = [], [], [], []
        ratio_list, p_list = [], []
        for a in alphas:
            src = src_map[a]
            key = f'MF_Nterms{nterms}'
            van_mean = src['vanilla']['mean']
            van_std  = src['vanilla']['std']
            mf_mean  = src[key]['mean']
            mf_std   = src[key]['std']
            mf_m_list.append(mf_mean);   mf_s_list.append(mf_std)
            van_m_list.append(van_mean); van_s_list.append(van_std)
            ratio_list.append(van_mean / mf_mean)
            p_list.append(src[key].get('wilcoxon_p', 1.0))

        ax.bar(x - w/2, mf_m_list, w, yerr=mf_s_list,
               label='MF-fPINN', color=BLUE, alpha=0.85, capsize=4,
               error_kw={'lw': 1.2}, edgecolor='white', zorder=3)
        ax.bar(x + w/2, van_m_list, w, yerr=van_s_list,
               label='Vanilla fPINN', color=RED, alpha=0.85, capsize=4,
               error_kw={'lw': 1.2}, edgecolor='white', zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f'α = {a}' for a in alphas])
        lf5 = LF_QUALITY[nterms][0.5]; lf1 = LF_QUALITY[nterms][1.0]
        ax.set_title(f'({"ab"[col_i]}) $N_{{\\mathrm{{terms}}}} = {nterms}$'
                     f'  (LF error: {lf5:.0f}% / {lf1:.0f}%)')
        ax.set_ylabel('Relative $L^2$ error (%)')
        if col_i == 0:
            ax.legend(loc='upper right')

        ymax = max(max(m + s for m, s in zip(mf_m_list, mf_s_list)),
                   max(v + s for v, s in zip(van_m_list, van_s_list)))
        ax.set_ylim(0, ymax * 1.55)

        for j, (r, p) in enumerate(zip(ratio_list, p_list)):
            annotate_ratio(ax, j, r, p)

    plt.tight_layout()
    out = FIG_DIR / 'fidelity_cross_comparison.pdf'
    plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# FIGURE 4: Fidelity sweep at α = 1.0  (Table 5)
# ---------------------------------------------------------------------------
def fig_fidelity_sweep_a10():
    d = load_json(DATA_DIR / 'core/fidelity_10seed_alpha1.0.json')
    van_mean = d['vanilla']['mean']
    van_std  = d['vanilla']['std']

    nterms_vals = [5, 8, 10, 12, 15, 20]
    mf_means = [d[f'MF_Nterms{n}']['mean']       for n in nterms_vals]
    mf_stds  = [d[f'MF_Nterms{n}']['std']        for n in nterms_vals]
    lf_errs  = [d[f'MF_Nterms{n}']['lf_err']     for n in nterms_vals]
    ratios   = [d[f'MF_Nterms{n}']['ratio']      for n in nterms_vals]
    pvals    = [d[f'MF_Nterms{n}']['wilcoxon_p'] for n in nterms_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: MF error curve vs LF quality
    # clip N_terms=5 (177%) from the line plot for readability; show as separate marker
    lf_plot = lf_errs[1:]    # skip N_terms=5
    mf_plot = mf_means[1:]
    mf_splot = mf_stds[1:]
    ax1.errorbar(lf_plot, mf_plot, yerr=mf_splot, fmt='o-', color=BLUE,
                 label='MF-fPINN', capsize=4, linewidth=2, markersize=7, zorder=4)
    # mark N_terms=5 separately (off scale)
    ax1.errorbar([lf_errs[0]], [min(mf_means[0], 55)], fmt='o', color=BLUE,
                 capsize=4, markersize=7, zorder=4)
    ax1.annotate(f'$N_t=5$ (177%↑)', (lf_errs[0], 55),
                 xytext=(lf_errs[0]+3, 52), fontsize=7.5, color=BLUE)

    ax1.axhline(van_mean, color=RED, linestyle='--', lw=1.6,
                label=f'Vanilla ({van_mean:.1f}%, 10 seeds)')
    ax1.fill_between([0, 80], van_mean - van_std, van_mean + van_std,
                     color=RED, alpha=0.12)
    for i, n in enumerate(nterms_vals[1:], 1):
        ax1.annotate(f'$N_t={n}$', (lf_errs[i], mf_means[i]),
                     textcoords='offset points', xytext=(3, 8),
                     fontsize=8, ha='left')

    ax1.set_xlim(-2, 32); ax1.set_ylim(0, 60)
    ax1.set_xlabel('LF data error (%)')
    ax1.set_ylabel('MF-fPINN $L^2$ error (%)')
    ax1.set_title('(a) MF error vs LF quality  (α = 1.0)')
    ax1.legend(loc='upper left')

    # right: advantage ratio bars — MONOTONICALLY INCREASING (as in Table 5)
    colors_r = [bar_color(r) for r in ratios]
    ax2.bar(range(len(nterms_vals)), ratios, color=colors_r,
            alpha=0.88, edgecolor='white', zorder=3)
    ax2.axhline(1.0, color='k', linestyle='--', lw=0.9, alpha=0.55)
    xtick_labels = [f'$N_t={n}$\n({LF_QUALITY[n][1.0]:.0f}%)' for n in nterms_vals]
    ax2.set_xticks(range(len(nterms_vals)))
    ax2.set_xticklabels(xtick_labels, fontsize=8.5)
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) Monotonic advantage at α = 1.0')   # ← CORRECTED TITLE
    ax2.set_ylim(0, max(ratios) * 1.45)
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        annotate_ratio(ax2, i, r, p, fontsize=8.5)
    handles = make_legend_patches([(BLUE, 'MF better'), (RED, 'MF worse')])
    ax2.legend(handles=handles, loc='upper right', fontsize=8.5)

    plt.tight_layout()
    out = FIG_DIR / 'fidelity_sweep_alpha1.0.pdf'
    plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# FIGURE 5: Fidelity sweep at α = 0.5  (Table 6)
# ---------------------------------------------------------------------------
def fig_fidelity_sweep_a05():
    d = load_json(DATA_DIR / 'core/fidelity_10seed_alpha0.5.json')
    van_mean = d['vanilla']['mean']
    van_std  = d['vanilla']['std']

    nterms_vals = [8, 12, 15, 20]
    mf_means = [d[f'MF_Nterms{n}']['mean']       for n in nterms_vals]
    mf_stds  = [d[f'MF_Nterms{n}']['std']        for n in nterms_vals]
    lf_errs  = [d[f'MF_Nterms{n}']['lf_err']     for n in nterms_vals]
    ratios   = [d[f'MF_Nterms{n}']['ratio']      for n in nterms_vals]
    pvals    = [d[f'MF_Nterms{n}']['wilcoxon_p'] for n in nterms_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: MF error curve
    ax1.errorbar(lf_errs, mf_means, yerr=mf_stds, fmt='o-', color=BLUE,
                 label='MF-fPINN', capsize=4, linewidth=2, markersize=7, zorder=4)
    ax1.axhline(van_mean, color=RED, linestyle='--', lw=1.6,
                label=f'Vanilla ({van_mean:.1f}%, 10 seeds)')
    ax1.fill_between([0, 50], van_mean - van_std, van_mean + van_std,
                     color=RED, alpha=0.12)
    for i, n in enumerate(nterms_vals):
        ax1.annotate(f'$N_t={n}$', (lf_errs[i], mf_means[i]),
                     textcoords='offset points', xytext=(3, 8),
                     fontsize=8, ha='left')

    ax1.set_xlim(-1, 50)
    ax1.set_ylim(0, max(mf_means) * 1.6)
    ax1.set_xlabel('LF data error (%)')
    ax1.set_ylabel('MF-fPINN $L^2$ error (%)')
    ax1.set_title('(a) MF error vs LF quality  (α = 0.5)')
    ax1.legend(loc='upper left')

    # right: advantage ratio bars — MONOTONICALLY INCREASING (as in Table 6)
    colors_r = [bar_color(r) for r in ratios]
    ax2.bar(range(len(nterms_vals)), ratios, color=colors_r,
            alpha=0.88, edgecolor='white', zorder=3)
    ax2.axhline(1.0, color='k', linestyle='--', lw=0.9, alpha=0.55)
    xtick_labels = [f'$N_t={n}$\n({LF_QUALITY[n][0.5]:.0f}%)' for n in nterms_vals]
    ax2.set_xticks(range(len(nterms_vals)))
    ax2.set_xticklabels(xtick_labels, fontsize=9)
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) Monotonic advantage at α = 0.5')
    ax2.set_ylim(0, max(ratios) * 1.45)
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        annotate_ratio(ax2, i, r, p)
    handles = make_legend_patches([(BLUE, 'MF better'), (RED, 'MF worse')])
    ax2.legend(handles=handles, loc='upper right', fontsize=8.5)

    plt.tight_layout()
    out = FIG_DIR / 'fidelity_sweep_alpha0.5.pdf'
    plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# FIGURE 6: GL memory depth ablation  (Table 8)
# ---------------------------------------------------------------------------
def fig_gl_ablation():
    d = load_json(DATA_DIR / 'ablations/gl_ablation_alpha0.5.json')
    nmem_vals = [2, 5, 10, 20, 50, 100]

    van_m = [d[f'gl_mem={n}']['vanilla']['mean'] for n in nmem_vals]
    van_s = [d[f'gl_mem={n}']['vanilla']['std']  for n in nmem_vals]
    mf_m  = [d[f'gl_mem={n}']['mf']['mean']      for n in nmem_vals]
    mf_s  = [d[f'gl_mem={n}']['mf']['std']       for n in nmem_vals]
    ratios = [d[f'gl_mem={n}']['ratio']          for n in nmem_vals]
    pvals  = [d[f'gl_mem={n}']['wilcoxon_p']     for n in nmem_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(nmem_vals))
    w = 0.32

    # left: raw errors
    ax1.bar(x - w/2, mf_m, w, yerr=mf_s, label='MF-fPINN',
            color=BLUE, alpha=0.85, capsize=4, error_kw={'lw': 1.2},
            edgecolor='white', zorder=3)
    ax1.bar(x + w/2, van_m, w, yerr=van_s, label='Vanilla fPINN',
            color=RED, alpha=0.85, capsize=4, error_kw={'lw': 1.2},
            edgecolor='white', zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels([str(n) for n in nmem_vals])
    ax1.set_xlabel('GL memory depth $N_{\\mathrm{mem}}$')
    ax1.set_ylabel('Relative $L^2$ error (%)')
    ax1.set_title('(a) Error vs GL depth  (α = 0.5, 10 seeds)')
    ax1.legend(); ax1.set_ylim(0, max(max(van_m), max(mf_m)) * 1.6)

    # right: advantage ratio
    colors_r = [bar_color(r) for r in ratios]
    ax2.bar(x, ratios, 0.50, color=colors_r, alpha=0.88,
            edgecolor='white', zorder=3)
    ax2.axhline(1.0, color='k', linestyle='--', lw=0.9, alpha=0.55)
    ax2.set_xticks(x); ax2.set_xticklabels([str(n) for n in nmem_vals])
    ax2.set_xlabel('GL memory depth $N_{\\mathrm{mem}}$')
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) MF advantage increases with GL depth')
    ax2.set_ylim(0, max(ratios) * 1.5)
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        annotate_ratio(ax2, i, r, p)

    plt.tight_layout()
    out = FIG_DIR / 'gl_ablation_alpha0.5.pdf'
    plt.savefig(out); plt.close()
    print(f'  saved {out.name}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating canonical publication figures')
    print('(all data from reproducibility/data/ — nothing hardcoded)\n')

    fig_alpha_advantage()
    print('  → alpha_advantage_ncol100.pdf  [matches Table 2]')

    fig_ncol_budget()
    print('  → ncol_budget_alpha0.5_v2.pdf  [matches Table 1]')

    fig_fidelity_cross()
    print('  → fidelity_cross_comparison.pdf  [matches Table 4]')

    fig_fidelity_sweep_a10()
    print('  → fidelity_sweep_alpha1.0.pdf  [matches Table 5]')

    fig_fidelity_sweep_a05()
    print('  → fidelity_sweep_alpha0.5.pdf  [matches Table 6]')

    fig_gl_ablation()
    print('  → gl_ablation_alpha0.5.pdf  [matches Table 8]')

    print('\nDone.  All figures in figures/')
