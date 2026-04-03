#!/usr/bin/env python3
"""
Generate publication figures from verified experiment results.
All data hardcoded from log files (N_terms=12 unless noted).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

COLORS = {
    'MF': '#1f77b4',      # blue
    'Vanilla': '#d62728', # red
    'Van50': '#2ca02c',   # green
    'RAR': '#ff7f0e',     # orange
    'ref': 'k',
}

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)


# ============================================================
# DATA (from verified log files, N_terms=12 unless noted)
# ============================================================

# N_col budget (α=0.5, N_terms=12, 5 seeds)
ncol_a05 = {
    50:  {'mf': (14.16, 3.92), 'van': (14.23, 3.51)},
    100: {'mf': (10.25, 3.44), 'van': (9.75, 5.17), 'van50': (7.37, 4.26)},
    200: {'mf': (8.61, 2.34),  'van': (5.76, 2.97), 'van50': (3.59, 1.37)},
}

# N_col budget (α=1.0, N_terms=12, 5 seeds)
ncol_a10 = {
    100: {'mf': (22.01, 12.44), 'van': (23.16, 10.85)},
}

# N_col budget (α=1.0, N_terms=8, 10 seeds)
ncol_a10_nterms8 = {
    100: {'mf': (21.93, 9.69), 'van': (30.87, 14.01), 'van50': (26.09, 14.86)},
    200: {'mf': (20.37, 11.78), 'van': (18.97, 10.13)},
}

# Multi-α sweep (N_col=100, N_terms=12)
alpha_sweep = {
    0.5: {'mf': (10.25, 3.44), 'van': (9.75, 5.17),  'seeds': 5},
    0.7: {'mf': (8.43, 3.02),  'van': (12.26, 1.02),  'seeds': 5},
    1.0: {'mf': (22.01, 12.44),'van': (23.16, 10.85),  'seeds': 5},
}

# α=0.7 also has N_col=200
alpha07_ncol200 = {'mf': (7.94, 3.27), 'van': (9.70, 4.55)}

# RAR comparison (α=0.5, N_col=100, 5 seeds, N_terms=12)
rar_a05 = {
    100: {'mf': (10.25, 3.44), 'van': (9.75, 5.17), 'rar': (16.16, 4.28)},
}

# Fidelity cross-comparison (N_col=100)
fidelity_cross = {
    # (α, N_terms): (MF_mean, MF_std, Van_mean, Van_std, seeds)
    (0.5, 8):  (22.76, 9.86, 11.39, 5.08, 10),
    (0.5, 12): (12.01, 3.85, 11.39, 5.08, 10),
    (1.0, 8):  (25.84, 18.91, 23.16, 10.85, 5),
    (1.0, 10): (13.13, 5.90, 23.16, 10.85, 5),
    (1.0, 12): (20.39, 12.90, 23.16, 10.85, 5),
}

# Fidelity sweep at α=1.0, N_col=100
# N_terms 5,20: 5 seeds (Van=23.16); N_terms 8,10,12,15: 10 seeds (Van=30.87)
fidelity_sweep_a10 = {
    # N_terms: (MF_mean, MF_std, Van_mean, Van_std)
    5:  (176.29, 118.29, 23.16, 10.85),
    8:  (37.90, 33.83, 30.87, 14.01),
    10: (18.67, 9.11, 30.87, 14.01),
    12: (28.20, 20.39, 30.87, 14.01),
    15: (23.64, 10.09, 30.87, 14.01),
    20: (22.20, 6.48, 23.16, 10.85),
}

# Fidelity sweep at α=0.5, N_col=100
# N_terms 8,10: 5 seeds (Van=9.75); N_terms 12,15,20: 10 seeds (Van=11.39)
fidelity_sweep_a05 = {
    # N_terms: (MF_mean, MF_std, Van_mean, Van_std)
    8:  (20.94, 5.09, 9.75, 5.17),
    10: (12.69, 2.30, 9.75, 5.17),
    12: (12.01, 3.85, 11.39, 5.08),
    15: (6.56, 1.10, 11.39, 5.08),
    20: (3.66, 1.73, 11.39, 5.08),
}

# LF quality table
lf_quality = {
    # N_terms: {α: error%}
    5:  {0.3: 130.9, 0.5: 123.7, 0.7: 111.8, 0.9: 96.2, 1.0: 71.2},
    8:  {0.3: 51.2, 0.5: 40.8, 0.7: 40.5, 0.9: 33.2, 1.0: 25.2},
    10: {0.3: 32.9, 0.5: 25.3, 0.7: 25.2, 0.9: 20.2, 1.0: 15.3},
    12: {0.3: 21.7, 0.5: 16.6, 0.7: 16.1, 0.9: 12.7, 1.0: 10.0},
    15: {0.3: 12.4, 0.5: 9.2, 0.7: 8.9, 0.9: 7.0, 1.0: 5.5},
    20: {0.3: 5.1, 0.5: 3.7, 0.7: 3.5, 0.9: 2.7, 1.0: 2.2},
}


# ============================================================
# FIGURE 1: Multi-α advantage at N_col=100
# ============================================================
def plot_alpha_advantage():
    """Bar chart: MF vs Vanilla across α values at N_col=100."""
    alphas = [0.5, 0.7, 1.0]
    mf_means = [alpha_sweep[a]['mf'][0] for a in alphas]
    mf_stds  = [alpha_sweep[a]['mf'][1] for a in alphas]
    van_means = [alpha_sweep[a]['van'][0] for a in alphas]
    van_stds  = [alpha_sweep[a]['van'][1] for a in alphas]
    ratios = [v/m for v, m in zip(van_means, mf_means)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(alphas))
    w = 0.35
    ax1.bar(x - w/2, mf_means, w, yerr=mf_stds, label='MF-fPINN',
            color=COLORS['MF'], alpha=0.85, capsize=4, edgecolor='white')
    ax1.bar(x + w/2, van_means, w, yerr=van_stds, label='Vanilla fPINN',
            color=COLORS['Vanilla'], alpha=0.85, capsize=4, edgecolor='white')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'α = {a}' for a in alphas])
    ax1.set_ylabel('Relative $L^2$ error (%)')
    ax1.set_title('(a) MF vs Vanilla at $N_{\\mathrm{col}} = 100$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: advantage ratio
    colors_bar = ['#aaa' if r < 1 else COLORS['MF'] for r in ratios]
    bars = ax2.bar(x, ratios, 0.5, color=colors_bar, alpha=0.85, edgecolor='white')
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'α = {a}' for a in alphas])
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) MF advantage factor')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (r, a) in enumerate(zip(ratios, alphas)):
        label = f'{r:.2f}×'
        ax2.text(i, r + 0.03, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = fig_dir / 'alpha_advantage_ncol100.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 2: Fidelity gap interaction with α
# ============================================================
def plot_fidelity_cross():
    """2x2 comparison: N_terms=8 vs 12, α=0.5 vs 1.0."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for i, nterms in enumerate([8, 12]):
        ax = axes[i]
        alphas = [0.5, 1.0]
        x = np.arange(len(alphas))
        w = 0.35

        mf_m = [fidelity_cross[(a, nterms)][0] for a in alphas]
        mf_s = [fidelity_cross[(a, nterms)][1] for a in alphas]
        van_m = [fidelity_cross[(a, nterms)][2] for a in alphas]
        van_s = [fidelity_cross[(a, nterms)][3] for a in alphas]

        ax.bar(x - w/2, mf_m, w, yerr=mf_s, label='MF-fPINN',
               color=COLORS['MF'], alpha=0.85, capsize=4, edgecolor='white')
        ax.bar(x + w/2, van_m, w, yerr=van_s, label='Vanilla',
               color=COLORS['Vanilla'], alpha=0.85, capsize=4, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([f'α = {a}' for a in alphas])
        ax.set_title(f'({"ab"[i]}) $N_{{\\mathrm{{terms}}}} = {nterms}$ '
                     f'(LF error: {lf_quality[nterms][0.5]:.0f}%/{lf_quality[nterms][1.0]:.0f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        if i == 0:
            ax.set_ylabel('Relative $L^2$ error (%)')
        ax.legend()

        # Annotate ratios
        for j, a in enumerate(alphas):
            ratio = van_m[j] / mf_m[j]
            label = f'{ratio:.2f}×'
            y_pos = max(mf_m[j] + mf_s[j], van_m[j] + van_s[j]) + 1
            color = COLORS['MF'] if ratio > 1.1 else ('red' if ratio < 0.9 else 'gray')
            ax.text(j, y_pos, label, ha='center', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    path = fig_dir / 'fidelity_cross_comparison.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 3: RAR comparison (α=0.5, N_col=100)
# ============================================================
def plot_rar_comparison():
    """Bar chart: MF vs Vanilla vs RAR at α=0.5, N_col=100."""
    data = rar_a05[100]
    methods = ['MF-fPINN', 'Vanilla', 'RAR']
    means = [data['mf'][0], data['van'][0], data['rar'][0]]
    stds  = [data['mf'][1], data['van'][1], data['rar'][1]]
    colors = [COLORS['MF'], COLORS['Vanilla'], COLORS['RAR']]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, 0.5, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Relative $L^2$ error (%)')
    ax.set_title('$\\alpha = 0.5$, $N_{\\mathrm{col}} = 100$, $N_{\\mathrm{terms}} = 12$')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f'{m:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = fig_dir / 'rar_comparison_alpha0.5.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 4: GL memory mechanism illustration
# ============================================================
def plot_gl_memory_mechanism():
    """Illustrate effective constraints per collocation point vs α."""
    alphas = np.linspace(0.1, 1.0, 50)
    N_mem = 100

    # GL weights decay: w_k ~ k^{-(1+α)} for large k
    # Effective constraints ≈ N_mem for α < 1, ≈ 1 for α = 1
    # More precisely, the number of non-negligible weights
    def effective_constraints(alpha, N_mem=100):
        if alpha >= 0.999:
            return 1.0
        weights = np.zeros(N_mem + 1)
        weights[0] = 1.0
        for k in range(1, N_mem + 1):
            weights[k] = weights[k-1] * (1 - (1 + alpha) / k)
        # Count weights > 1% of max
        threshold = 0.01 * np.max(np.abs(weights))
        return np.sum(np.abs(weights) > threshold)

    eff = [effective_constraints(a) for a in alphas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: effective constraints vs α
    ax1.plot(alphas, eff, 'b-', linewidth=2)
    ax1.set_xlabel('Fractional order $\\alpha$')
    ax1.set_ylabel('Effective temporal constraints per point')
    ax1.set_title('(a) GL memory depth vs $\\alpha$')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Classical ($\\alpha=1$)')
    ax1.legend()

    # Right: total effective constraints at N_col=100
    N_col = 100
    total = [N_col * e for e in eff]
    ax2.plot(alphas, total, 'b-', linewidth=2)
    ax2.set_xlabel('Fractional order $\\alpha$')
    ax2.set_ylabel('Total spatiotemporal constraints')
    ax2.set_title(f'(b) Total constraints at $N_{{\\mathrm{{col}}}} = {N_col}$')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=N_col, color='r', linestyle='--', alpha=0.5, label=f'$N_{{\\mathrm{{col}}}} = {N_col}$')
    ax2.legend()
    ax2.set_yscale('log')

    plt.tight_layout()
    path = fig_dir / 'gl_memory_mechanism.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 5: LF quality heatmap
# ============================================================
def plot_lf_quality_heatmap():
    """Heatmap of LF error vs N_terms and α."""
    nterms_list = [5, 8, 10, 12, 15, 20]
    alpha_list = [0.3, 0.5, 0.7, 0.9, 1.0]

    data = np.array([[lf_quality[nt][a] for a in alpha_list] for nt in nterms_list])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(alpha_list)))
    ax.set_xticklabels([f'{a}' for a in alpha_list])
    ax.set_yticks(range(len(nterms_list)))
    ax.set_yticklabels([str(n) for n in nterms_list])
    ax.set_xlabel('Fractional order $\\alpha$')
    ax.set_ylabel('$N_{\\mathrm{terms}}$ (Laplace inversion)')

    # Annotate each cell
    for i in range(len(nterms_list)):
        for j in range(len(alpha_list)):
            val = data[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='LF $L^2$ error (%)', shrink=0.8)
    ax.set_title('Low-fidelity data quality')
    plt.tight_layout()
    path = fig_dir / 'lf_quality_heatmap.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 6: N_col budget α=0.5 (with Van+50 and RAR)
# ============================================================
def plot_ncol_budget_a05():
    """N_col budget study at α=0.5 with all baselines."""
    ncols = [50, 100, 200]
    mf_m = [ncol_a05[n]['mf'][0] for n in ncols]
    mf_s = [ncol_a05[n]['mf'][1] for n in ncols]
    van_m = [ncol_a05[n]['van'][0] for n in ncols]
    van_s = [ncol_a05[n]['van'][1] for n in ncols]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(ncols, mf_m, yerr=mf_s, fmt='o-', color=COLORS['MF'],
                label='MF-fPINN', capsize=4, markersize=8)
    ax.errorbar(ncols, van_m, yerr=van_s, fmt='s-', color=COLORS['Vanilla'],
                label='Vanilla fPINN', capsize=4, markersize=8)

    # Van+50 where available
    van50_ncols = [n for n in ncols if 'van50' in ncol_a05[n]]
    if van50_ncols:
        v50_m = [ncol_a05[n]['van50'][0] for n in van50_ncols]
        v50_s = [ncol_a05[n]['van50'][1] for n in van50_ncols]
        ax.errorbar(van50_ncols, v50_m, yerr=v50_s,
                    fmt='^-', color=COLORS['Van50'], label='Vanilla+50',
                    capsize=4, markersize=8)

    # RAR at N_col=100
    if 100 in rar_a05:
        ax.errorbar([100], [rar_a05[100]['rar'][0]],
                    yerr=[rar_a05[100]['rar'][1]],
                    fmt='D', color=COLORS['RAR'], label='RAR',
                    capsize=4, markersize=8)

    ax.set_xlabel('$N_{\\mathrm{col}}$ (collocation points)')
    ax.set_ylabel('Relative $L^2$ error (%)')
    ax.set_title('$\\alpha = 0.5$, $N_{\\mathrm{terms}} = 12$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(ncols)
    ax.set_xticklabels([str(n) for n in ncols])

    plt.tight_layout()
    path = fig_dir / 'ncol_budget_alpha0.5_v2.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 7: Fidelity sweep at α=1.0
# ============================================================
def plot_fidelity_sweep_a10():
    """Line plot: MF error vs LF quality at α=1.0."""
    nterms_vals = sorted(fidelity_sweep_a10.keys())
    lf_errors = [lf_quality[n][1.0] for n in nterms_vals]
    mf_means = [fidelity_sweep_a10[n][0] for n in nterms_vals]
    mf_stds = [fidelity_sweep_a10[n][1] for n in nterms_vals]
    # Use 10-seed Vanilla baseline (30.87) for consistency with paper Table 5
    van_mean_10s = 30.87
    van_std_10s = 14.01

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: MF error vs LF error
    ax1.errorbar(lf_errors, mf_means, yerr=mf_stds, fmt='o-',
                 color=COLORS['MF'], label='MF-fPINN', capsize=4, linewidth=2)
    ax1.axhline(y=van_mean_10s, color=COLORS['Vanilla'], linestyle='--',
                linewidth=1.5, label=f'Vanilla ({van_mean_10s:.1f}%, 10s)')
    ax1.fill_between([0, 100], van_mean_10s - van_std_10s, van_mean_10s + van_std_10s,
                     color=COLORS['Vanilla'], alpha=0.15)

    for i, n in enumerate(nterms_vals):
        ax1.annotate(f'$N_{{\\mathrm{{terms}}}}={n}$',
                     (lf_errors[i], mf_means[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=8, ha='center')

    ax1.set_xlabel('LF data error (%)')
    ax1.set_ylabel('MF-fPINN $L^2$ error (%)')
    ax1.set_title('(a) MF error vs LF quality ($\\alpha = 1.0$)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, max(50, max(mf_means) * 0.5))

    # Right: advantage ratio (use per-entry Vanilla baseline)
    ratios = [fidelity_sweep_a10[n][2] / fidelity_sweep_a10[n][0] for n in nterms_vals]
    colors_bar = [COLORS['MF'] if r > 1.1 else ('red' if r < 0.9 else '#aaa') for r in ratios]
    bars = ax2.bar(range(len(nterms_vals)), ratios, color=colors_bar, alpha=0.85, edgecolor='white')
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(range(len(nterms_vals)))
    ax2.set_xticklabels([f'$N_{{t}}={n}$\n({lf_quality[n][1.0]:.0f}%)' for n in nterms_vals])
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) Non-monotonic advantage')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, r in enumerate(ratios):
        label = f'{r:.2f}×'
        y_off = 0.05 if r > 0 else -0.1
        ax2.text(i, max(r + y_off, 0.05), label, ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = fig_dir / 'fidelity_sweep_alpha1.0.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 8: Fidelity sweep at α=0.5
# ============================================================
def plot_fidelity_sweep_a05():
    """Line plot: MF error vs LF quality at α=0.5."""
    nterms_vals = sorted(fidelity_sweep_a05.keys())
    lf_errors = [lf_quality[n][0.5] for n in nterms_vals]
    mf_means = [fidelity_sweep_a05[n][0] for n in nterms_vals]
    mf_stds = [fidelity_sweep_a05[n][1] for n in nterms_vals]
    van_mean = fidelity_sweep_a05[8][2]
    van_std = fidelity_sweep_a05[8][3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.errorbar(lf_errors, mf_means, yerr=mf_stds, fmt='o-',
                 color=COLORS['MF'], label='MF-fPINN', capsize=4, linewidth=2)
    ax1.axhline(y=van_mean, color=COLORS['Vanilla'], linestyle='--',
                linewidth=1.5, label=f'Vanilla ({van_mean:.1f}%)')
    ax1.fill_between([0, 55], van_mean - van_std, van_mean + van_std,
                     color=COLORS['Vanilla'], alpha=0.15)

    for i, n in enumerate(nterms_vals):
        ax1.annotate(f'$N_{{\\mathrm{{terms}}}}={n}$',
                     (lf_errors[i], mf_means[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=8, ha='center')

    ax1.set_xlabel('LF data error (%)')
    ax1.set_ylabel('MF-fPINN $L^2$ error (%)')
    ax1.set_title('(a) MF error vs LF quality ($\\alpha = 0.5$)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 55)

    # Right: advantage ratio
    ratios = [van_mean / m for m in mf_means]
    colors_bar = [COLORS['MF'] if r > 1.1 else ('red' if r < 0.9 else '#aaa') for r in ratios]
    bars = ax2.bar(range(len(nterms_vals)), ratios, color=colors_bar, alpha=0.85, edgecolor='white')
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(range(len(nterms_vals)))
    ax2.set_xticklabels([f'$N_{{t}}={n}$\n({lf_quality[n][0.5]:.0f}%)' for n in nterms_vals])
    ax2.set_ylabel('Advantage ratio (Vanilla / MF)')
    ax2.set_title('(b) Monotonic advantage at $\\alpha = 0.5$')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, r in enumerate(ratios):
        label = f'{r:.2f}×'
        y_off = 0.05
        ax2.text(i, r + y_off, label, ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = fig_dir / 'fidelity_sweep_alpha0.5.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 9: BVTT decomposition (α=0.5, N_col=100)
# ============================================================
def plot_bvtt_decomposition():
    """Stacked bar: Bias² vs Variance for each method."""
    import json
    with open('results/fdr/bias_variance_alpha0.5_ncol100.json') as f:
        d = json.load(f)

    methods = ['vanilla', 'MF_Nterms8', 'MF_Nterms12', 'MF_Nterms15', 'MF_Nterms20']
    labels = ['Vanilla', 'MF $N_t{=}8$\n(41%)', 'MF $N_t{=}12$\n(17%)',
              'MF $N_t{=}15$\n(9%)', 'MF $N_t{=}20$\n(4%)']

    bias_sq = [d[m]['bias_sq'] * 1e4 for m in methods]
    variance = [d[m]['variance'] * 1e4 for m in methods]
    bias_frac = [d[m]['bias_frac'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(methods))
    w = 0.6
    ax1.bar(x, bias_sq, w, label='Bias$^2$', color='#e74c3c', alpha=0.85, edgecolor='white')
    ax1.bar(x, variance, w, bottom=bias_sq, label='Variance',
            color='#3498db', alpha=0.85, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('MSE components ($\\times 10^{-4}$)')
    ax1.set_title('(a) Bias-variance decomposition')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: fraction plot
    ax2.bar(x, [bf * 100 for bf in bias_frac], w,
            label='Bias$^2$/MSE', color='#e74c3c', alpha=0.85, edgecolor='white')
    ax2.bar(x, [(1 - bf) * 100 for bf in bias_frac], w,
            bottom=[bf * 100 for bf in bias_frac],
            label='Var/MSE', color='#3498db', alpha=0.85, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Fraction of MSE (%)')
    ax2.set_title('(b) Bias vs variance dominance')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='k', linestyle=':', alpha=0.3)

    for i, bf in enumerate(bias_frac):
        ax2.text(i, 50, f'{bf:.0%}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    plt.tight_layout()
    path = fig_dir / 'bvtt_decomposition_alpha0.5.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# FIGURE 10: Phase 2 stability curves
# ============================================================
def plot_phase2_stability():
    """Phase 2 error trajectory for α=0.5 vs α=1.0."""
    import json

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, alpha, title_letter in [(ax1, 0.5, 'a'), (ax2, 1.0, 'b')]:
        fn = f'results/fdr/stability_alpha{alpha}_nterms20_ncol100.json'
        with open(fn) as f:
            d = json.load(f)

        epochs = []
        means = []
        stds = []
        for k in sorted(d.keys()):
            if k.startswith('p2_') and isinstance(d[k], dict):
                ep = int(k.split('_')[1])
                epochs.append(ep)
                means.append(d[k]['l2_mean'])
                stds.append(d[k].get('l2_std', 0))

        # Sort by epochs
        order = np.argsort(epochs)
        epochs = [epochs[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]

        ax.errorbar(epochs, means, yerr=stds, fmt='o-', color=COLORS['MF'],
                    capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel('Phase 2 epochs')
        ax.set_ylabel('$L^2$ error (%)')
        Keff = 44 if alpha == 0.5 else 2
        ax.set_title(f'({title_letter}) $\\alpha = {alpha}$, $K_{{\\mathrm{{eff}}}} = {Keff}$')
        ax.grid(True, alpha=0.3)

        # Highlight transient degradation for α=1.0
        if alpha == 1.0 and len(means) > 3:
            peak_idx = np.argmax(means[1:]) + 1  # skip epoch 0
            if means[peak_idx] > means[1] * 1.1:  # significant increase
                ax.annotate('transient\ndegradation',
                           (epochs[peak_idx], means[peak_idx]),
                           textcoords="offset points", xytext=(30, 10),
                           fontsize=9, color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    path = fig_dir / 'phase2_stability.pdf'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Generating publication figures (v2 — verified data)...")
    print()

    print("Figure 1: Multi-α advantage")
    plot_alpha_advantage()

    print("Figure 2: Fidelity gap × α interaction")
    plot_fidelity_cross()

    print("Figure 3: RAR comparison")
    plot_rar_comparison()

    print("Figure 4: GL memory mechanism")
    plot_gl_memory_mechanism()

    print("Figure 5: LF quality heatmap")
    plot_lf_quality_heatmap()

    print("Figure 6: N_col budget α=0.5")
    plot_ncol_budget_a05()

    print("Figure 7: Fidelity sweep α=1.0")
    plot_fidelity_sweep_a10()

    print("Figure 8: Fidelity sweep α=0.5")
    plot_fidelity_sweep_a05()

    print("Figure 9: BVTT decomposition")
    plot_bvtt_decomposition()

    print("Figure 10: Phase 2 stability")
    plot_phase2_stability()

    print("\nDone! All figures in figures/")
