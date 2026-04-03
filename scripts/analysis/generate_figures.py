#!/usr/bin/env python3
"""
Generate all publication figures from completed experiment results.

Figures:
  1. N_col budget study (α=0.5) — KEY figure showing MF advantage at low N_col
  2. Transfer efficiency curve (α=0.5) — L2 vs N_LF
  3. N_col budget study (α=1.0) — if available
  4. Solution profiles — best model vs reference
"""

import json
import numpy as np
import torch
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
    'ref': 'k',
}

fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)
res_dir = Path('results/fdr')


def plot_ncol_budget(alpha=0.5):
    """Bar chart + line: MF vs Vanilla at different N_col."""
    path = res_dir / f'ncol_sweep_alpha{alpha}.json'
    if not path.exists():
        print(f"  Skipping N_col budget (α={alpha}): {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    # Extract
    n_cols = sorted(set(v['n_col'] for v in data.values()))
    mf_means = [data[f'MF_N{n}']['l2_mean'] * 100 for n in n_cols]
    mf_stds = [data[f'MF_N{n}']['l2_std'] * 100 for n in n_cols]
    van_means = [data[f'Vanilla_N{n}']['l2_mean'] * 100 for n in n_cols]
    van_stds = [data[f'Vanilla_N{n}']['l2_std'] * 100 for n in n_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: L2 error comparison
    x = np.arange(len(n_cols))
    w = 0.35
    ax1.bar(x - w/2, mf_means, w, yerr=mf_stds, label='MF-fPINN (M3)',
            color=COLORS['MF'], alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
    ax1.bar(x + w/2, van_means, w, yerr=van_stds, label='Vanilla fPINN (M1)',
            color=COLORS['Vanilla'], alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_cols])
    ax1.set_xlabel('$N_{\\mathrm{col}}$ (collocation points)')
    ax1.set_ylabel('Relative $L^2$ error (%)')
    ax1.set_title(f'(a) Accuracy vs. collocation budget ($\\alpha = {alpha}$)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Right: Speedup ratio
    ratios = [v / max(m, 1e-8) for v, m in zip(van_means, mf_means)]
    ax2.plot(n_cols, ratios, 'o-', color='#2ca02c', lw=2, markersize=8, zorder=5)
    ax2.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.fill_between(n_cols, 1, ratios, alpha=0.15, color='#2ca02c')

    # Annotate
    for i, (nc, r) in enumerate(zip(n_cols, ratios)):
        if r > 1.2:
            ax2.annotate(f'{r:.1f}×', (nc, r), textcoords='offset points',
                        xytext=(0, 12), ha='center', fontsize=10, fontweight='bold',
                        color='#2ca02c')

    ax2.set_xlabel('$N_{\\mathrm{col}}$ (collocation points)')
    ax2.set_ylabel('Vanilla / MF-fPINN error ratio')
    ax2.set_title(f'(b) MF-fPINN advantage factor')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = fig_dir / f'ncol_budget_alpha{alpha}.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_transfer_curve(alpha=0.5):
    """L2 error vs N_LF with vanilla baseline."""
    path = res_dir / f'transfer_curve_alpha{alpha}.json'
    if not path.exists():
        print(f"  Skipping transfer curve: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    n_lf_values = sorted([int(k) for k in data['transfer'].keys()])
    means = [data['transfer'][str(n)]['mean'] * 100 for n in n_lf_values]
    stds = [data['transfer'][str(n)]['std'] * 100 for n in n_lf_values]

    ax.errorbar(n_lf_values, means, yerr=stds, fmt='s-', color=COLORS['MF'],
                capsize=4, label='MF-fPINN', zorder=5, markersize=7)

    van_mean = data['vanilla']['mean'] * 100
    van_std = data['vanilla']['std'] * 100
    ax.axhline(van_mean, color=COLORS['Vanilla'], ls='--', lw=1.5,
               label=f'Vanilla fPINN ({van_mean:.2f}%)')
    ax.axhspan(van_mean - van_std, van_mean + van_std,
               color=COLORS['Vanilla'], alpha=0.1)

    ax.set_xlabel('$N_{\\mathrm{LF}}$ (low-fidelity Laplace data points)')
    ax.set_ylabel('Relative $L^2$ error (%)')
    ax.set_title(f'Transfer efficiency ($\\alpha = {alpha}$)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = fig_dir / f'transfer_curve_alpha{alpha}.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_solution_comparison(alpha=0.5):
    """Solution profiles from best saved model vs reference."""
    ref_path = res_dir / f'reference_alpha{alpha}.npz'
    model_mf = res_dir / f'M3_mf_alpha{alpha}_seed0.pt'
    model_van = res_dir / f'M1_vanilla_alpha{alpha}_seed0.pt'

    if not ref_path.exists():
        print(f"  Skipping solution profiles: reference not found")
        return

    ref_data = dict(np.load(ref_path, allow_pickle=True))

    from mf_fpinn.models.fdr_pinn import FDRNet
    from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC

    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    t_values = ref_data['t']
    x = ref_data['x']

    n_times = len(t_values)
    fig, axes = plt.subplots(1, n_times, figsize=(3.5 * n_times, 3.5), sharey=True)
    if n_times == 1:
        axes = [axes]

    models_to_plot = []
    for label, path, color, ls in [
        ('MF-fPINN', model_mf, COLORS['MF'], '--'),
        ('Vanilla', model_van, COLORS['Vanilla'], ':'),
    ]:
        if path.exists():
            net = FDRNet(
                n_layers=config['n_layers'], n_neurons=config['n_neurons'],
                activation=config['activation'], L=config['L'], T=config['T'],
                hard_bc=True, bc_type=config['bc_type'],
                bc_params=config['bc_params'],
                fourier_features=config['fourier_features'],
                fourier_sigma=config['fourier_sigma'],
            )
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                net.load_state_dict(ckpt['model_state_dict'])
            else:
                net.load_state_dict(ckpt)
            net.eval()
            models_to_plot.append((label, net, color, ls))

    for j, t_val in enumerate(t_values):
        ax = axes[j]
        ax.plot(x, ref_data['u'][:, j], 'k-', lw=2, label='Reference', zorder=10)

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        t_t = torch.full_like(x_t, float(t_val))

        for label, net, color, ls in models_to_plot:
            with torch.no_grad():
                u_pred = net(x_t, t_t).numpy().flatten()
            ax.plot(x, u_pred, ls, color=color, lw=1.5, label=label, alpha=0.8)

        ax.set_xlabel('$x$')
        ax.set_title(f'$t = {float(t_val):.2f}$')
        if j == 0:
            ax.set_ylabel('$u(x, t)$')
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc='upper right')
    fig.suptitle(f'Solution profiles ($\\alpha = {alpha}$)', y=1.02)
    plt.tight_layout()
    save_path = fig_dir / f'solution_profiles_alpha{alpha}.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_multi_alpha_sweep():
    """Multi-α advantage: bar chart showing MF/Vanilla ratio across α values."""
    path = res_dir / 'alpha_sweep.json'
    if not path.exists():
        print(f"  Skipping multi-α sweep: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: L2 errors at N_col=100
    x = np.arange(len(alphas))
    w = 0.35
    mf_100 = [data.get(f'alpha{a}_MF_N100', {}).get('l2_mean', 0) * 100 for a in alphas]
    mf_100_std = [data.get(f'alpha{a}_MF_N100', {}).get('l2_std', 0) * 100 for a in alphas]
    van_100 = [data.get(f'alpha{a}_Vanilla_N100', {}).get('l2_mean', 0) * 100 for a in alphas]
    van_100_std = [data.get(f'alpha{a}_Vanilla_N100', {}).get('l2_std', 0) * 100 for a in alphas]

    ax1.bar(x - w/2, mf_100, w, yerr=mf_100_std, label='MF-fPINN',
            color=COLORS['MF'], alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
    ax1.bar(x + w/2, van_100, w, yerr=van_100_std, label='Vanilla fPINN',
            color=COLORS['Vanilla'], alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{a}' for a in alphas])
    ax1.set_xlabel(r'Fractional order $\alpha$')
    ax1.set_ylabel('Relative $L^2$ error (%)')
    ax1.set_title(r'(a) Accuracy at $N_{\mathrm{col}} = 100$')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Right: advantage ratio across N_col for each α
    n_cols_sweep = [100, 200, 500]
    markers = ['o', 's', '^', 'D', 'v']
    for i, a in enumerate(alphas):
        ratios = []
        for nc in n_cols_sweep:
            mf_key = f'alpha{a}_MF_N{nc}'
            van_key = f'alpha{a}_Vanilla_N{nc}'
            if mf_key in data and van_key in data:
                r = data[van_key]['l2_mean'] / max(data[mf_key]['l2_mean'], 1e-8)
                ratios.append(r)
            else:
                ratios.append(1.0)
        ax2.plot(n_cols_sweep, ratios, f'{markers[i]}-', label=f'$\\alpha={a}$',
                 markersize=7, lw=1.5)

    ax2.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel('$N_{\\mathrm{col}}$ (collocation points)')
    ax2.set_ylabel('Vanilla / MF-fPINN error ratio')
    ax2.set_title('(b) MF advantage vs. collocation budget')
    ax2.set_xscale('log')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = fig_dir / 'fig6_multi_alpha_sweep.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_noise_robustness():
    """Noise robustness: MF L2 error vs noise level."""
    path = res_dir / 'noise_robustness_alpha0.5_ncol100.json'
    if not path.exists():
        print(f"  Skipping noise robustness: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    noise_levels = [0.0, 0.01, 0.05, 0.1]  # skip 0.2 (catastrophic)
    mf_means = []
    mf_stds = []
    for nl in noise_levels:
        key = f'MF_noise{nl}'
        if key in data:
            mf_means.append(data[key]['l2_mean'] * 100)
            mf_stds.append(data[key]['l2_std'] * 100)
        else:
            mf_means.append(np.nan)
            mf_stds.append(0)

    # Vanilla baseline
    van_mean = data['vanilla']['l2_mean'] * 100
    van_std = data['vanilla']['l2_std'] * 100

    ax.errorbar([n * 100 for n in noise_levels], mf_means, yerr=mf_stds,
                fmt='s-', color=COLORS['MF'], capsize=4, label='MF-fPINN', markersize=7, zorder=5)
    ax.axhline(van_mean, color=COLORS['Vanilla'], ls='--', lw=1.5,
               label=f'Vanilla fPINN ({van_mean:.1f}%)')
    ax.axhspan(van_mean - van_std, van_mean + van_std,
               color=COLORS['Vanilla'], alpha=0.1)

    # Annotate advantage factors
    for i, nl in enumerate(noise_levels):
        if not np.isnan(mf_means[i]) and mf_means[i] > 0:
            ratio = van_mean / mf_means[i]
            ax.annotate(f'{ratio:.1f}×', (nl * 100, mf_means[i]),
                       textcoords='offset points', xytext=(12, -5),
                       fontsize=9, color=COLORS['MF'])

    ax.set_xlabel('Noise level in LF data (%)')
    ax.set_ylabel('Relative $L^2$ error (%)')
    ax.set_title(r'Noise robustness ($\alpha = 0.5$, $N_{\mathrm{col}} = 100$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = fig_dir / 'fig7_noise_robustness.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_inverse_problem():
    """Inverse problem results: estimated vs true parameters."""
    path = res_dir / 'inverse_fixed.json'
    if not path.exists():
        print(f"  Skipping inverse problem: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    results = data['results']
    true_alpha = data['true_alpha']
    true_kappa = data['true_kappa']

    alpha_vals = [r['alpha'] for r in results]
    kappa_vals = [r['kappa'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # α estimation
    ax1.axhline(true_alpha, color='k', ls='--', lw=1.5, label=f'True $\\alpha = {true_alpha}$')
    ax1.scatter(range(1, len(alpha_vals)+1), alpha_vals, c=COLORS['MF'],
                s=80, zorder=5, edgecolors='white', linewidth=0.8)
    ax1.axhline(np.mean(alpha_vals), color=COLORS['MF'], ls=':', lw=1,
                label=f'Mean = {np.mean(alpha_vals):.3f}')
    ax1.set_xlabel('Seed')
    ax1.set_ylabel(r'Estimated $\alpha$')
    ax1.set_title(f'(a) Fractional order estimation')
    ax1.legend(fontsize=9)
    ax1.set_ylim(true_alpha - 0.1, true_alpha + 0.1)
    ax1.grid(True, alpha=0.3)

    # κ estimation
    ax2.axhline(true_kappa, color='k', ls='--', lw=1.5, label=f'True $\\kappa = {true_kappa}$')
    ax2.scatter(range(1, len(kappa_vals)+1), kappa_vals, c=COLORS['MF'],
                s=80, zorder=5, edgecolors='white', linewidth=0.8)
    ax2.axhline(np.mean(kappa_vals), color=COLORS['MF'], ls=':', lw=1,
                label=f'Mean = {np.mean(kappa_vals):.3f}')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel(r'Estimated $\kappa$')
    ax2.set_title(f'(b) Reaction rate estimation')
    ax2.legend(fontsize=9)
    ax2.set_ylim(true_kappa - 0.5, true_kappa + 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = fig_dir / 'fig8_inverse_problem.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_combined_ncol_both_alpha():
    """Combined figure: N_col sweep for both α=0.5 and α=1.0 (if available)."""
    path_05 = res_dir / 'ncol_sweep_alpha0.5.json'
    path_10 = res_dir / 'ncol_sweep_alpha1.0.json'

    if not path_05.exists():
        return
    if not path_10.exists():
        print("  α=1.0 N_col sweep not yet available, skipping combined plot")
        return

    with open(path_05) as f:
        data_05 = json.load(f)
    with open(path_10) as f:
        data_10 = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, data, alpha_val, panel in [(ax1, data_05, 0.5, '(a)'), (ax2, data_10, 1.0, '(b)')]:
        n_cols = sorted(set(v['n_col'] for v in data.values()))
        mf_means = [data[f'MF_N{n}']['l2_mean'] * 100 for n in n_cols]
        mf_stds = [data[f'MF_N{n}']['l2_std'] * 100 for n in n_cols]
        van_means = [data[f'Vanilla_N{n}']['l2_mean'] * 100 for n in n_cols]
        van_stds = [data[f'Vanilla_N{n}']['l2_std'] * 100 for n in n_cols]

        ax.errorbar(n_cols, mf_means, yerr=mf_stds, fmt='s-', color=COLORS['MF'],
                    capsize=4, label='MF-fPINN (M3)', markersize=7, zorder=5)
        ax.errorbar(n_cols, van_means, yerr=van_stds, fmt='o-', color=COLORS['Vanilla'],
                    capsize=4, label='Vanilla fPINN (M1)', markersize=7, zorder=5)

        ax.set_xlabel('$N_{\\mathrm{col}}$ (collocation points)')
        ax.set_ylabel('Relative $L^2$ error (%)')
        ax.set_title(f'{panel} $\\alpha = {alpha_val}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = fig_dir / 'ncol_budget_combined.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_fidelity_sweep(alpha=0.5):
    """KEY FIGURE: MF-fPINN error vs LF data quality (fidelity gap)."""
    import glob
    paths = glob.glob(str(res_dir / f'fidelity_sweep_alpha{alpha}_*.json'))
    if not paths:
        print(f"  Skipping fidelity sweep: no results found")
        return

    path = sorted(paths)[-1]  # latest
    with open(path) as f:
        data = json.load(f)

    vanilla_mean = data['vanilla']['l2_mean'] * 100
    vanilla_std = data['vanilla']['l2_std'] * 100

    n_terms_list = []
    lf_errors = []
    mf_means = []
    mf_stds = []
    advantages = []

    for key, val in sorted(data.items()):
        if key.startswith('MF_Nterms'):
            n_terms_list.append(val['n_terms'])
            lf_errors.append(val['lf_error'] * 100)
            mf_means.append(val['l2_mean'] * 100)
            mf_stds.append(val['l2_std'] * 100)
            advantages.append(val['advantage'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: MF error vs LF error
    ax1.errorbar(lf_errors, mf_means, yerr=mf_stds, fmt='s-', color=COLORS['MF'],
                 capsize=4, label='MF-fPINN', markersize=8, zorder=5, lw=2)
    ax1.axhline(vanilla_mean, color=COLORS['Vanilla'], ls='--', lw=2,
                label=f'Vanilla ({vanilla_mean:.1f}%)')
    ax1.axhspan(vanilla_mean - vanilla_std, vanilla_mean + vanilla_std,
                color=COLORS['Vanilla'], alpha=0.1)

    # Mark beneficial vs detrimental regions
    ax1.axvspan(0, 30, color='green', alpha=0.05, label='MF beneficial')
    ax1.axvspan(30, max(lf_errors) + 10, color='red', alpha=0.05, label='MF detrimental')

    # Annotate N_terms values
    for i, nt in enumerate(n_terms_list):
        offset = (0, 10) if mf_means[i] < vanilla_mean else (0, -15)
        ax1.annotate(f'$N_{{\\mathrm{{terms}}}}={nt}$',
                     (lf_errors[i], mf_means[i]),
                     textcoords='offset points', xytext=offset,
                     fontsize=8, ha='center')

    ax1.set_xlabel('LF data error (% relative $L^2$)')
    ax1.set_ylabel('MF-fPINN error (% relative $L^2$)')
    ax1.set_title(f'(a) MF accuracy vs. LF quality ($\\alpha={alpha}$, $N_{{\\mathrm{{col}}}}=100$)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Advantage factor vs LF error
    ax2.plot(lf_errors, advantages, 'o-', color=COLORS['MF'], markersize=8, lw=2)
    ax2.axhline(1.0, color='gray', ls='--', lw=1.5, alpha=0.5)
    ax2.fill_between([0, max(lf_errors) + 10], [1, 1], [0, 0],
                     color='red', alpha=0.05, label='MF worse')
    ax2.fill_between([0, max(lf_errors) + 10], [1, 1], [10, 10],
                     color='green', alpha=0.05, label='MF better')

    for i, nt in enumerate(n_terms_list):
        ax2.annotate(f'$N_{{\\mathrm{{terms}}}}={nt}$',
                     (lf_errors[i], advantages[i]),
                     textcoords='offset points', xytext=(10, 0),
                     fontsize=8)

    ax2.set_xlabel('LF data error (% relative $L^2$)')
    ax2.set_ylabel('Advantage factor (Vanilla / MF error)')
    ax2.set_title('(b) MF advantage factor')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = fig_dir / f'fig_fidelity_sweep_alpha{alpha}.pdf'
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'))
    print(f"  Saved: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    print("Generating publication figures...\n")

    print("0. Fidelity gap sweep (KEY figure)")
    plot_fidelity_sweep(alpha=0.5)

    print("\n1. N_col budget study (α=0.5)")
    plot_ncol_budget(alpha=0.5)

    print("\n2. Transfer efficiency curve (α=0.5)")
    plot_transfer_curve(alpha=0.5)

    print("\n3. Solution profiles (α=0.5)")
    plot_solution_comparison(alpha=0.5)

    print("\n4. Solution profiles (α=1.0)")
    plot_solution_comparison(alpha=1.0)

    print("\n5. N_col budget study (α=1.0)")
    plot_ncol_budget(alpha=1.0)

    print("\n6. Combined N_col (both α)")
    plot_combined_ncol_both_alpha()

    print("\n7. Multi-α sweep")
    plot_multi_alpha_sweep()

    print("\n8. Noise robustness")
    plot_noise_robustness()

    print("\n9. Inverse problem")
    plot_inverse_problem()

    print("\nDone! Figures saved to figures/")
