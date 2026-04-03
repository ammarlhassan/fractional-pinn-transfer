#!/usr/bin/env python3
"""Generate convergence speed plot from existing tracking data."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

def load_tracking(path):
    with open(path) as f:
        return json.load(f)

def extract_curves(data, method):
    """Extract epoch-L2 curves from multi-seed tracking data."""
    seeds = data[method]
    all_epochs = []
    all_l2 = []
    for s in seeds:
        epochs = np.array([p['epoch'] for p in s])
        l2 = np.array([p['l2'] for p in s])
        all_epochs.append(epochs)
        all_l2.append(l2)
    return all_epochs, all_l2

def plot_convergence():
    base = Path('results/fdr')

    # Load N_terms=12 tracking data
    d12 = load_tracking(base / 'convergence_tracking_alpha0.5_nterms12_ncol100.json')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): N_terms=12 convergence curves ---
    ax = axes[0]

    for method, color, label in [('Vanilla', '#2196F3', 'Vanilla fPINN'),
                                  ('MF', '#E91E63', 'MF-fPINN')]:
        all_epochs, all_l2 = extract_curves(d12, method)

        # For MF, Phase 2 starts at epoch 0 (post Phase 1)
        # For Vanilla, epochs start at 1000
        # Align: MF epoch 0 = "start of PDE training", Vanilla epoch 1000 = same
        # Actually let's plot total wall-clock epochs:
        # MF: Phase 1 = 5000 epochs, then Phase 2 epochs 0-20000 => total 5000-25000
        # Vanilla: epochs 1000-25000

        if method == 'MF':
            # Shift MF epochs by Phase 1 duration (5000)
            shifted = [e + 5000 for e in all_epochs]
        else:
            shifted = all_epochs

        # Find common epoch grid
        min_len = min(len(e) for e in shifted)
        epoch_grid = shifted[0][:min_len]
        l2_matrix = np.array([l[:min_len] for l in all_l2])

        mean_l2 = np.mean(l2_matrix, axis=0)
        std_l2 = np.std(l2_matrix, axis=0)

        ax.plot(epoch_grid, mean_l2, color=color, linewidth=2, label=label)
        ax.fill_between(epoch_grid, mean_l2 - std_l2, mean_l2 + std_l2,
                        alpha=0.2, color=color)

    # Add Phase 1 region
    ax.axvspan(0, 5000, alpha=0.08, color='#E91E63', label='MF Phase 1')
    ax.axvline(5000, color='#E91E63', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(2500, ax.get_ylim()[1]*0.85, 'Phase 1\n(data)', ha='center', fontsize=8,
            color='#E91E63', alpha=0.7)

    ax.set_xlabel('Total epochs')
    ax.set_ylabel('L2 relative error (%)')
    ax.set_title(r'(a) Convergence ($\alpha=0.5$, $N_{\rm col}=100$, $N_{\rm terms}=12$)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 25000)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): Single-seed convergence at multiple N_col ---
    ax = axes[1]
    d_multi = load_tracking(base / 'convergence_alpha0.5.json')

    colors_ncol = {'100': '#E91E63', '500': '#9C27B0', '2000': '#FF9800'}

    for ncol in ['100', '500', '2000']:
        mf_key = f'MF_N{ncol}'
        van_key = f'Vanilla_N{ncol}'

        if mf_key in d_multi and van_key in d_multi:
            mf_epochs = np.array(d_multi[mf_key]['epoch'])
            mf_l2 = np.array(d_multi[mf_key]['l2_error']) * 100
            van_epochs = np.array(d_multi[van_key]['epoch'])
            van_l2 = np.array(d_multi[van_key]['l2_error']) * 100

            c = colors_ncol[ncol]
            ax.plot(van_epochs, van_l2, color=c, linestyle='--', linewidth=1.5,
                    label=f'Vanilla $N_{{col}}$={ncol}')
            ax.plot(mf_epochs + 5000, mf_l2, color=c, linestyle='-', linewidth=2,
                    label=f'MF $N_{{col}}$={ncol}')

    ax.axvspan(0, 5000, alpha=0.08, color='gray')
    ax.axvline(5000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Total epochs')
    ax.set_ylabel('L2 relative error (%)')
    ax.set_title(r'(b) MF (solid) vs Vanilla (dashed) across $N_{\rm col}$')
    ax.legend(loc='upper right', fontsize=8, ncol=1)
    ax.set_ylim(0, 30)
    ax.set_xlim(0, 25000)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 30)

    plt.tight_layout()
    out = Path('figures/convergence_speed.pdf')
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    print(f'Saved to {out}')

    # Also print the "time to reach X%" table
    print('\nTime to reach accuracy thresholds:')
    print(f'{"Method":25s} {"10%":>8s} {"5%":>8s} {"3%":>8s}')
    print('-' * 50)
    for ncol in ['100', '500', '2000']:
        for prefix, shift in [('MF', 5000), ('Vanilla', 0)]:
            key = f'{prefix}_N{ncol}'
            if key in d_multi:
                epochs = np.array(d_multi[key]['epoch']) + shift
                l2 = np.array(d_multi[key]['l2_error']) * 100
                e10 = next((e for e,l in zip(epochs,l2) if l < 10), None)
                e5 = next((e for e,l in zip(epochs,l2) if l < 5), None)
                e3 = next((e for e,l in zip(epochs,l2) if l < 3), None)
                e10s = f'{e10}' if e10 else '---'
                e5s = f'{e5}' if e5 else '---'
                e3s = f'{e3}' if e3 else '---'
                print(f'{prefix:8s} N_col={ncol:>4s}      {e10s:>8s} {e5s:>8s} {e3s:>8s}')

if __name__ == '__main__':
    plot_convergence()
