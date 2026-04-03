"""
Publication-quality figures for the fractional diffusion-reaction study.

Generates:
  1. Solution profiles: u(x) at multiple times, comparing PINN vs reference
  2. Convergence curves: loss vs epoch for MF vs vanilla
  3. Transfer efficiency: L2 error vs N_LF
  4. Inverse convergence: α(epoch) and κ(epoch) traces
  5. Noise robustness: parameter error vs noise level
  6. α comparison: solution profiles for different α values
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
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

COLORS = {
    'M1': '#d62728',   # red
    'M2': '#ff7f0e',   # orange
    'M3': '#1f77b4',   # blue
    'M4': '#2ca02c',   # green
    'ref': 'k',        # black
}


def plot_solution_profiles(ref_data, pinn_results, alpha, save_path=None):
    """
    Plot u(x) at multiple time snapshots.

    Parameters
    ----------
    ref_data : dict with 'x', 't', 'u' (shape nx × nt)
    pinn_results : dict mapping method_name -> {'x': ..., 'u': dict of t->array}
    alpha : float
    save_path : str
    """
    t_values = ref_data['t']
    n_times = len(t_values)
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 3.5), sharey=True)
    if n_times == 1:
        axes = [axes]

    for j, t_val in enumerate(t_values):
        ax = axes[j]
        # Reference
        ax.plot(ref_data['x'], ref_data['u'][:, j], 'k-', lw=2,
                label='Reference', zorder=10)

        # PINN methods
        for name, res in pinn_results.items():
            color = COLORS.get(name[:2], '#7f7f7f')
            u_pred = res['u'][f't{float(t_val):.1f}']
            ax.plot(ref_data['x'], u_pred, '--', color=color, lw=1.5,
                    label=name, alpha=0.8)

        ax.set_xlabel('$x$')
        ax.set_title(f'$t = {float(t_val):.1f}$')
        if j == 0:
            ax.set_ylabel('$u(x, t)$')
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc='upper right')
    fig.suptitle(f'Solution profiles ($\\alpha = {alpha}$)', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_convergence(histories, labels=None, save_path=None):
    """
    Plot training loss convergence for multiple methods.

    Parameters
    ----------
    histories : list of dict (each from trainer.history)
    labels : list of str
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, (hist, label) in enumerate(zip(histories, labels or range(len(histories)))):
        color = list(COLORS.values())[i % len(COLORS)]
        epochs = np.arange(len(hist['total_loss']))

        axes[0].semilogy(epochs, hist['total_loss'], color=color, label=label, alpha=0.8)
        if any(v > 0 for v in hist['pde_loss']):
            axes[1].semilogy(epochs, hist['pde_loss'], color=color, label=label, alpha=0.8)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PDE Residual')
    axes[1].set_title('PDE Residual')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_transfer_curve(transfer_data, save_path=None):
    """
    Plot L2 error vs N_LF with vanilla baseline.

    Parameters
    ----------
    transfer_data : dict with 'vanilla', 'transfer' keys
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Transfer curve
    n_lf_values = sorted([int(k) for k in transfer_data['transfer'].keys()])
    means = [transfer_data['transfer'][str(n)]['mean'] for n in n_lf_values]
    stds = [transfer_data['transfer'][str(n)]['std'] for n in n_lf_values]

    ax.errorbar(n_lf_values, means, yerr=stds, fmt='o-', color=COLORS['M3'],
                capsize=4, label='MF-fPINN', zorder=5)

    # Vanilla baseline
    van_mean = transfer_data['vanilla']['mean']
    van_std = transfer_data['vanilla']['std']
    ax.axhline(van_mean, color=COLORS['M1'], ls='--', lw=1.5, label='Vanilla fPINN')
    ax.axhspan(van_mean - van_std, van_mean + van_std,
               color=COLORS['M1'], alpha=0.1)

    ax.set_xlabel('$N_{LF}$ (number of Laplace data points)')
    ax.set_ylabel('Relative $L^2$ error')
    ax.set_title(f'Transfer Efficiency ($\\alpha = {transfer_data.get("alpha", "?")}$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_inverse_convergence(history, true_alpha, true_kappa, save_path=None):
    """
    Plot parameter convergence during inverse problem.

    Parameters
    ----------
    history : dict with 'alpha', 'kappa', 'meas_loss', 'pde_loss'
    true_alpha, true_kappa : float
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    epochs = np.arange(len(history['alpha']))

    # α convergence
    ax = axes[0]
    ax.plot(epochs, history['alpha'], color=COLORS['M3'], lw=1.5)
    ax.axhline(true_alpha, color='k', ls='--', lw=1, label=f'True $\\alpha = {true_alpha}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$\\alpha$')
    ax.set_title('Fractional order')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # κ convergence
    ax = axes[1]
    ax.plot(epochs, history['kappa'], color=COLORS['M4'], lw=1.5)
    ax.axhline(true_kappa, color='k', ls='--', lw=1, label=f'True $\\kappa = {true_kappa}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$\\kappa$')
    ax.set_title('Reaction rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[2]
    ax.semilogy(epochs, history['meas_loss'], color=COLORS['M3'],
                label='Measurement', alpha=0.8)
    ax.semilogy(epochs, history['pde_loss'], color=COLORS['M1'],
                label='PDE', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_noise_robustness(results_by_noise, true_alpha, true_kappa, save_path=None):
    """
    Plot parameter estimation error vs noise level.

    Parameters
    ----------
    results_by_noise : dict mapping noise_level -> list of result dicts
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    noise_levels = sorted(results_by_noise.keys())

    for param, true_val, ax, color in [
        ('alpha', true_alpha, axes[0], COLORS['M3']),
        ('kappa', true_kappa, axes[1], COLORS['M4']),
    ]:
        means, stds = [], []
        for nl in noise_levels:
            errs = [r[f'{param}_error_pct'] for r in results_by_noise[nl]]
            means.append(np.mean(errs))
            stds.append(np.std(errs))

        ax.errorbar([nl * 100 for nl in noise_levels], means, yerr=stds,
                     fmt='o-', color=color, capsize=4)
        ax.set_xlabel('Noise level (%)')
        ax.set_ylabel(f'${param}$ error (%)')
        ax.set_title(f'${param}$ identification (true={true_val})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_factorial_comparison(results_all_methods, alpha, t_values, save_path=None):
    """
    Bar chart comparing M1-M4 across time snapshots.

    Parameters
    ----------
    results_all_methods : dict mapping method_name -> list of per-seed dicts
    alpha : float
    t_values : list of float
    """
    methods = list(results_all_methods.keys())
    n_methods = len(methods)
    n_times = len(t_values)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n_times), 4))

    x = np.arange(n_times)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        means, stds = [], []
        for t_val in t_values:
            key = f't{t_val:.2f}_u_l2'
            vals = [r[key] for r in results_all_methods[method]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        color = list(COLORS.values())[i % len(COLORS)]
        ax.bar(x + i * width - 0.4 + width/2, means, width, yerr=stds,
               label=method, color=color, alpha=0.8, capsize=3)

    ax.set_xlabel('Time snapshot')
    ax.set_ylabel('Relative $L^2$ error')
    ax.set_title(f'Method comparison ($\\alpha = {alpha}$)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$t={t:.1f}$' for t in t_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
