#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results for the paper.

Tables:
  1. N_col budget comparison (α=0.5 and α=1.0)
  2. Transfer efficiency summary
  3. M1 vs M3 main results at N_col=2000
  4. HPO results (when available)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

res_dir = Path('results/fdr')


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test between two paired samples."""
    try:
        stat, p = stats.wilcoxon(a, b)
        return p
    except Exception:
        return 1.0


def significance_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def table_ncol_budget(alpha=0.5):
    """LaTeX table for N_col budget study."""
    path = res_dir / f'ncol_sweep_alpha{alpha}.json'
    if not path.exists():
        print(f"  Skipping N_col budget table (α={alpha}): not found")
        return

    with open(path) as f:
        data = json.load(f)

    n_cols = sorted(set(v['n_col'] for v in data.values()))

    print(f"\n% Table: N_col Budget Study (α={alpha})")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(f"\\caption{{Collocation budget study ($\\alpha = {alpha}$, 5 seeds). "
          "MF-fPINN shows largest advantage when $N_{{\\mathrm{{col}}}}$ is limited.}}")
    print(f"\\label{{tab:ncol_alpha{alpha}}}")
    print(r"\begin{tabular}{r c c c c}")
    print(r"\toprule")
    print(r"$N_{\mathrm{col}}$ & MF-fPINN $L^2$ (\%) & Vanilla $L^2$ (\%) & Ratio & $p$-value \\")
    print(r"\midrule")

    for n_col in n_cols:
        mf = data[f'MF_N{n_col}']
        van = data[f'Vanilla_N{n_col}']
        ratio = van['l2_mean'] / max(mf['l2_mean'], 1e-8)
        p = wilcoxon_test(mf['l2_values'], van['l2_values'])
        stars = significance_stars(p)

        mf_str = f"{mf['l2_mean']*100:.2f} $\\pm$ {mf['l2_std']*100:.2f}"
        van_str = f"{van['l2_mean']*100:.2f} $\\pm$ {van['l2_std']*100:.2f}"

        bold = r"\textbf{" if ratio > 1.5 else ""
        bold_end = "}" if ratio > 1.5 else ""

        print(f"  {n_col} & {bold}{mf_str}{bold_end} & {van_str} & "
              f"{ratio:.1f}$\\times$ & {p:.3f} ({stars}) \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def table_main_results():
    """LaTeX table for main M1 vs M3 comparison at N_col=2000."""
    alphas = [0.5, 1.0]
    print(f"\n% Table: Main Results (M1 vs M3, N_col=2000)")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Forward problem: MF-fPINN (M3) vs. vanilla fPINN (M1) "
          r"at $N_{\mathrm{col}} = 2000$ (5 seeds).}")
    print(r"\label{tab:main_results}")
    print(r"\begin{tabular}{c c c c c}")
    print(r"\toprule")
    print(r"$\alpha$ & Method & Mean $L^2$ (\%) & Std & Wall time (s) \\")
    print(r"\midrule")

    for alpha in alphas:
        path = res_dir / f'results_alpha{alpha}.json'
        if not path.exists():
            print(f"  % Skipping α={alpha}: not found")
            continue

        with open(path) as f:
            data = json.load(f)

        for method_key, label in [('M3_mf', 'MF-fPINN'), ('M1_vanilla', 'Vanilla')]:
            if method_key not in data:
                continue
            seed_results = data[method_key]  # list of per-seed dicts

            # Average L2 across all time snapshots and seeds
            all_l2 = []
            walls = []
            for r in seed_results:
                l2_vals = [v for k, v in r.items() if k.endswith('_l2')]
                all_l2.append(np.mean(l2_vals))
                if 'wall_time' in r:
                    walls.append(r['wall_time'])

            mean_l2 = np.mean(all_l2) * 100
            std_l2 = np.std(all_l2) * 100
            wall_str = f"{np.mean(walls):.0f}" if walls else "—"

            print(f"  {alpha} & {label} & {mean_l2:.2f} & {std_l2:.2f} & {wall_str} \\\\")
        if alpha != alphas[-1]:
            print(r"  \midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def table_transfer_efficiency(alpha=0.5):
    """LaTeX table for transfer curve."""
    path = res_dir / f'transfer_curve_alpha{alpha}.json'
    if not path.exists():
        print(f"  Skipping transfer efficiency table: not found")
        return

    with open(path) as f:
        data = json.load(f)

    n_lf_values = sorted([int(k) for k in data['transfer'].keys()])

    print(f"\n% Table: Transfer Efficiency (α={alpha})")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(f"\\caption{{Transfer efficiency: effect of low-fidelity data size "
          f"($\\alpha = {alpha}$, $N_{{\\mathrm{{col}}}} = 2000$, 5 seeds).}}")
    print(r"\label{tab:transfer_efficiency}")
    print(r"\begin{tabular}{r c c}")
    print(r"\toprule")
    print(r"$N_{\mathrm{LF}}$ & MF-fPINN $L^2$ (\%) & vs. Vanilla \\")
    print(r"\midrule")

    van_mean = data['vanilla']['mean'] * 100
    van_std = data['vanilla']['std'] * 100

    for n_lf in n_lf_values:
        d = data['transfer'][str(n_lf)]
        mean = d['mean'] * 100
        std = d['std'] * 100
        diff = mean - van_mean
        sign = '+' if diff > 0 else ''
        print(f"  {n_lf} & {mean:.2f} $\\pm$ {std:.2f} & {sign}{diff:.2f}\\% \\\\")

    print(r"  \midrule")
    print(f"  Vanilla & {van_mean:.2f} $\\pm$ {van_std:.2f} & — \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == '__main__':
    print("=" * 60)
    print("  LaTeX Tables for MF-fPINN Paper")
    print("=" * 60)

    table_ncol_budget(alpha=0.5)
    table_ncol_budget(alpha=1.0)
    table_main_results()
    table_transfer_efficiency(alpha=0.5)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
