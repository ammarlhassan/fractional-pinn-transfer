#!/usr/bin/env python3
"""
Integrate seeds 5-9 with existing seeds 0-4 for 10-seed tables.

Handles the N_terms mismatch: existing alpha_sweep.json uses N_terms=15,
but paper tables use N_terms=12. This script uses the CORRECT sources:
  - α=0.5 seeds 0-4: rar_comparison_alpha0.5.json (N_terms=12)
  - α=0.7 seeds 0-4: ncol_sweep_10seeds_alpha0.7.json (N_terms=12)
  - α=1.0 seeds 0-4: needs to be extracted from the correct source

Usage:
  python3 integrate_10seed_results.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2 * 100)
    hi = np.percentile(means, (1+ci)/2 * 100)
    return lo, hi

def bootstrap_ratio_ci(mf_data, van_data, n_boot=10000, ci=0.95):
    """Bootstrap CI for ratio = mean(van) / mean(mf)."""
    rng = np.random.RandomState(42)
    ratios = []
    for _ in range(n_boot):
        idx = rng.choice(len(mf_data), size=len(mf_data), replace=True)
        m = np.mean([mf_data[i] for i in idx])
        v = np.mean([van_data[i] for i in idx])
        if m > 1e-10:
            ratios.append(v / m)
    lo = np.percentile(ratios, (1-ci)/2 * 100)
    hi = np.percentile(ratios, (1+ci)/2 * 100)
    return lo, hi

def load_new_seeds(alpha):
    """Load seeds 5-9 from alpha_sweep_seeds5to9_alpha{X}.json."""
    path = Path(f'results/fdr/alpha_sweep_seeds5to9_alpha{alpha}.json')
    if not path.exists():
        return None
    return json.load(open(path))

def load_ncol_new_seeds(alpha):
    """Load seeds 5-9 from ncol_budget_seeds5to9_alpha{X}_nterms12.json."""
    path = Path(f'results/fdr/ncol_budget_seeds5to9_alpha{alpha}_nterms12.json')
    if not path.exists():
        return None
    return json.load(open(path))

def get_seeds04_alpha05():
    """Seeds 0-4 for α=0.5 from rar_comparison (N_terms=12)."""
    d = json.load(open('results/fdr/rar_comparison_alpha0.5.json'))
    result = {}
    for key in d:
        if isinstance(d[key], dict) and 'l2_values' in d[key]:
            result[key] = d[key]['l2_values']
    return result

def get_seeds04_alpha07():
    """Seeds 0-4 for α=0.7 from ncol_sweep_10seeds (N_terms=12)."""
    d = json.load(open('results/fdr/ncol_sweep_10seeds_alpha0.7.json'))
    result = {}
    for key in d:
        if isinstance(d[key], dict) and 'l2_values' in d[key]:
            result[key] = d[key]['l2_values']
    return result

def get_seeds04_alpha10():
    """Seeds 0-4 for α=1.0.
    
    The ncol_sweep_10seeds_alpha1.0.json used N_terms=8 (stale).
    We need the first 5 seeds from an N_terms=12 run.
    For now, use the first 5 of ncol_sweep_10seeds_alpha1.0 as a placeholder
    since we don't have the original N_terms=12 source file.
    """
    # Check if we have a clean N_terms=12 source
    # TODO: Find the correct source for α=1.0 seeds 0-4 with N_terms=12
    d = json.load(open('results/fdr/ncol_sweep_10seeds_alpha1.0.json'))
    result = {}
    for key in d:
        if isinstance(d[key], dict) and 'l2_values' in d[key]:
            # Only take first 5 seeds
            result[key] = d[key]['l2_values'][:5]
    return result


def main():
    print("="*70)
    print("  10-Seed Integration Report")
    print("="*70)
    
    alphas = [0.5, 0.7, 1.0]
    seeds04_funcs = {
        0.5: get_seeds04_alpha05,
        0.7: get_seeds04_alpha07,
        1.0: get_seeds04_alpha10,
    }
    
    for alpha in alphas:
        new = load_new_seeds(alpha)
        if new is None:
            print(f"\n  α={alpha}: No new seed data found yet.")
            continue
            
        old_data = seeds04_funcs[alpha]()
        
        print(f"\n{'='*70}")
        print(f"  α = {alpha}")
        print(f"{'='*70}")
        
        for ncol in [100, 200, 500]:
            mf_key = f'alpha{alpha}_MF_N{ncol}'
            van_key = f'alpha{alpha}_Vanilla_N{ncol}'
            
            # Get new seeds
            new_mf = [r['l2'] for r in new.get(mf_key, [])] if mf_key in new else []
            new_van = [r['l2'] for r in new.get(van_key, [])] if van_key in new else []
            
            # Get old seeds (key format may differ)
            old_mf_key = f'MF_N{ncol}'
            old_van_key = f'Vanilla_N{ncol}'
            old_mf = old_data.get(old_mf_key, [])
            old_van = old_data.get(old_van_key, [])
            
            if not new_mf or not new_van:
                print(f"  N_col={ncol}: Missing new seed data")
                continue
            
            all_mf = old_mf + new_mf
            all_van = old_van + new_van
            
            mf_mean = np.mean(all_mf) * 100
            mf_std = np.std(all_mf) * 100
            van_mean = np.mean(all_van) * 100
            van_std = np.std(all_van) * 100
            ratio = van_mean / max(mf_mean, 1e-8)
            
            # Wilcoxon test (paired)
            if len(all_mf) == len(all_van) and len(all_mf) >= 5:
                try:
                    stat, p = wilcoxon(all_van, all_mf, alternative='greater')
                    p_str = f"p={p:.3f}"
                except:
                    p_str = "p=N/A"
            else:
                p_str = f"n_mf={len(all_mf)}, n_van={len(all_van)}"
            
            # Bootstrap CI for ratio
            if len(all_mf) >= 10:
                ci_lo, ci_hi = bootstrap_ratio_ci(all_mf, all_van)
                ci_str = f"CI [{ci_lo:.2f}, {ci_hi:.2f}]"
            else:
                ci_str = "CI N/A (need 10 seeds)"
            
            print(f"  N_col={ncol:4d}: MF={mf_mean:.2f}±{mf_std:.2f}%  "
                  f"Van={van_mean:.2f}±{van_std:.2f}%  "
                  f"Ratio={ratio:.2f}×  {p_str}  {ci_str}  "
                  f"(n={len(all_mf)}/{len(all_van)} seeds)")
        
        # Also check Van+50 if available in ncol budget
        ncol_new = load_ncol_new_seeds(alpha)
        if ncol_new:
            print(f"\n  Van+50 (from ncol_budget):")
            for ncol in [100, 200, 500, 2000]:
                key = f'Vanilla+50_N{ncol}'
                if key in ncol_new:
                    new_vals = [r['l2'] for r in ncol_new[key]]
                    old_vals = old_data.get(f'Vanilla+50_N{ncol}', [])
                    all_vals = old_vals + new_vals
                    print(f"    N_col={ncol}: {np.mean(all_vals)*100:.2f}±{np.std(all_vals)*100:.2f}% (n={len(all_vals)})")
    
    # Print LaTeX table update suggestions
    print(f"\n{'='*70}")
    print("  SUGGESTED LaTeX TABLE UPDATES")
    print(f"{'='*70}")
    print("  (Copy these into Tab:ncol and Tab:alpha_sweep)")
    print("  Remember to update caption from '5 seeds' to '10 seeds' where appropriate")

    # GL memory ablation
    gl_path = Path('results/gl_memory_ablation')
    for alpha in [0.5, 1.0]:
        gl_file = gl_path / f'gl_ablation_alpha{alpha}.json'
        if gl_file.exists():
            d = json.load(open(gl_file))
            print(f"\n{'='*70}")
            print(f"  GL MEMORY ABLATION α={alpha}")
            print(f"{'='*70}")
            for mem in [2, 5, 10, 20, 50, 100]:
                mf_key = f'MF_mem{mem}'
                van_key = f'Vanilla_mem{mem}'
                if mf_key in d and van_key in d:
                    mf_vals = d[mf_key].get('l2_values', [])
                    van_vals = d[van_key].get('l2_values', [])
                    mf_m = np.mean(mf_vals) * 100
                    van_m = np.mean(van_vals) * 100
                    ratio = van_m / max(mf_m, 1e-8)
                    print(f"  mem={mem:3d}: MF={mf_m:.2f}%  Van={van_m:.2f}%  Ratio={ratio:.2f}×  n={len(mf_vals)}")


if __name__ == '__main__':
    main()
