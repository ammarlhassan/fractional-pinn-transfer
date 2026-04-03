#!/usr/bin/env python3
"""
Causal training baseline (Wang et al. 2022) vs MF-fPINN vs Vanilla.

Tests whether improved PINN training (causal weighting) can match MF transfer.
This is a key ablation: if causal training alone matches MF, then the MF
benefit is just "training the PINN better" rather than genuine data transfer.

Fairness guarantees:
  - Same architecture (5 layers, 128 neurons, tanh, Fourier features)
  - Same total epochs (25000 for Vanilla and Causal; 5000+20000 for MF)
  - Same LR schedule (5e-4 cosine for Vanilla/Causal; 5e-4 Phase1 + 1e-4 Phase2 for MF)
  - Same collocation budget (N_col)
  - Same random seeds
  - Same evaluation protocol (L2 at 4 time snapshots)
  - Only difference: loss weighting scheme (uniform vs causal vs data+physics)

Usage:
  python3 scripts/experiments/run_causal_comparison.py --alpha 0.5 --device cuda:3 --seeds 10
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha, N_col=100):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': N_col}
    return cfg


def build_model(config):
    return FDRNet(
        n_layers=config['n_layers'],
        n_neurons=config['n_neurons'],
        activation=config['activation'],
        L=config['L'],
        T=config['T'],
        hard_bc=True,
        bc_type=config['bc_type'],
        bc_params=config['bc_params'],
        fourier_features=config['fourier_features'],
        fourier_sigma=config['fourier_sigma'],
    )


def evaluate_model(model, ref_data, device):
    model.eval()
    results = {}
    x_ref = ref_data['x']
    u_grid = ref_data['u']
    t_vals = ref_data['t']

    with torch.no_grad():
        for j, t_val in enumerate(t_vals):
            x_t = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = u_grid[:, j]
            l2 = compute_l2_relative_error(u_pred, u_ref)
            results[f't{t_val:.2f}_u_l2'] = l2
    return results


def run_single(method, config, lf_data, ref_data, seed, device,
               epsilon=10.0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config)
    trainer = FDRTrainer(model, config, lf_data, device=device)

    t0 = time.time()
    if method == 'vanilla':
        trainer.train_vanilla(verbose=verbose)
    elif method == 'mf':
        trainer.train_full(verbose=verbose)
    elif method.startswith('causal'):
        trainer.train_causal(verbose=verbose, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown method: {method}")
    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--n_col', type=int, default=100)
    parser.add_argument('--n_lf', type=int, default=50)
    parser.add_argument('--lf_nterms', type=int, default=20,
                        help='LF N_terms for Laplace degradation (20=high quality)')
    parser.add_argument('--epsilons', type=str, default='1,10,100',
                        help='Comma-separated causal epsilon values to sweep')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    epsilons = [float(e) for e in args.epsilons.split(',')]

    print(f"\n{'#'*70}")
    print(f"  Causal Training Baseline Comparison")
    print(f"  α={alpha}, N_col={args.n_col}, seeds={args.seeds}")
    print(f"  Epsilon values: {epsilons}")
    print(f"  LF N_terms: {args.lf_nterms}")
    print(f"{'#'*70}\n")

    # ── 1. Reference data ──────────────────────────────────
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/causal')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # ── 2. LF data (for MF method) ────────────────────────
    lf_N_terms = args.lf_nterms
    lf_precision = 20

    print(f"\nLF data: N_terms={lf_N_terms}, precision={lf_precision}")
    lf_err = compute_lf_error(solver, lf_N_terms=lf_N_terms,
                              lf_precision=lf_precision, N_test=50)
    lf_error_pct = lf_err['l2_relative'] * 100
    print(f"  LF error: L2_rel = {lf_error_pct:.2f}%")

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, lf_N_terms=lf_N_terms, lf_precision=lf_precision,
    )

    # ── 3. Run all methods ─────────────────────────────────
    config = build_config(alpha, N_col=args.n_col)
    all_results = {}

    # Methods to test
    methods = ['vanilla', 'mf']
    for eps in epsilons:
        methods.append(f'causal_eps{eps}')

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        epsilon = None
        if method.startswith('causal'):
            epsilon = float(method.split('eps')[1])

        method_results = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            metrics = run_single(
                method, config, lf_data, ref_data, seed, device,
                epsilon=epsilon, verbose=args.verbose,
            )
            method_results.append(metrics)

            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            avg = np.mean(l2_vals)
            print(f"  → Avg L2: {avg*100:.2f}%  |  Wall: {metrics['wall_time']:.1f}s")

        all_results[method] = method_results

    # ── 4. Summary ─────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  RESULTS SUMMARY  (α={alpha}, N_col={args.n_col})")
    print(f"{'#'*70}")

    avg_l2s = {}
    for method, results in all_results.items():
        l2_per_seed = []
        for r in results:
            vals = [v for k, v in r.items() if k.endswith('_l2')]
            l2_per_seed.append(np.mean(vals))
        mean_l2 = np.mean(l2_per_seed) * 100
        std_l2 = np.std(l2_per_seed, ddof=1) * 100
        avg_l2s[method] = (mean_l2, std_l2)
        print(f"  {method:>20s}: {mean_l2:.2f}% ± {std_l2:.2f}%")

    # Advantage ratios
    van_mean = avg_l2s['vanilla'][0]
    print(f"\n  Advantage over Vanilla ({van_mean:.2f}%):")
    for method, (mean, std) in avg_l2s.items():
        if method != 'vanilla':
            ratio = van_mean / mean if mean > 0 else float('inf')
            print(f"    {method:>20s}: {ratio:.2f}×")

    # Statistical tests
    from scipy.stats import wilcoxon
    print(f"\n  Wilcoxon signed-rank tests (vs Vanilla):")
    van_l2s = []
    for r in all_results['vanilla']:
        van_l2s.append(np.mean([v for k, v in r.items() if k.endswith('_l2')]))

    for method in all_results:
        if method == 'vanilla':
            continue
        m_l2s = []
        for r in all_results[method]:
            m_l2s.append(np.mean([v for k, v in r.items() if k.endswith('_l2')]))
        try:
            stat, p = wilcoxon(van_l2s, m_l2s)
            print(f"    {method:>20s}: p={p:.4f}")
        except Exception as e:
            print(f"    {method:>20s}: {e}")

    # ── 5. Save ────────────────────────────────────────────
    save_data = {
        'alpha': alpha,
        'N_col': args.n_col,
        'lf_N_terms': lf_N_terms,
        'lf_error_pct': lf_error_pct,
        'epsilons': epsilons,
        'seeds': args.seeds,
        'results': all_results,
        'summary': {m: {'mean': v[0], 'std': v[1]} for m, v in avg_l2s.items()},
    }
    json_path = out_dir / f'causal_alpha{alpha}_ncol{args.n_col}_nterms{lf_N_terms}_{args.seeds}seeds.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
