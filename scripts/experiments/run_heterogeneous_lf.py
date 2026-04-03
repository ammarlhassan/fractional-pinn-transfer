#!/usr/bin/env python3
"""
Heterogeneous LF source experiment: MF-fPINN with finite-difference LF data.

Tests whether MF transfer works when the LF source is a STRUCTURALLY DIFFERENT
solver (L1 scheme FD) rather than a degraded version of the same solver
(Laplace domain with fewer terms).

This addresses the concern: "Degrading the same solver is unrealistic;
does MF still work with data from a qualitatively different solver?"

Key design:
  - FD LF data uses L1 + implicit FD solver (coarse grid)
  - For comparison, Laplace LF data is generated at MATCHED error level
  - Both use the same (x,t) sample points (same random seed)
  - Evaluation is against Laplace HF reference (the ground truth)

Fairness guarantees:
  - Same architecture, epochs, LR, collocation budget across all methods
  - Same random seeds for training
  - Same (x,t) sample locations for LF data (only u-values differ)
  - LF error levels matched between FD and Laplace sources

Usage:
  python3 scripts/experiments/run_heterogeneous_lf.py --alpha 0.5 --device cuda:3 --seeds 10
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
from mf_fpinn.solvers.fdr_fd_solver import (
    solve_fdr_fd, generate_lf_data_fd, compute_fd_lf_error,
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


def compute_fd_lf_error_vs_laplace(alpha, lf_Nx, lf_Nt, ref_data,
                                    D=1.0, kappa=1.0, L=1.0, T=1.0,
                                    bc_type='pulse', bc_params=None):
    """Compute error of coarse FD vs Laplace HF reference on the evaluation grid."""
    bc_params = bc_params or {'a': 0.5}
    from scipy.interpolate import RegularGridInterpolator

    x_fd, t_fd, u_fd = solve_fdr_fd(alpha, D, kappa, lf_Nx, lf_Nt, L, T,
                                      bc_type, bc_params)
    interp = RegularGridInterpolator((x_fd, t_fd), u_fd, method='linear',
                                      bounds_error=False, fill_value=0.0)

    x_ref = ref_data['x']
    u_ref = ref_data['u']
    t_ref = ref_data['t']

    errors = []
    for j, tv in enumerate(t_ref):
        pts = np.column_stack([x_ref, np.full_like(x_ref, tv)])
        u_fd_vals = interp(pts)
        u_lap_vals = u_ref[:, j]
        norm_ref = np.linalg.norm(u_lap_vals)
        if norm_ref > 1e-12:
            errors.append(np.linalg.norm(u_fd_vals - u_lap_vals) / norm_ref)

    return np.mean(errors)


def generate_fd_lf_at_laplace_points(alpha, lf_Nx, lf_Nt, x_pts, t_pts,
                                      D=1.0, kappa=1.0, L=1.0, T=1.0,
                                      bc_type='pulse', bc_params=None):
    """Generate FD LF data AT THE SAME (x,t) points as Laplace LF data.

    This ensures a fair comparison: same sample locations, different u-values.
    """
    bc_params = bc_params or {'a': 0.5}
    from scipy.interpolate import RegularGridInterpolator

    x_fd, t_fd, u_fd = solve_fdr_fd(alpha, D, kappa, lf_Nx, lf_Nt, L, T,
                                      bc_type, bc_params)
    interp = RegularGridInterpolator((x_fd, t_fd), u_fd, method='linear',
                                      bounds_error=False, fill_value=0.0)

    pts = np.column_stack([x_pts, t_pts])
    u_vals = interp(pts)

    return {'x': x_pts, 't': t_pts, 'u': u_vals}


def run_single(method, config, lf_data, ref_data, seed, device, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config)
    trainer = FDRTrainer(model, config, lf_data, device=device)

    t0 = time.time()
    if method == 'vanilla':
        trainer.train_vanilla(verbose=verbose)
    elif method.startswith('mf'):
        trainer.train_full(verbose=verbose)
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
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    # FD coarseness configs and their matching Laplace N_terms
    # Goal: match error levels between FD and Laplace LF sources
    fd_configs = [
        {'Nx': 3, 'Nt': 30, 'label': 'FD_Nx3'},
        {'Nx': 5, 'Nt': 50, 'label': 'FD_Nx5'},
        {'Nx': 10, 'Nt': 100, 'label': 'FD_Nx10'},
        {'Nx': 20, 'Nt': 200, 'label': 'FD_Nx20'},
    ]

    # Laplace N_terms configs for comparison
    laplace_configs = [20, 15, 12]

    print(f"\n{'#'*70}")
    print(f"  Heterogeneous LF Source Experiment")
    print(f"  α={alpha}, N_col={args.n_col}, seeds={args.seeds}")
    print(f"  FD configs: {[c['label'] for c in fd_configs]}")
    print(f"  Laplace N_terms: {laplace_configs}")
    print(f"{'#'*70}\n")

    # ── 1. Reference data ──────────────────────────────────
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/heterogeneous_lf')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # ── 2. Measure FD LF errors vs Laplace HF ─────────────
    print(f"\nFD LF error vs Laplace HF reference:")
    fd_errors = {}
    for fc in fd_configs:
        err = compute_fd_lf_error_vs_laplace(
            alpha, fc['Nx'], fc['Nt'], ref_data,
            D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        )
        fd_errors[fc['label']] = err
        print(f"  {fc['label']}: L2_rel = {err*100:.2f}%")

    # Measure Laplace LF errors
    print(f"\nLaplace LF error vs HF:")
    lap_errors = {}
    for nt in laplace_configs:
        err = compute_lf_error(solver, lf_N_terms=nt, lf_precision=20, N_test=50)
        lap_errors[f'Lap_N{nt}'] = err['l2_relative']
        print(f"  N_terms={nt}: L2_rel = {err['l2_relative']*100:.2f}%")

    # ── 3. Generate LF data ───────────────────────────────
    # First generate Laplace LF to get the (x,t) sample points
    # Then generate FD LF at the SAME (x,t) points for fair comparison
    print(f"\nGenerating LF datasets...")

    # Base Laplace LF (N_terms=20, high quality)
    lf_data_base = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, lf_N_terms=20, lf_precision=20,
    )
    # Same (x,t) points for all LF sources
    x_pts = lf_data_base['x']
    t_pts = lf_data_base['t']

    # Generate all LF datasets
    lf_datasets = {}

    # Laplace LF at various N_terms
    for nt in laplace_configs:
        lf = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=args.n_lf, lf_N_terms=nt, lf_precision=20,
        )
        lf_datasets[f'Lap_N{nt}'] = lf

    # FD LF at various coarseness (using SAME (x,t) points)
    for fc in fd_configs:
        lf = generate_fd_lf_at_laplace_points(
            alpha, fc['Nx'], fc['Nt'], x_pts, t_pts,
            D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        )
        lf_datasets[fc['label']] = lf

    # Compute actual LF data error at training points vs HF
    print(f"\nActual LF data quality at training points:")
    hf_solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )
    u_hf_at_pts = np.array([hf_solver.evaluate(x_pts[i], t_pts[i])
                            for i in range(len(x_pts))])
    norm_hf = np.linalg.norm(u_hf_at_pts)

    data_errors = {}
    for name, lf in lf_datasets.items():
        err = np.linalg.norm(lf['u'] - u_hf_at_pts) / max(norm_hf, 1e-10)
        data_errors[name] = err
        print(f"  {name}: L2_rel at training points = {err*100:.2f}%")

    # ── 4. Run experiments ─────────────────────────────────
    config = build_config(alpha, N_col=args.n_col)
    all_results = {}

    # Vanilla baseline (same for all, no LF data used)
    print(f"\n{'='*60}")
    print(f"  Method: vanilla")
    print(f"{'='*60}")
    vanilla_results = []
    # Use dummy LF data for trainer (not used in vanilla mode)
    dummy_lf = lf_datasets[list(lf_datasets.keys())[0]]
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        metrics = run_single('vanilla', config, dummy_lf, ref_data, seed, device,
                             verbose=args.verbose)
        vanilla_results.append(metrics)
        l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
        print(f"  → Avg L2: {np.mean(l2_vals)*100:.2f}%")
    all_results['vanilla'] = vanilla_results

    # MF with each LF source
    for lf_name, lf_data in lf_datasets.items():
        print(f"\n{'='*60}")
        print(f"  Method: mf_{lf_name}")
        print(f"{'='*60}")
        method_results = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            metrics = run_single(f'mf_{lf_name}', config, lf_data, ref_data,
                                 seed, device, verbose=args.verbose)
            method_results.append(metrics)
            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            print(f"  → Avg L2: {np.mean(l2_vals)*100:.2f}%")
        all_results[f'mf_{lf_name}'] = method_results

    # ── 5. Summary ─────────────────────────────────────────
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
        avg_l2s[method] = (mean_l2, std_l2, l2_per_seed)
        lf_err_str = ""
        lf_key = method.replace('mf_', '')
        if lf_key in data_errors:
            lf_err_str = f"  (LF err: {data_errors[lf_key]*100:.1f}%)"
        print(f"  {method:>25s}: {mean_l2:.2f}% ± {std_l2:.2f}%{lf_err_str}")

    # Advantage table
    van_mean = avg_l2s['vanilla'][0]
    print(f"\n  Advantage over Vanilla ({van_mean:.2f}%):")
    print(f"  {'Method':>25s}  {'Ratio':>6s}  {'LF src':>8s}  {'LF err':>8s}")
    for method in all_results:
        if method == 'vanilla':
            continue
        mean_l2, std_l2, _ = avg_l2s[method]
        ratio = van_mean / mean_l2 if mean_l2 > 0 else float('inf')
        lf_key = method.replace('mf_', '')
        lf_err = data_errors.get(lf_key, 0) * 100
        src = 'FD' if 'FD' in method else 'Laplace'
        print(f"  {method:>25s}  {ratio:>5.2f}×  {src:>8s}  {lf_err:>7.1f}%")

    # Statistical tests
    from scipy.stats import wilcoxon
    print(f"\n  Wilcoxon tests:")
    van_l2s = avg_l2s['vanilla'][2]
    for method in all_results:
        if method == 'vanilla':
            continue
        m_l2s = avg_l2s[method][2]
        try:
            stat, p = wilcoxon(van_l2s, m_l2s)
            sig = '***' if p < 0.003 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f"    {method:>25s}: p={p:.4f} {sig}")
        except Exception as e:
            print(f"    {method:>25s}: {e}")

    # FD vs Laplace at matched error comparison
    print(f"\n  FD vs Laplace at similar error levels:")
    for fd_name in fd_errors:
        fd_err = fd_errors[fd_name] * 100
        # Find closest Laplace
        best_match = None
        best_diff = float('inf')
        for lap_name, lap_err in data_errors.items():
            if 'Lap' in lap_name:
                diff = abs(lap_err * 100 - fd_err)
                if diff < best_diff:
                    best_diff = diff
                    best_match = lap_name
        if best_match and f'mf_{fd_name}' in avg_l2s and f'mf_{best_match}' in avg_l2s:
            fd_l2 = avg_l2s[f'mf_{fd_name}'][0]
            lap_l2 = avg_l2s[f'mf_{best_match}'][0]
            print(f"    {fd_name} ({fd_err:.1f}% err) → {fd_l2:.2f}%  vs  "
                  f"{best_match} ({data_errors[best_match]*100:.1f}% err) → {lap_l2:.2f}%")

    # ── 6. Save ────────────────────────────────────────────
    save_data = {
        'alpha': alpha,
        'N_col': args.n_col,
        'seeds': args.seeds,
        'fd_errors': {k: float(v) for k, v in fd_errors.items()},
        'lap_errors': {k: float(v) for k, v in lap_errors.items()},
        'data_errors': {k: float(v) for k, v in data_errors.items()},
        'results': all_results,
        'summary': {m: {'mean': v[0], 'std': v[1]} for m, v in avg_l2s.items()},
    }
    json_path = out_dir / f'heterogeneous_alpha{alpha}_ncol{args.n_col}_{args.seeds}seeds.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
