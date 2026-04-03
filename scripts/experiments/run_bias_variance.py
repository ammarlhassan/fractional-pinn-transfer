#!/usr/bin/env python3
"""
Bias-Variance Decomposition for MF-fPINN vs Vanilla.

Computes bias² and variance from multi-seed predictions on a common grid.
Shows that MF trades variance for bias — the core theoretical insight.

Run:  python3 run_bias_variance.py --alpha 0.5 --device cuda:3 --seeds 10
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


def build_config(alpha, n_col):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': n_col}
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


def predict_on_grid(model, x_grid, t_grid, device):
    """Get predictions on a space-time grid. Returns (nx, nt) array."""
    model.eval()
    nx = len(x_grid)
    nt = len(t_grid)
    preds = np.zeros((nx, nt))

    with torch.no_grad():
        for j, t_val in enumerate(t_grid):
            x_t = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            preds[:, j] = u_pred

    return preds


def compute_bias_variance(predictions, reference):
    """
    Bias-variance decomposition.

    predictions: (S, nx, nt) — S seeds
    reference:   (nx, nt)

    Returns bias², variance, MSE (all scalars, normalized).
    """
    ref_norm_sq = np.sum(reference**2)
    if ref_norm_sq < 1e-15:
        ref_norm_sq = 1.0

    # Mean prediction across seeds
    mean_pred = np.mean(predictions, axis=0)  # (nx, nt)

    # Bias² = ||mean_pred - reference||² / ||reference||²
    bias_sq = np.sum((mean_pred - reference)**2) / ref_norm_sq

    # Variance = (1/S) Σ_i ||pred_i - mean_pred||² / ||reference||²
    variance = np.mean([
        np.sum((predictions[i] - mean_pred)**2) / ref_norm_sq
        for i in range(len(predictions))
    ])

    # MSE = Bias² + Variance
    mse = bias_sq + variance

    return {
        'bias_sq': float(bias_sq),
        'variance': float(variance),
        'mse': float(mse),
        'bias': float(np.sqrt(bias_sq)),
        'std': float(np.sqrt(variance)),
        'bias_frac': float(bias_sq / mse) if mse > 1e-15 else 0.0,
        'var_frac': float(variance / mse) if mse > 1e-15 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, nargs='+', default=[12, 15, 20])
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    n_col = args.n_col

    print(f"\n{'#'*70}")
    print(f"  BIAS-VARIANCE DECOMPOSITION")
    print(f"  α={alpha}, N_col={n_col}, Seeds={args.seeds}")
    print(f"  N_terms: {args.n_terms}")
    print(f"{'#'*70}\n")

    # 1. Reference solution on dense grid
    print("Generating reference solution...")
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    # Dense eval grid: 50 x-points × 20 t-points
    x_grid = np.linspace(0.01, PHYSICS['L'] - 0.01, 50)
    t_grid = np.linspace(0.02, PHYSICS['T'] * 0.5, 20)  # stay within pulse window

    ref_grid = np.zeros((len(x_grid), len(t_grid)))
    print(f"  Computing reference on {len(x_grid)}×{len(t_grid)} grid...")
    for j, t_val in enumerate(t_grid):
        for i, x_val in enumerate(x_grid):
            ref_grid[i, j] = float(solver.evaluate(x_val, t_val))
    print(f"  Reference peak: {ref_grid.max():.4f}")

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 2. Vanilla baseline (seed-varying predictions)
    print(f"\n{'='*60}")
    print(f"  Vanilla baseline ({args.seeds} seeds)")
    print(f"{'='*60}")

    config = build_config(alpha, n_col)
    van_preds = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(config)

        # Need LF data for trainer init even if vanilla
        if seed == 0:
            lf_data_dummy = generate_low_fidelity_data(
                alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                N_LF=50, lf_N_terms=12, lf_precision=20,
            )

        trainer = FDRTrainer(model, config, lf_data_dummy, device=device)
        trainer.train_vanilla(verbose=False)

        pred = predict_on_grid(model, x_grid, t_grid, device)
        van_preds.append(pred)
        l2 = np.sqrt(np.sum((pred - ref_grid)**2) / np.sum(ref_grid**2))
        print(f"  Vanilla seed {seed}: L2={l2*100:.2f}%")

    van_preds = np.array(van_preds)  # (S, nx, nt)
    van_bv = compute_bias_variance(van_preds, ref_grid)
    results['vanilla'] = van_bv
    print(f"\n  Vanilla decomposition:")
    print(f"    Bias  = {van_bv['bias']*100:.2f}%  (fraction: {van_bv['bias_frac']:.1%})")
    print(f"    Std   = {van_bv['std']*100:.2f}%  (fraction: {van_bv['var_frac']:.1%})")
    print(f"    MSE^½ = {np.sqrt(van_bv['mse'])*100:.2f}%")

    # 3. MF at each fidelity level
    for n_terms in args.n_terms:
        print(f"\n{'='*60}")
        print(f"  MF-fPINN N_terms={n_terms} ({args.seeds} seeds)")
        print(f"{'='*60}")

        # Generate LF data
        lf_err = compute_lf_error(solver, lf_N_terms=n_terms,
                                   lf_precision=20, N_test=50)
        lf_error_pct = lf_err['l2_relative'] * 100
        print(f"  LF error: {lf_error_pct:.1f}%")

        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, lf_N_terms=n_terms, lf_precision=20,
        )

        mf_preds = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(config)
            trainer = FDRTrainer(model, config, lf_data, device=device)
            trainer.train_full(verbose=False)

            pred = predict_on_grid(model, x_grid, t_grid, device)
            mf_preds.append(pred)
            l2 = np.sqrt(np.sum((pred - ref_grid)**2) / np.sum(ref_grid**2))
            print(f"  MF seed {seed}: L2={l2*100:.2f}%")

        mf_preds = np.array(mf_preds)
        mf_bv = compute_bias_variance(mf_preds, ref_grid)
        results[f'MF_Nterms{n_terms}'] = {
            **mf_bv,
            'lf_error_pct': lf_error_pct,
        }
        print(f"\n  MF (N_terms={n_terms}) decomposition:")
        print(f"    Bias  = {mf_bv['bias']*100:.2f}%  (fraction: {mf_bv['bias_frac']:.1%})")
        print(f"    Std   = {mf_bv['std']*100:.2f}%  (fraction: {mf_bv['var_frac']:.1%})")
        print(f"    MSE^½ = {np.sqrt(mf_bv['mse'])*100:.2f}%")

        # Delta analysis
        delta_var = van_bv['variance'] - mf_bv['variance']
        delta_bias = mf_bv['bias_sq'] - van_bv['bias_sq']
        print(f"\n  Transfer tradeoff:")
        print(f"    ΔVar  = {delta_var*1e4:.2f}×10⁻⁴  (positive = MF reduces variance)")
        print(f"    ΔBias²= {delta_bias*1e4:.2f}×10⁻⁴  (positive = MF increases bias)")
        print(f"    Net   = {(delta_var - delta_bias)*1e4:.2f}×10⁻⁴  (positive = MF helps)")
        print(f"    MF {'HELPS' if delta_var > delta_bias else 'HURTS'}: "
              f"ΔVar {'>' if delta_var > delta_bias else '<'} ΔBias²")

    # 4. Save
    json_path = out_dir / f'bias_variance_alpha{alpha}_ncol{n_col}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # Summary table
    print(f"\n{'#'*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'#'*70}")
    print(f"  {'Method':<20} {'Bias%':>8} {'Std%':>8} {'MSE½%':>8} {'Bias²/MSE':>10} {'Var/MSE':>10}")
    print(f"  {'-'*64}")
    print(f"  {'Vanilla':<20} {van_bv['bias']*100:>8.2f} {van_bv['std']*100:>8.2f} "
          f"{np.sqrt(van_bv['mse'])*100:>8.2f} {van_bv['bias_frac']:>10.1%} {van_bv['var_frac']:>10.1%}")
    for n_terms in args.n_terms:
        key = f'MF_Nterms{n_terms}'
        bv = results[key]
        lf_err = bv['lf_error_pct']
        print(f"  {'MF N='+str(n_terms)+f' ({lf_err:.0f}%)':<20} {bv['bias']*100:>8.2f} {bv['std']*100:>8.2f} "
              f"{np.sqrt(bv['mse'])*100:>8.2f} {bv['bias_frac']:>10.1%} {bv['var_frac']:>10.1%}")


if __name__ == '__main__':
    main()
