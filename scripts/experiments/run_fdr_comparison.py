#!/usr/bin/env python3
"""
MF-fPINN vs Vanilla fPINN comparison for fractional diffusion-reaction.

Run:  python3 run_fdr_comparison.py [--alpha 0.5] [--device cuda] [--seeds 5]
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver,
    generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha):
    """Merge physics + HP config for a given α."""
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    return cfg


def build_model(config):
    """Create a fresh FDRNet from config."""
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
    """Compute L2 errors at each time snapshot."""
    model.eval()
    results = {}
    x_ref = ref_data['x']
    u_grid = ref_data['u']   # shape (nx, nt)
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


def run_single(method, config, lf_data, ref_data, seed, device, verbose=True):
    """Run one seed of one method."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config)
    trainer = FDRTrainer(model, config, lf_data, device=device)

    t0 = time.time()
    if method == 'M3_mf':
        trainer.train_full(verbose=verbose)
    elif method == 'M1_vanilla':
        trainer.train_vanilla(verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics, model, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n_lf', type=int, default=50)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    verbose = not args.quiet
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Fractional Diffusion-Reaction: MF-fPINN vs Vanilla")
    print(f"  α = {alpha}, device = {device}, seeds = {args.seeds}")
    print(f"{'#'*70}\n")

    # ── 1. Generate reference data ──────────────────────────
    print("Generating reference solution...")
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # Quick sanity: print peak value at first time
    print(f"  u peak at t={EVAL_TIMES[0]}: {ref_data['u'][:, 0].max():.4f}")

    # ── 2. Generate GENUINE low-fidelity training data ─────
    LF_N_TERMS = 12      # degraded: 8 vs HF 30 (~40% L2 error)
    LF_PRECISION = 20   # degraded: 20 vs HF 30

    # Quantify LF degradation
    print(f"\nQuantifying LF solver degradation (N_terms={LF_N_TERMS}, precision={LF_PRECISION})...")
    lf_err = compute_lf_error(solver, lf_N_terms=LF_N_TERMS,
                              lf_precision=LF_PRECISION, N_test=50)
    print(f"  LF solver error vs HF: L2_rel = {lf_err['l2_relative']*100:.2f}%, "
          f"Linf_rel = {lf_err['linf_relative']*100:.2f}%")

    print(f"\nGenerating {args.n_lf} GENUINE low-fidelity training points...")
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
    )

    # ── 3. Run comparison ──────────────────────────────────
    config = build_config(alpha)
    all_results = {}

    for method in ['M3_mf', 'M1_vanilla']:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        method_results = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            metrics, model, trainer = run_single(
                method, config, lf_data, ref_data, seed, device, verbose
            )
            method_results.append(metrics)

            # Print summary for this seed
            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            avg = np.mean(l2_vals)
            print(f"  → Avg L2: {avg:.4f}  |  Wall: {metrics['wall_time']:.1f}s")

            # Save best model
            if seed == 0:
                ckpt_path = out_dir / f'{method}_alpha{alpha}_seed0.pt'
                trainer.save_checkpoint(str(ckpt_path))

        all_results[method] = method_results

    # ── 4. Summary ──────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  RESULTS SUMMARY  (α = {alpha})")
    print(f"{'#'*70}")

    for method, results in all_results.items():
        print(f"\n  {method}:")
        for t_val in EVAL_TIMES:
            key = f't{t_val:.2f}_u_l2'
            vals = [r[key] for r in results]
            print(f"    t={t_val:.2f}:  L2 = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        walls = [r['wall_time'] for r in results]
        print(f"    Wall time: {np.mean(walls):.1f}s ± {np.std(walls):.1f}s")

    # Compare
    print(f"\n  Improvement (MF over Vanilla):")
    for t_val in EVAL_TIMES:
        key = f't{t_val:.2f}_u_l2'
        mf = np.mean([r[key] for r in all_results['M3_mf']])
        van = np.mean([r[key] for r in all_results['M1_vanilla']])
        if van > 1e-8:
            ratio = van / mf
            print(f"    t={t_val:.2f}:  {ratio:.1f}× better")

    # Save
    json_path = out_dir / f'results_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
