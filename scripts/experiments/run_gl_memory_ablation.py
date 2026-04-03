#!/usr/bin/env python3
"""
GL Memory Depth Ablation — THE key mechanistic test.

If the GL memory hypothesis is correct, then:
  - Vanilla fPINN should get WORSE when we truncate gl_N_memory at fixed α=0.5
  - MF-fPINN should be LESS affected (it has the warm start)
  - The MF advantage ratio should INCREASE with truncation

This is the direct causal test the reviewer asked for.

Design:
  α = 0.5 (fixed), N_col = 100 (low collocation)
  gl_N_memory ∈ {2, 5, 10, 20, 50, 100}
  Methods: Vanilla, MF (N_terms=15 LF, ~9% error)
  10 seeds per configuration

Usage:
  python3 run_gl_memory_ablation.py --device cuda:0 --seeds 10
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver,
    generate_low_fidelity_data,
    compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha, gl_N_memory, N_col=100):
    """Build config with overridden gl_N_memory and N_col."""
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    cfg['gl_N_memory'] = gl_N_memory
    cfg['N_collocation'] = N_col
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


def run_single(method, config, lf_data, ref_data, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config)
    trainer = FDRTrainer(model, config, lf_data, device=device)

    t0 = time.time()
    if method == 'mf':
        trainer.train_full(verbose=False)
    else:
        trainer.train_vanilla(verbose=False)
    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, device)
    l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
    avg_l2 = float(np.mean(l2_vals))
    return {'avg_l2': avg_l2, 'seed': seed, 'wall_time': wall, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--ncol', type=int, default=100)
    parser.add_argument('--lf_nterms', type=int, default=15,
                        help='LF quality: 15 gives ~9%% error at alpha=0.5')
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    GL_MEMORY_VALUES = [2, 5, 10, 20, 50, 100]

    print(f"\n{'#'*70}")
    print(f"  GL MEMORY DEPTH ABLATION — Causal Mechanism Test")
    print(f"  α = {alpha}, N_col = {args.ncol}, device = {device}")
    print(f"  Seeds = {args.seeds}, LF N_terms = {args.lf_nterms}")
    print(f"  GL memory values: {GL_MEMORY_VALUES}")
    print(f"{'#'*70}\n")

    # Reference data
    print("Generating reference solution...")
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/gl_memory_ablation')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # LF data
    print(f"Generating LF data (N_terms={args.lf_nterms})...")
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=args.lf_nterms, lf_precision=20,
    )

    lf_err = compute_lf_error(solver, lf_N_terms=args.lf_nterms, lf_precision=20, N_test=50)
    print(f"  LF error: {lf_err['l2_relative']*100:.1f}%")

    # Run sweep
    all_results = {}

    for gl_mem in GL_MEMORY_VALUES:
        print(f"\n{'='*60}")
        print(f"  gl_N_memory = {gl_mem}")
        print(f"{'='*60}")

        config = build_config(alpha, gl_N_memory=gl_mem, N_col=args.ncol)

        for method in ['vanilla', 'mf']:
            key = f'gl{gl_mem}_{method}'
            results = []
            for seed in range(args.seeds):
                r = run_single(method, config, lf_data, ref_data, seed, device)
                results.append(r)
                print(f"  gl_mem={gl_mem:3d} | {method:7s} | seed {seed:2d} | L2={r['avg_l2']*100:.2f}% | {r['wall_time']:.0f}s")

            all_results[key] = results

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  GL MEMORY ABLATION SUMMARY  (α={alpha}, N_col={args.ncol}, {args.seeds} seeds)")
    print(f"{'#'*70}")
    print(f"\n{'gl_mem':>8} | {'Vanilla L2':>14} | {'MF L2':>14} | {'Ratio':>8} | {'Prediction':>12}")
    print('-' * 70)

    for gl_mem in GL_MEMORY_VALUES:
        van_key = f'gl{gl_mem}_vanilla'
        mf_key = f'gl{gl_mem}_mf'
        van_l2s = [r['avg_l2'] for r in all_results[van_key]]
        mf_l2s = [r['avg_l2'] for r in all_results[mf_key]]
        van_mean = np.mean(van_l2s) * 100
        van_std = np.std(van_l2s) * 100
        mf_mean = np.mean(mf_l2s) * 100
        mf_std = np.std(mf_l2s) * 100
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')

        # Prediction: if GL memory hypothesis is correct,
        # truncation should hurt Vanilla more than MF
        if gl_mem <= 5:
            pred = "MF should win"
        elif gl_mem >= 50:
            pred = "~neutral"
        else:
            pred = "MF advantage"

        print(f"  {gl_mem:6d} | {van_mean:6.2f}±{van_std:5.2f}% | {mf_mean:6.2f}±{mf_std:5.2f}% | {ratio:6.2f}x | {pred}")

    # Save
    save_path = out_dir / f'ablation_alpha{alpha}_ncol{args.ncol}_nterms{args.lf_nterms}.json'
    save_data = {
        'alpha': alpha, 'ncol': args.ncol, 'seeds': args.seeds,
        'lf_nterms': args.lf_nterms, 'lf_error_pct': lf_err['l2_relative'] * 100,
        'gl_memory_values': GL_MEMORY_VALUES,
        'results': {k: v for k, v in all_results.items()},
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nResults saved to {save_path}")

    # Also write log
    log_path = out_dir / f'log_ablation_alpha{alpha}.txt'
    with open(log_path, 'w') as f:
        f.write(f"GL Memory Depth Ablation\n")
        f.write(f"alpha={alpha}, ncol={args.ncol}, seeds={args.seeds}, lf_nterms={args.lf_nterms}\n")
        f.write(f"LF error: {lf_err['l2_relative']*100:.1f}%\n\n")
        for gl_mem in GL_MEMORY_VALUES:
            van_l2s = [r['avg_l2'] for r in all_results[f'gl{gl_mem}_vanilla']]
            mf_l2s = [r['avg_l2'] for r in all_results[f'gl{gl_mem}_mf']]
            ratio = np.mean(van_l2s) / np.mean(mf_l2s) if np.mean(mf_l2s) > 0 else float('inf')
            f.write(f"gl_mem={gl_mem:3d}: Van={np.mean(van_l2s)*100:.2f}±{np.std(van_l2s)*100:.2f}%  "
                    f"MF={np.mean(mf_l2s)*100:.2f}±{np.std(mf_l2s)*100:.2f}%  Ratio={ratio:.2f}x\n")
    print(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()
