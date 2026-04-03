#!/usr/bin/env python3
"""
Transfer efficiency curve: L2 error vs N_LF (number of Laplace data points).

This answers: "How many low-fidelity points does the MF-fPINN need?"

Run:  python3 run_fdr_transfer_curve.py [--alpha 0.5] [--device cuda] [--seeds 5]
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-lf-values', type=int, nargs='+',
                        default=[5, 10, 20, 30, 50, 100])
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    N_LF_values = args.n_lf_values

    print(f"\n{'#'*70}")
    print(f"  Transfer Efficiency Curve: α={alpha}")
    print(f"  N_LF values: {N_LF_values}")
    print(f"  Seeds: {args.seeds}")
    print(f"{'#'*70}\n")

    # Solver
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference data
    ref_path = out_dir / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}

    # Also run vanilla baseline (same total epochs as MF Phase1+Phase2)
    vanilla_results = {}
    print("\n--- Vanilla Baseline ---")
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        # Need dummy lf_data for trainer init
        dummy_lf = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=5, seed=seed, lf_N_terms=12, lf_precision=20,
        )
        model = FDRNet(
            n_layers=config['n_layers'], n_neurons=config['n_neurons'],
            activation=config['activation'], L=config['L'], T=config['T'],
            hard_bc=True, bc_type=config['bc_type'], bc_params=config['bc_params'],
            fourier_features=config['fourier_features'],
            fourier_sigma=config['fourier_sigma'],
        )
        trainer = FDRTrainer(model, config, dummy_lf, device=device)
        trainer.train_vanilla(verbose=False)

        model.eval()
        l2s = []
        with torch.no_grad():
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32,
                               device=device).unsqueeze(1)
            for j, t_val in enumerate(EVAL_TIMES):
                t_t = torch.full_like(x_t, t_val)
                u_pred = model(x_t, t_t).cpu().numpy().flatten()
                l2s.append(compute_l2_relative_error(u_pred, ref_data['u'][:, j]))
        vanilla_results.setdefault('l2_errors', []).append(np.mean(l2s))

    van_mean = np.mean(vanilla_results['l2_errors'])
    van_std = np.std(vanilla_results['l2_errors'])
    print(f"  Vanilla: L2 = {van_mean:.4f} ± {van_std:.4f}")

    # Transfer curve
    results = {}
    for n_lf in N_LF_values:
        print(f"\n--- N_LF = {n_lf} ---")
        l2_errors = []
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)

            lf_data = generate_low_fidelity_data(
                alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                N_LF=n_lf, seed=seed, lf_N_terms=12, lf_precision=20,
            )
            model = FDRNet(
                n_layers=config['n_layers'], n_neurons=config['n_neurons'],
                activation=config['activation'], L=config['L'], T=config['T'],
                hard_bc=True, bc_type=config['bc_type'], bc_params=config['bc_params'],
                fourier_features=config['fourier_features'],
                fourier_sigma=config['fourier_sigma'],
            )
            trainer = FDRTrainer(model, config, lf_data, device=device)
            trainer.train_full(verbose=False)

            model.eval()
            l2s = []
            with torch.no_grad():
                x_t = torch.tensor(ref_data['x'], dtype=torch.float32,
                                   device=device).unsqueeze(1)
                for j, t_val in enumerate(EVAL_TIMES):
                    t_t = torch.full_like(x_t, t_val)
                    u_pred = model(x_t, t_t).cpu().numpy().flatten()
                    l2s.append(compute_l2_relative_error(u_pred, ref_data['u'][:, j]))
            avg_l2 = np.mean(l2s)
            l2_errors.append(avg_l2)

        results[n_lf] = {
            'mean': float(np.mean(l2_errors)),
            'std': float(np.std(l2_errors)),
            'values': [float(v) for v in l2_errors],
        }
        print(f"  MF N_LF={n_lf}: L2 = {np.mean(l2_errors):.4f} ± {np.std(l2_errors):.4f}")

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  TRANSFER EFFICIENCY CURVE  α={alpha}")
    print(f"{'#'*70}")
    print(f"\n  {'N_LF':>5s}  {'L2 mean':>10s}  {'L2 std':>10s}  {'vs Vanilla':>12s}")
    print(f"  {'-'*45}")
    for n_lf in N_LF_values:
        r = results[n_lf]
        ratio = van_mean / r['mean'] if r['mean'] > 1e-8 else float('inf')
        print(f"  {n_lf:5d}  {r['mean']:10.4f}  {r['std']:10.4f}  {ratio:10.1f}×")
    print(f"  {'Van':>5s}  {van_mean:10.4f}  {van_std:10.4f}  {'1.0×':>12s}")

    # Save
    save_data = {
        'alpha': alpha,
        'vanilla': {'mean': float(van_mean), 'std': float(van_std),
                     'values': [float(v) for v in vanilla_results['l2_errors']]},
        'transfer': {str(k): v for k, v in results.items()},
    }
    json_path = out_dir / f'transfer_curve_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
