#!/usr/bin/env python3
"""
Phase 1 Duration Ablation — How many Phase 1 epochs are needed?

Varies Phase 1 epochs: {0, 500, 1000, 2000, 5000, 10000}
while keeping Phase 2 fixed at 20000 epochs.
Total compute budget increases with more Phase 1, showing
the cost-benefit tradeoff.

Run:  python3 run_phase1_ablation.py --alpha 0.5 --device cuda:4 --seeds 5
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
    l2_vals = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            l2 = compute_l2_relative_error(u_pred, ref_data['u'][:, j])
            l2_vals.append(l2)
    return np.mean(l2_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    n_col = args.n_col

    p1_values = [0, 500, 1000, 2000, 5000, 10000]

    print(f"\n{'#'*70}")
    print(f"  PHASE 1 DURATION ABLATION")
    print(f"  α={alpha}, N_col={n_col}, N_terms={args.n_terms}, Seeds={args.seeds}")
    print(f"  Phase 1 epochs: {p1_values}")
    print(f"{'#'*70}\n")

    # Reference
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
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # LF data
    lf_err = compute_lf_error(solver, lf_N_terms=args.n_terms, lf_precision=20, N_test=50)
    print(f"  LF error (N_terms={args.n_terms}): {lf_err['l2_relative']*100:.2f}%")

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=args.n_terms, lf_precision=20,
    )

    results = {}

    for p1_epochs in p1_values:
        print(f"\n{'='*60}")
        print(f"  Phase 1 = {p1_epochs} epochs")
        print(f"{'='*60}")

        l2_vals = []
        wall_vals = []

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = {**MANUAL_HP, **PHYSICS, **BC,
                      'alpha': alpha,
                      'N_collocation': n_col,
                      'phase1_epochs': p1_epochs}

            model = build_model(config)
            trainer = FDRTrainer(model, config, lf_data, device=device)

            t0 = time.time()
            if p1_epochs == 0:
                # Pure vanilla (same total epochs as Phase 2)
                trainer.train_vanilla(verbose=False)
            else:
                trainer.train_full(verbose=False)
            wall = time.time() - t0

            l2 = evaluate_model(model, ref_data, device)
            l2_vals.append(l2)
            wall_vals.append(wall)
            print(f"    Seed {seed}: L2={l2*100:.2f}%  ({wall:.0f}s)")

        mean_l2 = np.mean(l2_vals)
        std_l2 = np.std(l2_vals)
        mean_wall = np.mean(wall_vals)

        results[str(p1_epochs)] = {
            'l2_mean': mean_l2,
            'l2_std': std_l2,
            'l2_values': l2_vals,
            'wall_mean': mean_wall,
        }
        print(f"    → L2={mean_l2*100:.2f}±{std_l2*100:.2f}%  Wall={mean_wall:.0f}s")

    # Summary
    print(f"\n{'#'*70}")
    print(f"  SUMMARY")
    print(f"{'#'*70}")
    print(f"  {'P1 Epochs':>10} {'L2 Mean%':>10} {'L2 Std%':>10} {'Wall(s)':>10} {'vs P1=0':>10}")
    print(f"  {'-'*50}")
    base_l2 = results['0']['l2_mean']
    for p1 in p1_values:
        r = results[str(p1)]
        ratio = base_l2 / r['l2_mean'] if r['l2_mean'] > 1e-10 else float('inf')
        print(f"  {p1:>10d} {r['l2_mean']*100:>10.2f} {r['l2_std']*100:>10.2f} "
              f"{r['wall_mean']:>10.0f} {ratio:>10.2f}×")

    json_path = out_dir / f'phase1_ablation_alpha{alpha}_ncol{n_col}_nterms{args.n_terms}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
