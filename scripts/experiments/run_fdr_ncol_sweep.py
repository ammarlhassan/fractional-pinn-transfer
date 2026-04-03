#!/usr/bin/env python3
"""
Collocation budget study: MF-fPINN vs Vanilla at different N_col.

Shows that MF transfer helps most when collocation budget is limited.

Run:  python3 run_fdr_ncol_sweep.py --alpha 0.5 --device cuda --seeds 5
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


def eval_model(model, ref_data, device):
    model.eval()
    x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    errs = []
    with torch.no_grad():
        for j, tv in enumerate(ref_data['t']):
            t_t = torch.full_like(x_t, float(tv))
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            errs.append(compute_l2_relative_error(u_pred, ref_data['u'][:, j]))
    model.train()
    return np.mean(errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--ncol-values', type=int, nargs='+',
                        default=[100, 200, 500, 1000, 2000])
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Collocation Budget Study: MF-fPINN vs Vanilla")
    print(f"  α={alpha}, N_col values={args.ncol_values}")
    print(f"  Seeds={args.seeds}, Device={device}")
    print(f"{'#'*70}\n")

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

    # Generate GENUINE low-fidelity data (degraded solver)
    LF_N_TERMS = 12      # degraded: 8 vs HF 30 (~40% L2 error)
    LF_PRECISION = 20   # degraded: 20 vs HF 30

    print(f"\nQuantifying LF solver degradation (N_terms={LF_N_TERMS}, precision={LF_PRECISION})...")
    lf_err = compute_lf_error(solver, lf_N_terms=LF_N_TERMS,
                              lf_precision=LF_PRECISION, N_test=50)
    print(f"  LF solver error vs HF: L2_rel = {lf_err['l2_relative']*100:.2f}%, "
          f"Linf_rel = {lf_err['linf_relative']*100:.2f}%")

    print(f"\nGenerating {args.n_lf} GENUINE low-fidelity training points...")
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, seed=42,
        lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
    )

    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}

    all_results = {}

    for n_col in args.ncol_values:
        config['N_collocation'] = n_col
        print(f"\n{'='*60}")
        print(f"  N_col = {n_col}")
        print(f"{'='*60}")

        for method, use_transfer in [('MF', True), ('Vanilla', False)]:
            l2s = []
            walls = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = FDRNet(
                    n_layers=config['n_layers'], n_neurons=config['n_neurons'],
                    activation=config['activation'], L=config['L'], T=config['T'],
                    hard_bc=True, bc_type=config['bc_type'],
                    bc_params=config['bc_params'],
                    fourier_features=config['fourier_features'],
                    fourier_sigma=config['fourier_sigma'],
                )
                trainer = FDRTrainer(model, config, lf_data, device=device)

                t0 = time.time()
                if use_transfer:
                    trainer.train_full(verbose=False)
                else:
                    trainer.train_vanilla(verbose=False)
                wall = time.time() - t0

                l2 = eval_model(model, ref_data, device)
                l2s.append(l2)
                walls.append(wall)

            key = f'{method}_N{n_col}'
            all_results[key] = {
                'n_col': n_col, 'method': method,
                'l2_mean': float(np.mean(l2s)), 'l2_std': float(np.std(l2s)),
                'l2_values': [float(v) for v in l2s],
                'wall_mean': float(np.mean(walls)),
            }
            print(f"  {method:>8s}: L2 = {np.mean(l2s):.4f} ± {np.std(l2s):.4f}  "
                  f"(wall {np.mean(walls):.0f}s)")

    # Summary table
    print(f"\n\n{'#'*70}")
    print(f"  COLLOCATION BUDGET RESULTS  α={alpha}")
    print(f"{'#'*70}")
    print(f"\n  {'N_col':>6s}  {'MF L2 (%)':>12s}  {'Van L2 (%)':>12s}  {'MF/Van':>8s}")
    print(f"  {'-'*48}")
    for n_col in args.ncol_values:
        mf = all_results[f'MF_N{n_col}']
        van = all_results[f'Vanilla_N{n_col}']
        ratio = van['l2_mean'] / max(mf['l2_mean'], 1e-8)
        print(f"  {n_col:>6d}  {mf['l2_mean']*100:>8.2f}±{mf['l2_std']*100:.2f}  "
              f"{van['l2_mean']*100:>8.2f}±{van['l2_std']*100:.2f}  "
              f"{ratio:>6.1f}×")

    json_path = out_dir / f'ncol_sweep_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
