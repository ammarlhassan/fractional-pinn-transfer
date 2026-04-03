#!/usr/bin/env python3
"""
Multi-α sweep at low N_col: shows where in α-space MF transfer helps most.

Run:  python3 run_fdr_alpha_sweep.py --device cuda --seeds 5
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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--ncol-values', type=int, nargs='+', default=[100, 200, 500])
    parser.add_argument('--alpha-values', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7, 0.9, 1.0])
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'#'*70}")
    print(f"  Multi-α Sweep: MF-fPINN vs Vanilla")
    print(f"  α values={args.alpha_values}")
    print(f"  N_col values={args.ncol_values}")
    print(f"  Seeds={args.seeds}, Device={device}")
    print(f"{'#'*70}\n")

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for alpha in args.alpha_values:
        print(f"\n{'='*70}")
        print(f"  α = {alpha}")
        print(f"{'='*70}")

        # Build solver and reference for this alpha
        solver = FDRLaplaceSolver(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
        )

        ref_path = out_dir / f'reference_alpha{alpha}.npz'
        if ref_path.exists():
            ref_data = dict(np.load(ref_path, allow_pickle=True))
        else:
            ref_data = solver.generate_reference_data(
                nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
            )

        LF_N_TERMS = 12
        LF_PRECISION = 20
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=args.n_lf, seed=42,
            lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
        )
        config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}

        for n_col in args.ncol_values:
            config['N_collocation'] = n_col
            print(f"\n  N_col = {n_col}")

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
                    print(f"    {method} seed {seed}: L2={l2*100:.2f}%  ({wall:.0f}s)")

                key = f'alpha{alpha}_{method}_N{n_col}'
                all_results[key] = {
                    'alpha': alpha, 'n_col': n_col, 'method': method,
                    'l2_mean': float(np.mean(l2s)), 'l2_std': float(np.std(l2s)),
                    'l2_values': [float(v) for v in l2s],
                    'wall_mean': float(np.mean(walls)),
                    'wall_std': float(np.std(walls)),
                }
                print(f"    {method:>8s}: L2 = {np.mean(l2s)*100:.2f}% ± {np.std(l2s)*100:.2f}%  "
                      f"(wall {np.mean(walls):.0f}s)")

    # Summary table
    print(f"\n\n{'#'*70}")
    print(f"  MULTI-α RESULTS")
    print(f"{'#'*70}")
    for n_col in args.ncol_values:
        print(f"\n  N_col = {n_col}")
        print(f"  {'α':>5s}  {'MF L2 (%)':>12s}  {'Van L2 (%)':>12s}  {'Advantage':>10s}  {'MF wall(s)':>10s}")
        print(f"  {'-'*58}")
        for alpha in args.alpha_values:
            mf_key = f'alpha{alpha}_MF_N{n_col}'
            van_key = f'alpha{alpha}_Vanilla_N{n_col}'
            if mf_key in all_results and van_key in all_results:
                mf = all_results[mf_key]
                van = all_results[van_key]
                ratio = van['l2_mean'] / max(mf['l2_mean'], 1e-8)
                print(f"  {alpha:>5.1f}  {mf['l2_mean']*100:>8.2f}±{mf['l2_std']*100:.2f}  "
                      f"{van['l2_mean']*100:>8.2f}±{van['l2_std']*100:.2f}  "
                      f"{ratio:>8.1f}×  {mf['wall_mean']:>10.0f}")

    json_path = out_dir / 'alpha_sweep.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
