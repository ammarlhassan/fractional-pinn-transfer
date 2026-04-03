#!/usr/bin/env python3
"""
Complete N_col budget study to 10 seeds for each alpha (run seeds 5-9, merge).

Paper Table 1 currently has 5 seeds. This upgrades to 10.
Uses N_terms=12 (the default LF quality).
Also includes Vanilla+50 control.

Usage:
  python3 run_ncol_budget_10seeds.py --device cuda:0 --alpha 0.5
"""

import argparse
import json
import time
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
    return float(np.mean(errs))


def run_single(method, config, lf_data, ref_data, seed, device, extra_ncol=0):
    """Run single experiment.
    
    For 'Vanilla+50' control: extra_ncol=50 adds 50 random collocation points.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if extra_ncol > 0:
        config = dict(config)
        config['N_collocation'] = config['N_collocation'] + extra_ncol

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
    if method == 'MF':
        trainer.train_full(verbose=False)
    else:
        trainer.train_vanilla(verbose=False)
    wall = time.time() - t0

    l2 = eval_model(model, ref_data, device)
    return {'l2': l2, 'seed': seed, 'wall': wall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--start-seed', type=int, default=5)
    parser.add_argument('--end-seed', type=int, default=10)
    parser.add_argument('--ncol-values', type=int, nargs='+', default=[100, 200, 500, 2000])
    parser.add_argument('--lf-nterms', type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    seeds = list(range(args.start_seed, args.end_seed))

    print(f"\n{'#'*70}")
    print(f"  N_col Budget 10-Seed Completion")
    print(f"  α = {alpha}, seeds = {seeds}")
    print(f"  N_col = {args.ncol_values}, N_terms = {args.lf_nterms}")
    print(f"  Device = {device}")
    print(f"{'#'*70}\n")

    out_dir = Path('results/fdr')

    # Reference data
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

    # LF data
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42,
        lf_N_terms=args.lf_nterms, lf_precision=20,
    )

    new_results = {}

    for n_col in args.ncol_values:
        config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
        config['N_collocation'] = n_col

        # MF, Vanilla, Vanilla+50
        methods = [
            ('MF', True, 0),
            ('Vanilla', False, 0),
            ('Vanilla+50', False, 50),
        ]

        for method_name, use_transfer, extra_ncol in methods:
            key = f'alpha{alpha}_{method_name}_N{n_col}'
            results = []
            for seed in seeds:
                r = run_single(
                    'MF' if use_transfer else 'Vanilla',
                    config, lf_data, ref_data, seed, device,
                    extra_ncol=extra_ncol,
                )
                results.append(r)
                print(f"  α={alpha} N_col={n_col:4d} {method_name:12s} seed={seed:2d}: L2={r['l2']*100:.2f}%  ({r['wall']:.0f}s)")
            new_results[key] = results

    # Save new seed data
    save_path = out_dir / f'ncol_budget_seeds5to9_alpha{alpha}_nterms{args.lf_nterms}.json'
    with open(save_path, 'w') as f:
        json.dump(new_results, f, indent=2, default=float)
    print(f"\nNew seed results saved to {save_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  NEW SEEDS (5-9) for α={alpha}")
    print(f"{'='*70}")
    for n_col in args.ncol_values:
        for method_name in ['MF', 'Vanilla', 'Vanilla+50']:
            key = f'alpha{alpha}_{method_name}_N{n_col}'
            if key in new_results:
                l2s = [r['l2'] for r in new_results[key]]
                print(f"  N_col={n_col:4d} {method_name:12s}: {np.mean(l2s)*100:.2f}±{np.std(l2s)*100:.2f}% ({len(l2s)} seeds)")

    print("\nDone!")


if __name__ == '__main__':
    main()
