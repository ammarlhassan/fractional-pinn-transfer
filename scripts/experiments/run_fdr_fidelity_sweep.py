#!/usr/bin/env python3
"""
Fidelity gap sweep: How does LF data quality affect MF-fPINN advantage?

Varies the LF solver accuracy (N_terms) to create different fidelity gaps,
then compares MF-fPINN vs Vanilla at each level.

This is the KEY experiment: maps out when MF transfer helps vs hurts.

Run:  python3 run_fdr_fidelity_sweep.py --alpha 0.5 --device cuda --seeds 5
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
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--lf-nterms', type=int, nargs='+',
                        default=[5, 8, 10, 12, 15, 20])
    parser.add_argument('--lf-precision', type=int, default=20)
    parser.add_argument('--lbfgs', action='store_true', default=True,
                        help='Use L-BFGS in Phase 1 (default: True)')
    parser.add_argument('--no-lbfgs', dest='lbfgs', action='store_false')
    parser.add_argument('--w-data', type=float, default=10.0,
                        help='Data weight in Phase 2 (default: 10)')
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Fidelity Gap Sweep: MF-fPINN vs Vanilla")
    print(f"  α={alpha}, N_col={args.n_col}, N_LF={args.n_lf}")
    print(f"  LF N_terms values: {args.lf_nterms}")
    print(f"  Phase 1 L-BFGS: {args.lbfgs}, w_data={args.w_data}")
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

    # Run Vanilla baseline first (doesn't depend on LF quality)
    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha,
              'N_collocation': args.n_col}

    print("--- Vanilla Baseline ---")
    vanilla_l2s = []
    # Need dummy LF data for trainer constructor
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, seed=42, lf_N_terms=15, lf_precision=20,
    )
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
        trainer = FDRTrainer(model, config, dummy_lf, device=device)
        t0 = time.time()
        trainer.train_vanilla(verbose=False)
        wall = time.time() - t0
        l2 = eval_model(model, ref_data, device)
        vanilla_l2s.append(l2)
        print(f"  Vanilla seed {seed}: L2={l2*100:.2f}%  ({wall:.0f}s)")
    vanilla_mean = np.mean(vanilla_l2s)
    vanilla_std = np.std(vanilla_l2s)
    print(f"  Vanilla: {vanilla_mean*100:.2f}% ± {vanilla_std*100:.2f}%\n")

    all_results = {
        'vanilla': {
            'l2_mean': float(vanilla_mean),
            'l2_std': float(vanilla_std),
            'l2_values': [float(v) for v in vanilla_l2s],
        }
    }

    # Sweep over LF quality levels
    for n_terms in args.lf_nterms:
        print(f"\n{'='*60}")
        print(f"  LF N_terms = {n_terms}")
        print(f"{'='*60}")

        # Compute LF error for this setting
        lf_err = compute_lf_error(solver, lf_N_terms=n_terms,
                                   lf_precision=args.lf_precision, N_test=50)
        lf_l2 = lf_err['l2_relative']
        print(f"  LF quality: {lf_l2*100:.1f}% L2 error vs HF")

        # Generate LF data
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=args.n_lf, seed=42,
            lf_N_terms=n_terms, lf_precision=args.lf_precision,
        )

        # Override config for this run
        run_config = {**config,
                      'phase1_lbfgs': args.lbfgs,
                      'phase1_lbfgs_steps': 50 if args.lbfgs else 0,
                      'w_data': args.w_data}

        mf_l2s = []
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
            trainer = FDRTrainer(model, run_config, lf_data, device=device)
            t0 = time.time()
            trainer.train_full(verbose=False)
            wall = time.time() - t0
            l2 = eval_model(model, ref_data, device)
            mf_l2s.append(l2)
            print(f"    MF seed {seed}: L2={l2*100:.2f}%  ({wall:.0f}s)")

        mf_mean = np.mean(mf_l2s)
        mf_std = np.std(mf_l2s)
        ratio = vanilla_mean / max(mf_mean, 1e-8)
        print(f"  MF: {mf_mean*100:.2f}% ± {mf_std*100:.2f}%  "
              f"(advantage: {ratio:.2f}×)")

        all_results[f'MF_Nterms{n_terms}'] = {
            'n_terms': n_terms,
            'lf_error': float(lf_l2),
            'l2_mean': float(mf_mean),
            'l2_std': float(mf_std),
            'l2_values': [float(v) for v in mf_l2s],
            'advantage': float(ratio),
        }

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  FIDELITY GAP SWEEP SUMMARY (α={alpha}, N_col={args.n_col})")
    print(f"  Vanilla baseline: {vanilla_mean*100:.2f}% ± {vanilla_std*100:.2f}%")
    print(f"{'#'*70}")
    print(f"\n  {'N_terms':>7s}  {'LF err':>8s}  {'MF L2':>10s}  {'Advantage':>10s}")
    print(f"  {'-'*45}")
    for n_terms in args.lf_nterms:
        key = f'MF_Nterms{n_terms}'
        r = all_results[key]
        print(f"  {n_terms:>7d}  {r['lf_error']*100:>6.1f}%  "
              f"{r['l2_mean']*100:>6.2f}±{r['l2_std']*100:.2f}%  "
              f"{r['advantage']:>8.2f}×")

    lbfgs_str = 'lbfgs' if args.lbfgs else 'nolbfgs'
    json_path = out_dir / f'fidelity_sweep_alpha{alpha}_{lbfgs_str}_wdata{args.w_data}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
