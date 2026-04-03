#!/usr/bin/env python3
"""
Noise robustness study for MF-fPINN forward problem.

Shows MF transfer remains beneficial when LF data has noise,
addressing the criticism that "LF data is too clean."

Run:  python3 run_fdr_noise_robustness.py --device cuda --seeds 5
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
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--n-col', type=int, default=100,
                        help='Collocation points (use low value where MF shines)')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                        default=[0.0, 0.01, 0.05, 0.10, 0.20])
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Noise Robustness: MF-fPINN Forward Problem")
    print(f"  α={alpha}, N_col={args.n_col}, N_LF={args.n_lf}")
    print(f"  Noise levels: {args.noise_levels}")
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

    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': args.n_col}

    all_results = {}

    # First run vanilla baseline (no LF data, no noise effect)
    LF_N_TERMS = 12
    LF_PRECISION = 20
    print("Running Vanilla baseline (unaffected by LF noise)...")
    van_l2s = []
    lf_clean = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, seed=42,
        lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
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
        trainer = FDRTrainer(model, config, lf_clean, device=device)
        trainer.train_vanilla(verbose=False)
        van_l2s.append(eval_model(model, ref_data, device))

    van_mean = float(np.mean(van_l2s))
    van_std = float(np.std(van_l2s))
    all_results['vanilla'] = {
        'l2_mean': van_mean, 'l2_std': van_std,
        'l2_values': [float(v) for v in van_l2s],
    }
    print(f"  Vanilla: L2 = {van_mean*100:.2f}% ± {van_std*100:.2f}%\n")

    # Run MF at each noise level
    for noise in args.noise_levels:
        print(f"\n{'='*60}")
        print(f"  MF-fPINN with {noise*100:.0f}% noise on LF data")
        print(f"{'='*60}")

        mf_l2s = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate genuine LF data then add extra noise
            lf_data = generate_low_fidelity_data(
                alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                N_LF=args.n_lf, seed=42,
                lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
            )
            if noise > 0:
                rng = np.random.RandomState(seed + 1000)
                u_scale = np.std(lf_data['u'])
                noise_vec = rng.randn(len(lf_data['u'])) * noise * u_scale
                lf_data['u'] = lf_data['u'] + noise_vec

            model = FDRNet(
                n_layers=config['n_layers'], n_neurons=config['n_neurons'],
                activation=config['activation'], L=config['L'], T=config['T'],
                hard_bc=True, bc_type=config['bc_type'],
                bc_params=config['bc_params'],
                fourier_features=config['fourier_features'],
                fourier_sigma=config['fourier_sigma'],
            )
            trainer = FDRTrainer(model, config, lf_data, device=device)
            trainer.train_full(verbose=False)
            mf_l2s.append(eval_model(model, ref_data, device))

        key = f'MF_noise{noise}'
        all_results[key] = {
            'noise_level': noise,
            'l2_mean': float(np.mean(mf_l2s)),
            'l2_std': float(np.std(mf_l2s)),
            'l2_values': [float(v) for v in mf_l2s],
        }
        ratio = van_mean / max(np.mean(mf_l2s), 1e-8)
        print(f"  MF ({noise*100:.0f}% noise): L2 = {np.mean(mf_l2s)*100:.2f}% ± {np.std(mf_l2s)*100:.2f}%"
              f"  (vs Vanilla {van_mean*100:.2f}%, ratio={ratio:.1f}×)")

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  NOISE ROBUSTNESS SUMMARY (α={alpha}, N_col={args.n_col})")
    print(f"{'#'*70}")
    print(f"\n  Vanilla baseline: {van_mean*100:.2f}% ± {van_std*100:.2f}%\n")
    print(f"  {'Noise':>8s}  {'MF L2 (%)':>12s}  {'Advantage':>10s}")
    print(f"  {'-'*35}")
    for noise in args.noise_levels:
        r = all_results[f'MF_noise{noise}']
        ratio = van_mean / max(r['l2_mean'], 1e-8)
        print(f"  {noise*100:7.0f}%  {r['l2_mean']*100:>8.2f}±{r['l2_std']*100:.2f}  {ratio:>8.1f}×")

    json_path = out_dir / f'noise_robustness_alpha{alpha}_ncol{args.n_col}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
