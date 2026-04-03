#!/usr/bin/env python3
"""
10-seed N_col sweep at critical budget levels (100, 200, 500, 2000).

Purpose: Achieve statistical significance (p < 0.05) for the MF advantage
at low N_col. With 5 seeds, Wilcoxon minimum p = 0.031; with 10 seeds,
minimum p = 0.001.

Run:
  python3 run_fdr_ncol_10seeds.py --alpha 0.5 --device cpu --seeds 10
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
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--ncol-values', type=int, nargs='+',
                        default=[100, 200, 500, 2000])
    parser.add_argument('--n-lf', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  10-SEED N_col Sweep (Statistical Significance)")
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

    LF_N_TERMS = 12
    LF_PRECISION = 20
    print(f"Generating {args.n_lf} genuine low-fidelity points (N_terms={LF_N_TERMS}, prec={LF_PRECISION})...")
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

        # Three methods: MF, Vanilla, Vanilla+N_LF (control: extra collocation = LF budget)
        methods = [
            ('MF', True, n_col),
            ('Vanilla', False, n_col),
            (f'Vanilla+{args.n_lf}', False, n_col + args.n_lf),
        ]
        for method, use_transfer, effective_ncol in methods:
            l2s = []
            walls = []
            run_config = {**config, 'N_collocation': effective_ncol}
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = FDRNet(
                    n_layers=run_config['n_layers'], n_neurons=run_config['n_neurons'],
                    activation=run_config['activation'], L=run_config['L'], T=run_config['T'],
                    hard_bc=True, bc_type=run_config['bc_type'],
                    bc_params=run_config['bc_params'],
                    fourier_features=run_config['fourier_features'],
                    fourier_sigma=run_config['fourier_sigma'],
                )
                trainer = FDRTrainer(model, run_config, lf_data, device=device)

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

            key = f'{method}_N{n_col}'
            all_results[key] = {
                'n_col': effective_ncol, 'method': method,
                'l2_mean': float(np.mean(l2s)), 'l2_std': float(np.std(l2s)),
                'l2_values': [float(v) for v in l2s],
                'wall_mean': float(np.mean(walls)),
            }
            print(f"  {method:>12s}: L2 = {np.mean(l2s)*100:.2f}% ± {np.std(l2s)*100:.2f}%  "
                  f"(wall {np.mean(walls):.0f}s)")

    # Summary with statistics
    from scipy import stats
    print(f"\n\n{'#'*70}")
    print(f"  10-SEED N_COL RESULTS  α={alpha}")
    print(f"{'#'*70}")
    ctrl_name = f'Vanilla+{args.n_lf}'
    print(f"\n  {'N_col':>6s}  {'MF L2 (%)':>12s}  {'Van L2 (%)':>12s}  {ctrl_name+' (%)':>14s}  {'MF/Van':>8s}  {'p(MF<Van)':>10s}  {'p(MF<Ctrl)':>11s}")
    print(f"  {'-'*90}")
    for n_col in args.ncol_values:
        mf = all_results[f'MF_N{n_col}']
        van = all_results[f'Vanilla_N{n_col}']
        ctrl = all_results.get(f'{ctrl_name}_N{n_col}', None)
        ratio = van['l2_mean'] / max(mf['l2_mean'], 1e-8)
        try:
            _, p_van = stats.wilcoxon(mf['l2_values'], van['l2_values'])
        except Exception:
            p_van = 1.0
        try:
            _, p_ctrl = stats.wilcoxon(mf['l2_values'], ctrl['l2_values']) if ctrl else (None, 1.0)
        except Exception:
            p_ctrl = 1.0
        sig_van = '***' if p_van < 0.001 else '**' if p_van < 0.01 else '*' if p_van < 0.05 else 'n.s.'
        sig_ctrl = '***' if p_ctrl < 0.001 else '**' if p_ctrl < 0.01 else '*' if p_ctrl < 0.05 else 'n.s.'
        ctrl_str = f"{ctrl['l2_mean']*100:>8.2f}±{ctrl['l2_std']*100:.2f}" if ctrl else "N/A"
        print(f"  {n_col:>6d}  {mf['l2_mean']*100:>8.2f}±{mf['l2_std']*100:.2f}  "
              f"{van['l2_mean']*100:>8.2f}±{van['l2_std']*100:.2f}  "
              f"{ctrl_str:>14s}  "
              f"{ratio:>6.1f}×  p={p_van:.4f}({sig_van})  p={p_ctrl:.4f}({sig_ctrl})")

    json_path = out_dir / f'ncol_sweep_10seeds_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
