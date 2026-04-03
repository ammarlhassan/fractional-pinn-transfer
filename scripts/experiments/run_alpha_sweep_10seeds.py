#!/usr/bin/env python3
"""
Complete alpha sweep to 10 seeds (run seeds 5-9, merge with existing 0-4).

This addresses the reviewer critique: "Why are some regimes 5 seeds and others 10?"

Also runs N_col budget at 10 seeds for the key table entries.

Usage:
  python3 run_alpha_sweep_10seeds.py --device cuda:2 --alpha 0.5
  python3 run_alpha_sweep_10seeds.py --device cuda:3 --alpha 0.7
  python3 run_alpha_sweep_10seeds.py --device cuda:4 --alpha 1.0
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
    return float(np.mean(errs))


def run_single(method, config, lf_data, ref_data, seed, device):
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
    parser.add_argument('--ncol-values', type=int, nargs='+', default=[100, 200, 500])
    parser.add_argument('--lf-nterms', type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    seeds = list(range(args.start_seed, args.end_seed))

    print(f"\n{'#'*70}")
    print(f"  Alpha Sweep 10-Seed Completion")
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

    # LF data (same deterministic seed=42 as original)
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

        for method in ['MF', 'Vanilla']:
            key = f'alpha{alpha}_{method}_N{n_col}'
            results = []
            for seed in seeds:
                r = run_single(method, config, lf_data, ref_data, seed, device)
                results.append(r)
                print(f"  α={alpha} N_col={n_col:4d} {method:7s} seed={seed:2d}: L2={r['l2']*100:.2f}%  ({r['wall']:.0f}s)")
            new_results[key] = results

    # Save new seeds
    save_path = out_dir / f'alpha_sweep_seeds{args.start_seed}to{args.end_seed-1}_alpha{alpha}.json'
    with open(save_path, 'w') as f:
        json.dump(new_results, f, indent=2, default=float)
    print(f"\nNew seed results saved to {save_path}")

    # Merge with existing data
    existing_path = out_dir / 'alpha_sweep.json'
    if existing_path.exists():
        existing = json.load(open(existing_path))
        merged = {}
        for key in set(list(existing.keys()) + list(new_results.keys())):
            if key in existing and key in new_results:
                old_l2 = existing[key].get('l2_values', [])
                new_l2 = [r['l2'] for r in new_results[key]]
                all_l2 = old_l2 + new_l2
                old_walls = [existing[key].get('wall_mean', 0)] * len(old_l2)
                new_walls = [r['wall'] for r in new_results[key]]
                all_walls = old_walls + new_walls
                merged[key] = {
                    'alpha': existing[key]['alpha'],
                    'n_col': existing[key]['n_col'],
                    'method': existing[key]['method'],
                    'l2_mean': float(np.mean(all_l2)),
                    'l2_std': float(np.std(all_l2)),
                    'l2_values': [float(v) for v in all_l2],
                    'wall_mean': float(np.mean(all_walls)),
                    'n_seeds': len(all_l2),
                }
            elif key in existing:
                merged[key] = existing[key]
                merged[key]['n_seeds'] = len(existing[key].get('l2_values', []))
            else:
                l2s = [r['l2'] for r in new_results[key]]
                merged[key] = {
                    'l2_mean': float(np.mean(l2s)),
                    'l2_std': float(np.std(l2s)),
                    'l2_values': [float(v) for v in l2s],
                    'n_seeds': len(l2s),
                }

        merged_path = out_dir / 'alpha_sweep_10seeds.json'
        with open(merged_path, 'w') as f:
            json.dump(merged, f, indent=2, default=float)
        print(f"Merged 10-seed results saved to {merged_path}")

        # Print summary for this alpha
        print(f"\n{'='*70}")
        print(f"  10-SEED RESULTS for α={alpha}")
        print(f"{'='*70}")
        for n_col in args.ncol_values:
            mf_key = f'alpha{alpha}_MF_N{n_col}'
            van_key = f'alpha{alpha}_Vanilla_N{n_col}'
            if mf_key in merged and van_key in merged:
                mf = merged[mf_key]
                van = merged[van_key]
                ratio = van['l2_mean'] / max(mf['l2_mean'], 1e-8)
                print(f"  N_col={n_col:4d}: MF={mf['l2_mean']*100:.2f}±{mf['l2_std']*100:.2f}%  "
                      f"Van={van['l2_mean']*100:.2f}±{van['l2_std']*100:.2f}%  "
                      f"Ratio={ratio:.2f}×  (n={mf.get('n_seeds','?')}/{van.get('n_seeds','?')} seeds)")

    print("\nDone!")


if __name__ == '__main__':
    main()
