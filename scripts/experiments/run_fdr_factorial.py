#!/usr/bin/env python3
"""
Full 2×2 factorial experiment: M1 (vanilla), M2 (vanilla+HPO), M3 (MF), M4 (MF+HPO).

Run:  python3 run_fdr_factorial.py --alpha 0.5 --device cuda --seeds 10 --hpo-trials 50
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
from mf_fpinn.training.fdr_hpo import FDRHPORunner
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_model(config):
    return FDRNet(
        n_layers=config['n_layers'], n_neurons=config['n_neurons'],
        activation=config['activation'], L=config['L'], T=config['T'],
        hard_bc=True, bc_type=config['bc_type'], bc_params=config['bc_params'],
        fourier_features=config['fourier_features'],
        fourier_sigma=config['fourier_sigma'],
    )


def evaluate_model(model, ref_data, device):
    model.eval()
    results = {}
    x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            t_t = torch.full_like(x_t, float(t_val))
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            results[f't{float(t_val):.2f}_u_l2'] = compute_l2_relative_error(
                u_pred, ref_data['u'][:, j])
    return results


def run_method(method, config, lf_data, ref_data, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(config)
    trainer = FDRTrainer(model, config, lf_data, device=device)

    t0 = time.time()
    if method in ('M1_vanilla', 'M2_vanilla_hpo'):
        trainer.train_vanilla(verbose=False)
    else:
        trainer.train_full(verbose=False)
    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--hpo-trials', type=int, default=50)
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip HPO, use manual HPs for all methods')
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Full 2×2 Factorial | α={alpha} | seeds={args.seeds}")
    print(f"  HPO trials: {args.hpo_trials} | Device: {device}")
    print(f"{'#'*70}\n")

    # Setup
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

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, seed=42, lf_N_terms=8, lf_precision=20,
    )

    # Base config (manual HPs)
    base_config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}

    all_results = {}
    configs_used = {}

    # --- M1: Vanilla with manual HPs ---
    print(f"\n{'='*60}")
    print(f"  M1: Vanilla fPINN (manual HPs)")
    print(f"{'='*60}")
    m1_results = []
    for seed in range(args.seeds):
        metrics = run_method('M1_vanilla', base_config, lf_data, ref_data, seed, device)
        m1_results.append(metrics)
        l2s = [v for k, v in metrics.items() if k.endswith('_l2')]
        print(f"  Seed {seed}: avg L2={np.mean(l2s):.4f}  wall={metrics['wall_time']:.1f}s")
    all_results['M1_vanilla'] = m1_results
    configs_used['M1'] = base_config

    # --- M3: MF-fPINN with manual HPs ---
    print(f"\n{'='*60}")
    print(f"  M3: MF-fPINN (manual HPs)")
    print(f"{'='*60}")
    m3_results = []
    for seed in range(args.seeds):
        metrics = run_method('M3_mf', base_config, lf_data, ref_data, seed, device)
        m3_results.append(metrics)
        l2s = [v for k, v in metrics.items() if k.endswith('_l2')]
        print(f"  Seed {seed}: avg L2={np.mean(l2s):.4f}  wall={metrics['wall_time']:.1f}s")
    all_results['M3_mf'] = m3_results
    configs_used['M3'] = base_config

    # --- HPO for M2 and M4 ---
    if not args.skip_hpo:
        physics = {'alpha': alpha, **PHYSICS}
        bc_cfg = dict(BC)

        # M2: Vanilla + HPO
        print(f"\n{'='*60}")
        print(f"  M2: HPO for Vanilla fPINN ({args.hpo_trials} trials)")
        print(f"{'='*60}")
        runner_m2 = FDRHPORunner(
            lf_data, ref_data, physics, bc_cfg, device,
            study_name=f'fdr_M2_alpha{alpha}',
        )
        study_m2 = runner_m2.run_study(args.hpo_trials, use_transfer=False)
        runner_m2.analyze_importance(study_m2)
        m2_results = runner_m2.run_best_multi_seed(study_m2, args.seeds, use_transfer=False)
        all_results['M2_vanilla_hpo'] = m2_results
        configs_used['M2'] = study_m2.best_params

        # M4: MF-fPINN + HPO
        print(f"\n{'='*60}")
        print(f"  M4: HPO for MF-fPINN ({args.hpo_trials} trials)")
        print(f"{'='*60}")
        runner_m4 = FDRHPORunner(
            lf_data, ref_data, physics, bc_cfg, device,
            study_name=f'fdr_M4_alpha{alpha}',
        )
        study_m4 = runner_m4.run_study(args.hpo_trials, use_transfer=True)
        runner_m4.analyze_importance(study_m4)
        m4_results = runner_m4.run_best_multi_seed(study_m4, args.seeds, use_transfer=True)
        all_results['M4_mf_hpo'] = m4_results
        configs_used['M4'] = study_m4.best_params

    # --- Summary ---
    print(f"\n\n{'#'*70}")
    print(f"  FACTORIAL RESULTS  α={alpha}")
    print(f"{'#'*70}")

    t_keys = sorted([k for k in all_results[list(all_results.keys())[0]][0].keys()
                      if k.endswith('_l2')])

    for method, results in all_results.items():
        print(f"\n  {method}:")
        for key in t_keys:
            vals = [r[key] for r in results]
            print(f"    {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        walls = [r['wall_time'] for r in results]
        print(f"    Wall: {np.mean(walls):.1f}s ± {np.std(walls):.1f}s")

    # Save
    json_path = out_dir / f'factorial_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'alpha': alpha,
            'seeds': args.seeds,
            'results': all_results,
            'configs': {k: str(v) for k, v in configs_used.items()},
        }, f, indent=2, default=str)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
