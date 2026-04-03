#!/usr/bin/env python3
"""
HPO for fPINN variants (M2: vanilla+HPO, M4: MF-fPINN+HPO).

Run:
  python3 run_fdr_hpo.py --method M2 --alpha 0.5 --device cuda --trials 50
  python3 run_fdr_hpo.py --method M4 --alpha 0.5 --device cuda --trials 50
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data,
)
from mf_fpinn.training.fdr_hpo import FDRHPORunner
from mf_fpinn.experiments.fdr_configs import PHYSICS, BC, EVAL_TIMES, REFERENCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['M2', 'M4'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--final-seeds', type=int, default=10)
    parser.add_argument('--n-lf', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    use_transfer = args.method == 'M4'

    print(f"\n{'#'*70}")
    print(f"  HPO for {args.method} | α={alpha} | trials={args.trials}")
    print(f"  Transfer: {use_transfer} | Device: {device}")
    print(f"{'#'*70}\n")

    # Reference and solver
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
        print(f"  Loaded reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, seed=42, lf_N_terms=8, lf_precision=20,
    )

    physics = {'alpha': alpha, **PHYSICS}
    bc_cfg = dict(BC)

    runner = FDRHPORunner(
        lf_data=lf_data, ref_data=ref_data,
        physics=physics, bc_cfg=bc_cfg,
        device=device,
        study_name=f'fdr_{args.method}_alpha{alpha}',
    )

    # Run HPO
    study = runner.run_study(
        n_trials=args.trials, use_transfer=use_transfer, seed=42,
    )

    # Importance analysis
    importance = runner.analyze_importance(study)

    # Multi-seed evaluation with best HPs
    print(f"\n{'='*60}")
    print(f"  Re-training best config with {args.final_seeds} seeds")
    print(f"{'='*60}")
    final_results = runner.run_best_multi_seed(
        study, n_seeds=args.final_seeds, use_transfer=use_transfer,
    )

    # Save everything
    save_data = {
        'method': args.method,
        'alpha': alpha,
        'use_transfer': use_transfer,
        'n_trials': args.trials,
        'best_params': study.best_params,
        'best_l2': study.best_value,
        'importance': importance,
        'final_results': final_results,
    }
    json_path = out_dir / f'hpo_{args.method}_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
