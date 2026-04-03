#!/usr/bin/env python3
"""
LR Control Experiment: Test whether MF advantage is confounded by learning rate.

The MF two-phase protocol uses:
  Phase 1: LR=5e-4, 5k epochs (data fitting)
  Phase 2: LR=1e-4, 20k epochs (physics fine-tuning)

Vanilla uses:
  LR=5e-4, 25k epochs (physics only)

This experiment runs Vanilla at LR=1e-4 (matching Phase 2) for 25k epochs
to test whether the lower LR alone explains the MF advantage.

Run:
  python3 scripts/experiments/run_lr_control.py --device cuda:3 --seeds 5
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
    results = {}
    x_ref = ref_data['x']
    u_grid = ref_data['u']
    t_vals = ref_data['t']
    with torch.no_grad():
        for j, t_val in enumerate(t_vals):
            x_t = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = u_grid[:, j]
            l2 = compute_l2_relative_error(u_pred, u_ref)
            results[f't{t_val:.2f}_u_l2'] = l2
    return results


CONFIGS = [
    # (alpha, N_col, N_terms, description)
    (0.5, 100, 20, 'alpha0.5_ncol100_nt20'),   # strongest MF advantage (3.11x)
    (0.7, 100, 12, 'alpha0.7_ncol100_nt12'),   # significant MF advantage (1.47x)
    (0.5, 100, 15, 'alpha0.5_ncol100_nt15'),   # moderate MF advantage (1.74x)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path('results/lr_control')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for alpha, n_col, n_terms, desc in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"  LR Control: alpha={alpha}, N_col={n_col}, N_terms={n_terms}")
        print(f"{'#'*70}")

        # Generate reference
        solver = FDRLaplaceSolver(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
        )
        ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
        if ref_path.exists():
            ref_data = dict(np.load(ref_path, allow_pickle=True))
        else:
            ref_data = solver.generate_reference_data(
                nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
            )

        # Generate LF data (used by MF, not by vanilla, but needed for trainer)
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, lf_N_terms=n_terms, lf_precision=20,
        )

        config_results = {'vanilla_lr5e4': [], 'vanilla_lr1e4': []}

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")

            # 1. Standard Vanilla (LR=5e-4, 25k epochs) as control
            print(f"  Vanilla (LR=5e-4)...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_std = {**MANUAL_HP, **PHYSICS, **BC,
                       'alpha': alpha, 'N_collocation': n_col}
            model_std = build_model(cfg_std)
            trainer_std = FDRTrainer(model_std, cfg_std, lf_data, device=device)
            t0 = time.time()
            trainer_std.train_vanilla(verbose=False)
            wall_std = time.time() - t0
            m_std = evaluate_model(model_std, ref_data, device)
            m_std['seed'] = seed
            m_std['wall_time'] = wall_std
            avg_std = np.mean([v for k, v in m_std.items() if k.endswith('_l2')])
            print(f"    L2={avg_std*100:.2f}%, wall={wall_std:.0f}s")
            config_results['vanilla_lr5e4'].append(m_std)

            # 2. Vanilla with LR=1e-4 (matching Phase 2)
            print(f"  Vanilla (LR=1e-4)...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_low = {**MANUAL_HP, **PHYSICS, **BC,
                       'alpha': alpha, 'N_collocation': n_col,
                       'lr': 1e-4}  # Override LR
            model_low = build_model(cfg_low)
            trainer_low = FDRTrainer(model_low, cfg_low, lf_data, device=device)
            t0 = time.time()
            trainer_low.train_vanilla(verbose=False)
            wall_low = time.time() - t0
            m_low = evaluate_model(model_low, ref_data, device)
            m_low['seed'] = seed
            m_low['wall_time'] = wall_low
            avg_low = np.mean([v for k, v in m_low.items() if k.endswith('_l2')])
            print(f"    L2={avg_low*100:.2f}%, wall={wall_low:.0f}s")
            config_results['vanilla_lr1e4'].append(m_low)

        # Summary for this config
        for method, results in config_results.items():
            avgs = [np.mean([v for k, v in r.items() if k.endswith('_l2')]) * 100
                    for r in results]
            print(f"\n  {method}: {np.mean(avgs):.2f}% ± {np.std(avgs, ddof=1):.2f}%")

        all_results[desc] = config_results

        # Save incrementally
        save_data = {}
        for cfg_key, cfg_results in all_results.items():
            save_data[cfg_key] = {}
            for method, results in cfg_results.items():
                avgs = [np.mean([v for k, v in r.items() if k.endswith('_l2')]) * 100
                        for r in results]
                save_data[cfg_key][method] = {
                    'l2_errors_pct': avgs,
                    'mean': float(np.mean(avgs)),
                    'std': float(np.std(avgs, ddof=1)),
                    'per_seed': results,
                }

        with open(out_dir / 'lr_control_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_dir / 'lr_control_results.json'}")

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"  FINAL LR CONTROL SUMMARY")
    print(f"{'#'*70}")
    for desc, config_results in all_results.items():
        print(f"\n  {desc}:")
        for method, results in config_results.items():
            avgs = [np.mean([v for k, v in r.items() if k.endswith('_l2')]) * 100
                    for r in results]
            print(f"    {method}: {np.mean(avgs):.2f}% ± {np.std(avgs, ddof=1):.2f}%")
        # Ratio
        std_avgs = [np.mean([v for k, v in r.items() if k.endswith('_l2')]) * 100
                    for r in config_results['vanilla_lr5e4']]
        low_avgs = [np.mean([v for k, v in r.items() if k.endswith('_l2')]) * 100
                    for r in config_results['vanilla_lr1e4']]
        ratio = np.mean(std_avgs) / np.mean(low_avgs) if np.mean(low_avgs) > 0 else float('inf')
        print(f"    LR=5e-4/LR=1e-4 ratio: {ratio:.2f}x")


if __name__ == '__main__':
    main()
