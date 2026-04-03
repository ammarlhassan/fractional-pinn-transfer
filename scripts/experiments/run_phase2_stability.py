#!/usr/bin/env python3
"""
Phase 2 Stability Test — Does Phase 2 degrade a good Phase 1 initialization?

At α=1.0 with high-quality LF data, Phase 2 has few effective PDE constraints
(K_eff=2). This test evaluates L2 error at multiple Phase 2 checkpoints to see
if error rises during Phase 2 (instability) vs monotonically decreasing (stable).

This validates the GL memory stability hypothesis:
- α=0.5 (K_eff=44): Phase 2 should be stable (error decreases)
- α=1.0 (K_eff=2):  Phase 2 may be unstable (error rises after initial decrease)

Run:  python3 run_phase2_stability.py --alpha 1.0 --device cuda:3 --seeds 5
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


def build_config(alpha, n_col, phase2_epochs):
    cfg = {**MANUAL_HP, **PHYSICS, **BC,
           'alpha': alpha,
           'N_collocation': n_col,
           'phase2_epochs': phase2_epochs}
    return cfg


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
    """Compute average L2 across time snapshots."""
    model.eval()
    l2_vals = []
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
            l2_vals.append(l2)

    return np.mean(l2_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    n_col = args.n_col

    # Phase 2 checkpoints to evaluate
    p2_checkpoints = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]

    print(f"\n{'#'*70}")
    print(f"  PHASE 2 STABILITY TEST")
    print(f"  α={alpha}, N_col={n_col}, Seeds={args.seeds}")
    print(f"  N_terms={args.n_terms}, Phase 2 checkpoints: {p2_checkpoints}")
    print(f"{'#'*70}\n")

    # 1. Reference
    print("Generating reference...")
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
        print(f"  Loaded cached reference")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # 2. LF data
    lf_err = compute_lf_error(solver, lf_N_terms=args.n_terms,
                               lf_precision=20, N_test=50)
    print(f"  LF error (N_terms={args.n_terms}): {lf_err['l2_relative']*100:.2f}%")

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=args.n_terms, lf_precision=20,
    )

    # 3. For each seed: train Phase 1 once, then vary Phase 2 duration
    results = {'checkpoints': p2_checkpoints, 'seeds': {}}

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        seed_results = []

        for p2_epochs in p2_checkpoints:
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = build_config(alpha, n_col, p2_epochs)
            model = build_model(config)
            trainer = FDRTrainer(model, config, lf_data, device=device)

            if p2_epochs == 0:
                # Phase 1 only
                trainer.train_phase1(verbose=False)
            else:
                # Phase 1 + Phase 2
                trainer.train_full(verbose=False)

            l2 = evaluate_model(model, ref_data, device)
            seed_results.append(l2)
            print(f"    P2={p2_epochs:>5d} epochs → L2={l2*100:.2f}%")

        results['seeds'][str(seed)] = seed_results

    # 4. Summary
    print(f"\n{'#'*70}")
    print(f"  PHASE 2 STABILITY SUMMARY (α={alpha})")
    print(f"{'#'*70}")

    all_seeds = np.array([results['seeds'][str(s)] for s in range(args.seeds)])  # (S, n_checkpoints)
    means = np.mean(all_seeds, axis=0)
    stds = np.std(all_seeds, axis=0)

    results['summary'] = {
        'means': means.tolist(),
        'stds': stds.tolist(),
    }

    print(f"  {'P2 Epochs':>10} {'L2 Mean%':>10} {'L2 Std%':>10} {'Trend':>10}")
    print(f"  {'-'*40}")
    for i, p2 in enumerate(p2_checkpoints):
        trend = ''
        if i > 0:
            if means[i] < means[i-1] - 0.001:
                trend = '↓ better'
            elif means[i] > means[i-1] + 0.001:
                trend = '↑ WORSE'
            else:
                trend = '→ flat'
        print(f"  {p2:>10d} {means[i]*100:>10.2f} {stds[i]*100:>10.2f} {trend:>10}")

    # Detect instability: error at 20000 > error at some earlier checkpoint
    best_idx = np.argmin(means)
    final_idx = len(means) - 1
    if best_idx < final_idx and means[final_idx] > means[best_idx] * 1.05:
        print(f"\n  ⚠ INSTABILITY DETECTED: Best at P2={p2_checkpoints[best_idx]} "
              f"({means[best_idx]*100:.2f}%), "
              f"final at P2={p2_checkpoints[final_idx]} ({means[final_idx]*100:.2f}%)")
        print(f"  Phase 2 degrades by {(means[final_idx]/means[best_idx] - 1)*100:.1f}% "
              f"after optimal point")
    else:
        print(f"\n  ✓ STABLE: Error monotonically improves or plateaus during Phase 2")

    # Save
    json_path = out_dir / f'phase2_stability_alpha{alpha}_ncol{n_col}_nterms{args.n_terms}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
