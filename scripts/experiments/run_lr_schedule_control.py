#!/usr/bin/env python3
"""
LR Schedule Control: Test whether the two-phase LR schedule itself
(not the LF data) explains MF advantage.

Conditions:
  1. vanilla_standard: 25k epochs @ LR=5e-4 (cosine)
  2. vanilla_schedule: 5k epochs @ LR=5e-4 (cosine) + 20k epochs @ LR=1e-4 (cosine)
     — matches MF Phase 1/2 durations and LRs, but NO data fitting
  3. mf_twophase: 5k epochs data fitting @ LR=5e-4 + 20k physics @ LR=1e-4
     — the actual MF protocol

If vanilla_schedule ≈ mf_twophase >> vanilla_standard:
  → MF advantage is just LR scheduling, not data transfer
If mf_twophase >> vanilla_schedule ≈ vanilla_standard:
  → MF advantage is genuinely from data transfer

Run:
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/experiments/run_lr_schedule_control.py \
      --device cuda:0 --seeds 5
"""

import argparse
import gc
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
    return float(np.mean(l2_vals))


CONFIGS = [
    # (alpha, N_col, N_terms, description)
    (0.5, 100, 20, 'alpha0.5_ncol100_nt20'),   # strongest MF advantage (3.11x)
    (0.7, 100, 12, 'alpha0.7_ncol100_nt12'),   # significant MF advantage
    (0.5, 100, 15, 'alpha0.5_ncol100_nt15'),   # moderate MF advantage (1.74x)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path('results/lr_schedule_control')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for alpha, n_col, n_terms, desc in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"  LR Schedule Control: alpha={alpha}, N_col={n_col}, N_terms={n_terms}")
        print(f"{'#'*70}", flush=True)

        # Generate reference
        ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
        if ref_path.exists():
            ref_data = dict(np.load(ref_path, allow_pickle=True))
        else:
            solver = FDRLaplaceSolver(
                alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
            )
            ref_data = solver.generate_reference_data(
                nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
            )

        # Generate LF data
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, lf_N_terms=n_terms, lf_precision=20,
        )

        config_results = {
            'vanilla_standard': [],
            'vanilla_schedule': [],
            'mf_twophase': [],
        }

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---", flush=True)

            # ── 1. Vanilla standard: 25k @ LR=5e-4 ──
            print(f"  [1/3] Vanilla standard (25k @ LR=5e-4)...", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = {**MANUAL_HP, **PHYSICS, **BC,
                   'alpha': alpha, 'N_collocation': n_col}
            model = build_model(cfg)
            trainer = FDRTrainer(model, cfg, lf_data, device=device)
            t0 = time.time()
            trainer.train_vanilla(verbose=False)
            wall = time.time() - t0
            l2 = evaluate_model(model, ref_data, device)
            print(f"    L2={l2*100:.2f}%, wall={wall:.0f}s", flush=True)
            config_results['vanilla_standard'].append({
                'seed': seed, 'l2': l2, 'wall_time': wall
            })
            del model, trainer; torch.cuda.empty_cache(); gc.collect()

            # ── 2. Vanilla schedule: 5k @ LR=5e-4 + 20k @ LR=1e-4 (PDE only) ──
            print(f"  [2/3] Vanilla schedule (5k@5e-4 + 20k@1e-4, PDE only)...",
                  flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_p1 = {**MANUAL_HP, **PHYSICS, **BC,
                      'alpha': alpha, 'N_collocation': n_col,
                      'vanilla_epochs': 5000, 'lr': 5e-4}
            model = build_model(cfg_p1)
            trainer = FDRTrainer(model, cfg_p1, lf_data, device=device)
            t0 = time.time()
            # Phase 1: 5k epochs PDE @ LR=5e-4
            trainer.train_vanilla(epochs=5000, verbose=False)
            # Phase 2: 20k epochs PDE @ LR=1e-4 (new optimizer, matching MF Phase 2)
            cfg_p2 = {**cfg_p1, 'lr': 1e-4, 'vanilla_epochs': 20000}
            trainer.config = cfg_p2
            trainer.train_vanilla(epochs=20000, verbose=False)
            wall = time.time() - t0
            l2 = evaluate_model(model, ref_data, device)
            print(f"    L2={l2*100:.2f}%, wall={wall:.0f}s", flush=True)
            config_results['vanilla_schedule'].append({
                'seed': seed, 'l2': l2, 'wall_time': wall
            })
            del model, trainer; torch.cuda.empty_cache(); gc.collect()

            # ── 3. MF two-phase: 5k data + 20k physics ──
            print(f"  [3/3] MF two-phase (5k data + 20k physics)...", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_mf = {**MANUAL_HP, **PHYSICS, **BC,
                      'alpha': alpha, 'N_collocation': n_col}
            model = build_model(cfg_mf)
            trainer = FDRTrainer(model, cfg_mf, lf_data, device=device)
            t0 = time.time()
            trainer.train_phase1(verbose=False)
            trainer.train_phase2(verbose=False)
            wall = time.time() - t0
            l2 = evaluate_model(model, ref_data, device)
            print(f"    L2={l2*100:.2f}%, wall={wall:.0f}s", flush=True)
            config_results['mf_twophase'].append({
                'seed': seed, 'l2': l2, 'wall_time': wall
            })
            del model, trainer; torch.cuda.empty_cache(); gc.collect()

        # Summary for this config
        print(f"\n{'='*60}", flush=True)
        print(f"  Summary: {desc}", flush=True)
        print(f"{'='*60}", flush=True)
        for method, results in config_results.items():
            l2s = [r['l2'] * 100 for r in results]
            print(f"  {method:20s}: {np.mean(l2s):.2f}% ± {np.std(l2s, ddof=1):.2f}%",
                  flush=True)

        all_results[desc] = config_results

        # Save incrementally
        save_data = {}
        for cfg_key, cfg_results in all_results.items():
            save_data[cfg_key] = {}
            for method, results in cfg_results.items():
                l2s = [r['l2'] * 100 for r in results]
                save_data[cfg_key][method] = {
                    'l2_errors_pct': l2s,
                    'mean': float(np.mean(l2s)),
                    'std': float(np.std(l2s, ddof=1)),
                    'per_seed': results,
                }

        with open(out_dir / 'lr_schedule_control_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_dir / 'lr_schedule_control_results.json'}",
              flush=True)

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"  FINAL LR SCHEDULE CONTROL SUMMARY")
    print(f"{'#'*70}", flush=True)
    for desc, config_results in all_results.items():
        print(f"\n  {desc}:", flush=True)
        methods = list(config_results.keys())
        for method in methods:
            l2s = [r['l2'] * 100 for r in config_results[method]]
            print(f"    {method:20s}: {np.mean(l2s):.2f}% ± {np.std(l2s, ddof=1):.2f}%",
                  flush=True)
        # Key ratio
        std_l2s = [r['l2'] for r in config_results['vanilla_standard']]
        sched_l2s = [r['l2'] for r in config_results['vanilla_schedule']]
        mf_l2s = [r['l2'] for r in config_results['mf_twophase']]
        print(f"    Schedule vs Standard: {np.mean(std_l2s)/np.mean(sched_l2s):.2f}x",
              flush=True)
        print(f"    MF vs Standard:       {np.mean(std_l2s)/np.mean(mf_l2s):.2f}x",
              flush=True)
        print(f"    MF vs Schedule:       {np.mean(sched_l2s)/np.mean(mf_l2s):.2f}x",
              flush=True)

    print("\nDONE.", flush=True)


if __name__ == '__main__':
    main()
