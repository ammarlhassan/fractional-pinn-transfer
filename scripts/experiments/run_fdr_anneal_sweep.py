#!/usr/bin/env python3
"""
Test adaptive w_data annealing on the fidelity sweep at α=1.0.

Hypothesis: Annealing w_data to zero during Phase 2 eliminates the
non-monotonic fidelity gap response by preventing Phase 2 from
over-relying on accurate LF data.

Run:  python3 run_fdr_anneal_sweep.py --device cuda:1 --seeds 5
"""

import argparse
import json
import sys
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


def build_config(alpha, n_col=100, n_terms=12, anneal_epochs=0):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    cfg['N_collocation'] = n_col
    cfg['w_data_anneal_epochs'] = anneal_epochs
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
    model.eval()
    errors = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, j]
            errors.append(compute_l2_relative_error(u_pred, u_ref))
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    n_seeds = args.seeds
    n_col = args.n_col

    # Annealing schedules to test
    anneal_configs = [0, 5000, 10000, 15000]  # 0 = no annealing (baseline)
    nterms_values = [10, 12, 15, 20]

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'log_anneal_sweep_alpha{alpha}.txt'

    # Redirect output
    log_file = open(log_path, 'w')
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"\n{'#'*70}")
    print(f"  W_DATA ANNEALING SWEEP")
    print(f"  α={alpha}, N_col={n_col}, Seeds={n_seeds}")
    print(f"  Anneal schedules: {anneal_configs}")
    print(f"  N_terms values: {nterms_values}")
    print(f"  Device={device}")
    print(f"{'#'*70}\n")

    # Reference solution
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

    # Vanilla baseline (same for all)
    print("--- Vanilla Baseline ---")
    van_errors = []
    # Dummy LF data for trainer init (not used by vanilla)
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=15, lf_precision=20,
    )
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = build_config(alpha, n_col=n_col)
        model = build_model(cfg)
        trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
        trainer.train_vanilla(verbose=False)
        err = evaluate_model(model, ref_data, device)
        van_errors.append(err * 100)
        print(f"  Vanilla seed {seed}: L2={err*100:.2f}%")
    van_mean = np.mean(van_errors)
    van_std = np.std(van_errors)
    print(f"  Vanilla: {van_mean:.2f}% ± {van_std:.2f}%\n")

    # Sweep: N_terms × anneal_epochs
    results = {}
    for n_terms in nterms_values:
        print(f"\n{'='*60}")
        print(f"  N_terms = {n_terms}")
        print(f"{'='*60}")

        # Compute LF degradation
        lf_err_dict = compute_lf_error(solver, lf_N_terms=n_terms, lf_precision=20, N_test=50)
        lf_err = lf_err_dict['l2_relative']
        print(f"  LF quality: {lf_err*100:.1f}% L2 error vs HF")

        # Generate LF data once for this N_terms
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, seed=42, lf_N_terms=n_terms, lf_precision=20,
        )

        for anneal in anneal_configs:
            label = f"anneal={anneal}" if anneal > 0 else "no_anneal"
            print(f"\n  --- {label} ---")

            mf_errors = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg = build_config(alpha, n_col=n_col, anneal_epochs=anneal)
                model = build_model(cfg)
                trainer = FDRTrainer(model, cfg, lf_data, device=device)
                t0 = time.time()
                trainer.train_full(verbose=False)
                wall = time.time() - t0
                err = evaluate_model(model, ref_data, device)
                mf_errors.append(err * 100)
                print(f"    MF seed {seed}: L2={err*100:.2f}%  ({wall:.0f}s)")

            mf_mean = np.mean(mf_errors)
            mf_std = np.std(mf_errors)
            ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
            print(f"  MF ({label}): {mf_mean:.2f}% ± {mf_std:.2f}%  "
                  f"(advantage: {ratio:.2f}×)")

            results[(n_terms, anneal)] = {
                'mf_mean': mf_mean, 'mf_std': mf_std,
                'van_mean': van_mean, 'van_std': van_std,
                'ratio': ratio, 'lf_err': lf_err * 100,
            }

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  ANNEALING SWEEP SUMMARY (α={alpha}, N_col={n_col})")
    print(f"  Vanilla baseline: {van_mean:.2f}% ± {van_std:.2f}%")
    print(f"{'#'*70}\n")
    print(f"  {'N_terms':>7s}  {'Anneal':>8s}  {'LF err':>7s}  {'MF L2':>16s}  {'Ratio':>7s}")
    print(f"  {'-'*55}")
    for (nt, ann), r in sorted(results.items()):
        label = f"{ann}" if ann > 0 else "none"
        print(f"  {nt:>7d}  {label:>8s}  {r['lf_err']:>5.1f}%  "
              f"{r['mf_mean']:>6.2f}±{r['mf_std']:<6.2f}%  {r['ratio']:>6.2f}×")

    # Save JSON
    json_results = {str(k): v for k, v in results.items()}
    json_path = out_dir / f'anneal_sweep_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
