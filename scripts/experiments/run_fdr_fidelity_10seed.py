#!/usr/bin/env python3
"""
10-seed validation of key fidelity sweep results at α=0.5.
Tests N_terms={15,20} which showed the largest MF advantages (1.58×, 3.15×).

Run:  python3 run_fdr_fidelity_10seed.py --device cuda:1 --seeds 10
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


def build_config(alpha, n_col=100):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    cfg['N_collocation'] = n_col
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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--n-col', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha
    n_seeds = args.seeds
    n_col = args.n_col

    if alpha >= 0.9:
        nterms_values = [8, 10, 12, 15]  # α≈1: test non-monotonic peak at N_terms=10
    else:
        nterms_values = [12, 15, 20]  # α<1: crossover, moderate, high advantage

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'log_fidelity_10seed_alpha{alpha}.txt'

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
    print(f"  FIDELITY SWEEP 10-SEED VALIDATION")
    print(f"  α={alpha}, N_col={n_col}, Seeds={n_seeds}")
    print(f"  N_terms: {nterms_values}")
    print(f"{'#'*70}\n")

    # Reference
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

    # Vanilla baseline (10 seeds)
    print("--- Vanilla Baseline ---")
    van_errors = []
    # Dummy LF data for trainer init
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=12, lf_precision=20,
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
    print(f"  Vanilla: {np.mean(van_errors):.2f}% ± {np.std(van_errors):.2f}%\n")

    # MF for each N_terms
    results = {'vanilla': {'l2_values': van_errors, 'l2_mean': np.mean(van_errors),
                           'l2_std': np.std(van_errors)}}

    for n_terms in nterms_values:
        print(f"\n--- N_terms = {n_terms} ---")
        lf_err_dict = compute_lf_error(solver, lf_N_terms=n_terms, lf_precision=20, N_test=50)
        lf_err = lf_err_dict['l2_relative']
        print(f"  LF quality: {lf_err*100:.1f}%")

        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, seed=42, lf_N_terms=n_terms, lf_precision=20,
        )

        mf_errors = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = build_config(alpha, n_col=n_col)
            model = build_model(cfg)
            trainer = FDRTrainer(model, cfg, lf_data, device=device)
            t0 = time.time()
            trainer.train_full(verbose=False)
            wall = time.time() - t0
            err = evaluate_model(model, ref_data, device)
            mf_errors.append(err * 100)
            print(f"  MF seed {seed}: L2={err*100:.2f}% ({wall:.0f}s)")

        mf_mean = np.mean(mf_errors)
        mf_std = np.std(mf_errors)
        van_mean = np.mean(van_errors)
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')

        # Wilcoxon test
        from scipy import stats
        n = min(len(mf_errors), len(van_errors))
        stat, p = stats.wilcoxon(mf_errors[:n], van_errors[:n])
        sig = '*' if p < 0.05 else 'n.s.'

        print(f"\n  MF (N_terms={n_terms}): {mf_mean:.2f}% ± {mf_std:.2f}%")
        print(f"  Ratio: {ratio:.2f}×, Wilcoxon p={p:.4f} ({sig})")

        results[f'MF_Nterms{n_terms}'] = {
            'l2_values': mf_errors, 'l2_mean': mf_mean, 'l2_std': mf_std,
            'lf_err': lf_err * 100, 'ratio': ratio, 'wilcoxon_p': p,
        }

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY (α={alpha}, N_col={n_col}, {n_seeds} seeds)")
    print(f"  Vanilla: {np.mean(van_errors):.2f}% ± {np.std(van_errors):.2f}%")
    print(f"{'#'*70}")
    for nt in nterms_values:
        r = results[f'MF_Nterms{nt}']
        print(f"  N_terms={nt}: {r['l2_mean']:.2f}%±{r['l2_std']:.2f}% "
              f"ratio={r['ratio']:.2f}× p={r['wilcoxon_p']:.4f}")

    json_path = out_dir / f'fidelity_10seed_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
