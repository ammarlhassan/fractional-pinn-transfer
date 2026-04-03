#!/usr/bin/env python3
"""
Fine α grid sweep: MF vs Vanilla vs Van+50 at N_col=100,200.

Fills in missing α values {0.4, 0.6, 0.8} to complement existing {0.3, 0.5, 0.7, 0.9, 1.0}.
Uses genuine LF data (N_terms=12, precision=20).

Run:  python3 run_fdr_alpha_grid.py --device cuda:4 --seeds 5
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


LF_N_TERMS = 12
LF_PRECISION = 20


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


def run_method(method, alpha, n_col, lf_data, ref_data, seed, device):
    """Run a single method/seed and return L2 error."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col=n_col)

    if method == 'vanilla+50':
        cfg['N_collocation'] = n_col + 50

    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, lf_data, device=device)

    t0 = time.time()
    if method == 'mf':
        trainer.train_full(verbose=False)
    else:
        trainer.train_vanilla(verbose=False)
    wall = time.time() - t0

    err = evaluate_model(model, ref_data, device)
    return err, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.4, 0.6, 0.8])
    parser.add_argument('--ncols', nargs='+', type=int, default=[100, 200])
    args = parser.parse_args()

    device = torch.device(args.device)
    n_seeds = args.seeds
    alphas = args.alphas
    ncols = args.ncols

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log_alpha_grid_fine.txt'

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
    print(f"  FINE α GRID SWEEP")
    print(f"  α values: {alphas}")
    print(f"  N_col: {ncols}")
    print(f"  Seeds: {n_seeds}, Device: {device}")
    print(f"  LF: N_terms={LF_N_TERMS}, precision={LF_PRECISION}")
    print(f"{'#'*70}\n")

    results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"  α = {alpha}")
        print(f"{'='*60}")

        # Reference solution
        solver = FDRLaplaceSolver(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
        )
        ref_path = out_dir / f'reference_alpha{alpha}.npz'
        if ref_path.exists():
            ref_data = dict(np.load(ref_path, allow_pickle=True))
            print(f"  Loaded reference from {ref_path}")
        else:
            ref_data = solver.generate_reference_data(
                nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
            )
            print(f"  Generated reference → {ref_path}")

        # LF quality
        lf_err = compute_lf_error(solver, lf_N_terms=LF_N_TERMS,
                                  lf_precision=LF_PRECISION, N_test=50)
        print(f"  LF quality: {lf_err['l2_relative']*100:.1f}% L2 error vs HF")

        # Generate LF data
        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=50, seed=42, lf_N_terms=LF_N_TERMS, lf_precision=LF_PRECISION,
        )

        for n_col in ncols:
            print(f"\n  --- N_col = {n_col} ---")
            for method in ['mf', 'vanilla', 'vanilla+50']:
                errors = []
                walls = []
                for seed in range(n_seeds):
                    err, wall = run_method(method, alpha, n_col, lf_data,
                                          ref_data, seed, device)
                    errors.append(err * 100)
                    walls.append(wall)
                    print(f"    {method} seed {seed}: L2={err*100:.2f}% ({wall:.0f}s)")

                mean_err = np.mean(errors)
                std_err = np.std(errors)
                key = f"alpha{alpha}_{method}_N{n_col}"
                results[key] = {
                    'alpha': alpha, 'n_col': n_col, 'method': method,
                    'l2_mean': mean_err, 'l2_std': std_err,
                    'l2_values': errors,
                    'lf_err': lf_err['l2_relative'] * 100,
                }
                print(f"  {method} N={n_col}: {mean_err:.2f}% ± {std_err:.2f}%")

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  FINE α GRID SUMMARY")
    print(f"{'#'*70}\n")
    print(f"  {'α':>5} {'N_col':>5} {'MF':>12} {'Vanilla':>12} {'Van+50':>12} {'Ratio':>7}")
    print(f"  {'-'*60}")
    for alpha in alphas:
        for n_col in ncols:
            mf = results.get(f"alpha{alpha}_mf_N{n_col}", {})
            van = results.get(f"alpha{alpha}_vanilla_N{n_col}", {})
            vp = results.get(f"alpha{alpha}_vanilla+50_N{n_col}", {})
            ratio = van.get('l2_mean', 0) / mf['l2_mean'] if mf.get('l2_mean', 0) > 0 else 0
            print(f"  {alpha:>5.1f} {n_col:>5d} "
                  f"{mf.get('l2_mean',0):>5.2f}±{mf.get('l2_std',0):<5.2f}"
                  f"{van.get('l2_mean',0):>5.2f}±{van.get('l2_std',0):<5.2f}"
                  f"{vp.get('l2_mean',0):>5.2f}±{vp.get('l2_std',0):<5.2f}"
                  f"  {ratio:>5.2f}×")

    # Save
    json_path = out_dir / 'alpha_grid_fine.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
