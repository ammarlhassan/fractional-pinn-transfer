#!/usr/bin/env python3
"""
Crossover formula validation: checks whether C ≈ 5.8 in Proposition 3
    ε*(α, N_col) = C / sqrt(N_col · K_eff(α))
transfers to a different reaction rate κ.

Runs a fidelity sweep (N_terms ∈ {5,8,10,12,15,20}) for a given
(alpha, kappa) pair, 5 seeds each, and reports the empirical crossover.

Usage — one job per GPU:
  CUDA_VISIBLE_DEVICES=0 python3 run_crossover_validation.py --alpha 0.5 --kappa 0.5  --device cuda:0
  CUDA_VISIBLE_DEVICES=3 python3 run_crossover_validation.py --alpha 1.0 --kappa 0.5  --device cuda:0
  CUDA_VISIBLE_DEVICES=4 python3 run_crossover_validation.py --alpha 0.5 --kappa 2.0  --device cuda:0
  CUDA_VISIBLE_DEVICES=5 python3 run_crossover_validation.py --alpha 1.0 --kappa 2.0  --device cuda:0
"""

import argparse
import json
import sys
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver,
    generate_low_fidelity_data,
    compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


NTERMS_VALUES = [5, 8, 10, 12, 15, 20]
N_SEEDS = 5
N_COL = 100
N_LF = 50


def build_config(alpha, kappa, D=1.0, L=1.0, T=1.0):
    cfg = {**MANUAL_HP, **BC}
    cfg.update({
        'alpha': alpha,
        'D': D,
        'kappa': kappa,
        'L': L,
        'T': T,
        'N_collocation': N_COL,
    })
    return cfg


def build_model(cfg):
    return FDRNet(
        n_layers=cfg['n_layers'],
        n_neurons=cfg['n_neurons'],
        activation=cfg['activation'],
        L=cfg['L'],
        T=cfg['T'],
        hard_bc=True,
        bc_type=cfg['bc_type'],
        bc_params=cfg['bc_params'],
        fourier_features=cfg['fourier_features'],
        fourier_sigma=cfg['fourier_sigma'],
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


def wilcoxon_test(a, b):
    n = min(len(a), len(b))
    try:
        _, p = stats.wilcoxon(a[:n], b[:n])
    except Exception:
        p = float('nan')
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',  type=float, required=True)
    parser.add_argument('--kappa',  type=float, required=True)
    parser.add_argument('--D',      type=float, default=1.0)
    parser.add_argument('--L',      type=float, default=1.0)
    parser.add_argument('--device', type=str,   default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    alpha, kappa = args.alpha, args.kappa

    out_dir = Path('results/crossover_validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'alpha{alpha}_kappa{kappa}'
    log_path = out_dir / f'log_{tag}.txt'

    log_file = open(log_path, 'w')
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"\n{'#'*70}")
    print(f"  CROSSOVER FORMULA VALIDATION")
    print(f"  α={alpha}  κ={kappa}  D={args.D}  L={args.L}")
    print(f"  N_col={N_COL}  N_LF={N_LF}  seeds={N_SEEDS}")
    print(f"  Device: {device}")
    print(f"  N_terms: {NTERMS_VALUES}")
    print(f"{'#'*70}\n")

    # Reference HF solver
    solver = FDRLaplaceSolver(
        alpha=alpha, D=args.D, kappa=kappa, L=args.L,
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )
    ref_path = out_dir / f'reference_{tag}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )
        print(f"Generated reference data → {ref_path}")

    # Shared LF data for vanilla trainer init
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=args.D, kappa=kappa, L=args.L,
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=N_LF, seed=42, lf_N_terms=12, lf_precision=20,
    )

    # --- Vanilla baseline ---
    print("--- Vanilla Baseline ---")
    van_errors = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = build_config(alpha, kappa, D=args.D, L=args.L)
        model = build_model(cfg).to(device)
        trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
        t0 = time.time()
        trainer.train_vanilla(verbose=False)
        wall = time.time() - t0
        err = evaluate_model(model, ref_data, device) * 100
        van_errors.append(err)
        print(f"  Vanilla seed {seed}: L2={err:.2f}%  ({wall:.0f}s)")
    van_mean = float(np.mean(van_errors))
    van_std  = float(np.std(van_errors))
    print(f"  Vanilla: {van_mean:.2f}% ± {van_std:.2f}%\n")

    results = {
        'meta': {'alpha': alpha, 'kappa': kappa, 'D': args.D, 'L': args.L,
                 'N_col': N_COL, 'N_LF': N_LF, 'n_seeds': N_SEEDS,
                 'nterms_values': NTERMS_VALUES},
        'vanilla': {'l2_values': van_errors, 'l2_mean': van_mean, 'l2_std': van_std},
    }

    # --- MF sweep ---
    for n_terms in NTERMS_VALUES:
        print(f"\n--- N_terms = {n_terms} ---")
        lf_err_dict = compute_lf_error(solver, lf_N_terms=n_terms, lf_precision=20, N_test=50)
        lf_err_pct = lf_err_dict['l2_relative'] * 100
        print(f"  LF quality: {lf_err_pct:.1f}%")

        lf_data = generate_low_fidelity_data(
            alpha=alpha, D=args.D, kappa=kappa, L=args.L,
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_LF=N_LF, seed=42, lf_N_terms=n_terms, lf_precision=20,
        )

        mf_errors = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = build_config(alpha, kappa, D=args.D, L=args.L)
            model = build_model(cfg).to(device)
            trainer = FDRTrainer(model, cfg, lf_data, device=device)
            t0 = time.time()
            trainer.train_full(verbose=False)
            wall = time.time() - t0
            err = evaluate_model(model, ref_data, device) * 100
            mf_errors.append(err)
            print(f"  MF seed {seed}: L2={err:.2f}%  ({wall:.0f}s)")

        mf_mean = float(np.mean(mf_errors))
        mf_std  = float(np.std(mf_errors))
        ratio   = van_mean / mf_mean if mf_mean > 0 else float('inf')
        p_val   = wilcoxon_test(mf_errors, van_errors)

        print(f"\n  MF (N_terms={n_terms}): {mf_mean:.2f}% ± {mf_std:.2f}%")
        print(f"  Ratio={ratio:.2f}×  p={p_val:.4f}")

        results[f'MF_Nterms{n_terms}'] = {
            'lf_err_pct': lf_err_pct,
            'l2_values': mf_errors,
            'l2_mean': mf_mean,
            'l2_std': mf_std,
            'ratio': ratio,
            'wilcoxon_p': p_val,
        }

    # --- Summary ---
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY  α={alpha}  κ={kappa}")
    print(f"  Vanilla: {van_mean:.2f}% ± {van_std:.2f}%")
    print(f"{'='*70}")
    print(f"  {'N_terms':>8}  {'LF_err%':>8}  {'MF%':>10}  {'Ratio':>7}  {'p':>7}")
    print(f"  {'-'*55}")
    for nt in NTERMS_VALUES:
        r = results[f'MF_Nterms{nt}']
        sig = '**' if r['wilcoxon_p'] < 0.05 else 'n.s.'
        print(f"  {nt:>8}  {r['lf_err_pct']:>8.1f}  "
              f"{r['l2_mean']:>6.2f}±{r['l2_std']:>5.2f}  "
              f"{r['ratio']:>6.2f}×  {r['wilcoxon_p']:>6.4f} {sig}")
    print(f"{'#'*70}\n")

    json_path = out_dir / f'crossover_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {json_path}")


if __name__ == '__main__':
    main()
