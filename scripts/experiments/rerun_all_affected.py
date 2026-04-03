#!/usr/bin/env python3
"""
Re-run ALL experiments affected by the config change (pde_ramp 2000→3000).
Uses current MANUAL_HP config for reproducibility.

Distributes work across multiple GPUs. Each GPU handles a batch of experiments.

Usage:
  # Run a specific GPU batch (0-3):
  python3 scripts/experiments/rerun_all_affected.py --gpu-batch 0 --device cuda:1
  python3 scripts/experiments/rerun_all_affected.py --gpu-batch 1 --device cuda:2
  python3 scripts/experiments/rerun_all_affected.py --gpu-batch 2 --device cuda:4
  python3 scripts/experiments/rerun_all_affected.py --gpu-batch 3 --device cuda:5
"""

import argparse
import json
import time
import sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


# ── Helpers ──────────────────────────────────────────────────────────

def build_config(alpha, n_col=100, **overrides):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': n_col}
    cfg.update(overrides)
    return cfg


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
    errors = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, j]
            errors.append(compute_l2_relative_error(u_pred, u_ref))
    return float(np.mean(errors))


def get_reference(alpha, device):
    ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
    if ref_path.exists():
        return dict(np.load(ref_path, allow_pickle=True))
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )
    return solver.generate_reference_data(
        nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
    )


def train_vanilla_single(alpha, n_col, seed, ref_data, device, gl_N_memory=None):
    """Train a single vanilla seed, return L2 error %."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    if gl_N_memory is not None:
        cfg['gl_N_memory'] = gl_N_memory
    # Need dummy LF for trainer init
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=12, lf_precision=20,
    )
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
    t0 = time.time()
    trainer.train_vanilla(verbose=False)
    wall = time.time() - t0
    err = evaluate_model(model, ref_data, device)
    return err * 100, wall


def train_mf_single(alpha, n_col, n_terms, seed, ref_data, device,
                     gl_N_memory=None, noise_pct=0.0):
    """Train a single MF seed, return L2 error %."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    if gl_N_memory is not None:
        cfg['gl_N_memory'] = gl_N_memory
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=n_terms, lf_precision=20,
    )
    # Add noise if requested
    if noise_pct > 0:
        rng = np.random.default_rng(seed + 1000)
        noise_level = noise_pct / 100.0
        u_scale = np.abs(lf_data['u']).max()
        lf_data['u'] = lf_data['u'] + rng.normal(0, noise_level * u_scale, lf_data['u'].shape)
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, lf_data, device=device)
    t0 = time.time()
    trainer.train_full(verbose=False)
    wall = time.time() - t0
    err = evaluate_model(model, ref_data, device)
    return err * 100, wall


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}", flush=True)


# ── Experiment Functions ─────────────────────────────────────────────

def run_ncol_budget(alpha, n_col_list, n_terms, n_seeds, ref_data, device, out_path):
    """N_col budget study: vanilla + MF at multiple N_col values."""
    print(f"\n{'#'*70}")
    print(f"  N_COL BUDGET: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"  N_col values: {n_col_list}")
    print(f"{'#'*70}\n", flush=True)

    results = {}
    for n_col in n_col_list:
        van_errs, mf_errs = [], []

        # Vanilla
        for seed in range(n_seeds):
            err, wall = train_vanilla_single(alpha, n_col, seed, ref_data, device)
            van_errs.append(err)
            print(f"  [N_col={n_col}] Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        # MF
        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, n_col, n_terms, seed, ref_data, device)
            mf_errs.append(err)
            print(f"  [N_col={n_col}] MF seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        van_mean, van_std = np.mean(van_errs), np.std(van_errs, ddof=1)
        mf_mean, mf_std = np.mean(mf_errs), np.std(mf_errs, ddof=1)
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
        n = min(len(van_errs), len(mf_errs))
        try:
            stat, p = stats.wilcoxon(mf_errs[:n], van_errs[:n])
        except:
            p = 1.0

        results[f'N_col={n_col}'] = {
            'vanilla': {'l2_values': van_errs, 'mean': van_mean, 'std': van_std},
            'mf': {'l2_values': mf_errs, 'mean': mf_mean, 'std': mf_std},
            'ratio': ratio, 'wilcoxon_p': p,
        }
        print(f"\n  N_col={n_col}: Van {van_mean:.2f}±{van_std:.2f}% | "
              f"MF {mf_mean:.2f}±{mf_std:.2f}% | {ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_fidelity_sweep(alpha, nterms_list, n_seeds, ref_data, device, out_path):
    """Fidelity sweep: shared vanilla + MF at multiple N_terms."""
    print(f"\n{'#'*70}")
    print(f"  FIDELITY SWEEP: α={alpha}, {n_seeds} seeds")
    print(f"  N_terms: {nterms_list}")
    print(f"{'#'*70}\n", flush=True)

    # Shared vanilla baseline
    van_errs = []
    dummy_lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=12, lf_precision=20,
    )
    for seed in range(n_seeds):
        err, wall = train_vanilla_single(alpha, 100, seed, ref_data, device)
        van_errs.append(err)
        print(f"  Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)
    print(f"  Vanilla: {np.mean(van_errs):.2f}±{np.std(van_errs, ddof=1):.2f}%\n", flush=True)

    results = {
        'vanilla': {'l2_values': van_errs, 'l2_mean': np.mean(van_errs),
                     'l2_std': np.std(van_errs, ddof=1)},
    }

    for n_terms in nterms_list:
        # LF quality
        solver = FDRLaplaceSolver(
            alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
            bc_type=BC['bc_type'], bc_params=BC['bc_params'],
            N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
        )
        lf_err_dict = compute_lf_error(solver, lf_N_terms=n_terms, lf_precision=20, N_test=50)
        lf_err = lf_err_dict['l2_relative'] * 100
        print(f"  N_terms={n_terms}: LF quality = {lf_err:.1f}%", flush=True)

        mf_errs = []
        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, 100, n_terms, seed, ref_data, device)
            mf_errs.append(err)
            print(f"  MF(N_terms={n_terms}) seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        mf_mean = np.mean(mf_errs)
        mf_std = np.std(mf_errs, ddof=1)
        ratio = np.mean(van_errs) / mf_mean if mf_mean > 0 else float('inf')
        n = min(len(van_errs), len(mf_errs))
        try:
            stat, p = stats.wilcoxon(mf_errs[:n], van_errs[:n])
        except:
            p = 1.0

        results[f'MF_Nterms{n_terms}'] = {
            'l2_values': mf_errs, 'l2_mean': mf_mean, 'l2_std': mf_std,
            'lf_err': lf_err, 'ratio': ratio, 'wilcoxon_p': p,
        }
        print(f"  MF(N_terms={n_terms}): {mf_mean:.2f}±{mf_std:.2f}% | "
              f"{ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_gl_ablation(alpha, gl_mem_list, n_terms, n_seeds, ref_data, device, out_path):
    """GL memory ablation: vanilla + MF at different gl_N_memory values."""
    print(f"\n{'#'*70}")
    print(f"  GL MEMORY ABLATION: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"  gl_N_memory: {gl_mem_list}")
    print(f"{'#'*70}\n", flush=True)

    results = {}
    for gl_mem in gl_mem_list:
        van_errs, mf_errs = [], []

        for seed in range(n_seeds):
            err, wall = train_vanilla_single(alpha, 100, seed, ref_data, device,
                                              gl_N_memory=gl_mem)
            van_errs.append(err)
            print(f"  [gl_mem={gl_mem}] Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, 100, n_terms, seed, ref_data, device,
                                         gl_N_memory=gl_mem)
            mf_errs.append(err)
            print(f"  [gl_mem={gl_mem}] MF seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        van_mean, van_std = np.mean(van_errs), np.std(van_errs, ddof=1)
        mf_mean, mf_std = np.mean(mf_errs), np.std(mf_errs, ddof=1)
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
        n = min(len(van_errs), len(mf_errs))
        try:
            stat, p = stats.wilcoxon(mf_errs[:n], van_errs[:n])
        except:
            p = 1.0

        results[f'gl_mem={gl_mem}'] = {
            'vanilla': {'l2_values': van_errs, 'mean': van_mean, 'std': van_std},
            'mf': {'l2_values': mf_errs, 'mean': mf_mean, 'std': mf_std},
            'ratio': ratio, 'wilcoxon_p': p,
        }
        print(f"  gl_mem={gl_mem}: Van {van_mean:.2f}±{van_std:.2f}% | "
              f"MF {mf_mean:.2f}±{mf_std:.2f}% | {ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_noise_robustness(alpha, noise_pcts, n_terms, n_seeds, ref_data, device, out_path):
    """Noise robustness: vanilla + MF at different noise levels."""
    print(f"\n{'#'*70}")
    print(f"  NOISE ROBUSTNESS: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"  Noise levels: {noise_pcts}%")
    print(f"{'#'*70}\n", flush=True)

    # Shared vanilla (no noise affects vanilla)
    van_errs = []
    for seed in range(n_seeds):
        err, wall = train_vanilla_single(alpha, 100, seed, ref_data, device)
        van_errs.append(err)
        print(f"  Vanilla seed {seed}: {err:.2f}%", flush=True)
    print(f"  Vanilla: {np.mean(van_errs):.2f}±{np.std(van_errs, ddof=1):.2f}%\n", flush=True)

    results = {
        'vanilla': {'l2_values': van_errs, 'mean': np.mean(van_errs),
                     'std': np.std(van_errs, ddof=1)},
    }

    for noise_pct in noise_pcts:
        mf_errs = []
        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, 100, n_terms, seed, ref_data, device,
                                         noise_pct=noise_pct)
            mf_errs.append(err)
            print(f"  MF(noise={noise_pct}%) seed {seed}: {err:.2f}%", flush=True)

        mf_mean = np.mean(mf_errs)
        ratio = np.mean(van_errs) / mf_mean if mf_mean > 0 else float('inf')
        results[f'noise_{noise_pct}pct'] = {
            'l2_values': mf_errs, 'mean': mf_mean,
            'std': np.std(mf_errs, ddof=1), 'ratio': ratio,
        }
        print(f"  Noise {noise_pct}%: MF {mf_mean:.2f}% | {ratio:.2f}×\n", flush=True)

    save_json(results, out_path)
    return results


def run_composite(alpha, n_terms, n_seeds, ref_data, device, out_path):
    """Composite NN comparison: vanilla + MF + composite."""
    from mf_fpinn.models.fdr_pinn import FDRNet
    print(f"\n{'#'*70}")
    print(f"  COMPOSITE NN: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"{'#'*70}\n", flush=True)

    # Import composite components
    try:
        from scripts.experiments.run_composite_comparison import CompositeMFNet
    except ImportError:
        # Define inline if import fails
        import torch.nn as nn
        class CompositeMFNet(nn.Module):
            """Composite = linear combination: u = α_lo·u_lo + α_hi·u_hi + correction."""
            def __init__(self, lo_net, hi_net, L=1.0, T=1.0):
                super().__init__()
                self.lo_net = lo_net
                self.hi_net = hi_net
                self.alpha_lo = nn.Parameter(torch.tensor(0.5))
                self.alpha_hi = nn.Parameter(torch.tensor(0.5))
                self.L = L
                self.T = T
            def forward(self, x, t):
                with torch.no_grad():
                    u_lo = self.lo_net(x, t)
                u_hi = self.hi_net(x, t)
                return self.alpha_lo * u_lo + self.alpha_hi * u_hi

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=n_terms, lf_precision=20,
    )

    results = {'M1_vanilla': [], 'M3_mf': [], 'composite': []}

    for method in ['M1_vanilla', 'M3_mf', 'composite']:
        print(f"\n  --- {method} ---", flush=True)
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = build_config(alpha, n_col=100)

            if method == 'M1_vanilla':
                model = build_model(cfg)
                trainer = FDRTrainer(model, cfg, lf_data, device=device)
                t0 = time.time()
                trainer.train_vanilla(verbose=False)
                wall = time.time() - t0

            elif method == 'M3_mf':
                model = build_model(cfg)
                trainer = FDRTrainer(model, cfg, lf_data, device=device)
                t0 = time.time()
                trainer.train_full(verbose=False)
                wall = time.time() - t0

            elif method == 'composite':
                # Train lo_net on LF data, hi_net as vanilla, then combine
                lo_net = build_model(cfg)
                trainer_lo = FDRTrainer(lo_net, cfg, lf_data, device=device)
                t0 = time.time()
                trainer_lo.train_phase1(verbose=False)
                lo_net.eval()

                torch.manual_seed(seed)
                np.random.seed(seed)
                hi_net = build_model(cfg)
                trainer_hi = FDRTrainer(hi_net, cfg, lf_data, device=device)
                trainer_hi.train_vanilla(verbose=False)

                model = CompositeMFNet(lo_net, hi_net, L=cfg['L'], T=cfg['T']).to(device)
                # Fine-tune composite weights
                opt = torch.optim.Adam(
                    [model.alpha_lo, model.alpha_hi], lr=1e-3
                )
                x_lf_t = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
                t_lf_t = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
                u_lf_t = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)
                for _ in range(500):
                    opt.zero_grad()
                    u_pred = model(x_lf_t, t_lf_t)
                    loss = torch.nn.functional.mse_loss(u_pred, u_lf_t)
                    loss.backward()
                    opt.step()
                wall = time.time() - t0

            model.eval()
            errors = {}
            with torch.no_grad():
                for j, t_val in enumerate(ref_data['t']):
                    x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
                    t_t = torch.full_like(x_t, t_val)
                    u_pred = model(x_t, t_t).cpu().numpy().flatten()
                    u_ref = ref_data['u'][:, j]
                    errors[f't{t_val:.2f}_u_l2'] = compute_l2_relative_error(u_pred, u_ref)
            errors['seed'] = seed
            errors['wall_time'] = wall
            avg = np.mean([v for k, v in errors.items() if k.endswith('_l2')])
            results[method].append(errors)
            print(f"  {method} seed {seed}: {avg*100:.2f}% ({wall:.0f}s)", flush=True)

    save_json({'config': {'alpha': alpha, 'n_col': 100, 'n_terms': n_terms, 'seeds': n_seeds},
               'results': results}, out_path)
    return results


def run_bvtt(alpha, n_terms, n_seeds, ref_data, device, out_path):
    """Bias-variance decomposition: many seeds for van + MF."""
    print(f"\n{'#'*70}")
    print(f"  BVTT DECOMPOSITION: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"{'#'*70}\n", flush=True)

    results = {'vanilla': {'predictions': {}}, 'mf': {'predictions': {}}}

    # Store per-seed predictions at each time snapshot
    for method in ['vanilla', 'mf']:
        print(f"\n  --- {method} ---", flush=True)
        for seed in range(n_seeds):
            if method == 'vanilla':
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg = build_config(alpha, n_col=100)
                dummy_lf = generate_low_fidelity_data(
                    alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                    bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                    N_LF=50, seed=42, lf_N_terms=12, lf_precision=20,
                )
                model = build_model(cfg)
                trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
                t0 = time.time()
                trainer.train_vanilla(verbose=False)
                wall = time.time() - t0
            else:
                err, wall = 0, 0  # placeholder
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg = build_config(alpha, n_col=100)
                lf_data = generate_low_fidelity_data(
                    alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
                    bc_type=BC['bc_type'], bc_params=BC['bc_params'],
                    N_LF=50, seed=42, lf_N_terms=n_terms, lf_precision=20,
                )
                model = build_model(cfg)
                trainer = FDRTrainer(model, cfg, lf_data, device=device)
                t0 = time.time()
                trainer.train_full(verbose=False)
                wall = time.time() - t0

            model.eval()
            seed_preds = {}
            errors = []
            with torch.no_grad():
                for j, t_val in enumerate(ref_data['t']):
                    x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
                    t_t = torch.full_like(x_t, t_val)
                    u_pred = model(x_t, t_t).cpu().numpy().flatten()
                    u_ref = ref_data['u'][:, j]
                    seed_preds[f't{t_val:.2f}'] = u_pred.tolist()
                    errors.append(compute_l2_relative_error(u_pred, u_ref))
            avg_err = np.mean(errors) * 100
            results[method]['predictions'][str(seed)] = {
                'preds': seed_preds, 'l2_pct': avg_err
            }
            if seed % 10 == 0 or seed == n_seeds - 1:
                print(f"  {method} seed {seed}: {avg_err:.2f}% ({wall:.0f}s)", flush=True)

    save_json(results, out_path)
    return results


# ── GPU Batch Definitions ────────────────────────────────────────────

def gpu_batch_0(device):
    """Batch 0: fidelity sweep α=0.5 + α=1.0 (~12.5 GPU-hr)"""
    ref_05 = get_reference(0.5, device)
    ref_10 = get_reference(1.0, device)

    run_fidelity_sweep(0.5, [12, 15, 20], 10, ref_05, device,
                       'results/rerun/fidelity_10seed_alpha0.5.json')
    run_fidelity_sweep(1.0, [8, 10, 12, 15], 10, ref_10, device,
                       'results/rerun/fidelity_10seed_alpha1.0.json')


def gpu_batch_1(device):
    """Batch 1: N_col budget α=0.5 + α=0.7 (~13 GPU-hr)"""
    ref_05 = get_reference(0.5, device)
    ref_07 = get_reference(0.7, device)

    # α=0.5: N_col={100,200,500} with N_terms=8 (for tab:ncol N_terms=8 rows)
    run_ncol_budget(0.5, [100, 200, 500], 8, 10, ref_05, device,
                    'results/rerun/ncol_alpha0.5_nterms8.json')
    # α=0.5: N_col={100,200,500} with N_terms=12
    run_ncol_budget(0.5, [100, 200, 500], 12, 10, ref_05, device,
                    'results/rerun/ncol_alpha0.5_nterms12.json')
    # α=0.7: N_col=100 with N_terms=12
    run_ncol_budget(0.7, [100, 200, 500], 12, 10, ref_07, device,
                    'results/rerun/ncol_alpha0.7_nterms12.json')


def gpu_batch_2(device):
    """Batch 2: N_col α=1.0 + noise + composite (~14.5 GPU-hr)"""
    ref_10 = get_reference(1.0, device)
    ref_05 = get_reference(0.5, device)

    # α=1.0: N_col={100,200,500}
    run_ncol_budget(1.0, [100, 200, 500], 12, 10, ref_10, device,
                    'results/rerun/ncol_alpha1.0_nterms12.json')
    # Noise robustness α=1.0
    run_noise_robustness(1.0, [0, 1, 5, 10, 20], 12, 5, ref_10, device,
                         'results/rerun/noise_alpha1.0.json')
    # Composite α=0.5
    run_composite(0.5, 20, 10, ref_05, device,
                  'results/rerun/composite_alpha0.5_nterms20.json')


def gpu_batch_3(device):
    """Batch 3: GL ablation + BVTT (~23.6 GPU-hr — heaviest)"""
    ref_05 = get_reference(0.5, device)
    ref_10 = get_reference(1.0, device)

    # GL ablation α=0.5
    run_gl_ablation(0.5, [2, 5, 10, 20, 50, 100], 15, 10, ref_05, device,
                    'results/rerun/gl_ablation_alpha0.5.json')
    # GL ablation α=1.0 (control)
    run_gl_ablation(1.0, [2, 5, 10, 20, 50, 100], 15, 10, ref_10, device,
                    'results/rerun/gl_ablation_alpha1.0.json')
    # BVTT α=0.5 (50 seeds)
    run_bvtt(0.5, 20, 50, ref_05, device,
             'results/rerun/bvtt_alpha0.5.json')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-batch', type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  RE-RUN BATCH {args.gpu_batch} on {args.device}")
    print(f"  Config: pde_ramp={MANUAL_HP['pde_ramp_epochs']}, "
          f"wd={MANUAL_HP['weight_decay']}, epochs={MANUAL_HP['vanilla_epochs']}")
    print(f"{'='*70}\n", flush=True)

    Path('results/rerun').mkdir(parents=True, exist_ok=True)

    batch_funcs = [gpu_batch_0, gpu_batch_1, gpu_batch_2, gpu_batch_3]
    t0 = time.time()
    batch_funcs[args.gpu_batch](device)
    total = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  BATCH {args.gpu_batch} COMPLETE in {total/3600:.1f} hours")
    print(f"{'='*70}\n", flush=True)


if __name__ == '__main__':
    main()
