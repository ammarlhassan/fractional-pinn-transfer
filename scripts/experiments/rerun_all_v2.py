#!/usr/bin/env python3
"""
Re-run ALL experiments affected by the fdr_configs.py change (March 29).
Uses current MANUAL_HP for reproducibility. Results go to results/rerun/.

Batch 0 (GPU 1): Fidelity sweeps α=0.5 + α=1.0           ~14 hrs
Batch 1 (GPU 2): N_col budget α=0.5, α=0.7                ~13 hrs
Batch 2 (GPU 4): N_col budget α=1.0, Noise, BVTT           ~14 hrs
Batch 3 (GPU 5): GL ablation α=0.5, Composite              ~17 hrs

Usage:
  python3 scripts/experiments/rerun_all_v2.py --gpu-batch 0 --device cuda:1
  python3 scripts/experiments/rerun_all_v2.py --gpu-batch 1 --device cuda:2
  python3 scripts/experiments/rerun_all_v2.py --gpu-batch 2 --device cuda:4
  python3 scripts/experiments/rerun_all_v2.py --gpu-batch 3 --device cuda:5
"""

import argparse
import json
import time
import sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error

OUT_DIR = Path('results/rerun')


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
    """Return mean L2 relative error (as fraction, not %)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, float(t_val))
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, j]
            errors.append(compute_l2_relative_error(u_pred, u_ref))
    return float(np.mean(errors))


def evaluate_model_predictions(model, ref_data, device):
    """Return per-snapshot predictions (for BVTT) and L2 errors."""
    model.eval()
    preds = {}
    errors = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, float(t_val))
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, j]
            preds[f't{float(t_val):.2f}'] = u_pred.tolist()
            errors.append(compute_l2_relative_error(u_pred, u_ref))
    return preds, float(np.mean(errors))


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


def make_lf(alpha, n_terms=12, seed=42):
    return generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=seed, lf_N_terms=n_terms, lf_precision=20,
    )


def train_vanilla_single(alpha, n_col, seed, ref_data, device, gl_N_memory=None):
    """Train single vanilla, return (L2 error %, wall time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    if gl_N_memory is not None:
        cfg['gl_N_memory'] = gl_N_memory
    dummy_lf = make_lf(alpha)
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
    t0 = time.time()
    trainer.train_vanilla(verbose=False)
    wall = time.time() - t0
    err = evaluate_model(model, ref_data, device)
    return err * 100, wall


def train_vanilla_with_preds(alpha, n_col, seed, ref_data, device):
    """Train vanilla, return (preds_dict, L2 error %, wall time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    dummy_lf = make_lf(alpha)
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, dummy_lf, device=device)
    t0 = time.time()
    trainer.train_vanilla(verbose=False)
    wall = time.time() - t0
    preds, err = evaluate_model_predictions(model, ref_data, device)
    return preds, err * 100, wall


def train_mf_single(alpha, n_col, n_terms, seed, ref_data, device,
                     gl_N_memory=None, noise_pct=0.0):
    """Train single MF, return (L2 error %, wall time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    if gl_N_memory is not None:
        cfg['gl_N_memory'] = gl_N_memory
    lf_data = make_lf(alpha, n_terms=n_terms)
    if noise_pct > 0:
        rng = np.random.default_rng(seed + 1000)
        u_scale = np.abs(lf_data['u']).max()
        lf_data['u'] = lf_data['u'] + rng.normal(0, noise_pct / 100.0 * u_scale, lf_data['u'].shape)
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, lf_data, device=device)
    t0 = time.time()
    trainer.train_full(verbose=False)
    wall = time.time() - t0
    err = evaluate_model(model, ref_data, device)
    return err * 100, wall


def train_mf_with_preds(alpha, n_col, n_terms, seed, ref_data, device):
    """Train MF, return (preds_dict, L2 error %, wall time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = build_config(alpha, n_col)
    lf_data = make_lf(alpha, n_terms=n_terms)
    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, lf_data, device=device)
    t0 = time.time()
    trainer.train_full(verbose=False)
    wall = time.time() - t0
    preds, err = evaluate_model_predictions(model, ref_data, device)
    return preds, err * 100, wall


def wilcoxon_p(a, b):
    n = min(len(a), len(b))
    try:
        _, p = stats.wilcoxon(a[:n], b[:n])
        return float(p)
    except Exception:
        return 1.0


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [SAVED] {path}", flush=True)


# ── Experiment Functions ─────────────────────────────────────────────

def run_ncol_budget(alpha, n_col_list, n_terms, n_seeds, ref_data, device, out_path):
    """N_col budget: Vanilla + MF + Van+50 control.
    Feeds: tab:ncol, tab:alpha_sweep."""
    print(f"\n{'#'*70}")
    print(f"  N_COL BUDGET: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"  N_col: {n_col_list}")
    print(f"{'#'*70}\n", flush=True)

    results = {}
    for n_col in n_col_list:
        van_errs, mf_errs, van50_errs = [], [], []

        # Vanilla (10 seeds)
        for seed in range(n_seeds):
            err, wall = train_vanilla_single(alpha, n_col, seed, ref_data, device)
            van_errs.append(err)
            print(f"  [N_col={n_col}] Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        # MF (10 seeds)
        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, n_col, n_terms, seed, ref_data, device)
            mf_errs.append(err)
            print(f"  [N_col={n_col}] MF seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        # Van+50 control (5 seeds)
        for seed in range(5):
            err, wall = train_vanilla_single(alpha, n_col + 50, seed, ref_data, device)
            van50_errs.append(err)
            print(f"  [N_col={n_col}] Van+50 seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        van_mean = np.mean(van_errs)
        mf_mean = np.mean(mf_errs)
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
        p = wilcoxon_p(mf_errs, van_errs)

        results[f'N_col={n_col}'] = {
            'vanilla': {'l2_values': van_errs, 'mean': float(van_mean),
                        'std': float(np.std(van_errs, ddof=1))},
            'mf': {'l2_values': mf_errs, 'mean': float(mf_mean),
                    'std': float(np.std(mf_errs, ddof=1))},
            'van50': {'l2_values': van50_errs, 'mean': float(np.mean(van50_errs)),
                      'std': float(np.std(van50_errs, ddof=1))},
            'ratio': float(ratio), 'wilcoxon_p': p,
        }
        print(f"\n  N_col={n_col}: Van {van_mean:.2f}±{np.std(van_errs, ddof=1):.2f}% | "
              f"MF {mf_mean:.2f}±{np.std(mf_errs, ddof=1):.2f}% | "
              f"Van+50 {np.mean(van50_errs):.2f}±{np.std(van50_errs, ddof=1):.2f}% | "
              f"{ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_fidelity_sweep(alpha, nterms_list, n_seeds, ref_data, device, out_path):
    """Fidelity sweep: shared vanilla + MF at multiple N_terms, N_col=100.
    Feeds: tab:fidelity_sweep, tab:fidelity_sweep_a05, tab:fidelity_cross."""
    print(f"\n{'#'*70}")
    print(f"  FIDELITY SWEEP: α={alpha}, {n_seeds} seeds")
    print(f"  N_terms: {nterms_list}")
    print(f"{'#'*70}\n", flush=True)

    # Shared vanilla baseline
    van_errs = []
    for seed in range(n_seeds):
        err, wall = train_vanilla_single(alpha, 100, seed, ref_data, device)
        van_errs.append(err)
        print(f"  Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)
    van_mean = np.mean(van_errs)
    van_std = np.std(van_errs, ddof=1)
    print(f"  Vanilla: {van_mean:.2f}±{van_std:.2f}%\n", flush=True)

    # LF quality info
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    results = {
        'vanilla': {'l2_values': van_errs, 'mean': float(van_mean),
                     'std': float(van_std)},
    }

    for n_terms in nterms_list:
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
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
        p = wilcoxon_p(mf_errs, van_errs)

        results[f'MF_Nterms{n_terms}'] = {
            'l2_values': mf_errs, 'mean': float(mf_mean), 'std': float(mf_std),
            'lf_err': float(lf_err), 'ratio': float(ratio), 'wilcoxon_p': p,
        }
        print(f"  MF(N_terms={n_terms}): {mf_mean:.2f}±{mf_std:.2f}% | "
              f"{ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_gl_ablation(alpha, gl_mem_list, n_terms, n_seeds, ref_data, device, out_path):
    """GL memory ablation: vanilla + MF at different gl_N_memory values.
    Feeds: tab:gl_ablation."""
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
            print(f"  [gl={gl_mem}] Van seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, 100, n_terms, seed, ref_data, device,
                                         gl_N_memory=gl_mem)
            mf_errs.append(err)
            print(f"  [gl={gl_mem}] MF seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

        van_mean, van_std = float(np.mean(van_errs)), float(np.std(van_errs, ddof=1))
        mf_mean, mf_std = float(np.mean(mf_errs)), float(np.std(mf_errs, ddof=1))
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')
        p = wilcoxon_p(mf_errs, van_errs)

        results[f'gl_mem={gl_mem}'] = {
            'vanilla': {'l2_values': van_errs, 'mean': van_mean, 'std': van_std},
            'mf': {'l2_values': mf_errs, 'mean': mf_mean, 'std': mf_std},
            'ratio': float(ratio), 'wilcoxon_p': p,
        }
        print(f"  gl={gl_mem}: Van {van_mean:.2f}±{van_std:.2f}% | "
              f"MF {mf_mean:.2f}±{mf_std:.2f}% | {ratio:.2f}× | p={p:.4f}\n", flush=True)

    save_json(results, out_path)
    return results


def run_noise_robustness(alpha, noise_pcts, n_terms, n_seeds, ref_data, device, out_path):
    """Noise robustness: shared vanilla + noisy MF.
    Feeds: tab:noise."""
    print(f"\n{'#'*70}")
    print(f"  NOISE ROBUSTNESS: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"  Noise: {noise_pcts}%")
    print(f"{'#'*70}\n", flush=True)

    # Shared vanilla (noise doesn't affect vanilla)
    van_errs = []
    for seed in range(n_seeds):
        err, wall = train_vanilla_single(alpha, 100, seed, ref_data, device)
        van_errs.append(err)
        print(f"  Vanilla seed {seed}: {err:.2f}%", flush=True)
    van_mean = float(np.mean(van_errs))
    van_std = float(np.std(van_errs, ddof=1))
    print(f"  Vanilla: {van_mean:.2f}±{van_std:.2f}%\n", flush=True)

    results = {
        'vanilla': {'l2_values': van_errs, 'mean': van_mean, 'std': van_std},
    }
    for noise in noise_pcts:
        mf_errs = []
        for seed in range(n_seeds):
            err, wall = train_mf_single(alpha, 100, n_terms, seed, ref_data, device,
                                         noise_pct=noise)
            mf_errs.append(err)
            print(f"  MF(noise={noise}%) seed {seed}: {err:.2f}%", flush=True)

        mf_mean = float(np.mean(mf_errs))
        mf_std = float(np.std(mf_errs, ddof=1))
        ratio = van_mean / mf_mean if mf_mean > 0 else float('inf')

        results[f'noise_{noise}pct'] = {
            'l2_values': mf_errs, 'mean': mf_mean, 'std': mf_std,
            'ratio': float(ratio),
        }
        print(f"  Noise {noise}%: MF {mf_mean:.2f}±{mf_std:.2f}% | {ratio:.2f}×\n", flush=True)

    save_json(results, out_path)
    return results


def run_composite(alpha, n_terms, n_seeds, ref_data, device, out_path):
    """Composite NN comparison: vanilla + MF + composite.
    Feeds: tab:composite."""
    import torch.nn as nn

    class CompositeMFNet(nn.Module):
        def __init__(self, lo_net, hi_net, L=1.0, T=1.0):
            super().__init__()
            self.lo_net = lo_net
            self.hi_net = hi_net
            self.alpha_lo = nn.Parameter(torch.tensor(0.5))
            self.alpha_hi = nn.Parameter(torch.tensor(0.5))
            self.L, self.T = L, T

        def forward(self, x, t):
            with torch.no_grad():
                u_lo = self.lo_net(x, t)
            u_hi = self.hi_net(x, t)
            return self.alpha_lo * u_lo + self.alpha_hi * u_hi

    print(f"\n{'#'*70}")
    print(f"  COMPOSITE NN: α={alpha}, N_terms={n_terms}, {n_seeds} seeds")
    print(f"{'#'*70}\n", flush=True)

    lf_data = make_lf(alpha, n_terms=n_terms)
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
                opt = torch.optim.Adam([model.alpha_lo, model.alpha_hi], lr=1e-3)
                x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
                t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
                u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)
                for _ in range(500):
                    opt.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(x_lf, t_lf), u_lf)
                    loss.backward()
                    opt.step()
                wall = time.time() - t0

            err = evaluate_model(model, ref_data, device) * 100
            results[method].append({'seed': seed, 'l2_pct': err, 'wall': wall})
            print(f"  {method} seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)

    # Summary stats
    summary = {}
    for method in results:
        errs = [r['l2_pct'] for r in results[method]]
        summary[method] = {'mean': float(np.mean(errs)), 'std': float(np.std(errs, ddof=1)),
                           'l2_values': errs}

    save_json({'config': {'alpha': alpha, 'n_col': 100, 'n_terms': n_terms,
                          'seeds': n_seeds}, 'results': results, 'summary': summary},
              out_path)
    return results


def run_bvtt(alpha, nterms_list, n_seeds, ref_data, device, out_path):
    """BVTT decomposition: stores spatial predictions for bias/variance calculation.
    Feeds: tab:bvtt.

    Runs vanilla + MF at multiple N_terms, 10 seeds, storing full predictions.
    """
    print(f"\n{'#'*70}")
    print(f"  BVTT DECOMPOSITION: α={alpha}, N_terms={nterms_list}, {n_seeds} seeds")
    print(f"{'#'*70}\n", flush=True)

    results = {
        'vanilla': {'predictions': {}, 'l2_values': []},
        'ref_x': ref_data['x'].tolist() if hasattr(ref_data['x'], 'tolist') else list(ref_data['x']),
        'ref_t': [float(t) for t in ref_data['t']],
    }

    # Vanilla predictions
    print(f"  --- Vanilla ---", flush=True)
    for seed in range(n_seeds):
        preds, err, wall = train_vanilla_with_preds(alpha, 100, seed, ref_data, device)
        results['vanilla']['predictions'][str(seed)] = preds
        results['vanilla']['l2_values'].append(err)
        print(f"  Vanilla seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)
    print(f"  Vanilla mean: {np.mean(results['vanilla']['l2_values']):.2f}%\n", flush=True)

    # MF at each N_terms
    for n_terms in nterms_list:
        key = f'MF_Nterms{n_terms}'
        results[key] = {'predictions': {}, 'l2_values': []}
        print(f"  --- MF N_terms={n_terms} ---", flush=True)
        for seed in range(n_seeds):
            preds, err, wall = train_mf_with_preds(alpha, 100, n_terms, seed, ref_data, device)
            results[key]['predictions'][str(seed)] = preds
            results[key]['l2_values'].append(err)
            print(f"  MF(N_terms={n_terms}) seed {seed}: {err:.2f}% ({wall:.0f}s)", flush=True)
        mf_mean = np.mean(results[key]['l2_values'])
        ratio = np.mean(results['vanilla']['l2_values']) / mf_mean if mf_mean > 0 else float('inf')
        print(f"  MF(N_terms={n_terms}) mean: {mf_mean:.2f}% | {ratio:.2f}×\n", flush=True)

    # Store reference for BVTT computation
    ref_u = {}
    for j, t_val in enumerate(ref_data['t']):
        ref_u[f't{float(t_val):.2f}'] = ref_data['u'][:, j].tolist() if hasattr(ref_data['u'][:, j], 'tolist') else list(ref_data['u'][:, j])
    results['ref_u'] = ref_u

    save_json(results, out_path)
    return results


# ── GPU Batch Definitions ────────────────────────────────────────────

def gpu_batch_0(device):
    """Fidelity sweeps α=0.5 and α=1.0 (~14 hrs).
    Produces: fidelity_10seed_alpha0.5.json, fidelity_10seed_alpha1.0.json."""
    ref_05 = get_reference(0.5, device)
    ref_10 = get_reference(1.0, device)

    # α=0.5: N_terms={8,12,15,20} — feeds tab:fidelity_sweep_a05 + tab:fidelity_cross
    run_fidelity_sweep(0.5, [8, 12, 15, 20], 10, ref_05, device,
                       str(OUT_DIR / 'fidelity_10seed_alpha0.5.json'))

    # α=1.0: N_terms={5,8,10,12,15,20} — feeds tab:fidelity_sweep + tab:fidelity_cross
    run_fidelity_sweep(1.0, [5, 8, 10, 12, 15, 20], 10, ref_10, device,
                       str(OUT_DIR / 'fidelity_10seed_alpha1.0.json'))


def gpu_batch_1(device):
    """N_col budget α=0.5 and α=0.7 (~13 hrs).
    Produces: ncol_alpha0.5_nterms12.json, ncol_alpha0.7_nterms12.json."""
    ref_05 = get_reference(0.5, device)
    ref_07 = get_reference(0.7, device)

    run_ncol_budget(0.5, [100, 200, 500], 12, 10, ref_05, device,
                    str(OUT_DIR / 'ncol_alpha0.5_nterms12.json'))
    run_ncol_budget(0.7, [100, 200, 500], 12, 10, ref_07, device,
                    str(OUT_DIR / 'ncol_alpha0.7_nterms12.json'))


def gpu_batch_2(device):
    """N_col α=1.0 + Noise + BVTT (~14 hrs).
    Produces: ncol_alpha1.0_nterms12.json, noise_alpha1.0.json, bvtt_alpha0.5.json."""
    ref_10 = get_reference(1.0, device)
    ref_05 = get_reference(0.5, device)

    # N_col budget α=1.0
    run_ncol_budget(1.0, [100, 200, 500], 12, 10, ref_10, device,
                    str(OUT_DIR / 'ncol_alpha1.0_nterms12.json'))

    # Noise robustness α=1.0 (5 seeds)
    run_noise_robustness(1.0, [0, 1, 5, 10, 20], 12, 5, ref_10, device,
                         str(OUT_DIR / 'noise_alpha1.0.json'))

    # BVTT α=0.5 (10 seeds, multiple N_terms for BVTT table)
    run_bvtt(0.5, [8, 12, 15, 20], 10, ref_05, device,
             str(OUT_DIR / 'bvtt_alpha0.5.json'))


def gpu_batch_3(device):
    """GL ablation α=0.5 + Composite (~17 hrs).
    Produces: gl_ablation_alpha0.5.json, composite_alpha0.5.json."""
    ref_05 = get_reference(0.5, device)

    # GL ablation α=0.5 (6 values × 10 seeds × 2 methods)
    run_gl_ablation(0.5, [2, 5, 10, 20, 50, 100], 15, 10, ref_05, device,
                    str(OUT_DIR / 'gl_ablation_alpha0.5.json'))

    # Composite comparison (10 seeds × 3 methods)
    run_composite(0.5, 20, 10, ref_05, device,
                  str(OUT_DIR / 'composite_alpha0.5.json'))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Re-run affected experiments')
    parser.add_argument('--gpu-batch', type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  RE-RUN BATCH {args.gpu_batch} on {args.device}")
    print(f"  Config: pde_ramp={MANUAL_HP['pde_ramp_epochs']}, "
          f"wd={MANUAL_HP['weight_decay']}, lr={MANUAL_HP['lr']}, "
          f"epochs={MANUAL_HP['vanilla_epochs']}")
    print(f"{'='*70}\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_funcs = [gpu_batch_0, gpu_batch_1, gpu_batch_2, gpu_batch_3]
    t0 = time.time()
    batch_funcs[args.gpu_batch](device)
    total = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  BATCH {args.gpu_batch} COMPLETE in {total/3600:.1f} hours")
    print(f"{'='*70}\n", flush=True)


if __name__ == '__main__':
    main()
