#!/usr/bin/env python3
"""
Phase 2 Stability Test + Phase 1 Ablation + Convergence Tracking.

Three experiments that probe the MF training dynamics:

1. STABILITY (--mode stability):
   Train Phase 1 once, then run Phase 2 for varying epochs.
   Tests whether Phase 2 degrades good Phase 1 init (esp. at α=1.0).

2. ABLATION (--mode ablation):
   Vary Phase 1 epochs: {0, 500, 1000, 2000, 5000, 10000}.
   Shows sensitivity to Phase 1 budget.

3. CONVERGENCE (--mode convergence):
   Track L2 error during Phase 2 by saving checkpoints and evaluating.

Run:
  python3 run_stability_ablation.py --mode stability --alpha 1.0 --device cuda:3 --seeds 5 --n-terms 20
  python3 run_stability_ablation.py --mode stability --alpha 0.5 --device cuda:3 --seeds 5 --n-terms 20
  python3 run_stability_ablation.py --mode ablation --alpha 0.5 --device cuda:4 --seeds 5
  python3 run_stability_ablation.py --mode convergence --alpha 0.5 --device cuda:4 --seeds 5
"""

import argparse
import json
import time
import copy
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.training.fdr_trainer import FDRTrainer, CollocationSampler
from mf_fpinn.models.fdr_loss import FDRDataLoss, FDRPDELoss, FDRBCLoss, FDRICLoss
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha, **overrides):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}
    cfg.update(overrides)
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


def compute_snapshot_l2(model, ref_data, device):
    """Compute average L2 error across evaluation time snapshots."""
    model.eval()
    l2s = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, j]
            l2s.append(compute_l2_relative_error(u_pred, u_ref))
    model.train()
    return np.mean(l2s)


def setup_data(alpha, n_terms, device):
    """Generate reference + LF data."""
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
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    lf_err = compute_lf_error(solver, lf_N_terms=n_terms, lf_precision=20, N_test=50)
    print(f"  LF error (N_terms={n_terms}): {lf_err['l2_relative']*100:.2f}%")

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=n_terms, lf_precision=20,
    )

    return ref_data, lf_data, lf_err['l2_relative']


# ═══════════════════════════════════════════════════════════════
#  MODE 1: PHASE 2 STABILITY TEST
# ═══════════════════════════════════════════════════════════════

def run_stability(alpha, n_terms, n_col, seeds, device):
    """Train Phase 1 once per seed, then test Phase 2 at varying durations."""
    p2_values = [0, 500, 1000, 2000, 5000, 10000, 15000, 20000]

    print(f"\n{'#'*70}")
    print(f"  PHASE 2 STABILITY TEST")
    print(f"  α={alpha}, N_terms={n_terms}, N_col={n_col}, Seeds={seeds}")
    print(f"  Phase 2 epochs: {p2_values}")
    print(f"{'#'*70}\n")

    ref_data, lf_data, lf_err = setup_data(alpha, n_terms, device)

    results = {'alpha': alpha, 'n_terms': n_terms, 'n_col': n_col,
               'lf_error_pct': float(lf_err * 100)}

    for p2_ep in p2_values:
        l2s = []
        for s in range(seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            model = build_model(build_config(alpha, N_collocation=n_col,
                                              phase2_epochs=p2_ep))
            config = build_config(alpha, N_collocation=n_col, phase2_epochs=p2_ep)
            trainer = FDRTrainer(model, config, lf_data, device=device)

            # Phase 1 always
            trainer.train_phase1(verbose=False)

            if p2_ep > 0:
                trainer.train_phase2(epochs=p2_ep, verbose=False)

            l2 = compute_snapshot_l2(model, ref_data, device)
            l2s.append(float(l2 * 100))

        mean_l2 = np.mean(l2s)
        std_l2 = np.std(l2s)
        print(f"  P2={p2_ep:>6d}: L2={mean_l2:.2f}% ± {std_l2:.2f}%")

        results[f'p2_{p2_ep}'] = {
            'phase2_epochs': p2_ep,
            'l2_mean': float(mean_l2),
            'l2_std': float(std_l2),
            'l2_values': l2s,
        }

    # Analyze degradation
    l2_0 = results['p2_0']['l2_mean']
    best = min((v for k, v in results.items() if k.startswith('p2_')),
               key=lambda v: v['l2_mean'])
    l2_final = results[f'p2_{p2_values[-1]}']['l2_mean']

    print(f"\n  Phase 1 only: {l2_0:.2f}%")
    print(f"  Best at P2={best['phase2_epochs']}: {best['l2_mean']:.2f}%")
    print(f"  Full P2={p2_values[-1]}: {l2_final:.2f}%")

    if l2_final > best['l2_mean'] * 1.1:
        print(f"  ⚠ DEGRADATION: Full P2 is {l2_final/best['l2_mean']:.2f}× worse than optimal")
        results['degradation'] = True
        results['optimal_p2'] = best['phase2_epochs']
    else:
        print(f"  ✓ No degradation — Phase 2 is stable")
        results['degradation'] = False

    return results


# ═══════════════════════════════════════════════════════════════
#  MODE 2: PHASE 1 DURATION ABLATION
# ═══════════════════════════════════════════════════════════════

def run_ablation(alpha, n_terms, n_col, seeds, device):
    """Vary Phase 1 duration, keeping Phase 2 fixed."""
    p1_values = [0, 500, 1000, 2000, 5000, 10000]

    print(f"\n{'#'*70}")
    print(f"  PHASE 1 DURATION ABLATION")
    print(f"  α={alpha}, N_terms={n_terms}, N_col={n_col}, Seeds={seeds}")
    print(f"  Phase 1 epochs: {p1_values}")
    print(f"{'#'*70}\n")

    ref_data, lf_data, lf_err = setup_data(alpha, n_terms, device)

    results = {'alpha': alpha, 'n_terms': n_terms, 'n_col': n_col,
               'lf_error_pct': float(lf_err * 100)}

    # Also run vanilla baseline for reference
    print("  --- Vanilla baseline ---")
    van_l2s = []
    for s in range(seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        config = build_config(alpha, N_collocation=n_col)
        model = build_model(config)
        trainer = FDRTrainer(model, config, lf_data, device=device)
        trainer.train_vanilla(verbose=False)
        l2 = compute_snapshot_l2(model, ref_data, device)
        van_l2s.append(float(l2 * 100))
    print(f"  Vanilla: L2={np.mean(van_l2s):.2f}% ± {np.std(van_l2s):.2f}%")
    results['vanilla'] = {
        'l2_mean': float(np.mean(van_l2s)),
        'l2_std': float(np.std(van_l2s)),
        'l2_values': van_l2s,
    }

    for p1_ep in p1_values:
        l2s = []
        for s in range(seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            config = build_config(alpha, N_collocation=n_col,
                                  phase1_epochs=max(p1_ep, 1))
            if p1_ep == 0:
                config['phase1_lbfgs'] = False
            model = build_model(config)
            trainer = FDRTrainer(model, config, lf_data, device=device)

            if p1_ep == 0:
                # Skip Phase 1 entirely, but still use MF Phase 2 (with data loss)
                trainer.train_phase2(verbose=False)
            else:
                trainer.train_phase1(epochs=p1_ep, verbose=False)
                trainer.train_phase2(verbose=False)

            l2 = compute_snapshot_l2(model, ref_data, device)
            l2s.append(float(l2 * 100))

        mean_l2 = np.mean(l2s)
        std_l2 = np.std(l2s)
        print(f"  P1={p1_ep:>6d}: L2={mean_l2:.2f}% ± {std_l2:.2f}%")

        results[f'p1_{p1_ep}'] = {
            'phase1_epochs': p1_ep,
            'l2_mean': float(mean_l2),
            'l2_std': float(std_l2),
            'l2_values': l2s,
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  MODE 3: CONVERGENCE TRACKING (manual training loop)
# ═══════════════════════════════════════════════════════════════

def train_with_tracking(model, config, lf_data, ref_data, device,
                        method='MF', eval_every=1000):
    """Train with periodic L2 evaluation. Returns list of (epoch, l2) tuples."""
    L = config.get('L', 1.0)
    T = config.get('T', 1.0)
    alpha = config.get('alpha', 1.0)

    # Setup loss functions
    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = FDRDataLoss(u_scale=u_scale)
    pde_loss_fn = FDRPDELoss(
        alpha=alpha, D=config.get('D', 1.0), kappa=config.get('kappa', 1.0),
        gl_h=config.get('gl_h', 0.01), gl_N_memory=config.get('gl_N_memory', 100),
    ).to(device)
    bc_loss_fn = FDRBCLoss(L=L)
    ic_loss_fn = FDRICLoss()

    sampler = CollocationSampler(
        N_col=config.get('N_collocation', 2000),
        x_range=(0.0, L), t_range=(0.01, T), device=device,
    )
    t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, L, 50, device=device).unsqueeze(1)

    curve = []

    if method == 'MF':
        # Phase 1 (data pre-training)
        p1_epochs = config.get('phase1_epochs', 5000)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-4),
                                     weight_decay=config.get('weight_decay', 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1_epochs, eta_min=1e-6)

        model.train()
        for ep in range(p1_epochs):
            optimizer.zero_grad()
            u_pred = model(x_lf, t_lf)
            loss = data_loss_fn(u_pred, u_lf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # L-BFGS refinement
        if config.get('phase1_lbfgs', True):
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=50, line_search_fn='strong_wolfe')
            def closure():
                lbfgs.zero_grad()
                u_pred = model(x_lf, t_lf)
                loss = data_loss_fn(u_pred, u_lf)
                loss.backward()
                return loss
            try:
                lbfgs.step(closure)
            except Exception:
                pass

        # Evaluate after Phase 1
        l2_p1 = compute_snapshot_l2(model, ref_data, device)
        curve.append({'epoch': 0, 'l2': float(l2_p1 * 100), 'phase': 'post_p1'})

        # Phase 2
        p2_epochs = config.get('phase2_epochs', 20000)
        lr2 = config.get('phase2_lr', 1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr2,
                                     weight_decay=config.get('weight_decay', 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=1e-6)

        w_data = config.get('w_data', 10.0)
        w_pde = config.get('w_pde', 1.0)
        w_bc = config.get('w_bc', 1.0)
        w_ic = config.get('w_ic', 1.0)
        ramp = config.get('pde_ramp_epochs', 3000)

        model.train()
        for ep in range(p2_epochs):
            optimizer.zero_grad()
            u_pred = model(x_lf, t_lf)
            l_data = data_loss_fn(u_pred, u_lf)
            x_col, t_col = sampler.sample(ep)
            l_pde, _ = pde_loss_fn(model, x_col, t_col)
            l_bc = bc_loss_fn(model, t_bc)
            l_ic = ic_loss_fn(model, x_ic)
            pde_scale = min(1.0, ep / ramp) if ramp > 0 else 1.0
            total = w_data * l_data + w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 5.0))
            optimizer.step()
            scheduler.step()

            if (ep + 1) % eval_every == 0:
                l2 = compute_snapshot_l2(model, ref_data, device)
                curve.append({'epoch': ep + 1, 'l2': float(l2 * 100), 'phase': 'p2'})

    else:  # Vanilla
        total_epochs = config.get('vanilla_epochs', 25000)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-4),
                                     weight_decay=config.get('weight_decay', 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        w_pde = config.get('w_pde', 1.0)
        w_bc = config.get('w_bc', 1.0)
        w_ic = config.get('w_ic', 1.0)
        ramp = config.get('pde_ramp_epochs', 3000)

        model.train()
        for ep in range(total_epochs):
            optimizer.zero_grad()
            x_col, t_col = sampler.sample(ep)
            l_pde, _ = pde_loss_fn(model, x_col, t_col)
            l_bc = bc_loss_fn(model, t_bc)
            l_ic = ic_loss_fn(model, x_ic)
            pde_scale = min(1.0, ep / ramp) if ramp > 0 else 1.0
            total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (ep + 1) % eval_every == 0:
                l2 = compute_snapshot_l2(model, ref_data, device)
                curve.append({'epoch': ep + 1, 'l2': float(l2 * 100), 'phase': 'vanilla'})

    return curve


def run_convergence(alpha, n_terms, n_col, seeds, device, eval_every=1000):
    """Track L2 error vs epoch for MF and Vanilla."""
    print(f"\n{'#'*70}")
    print(f"  CONVERGENCE TRACKING (L2 vs Epoch)")
    print(f"  α={alpha}, N_terms={n_terms}, N_col={n_col}, Seeds={seeds}")
    print(f"  Evaluate every {eval_every} epochs")
    print(f"{'#'*70}\n")

    ref_data, lf_data, lf_err = setup_data(alpha, n_terms, device)
    config = build_config(alpha, N_collocation=n_col)

    results = {'alpha': alpha, 'n_terms': n_terms, 'n_col': n_col,
               'lf_error_pct': float(lf_err * 100), 'eval_every': eval_every}

    for method in ['MF', 'Vanilla']:
        print(f"\n--- {method} ---")
        all_curves = []
        for s in range(seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            model = build_model(config).to(device)
            curve = train_with_tracking(model, config, lf_data, ref_data, device,
                                        method=method, eval_every=eval_every)
            all_curves.append(curve)
            final = curve[-1]['l2']
            print(f"  Seed {s}: final L2={final:.2f}%")

        results[method] = all_curves

        # Average curve
        if method == 'MF':
            # Phase 2 epochs only (skip post_p1 marker)
            p2_epochs = [c['epoch'] for c in all_curves[0] if c['phase'] == 'p2']
            p1_l2 = np.mean([c[0]['l2'] for c in all_curves])
            print(f"  After Phase 1: {p1_l2:.2f}%")
        else:
            final_mean = np.mean([c[-1]['l2'] for c in all_curves])
            final_std = np.std([c[-1]['l2'] for c in all_curves])
            print(f"  Final: {final_mean:.2f}% ± {final_std:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['stability', 'ablation', 'convergence', 'all'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    n_terms = args.n_terms or (12 if args.alpha < 0.9 else 10)

    out_dir = Path('results/fdr')
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [args.mode] if args.mode != 'all' else ['stability', 'ablation', 'convergence']

    for mode in modes:
        if mode == 'stability':
            for nt in [15, 20]:
                results = run_stability(args.alpha, nt, args.n_col, args.seeds, device)
                path = out_dir / f'stability_alpha{args.alpha}_nterms{nt}_ncol{args.n_col}.json'
                with open(path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n  Saved to {path}")

        elif mode == 'ablation':
            results = run_ablation(args.alpha, n_terms, args.n_col, args.seeds, device)
            path = out_dir / f'ablation_p1_alpha{args.alpha}_nterms{n_terms}_ncol{args.n_col}.json'
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {path}")

        elif mode == 'convergence':
            results = run_convergence(args.alpha, n_terms, args.n_col, args.seeds, device)
            path = out_dir / f'convergence_tracking_alpha{args.alpha}_nterms{n_terms}_ncol{args.n_col}.json'
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {path}")


if __name__ == '__main__':
    main()
