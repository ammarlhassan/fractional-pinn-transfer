#!/usr/bin/env python3
"""
Inverse problem: MF vs Vanilla comparison for parameter identification (α, κ).

Architecture: plain FDRNet (NO FiLM). α and κ are external nn.Parameters that
enter ONLY through the scale-normalized PDE residual — not through the network
output. This avoids FiLM gradient conflicts that push α away from the true value.

Protocol:
  MF:      Phase1 (LF data pretraining) → Phase2 (measurement fitting) → Phase3 (α,κ identification)
  Vanilla: Phase2 (measurement fitting) → Phase3 (α,κ identification)

Phase3 fixes the network and optimizes only (α, κ) via PDE residual.
This isolates the effect of MF warm start on parameter identification.

Run:  python3 run_inverse_mf_vs_vanilla.py --device cuda:4 --seeds 5
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import FDRLaplaceSolver, generate_sparse_training_data
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.models.fdr_loss import (
    FDRDataLoss, DifferentiableFDRPDELoss, FDRBCLoss, FDRICLoss
)
from mf_fpinn.training.fdr_trainer import CollocationSampler

TRUE_ALPHA = 0.7
TRUE_KAPPA = 1.5
INIT_ALPHA = 0.5
INIT_KAPPA = 1.0
D, L, T = 1.0, 1.0, 1.0
NOISE = 0.05
N_SENSORS, N_TIMES = 20, 10
N_LF = 50
PHASE1_EPOCHS = 5000   # MF only: LF pre-training
PHASE2_EPOCHS = 5000   # measurement fitting (network weights only)
PHASE3_EPOCHS = 15000  # α, κ identification (network FIXED)
ALPHA_RANGE = (0.05, 0.99)
KAPPA_RANGE = (0.01, 5.0)


def generate_measurements(true_solver, noise_level, seed):
    rng = np.random.default_rng(seed)
    x_s = np.linspace(0.05, 0.8, N_SENSORS)
    t_s = np.linspace(0.1, 0.8, N_TIMES)
    xx, tt = np.meshgrid(x_s, t_s, indexing='ij')
    x_flat, t_flat = xx.flatten(), tt.flatten()
    u_clean = np.array([true_solver.evaluate(x_flat[i], t_flat[i])
                        for i in range(len(x_flat))])
    u_max = np.max(np.abs(u_clean))
    u_noisy = u_clean + rng.normal(0, noise_level * u_max, len(x_flat))
    return x_flat, t_flat, u_noisy


def build_net():
    return FDRNet(
        n_layers=5, n_neurons=128, activation='tanh', L=L, T=T, hard_bc=True,
        bc_type='pulse', bc_params={'a': 0.5},
        fourier_features=64, fourier_sigma=5.0,
    )


def run_seed(method, seed, true_solver, lf_solver, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  [{method}] seed {seed}", flush=True)
    t0 = time.time()

    # Measurement tensors
    x_flat, t_flat, u_noisy = generate_measurements(true_solver, NOISE, seed)
    x_meas = torch.tensor(x_flat, dtype=torch.float32, device=device).unsqueeze(1)
    t_meas = torch.tensor(t_flat, dtype=torch.float32, device=device).unsqueeze(1)
    u_meas = torch.tensor(u_noisy, dtype=torch.float32, device=device).unsqueeze(1)

    net = build_net().to(device)
    data_loss_fn = FDRDataLoss()

    # ── Phase 1 (MF only): pre-train on LF data ──────────────────────
    if method == 'MF':
        lf_data = generate_sparse_training_data(lf_solver, N_LF=N_LF)
        x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
        t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
        u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

        opt1 = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EPOCHS, eta_min=1e-6)
        net.train()
        for epoch in range(PHASE1_EPOCHS):
            opt1.zero_grad()
            loss = data_loss_fn(net(x_lf, t_lf), u_lf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt1.step()
            sch1.step()

        # L-BFGS polish on LF data
        lbfgs1 = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                     history_size=50, line_search_fn='strong_wolfe')
        for _ in range(30):
            def c1():
                lbfgs1.zero_grad()
                l = data_loss_fn(net(x_lf, t_lf), u_lf)
                l.backward()
                return l
            lbfgs1.step(c1)
        print(f"    Phase1 done ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 2: fit measurements (network weights only) ─────────────
    opt2 = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    net.train()
    for epoch in range(PHASE2_EPOCHS):
        opt2.zero_grad()
        loss = data_loss_fn(net(x_meas, t_meas), u_meas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt2.step()
        sch2.step()

    # L-BFGS polish on measurements
    lbfgs2 = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                 history_size=50, line_search_fn='strong_wolfe')
    for _ in range(50):
        def c2():
            lbfgs2.zero_grad()
            l = data_loss_fn(net(x_meas, t_meas), u_meas)
            l.backward()
            return l
        lbfgs2.step(c2)

    with torch.no_grad():
        meas_fit = data_loss_fn(net(x_meas, t_meas), u_meas).item()
    print(f"    Phase2 done, meas_fit={meas_fit:.3e} ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 3: identify (α, κ) with network FROZEN ──────────────────
    # Network fixed; only α_raw and κ_raw are optimized via PDE residual.
    # No FiLM → no gradient conflicts through network output.
    for p in net.parameters():
        p.requires_grad_(False)

    # Sigmoid-reparameterized physical parameters
    a_min, a_max = ALPHA_RANGE
    alpha_init = np.clip((INIT_ALPHA - a_min) / (a_max - a_min), 0.01, 0.99)
    alpha_raw = nn.Parameter(
        torch.tensor(float(np.log(alpha_init / (1 - alpha_init))), dtype=torch.float32,
                     device=device)
    )
    k_min, k_max = KAPPA_RANGE
    kappa_init = np.clip((INIT_KAPPA - k_min) / (k_max - k_min), 0.01, 0.99)
    kappa_raw = nn.Parameter(
        torch.tensor(float(np.log(kappa_init / (1 - kappa_init))), dtype=torch.float32,
                     device=device)
    )

    def get_alpha():
        return a_min + (a_max - a_min) * torch.sigmoid(alpha_raw)

    def get_kappa():
        return k_min + (k_max - k_min) * torch.sigmoid(kappa_raw)

    pde_loss_fn = DifferentiableFDRPDELoss(D=D, gl_h=0.005, gl_N_memory=200)
    sampler = CollocationSampler(N_col=500, x_range=(0, L), t_range=(0.01, T), device=device)

    phys_opt = torch.optim.Adam([alpha_raw, kappa_raw], lr=1e-3)
    phys_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        phys_opt, T_max=PHASE3_EPOCHS, eta_min=1e-4)

    net.eval()  # frozen, eval mode
    for epoch in range(PHASE3_EPOCHS):
        phys_opt.zero_grad()
        alpha_est = get_alpha()
        kappa_est = get_kappa()
        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(net, x_col, t_col, alpha_est, kappa_est)
        l_pde.backward()
        alpha_raw.grad.clamp_(-2.0, 2.0)
        kappa_raw.grad.clamp_(-2.0, 2.0)
        phys_opt.step()
        phys_sched.step()

        if (epoch + 1) % 3000 == 0:
            print(f"    Phase3 epoch {epoch+1}/{PHASE3_EPOCHS}: "
                  f"α={alpha_est.item():.4f} κ={kappa_est.item():.4f} "
                  f"PDE={l_pde.item():.3e}", flush=True)

    final_alpha = get_alpha().item()
    final_kappa = get_kappa().item()
    alpha_err = abs(final_alpha - TRUE_ALPHA) / TRUE_ALPHA * 100
    kappa_err = abs(final_kappa - TRUE_KAPPA) / TRUE_KAPPA * 100
    wall = time.time() - t0
    print(f"    → α={final_alpha:.4f} (err={alpha_err:.1f}%) "
          f"κ={final_kappa:.4f} (err={kappa_err:.1f}%)  wall={wall:.0f}s", flush=True)

    return {
        'alpha': final_alpha, 'kappa': final_kappa,
        'alpha_error_pct': alpha_err, 'kappa_error_pct': kappa_err,
        'meas_fit': float(meas_fit), 'wall_time': wall, 'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)

    out_dir = Path('results/inverse_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Inverse Problem: MF vs Vanilla (plain FDRNet, no FiLM)")
    print(f"  True: α={TRUE_ALPHA}, κ={TRUE_KAPPA}, noise={NOISE*100:.0f}%")
    print(f"  Phase3: network FROZEN, optimize (α,κ) via PDE residual only")
    print(f"  device={device}, seeds={args.seeds}")
    print(f"{'#'*70}\n")

    true_solver = FDRLaplaceSolver(
        alpha=TRUE_ALPHA, D=D, kappa=TRUE_KAPPA, L=L,
        bc_type='pulse', bc_params={'a': 0.5}, N_terms=30, precision=30,
    )
    lf_solver = FDRLaplaceSolver(
        alpha=INIT_ALPHA, D=D, kappa=INIT_KAPPA, L=L,
        bc_type='pulse', bc_params={'a': 0.5}, N_terms=20, precision=20,
    )

    all_results = {'MF': [], 'Vanilla': []}

    for method in ['MF', 'Vanilla']:
        print(f"\n{'='*60}  {method}  {'='*60}")
        for seed in range(args.seeds):
            r = run_seed(method, seed, true_solver, lf_solver, device)
            all_results[method].append(r)

    # Summary
    print(f"\n{'#'*70}")
    for method in ['MF', 'Vanilla']:
        res = all_results[method]
        ae = [r['alpha_error_pct'] for r in res]
        ke = [r['kappa_error_pct'] for r in res]
        print(f"  {method}: α_err={np.mean(ae):.2f}±{np.std(ae, ddof=1):.2f}%  "
              f"κ_err={np.mean(ke):.2f}±{np.std(ke, ddof=1):.2f}%")

    out_path = out_dir / f'inverse_mfvsvanilla_alpha{TRUE_ALPHA}_noise{NOISE}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
