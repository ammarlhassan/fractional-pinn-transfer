#!/usr/bin/env python3
"""
Inverse problem: Laplace-domain parameter identification for (α, κ).

NOVEL APPROACH — bypasses the GL scheme entirely for parameter identification.

The Grünwald-Letnikov time-domain PDE residual has a systematic bias toward
α→1.0 because at α≈1, the GL sum reduces to a backward difference that any
smooth function approximately satisfies. This creates a spurious minimum in
the PDE residual loss landscape near α=1, regardless of the true α.

Our solution: match the NUMERICAL Laplace transform of the frozen network
to the ANALYTICAL Laplace-domain solution:

    ū_exact(x, s; α, κ) = ḡ(s) · sinh(λ(L-x)) / sinh(λL)
    where λ = √((s^α + κ) / D)

Here α enters through s^α, which is:
  - Smooth and monotonic in α for any fixed s > 0
  - Free of discretization artifacts (no h^{-α} scaling)
  - Well-conditioned for gradient-based optimization
  - A natural spectral characterization of the fractional order

Protocol:
  MF:      Phase1 (LF data) → Phase2 (measurements) → Phase3 (Laplace identification)
  Vanilla: Phase2 (measurements) → Phase3 (Laplace identification)

Run:  CUDA_VISIBLE_DEVICES=4 python3 run_inverse_laplace_id.py --device cuda:0 --seeds 5
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
from mf_fpinn.models.fdr_loss import FDRDataLoss

# ── Problem setup ────────────────────────────────────────────────────
TRUE_ALPHA = 0.7
TRUE_KAPPA = 1.5
INIT_ALPHA = 0.5
INIT_KAPPA = 1.0
D, L, T = 1.0, 1.0, 1.0
NOISE = 0.05
N_SENSORS, N_TIMES = 20, 10
N_LF = 50
BC_A = 0.5  # pulse BC: g(t) = sin(πt/a)

# ── Training epochs ──────────────────────────────────────────────────
PHASE1_EPOCHS = 5000   # MF only: LF pre-training
PHASE2_EPOCHS = 5000   # measurement fitting
PHASE3_EPOCHS = 5000   # Laplace-domain identification (much fewer needed)
ALPHA_RANGE = (0.05, 0.99)
KAPPA_RANGE = (0.01, 5.0)

# ── Laplace identification config ────────────────────────────────────
# Real positive s values — chosen to span the range where the Laplace
# transform is sensitive to α. At large s, s^α dominates κ (constrains α).
# At small s, κ contributes more (constrains κ jointly).
S_VALUES = [2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0]
N_X_LAPLACE = 20      # spatial evaluation points for Laplace comparison
N_T_QUADRATURE = 500   # time points for numerical Laplace integral


def generate_measurements(true_solver, noise_level, seed):
    """Generate noisy sensor measurements."""
    rng = np.random.default_rng(seed)
    x_s = np.linspace(0.05, 0.8, N_SENSORS)
    t_s = np.linspace(0.1, 0.8, N_TIMES)
    xx, tt = np.meshgrid(x_s, t_s, indexing='ij')
    x_flat, t_flat = xx.flatten(), tt.flatten()
    N = len(x_flat)
    print(f"Generating {N} sparse training points...")
    u_clean = np.zeros(N)
    for i in range(N):
        u_clean[i] = true_solver.evaluate(x_flat[i], t_flat[i])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}")
    u_max = np.max(np.abs(u_clean))
    u_noisy = u_clean + rng.normal(0, noise_level * u_max, N)
    return x_flat, t_flat, u_noisy


def build_net():
    """Build the same FDRNet architecture used in other experiments."""
    return FDRNet(
        n_layers=5, n_neurons=128, activation='tanh', L=L, T=T, hard_bc=True,
        bc_type='pulse', bc_params={'a': BC_A},
        fourier_features=64, fourier_sigma=5.0,
    )


def analytical_laplace_solution(x, s, alpha, kappa):
    """
    Differentiable analytical Laplace-domain solution (PyTorch).

    ū(x, s; α, κ) = ḡ(s) · sinh(λ(L-x)) / sinh(λL)
    λ = √((s^α + κ) / D)

    Parameters
    ----------
    x : Tensor (N_x, 1) — spatial points
    s : Tensor (N_s,)   — Laplace variable values
    alpha : Tensor (scalar, requires_grad)
    kappa : Tensor (scalar, requires_grad)

    Returns
    -------
    u_bar : Tensor (N_x, N_s) — ū(x_i, s_j; α, κ)
    """
    omega = torch.tensor(np.pi / BC_A, device=alpha.device, dtype=alpha.dtype)
    a = torch.tensor(BC_A, device=alpha.device, dtype=alpha.dtype)

    # ḡ(s) = ω(1 + exp(-as)) / (s² + ω²)  [Laplace of sin(ωt)·H(t)·H(a-t)]
    g_bar = omega * (1.0 + torch.exp(-a * s)) / (s ** 2 + omega ** 2)  # (N_s,)

    # λ = √((s^α + κ) / D)
    s_alpha = torch.exp(alpha * torch.log(s))  # s^α, differentiable in α
    lambda_sq = (s_alpha + kappa) / D
    lam = torch.sqrt(lambda_sq)  # (N_s,)

    # ū(x, s) = ḡ(s) · sinh(λ(L-x)) / sinh(λL)
    x_flat = x.squeeze(-1)  # (N_x,)
    Lmx = L - x_flat.unsqueeze(1)  # (N_x, 1)
    lam_row = lam.unsqueeze(0)      # (1, N_s)

    # Use numerically stable sinh ratio: sinh(a)/sinh(b) for large arguments
    arg_top = lam_row * Lmx          # (N_x, N_s)
    arg_bot = lam_row * L            # (1, N_s)

    # For large arguments, sinh(a)/sinh(b) ≈ exp(a-b) if both positive
    # Use log-space for stability: log(sinh(a)/sinh(b)) = log(sinh(a)) - log(sinh(b))
    u_bar = g_bar.unsqueeze(0) * torch.sinh(arg_top) / torch.sinh(arg_bot)

    return u_bar  # (N_x, N_s)


def numerical_laplace_transform(net, x_eval, s_values, device):
    """
    Compute numerical Laplace transform of the frozen network output.

    ū_num(x, s) = ∫₀ᵀ û(x, t) e^{-st} dt  (trapezoidal rule)

    Parameters
    ----------
    net : frozen FDRNet
    x_eval : Tensor (N_x, 1)
    s_values : Tensor (N_s,)

    Returns
    -------
    u_bar_num : Tensor (N_x, N_s)
    """
    N_x = x_eval.shape[0]
    N_s = s_values.shape[0]

    # Dense time grid for numerical integration [ε, T]
    # Start from small ε > 0 to avoid t=0 where u=0 exactly (IC)
    t_grid = torch.linspace(0.002, T, N_T_QUADRATURE, device=device)  # (N_t,)
    dt = t_grid[1] - t_grid[0]

    # Evaluate network on full (x, t) grid
    with torch.no_grad():
        # Shape: (N_x, N_t) — evaluate for each x at all t
        u_vals = torch.zeros(N_x, N_T_QUADRATURE, device=device)
        # Batch evaluate: each x_i at all t_k
        batch_size = 50  # process time points in chunks to avoid OOM
        for ti_start in range(0, N_T_QUADRATURE, batch_size):
            ti_end = min(ti_start + batch_size, N_T_QUADRATURE)
            t_chunk = t_grid[ti_start:ti_end]
            n_t_chunk = t_chunk.shape[0]

            # Expand: (N_x * n_t_chunk, 1)
            x_batch = x_eval.repeat(n_t_chunk, 1)
            t_batch = t_chunk.repeat_interleave(N_x).unsqueeze(1)
            # Actually we need (x_i, t_k) for all i,k
            x_batch = x_eval.expand(N_x, n_t_chunk).reshape(-1, 1)
            t_batch = t_chunk.unsqueeze(0).expand(N_x, n_t_chunk).reshape(-1, 1)

            u_out = net(x_batch, t_batch).reshape(N_x, n_t_chunk)
            u_vals[:, ti_start:ti_end] = u_out

    # Numerical Laplace transform via trapezoidal rule
    # ū(x_i, s_j) = Σ_k u(x_i, t_k) · exp(-s_j · t_k) · dt
    # with trapezoidal weights (half weight at endpoints)
    exp_st = torch.exp(-s_values.unsqueeze(1) * t_grid.unsqueeze(0))  # (N_s, N_t)

    # Trapezoidal weights
    trap_weights = torch.ones(N_T_QUADRATURE, device=device) * dt
    trap_weights[0] *= 0.5
    trap_weights[-1] *= 0.5

    # ū_num(x_i, s_j) = Σ_k u(x_i, t_k) · exp(-s_j · t_k) · w_k
    weighted_exp = exp_st * trap_weights.unsqueeze(0)  # (N_s, N_t)
    u_bar_num = torch.matmul(u_vals, weighted_exp.T)  # (N_x, N_s)

    return u_bar_num


def run_seed(method, seed, true_solver, lf_solver, device):
    """Run one inverse experiment with Laplace-domain identification."""
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

        # L-BFGS polish
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

    # ── Phase 3: Laplace-domain identification ─────────────────────────
    # Freeze the network — only α and κ are optimized.
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()

    # Spatial evaluation grid for Laplace comparison
    x_eval = torch.linspace(0.05, 0.95, N_X_LAPLACE, device=device).unsqueeze(1)
    s_vals = torch.tensor(S_VALUES, dtype=torch.float32, device=device)

    # Precompute numerical Laplace transform (frozen, computed once)
    u_bar_num = numerical_laplace_transform(net, x_eval, s_vals, device)
    print(f"    Numerical Laplace computed: shape={u_bar_num.shape}, "
          f"range=[{u_bar_num.min():.4e}, {u_bar_num.max():.4e}] "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Sigmoid-parameterized physical parameters
    a_min, a_max = ALPHA_RANGE
    alpha_init = np.clip((INIT_ALPHA - a_min) / (a_max - a_min), 0.01, 0.99)
    alpha_raw = nn.Parameter(
        torch.tensor(float(np.log(alpha_init / (1 - alpha_init))),
                     dtype=torch.float32, device=device)
    )
    k_min, k_max = KAPPA_RANGE
    kappa_init = np.clip((INIT_KAPPA - k_min) / (k_max - k_min), 0.01, 0.99)
    kappa_raw = nn.Parameter(
        torch.tensor(float(np.log(kappa_init / (1 - kappa_init))),
                     dtype=torch.float32, device=device)
    )

    def get_alpha():
        return a_min + (a_max - a_min) * torch.sigmoid(alpha_raw)

    def get_kappa():
        return k_min + (k_max - k_min) * torch.sigmoid(kappa_raw)

    # Optimizer for Laplace identification
    phys_opt = torch.optim.Adam([alpha_raw, kappa_raw], lr=5e-3)
    phys_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        phys_opt, T_max=PHASE3_EPOCHS, eta_min=1e-4)

    history = {'alpha': [], 'kappa': [], 'loss': []}

    for epoch in range(PHASE3_EPOCHS):
        phys_opt.zero_grad()
        alpha_est = get_alpha()
        kappa_est = get_kappa()

        # Analytical Laplace solution at current (α, κ)
        u_bar_exact = analytical_laplace_solution(x_eval, s_vals, alpha_est, kappa_est)

        # Loss: MSE between numerical and analytical Laplace transforms
        loss = torch.mean((u_bar_num - u_bar_exact) ** 2)
        loss.backward()

        # Gentle gradient clipping (shouldn't need aggressive clipping here)
        alpha_raw.grad.clamp_(-5.0, 5.0)
        kappa_raw.grad.clamp_(-5.0, 5.0)

        phys_opt.step()
        phys_sched.step()

        if (epoch + 1) % 1000 == 0:
            history['alpha'].append(alpha_est.item())
            history['kappa'].append(kappa_est.item())
            history['loss'].append(loss.item())
            print(f"    Phase3 epoch {epoch+1}/{PHASE3_EPOCHS}: "
                  f"α={alpha_est.item():.4f} κ={kappa_est.item():.4f} "
                  f"Laplace_MSE={loss.item():.3e}", flush=True)

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
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)

    out_dir = Path('results/inverse_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Inverse Problem: LAPLACE-DOMAIN Identification")
    print(f"  True: α={TRUE_ALPHA}, κ={TRUE_KAPPA}, noise={NOISE*100:.0f}%")
    print(f"  Phase3: Laplace-domain matching (NO GL scheme)")
    print(f"  s values: {S_VALUES}")
    print(f"  device={device}, seeds={args.seeds}")
    print(f"{'#'*70}\n")

    true_solver = FDRLaplaceSolver(
        alpha=TRUE_ALPHA, D=D, kappa=TRUE_KAPPA, L=L,
        bc_type='pulse', bc_params={'a': BC_A}, N_terms=30, precision=30,
    )
    lf_solver = FDRLaplaceSolver(
        alpha=INIT_ALPHA, D=D, kappa=INIT_KAPPA, L=L,
        bc_type='pulse', bc_params={'a': BC_A}, N_terms=20, precision=20,
    )

    all_results = {'MF': [], 'Vanilla': []}

    for method_name in ['MF', 'Vanilla']:
        print(f"\n{'='*60}  {method_name}  {'='*60}")
        for seed in range(args.seeds):
            r = run_seed(method_name, seed, true_solver, lf_solver, device)
            all_results[method_name].append(r)

    # Summary
    print(f"\n{'#'*70}")
    print(f"  LAPLACE-DOMAIN INVERSE RESULTS")
    print(f"  True: α={TRUE_ALPHA}, κ={TRUE_KAPPA}")
    print(f"{'#'*70}")
    for method_name in ['MF', 'Vanilla']:
        res = all_results[method_name]
        ae = [r['alpha_error_pct'] for r in res]
        ke = [r['kappa_error_pct'] for r in res]
        print(f"  {method_name}: α_err={np.mean(ae):.2f}±{np.std(ae, ddof=1):.2f}%  "
              f"κ_err={np.mean(ke):.2f}±{np.std(ke, ddof=1):.2f}%")

    # Save (strip history for compactness)
    save_results = {}
    for method_name in ['MF', 'Vanilla']:
        save_results[method_name] = [
            {k: v for k, v in r.items() if k != 'history'}
            for r in all_results[method_name]
        ]
    out_path = out_dir / f'inverse_laplace_id_alpha{TRUE_ALPHA}_noise{NOISE}.json'
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
