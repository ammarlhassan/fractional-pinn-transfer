"""
Burgers equation with NEUTRAL ansatz (no exp(-t) bias).

The original ansatz: u(x,t) = sin(πx)·exp(-t) + x(1-x)t·NN(x,t)
forces exponential decay — correct for classical diffusion (α=1)
but WRONG for fractional diffusion (α<1) where decay follows
Mittag-Leffler (algebraic, not exponential).

Neutral ansatz: u(x,t) = sin(πx) + x(1-x)t·NN(x,t)
Still satisfies BCs: u(0,t)=0, u(1,t)=0, u(x,0)=sin(πx)
But does NOT pre-impose temporal decay profile.

Compare both ansatze for α=0.5 and α=1.0.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

from mf_fpinn.models.burgers_pinn import BurgersNet
from mf_fpinn.models.burgers_loss import BurgersPDELoss, BurgersBCLoss, BurgersICLoss
from mf_fpinn.solvers.burgers_solver import (
    solve_fractional_burgers,
    generate_reference_burgers,
    generate_lf_data_burgers,
)
from scipy.interpolate import RegularGridInterpolator


class BurgersNetNeutral(BurgersNet):
    """BurgersNet with neutral ansatz (no exp(-t))."""

    def forward(self, x, t):
        x_norm = x * self.x_scale - 1.0
        t_norm = t * self.t_scale - 1.0
        inp = torch.cat([x_norm, t_norm], dim=-1)

        if self.use_fourier:
            inp = self.fourier(inp)

        nn_out = self.head(self.trunk(inp))

        if self.hard_bc:
            # NEUTRAL: u(x,t) = sin(πx) + x(1-x)t·NN(x,t)
            # BCs: u(0,t) = sin(0) + 0 = 0 ✓
            #      u(1,t) = sin(π) + 0 = 0 ✓
            #      u(x,0) = sin(πx) + 0 = sin(πx) ✓
            L = self.L
            ic_term = torch.sin(np.pi * x / L)  # NO exp(-t) !!
            correction = x * (L - x) * t * nn_out
            u = ic_term + correction
        else:
            u = nn_out

        return u


def run_comparison(alpha, n_col_list, device, seeds=5, verbose=True):
    """Run Burgers comparison with both ansatze."""
    nu = 0.02
    L = 1.0
    T = 1.0
    N_LF = 50

    # Reference solver
    print(f'Computing reference solution (α={alpha})...')
    x_ref, t_ref, u_ref = solve_fractional_burgers(alpha, nu, Nx=400, Nt=4000, L=L, T=T)
    ref_interp = RegularGridInterpolator(
        (x_ref, t_ref), u_ref, method='linear', bounds_error=False, fill_value=0.0)

    # LF solver
    x_lf_grid, t_lf_grid, u_lf_grid = solve_fractional_burgers(alpha, nu, Nx=8, Nt=80, L=L, T=T)
    lf_interp = RegularGridInterpolator(
        (x_lf_grid, t_lf_grid), u_lf_grid, method='linear', bounds_error=False, fill_value=0.0)
    print(f'LF solver done.')

    # Generate LF data
    rng = np.random.RandomState(42)
    x_lf = rng.beta(0.5, 2.0, N_LF) * L
    t_lf = 0.01 + 0.99 * rng.beta(0.5, 2.0, N_LF)
    u_lf = np.array([lf_interp(np.array([xi, ti])) for xi, ti in zip(x_lf, t_lf)])

    x_lf_t = torch.tensor(x_lf, dtype=torch.float32, device=device).unsqueeze(1)
    t_lf_t = torch.tensor(t_lf, dtype=torch.float32, device=device).unsqueeze(1)
    u_lf_t = torch.tensor(u_lf, dtype=torch.float32, device=device).unsqueeze(1)

    results_all = {}

    for n_col in n_col_list:
        print(f'\n{"="*60}')
        print(f'  N_col = {n_col}, α = {alpha}')
        print(f'{"="*60}')

        for ansatz_name, NetClass in [('original_exp', BurgersNet),
                                       ('neutral', BurgersNetNeutral)]:
            for method in ['mf', 'vanilla']:
                key = f'{ansatz_name}_{method}'
                seed_results = []

                for seed in range(seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = NetClass(
                        n_layers=5, n_neurons=128, activation='tanh',
                        fourier_features=64, fourier_sigma=5.0,
                        L=L, T=T, hard_bc=True
                    ).to(device)

                    pde_loss = BurgersPDELoss(
                        alpha=alpha, nu=nu,
                        gl_h=0.01, gl_N_memory=100
                    ).to(device)
                    bc_loss = BurgersBCLoss(L=L)
                    ic_loss = BurgersICLoss(L=L)

                    # BC/IC evaluation points
                    t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
                    x_ic = torch.linspace(0.01, L, 50, device=device).unsqueeze(1)

                    # Training
                    total_epochs = 25000
                    phase1_epochs = 5000 if method == 'mf' else 0
                    phase2_epochs = 20000 if method == 'mf' else total_epochs

                    t_start = time.time()

                    if method == 'mf':
                        # Phase 1
                        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, phase1_epochs, 1e-6)

                        for ep in range(phase1_epochs):
                            opt.zero_grad()
                            pred = model(x_lf_t, t_lf_t)
                            loss = torch.mean((pred - u_lf_t)**2)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                            opt.step()
                            sched.step()

                    # Phase 2 / Vanilla training
                    lr = 1e-4 if method == 'mf' else 5e-4
                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, phase2_epochs, 1e-6)

                    for ep in range(phase2_epochs):
                        opt.zero_grad()

                        # Collocation
                        if ep % 1000 == 0:
                            x_c = torch.tensor(
                                np.random.beta(0.5, 2.0, n_col) * L,
                                dtype=torch.float32, device=device
                            ).unsqueeze(1)
                            t_c = torch.tensor(
                                0.01 + 0.99 * np.random.beta(0.5, 2.0, n_col),
                                dtype=torch.float32, device=device
                            ).unsqueeze(1)

                        x_c.requires_grad_(True)
                        t_c.requires_grad_(True)

                        pde_scale = min(1.0, (ep + 1) / 3000)
                        l_pde_raw, _ = pde_loss(model, x_c, t_c)
                        l_pde = pde_scale * l_pde_raw

                        l_bc = bc_loss(model, t_bc)
                        l_ic = ic_loss(model, x_ic)

                        l_data = torch.tensor(0.0, device=device)
                        if method == 'mf':
                            pred = model(x_lf_t, t_lf_t)
                            l_data = 10.0 * torch.mean((pred - u_lf_t)**2)

                        loss = l_pde + l_bc + l_ic + l_data
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        sched.step()

                    wall_time = time.time() - t_start

                    # Evaluate
                    t_evals = [0.10, 0.30, 0.50, 0.80]
                    x_eval = np.linspace(0.01, L - 0.01, 100)
                    l2_per_t = []
                    for t_val in t_evals:
                        ref_vals = np.array([ref_interp(np.array([xi, t_val])) for xi in x_eval])
                        xt = torch.tensor(x_eval, dtype=torch.float32, device=device).unsqueeze(1)
                        tt = torch.full_like(xt, t_val)
                        with torch.no_grad():
                            pred_vals = model(xt, tt).cpu().numpy().flatten()
                        l2 = np.linalg.norm(pred_vals - ref_vals) / (np.linalg.norm(ref_vals) + 1e-10)
                        l2_per_t.append(float(l2) * 100)

                    avg_l2 = np.mean(l2_per_t)
                    seed_results.append({
                        'seed': seed,
                        'avg_l2': avg_l2,
                        'l2_per_t': l2_per_t,
                        'wall_time': wall_time,
                    })

                    if verbose:
                        print(f'  {key} seed {seed}: L2={avg_l2:.2f}%')

                l2s = [r['avg_l2'] for r in seed_results]
                results_all[f'ncol{n_col}_{key}'] = {
                    'method': key,
                    'n_col': n_col,
                    'l2_mean': float(np.mean(l2s)),
                    'l2_std': float(np.std(l2s)),
                    'seeds': seed_results,
                }

    return results_all


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, nargs='+', default=[100, 200])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    results = run_comparison(args.alpha, args.n_col, device, args.seeds)

    # Summary
    print(f'\n{"="*70}')
    print(f'ANSATZ COMPARISON α={args.alpha}')
    print(f'{"="*70}')
    for key, val in sorted(results.items()):
        print(f'{key}: {val["l2_mean"]:.2f}±{val["l2_std"]:.2f}%')

    os.makedirs('results/burgers_ansatz', exist_ok=True)
    outpath = f'results/burgers_ansatz/ansatz_alpha{args.alpha}.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
