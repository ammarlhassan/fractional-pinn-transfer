#!/usr/bin/env python3
"""
Inverse problem: identify (α, κ) from noisy measurements of u(x, t).

Two-stage approach:
  Stage 1: Pre-train on Laplace data (warm start near true solution)
  Stage 2: Joint optimization — network weights + physical parameters

Run:  python3 run_fdr_inverse.py [--device cuda] [--noise 0.05]
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import FDRLaplaceSolver, generate_sparse_training_data
from mf_fpinn.models.fdr_pinn import FDRNet, InverseFDRNet
from mf_fpinn.models.fdr_loss import (
    FDRDataLoss, DifferentiableFDRPDELoss, FDRBCLoss, FDRICLoss
)
from mf_fpinn.training.fdr_trainer import CollocationSampler


def generate_measurements(solver, n_sensors=20, n_times=10,
                          x_range=(0.05, 0.8), t_range=(0.1, 0.8),
                          noise_level=0.05, seed=0):
    """
    Generate synthetic measurement data with Gaussian noise.

    Parameters
    ----------
    solver : FDRLaplaceSolver
        Solver with TRUE parameters.
    n_sensors : int
        Number of spatial sensor locations.
    n_times : int
        Number of time snapshots.
    noise_level : float
        Relative noise level (e.g. 0.05 = 5%).
    seed : int

    Returns
    -------
    meas : dict with 'x', 't', 'u_clean', 'u_noisy', 'noise_level'
    """
    rng = np.random.default_rng(seed)

    x_sensors = np.linspace(x_range[0], x_range[1], n_sensors)
    t_snapshots = np.linspace(t_range[0], t_range[1], n_times)

    # All (x, t) pairs
    xx, tt = np.meshgrid(x_sensors, t_snapshots, indexing='ij')
    x_flat = xx.flatten()
    t_flat = tt.flatten()
    N = len(x_flat)

    u_clean = np.zeros(N)
    print(f"Generating {N} measurements ({n_sensors} sensors × {n_times} times)...")
    for i in range(N):
        u_clean[i] = solver.evaluate(x_flat[i], t_flat[i])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}")

    # Add noise
    u_max = np.max(np.abs(u_clean))
    noise = rng.normal(0, noise_level * u_max, N)
    u_noisy = u_clean + noise

    return {
        'x': x_flat, 't': t_flat,
        'u_clean': u_clean, 'u_noisy': u_noisy,
        'noise_level': noise_level,
        'x_sensors': x_sensors, 't_snapshots': t_snapshots,
    }


def run_inverse(true_alpha, true_kappa, D, L, T,
                bc_type, bc_params,
                init_alpha, init_kappa,
                noise_level, device, seed=0,
                n_lf=50, pretrain_epochs=3000,
                inverse_epochs=10000, verbose=True):
    """
    Run one inverse problem experiment.

    Returns
    -------
    result : dict with identified params, errors, convergence history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. True solver + measurements
    true_solver = FDRLaplaceSolver(
        alpha=true_alpha, D=D, kappa=true_kappa, L=L,
        bc_type=bc_type, bc_params=bc_params,
        N_terms=30, precision=30,
    )
    meas = generate_measurements(
        true_solver, n_sensors=20, n_times=10,
        noise_level=noise_level, seed=seed,
    )

    # 2. LF data for warm start (using INITIAL GUESS parameters)
    init_solver = FDRLaplaceSolver(
        alpha=init_alpha, D=D, kappa=init_kappa, L=L,
        bc_type=bc_type, bc_params=bc_params,
        N_terms=20, precision=20,
    )
    lf_data = generate_sparse_training_data(init_solver, N_LF=n_lf)

    # 3. Build network + inverse wrapper
    net = FDRNet(
        n_layers=5, n_neurons=128, activation='tanh',
        L=L, T=T, hard_bc=True,
        bc_type=bc_type, bc_params=bc_params,
        fourier_features=64, fourier_sigma=5.0,
    ).to(device)

    inv_model = InverseFDRNet(
        net, init_alpha=init_alpha, init_kappa=init_kappa,
        alpha_range=(0.05, 0.99), kappa_range=(0.01, 5.0),
    ).to(device)

    # Measurement tensors
    x_meas = torch.tensor(meas['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_meas = torch.tensor(meas['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_meas = torch.tensor(meas['u_noisy'], dtype=torch.float32, device=device).unsqueeze(1)

    # LF tensors (for pre-training)
    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    # Loss functions
    data_loss_fn = FDRDataLoss()
    pde_loss_fn = DifferentiableFDRPDELoss(D=D, gl_h=0.005, gl_N_memory=200)
    bc_loss_fn = FDRBCLoss(L=L)
    ic_loss_fn = FDRICLoss()

    sampler = CollocationSampler(N_col=500, x_range=(0, L),
                                  t_range=(0.01, T), device=device)

    # 4. Stage 1: Pre-train network on LF data (fix physical params)
    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Pre-training ({pretrain_epochs} epochs)")
        print(f"  Init α={init_alpha}, κ={init_kappa}")
        print(f"{'='*60}")

    # Only train network params in stage 1
    net_optimizer = torch.optim.Adam(list(inv_model.network_params()), lr=5e-4)
    net_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        net_optimizer, T_max=pretrain_epochs, eta_min=1e-6
    )

    inv_model.train()
    for epoch in range(pretrain_epochs):
        net_optimizer.zero_grad()
        u_pred = inv_model(x_lf, t_lf)
        loss = data_loss_fn(u_pred, u_lf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), 1.0)
        net_optimizer.step()
        net_scheduler.step()

        if verbose and (epoch + 1) % (pretrain_epochs // 5) == 0:
            print(f"  Epoch {epoch+1:>6d}  |  Loss: {loss.item():.6e}")

    # L-BFGS refinement
    if verbose:
        print("  L-BFGS refinement...")
    lbfgs = torch.optim.LBFGS(list(inv_model.network_params()), lr=1.0,
                                max_iter=20, history_size=50,
                                line_search_fn='strong_wolfe')
    for _ in range(30):
        def closure():
            lbfgs.zero_grad()
            pred = inv_model(x_lf, t_lf)
            l = data_loss_fn(pred, u_lf)
            l.backward()
            return l
        lbfgs.step(closure)

    # 5. Stage 2: Fit measurements (network only, physical params frozen)
    meas_epochs = inverse_epochs // 4
    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Measurement fitting ({meas_epochs} epochs)")
        print(f"  Physical params FROZEN: α={inv_model.get_alpha().item():.4f}, κ={inv_model.get_kappa().item():.4f}")
        print(f"{'='*60}")

    meas_opt = torch.optim.Adam(list(inv_model.network_params()), lr=5e-4)
    meas_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        meas_opt, T_max=meas_epochs, eta_min=1e-6
    )

    inv_model.train()
    for epoch in range(meas_epochs):
        meas_opt.zero_grad()
        u_pred_meas = inv_model(x_meas, t_meas)
        loss = data_loss_fn(u_pred_meas, u_meas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), 1.0)
        meas_opt.step()
        meas_sched.step()

        if verbose and (epoch + 1) % max(1, meas_epochs // 5) == 0:
            print(f"  Epoch {epoch+1:>6d}  |  Meas loss: {loss.item():.6e}")

    # L-BFGS refinement on measurements
    if verbose:
        print("  L-BFGS refinement on measurements...")
    lbfgs2 = torch.optim.LBFGS(list(inv_model.network_params()), lr=1.0,
                                 max_iter=20, history_size=50,
                                 line_search_fn='strong_wolfe')
    for _ in range(30):
        def closure2():
            lbfgs2.zero_grad()
            pred = inv_model(x_meas, t_meas)
            l = data_loss_fn(pred, u_meas)
            l.backward()
            return l
        lbfgs2.step(closure2)

    if verbose:
        with torch.no_grad():
            final_meas = data_loss_fn(inv_model(x_meas, t_meas), u_meas)
        print(f"  Final meas loss: {final_meas.item():.6e}")

    # 6. Stage 3: Joint optimization (network + physical params with PDE)
    joint_epochs = inverse_epochs - meas_epochs
    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 3: Joint Inverse ({joint_epochs} epochs)")
        print(f"  True α={true_alpha}, κ={true_kappa}")
        print(f"  Starting: α={inv_model.get_alpha().item():.4f}, κ={inv_model.get_kappa().item():.4f}")
        print(f"{'='*60}")

    # Separate optimizers: low LR for network (preserve measurement fit), moderate for physical
    net_opt = torch.optim.Adam(list(inv_model.network_params()), lr=5e-5,
                                weight_decay=1e-5)
    phys_opt = torch.optim.Adam(inv_model.physical_params(), lr=2e-3)
    net_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        net_opt, T_max=joint_epochs, eta_min=1e-6
    )
    phys_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        phys_opt, T_max=joint_epochs, eta_min=2e-4
    )

    # Weights: both measurement and PDE significant — PDE constrains (α, κ)
    w_meas = 100.0
    w_pde = 10.0
    w_bc = 1.0
    w_ic = 1.0
    pde_ramp = 3000

    history = {'alpha': [], 'kappa': [], 'meas_loss': [], 'pde_loss': [],
               'total_loss': [], 'epoch': []}

    t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, L, 50, device=device).unsqueeze(1)

    for epoch in range(joint_epochs):
        net_opt.zero_grad()
        phys_opt.zero_grad()

        alpha_est = inv_model.get_alpha()
        kappa_est = inv_model.get_kappa()

        # Measurement loss
        u_pred_meas = inv_model(x_meas, t_meas)
        l_meas = data_loss_fn(u_pred_meas, u_meas)

        # PDE loss (differentiable w.r.t. α and κ)
        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(inv_model, x_col, t_col, alpha_est, kappa_est)

        # BC/IC
        l_bc = bc_loss_fn(inv_model, t_bc)
        l_ic = ic_loss_fn(inv_model, x_ic)

        pde_scale = min(1.0, epoch / pde_ramp) if pde_ramp > 0 else 1.0
        total = w_meas * l_meas + w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), 5.0)
        # Clip physical param gradients more aggressively to prevent jumps
        for p in inv_model.physical_params():
            if p.grad is not None:
                p.grad.clamp_(-1.0, 1.0)
        net_opt.step()
        phys_opt.step()
        net_sched.step()
        phys_sched.step()

        # Log
        history['alpha'].append(alpha_est.item())
        history['kappa'].append(kappa_est.item())
        history['meas_loss'].append(l_meas.item())
        history['pde_loss'].append(l_pde.item())
        history['total_loss'].append(total.item())
        history['epoch'].append(epoch)

        if verbose and (epoch + 1) % max(1, joint_epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>6d}  |  "
                  f"α={alpha_est.item():.4f} (true={true_alpha})  "
                  f"κ={kappa_est.item():.4f} (true={true_kappa})  |  "
                  f"Meas: {l_meas.item():.4e}  PDE: {l_pde.item():.4e}")

    # Final estimates
    final_alpha = inv_model.get_alpha().item()
    final_kappa = inv_model.get_kappa().item()
    alpha_err = abs(final_alpha - true_alpha) / true_alpha * 100
    kappa_err = abs(final_kappa - true_kappa) / true_kappa * 100

    if verbose:
        print(f"\n{'='*60}")
        print(f"INVERSE RESULTS")
        print(f"  α: {final_alpha:.4f}  (true={true_alpha}, error={alpha_err:.2f}%)")
        print(f"  κ: {final_kappa:.4f}  (true={true_kappa}, error={kappa_err:.2f}%)")
        print(f"{'='*60}")

    return {
        'true_alpha': true_alpha, 'true_kappa': true_kappa,
        'est_alpha': final_alpha, 'est_kappa': final_kappa,
        'alpha_error_pct': alpha_err, 'kappa_error_pct': kappa_err,
        'noise_level': noise_level, 'seed': seed,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--true-alpha', type=float, default=0.7)
    parser.add_argument('--true-kappa', type=float, default=1.5)
    parser.add_argument('--pretrain', type=int, default=5000)
    parser.add_argument('--inverse-epochs', type=int, default=20000)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'#'*70}")
    print(f"  Inverse Problem: Identify (α, κ)")
    print(f"  True: α={args.true_alpha}, κ={args.true_kappa}")
    print(f"  Noise: {args.noise*100:.0f}%  |  Seeds: {args.seeds}")
    print(f"{'#'*70}\n")

    out_dir = Path('results/fdr/inverse')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        result = run_inverse(
            true_alpha=args.true_alpha,
            true_kappa=args.true_kappa,
            D=1.0, L=1.0, T=1.0,
            bc_type='pulse', bc_params={'a': 0.5},
            init_alpha=0.5,      # deliberately wrong initial guess
            init_kappa=1.0,      # deliberately wrong initial guess
            noise_level=args.noise,
            device=device,
            seed=seed,
            n_lf=50,
            pretrain_epochs=args.pretrain,
            inverse_epochs=args.inverse_epochs,
        )
        all_results.append(result)

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  INVERSE PROBLEM SUMMARY")
    print(f"  True: α={args.true_alpha}, κ={args.true_kappa}, noise={args.noise*100:.0f}%")
    print(f"{'#'*70}")

    alphas = [r['est_alpha'] for r in all_results]
    kappas = [r['est_kappa'] for r in all_results]
    alpha_errs = [r['alpha_error_pct'] for r in all_results]
    kappa_errs = [r['kappa_error_pct'] for r in all_results]

    print(f"\n  α: {np.mean(alphas):.4f} ± {np.std(alphas):.4f}  "
          f"(error: {np.mean(alpha_errs):.2f}% ± {np.std(alpha_errs):.2f}%)")
    print(f"  κ: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}  "
          f"(error: {np.mean(kappa_errs):.2f}% ± {np.std(kappa_errs):.2f}%)")

    # Save
    # Strip history from saved results (too large for JSON)
    save_results = [{k: v for k, v in r.items() if k != 'history'}
                    for r in all_results]
    json_path = out_dir / f'inverse_noise{args.noise}_alpha{args.true_alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved to {json_path}")

    # Run noise sweep if multiple noise levels needed
    if args.seeds >= 3:
        print(f"\n  To run noise robustness sweep:")
        print(f"    for noise in 0.01 0.05 0.10; do")
        print(f"      python3 run_fdr_inverse.py --noise $noise --device {args.device} --seeds {args.seeds}")
        print(f"    done")


if __name__ == '__main__':
    main()
