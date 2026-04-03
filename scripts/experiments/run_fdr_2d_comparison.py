#!/usr/bin/env python3
"""
2D MF-fPINN vs Vanilla comparison for fractional diffusion-reaction.

Runs M1 vs M3 (+ Vanilla+50 control) at different N_col values to show
MF advantage extends to 2D.

Run:  python3 run_fdr_2d_comparison.py --alpha 0.5 --device cuda:2 --seeds 5
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from mf_fpinn.solvers.fdr_solver_2d import (
    FDRLaplaceSolver2D,
    generate_low_fidelity_data_2d,
    compute_lf_error_2d,
)
from mf_fpinn.models.fdr_pinn_2d import FDRNet2D
from mf_fpinn.models.fdr_loss_2d import FDRPDELoss2D, FDRDataLoss2D, FDRBCICLoss2D


def generate_reference_2d(solver, nx=20, ny=20, t_values=(0.05, 0.15, 0.25, 0.40),
                           save_path=None):
    """Generate dense 2D reference solution."""
    x_array = np.linspace(0.02, float(solver.Lx) - 0.02, nx)
    y_array = np.linspace(0.02, float(solver.Ly) - 0.02, ny)
    t_array = np.array(t_values)

    print(f"Generating 2D reference: {nx}x{ny}x{len(t_array)} points...")
    u_grid = solver.evaluate_grid(x_array, y_array, t_array)

    data = {'x': x_array, 'y': y_array, 't': t_array, 'u': u_grid}
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved reference to {save_path}")
    return data


def compute_l2_error_2d(model, ref_data, device):
    """Compute relative L2 error on reference grid."""
    x_arr, y_arr, t_arr = ref_data['x'], ref_data['y'], ref_data['t']
    u_ref = ref_data['u']  # (nx, ny, nt)

    errors = []
    model.eval()
    for k, tk in enumerate(t_arr):
        xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
        x_flat = torch.tensor(xx.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        y_flat = torch.tensor(yy.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        t_flat = torch.full_like(x_flat, float(tk))

        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat).cpu().numpy().reshape(xx.shape)

        u_true = u_ref[:, :, k]
        err = np.sqrt(np.mean((u_pred - u_true) ** 2)) / max(np.sqrt(np.mean(u_true ** 2)), 1e-10)
        errors.append(err)

    return float(np.mean(errors))


def train_2d(model, lf_data, pde_loss_fn, data_loss_fn, bcic_loss_fn,
             N_col, device, epochs_phase1=2000, epochs_phase2=3000,
             lr=1e-3, use_transfer=True, Lx=1.0, Ly=1.0, T=0.5,
             verbose=False, bc_aware_phase1=True):
    """Train 2D PINN with soft BC/IC and optional Phase 1 transfer.

    Args:
        bc_aware_phase1: If True, include BC/IC losses in Phase 1 alongside
            data loss. This ensures the warm start respects boundary structure.
    """
    model.to(device)

    # Prepare LF data tensors
    x_lf = y_lf = t_lf = u_lf = None
    if use_transfer and lf_data is not None:
        x_lf = torch.tensor(lf_data['x'], dtype=torch.float32).unsqueeze(1).to(device)
        y_lf = torch.tensor(lf_data['y'], dtype=torch.float32).unsqueeze(1).to(device)
        t_lf = torch.tensor(lf_data['t'], dtype=torch.float32).unsqueeze(1).to(device)
        u_lf = torch.tensor(lf_data['u'], dtype=torch.float32).unsqueeze(1).to(device)

        # Phase 1: data fitting + BC/IC awareness
        opt_p1 = torch.optim.Adam(model.parameters(), lr=lr)
        w_bcic_p1 = 10.0 if bc_aware_phase1 else 0.0
        for epoch in range(epochs_phase1):
            opt_p1.zero_grad()
            u_pred = model(x_lf, y_lf, t_lf)
            loss = data_loss_fn(u_pred, u_lf)
            if bc_aware_phase1:
                l_bcic_p1, _ = bcic_loss_fn(model, device)
                loss = loss + w_bcic_p1 * l_bcic_p1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_p1.step()

        # Phase 1 L-BFGS refinement (data + BC/IC)
        try:
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50,
                                        line_search_fn='strong_wolfe')
            for _ in range(3):
                def closure():
                    lbfgs.zero_grad()
                    u_pred = model(x_lf, y_lf, t_lf)
                    loss = data_loss_fn(u_pred, u_lf)
                    if bc_aware_phase1:
                        l_bcic_p1, _ = bcic_loss_fn(model, device)
                        loss = loss + w_bcic_p1 * l_bcic_p1
                    loss.backward()
                    return loss
                lbfgs.step(closure)
        except Exception:
            pass  # L-BFGS may fail on some configs; Adam result is sufficient

    # Phase 2 (or vanilla): physics + BC/IC
    total_epochs = epochs_phase2 if use_transfer else (epochs_phase1 + epochs_phase2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6)

    pde_ramp_epochs = min(2000, total_epochs // 5)

    for epoch in range(total_epochs):
        optimizer.zero_grad()
        model.train()

        w_pde = min(1.0, epoch / max(pde_ramp_epochs, 1))

        # Beta-distributed collocation points (cluster near boundaries)
        rng_col = torch.Generator(device=device)
        x_c = torch.rand(N_col, 1, device=device) * Lx
        y_c = torch.rand(N_col, 1, device=device) * Ly
        t_c = torch.rand(N_col, 1, device=device) * T * 0.95 + T * 0.05

        l_pde, _ = pde_loss_fn(model, x_c, y_c, t_c)
        l_bcic, _ = bcic_loss_fn(model, device)

        # Data regularization in MF mode (anneal to zero)
        l_data = torch.tensor(0.0, device=device)
        if use_transfer and x_lf is not None:
            anneal = max(0.0, 1.0 - epoch / max(total_epochs * 0.5, 1))
            u_pred = model(x_lf, y_lf, t_lf)
            l_data = anneal * data_loss_fn(u_pred, u_lf)

        w_data = 10.0 if use_transfer else 0.0
        w_bcic = 10.0
        loss = w_pde * l_pde + w_bcic * l_bcic + w_data * l_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"    Epoch {epoch+1}: PDE={l_pde.item():.4e} BCIC={l_bcic.item():.4e}")

    return model


def main():
    parser = argparse.ArgumentParser(description='2D MF-fPINN comparison')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-cols', type=int, nargs='+', default=[200, 500, 1500])
    parser.add_argument('--n-lf', type=int, default=100)
    parser.add_argument('--lf-n-terms', type=int, default=8)
    parser.add_argument('--lf-n-modes', type=int, default=5)
    parser.add_argument('--epochs-p1', type=int, default=2000)
    parser.add_argument('--epochs-p2', type=int, default=3000)
    parser.add_argument('--gl-memory', type=int, default=40)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    res_dir = Path('results/fdr')
    res_dir.mkdir(parents=True, exist_ok=True)

    D, kappa, Lx, Ly, T = 1.0, 1.0, 1.0, 1.0, 0.5
    alpha = args.alpha

    print("#" * 60)
    print(f"  2D MF-fPINN vs Vanilla Comparison")
    print(f"  alpha={alpha}, D={D}, kappa={kappa}, Lx={Lx}, Ly={Ly}")
    print(f"  N_col values={args.n_cols}, seeds={args.seeds}")
    print(f"  LF: N_terms={args.lf_n_terms}, n_modes={args.lf_n_modes}")
    print(f"  Device: {device}")
    print("#" * 60)

    # HF solver for reference
    solver_hf = FDRLaplaceSolver2D(
        alpha=alpha, D=D, kappa=kappa, Lx=Lx, Ly=Ly,
        n_modes=15, bc_type='pulse', bc_params={'a': 0.5},
        N_terms=40, precision=40)

    # Reference data (cache to disk)
    ref_path = res_dir / f'reference_2d_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"Loaded reference from {ref_path}")
    else:
        ref_data = generate_reference_2d(solver_hf, nx=20, ny=20,
                                          t_values=(0.05, 0.15, 0.25, 0.40),
                                          save_path=str(ref_path))

    # Quantify LF fidelity gap
    lf_err = compute_lf_error_2d(solver_hf,
                                  lf_N_terms=args.lf_n_terms,
                                  lf_precision=20,
                                  lf_n_modes=args.lf_n_modes,
                                  N_test=30, seed=123)
    print(f"  LF fidelity gap: {lf_err['l2_relative']*100:.1f}% L2 error")

    # Generate genuine LF data
    lf_data = generate_low_fidelity_data_2d(
        alpha=alpha, D=D, kappa=kappa, Lx=Lx, Ly=Ly,
        bc_type='pulse', bc_params={'a': 0.5},
        N_LF=args.n_lf, seed=999,
        lf_N_terms=args.lf_n_terms, lf_precision=20, lf_n_modes=args.lf_n_modes)

    # Model config
    hp = dict(n_layers=5, n_neurons=200, activation='tanh',
              Lx=Lx, Ly=Ly, T=T, bc_type='pulse', bc_params={'a': 0.5},
              fourier_features=64, fourier_sigma=5.0)

    results = {'lf_error_pct': lf_err['l2_relative'] * 100}

    for N_col in args.n_cols:
        print(f"\n{'='*50}")
        print(f"  N_col = {N_col}")
        print(f"{'='*50}")

        # Three methods: MF, Vanilla, Vanilla+N_LF (data control)
        methods = [
            ('MF', True, False),
            ('Vanilla', False, False),
            (f'Van+{args.n_lf}', False, True),
        ]

        for method_name, use_transfer, extra_col in methods:
            l2_values = []
            t0 = time.time()
            effective_ncol = N_col + args.n_lf if extra_col else N_col

            for seed in range(args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = FDRNet2D(**hp)
                pde_loss = FDRPDELoss2D(alpha=alpha, D=D, kappa=kappa,
                                         gl_h=0.01, gl_N_memory=args.gl_memory)
                data_loss = FDRDataLoss2D()
                bcic_loss = FDRBCICLoss2D(Lx=Lx, Ly=Ly, T=T,
                                           bc_type='pulse', bc_params={'a': 0.5},
                                           N_bc=50)

                model = train_2d(model, lf_data if use_transfer else None,
                                 pde_loss, data_loss, bcic_loss,
                                 N_col=effective_ncol, device=device,
                                 epochs_phase1=args.epochs_p1,
                                 epochs_phase2=args.epochs_p2,
                                 lr=1e-3, use_transfer=use_transfer,
                                 Lx=Lx, Ly=Ly, T=T)

                l2 = compute_l2_error_2d(model, ref_data, device)
                l2_values.append(l2 * 100)
                print(f"    {method_name} seed {seed}: L2={l2*100:.2f}%")

            wall = time.time() - t0
            mean_l2 = np.mean(l2_values)
            std_l2 = np.std(l2_values)
            print(f"  {method_name:>8s}: {mean_l2:.2f}% +/- {std_l2:.2f}%  ({wall:.0f}s)")

            results[f'{method_name}_N{N_col}'] = {
                'method': method_name,
                'n_col': N_col,
                'l2_mean': float(mean_l2),
                'l2_std': float(std_l2),
                'l2_values': [float(v) for v in l2_values],
                'wall_time': float(wall),
            }

    # Summary with Wilcoxon tests
    print(f"\n{'='*60}")
    print(f"  SUMMARY (2D, alpha={alpha})")
    print(f"  LF fidelity gap: {lf_err['l2_relative']*100:.1f}%")
    print(f"{'='*60}")
    print(f"  {'N_col':>6} {'MF L2(%)':>10} {'Van L2(%)':>10} {'Van+N(%)':>10} {'Ratio':>7} {'p-val':>8}")
    print(f"  {'-'*55}")
    for N_col in args.n_cols:
        mf = results.get(f'MF_N{N_col}', {})
        va = results.get(f'Vanilla_N{N_col}', {})
        vc = results.get(f'Van+{args.n_lf}_N{N_col}', {})
        if mf and va:
            ratio = va['l2_mean'] / max(mf['l2_mean'], 1e-8)
            # Wilcoxon signed-rank test (one-sided)
            n = min(len(mf['l2_values']), len(va['l2_values']))
            try:
                stat, p = stats.wilcoxon(va['l2_values'][:n], mf['l2_values'][:n],
                                          alternative='greater')
            except Exception:
                p = 1.0
            sig = '*' if p < 0.05 else 'n.s.'
            vc_str = f"{vc['l2_mean']:.2f}+/-{vc['l2_std']:.2f}" if vc else "N/A"
            print(f"  {N_col:>6} {mf['l2_mean']:>6.2f}+/-{mf['l2_std']:.2f}"
                  f" {va['l2_mean']:>6.2f}+/-{va['l2_std']:.2f}"
                  f" {vc_str:>10}"
                  f" {ratio:>5.1f}x {p:>7.4f}{sig}")

    save_path = res_dir / f'ncol_sweep_2d_alpha{alpha}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {save_path}")


if __name__ == '__main__':
    main()
