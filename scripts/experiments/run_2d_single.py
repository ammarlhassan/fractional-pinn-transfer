#!/usr/bin/env python3
"""Run a single 2D experiment (one alpha, one N_col) for parallel execution."""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver_2d import FDRLaplaceSolver2D, generate_sparse_training_data_2d
from mf_fpinn.models.fdr_pinn_2d import FDRNet2D
from mf_fpinn.models.fdr_loss_2d import FDRPDELoss2D, FDRDataLoss2D, FDRBCICLoss2D


def compute_l2_error_2d(model, ref_data, device):
    model.eval()
    x_ref = ref_data['x']
    y_ref = ref_data['y']
    t_ref = ref_data['t_values']
    u_ref = ref_data['u']

    all_pred, all_true = [], []
    for k, tk in enumerate(t_ref):
        xx, yy = np.meshgrid(x_ref, y_ref, indexing='ij')
        x_flat = torch.tensor(xx.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        y_flat = torch.tensor(yy.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        t_flat = torch.full_like(x_flat, float(tk))
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat).cpu().numpy().flatten()
        u_true = u_ref[:, :, k].flatten()
        all_pred.append(u_pred)
        all_true.append(u_true)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return np.sqrt(np.mean((pred - true)**2)) / max(np.sqrt(np.mean(true**2)), 1e-10)


def train_2d(model, lf_data, pde_loss_fn, data_loss_fn, bcic_loss_fn,
             N_col, device, epochs_phase1=3000, epochs_phase2=15000,
             lr=1e-3, use_transfer=True, Lx=1.0, Ly=1.0, T=0.5):
    model.to(device)

    x_lf = y_lf = t_lf = u_lf = None
    if use_transfer and lf_data is not None:
        x_lf = torch.tensor(lf_data['x'], dtype=torch.float32).unsqueeze(1).to(device)
        y_lf = torch.tensor(lf_data['y'], dtype=torch.float32).unsqueeze(1).to(device)
        t_lf = torch.tensor(lf_data['t'], dtype=torch.float32).unsqueeze(1).to(device)
        u_lf = torch.tensor(lf_data['u'], dtype=torch.float32).unsqueeze(1).to(device)

        # Phase 1: data fitting WITH BC/IC regularization
        # (prevents overfitting to sparse data and violating BCs)
        opt_p1 = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs_phase1):
            opt_p1.zero_grad()
            u_pred = model(x_lf, y_lf, t_lf)
            l_data_p1 = data_loss_fn(u_pred, u_lf)
            l_bcic_p1, _ = bcic_loss_fn(model, device)
            loss = l_data_p1 + 5.0 * l_bcic_p1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_p1.step()

    total_epochs = epochs_phase2 if use_transfer else (epochs_phase1 + epochs_phase2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6)
    pde_ramp_epochs = min(2000, total_epochs // 5)

    for epoch in range(total_epochs):
        optimizer.zero_grad()
        w_pde = min(1.0, epoch / max(pde_ramp_epochs, 1))

        x_c = torch.rand(N_col, 1, device=device) * Lx
        y_c = torch.rand(N_col, 1, device=device) * Ly
        t_c = torch.rand(N_col, 1, device=device) * T * 0.95 + T * 0.05

        l_pde, _ = pde_loss_fn(model, x_c, y_c, t_c)
        l_bcic, _ = bcic_loss_fn(model, device)

        l_data = torch.tensor(0.0, device=device)
        if use_transfer and x_lf is not None:
            u_pred = model(x_lf, y_lf, t_lf)
            l_data = data_loss_fn(u_pred, u_lf)

        # Reduced data weight to avoid fighting BC/IC losses
        w_data = 1.0 if use_transfer else 0.0
        loss = w_pde * l_pde + 10.0 * l_bcic + w_data * l_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n-col', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--epochs-p1', type=int, default=3000)
    parser.add_argument('--epochs-p2', type=int, default=15000)
    parser.add_argument('--gl-memory', type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    alpha = args.alpha
    N_col = args.n_col

    D, kappa, Lx, Ly, T = 1.0, 1.0, 1.0, 1.0, 0.5

    print(f"2D: α={alpha}, N_col={N_col}, seeds={args.seeds}")

    solver = FDRLaplaceSolver2D(
        alpha=alpha, D=D, kappa=kappa, Lx=Lx, Ly=Ly,
        bc_type='pulse', bc_params={'a': 0.5}, N_terms=40, precision=40)

    # Reference
    x_ref = np.linspace(0.05, 0.95, 20)
    y_ref = np.linspace(0.05, 0.95, 20)
    t_values = [0.05, 0.15, 0.25, 0.40]
    print(f"Generating reference: {len(x_ref)}×{len(y_ref)}×{len(t_values)} points...")
    u_ref = solver.evaluate_grid(x_ref, y_ref, np.array(t_values))
    ref_data = {'x': x_ref, 'y': y_ref, 't_values': t_values, 'u': u_ref}

    # LF data
    lf_data = generate_sparse_training_data_2d(solver, N_LF=100, seed=999)

    hp = dict(n_layers=5, n_neurons=200, activation='tanh',
              Lx=Lx, Ly=Ly, T=T, bc_type='pulse', bc_params={'a': 0.5},
              fourier_features=64, fourier_sigma=5.0)

    results = {}
    for method, use_transfer in [('MF', True), ('Vanilla', False)]:
        l2_values = []
        t0 = time.time()

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FDRNet2D(**hp)
            pde_loss = FDRPDELoss2D(alpha=alpha, D=D, kappa=kappa,
                                     gl_h=0.01, gl_N_memory=args.gl_memory)
            data_loss = FDRDataLoss2D()
            bcic_loss = FDRBCICLoss2D(Lx=Lx, Ly=Ly, T=T,
                                       bc_type='pulse', bc_params={'a': 0.5})

            model = train_2d(model, lf_data, pde_loss, data_loss, bcic_loss,
                             N_col=N_col, device=device,
                             epochs_phase1=args.epochs_p1,
                             epochs_phase2=args.epochs_p2,
                             lr=1e-3, use_transfer=use_transfer,
                             Lx=Lx, Ly=Ly, T=T)

            l2 = compute_l2_error_2d(model, ref_data, device)
            l2_values.append(l2)
            print(f"    {method} seed {seed}: L2 = {l2*100:.2f}%")

        wall = time.time() - t0
        mean_l2 = np.mean(l2_values)
        std_l2 = np.std(l2_values)
        print(f"  {method}: L2 = {mean_l2*100:.2f}% ± {std_l2*100:.2f}%  (wall {wall:.0f}s)")

        results[method] = {
            'l2_mean': float(mean_l2),
            'l2_std': float(std_l2),
            'l2_values': [float(v) for v in l2_values],
            'wall_time': wall,
        }

    ratio = results['Vanilla']['l2_mean'] / max(results['MF']['l2_mean'], 1e-8)
    print(f"\n  MF/Vanilla advantage: {ratio:.1f}×")

    # Save
    res_dir = Path('results/fdr/2d')
    res_dir.mkdir(parents=True, exist_ok=True)
    out = {'alpha': alpha, 'N_col': N_col, 'seeds': args.seeds, **results}
    fname = res_dir / f'2d_alpha{alpha}_ncol{N_col}.json'
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {fname}")


if __name__ == '__main__':
    main()
