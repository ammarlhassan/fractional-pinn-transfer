#!/usr/bin/env python3
"""
Training convergence curves: records loss components per epoch for MF vs Vanilla.

Produces data for Figure: loss decomposition showing why MF converges faster
at low N_col.

Run:  python3 run_fdr_convergence.py --device cuda
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.models.fdr_loss import FDRDataLoss, FDRPDELoss, FDRBCLoss, FDRICLoss
from mf_fpinn.training.fdr_trainer import FDRTrainer, CollocationSampler
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def eval_model(model, ref_data, device):
    model.eval()
    x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    errs = []
    with torch.no_grad():
        for j, tv in enumerate(ref_data['t']):
            t_t = torch.full_like(x_t, float(tv))
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            errs.append(compute_l2_relative_error(u_pred, ref_data['u'][:, j]))
    model.train()
    return np.mean(errs)


def train_with_logging(model, config, lf_data, ref_data, device,
                       use_transfer, log_every=50):
    """Train and log loss components + L2 error at regular intervals."""
    trainer = FDRTrainer(model, config, lf_data, device=device)

    # Loss functions for evaluation
    pde_loss_fn = FDRPDELoss(
        D=config['D'], alpha=config['alpha'],
        gl_h=config.get('gl_h', 0.01),
        gl_N_memory=config.get('gl_N_memory', 100),
    ).to(device)
    data_loss_fn = FDRDataLoss()

    history = {
        'epoch': [], 'total_loss': [], 'pde_loss': [],
        'data_loss': [], 'l2_error': [], 'phase': [],
    }

    # We need to manually implement training with logging
    model.to(device)
    model.train()

    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    sampler = CollocationSampler(
        N_col=config['N_collocation'],
        x_range=(0, config['L']),
        t_range=(0.01, config['T']),
        device=device,
    )

    if use_transfer:
        # Phase 1: data fitting
        phase1_epochs = config.get('phase1_epochs', 3000)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase1_epochs, eta_min=1e-6
        )

        for epoch in range(phase1_epochs):
            optimizer.zero_grad()
            u_pred = model(x_lf, t_lf)
            loss = data_loss_fn(u_pred, u_lf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % log_every == 0:
                # PDE loss needs gradients, so don't use no_grad
                x_col_eval, t_col_eval = sampler.sample(epoch)
                pde_l, _ = pde_loss_fn(model, x_col_eval, t_col_eval)
                l2 = eval_model(model, ref_data, device)
                model.train()
                history['epoch'].append(epoch + 1)
                history['total_loss'].append(loss.item())
                history['pde_loss'].append(pde_l.item())
                history['data_loss'].append(loss.item())
                history['l2_error'].append(l2)
                history['phase'].append('phase1')

        # Phase 2: physics
        phase2_epochs = config.get('phase2_epochs', 10000)
        optimizer2 = torch.optim.Adam(model.parameters(),
                                       lr=config.get('phase2_lr', 1e-4))
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=phase2_epochs, eta_min=1e-6
        )

        w_data = config.get('w_data', 10.0)
        w_pde = config.get('w_pde', 1.0)
        pde_ramp = config.get('pde_ramp_epochs', 2000)

        for epoch in range(phase2_epochs):
            optimizer2.zero_grad()

            u_pred_data = model(x_lf, t_lf)
            l_data = data_loss_fn(u_pred_data, u_lf)

            x_col, t_col = sampler.sample(epoch)
            l_pde, _ = pde_loss_fn(model, x_col, t_col)

            pde_scale = min(1.0, epoch / pde_ramp) if pde_ramp > 0 else 1.0
            total = w_data * l_data + w_pde * pde_scale * l_pde

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.get('grad_clip', 1.0))
            optimizer2.step()
            scheduler2.step()

            if (epoch + 1) % log_every == 0:
                with torch.no_grad():
                    l2 = eval_model(model, ref_data, device)
                    model.train()
                history['epoch'].append(phase1_epochs + epoch + 1)
                history['total_loss'].append(total.item())
                history['pde_loss'].append(l_pde.item())
                history['data_loss'].append(l_data.item())
                history['l2_error'].append(l2)
                history['phase'].append('phase2')
    else:
        # Vanilla: PDE-only training
        total_epochs = config.get('phase1_epochs', 3000) + config.get('phase2_epochs', 10000)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6
        )

        w_pde = config.get('w_pde', 1.0)

        for epoch in range(total_epochs):
            optimizer.zero_grad()

            x_col, t_col = sampler.sample(epoch)
            l_pde, _ = pde_loss_fn(model, x_col, t_col)
            total = w_pde * l_pde

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % log_every == 0:
                with torch.no_grad():
                    l2 = eval_model(model, ref_data, device)
                    model.train()
                    # Also compute data loss for comparison
                    u_pred_lf = model(x_lf, t_lf)
                    l_data = data_loss_fn(u_pred_lf, u_lf).item()
                history['epoch'].append(epoch + 1)
                history['total_loss'].append(total.item())
                history['pde_loss'].append(l_pde.item())
                history['data_loss'].append(l_data)
                history['l2_error'].append(l2)
                history['phase'].append('vanilla')

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ncol-values', type=int, nargs='+', default=[100, 500, 2000])
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Training Convergence Curves")
    print(f"  α={alpha}, N_col={args.ncol_values}")
    print(f"  Device={device}, log_every={args.log_every}")
    print(f"{'#'*70}\n")

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
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, seed=42, lf_N_terms=8, lf_precision=20,
    )
    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha}

    all_curves = {}

    for n_col in args.ncol_values:
        config['N_collocation'] = n_col
        print(f"\n{'='*60}")
        print(f"  N_col = {n_col}")
        print(f"{'='*60}")

        for method, use_transfer in [('MF', True), ('Vanilla', False)]:
            print(f"\n  {method}...")
            torch.manual_seed(0)
            np.random.seed(0)

            model = FDRNet(
                n_layers=config['n_layers'], n_neurons=config['n_neurons'],
                activation=config['activation'], L=config['L'], T=config['T'],
                hard_bc=True, bc_type=config['bc_type'],
                bc_params=config['bc_params'],
                fourier_features=config['fourier_features'],
                fourier_sigma=config['fourier_sigma'],
            )

            t0 = time.time()
            history = train_with_logging(
                model, config, lf_data, ref_data, device,
                use_transfer=use_transfer, log_every=args.log_every,
            )
            wall = time.time() - t0

            final_l2 = history['l2_error'][-1]
            print(f"    Final L2: {final_l2*100:.2f}%  ({wall:.0f}s)")

            key = f'{method}_N{n_col}'
            all_curves[key] = history

    # Save
    json_path = out_dir / f'convergence_alpha{alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(all_curves, f)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
