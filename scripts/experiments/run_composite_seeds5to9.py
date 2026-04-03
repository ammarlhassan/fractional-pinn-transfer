#!/usr/bin/env python3
"""
Run composite comparison seeds 5-9 to upgrade from 5 to 10 seeds.
Appends results to existing composite_alpha0.5_ncol100_nterms20.json.

Run:  python3 run_composite_seeds5to9.py --device cuda:0
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.models.composite_nn import CompositeMFNet
from mf_fpinn.models.fdr_loss import FDRDataLoss, FDRPDELoss, FDRBCLoss, FDRICLoss
from mf_fpinn.training.fdr_trainer import FDRTrainer, CollocationSampler
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha, n_col=100):
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': n_col}
    return cfg


def build_fdrnet(config):
    return FDRNet(
        n_layers=config['n_layers'], n_neurons=config['n_neurons'],
        activation=config['activation'], L=config['L'], T=config['T'],
        hard_bc=True, bc_type=config['bc_type'], bc_params=config['bc_params'],
        fourier_features=config['fourier_features'], fourier_sigma=config['fourier_sigma'],
    )


def build_composite(config):
    return CompositeMFNet(
        n_layers_lf=config['n_layers'], n_neurons_lf=config['n_neurons'],
        n_layers_lin=2, n_neurons_lin=64, n_layers_nl=4, n_neurons_nl=128,
        activation=config['activation'], L=config['L'], T=config['T'],
        hard_bc=True, bc_type=config['bc_type'], bc_params=config['bc_params'],
        fourier_features=config['fourier_features'], fourier_sigma=config['fourier_sigma'],
    )


def evaluate_model(model, ref_data, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            l2 = compute_l2_relative_error(u_pred, ref_data['u'][:, j])
            results[f't{t_val:.2f}_u_l2'] = l2
    return results


def train_composite(model, config, lf_data, device):
    model = model.to(device)
    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = FDRDataLoss(u_scale=u_scale)
    pde_loss_fn = FDRPDELoss(
        alpha=config['alpha'], D=config['D'], kappa=config['kappa'],
        gl_h=config.get('gl_h', 0.01), gl_N_memory=config.get('gl_N_memory', 100),
    ).to(device)
    bc_loss_fn = FDRBCLoss(L=config['L'])
    ic_loss_fn = FDRICLoss()
    sampler = CollocationSampler(
        N_col=config.get('N_collocation', 100),
        x_range=(0.0, config['L']), t_range=(0.01, config['T']), device=device,
    )

    epochs = config.get('vanilla_epochs', 25000)
    t_bc = torch.linspace(0.01, config['T'], 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, config['L'], 50, device=device).unsqueeze(1)
    ramp = config.get('pde_ramp_epochs', 3000)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 5e-4),
                                  weight_decay=config.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    model.train()
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_lf_pred = model.forward_lf(x_lf, t_lf)
        l_data = data_loss_fn(u_lf_pred, u_lf)
        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(model, x_col, t_col)
        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)
        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
        total = config.get('w_data', 10.0) * l_data + pde_scale * l_pde + l_bc + l_ic
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 5.0))
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 5000 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Total: {total.item():.4e} | "
                  f"LF: {l_data.item():.4e} | PDE: {l_pde.item():.4e} | "
                  f"{time.time()-start:.0f}s", flush=True)


def run_single(method, config, lf_data, ref_data, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    if method == 'composite':
        model = build_composite(config)
        train_composite(model, config, lf_data, device)
    elif method == 'M3_mf':
        model = build_fdrnet(config)
        trainer = FDRTrainer(model, config, lf_data, device=device)
        trainer.train_full(verbose=False)
    elif method == 'M1_vanilla':
        model = build_fdrnet(config)
        trainer = FDRTrainer(model, config, lf_data, device=device)
        trainer.train_vanilla(verbose=False)

    wall = time.time() - t0
    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_seed', type=int, default=5)
    parser.add_argument('--end_seed', type=int, default=10)  # exclusive
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = 0.5
    n_col = 100
    n_terms = 20

    print(f"\n{'#'*70}")
    print(f"  Composite NN seeds {args.start_seed}..{args.end_seed-1}")
    print(f"  alpha={alpha}, N_col={n_col}, N_terms={n_terms}")
    print(f"  device={device}")
    print(f"{'#'*70}\n")

    # Load existing results
    existing_path = Path('results/fdr/composite/composite_alpha0.5_ncol100_nterms20.json')
    with open(existing_path) as f:
        existing = json.load(f)

    # Reference data
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )
    ref_path = Path('results/fdr') / f'reference_alpha{alpha}.npz'
    ref_data = dict(np.load(ref_path, allow_pickle=True))
    print(f"  Loaded cached reference")

    # LF data
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=n_terms, lf_precision=20,
    )

    config = build_config(alpha, n_col=n_col)
    methods = ['composite', 'M3_mf', 'M1_vanilla']
    new_results = {m: [] for m in methods}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        for seed in range(args.start_seed, args.end_seed):
            print(f"\n--- Seed {seed} ---", flush=True)
            metrics = run_single(method, config, lf_data, ref_data, seed, device)
            new_results[method].append(metrics)
            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            print(f"  -> Avg L2: {np.mean(l2_vals)*100:.2f}%  Wall: {metrics['wall_time']:.1f}s")

    # Merge with existing and save combined
    combined = {
        'config': {**existing['config'], 'seeds': args.end_seed},
        'results': {}
    }
    for method in methods:
        combined['results'][method] = existing['results'][method] + new_results[method]

    out_path = Path('results/fdr/composite/composite_alpha0.5_ncol100_nterms20_10seeds.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results saved to {out_path}")

    # Summary
    print(f"\n{'#'*70}")
    print(f"  10-SEED SUMMARY")
    print(f"{'#'*70}")
    for method in methods:
        all_seeds = combined['results'][method]
        per_seed_avg = []
        for r in all_seeds:
            l2_vals = [v for k, v in r.items() if k.endswith('_l2')]
            per_seed_avg.append(np.mean(l2_vals))
        m = np.mean(per_seed_avg)
        s = np.std(per_seed_avg, ddof=1)
        print(f"  {method:15s}: {m*100:.2f}% ± {s*100:.2f}%")

    # Wilcoxon tests
    from scipy.stats import wilcoxon, mannwhitneyu
    avg_l2 = {}
    for method in methods:
        avg_l2[method] = [np.mean([v for k, v in r.items() if k.endswith('_l2')])
                          for r in combined['results'][method]]

    print("\n  Wilcoxon tests (two-sided):")
    for name, m1, m2 in [('MF vs Vanilla', 'M3_mf', 'M1_vanilla'),
                          ('Composite vs Vanilla', 'composite', 'M1_vanilla'),
                          ('MF vs Composite', 'M3_mf', 'composite')]:
        stat, p = wilcoxon(avg_l2[m1], avg_l2[m2])
        print(f"    {name:30s}: p={p:.4f}")


if __name__ == '__main__':
    main()
