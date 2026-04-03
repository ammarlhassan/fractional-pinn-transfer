#!/usr/bin/env python3
"""
Quick control test: does grad_clip=1.0 vs 5.0 materially affect vanilla fPINN?

Background: train_vanilla() hardcodes clip=1.0, while train_phase2() uses
config['grad_clip']=5.0. This tests whether that discrepancy matters.

Tests:
  - Vanilla with clip=1.0 (current code behavior)
  - Vanilla with clip=5.0 (what config says)
  - MF with clip=5.0 (current code behavior)
  - MF with clip=1.0 (matched to vanilla)
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
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


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


def evaluate_model(model, ref_data, device):
    model.eval()
    results = {}
    x_ref = ref_data['x']
    u_grid = ref_data['u']
    t_vals = ref_data['t']
    with torch.no_grad():
        for j, t_val in enumerate(t_vals):
            x_t = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = u_grid[:, j]
            l2 = compute_l2_relative_error(u_pred, u_ref)
            results[f't{t_val:.2f}_u_l2'] = l2
    return results


def run_single(method, config, lf_data, ref_data, seed, device, grad_clip_override=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Override grad_clip in config if requested
    cfg = dict(config)
    if grad_clip_override is not None:
        cfg['grad_clip'] = grad_clip_override

    model = build_model(cfg)
    trainer = FDRTrainer(model, cfg, lf_data, device=device)

    # Monkey-patch vanilla to use config grad_clip instead of hardcoded 1.0
    if method.startswith('vanilla_clip'):
        clip_val = float(method.split('clip')[1])
        original_train = trainer.train_vanilla

        def patched_train(verbose=False, _clip=clip_val, _trainer=trainer):
            """Vanilla with configurable grad_clip."""
            import types
            epochs = _trainer.config.get('vanilla_epochs',
                         _trainer.config.get('phase2_epochs', 20000))
            optimizer = _trainer._build_optimizer()
            scheduler = _trainer._build_scheduler(optimizer, epochs)

            w_pde = _trainer.config.get('w_pde', 1.0)
            w_bc = _trainer.config.get('w_bc', 1.0)
            w_ic = _trainer.config.get('w_ic', 1.0)
            ramp = _trainer.config.get('pde_ramp_epochs', 2000)

            _trainer.model.train()
            start = time.time()
            for epoch in range(epochs):
                optimizer.zero_grad()
                x_col, t_col = _trainer.sampler.sample(epoch)
                l_pde, _ = _trainer.pde_loss_fn(_trainer.model, x_col, t_col)
                l_bc = _trainer.bc_loss_fn(_trainer.model, _trainer.t_bc)
                l_ic = _trainer.ic_loss_fn(_trainer.model, _trainer.x_ic)
                pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
                total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic
                total.backward()
                torch.nn.utils.clip_grad_norm_(_trainer.model.parameters(), _clip)
                optimizer.step()
                scheduler.step()
                _trainer._log(0, epoch, total.item(), 0, l_pde.item(),
                              l_bc.item(), l_ic.item(),
                              optimizer.param_groups[0]['lr'], start)
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                          f"Total: {total.item():.4e}", flush=True)
            return total.item()

        t0 = time.time()
        patched_train(verbose=False)
        wall = time.time() - t0
    elif method.startswith('mf_clip'):
        clip_val = float(method.split('clip')[1])
        cfg['grad_clip'] = clip_val
        model2 = build_model(cfg)
        trainer2 = FDRTrainer(model2, cfg, lf_data, device=device)
        t0 = time.time()
        trainer2.train_full(verbose=False)
        wall = time.time() - t0
        model = model2
    else:
        raise ValueError(method)

    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n_col', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  Grad Clip Control Test")
    print(f"  α={alpha}, N_col={args.n_col}, seeds={args.seeds}")
    print(f"{'#'*70}\n")

    # Reference
    ref_path = Path(f'results/fdr/reference_alpha{alpha}.npz')
    ref_data = dict(np.load(ref_path, allow_pickle=True))
    print(f"  Loaded reference from {ref_path}")

    # LF data (N_terms=20 for MF)
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=20, lf_precision=20,
    )

    config = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': args.n_col}

    methods = ['vanilla_clip1.0', 'vanilla_clip5.0', 'mf_clip1.0', 'mf_clip5.0']

    all_results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        method_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...", end=' ', flush=True)
            metrics = run_single(method, config, lf_data, ref_data, seed, device)
            method_results.append(metrics)
            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            avg = np.mean(l2_vals)
            print(f"L2={avg*100:.2f}%  Wall={metrics['wall_time']:.0f}s")

        all_results[method] = method_results

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  GRAD CLIP CONTROL RESULTS  (α={alpha}, N_col={args.n_col})")
    print(f"{'#'*70}")
    print(f"\n  {'Method':>20s}  {'Mean L2':>10s}  {'Std':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}")

    for method, results in all_results.items():
        l2s = []
        for r in results:
            l2s.append(np.mean([v for k, v in r.items() if k.endswith('_l2')]))
        mean = np.mean(l2s) * 100
        std = np.std(l2s, ddof=1) * 100
        print(f"  {method:>20s}  {mean:>9.2f}%  {std:>7.2f}%")

    # Key comparison
    print(f"\n  Key comparisons:")
    van1 = np.mean([np.mean([v for k, v in r.items() if k.endswith('_l2')])
                     for r in all_results['vanilla_clip1.0']]) * 100
    van5 = np.mean([np.mean([v for k, v in r.items() if k.endswith('_l2')])
                     for r in all_results['vanilla_clip5.0']]) * 100
    mf1 = np.mean([np.mean([v for k, v in r.items() if k.endswith('_l2')])
                    for r in all_results['mf_clip1.0']]) * 100
    mf5 = np.mean([np.mean([v for k, v in r.items() if k.endswith('_l2')])
                    for r in all_results['mf_clip5.0']]) * 100

    print(f"  Vanilla clip effect: clip=1.0 {van1:.2f}% vs clip=5.0 {van5:.2f}%  "
          f"(ratio: {van1/van5:.2f}×)")
    print(f"  MF clip effect:      clip=1.0 {mf1:.2f}% vs clip=5.0 {mf5:.2f}%  "
          f"(ratio: {mf1/mf5:.2f}×)")
    print(f"  Fair comparison:     Vanilla(clip=5) {van5:.2f}% vs MF(clip=5) {mf5:.2f}%  "
          f"(MF advantage: {van5/mf5:.2f}×)")
    print(f"  Current (buggy):     Vanilla(clip=1) {van1:.2f}% vs MF(clip=5) {mf5:.2f}%  "
          f"(MF advantage: {van1/mf5:.2f}×)")

    # Save
    out_dir = Path('results/gradclip_control')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        'alpha': alpha, 'N_col': args.n_col, 'seeds': args.seeds,
        'results': all_results,
    }
    json_path = out_dir / f'gradclip_alpha{alpha}_ncol{args.n_col}_{args.seeds}seeds.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
