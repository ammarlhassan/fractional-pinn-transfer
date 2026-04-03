#!/usr/bin/env python3
"""
Convergence Speed Tracking — L2 error vs training epoch.

Evaluates L2 error every `eval_interval` epochs during training for both
MF and Vanilla. Shows:
1. MF head start after Phase 1
2. Convergence speed during Phase 2 vs Vanilla
3. Epochs-to-target-error comparison

Run:  python3 run_convergence_tracking.py --alpha 0.5 --device cuda:5 --seeds 5
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
from mf_fpinn.models.fdr_loss import FDRPDELoss
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


def build_config(alpha, n_col):
    return {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': n_col}


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


def evaluate_l2(model, ref_data, device):
    model.eval()
    l2_vals = []
    with torch.no_grad():
        for j, t_val in enumerate(ref_data['t']):
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            l2 = compute_l2_relative_error(u_pred, ref_data['u'][:, j])
            l2_vals.append(l2)
    model.train()
    return np.mean(l2_vals)


def train_with_tracking(model, config, lf_data, ref_data, device,
                         method='mf', eval_interval=500):
    """Train with periodic L2 evaluation."""
    from mf_fpinn.training.fdr_trainer import FDRTrainer

    tracker = {'epochs': [], 'l2': [], 'loss': [], 'phase': []}

    trainer = FDRTrainer(model, config, lf_data, device=device)

    if method == 'mf':
        # Phase 1
        trainer.train_phase1(verbose=False)
        l2_after_p1 = evaluate_l2(model, ref_data, device)
        tracker['epochs'].append(0)  # epoch 0 of Phase 2
        tracker['l2'].append(l2_after_p1)
        tracker['loss'].append(float(trainer.history['total_loss'][-1]) if trainer.history['total_loss'] else 0)
        tracker['phase'].append(1)

        # Phase 2 with periodic eval
        p2_epochs = config.get('phase2_epochs', 20000)
        total_eval_points = p2_epochs // eval_interval

        # We need to train Phase 2 in chunks
        # Save the original phase2_epochs and do manual chunked training
        for chunk_end in range(eval_interval, p2_epochs + 1, eval_interval):
            chunk_config = {**config, 'phase2_epochs': eval_interval}

            if chunk_end == eval_interval:
                # First chunk: normal Phase 2 start
                trainer.config['phase2_epochs'] = eval_interval
                trainer.train_phase2(verbose=False)
            else:
                # Subsequent chunks: continue training
                trainer.config['phase2_epochs'] = eval_interval
                trainer.train_phase2(verbose=False)

            l2 = evaluate_l2(model, ref_data, device)
            tracker['epochs'].append(chunk_end)
            tracker['l2'].append(l2)
            tracker['loss'].append(float(trainer.history['total_loss'][-1]) if trainer.history['total_loss'] else 0)
            tracker['phase'].append(2)

    else:
        # Vanilla: train in chunks
        van_epochs = config.get('vanilla_epochs', 25000)
        for chunk_end in range(eval_interval, van_epochs + 1, eval_interval):
            trainer.config['vanilla_epochs'] = eval_interval
            trainer.train_vanilla(verbose=False)

            l2 = evaluate_l2(model, ref_data, device)
            tracker['epochs'].append(chunk_end)
            tracker['l2'].append(l2)
            tracker['loss'].append(float(trainer.history['total_loss'][-1]) if trainer.history['total_loss'] else 0)
            tracker['phase'].append(0)

    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, default=15)
    parser.add_argument('--eval-interval', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    alpha = args.alpha

    print(f"\n{'#'*70}")
    print(f"  CONVERGENCE TRACKING")
    print(f"  α={alpha}, N_col={args.n_col}, N_terms={args.n_terms}")
    print(f"  Eval every {args.eval_interval} epochs, Seeds={args.seeds}")
    print(f"{'#'*70}\n")

    # Reference
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/fdr')
    ref_path = out_dir / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    # LF data
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=50, lf_N_terms=args.n_terms, lf_precision=20,
    )
    lf_err = compute_lf_error(solver, lf_N_terms=args.n_terms, lf_precision=20, N_test=50)
    print(f"  LF error: {lf_err['l2_relative']*100:.2f}%")

    config = build_config(alpha, args.n_col)
    results = {'mf': {}, 'vanilla': {}}

    for method in ['mf', 'vanilla']:
        print(f"\n{'='*60}")
        print(f"  {method.upper()}")
        print(f"{'='*60}")

        all_tracks = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...", end='', flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_model(config)
            tracker = train_with_tracking(
                model, config, lf_data, ref_data, device,
                method=method, eval_interval=args.eval_interval
            )
            all_tracks.append(tracker)
            final_l2 = tracker['l2'][-1]
            print(f" final L2={final_l2*100:.2f}%")

        # Aggregate
        epochs = all_tracks[0]['epochs']
        l2_matrix = np.array([t['l2'] for t in all_tracks])  # (S, n_checkpoints)
        results[method] = {
            'epochs': epochs,
            'l2_mean': np.mean(l2_matrix, axis=0).tolist(),
            'l2_std': np.std(l2_matrix, axis=0).tolist(),
            'l2_all': l2_matrix.tolist(),
        }

    # Compute convergence speed: epochs to reach various thresholds
    print(f"\n{'#'*70}")
    print(f"  CONVERGENCE SPEED COMPARISON")
    print(f"{'#'*70}")

    thresholds = [0.20, 0.15, 0.10, 0.08, 0.05]
    print(f"  {'Threshold':>12} {'MF epoch':>10} {'Van epoch':>10} {'Speedup':>10}")
    print(f"  {'-'*42}")

    convergence_speeds = {}
    for thresh in thresholds:
        mf_epochs = results['mf']['epochs']
        mf_l2 = results['mf']['l2_mean']
        van_epochs = results['vanilla']['epochs']
        van_l2 = results['vanilla']['l2_mean']

        mf_reach = None
        for i, (e, l) in enumerate(zip(mf_epochs, mf_l2)):
            if l <= thresh:
                mf_reach = e
                break

        van_reach = None
        for i, (e, l) in enumerate(zip(van_epochs, van_l2)):
            if l <= thresh:
                van_reach = e
                break

        mf_str = str(mf_reach) if mf_reach else 'never'
        van_str = str(van_reach) if van_reach else 'never'
        speedup = ''
        if mf_reach and van_reach:
            speedup = f'{van_reach/mf_reach:.1f}×'
        elif mf_reach and not van_reach:
            speedup = '∞'

        print(f"  {thresh*100:>10.0f}% {mf_str:>10} {van_str:>10} {speedup:>10}")
        convergence_speeds[f'{thresh*100:.0f}pct'] = {
            'mf_epoch': mf_reach, 'van_epoch': van_reach
        }

    results['convergence_speeds'] = convergence_speeds

    json_path = out_dir / f'convergence_tracking_alpha{alpha}_ncol{args.n_col}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
