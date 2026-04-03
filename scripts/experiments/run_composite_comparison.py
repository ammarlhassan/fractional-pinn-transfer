#!/usr/bin/env python3
"""
Composite NN (Meng-Karniadakis 2020) vs MF-fPINN vs Vanilla fPINN comparison.

Benchmarks the Meng-Karniadakis composite neural network architecture against
our two-phase MF-fPINN and vanilla fPINN on the 1D fractional diffusion-reaction
equation.

Run:
    python3 run_composite_comparison.py --device cuda --seeds 5
    python3 run_composite_comparison.py --device cuda --seeds 5 --n_terms 12
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver,
    generate_low_fidelity_data, compute_lf_error,
)
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.models.composite_nn import CompositeMFNet
from mf_fpinn.models.fdr_loss import FDRDataLoss, FDRPDELoss, FDRBCLoss, FDRICLoss
from mf_fpinn.training.fdr_trainer import FDRTrainer, CollocationSampler
from mf_fpinn.experiments.fdr_configs import MANUAL_HP, PHYSICS, BC, EVAL_TIMES, REFERENCE
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


# ── Config helpers ────────────────────────────────────────────

def build_config(alpha, n_col=100):
    """Merge physics + HP config for a given alpha."""
    cfg = {**MANUAL_HP, **PHYSICS, **BC, 'alpha': alpha, 'N_collocation': n_col}
    return cfg


def build_fdrnet(config):
    """Create a fresh FDRNet from config."""
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


def build_composite(config):
    """Create a fresh CompositeMFNet from config."""
    return CompositeMFNet(
        n_layers_lf=config['n_layers'],
        n_neurons_lf=config['n_neurons'],
        n_layers_lin=2,
        n_neurons_lin=64,
        n_layers_nl=4,
        n_neurons_nl=128,
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
    """Compute L2 errors at each time snapshot."""
    model.eval()
    results = {}
    x_ref = ref_data['x']
    u_grid = ref_data['u']   # shape (nx, nt)
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


# ── Composite NN Trainer ──────────────────────────────────────

def train_composite(model, config, lf_data, device, verbose=True):
    """
    Train the composite NN with joint loss:
        L = L_PDE + w_lf * L_data_LF

    L_data_LF: MSE between model.forward_lf(x,t) and u_LF (trains NN_L).
    L_PDE: GL fractional derivative loss on model(x,t) (trains all three nets).

    Same optimizer settings as MF-fPINN/Vanilla: Adam + cosine annealing.
    Total epochs = vanilla_epochs (25000) for fair budget comparison.
    """
    model = model.to(device)

    # LF data tensors
    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    # Loss functions
    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = FDRDataLoss(u_scale=u_scale)
    pde_loss_fn = FDRPDELoss(
        alpha=config['alpha'], D=config['D'], kappa=config['kappa'],
        gl_h=config.get('gl_h', 0.01),
        gl_N_memory=config.get('gl_N_memory', 100),
    ).to(device)
    bc_loss_fn = FDRBCLoss(L=config['L'])
    ic_loss_fn = FDRICLoss()

    # Collocation sampler
    sampler = CollocationSampler(
        N_col=config.get('N_collocation', 2000),
        x_range=(0.0, config['L']),
        t_range=(0.01, config['T']),
        device=device,
    )

    # BC/IC evaluation points
    T = config['T']
    L = config['L']
    t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, L, 50, device=device).unsqueeze(1)

    # Training config
    epochs = config.get('vanilla_epochs', 25000)  # Same total budget
    lr = config.get('lr', 5e-4)
    w_lf = config.get('w_data', 10.0)
    w_pde = config.get('w_pde', 1.0)
    w_bc = config.get('w_bc', 1.0)
    w_ic = config.get('w_ic', 1.0)
    ramp = config.get('pde_ramp_epochs', 3000)
    grad_clip = config.get('grad_clip', 5.0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
        weight_decay=config.get('weight_decay', 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPOSITE NN (Meng-Karniadakis) — {epochs} epochs")
        print(f"  LF data points: {len(x_lf)}")
        print(f"  Weights: w_lf={w_lf}, w_pde={w_pde}, w_bc={w_bc}, w_ic={w_ic}")
        print(f"  PDE ramp: {ramp} epochs")
        params = model.param_summary()
        print(f"  Params: NN_L={params['nn_lf']:,}, NN_lin={params['nn_lin']:,}, "
              f"NN_nl={params['nn_nl']:,}, total={params['total']:,}")
        print(f"{'='*60}", flush=True)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # LF data loss (on NN_L only)
        u_lf_pred = model.forward_lf(x_lf, t_lf)
        l_data_lf = data_loss_fn(u_lf_pred, u_lf)

        # PDE loss (on composite output)
        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(model, x_col, t_col)

        # BC/IC losses (monitoring; hard BC enforces them)
        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)

        # PDE ramp-up
        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0

        total = w_lf * l_data_lf + w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                  f"Total: {total.item():.4e}  LF_data: {l_data_lf.item():.4e}  "
                  f"PDE: {l_pde.item():.4e}  BC: {l_bc.item():.4e}  "
                  f"IC: {l_ic.item():.4e}  |  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start
    if verbose:
        print(f"  Composite done in {elapsed:.1f}s  |  "
              f"Final: {total.item():.6e}", flush=True)

    return total.item()


# ── Single-run helpers ────────────────────────────────────────

def run_single(method, config, lf_data, ref_data, seed, device, verbose=True):
    """Run one seed of one method. Returns metrics dict and wall time."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()

    if method == 'composite':
        model = build_composite(config)
        train_composite(model, config, lf_data, device, verbose=verbose)
    elif method == 'M3_mf':
        model = build_fdrnet(config)
        trainer = FDRTrainer(model, config, lf_data, device=device)
        trainer.train_full(verbose=verbose)
    elif method == 'M1_vanilla':
        model = build_fdrnet(config)
        trainer = FDRTrainer(model, config, lf_data, device=device)
        trainer.train_vanilla(verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")

    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics, model


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Composite NN vs MF-fPINN vs Vanilla comparison"
    )
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n_lf', type=int, default=50)
    parser.add_argument('--n_col', type=int, default=100,
                        help='Number of collocation points')
    parser.add_argument('--n_terms', type=int, default=20,
                        help='LF solver N_terms (20=good LF, 12=moderate LF)')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    verbose = not args.quiet
    alpha = args.alpha
    n_terms = args.n_terms

    print(f"\n{'#'*70}")
    print(f"  Composite NN vs MF-fPINN vs Vanilla fPINN")
    print(f"  alpha={alpha}, N_col={args.n_col}, N_terms={n_terms}")
    print(f"  device={device}, seeds={args.seeds}")
    print(f"{'#'*70}\n")

    # ── 1. Generate reference data ──────────────────────────
    print("Generating reference solution...")
    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_terms=REFERENCE['N_terms'], precision=REFERENCE['precision'],
    )

    out_dir = Path('results/fdr/composite')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path('results/fdr') / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = solver.generate_reference_data(
            nx=REFERENCE['nx'], t_values=EVAL_TIMES, save_path=str(ref_path)
        )

    print(f"  u peak at t={EVAL_TIMES[0]}: {ref_data['u'][:, 0].max():.4f}")

    # ── 2. Generate genuine low-fidelity data ───────────────
    lf_precision = 20

    print(f"\nQuantifying LF solver degradation (N_terms={n_terms}, "
          f"precision={lf_precision})...")
    lf_err = compute_lf_error(solver, lf_N_terms=n_terms,
                              lf_precision=lf_precision, N_test=50)
    print(f"  LF solver error vs HF: L2_rel = {lf_err['l2_relative']*100:.2f}%, "
          f"Linf_rel = {lf_err['linf_relative']*100:.2f}%")

    print(f"\nGenerating {args.n_lf} genuine low-fidelity training points "
          f"(N_terms={n_terms})...")
    lf_data = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS['D'], kappa=PHYSICS['kappa'], L=PHYSICS['L'],
        bc_type=BC['bc_type'], bc_params=BC['bc_params'],
        N_LF=args.n_lf, lf_N_terms=n_terms, lf_precision=lf_precision,
    )

    # ── 3. Run comparison ──────────────────────────────────
    config = build_config(alpha, n_col=args.n_col)

    # Report parameter counts
    fdr_model = build_fdrnet(config)
    comp_model = build_composite(config)
    n_fdr = sum(p.numel() for p in fdr_model.parameters())
    comp_summary = comp_model.param_summary()
    print(f"\nParameter counts:")
    print(f"  FDRNet (MF/Vanilla): {n_fdr:,}")
    print(f"  Composite NN: {comp_summary['total']:,} "
          f"(NN_L={comp_summary['nn_lf']:,}, "
          f"NN_lin={comp_summary['nn_lin']:,}, "
          f"NN_nl={comp_summary['nn_nl']:,})")
    del fdr_model, comp_model

    methods = ['composite', 'M3_mf', 'M1_vanilla']
    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        method_results = []

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            metrics, model = run_single(
                method, config, lf_data, ref_data, seed, device, verbose
            )
            method_results.append(metrics)

            # Print summary for this seed
            l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
            avg = np.mean(l2_vals)
            print(f"  -> Avg L2: {avg*100:.2f}%  |  Wall: {metrics['wall_time']:.1f}s")

        all_results[method] = method_results

    # ── 4. Summary ──────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"  alpha={alpha}, N_col={args.n_col}, N_terms={n_terms} "
          f"(LF error: {lf_err['l2_relative']*100:.1f}%)")
    print(f"{'#'*70}")

    # Per-snapshot results
    for method, results in all_results.items():
        print(f"\n  {method}:")
        for t_val in EVAL_TIMES:
            key = f't{t_val:.2f}_u_l2'
            vals = [r[key] for r in results]
            print(f"    t={t_val:.2f}:  L2 = {np.mean(vals)*100:.2f}% "
                  f"+/- {np.std(vals)*100:.2f}%")
        walls = [r['wall_time'] for r in results]
        print(f"    Wall time: {np.mean(walls):.1f}s +/- {np.std(walls):.1f}s")

    # Average L2 across snapshots
    print(f"\n  Average L2 across snapshots:")
    avg_l2 = {}
    for method, results in all_results.items():
        per_seed_avg = []
        for r in results:
            l2_vals = [v for k, v in r.items() if k.endswith('_l2')]
            per_seed_avg.append(np.mean(l2_vals))
        avg_l2[method] = per_seed_avg
        mean_l2 = np.mean(per_seed_avg)
        std_l2 = np.std(per_seed_avg)
        print(f"    {method:15s}: {mean_l2*100:.2f}% +/- {std_l2*100:.2f}%")

    # Advantage ratios vs Vanilla
    van_avg = np.mean(avg_l2['M1_vanilla'])
    print(f"\n  Advantage ratios (Vanilla / Method):")
    for method in ['composite', 'M3_mf']:
        method_avg = np.mean(avg_l2[method])
        if method_avg > 1e-8:
            ratio = van_avg / method_avg
            print(f"    {method:15s}: {ratio:.2f}x")

    # Wilcoxon signed-rank test (composite vs MF, composite vs vanilla)
    try:
        from scipy.stats import wilcoxon
        print(f"\n  Wilcoxon signed-rank tests (paired, per-seed avg L2):")

        for pair_name, m1, m2 in [
            ('Composite vs Vanilla', 'composite', 'M1_vanilla'),
            ('MF vs Vanilla', 'M3_mf', 'M1_vanilla'),
            ('Composite vs MF', 'composite', 'M3_mf'),
        ]:
            a = np.array(avg_l2[m1])
            b = np.array(avg_l2[m2])
            if len(a) >= 5:
                stat, p = wilcoxon(a, b)
                direction = "better" if np.mean(a) < np.mean(b) else "worse"
                print(f"    {pair_name:30s}: p={p:.4f} ({m1} {direction})")
            else:
                print(f"    {pair_name:30s}: not enough seeds for Wilcoxon "
                      f"(need >=5, got {len(a)})")
    except ImportError:
        print("\n  (scipy not available for Wilcoxon test)")

    # ── 5. Save results ────────────────────────────────────
    save_data = {
        'config': {
            'alpha': alpha, 'N_col': args.n_col, 'N_terms': n_terms,
            'seeds': args.seeds, 'n_lf': args.n_lf,
            'lf_l2_error': lf_err['l2_relative'],
        },
        'results': {},
    }
    for method, results in all_results.items():
        save_data['results'][method] = results

    json_path = out_dir / f'composite_alpha{alpha}_ncol{args.n_col}_nterms{n_terms}.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
