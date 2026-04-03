#!/usr/bin/env python3
"""
MF-fPINN vs Vanilla fPINN comparison for the time-fractional Burgers equation.

Run:  python3 run_burgers_comparison.py [--alpha 0.5] [--device cuda] [--seeds 5]
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from mf_fpinn.solvers.burgers_solver import (
    solve_fractional_burgers,
    generate_reference_burgers,
    generate_lf_data_burgers,
    compute_lf_error_burgers,
)
from mf_fpinn.models.burgers_pinn import BurgersNet
from mf_fpinn.models.burgers_loss import (
    BurgersDataLoss, BurgersPDELoss, BurgersBCLoss, BurgersICLoss,
)
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


# ── Default configuration ────────────────────────────────────
DEFAULT_CONFIG = {
    # Physics
    'nu': 0.02,        # Low viscosity for steeper gradients / larger LF gap
    'L': 1.0,
    'T': 1.0,
    # Architecture
    'n_layers': 5,
    'n_neurons': 128,
    'activation': 'tanh',
    'fourier_features': 64,
    'fourier_sigma': 5.0,
    # Training
    'lr': 5e-4,
    'phase2_lr': 1e-4,
    'weight_decay': 1e-5,
    'lr_schedule': 'cosine',
    'N_collocation': 100,
    'phase1_epochs': 5000,
    'phase1_lbfgs': True,
    'phase1_lbfgs_steps': 50,
    'phase2_epochs': 20000,
    'vanilla_epochs': 25000,
    'w_data': 10.0,
    'w_pde': 1.0,
    'w_bc': 1.0,
    'w_ic': 1.0,
    'pde_ramp_epochs': 3000,
    'grad_clip': 5.0,
    'gl_h': 0.01,
    'gl_N_memory': 100,
    # LF data (Nx=8,Nt=80 → ~18% LF error at ν=0.02; comparable to FDR N_terms=12)
    'N_LF': 50,
    'lf_Nx': 8,
    'lf_Nt': 80,
    # HF reference
    'hf_Nx': 400,
    'hf_Nt': 4000,
}

# Evaluation time snapshots
EVAL_TIMES = [0.1, 0.3, 0.5, 0.8]


def build_model(config):
    """Create a fresh BurgersNet from config."""
    return BurgersNet(
        n_layers=config['n_layers'],
        n_neurons=config['n_neurons'],
        activation=config['activation'],
        L=config['L'],
        T=config['T'],
        hard_bc=True,
        fourier_features=config['fourier_features'],
        fourier_sigma=config['fourier_sigma'],
    )


def evaluate_model(model, ref_data, eval_times, device):
    """Compute L2 errors at each time snapshot by interpolating from the full grid."""
    from scipy.interpolate import RegularGridInterpolator
    model.eval()
    results = {}

    x_ref = ref_data['x']
    t_ref = ref_data['t']
    u_grid = ref_data['u']  # shape (Nx+1, Nt+1)

    # Build interpolator from the full grid
    interp = RegularGridInterpolator(
        (x_ref, t_ref), u_grid, method='linear',
        bounds_error=False, fill_value=0.0
    )

    # Evaluate at interior x points for each time snapshot
    nx_eval = 100
    x_eval = np.linspace(0.01, ref_data['L'] - 0.01, nx_eval)

    with torch.no_grad():
        for t_val in eval_times:
            x_t = torch.tensor(x_eval, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, t_t).cpu().numpy().flatten()

            # Reference at these points
            pts = np.column_stack([x_eval, np.full(nx_eval, t_val)])
            u_ref = interp(pts)

            l2 = compute_l2_relative_error(u_pred, u_ref)
            results[f't{t_val:.2f}_l2'] = l2

    return results


class BurgersCollocationSampler:
    """Sample collocation points with Beta-biased distribution."""

    def __init__(self, N_col=100, x_range=(0.0, 1.0), t_range=(0.01, 1.0),
                 device=torch.device('cpu'), resample_every=1000):
        self.N_col = N_col
        self.x_range = x_range
        self.t_range = t_range
        self.device = device
        self.resample_every = resample_every
        self._current = None

    def sample(self, epoch=0):
        if self._current is None or epoch % self.resample_every == 0:
            rng = np.random.default_rng(epoch)
            x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * \
                rng.beta(0.5, 2.0, self.N_col)
            t = self.t_range[0] + (self.t_range[1] - self.t_range[0]) * \
                rng.beta(0.5, 2.0, self.N_col)
            self._current = (
                torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(1),
                torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(1),
            )
        return self._current


def train_phase1(model, lf_data, config, device, verbose=True):
    """Phase 1: Pure data fitting on LF data."""
    epochs = config.get('phase1_epochs', 5000)

    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = BurgersDataLoss(u_scale=u_scale)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Data Pre-training ({epochs} epochs)")
        print(f"  Data points: {len(x_lf)}")
        print(f"{'='*60}", flush=True)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(x_lf, t_lf)
        loss = data_loss_fn(u_pred, u_lf)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                  f"Loss: {loss.item():.6e}  |  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    # L-BFGS refinement
    if config.get('phase1_lbfgs', True):
        n_steps = config.get('phase1_lbfgs_steps', 50)
        if verbose:
            print(f"  L-BFGS refinement ({n_steps} steps)...", flush=True)

        with torch.no_grad():
            pre_lbfgs_loss = data_loss_fn(model(x_lf, t_lf), u_lf).item()
        pre_lbfgs_state = {k: v.clone() for k, v in model.state_dict().items()}

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=20, history_size=50,
            line_search_fn='strong_wolfe'
        )
        for _ in range(n_steps):
            def closure():
                lbfgs.zero_grad()
                pred = model(x_lf, t_lf)
                l = data_loss_fn(pred, u_lf)
                l.backward()
                return l
            lbfgs.step(closure)

        with torch.no_grad():
            post_lbfgs_loss = data_loss_fn(model(x_lf, t_lf), u_lf).item()

        if post_lbfgs_loss > pre_lbfgs_loss * 1.5:
            model.load_state_dict(pre_lbfgs_state)
            if verbose:
                print(f"  L-BFGS rolled back (diverged)", flush=True)

    with torch.no_grad():
        final = data_loss_fn(model(x_lf, t_lf), u_lf).item()
    if verbose:
        print(f"  Phase 1 done in {time.time()-start:.1f}s  |  Final: {final:.6e}", flush=True)
    return final


def train_phase2(model, lf_data, config, device, verbose=True):
    """Phase 2: Physics-informed fine-tuning."""
    epochs = config.get('phase2_epochs', 20000)
    alpha = config['alpha']

    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = BurgersDataLoss(u_scale=u_scale)
    pde_loss_fn = BurgersPDELoss(
        alpha=alpha, nu=config['nu'],
        gl_h=config['gl_h'], gl_N_memory=config['gl_N_memory'],
    ).to(device)
    bc_loss_fn = BurgersBCLoss(L=config['L'])
    ic_loss_fn = BurgersICLoss(L=config['L'])

    sampler = BurgersCollocationSampler(
        N_col=config['N_collocation'],
        x_range=(0.0, config['L']),
        t_range=(0.01, config['T']),
        device=device,
    )

    t_bc = torch.linspace(0.01, config['T'], 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, config['L'], 50, device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get('phase2_lr', 1e-4),
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    w_data = config['w_data']
    w_pde = config['w_pde']
    w_bc = config['w_bc']
    w_ic = config['w_ic']
    ramp = config['pde_ramp_epochs']

    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Physics-Informed ({epochs} epochs)")
        print(f"  Weights: data={w_data}, pde={w_pde}, bc={w_bc}, ic={w_ic}")
        print(f"{'='*60}", flush=True)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        u_pred = model(x_lf, t_lf)
        l_data = data_loss_fn(u_pred, u_lf)

        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(model, x_col, t_col)

        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)

        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
        total = w_data * l_data + w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                  f"Total: {total.item():.4e}  Data: {l_data.item():.4e}  "
                  f"PDE: {l_pde.item():.4e}", flush=True)

    if verbose:
        print(f"  Phase 2 done in {time.time()-start:.1f}s", flush=True)
    return total.item()


def train_vanilla(model, config, device, verbose=True):
    """Vanilla fPINN: PDE + BC + IC only, no data."""
    epochs = config.get('vanilla_epochs', 25000)
    alpha = config['alpha']

    pde_loss_fn = BurgersPDELoss(
        alpha=alpha, nu=config['nu'],
        gl_h=config['gl_h'], gl_N_memory=config['gl_N_memory'],
    ).to(device)
    bc_loss_fn = BurgersBCLoss(L=config['L'])
    ic_loss_fn = BurgersICLoss(L=config['L'])

    sampler = BurgersCollocationSampler(
        N_col=config['N_collocation'],
        x_range=(0.0, config['L']),
        t_range=(0.01, config['T']),
        device=device,
    )

    t_bc = torch.linspace(0.01, config['T'], 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, config['L'], 50, device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    w_pde = config['w_pde']
    w_bc = config['w_bc']
    w_ic = config['w_ic']
    ramp = config['pde_ramp_epochs']

    if verbose:
        print(f"\n{'='*60}")
        print(f"VANILLA fPINN ({epochs} epochs)  |  NO data")
        print(f"  N_col={config['N_collocation']}, pde ramp={ramp}")
        print(f"{'='*60}", flush=True)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(model, x_col, t_col)
        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)

        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
        total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                  f"Total: {total.item():.4e}  PDE: {l_pde.item():.4e}  "
                  f"BC: {l_bc.item():.4e}  IC: {l_ic.item():.4e}", flush=True)

    if verbose:
        print(f"  Vanilla done in {time.time()-start:.1f}s", flush=True)
    return total.item()


def train_vanilla_with_data(model, lf_data, config, device, verbose=True):
    """Vanilla+N_LF control: same vanilla loss + extra data points as collocation."""
    epochs = config.get('vanilla_epochs', 25000)
    alpha = config['alpha']

    # Convert LF data to collocation-like tensors (just extra PDE residual points)
    x_extra = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_extra = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)

    pde_loss_fn = BurgersPDELoss(
        alpha=alpha, nu=config['nu'],
        gl_h=config['gl_h'], gl_N_memory=config['gl_N_memory'],
    ).to(device)
    bc_loss_fn = BurgersBCLoss(L=config['L'])
    ic_loss_fn = BurgersICLoss(L=config['L'])

    sampler = BurgersCollocationSampler(
        N_col=config['N_collocation'],
        x_range=(0.0, config['L']),
        t_range=(0.01, config['T']),
        device=device,
    )

    t_bc = torch.linspace(0.01, config['T'], 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.01, config['L'], 50, device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    w_pde = config['w_pde']
    w_bc = config['w_bc']
    w_ic = config['w_ic']
    ramp = config['pde_ramp_epochs']

    if verbose:
        print(f"\n{'='*60}")
        print(f"VANILLA+N_LF CONTROL ({epochs} epochs)")
        print(f"  N_col={config['N_collocation']} + {len(x_extra)} extra PDE pts")
        print(f"{'='*60}", flush=True)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        x_col, t_col = sampler.sample(epoch)
        # Concatenate extra points
        x_all = torch.cat([x_col, x_extra], dim=0)
        t_all = torch.cat([t_col, t_extra], dim=0)

        l_pde, _ = pde_loss_fn(model, x_all, t_all)
        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)

        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
        total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                  f"Total: {total.item():.4e}  PDE: {l_pde.item():.4e}", flush=True)

    if verbose:
        print(f"  Vanilla+N_LF done in {time.time()-start:.1f}s", flush=True)
    return total.item()


def run_single(method, config, lf_data, ref_data, seed, device, verbose=True):
    """Run one seed of one method."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config).to(device)

    t0 = time.time()
    if method == 'M3_mf':
        train_phase1(model, lf_data, config, device, verbose)
        train_phase2(model, lf_data, config, device, verbose)
    elif method == 'M1_vanilla':
        train_vanilla(model, config, device, verbose)
    elif method == 'vanilla_nlf':
        train_vanilla_with_data(model, lf_data, config, device, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
    wall = time.time() - t0

    metrics = evaluate_model(model, ref_data, EVAL_TIMES, device)
    metrics['seed'] = seed
    metrics['wall_time'] = wall
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Fractional Burgers: MF-fPINN vs Vanilla comparison'
    )
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-lf', type=int, default=50)
    parser.add_argument('--n-cols', type=int, nargs='+', default=[100])
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    verbose = not args.quiet
    alpha = args.alpha

    config = {**DEFAULT_CONFIG, 'alpha': alpha}

    print(f"\n{'#'*70}")
    print(f"  Fractional Burgers: MF-fPINN vs Vanilla")
    print(f"  alpha = {alpha}, nu = {config['nu']}, device = {device}, seeds = {args.seeds}")
    print(f"{'#'*70}\n")

    # ── 1. Generate HF reference ──────────────────────────────
    out_dir = Path('results/burgers')
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir / f'reference_alpha{alpha}.npz'
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference from {ref_path}")
    else:
        ref_data = generate_reference_burgers(
            alpha, nu=config['nu'], save_path=str(ref_path)
        )

    print(f"  Solution max: {ref_data['u'].max():.4f}")

    # ── 2. Quantify LF error ─────────────────────────────────
    lf_err = compute_lf_error_burgers(
        alpha, nu=config['nu'],
        lf_Nx=config['lf_Nx'], lf_Nt=config['lf_Nt'],
        hf_Nx=config['hf_Nx'], hf_Nt=config['hf_Nt'],
    )
    print(f"  LF error vs HF: {lf_err*100:.2f}%")

    # ── 3. Generate LF training data ─────────────────────────
    lf_data = generate_lf_data_burgers(
        alpha, nu=config['nu'], N_LF=args.n_lf,
        lf_Nx=config['lf_Nx'], lf_Nt=config['lf_Nt'],
    )

    # ── 4. Run comparison for each N_col ─────────────────────
    for n_col in args.n_cols:
        config['N_collocation'] = n_col

        print(f"\n{'#'*70}")
        print(f"  N_col = {n_col}")
        print(f"{'#'*70}")

        all_results = {}

        for method in ['M3_mf', 'M1_vanilla', 'vanilla_nlf']:
            print(f"\n{'='*60}")
            print(f"  Method: {method}  (N_col={n_col})")
            print(f"{'='*60}")

            method_results = []
            for seed in range(args.seeds):
                print(f"\n--- Seed {seed} ---")
                metrics = run_single(
                    method, config, lf_data, ref_data, seed, device, verbose
                )
                method_results.append(metrics)

                l2_vals = [v for k, v in metrics.items() if k.endswith('_l2')]
                avg = np.mean(l2_vals)
                print(f"  -> Avg L2: {avg:.4f}  |  Wall: {metrics['wall_time']:.1f}s")

            all_results[method] = method_results

        # ── 5. Summary ────────────────────────────────────────
        print(f"\n\n{'#'*70}")
        print(f"  RESULTS SUMMARY  (alpha={alpha}, N_col={n_col})")
        print(f"{'#'*70}")

        for method, results in all_results.items():
            print(f"\n  {method}:")
            for t_val in EVAL_TIMES:
                key = f't{t_val:.2f}_l2'
                vals = [r[key] for r in results]
                print(f"    t={t_val:.2f}:  L2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            walls = [r['wall_time'] for r in results]
            print(f"    Wall time: {np.mean(walls):.1f}s +/- {np.std(walls):.1f}s")

        # Aggregate: average across time snapshots per seed
        for method, results in all_results.items():
            for r in results:
                l2_vals = [v for k, v in r.items() if k.endswith('_l2')]
                r['avg_l2'] = np.mean(l2_vals)

        print(f"\n  Aggregate (avg over snapshots):")
        for method in all_results:
            vals = [r['avg_l2'] for r in all_results[method]]
            print(f"    {method}: {np.mean(vals)*100:.2f}% +/- {np.std(vals)*100:.2f}%")

        # MF vs Vanilla ratio
        mf_vals = [r['avg_l2'] for r in all_results['M3_mf']]
        van_vals = [r['avg_l2'] for r in all_results['M1_vanilla']]
        mf_mean = np.mean(mf_vals)
        van_mean = np.mean(van_vals)
        if mf_mean > 1e-8:
            ratio = van_mean / mf_mean
            print(f"\n  Advantage: Van/MF = {ratio:.2f}x")

        # Wilcoxon test if enough seeds
        if len(mf_vals) >= 5:
            try:
                stat, p = stats.wilcoxon(mf_vals, van_vals, alternative='two-sided')
                sig = '*' if p < 0.05 else 'n.s.'
                print(f"  Wilcoxon: p={p:.4f} {sig}")
            except Exception:
                pass

        # Save results
        save_results = {}
        for method, results in all_results.items():
            save_results[method] = []
            for r in results:
                save_results[method].append(
                    {k: v for k, v in r.items() if isinstance(v, (int, float))}
                )

        json_path = out_dir / f'results_alpha{alpha}_ncol{n_col}.json'
        with open(json_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Results saved to {json_path}")


if __name__ == '__main__':
    main()
