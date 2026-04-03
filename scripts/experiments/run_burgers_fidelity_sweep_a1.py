#!/usr/bin/env python3
"""
Burgers α=1.0 fidelity sweep: vary LF grid resolution to test whether
non-monotonic fidelity response (seen in FDR at α=1.0) transfers to Burgers.

LF resolutions → approximate L2 errors:
  Nx=4,  Nt=40:  ~36% (poor LF)
  Nx=8,  Nt=80:  ~25% (current default)
  Nx=12, Nt=120: ~15% (intermediate, analogous to FDR N_terms=10 peak)
  Nx=16, Nt=160: ~5%  (good)
  Nx=20, Nt=200: ~2%  (near-perfect)

Run:  python3 run_burgers_fidelity_sweep_a1.py --device cuda:3 --seeds 5
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from mf_fpinn.solvers.burgers_solver import (
    generate_reference_burgers,
    generate_lf_data_burgers,
    compute_lf_error_burgers,
)
from mf_fpinn.models.burgers_pinn import BurgersNet
from mf_fpinn.models.burgers_loss import (
    BurgersDataLoss, BurgersPDELoss, BurgersBCLoss, BurgersICLoss,
)
from mf_fpinn.evaluation.metrics import compute_l2_relative_error


ALPHA = 1.0
NU = 0.02
L, T = 1.0, 1.0
EVAL_TIMES = [0.1, 0.3, 0.5, 0.8]

# LF configurations to sweep (analogous to N_terms sweep in FDR)
LF_CONFIGS = [
    (4,  40),   # ~36% error
    (8,  80),   # ~25% error (current default)
    (12, 120),  # ~15% error
    (16, 160),  # ~5% error
    (20, 200),  # ~2% error
]

CONFIG = {
    'nu': NU, 'L': L, 'T': T,
    'n_layers': 5, 'n_neurons': 128, 'activation': 'tanh',
    'fourier_features': 64, 'fourier_sigma': 5.0,
    'lr': 5e-4, 'phase2_lr': 1e-4, 'weight_decay': 1e-5,
    'N_collocation': 100,
    'phase1_epochs': 5000, 'phase1_lbfgs': True, 'phase1_lbfgs_steps': 50,
    'phase2_epochs': 20000, 'vanilla_epochs': 25000,
    'w_data': 10.0, 'w_pde': 1.0, 'w_bc': 1.0, 'w_ic': 1.0,
    'pde_ramp_epochs': 3000, 'grad_clip': 5.0,
    'gl_h': 0.01, 'gl_N_memory': 100,
    'N_LF': 50, 'hf_Nx': 400, 'hf_Nt': 4000,
}


def build_model():
    return BurgersNet(
        n_layers=CONFIG['n_layers'], n_neurons=CONFIG['n_neurons'],
        activation=CONFIG['activation'], L=CONFIG['L'], T=CONFIG['T'],
        fourier_features=CONFIG['fourier_features'],
        fourier_sigma=CONFIG['fourier_sigma'],
    )


def evaluate_model(model, ref_data, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for t_val in EVAL_TIMES:
            t_idx = np.argmin(np.abs(ref_data['t'] - t_val))
            x_t = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, ref_data['t'][t_idx])
            u_pred = model(x_t, t_t).cpu().numpy().flatten()
            u_ref = ref_data['u'][:, t_idx]
            results[f't{t_val:.2f}'] = compute_l2_relative_error(u_pred, u_ref)
    return results


def train_one_seed(method, seed, lf_data, ref_data, device):
    """Train one seed. method: 'MF' or 'Vanilla'."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model().to(device)

    # Tensors
    x_lf = torch.tensor(lf_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    t_lf = torch.tensor(lf_data['t'], dtype=torch.float32, device=device).unsqueeze(1)
    u_lf = torch.tensor(lf_data['u'], dtype=torch.float32, device=device).unsqueeze(1)

    u_scale = max(u_lf.abs().max().item(), 0.01)
    data_loss_fn = BurgersDataLoss(u_scale=u_scale)
    pde_loss_fn = BurgersPDELoss(
        alpha=ALPHA, nu=NU,
        gl_h=CONFIG['gl_h'], gl_N_memory=CONFIG['gl_N_memory'],
    ).to(device)
    bc_loss_fn = BurgersBCLoss()
    ic_loss_fn = BurgersICLoss()

    from mf_fpinn.training.fdr_trainer import CollocationSampler
    sampler = CollocationSampler(
        N_col=CONFIG['N_collocation'],
        x_range=(0.0, L), t_range=(0.01, T), device=device,
    )

    t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
    x_ic = torch.linspace(0.0, L, 50, device=device).unsqueeze(1)

    t0 = time.time()

    if method == 'MF':
        # Phase 1: data pretraining
        opt1 = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'],
                                weight_decay=CONFIG['weight_decay'])
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=CONFIG['phase1_epochs'], eta_min=1e-6)
        model.train()
        for epoch in range(CONFIG['phase1_epochs']):
            opt1.zero_grad()
            loss = CONFIG['w_data'] * data_loss_fn(model(x_lf, t_lf), u_lf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            opt1.step()
            sched1.step()

        if CONFIG['phase1_lbfgs']:
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                       history_size=50, line_search_fn='strong_wolfe')
            for _ in range(CONFIG['phase1_lbfgs_steps']):
                def closure():
                    lbfgs.zero_grad()
                    l = CONFIG['w_data'] * data_loss_fn(model(x_lf, t_lf), u_lf)
                    l.backward()
                    return l
                lbfgs.step(closure)

        print(f"    Phase1 done ({time.time()-t0:.0f}s)", flush=True)
        phase2_epochs = CONFIG['phase2_epochs']
    else:
        phase2_epochs = CONFIG['vanilla_epochs']

    # Phase 2 (or full vanilla training)
    opt2 = torch.optim.Adam(model.parameters(), lr=CONFIG['phase2_lr'],
                             weight_decay=CONFIG['weight_decay'])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=phase2_epochs, eta_min=1e-6)
    ramp = CONFIG['pde_ramp_epochs']
    model.train()
    for epoch in range(phase2_epochs):
        opt2.zero_grad()
        x_col, t_col = sampler.sample(epoch)
        l_pde, _ = pde_loss_fn(model, x_col, t_col)
        l_bc = bc_loss_fn(model, t_bc)
        l_ic = ic_loss_fn(model, x_ic)
        pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
        total = pde_scale * l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        opt2.step()
        sched2.step()
        if (epoch + 1) % 5000 == 0:
            print(f"    Epoch {epoch+1}/{phase2_epochs} PDE:{l_pde.item():.3e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # L-BFGS polish
    lbfgs2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                 history_size=50, line_search_fn='strong_wolfe')
    for _ in range(50):
        def closure2():
            lbfgs2.zero_grad()
            x_col, t_col = sampler.sample(0)
            l_pde, _ = pde_loss_fn(model, x_col, t_col)
            l_bc = bc_loss_fn(model, t_bc)
            l_ic = ic_loss_fn(model, x_ic)
            l = l_pde + CONFIG['w_bc'] * l_bc + CONFIG['w_ic'] * l_ic
            l.backward()
            return l
        lbfgs2.step(closure2)

    wall = time.time() - t0
    metrics = evaluate_model(model, ref_data, device)
    avg_l2 = np.mean(list(metrics.values()))
    print(f"    Done: avg_L2={avg_l2*100:.2f}%  wall={wall:.0f}s", flush=True)
    return metrics, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path('results/burgers_fidelity')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Burgers α={ALPHA} Fidelity Sweep")
    print(f"  LF resolutions: {LF_CONFIGS}")
    print(f"  device={device}, seeds={args.seeds}")
    print(f"{'#'*70}\n")

    # Generate HF reference
    print("Generating HF reference...", flush=True)
    ref_path = Path('results/burgers_fidelity/reference_alpha1.0.npz')
    if ref_path.exists():
        ref_data = dict(np.load(ref_path, allow_pickle=True))
        print(f"  Loaded cached reference")
    else:
        ref_data = generate_reference_burgers(
            alpha=ALPHA, nu=NU, Nx=CONFIG['hf_Nx'], Nt=CONFIG['hf_Nt'],
            L=L, T=T,
        )
        np.savez(ref_path, **ref_data)
        print(f"  Saved reference to {ref_path}")

    # Also run Vanilla once per LF config (LF data doesn't affect Vanilla)
    # Actually Vanilla doesn't use LF data at all, so run it once
    # We'll use the same Vanilla results for all LF configs
    print("\n" + "="*60)
    print("Running Vanilla (shared across all LF configs)")
    print("="*60)
    vanilla_results = []
    for seed in range(args.seeds):
        print(f"\n  Vanilla seed {seed}", flush=True)
        # Use dummy lf_data (won't be used for Vanilla)
        dummy_lf = generate_lf_data_burgers(
            alpha=ALPHA, nu=NU, N_LF=CONFIG['N_LF'], seed=seed,
            lf_Nx=8, lf_Nt=80, L=L, T=T,
        )
        metrics, wall = train_one_seed('Vanilla', seed, dummy_lf, ref_data, device)
        vanilla_results.append({'metrics': metrics, 'wall_time': wall, 'seed': seed})

    vanilla_l2 = [np.mean(list(r['metrics'].values())) for r in vanilla_results]
    print(f"\nVanilla: {np.mean(vanilla_l2)*100:.2f}% ± {np.std(vanilla_l2, ddof=1)*100:.2f}%")

    # Sweep LF configurations
    all_results = {'vanilla': vanilla_results, 'mf': {}}
    lf_errors = {}

    for lf_nx, lf_nt in LF_CONFIGS:
        key = f'Nx{lf_nx}'
        print(f"\n{'='*60}")
        print(f"MF with LF Nx={lf_nx}, Nt={lf_nt}")
        print(f"{'='*60}")

        # Compute LF error
        lf_err = compute_lf_error_burgers(
            alpha=ALPHA, nu=NU, lf_Nx=lf_nx, lf_Nt=lf_nt,
            hf_Nx=CONFIG['hf_Nx'], hf_Nt=CONFIG['hf_Nt'], L=L, T=T,
        )
        lf_errors[key] = float(lf_err)
        print(f"  LF error: {lf_err*100:.1f}%", flush=True)

        mf_results = []
        for seed in range(args.seeds):
            print(f"\n  MF seed {seed}", flush=True)
            lf_data = generate_lf_data_burgers(
                alpha=ALPHA, nu=NU, N_LF=CONFIG['N_LF'], seed=seed,
                lf_Nx=lf_nx, lf_Nt=lf_nt, L=L, T=T,
            )
            metrics, wall = train_one_seed('MF', seed, lf_data, ref_data, device)
            mf_results.append({'metrics': metrics, 'wall_time': wall, 'seed': seed})

        all_results['mf'][key] = mf_results
        mf_l2 = [np.mean(list(r['metrics'].values())) for r in mf_results]
        ratio = np.mean(vanilla_l2) / np.mean(mf_l2)
        print(f"\n  MF: {np.mean(mf_l2)*100:.2f}% ± {np.std(mf_l2, ddof=1)*100:.2f}%  "
              f"Ratio: {ratio:.2f}x  (LF err: {lf_err*100:.1f}%)")

    # Save
    save_data = {
        'config': {'alpha': ALPHA, 'nu': NU, 'N_col': CONFIG['N_collocation'], 'seeds': args.seeds},
        'lf_errors': lf_errors,
        'vanilla_l2': [float(v) for v in vanilla_l2],
        'mf_by_lf': {key: [float(np.mean(list(r['metrics'].values())))
                            for r in all_results['mf'][key]]
                     for key in all_results['mf']},
    }
    out_path = out_dir / f'fidelity_sweep_alpha{ALPHA}_ncol{CONFIG["N_collocation"]}.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'#'*70}")
    print(f"  FIDELITY SWEEP SUMMARY (α={ALPHA})")
    print(f"  Vanilla: {np.mean(vanilla_l2)*100:.2f}% ± {np.std(vanilla_l2, ddof=1)*100:.2f}%")
    print(f"{'#'*70}")
    print(f"{'LF Nx':>8} {'LF err%':>10} {'MF L2%':>10} {'±':>8} {'Ratio':>8}")
    print('-'*50)
    for lf_nx, lf_nt in LF_CONFIGS:
        key = f'Nx{lf_nx}'
        mf_l2 = save_data['mf_by_lf'][key]
        le = lf_errors[key]
        ratio = np.mean(vanilla_l2) / np.mean(mf_l2)
        print(f"{lf_nx:>8} {le*100:>10.1f} {np.mean(mf_l2)*100:>10.2f} "
              f"{np.std(mf_l2, ddof=1)*100:>8.2f} {ratio:>8.2f}x")


if __name__ == '__main__':
    main()
