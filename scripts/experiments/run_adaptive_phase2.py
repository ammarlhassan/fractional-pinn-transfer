"""
Adaptive Phase 2 Scheduler: GL-aware training duration and learning rate.

Theory: K_eff(α) determines how much PDE gradient signal is available.
- Low K_eff (α≈1): Less PDE info per point → lower LR, risk of overshooting
- High K_eff (α≈0.5): Rich PDE signal → can use higher LR, more epochs

Adaptive rules:
1. Phase 2 LR ∝ K_eff(α) / K_eff(1) (scale relative to classical case)
2. Phase 2 epochs ∝ 1/K_eff(α) (more epochs when signal is weak) -- OR
   Phase 2 early-stop based on validation (LF data as proxy)
3. PDE ramp ∝ 1/K_eff(α) (slower ramp when signal is weak)

Compare: Fixed schedule vs. GL-adaptive schedule at each α.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))
from mf_fpinn.models.fdr_pinn import FDRNet
from mf_fpinn.models.fdr_loss import FDRPDELoss
from mf_fpinn.solvers.fdr_solver import FDRLaplaceSolver as FDRSolver


def compute_keff(alpha, threshold=1e-3, N_mem=100):
    """Compute K_eff for given alpha."""
    w = [1.0]
    for k in range(1, N_mem + 1):
        w.append(w[-1] * (1.0 - (1.0 + alpha) / k))
    return sum(1 for wi in w if abs(wi) > threshold)


def adaptive_schedule(alpha, base_lr=1e-4, base_epochs=20000, base_ramp=3000):
    """
    Compute adaptive Phase 2 hyperparameters based on K_eff(α).

    The key insight: K_eff controls the gradient signal quality.
    - More GL evaluations → richer gradient → can be more aggressive
    - Fewer GL evaluations → weaker signal → must be more cautious
    """
    keff = compute_keff(alpha)
    keff_1 = compute_keff(1.0)  # = 2

    # Ratio: how much more info we have compared to classical
    ratio = keff / keff_1  # e.g., 22 at α=0.5, 1 at α=1.0

    # LR: scale with sqrt of ratio (more info → can step faster, but not linearly)
    lr = base_lr * min(3.0, max(0.3, np.sqrt(ratio)))

    # Epochs: inversely proportional to sqrt(ratio)
    # More info → converge faster → need fewer epochs
    epochs = int(base_epochs / max(1, np.sqrt(ratio / 2)))
    epochs = max(5000, min(30000, epochs))  # clamp

    # Ramp: slower ramp when less PDE signal
    ramp = int(base_ramp / max(1, np.sqrt(ratio / 2)))
    ramp = max(1000, min(5000, ramp))

    return {
        'lr': lr,
        'epochs': epochs,
        'ramp': ramp,
        'keff': keff,
        'ratio': ratio,
    }


def train_mf_fpinn(model, pde_loss_fn, x_lf, t_lf, u_lf, device,
                    phase1_epochs=5000, phase2_lr=1e-4, phase2_epochs=20000,
                    ramp_epochs=3000, N_col=100, verbose=False):
    """Train MF-fPINN with given Phase 2 schedule."""
    # Phase 1: data pre-training
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, phase1_epochs, 1e-6)

    for ep in range(phase1_epochs):
        opt.zero_grad()
        pred = model(x_lf, t_lf)
        loss = torch.mean((pred - u_lf)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

    # L-BFGS polish
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=50, line_search_fn='strong_wolfe')
    def closure():
        lbfgs.zero_grad()
        pred = model(x_lf, t_lf)
        loss = torch.mean((pred - u_lf)**2)
        loss.backward()
        return loss
    try:
        lbfgs.step(closure)
    except:
        pass

    # Phase 2: physics fine-tuning with given schedule
    opt = torch.optim.Adam(model.parameters(), lr=phase2_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, phase2_epochs, 1e-6)

    # Track convergence
    checkpoints = {}
    eval_epochs = set([0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000])
    eval_epochs = eval_epochs.intersection(range(phase2_epochs + 1))

    for ep in range(phase2_epochs):
        opt.zero_grad()

        # Resample collocation
        if ep % 1000 == 0:
            x_c = torch.tensor(
                np.random.beta(0.5, 2.0, N_col),
                dtype=torch.float32, device=device
            ).unsqueeze(1)
            t_c = torch.tensor(
                0.01 + 0.99 * np.random.beta(0.5, 2.0, N_col),
                dtype=torch.float32, device=device
            ).unsqueeze(1)

        pde_scale = min(1.0, (ep + 1) / ramp_epochs)

        x_c_r = x_c.detach().requires_grad_(True)
        t_c_r = t_c.detach().requires_grad_(True)
        l_pde_raw, _ = pde_loss_fn(model, x_c_r, t_c_r)
        l_pde = pde_scale * l_pde_raw

        pred_lf = model(x_lf, t_lf)
        l_data = 10.0 * torch.mean((pred_lf - u_lf)**2)

        loss = l_pde + l_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

    return model


def evaluate_model(model, solver_hf, device, L=1.0):
    """Evaluate against HF reference."""
    x_eval = np.linspace(0.01, L - 0.01, 100)
    t_evals = [0.05, 0.15, 0.25, 0.40]
    errors = []
    for t_val in t_evals:
        ref = np.array([solver_hf.evaluate(xi, t_val) for xi in x_eval])
        xt = torch.tensor(x_eval, dtype=torch.float32, device=device).unsqueeze(1)
        tt = torch.full_like(xt, t_val)
        with torch.no_grad():
            pred = model(xt, tt).cpu().numpy().flatten()
        l2 = np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10)
        errors.append(l2)
    return float(np.mean(errors)) * 100


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=100)
    parser.add_argument('--n-terms', type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    alphas = [0.5, 0.7, 0.9, 1.0]
    N_LF = 50
    L = 1.0
    T = 1.0

    results = {}

    # Show adaptive schedules
    print('=== ADAPTIVE SCHEDULES ===')
    for alpha in alphas:
        sched = adaptive_schedule(alpha)
        print(f'α={alpha}: K_eff={sched["keff"]}, ratio={sched["ratio"]:.1f}×, '
              f'LR={sched["lr"]:.2e}, epochs={sched["epochs"]}, ramp={sched["ramp"]}')
    print()

    for alpha in alphas:
        print(f'\n{"#"*60}')
        print(f'  α = {alpha}')
        print(f'{"#"*60}')

        # Reference and LF data
        solver_hf = FDRSolver(alpha=alpha, D=1.0, kappa=1.0, L=L,
                              N_terms=30, precision=30)
        solver_lf = FDRSolver(alpha=alpha, D=1.0, kappa=1.0, L=L,
                              N_terms=args.n_terms, precision=20)

        rng = np.random.RandomState(42)
        x_lf_np = rng.beta(0.5, 2.0, N_LF) * L
        t_lf_np = 0.01 + 0.99 * rng.beta(0.5, 2.0, N_LF) * T
        u_lf_np = np.array([solver_lf.evaluate(xi, ti) for xi, ti in zip(x_lf_np, t_lf_np)])

        x_lf = torch.tensor(x_lf_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_lf = torch.tensor(t_lf_np, dtype=torch.float32, device=device).unsqueeze(1)
        u_lf = torch.tensor(u_lf_np, dtype=torch.float32, device=device).unsqueeze(1)

        pde_loss_fn = FDRPDELoss(alpha=alpha, D=1.0, kappa=1.0,
                                  gl_h=0.01, gl_N_memory=100).to(device)

        # Get schedules
        fixed_sched = {'lr': 1e-4, 'epochs': 20000, 'ramp': 3000}
        adapt_sched = adaptive_schedule(alpha)

        for sched_name, sched in [('fixed', fixed_sched), ('adaptive', adapt_sched)]:
            seed_l2s = []

            for seed in range(args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = FDRNet(
                    n_layers=5, n_neurons=128, activation='tanh',
                    fourier_features=64, fourier_sigma=5.0, L=L
                ).to(device)

                t_start = time.time()
                model = train_mf_fpinn(
                    model, pde_loss_fn, x_lf, t_lf, u_lf, device,
                    phase2_lr=sched['lr'],
                    phase2_epochs=sched['epochs'],
                    ramp_epochs=sched['ramp'],
                    N_col=args.n_col,
                )
                wall = time.time() - t_start

                l2 = evaluate_model(model, solver_hf, device)
                seed_l2s.append(l2)
                print(f'  {sched_name} seed {seed}: L2={l2:.2f}% ({wall:.0f}s)')

            results[f'alpha{alpha}_{sched_name}'] = {
                'alpha': alpha,
                'schedule': sched_name,
                'l2_mean': float(np.mean(seed_l2s)),
                'l2_std': float(np.std(seed_l2s)),
                'l2_values': seed_l2s,
                'config': {k: v for k, v in sched.items() if k != 'keff'},
            }

    # Summary
    print(f'\n{"="*70}')
    print(f'ADAPTIVE vs FIXED PHASE 2 SCHEDULE')
    print(f'{"="*70}')
    print(f'{"α":>5} {"Fixed":>15} {"Adaptive":>15} {"Improve":>10}')
    print('-'*50)
    for alpha in alphas:
        f = results[f'alpha{alpha}_fixed']
        a = results[f'alpha{alpha}_adaptive']
        improve = f['l2_mean'] / a['l2_mean'] if a['l2_mean'] > 0 else 0
        print(f'{alpha:>5.1f} {f["l2_mean"]:>7.2f}±{f["l2_std"]:>5.2f}% '
              f'{a["l2_mean"]:>7.2f}±{a["l2_std"]:>5.2f}% {improve:>9.2f}×')

    os.makedirs('results/adaptive', exist_ok=True)
    outpath = f'results/adaptive/adaptive_phase2_ncol{args.n_col}_nterms{args.n_terms}.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
