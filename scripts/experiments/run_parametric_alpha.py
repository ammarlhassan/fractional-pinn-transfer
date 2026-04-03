"""
Parametric α-PINN: Train a single network NN(x, t, α) → u that covers all
fractional orders simultaneously.

Key hypothesis: The GL memory mechanism predicts which α values benefit from
MF transfer. A parametric model validates this by showing accuracy vs α
follows the K_eff curve.

Two experiments:
1. Vanilla parametric fPINN: train on PDE loss for all α simultaneously
2. MF parametric fPINN: Phase 1 on LF data for all α, Phase 2 on PDE

Compare accuracy(α) curves between methods — does MF advantage follow K_eff?
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
from mf_fpinn.solvers.fdr_solver import FDRLaplaceSolver as FDRSolver


class ParametricAlphaNet(nn.Module):
    """Network that takes (x, t, alpha) as input and outputs u."""

    def __init__(self, n_layers=5, n_neurons=128, activation='tanh',
                 fourier_features=64, fourier_sigma=5.0, L=1.0, a=0.5):
        super().__init__()
        self.L = L
        self.a = a

        # Fourier features for (x, t, alpha) — 3D input
        self.n_ff = fourier_features
        self.register_buffer('B', torch.randn(3, fourier_features) * fourier_sigma)

        input_dim = 2 * fourier_features  # cos + sin
        act = {'tanh': nn.Tanh, 'gelu': nn.GELU, 'swish': nn.SiLU}[activation]

        layers = [nn.Linear(input_dim, n_neurons), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def _fourier_encode(self, x, t, alpha):
        """Encode (x, t, alpha) with Fourier features."""
        inp = torch.cat([x, t, alpha], dim=1)  # (N, 3)
        proj = inp @ self.B  # (N, n_ff)
        return torch.cat([torch.cos(2 * np.pi * proj),
                          torch.sin(2 * np.pi * proj)], dim=1)

    def _bc_function(self, t):
        """Pulse BC g(t) = sin(pi*t/a) for t in [0, a], else 0."""
        omega = np.pi / self.a
        mask = torch.sigmoid(100.0 * (self.a - t)) * torch.sigmoid(100.0 * t)
        return torch.sin(omega * t) * mask

    def forward(self, x, t, alpha):
        """Forward with hard BC enforcement."""
        enc = self._fourier_encode(x, t, alpha)
        nn_out = self.net(enc)
        g_t = self._bc_function(t)
        L = self.L
        u = g_t * (L - x) / L + x * (L - x) * t * nn_out
        return u


def compute_gl_weights(alpha_val, N_mem):
    """GL weights for a single alpha value."""
    w = [1.0]
    for k in range(1, N_mem + 1):
        w.append(w[-1] * (1.0 - (1.0 + alpha_val) / k))
    return w


def pde_residual_parametric(model, x, t, alpha, D=1.0, kappa=1.0,
                             gl_h=0.01, gl_N_mem=100):
    """
    Compute PDE residual for batch with mixed alpha values.
    Each point has its own alpha, so we group by unique alpha values.
    """
    unique_alphas = torch.unique(alpha)
    residuals = torch.zeros(x.shape[0], device=x.device)

    for a_val in unique_alphas:
        mask = (alpha == a_val).squeeze()
        xi = x[mask]
        ti = t[mask]
        ai = alpha[mask]

        a_float = a_val.item()

        if a_float >= 0.999:
            # Classical: autograd du/dt
            xi_r = xi.detach().requires_grad_(True)
            ti_r = ti.detach().requires_grad_(True)
            u = model(xi_r, ti_r, ai)
            du_dt = torch.autograd.grad(u.sum(), ti_r, create_graph=True)[0]
            du_dx = torch.autograd.grad(u.sum(), xi_r, create_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), xi_r, create_graph=True)[0]
            res = du_dt - D * d2u_dx2 + kappa * u
        else:
            # Fractional GL
            h = gl_h
            N_mem_actual = min(gl_N_mem, max(1, int(ti.max().item() / h)))
            weights = compute_gl_weights(a_float, N_mem_actual)
            scale = h ** (-a_float)

            # Chunked GL evaluation
            gl_sum = torch.zeros_like(xi)
            chunk_size = 20
            for c_start in range(0, N_mem_actual + 1, chunk_size):
                c_end = min(c_start + chunk_size, N_mem_actual + 1)
                for k in range(c_start, c_end):
                    t_shifted = torch.clamp(ti - k * h, min=0.0)
                    u_k = model(xi, t_shifted, ai)
                    gl_sum = gl_sum + weights[k] * u_k

            caputo_u = scale * gl_sum

            # Spatial derivatives
            xi_r = xi.detach().requires_grad_(True)
            u_at = model(xi_r, ti, ai)
            du_dx = torch.autograd.grad(u_at.sum(), xi_r, create_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), xi_r, create_graph=True)[0]

            res = caputo_u - D * d2u_dx2 + kappa * u_at

        residuals[mask] = res.squeeze(-1) if res.dim() > 1 else res

    return residuals


def generate_lf_data_multi_alpha(alphas, N_per_alpha=50, N_terms=12, L=1.0, T=1.0):
    """Generate LF data for multiple alpha values."""
    all_x, all_t, all_a, all_u = [], [], [], []

    for alpha in alphas:
        solver = FDRSolver(alpha=alpha, D=1.0, kappa=1.0, L=L,
                           N_terms=N_terms, precision=20)
        rng = np.random.RandomState(42)
        x_pts = rng.beta(0.5, 2.0, N_per_alpha) * L
        t_pts = 0.01 + 0.99 * rng.beta(0.5, 2.0, N_per_alpha) * T
        u_vals = np.array([solver.evaluate(xi, ti) for xi, ti in zip(x_pts, t_pts)])

        all_x.append(x_pts)
        all_t.append(t_pts)
        all_a.append(np.full(N_per_alpha, alpha))
        all_u.append(u_vals)

    return (np.concatenate(all_x), np.concatenate(all_t),
            np.concatenate(all_a), np.concatenate(all_u))


def evaluate_parametric(model, alphas, device, N_terms_hf=30, L=1.0, T=1.0):
    """Evaluate model at each alpha against HF reference."""
    results = {}
    x_eval = np.linspace(0.01, L - 0.01, 50)
    t_eval = [0.05, 0.15, 0.25, 0.40]

    for alpha in alphas:
        solver = FDRSolver(alpha=alpha, D=1.0, kappa=1.0, L=L,
                           N_terms=N_terms_hf, precision=30)
        errors = []
        for t_val in t_eval:
            ref = np.array([solver.evaluate(xi, t_val) for xi in x_eval])
            x_t = torch.tensor(x_eval, dtype=torch.float32, device=device).unsqueeze(1)
            t_t = torch.full_like(x_t, t_val)
            a_t = torch.full_like(x_t, alpha)
            with torch.no_grad():
                pred = model(x_t, t_t, a_t).cpu().numpy().flatten()
            l2 = np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-10)
            errors.append(l2)
        results[alpha] = {
            'l2_per_t': [float(e) * 100 for e in errors],
            'l2_mean': float(np.mean(errors)) * 100,
        }
    return results


def train_parametric(model, method, device, alphas, N_col=200, N_lf_per_alpha=50,
                     phase1_epochs=5000, phase2_epochs=20000, lf_N_terms=12,
                     verbose=True):
    """Train parametric alpha-PINN."""
    L, T = 1.0, 1.0

    # Generate LF data if MF
    if method == 'mf':
        x_lf, t_lf, a_lf, u_lf = generate_lf_data_multi_alpha(
            alphas, N_lf_per_alpha, lf_N_terms, L, T)
        x_lf_t = torch.tensor(x_lf, dtype=torch.float32, device=device).unsqueeze(1)
        t_lf_t = torch.tensor(t_lf, dtype=torch.float32, device=device).unsqueeze(1)
        a_lf_t = torch.tensor(a_lf, dtype=torch.float32, device=device).unsqueeze(1)
        u_lf_t = torch.tensor(u_lf, dtype=torch.float32, device=device).unsqueeze(1)

        if verbose:
            print(f'  LF data: {len(x_lf)} points across {len(alphas)} alphas')

    total_epochs = phase1_epochs + phase2_epochs

    if method == 'mf':
        # Phase 1: data pre-training
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, phase1_epochs, 1e-6)

        if verbose:
            print(f'\n  PHASE 1: Data pre-training ({phase1_epochs} epochs)')

        for epoch in range(phase1_epochs):
            optimizer.zero_grad()
            pred = model(x_lf_t, t_lf_t, a_lf_t)
            loss = torch.mean((pred - u_lf_t) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            if verbose and (epoch + 1) % 1000 == 0:
                print(f'    Epoch {epoch+1}/{phase1_epochs} | Loss: {loss.item():.4e}')

        # Phase 2: physics fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, phase2_epochs, 1e-6)
        start_epoch = phase1_epochs
    else:
        # Vanilla: all epochs on PDE
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, 1e-6)
        start_epoch = 0
        phase2_epochs = total_epochs

    if verbose:
        label = 'PHASE 2' if method == 'mf' else 'TRAINING'
        print(f'\n  {label}: Physics-informed ({phase2_epochs} epochs)')

    ramp_epochs = 3000

    for epoch in range(phase2_epochs):
        optimizer.zero_grad()

        # Sample collocation points with random alphas
        x_np = np.random.beta(0.5, 2.0, N_col) * L
        t_np = 0.01 + 0.99 * np.random.beta(0.5, 2.0, N_col) * T
        a_np = np.random.choice(alphas, N_col)

        x_c = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_c = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(1)
        a_c = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(1)

        # PDE residual
        res = pde_residual_parametric(model, x_c, t_c, a_c)
        pde_scale = min(1.0, (epoch + 1) / ramp_epochs)
        loss_pde = pde_scale * torch.mean(res ** 2)

        # Data loss (if MF)
        loss_data = torch.tensor(0.0, device=device)
        if method == 'mf':
            pred_lf = model(x_lf_t, t_lf_t, a_lf_t)
            loss_data = 10.0 * torch.mean((pred_lf - u_lf_t) ** 2)

        loss = loss_pde + loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 2000 == 0:
            print(f'    Epoch {epoch+1}/{phase2_epochs} | PDE: {loss_pde.item():.4e} '
                  f'| Data: {loss_data.item():.4e}')

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n-col', type=int, default=200)
    parser.add_argument('--lf-nterms', type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    alphas_train = [0.3, 0.5, 0.7, 0.9, 1.0]
    alphas_eval = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # include interpolation

    results = {'vanilla': {}, 'mf': {}}

    for method in ['vanilla', 'mf']:
        print(f'\n{"#"*60}')
        print(f'  Method: {method.upper()} Parametric α-PINN')
        print(f'{"#"*60}')

        all_seed_results = []

        for seed in range(args.seeds):
            print(f'\n--- Seed {seed} ---')
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ParametricAlphaNet(
                n_layers=5, n_neurons=128, activation='tanh',
                fourier_features=64, fourier_sigma=5.0
            ).to(device)

            t_start = time.time()
            model = train_parametric(
                model, method, device, alphas_train,
                N_col=args.n_col, lf_N_terms=args.lf_nterms,
            )
            wall_time = time.time() - t_start

            # Evaluate
            eval_results = evaluate_parametric(model, alphas_eval, device)
            eval_results['wall_time'] = wall_time
            eval_results['seed'] = seed
            all_seed_results.append(eval_results)

            print(f'\n  Results (seed {seed}):')
            for alpha in alphas_eval:
                print(f'    α={alpha}: L2={eval_results[alpha]["l2_mean"]:.2f}%')

        # Aggregate across seeds
        for alpha in alphas_eval:
            l2s = [r[alpha]['l2_mean'] for r in all_seed_results]
            results[method][str(alpha)] = {
                'l2_mean': float(np.mean(l2s)),
                'l2_std': float(np.std(l2s)),
                'l2_values': l2s,
            }

    # Summary
    print(f'\n{"="*70}')
    print(f'PARAMETRIC α-PINN SUMMARY ({args.seeds} seeds)')
    print(f'{"="*70}')
    print(f'{"alpha":>6} {"Vanilla":>15} {"MF":>15} {"Ratio":>8}')
    print('-'*50)
    for alpha in alphas_eval:
        v = results['vanilla'][str(alpha)]
        m = results['mf'][str(alpha)]
        ratio = v['l2_mean'] / m['l2_mean'] if m['l2_mean'] > 0 else 0
        print(f'{alpha:>6.1f} {v["l2_mean"]:>7.2f}±{v["l2_std"]:>5.2f}% '
              f'{m["l2_mean"]:>7.2f}±{m["l2_std"]:>5.2f}% {ratio:>7.2f}×')

    # Save
    os.makedirs('results/parametric', exist_ok=True)
    outpath = f'results/parametric/parametric_alpha_ncol{args.n_col}_nterms{args.lf_nterms}.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
