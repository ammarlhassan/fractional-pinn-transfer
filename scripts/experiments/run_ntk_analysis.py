"""
NTK Eigenvalue Analysis for GL-Modified Fractional PDE Loss.

Computes the Neural Tangent Kernel matrix K_{ij} = <dR_i/dtheta, dR_j/dtheta>
for the GL-discretized PDE residual and analyzes how the condition number
and eigenvalue spectrum change with fractional order alpha.

Key hypothesis: GL memory enriches the NTK → lower condition number → better training.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mf_fpinn.models.fdr_pinn import FDRNet


def compute_gl_weights(alpha, N_mem):
    """Compute GL weights for fractional order alpha."""
    weights = [1.0]
    for k in range(1, N_mem + 1):
        w_k = weights[-1] * (1.0 - (1.0 + alpha) / k)
        weights.append(w_k)
    return np.array(weights)


def compute_pde_residual_jacobian(model, x_col, t_col, alpha, D=1.0, kappa=1.0,
                                   gl_h=0.01, gl_N_mem=100, L=1.0, a=0.5):
    """
    Compute the Jacobian dR_i/dtheta for each collocation point.
    R_i = GL_alpha(u)(x_i, t_i) - D*u_xx(x_i, t_i) + kappa*u(x_i, t_i)

    Returns: (N_col, N_params) Jacobian matrix
    """
    N_col = x_col.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    N_params = sum(p.numel() for p in params)

    jacobians = torch.zeros(N_col, N_params, device=x_col.device)

    for i in range(N_col):
        model.zero_grad()

        xi = x_col[i:i+1].detach().requires_grad_(True)
        ti = t_col[i:i+1].detach().requires_grad_(True)

        # Compute PDE residual at point i
        if alpha >= 0.999:
            # Classical: du/dt - D*d2u/dx2 + kappa*u
            u = model(xi, ti)
            du_dt = torch.autograd.grad(u, ti, create_graph=True, retain_graph=True)[0]
            du_dx = torch.autograd.grad(u, xi, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx, xi, create_graph=True, retain_graph=True)[0]
            residual = du_dt - D * d2u_dx2 + kappa * u
        else:
            # Fractional GL: h^{-alpha} * sum(w_k * u(x, t-k*h)) - D*u_xx + kappa*u
            h = gl_h
            t_val = ti.item()
            N_mem_actual = min(gl_N_mem, max(1, int(t_val / h)))
            weights = compute_gl_weights(alpha, N_mem_actual)
            scale = h ** (-alpha)

            # GL sum
            gl_sum = torch.zeros(1, 1, device=xi.device)
            for k in range(N_mem_actual + 1):
                t_shifted = max(0.0, t_val - k * h)
                t_k = torch.tensor([[t_shifted]], device=xi.device, dtype=xi.dtype)
                u_k = model(xi, t_k)
                gl_sum = gl_sum + weights[k] * u_k

            caputo_u = scale * gl_sum

            # Spatial derivatives
            u_at_point = model(xi, ti)
            du_dx = torch.autograd.grad(u_at_point, xi, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx, xi, create_graph=True, retain_graph=True)[0]

            residual = caputo_u - D * d2u_dx2 + kappa * u_at_point

        # Backprop to get dR_i/dtheta
        residual.backward()

        # Collect gradients
        idx = 0
        for p in params:
            if p.grad is not None:
                n = p.numel()
                jacobians[i, idx:idx+n] = p.grad.view(-1)
                idx += n

    return jacobians


def compute_ntk_matrix(jacobians):
    """Compute NTK matrix K = J @ J^T."""
    return jacobians @ jacobians.T


def analyze_eigenvalues(K):
    """Analyze eigenvalue spectrum of NTK matrix."""
    eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy()
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Filter positive eigenvalues
    pos_eigs = eigenvalues[eigenvalues > 1e-15]

    if len(pos_eigs) < 2:
        return {
            'eigenvalues': eigenvalues.tolist(),
            'condition_number': float('inf'),
            'lambda_max': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
            'lambda_min': 0,
            'effective_rank': 0,
            'trace': float(np.sum(eigenvalues)),
        }

    cond = pos_eigs[0] / pos_eigs[-1]

    # Effective rank (participation ratio)
    normalized = pos_eigs / np.sum(pos_eigs)
    entropy = -np.sum(normalized * np.log(normalized + 1e-30))
    eff_rank = np.exp(entropy)

    return {
        'eigenvalues': eigenvalues.tolist(),
        'condition_number': float(cond),
        'lambda_max': float(pos_eigs[0]),
        'lambda_min': float(pos_eigs[-1]),
        'effective_rank': float(eff_rank),
        'trace': float(np.sum(eigenvalues)),
        'n_positive': int(len(pos_eigs)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-col', type=int, default=30, help='Number of collocation points (keep small for NTK)')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--layers', type=int, default=3, help='Network layers (smaller for NTK tractability)')
    parser.add_argument('--neurons', type=int, default=32, help='Neurons per layer')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Fixed collocation points
    np.random.seed(42)
    x_np = np.random.beta(0.5, 2.0, size=args.n_col)
    t_np = 0.01 + 0.99 * np.random.beta(0.5, 2.0, size=args.n_col)
    x_col = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_col = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(1)

    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}

    for alpha in alphas:
        print(f'\n{"="*60}')
        print(f'  alpha = {alpha}')
        print(f'{"="*60}')

        seed_results = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create a small network (must be tractable for NTK)
            model = FDRNet(
                n_layers=args.layers,
                n_neurons=args.neurons,
                activation='tanh',
                fourier_features=16,
                fourier_sigma=5.0,
                L=1.0,
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if seed == 0:
                print(f'  Network: {args.layers} layers, {args.neurons} neurons, {n_params} params')
                print(f'  Collocation: {args.n_col} points')

            t_start = time.time()
            with torch.no_grad():
                pass  # ensure no lingering graphs

            # Compute Jacobian
            J = compute_pde_residual_jacobian(
                model, x_col, t_col, alpha,
                D=1.0, kappa=1.0, gl_h=0.01, gl_N_mem=100
            )

            # Compute NTK
            K = compute_ntk_matrix(J)

            # Analyze
            analysis = analyze_eigenvalues(K)
            elapsed = time.time() - t_start

            seed_results.append({
                'seed': seed,
                'condition_number': analysis['condition_number'],
                'lambda_max': analysis['lambda_max'],
                'lambda_min': analysis['lambda_min'],
                'effective_rank': analysis['effective_rank'],
                'trace': analysis['trace'],
                'n_positive': analysis.get('n_positive', 0),
                'wall_time': elapsed,
            })

            print(f'  Seed {seed}: cond={analysis["condition_number"]:.2e}, '
                  f'λ_max={analysis["lambda_max"]:.4e}, λ_min={analysis["lambda_min"]:.4e}, '
                  f'eff_rank={analysis["effective_rank"]:.1f}, '
                  f'time={elapsed:.1f}s')

        # Aggregate
        conds = [s['condition_number'] for s in seed_results]
        traces = [s['trace'] for s in seed_results]
        eff_ranks = [s['effective_rank'] for s in seed_results]
        lmax = [s['lambda_max'] for s in seed_results]
        lmin = [s['lambda_min'] for s in seed_results]

        results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'seeds': seed_results,
            'condition_number_mean': float(np.mean(conds)),
            'condition_number_std': float(np.std(conds)),
            'trace_mean': float(np.mean(traces)),
            'effective_rank_mean': float(np.mean(eff_ranks)),
            'lambda_max_mean': float(np.mean(lmax)),
            'lambda_min_mean': float(np.mean(lmin)),
        }

        print(f'\n  SUMMARY α={alpha}: cond={np.mean(conds):.2e}±{np.std(conds):.2e}, '
              f'trace={np.mean(traces):.2e}, eff_rank={np.mean(eff_ranks):.1f}')

    # Save results
    os.makedirs('results/ntk', exist_ok=True)
    outpath = f'results/ntk/ntk_analysis_ncol{args.n_col}.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {outpath}')

    # Summary table
    print(f'\n{"="*70}')
    print(f'{"alpha":>6} {"cond(K)":>14} {"λ_max":>12} {"λ_min":>12} {"eff_rank":>10} {"trace":>12}')
    print('-'*70)
    for alpha in alphas:
        r = results[f'alpha_{alpha}']
        print(f'{alpha:>6.1f} {r["condition_number_mean"]:>14.2e} '
              f'{r["lambda_max_mean"]:>12.4e} {r["lambda_min_mean"]:>12.4e} '
              f'{r["effective_rank_mean"]:>10.1f} {r["trace_mean"]:>12.4e}')


if __name__ == '__main__':
    main()
