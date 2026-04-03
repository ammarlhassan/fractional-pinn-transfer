"""
Finite-difference solver for the fractional diffusion-reaction equation.

    ∂_t^α u = D ∂²u/∂x² − κ u,   x ∈ (0, L), t ∈ (0, T)

    u(x, 0) = 0
    u(0, t) = g(t),   u(L, t) = 0

Caputo fractional derivative via L1 scheme:
    ∂_t^α u(x, t_n) ≈ σ Σ_{k=0}^{n-1} b_k [u^{n-k} - u^{n-k-1}]
    where σ = (Δt)^{-α}/Γ(2-α), b_k = (k+1)^{1-α} - k^{1-α}

Fully implicit time stepping: diffusion and reaction both implicit.

This provides a STRUCTURALLY DIFFERENT low-fidelity source compared to the
Laplace-domain solver, for heterogeneous multi-fidelity experiments.
"""

import numpy as np
from math import gamma
from scipy.interpolate import RegularGridInterpolator


def _thomas_solve(a, b, c, d):
    """Solve tridiagonal system Ax = d using Thomas algorithm."""
    n = len(b)
    c = c.copy()
    d = d.copy()
    b = b.copy()

    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x


def _bc_pulse(t, a=0.5):
    """g(t) = sin(πt/a) for t ∈ [0, a], 0 otherwise."""
    if t <= 0 or t > a:
        return 0.0
    return np.sin(np.pi * t / a)


def solve_fdr_fd(alpha, D=1.0, kappa=1.0, Nx=100, Nt=1000, L=1.0, T=1.0,
                 bc_type='pulse', bc_params=None):
    """
    Solve the time-fractional diffusion-reaction equation using L1 + fully implicit.

    Parameters
    ----------
    alpha : float
        Fractional order α ∈ (0, 1].
    D : float
        Diffusion coefficient.
    kappa : float
        Reaction rate.
    Nx : int
        Number of spatial intervals.
    Nt : int
        Number of time steps.
    L : float
        Domain length.
    T : float
        Final time.
    bc_type : str
        'pulse' only for now.
    bc_params : dict
        e.g. {'a': 0.5} for pulse BC.

    Returns
    -------
    x_grid : (Nx+1,)
    t_grid : (Nt+1,)
    u : (Nx+1, Nt+1)
    """
    bc_params = bc_params or {'a': 0.5}
    dx = L / Nx
    dt = T / Nt
    x_grid = np.linspace(0, L, Nx + 1)
    t_grid = np.linspace(0, T, Nt + 1)

    u = np.zeros((Nx + 1, Nt + 1))
    # IC: u(x, 0) = 0 — already initialized
    # BCs: u(0, t) = g(t),  u(L, t) = 0

    N_int = Nx - 1  # number of interior points
    r = D / dx**2

    if alpha >= 0.999:
        # Classical case: backward Euler (fully implicit)
        # (1 + 2r·dt + κ·dt) u_i^{n+1} - r·dt (u_{i-1}^{n+1} + u_{i+1}^{n+1}) = u_i^n + BC
        coeff = 1.0 + 2.0 * r * dt + kappa * dt
        diag_main = np.ones(N_int) * coeff
        diag_lower = np.ones(N_int - 1) * (-r * dt)
        diag_upper = np.ones(N_int - 1) * (-r * dt)

        for n in range(Nt):
            t_new = t_grid[n + 1]

            rhs = u[1:Nx, n].copy()

            # Left BC contribution: u(0, t_{n+1}) = g(t_{n+1})
            g_new = _bc_pulse(t_new, **bc_params) if bc_type == 'pulse' else 0.0
            rhs[0] += r * dt * g_new

            # Right BC: u(L, t_{n+1}) = 0 → no contribution

            u[1:Nx, n+1] = _thomas_solve(diag_lower.copy(), diag_main.copy(),
                                          diag_upper.copy(), rhs)
            u[0, n+1] = g_new
            u[Nx, n+1] = 0.0

    else:
        # Fractional case: L1 scheme for Caputo derivative
        # At time step n (computing u^{n+1}):
        #
        # σ Σ_{k=0}^{n} b_k [u^{n+1-k} - u^{n-k}] = D u_xx^{n+1} - κ u^{n+1}
        #
        # k=0 term: σ b_0 (u^{n+1} - u^n)
        # Rearrange:
        #   (σ b_0 + κ + 2r) u_i^{n+1} - r (u_{i-1}^{n+1} + u_{i+1}^{n+1})
        #       = σ b_0 u_i^n - σ Σ_{k=1}^{n} b_k [u_i^{n+1-k} - u_i^{n-k}] + BC contribution

        sigma = dt**(-alpha) / gamma(2 - alpha)

        # Precompute b_k coefficients
        b = np.zeros(Nt + 1)
        for k in range(Nt + 1):
            b[k] = (k + 1)**(1 - alpha) - k**(1 - alpha)

        for n in range(Nt):
            t_new = t_grid[n + 1]

            # History sum: Σ_{k=1}^{n} b_k [u^{n+1-k} - u^{n-k}]
            history = np.zeros(N_int)
            for k in range(1, n + 1):
                history += b[k] * (u[1:Nx, n + 1 - k] - u[1:Nx, n - k])

            # RHS
            rhs = sigma * b[0] * u[1:Nx, n] - sigma * history

            # BC contribution
            g_new = _bc_pulse(t_new, **bc_params) if bc_type == 'pulse' else 0.0
            rhs[0] += r * g_new  # left BC

            # Tridiagonal system
            coeff = sigma * b[0] + kappa + 2.0 * r
            diag_main = np.ones(N_int) * coeff
            diag_lower = np.ones(N_int - 1) * (-r)
            diag_upper = np.ones(N_int - 1) * (-r)

            u[1:Nx, n+1] = _thomas_solve(diag_lower, diag_main, diag_upper, rhs)
            u[0, n+1] = g_new
            u[Nx, n+1] = 0.0

    return x_grid, t_grid, u


def generate_lf_data_fd(alpha, D=1.0, kappa=1.0, L=1.0, T=1.0,
                        bc_type='pulse', bc_params=None,
                        N_LF=50, lf_Nx=20, lf_Nt=200,
                        seed=42, t_range=(0.01, 1.0)):
    """
    Generate low-fidelity data from a COARSE finite-difference solver.

    This provides a structurally different LF source: the FD solver uses
    L1 time discretization (not Laplace-domain inversion), so its errors
    are qualitatively different from a degraded Laplace solver.

    Parameters
    ----------
    alpha : float
    D, kappa, L, T : float
        Physical parameters.
    bc_type : str
    bc_params : dict
    N_LF : int
        Number of LF sample points.
    lf_Nx, lf_Nt : int
        COARSE grid resolution for LF solver.
    seed : int
    t_range : tuple

    Returns
    -------
    data : dict with keys 'x', 't', 'u'
    """
    bc_params = bc_params or {'a': 0.5}

    # Solve on coarse grid
    x_grid, t_grid, u_grid = solve_fdr_fd(
        alpha, D, kappa, lf_Nx, lf_Nt, L, T, bc_type, bc_params
    )

    # Build interpolator
    interp = RegularGridInterpolator(
        (x_grid, t_grid), u_grid, method='linear',
        bounds_error=False, fill_value=0.0
    )

    # Sample random points with Beta-biased distribution (same as Laplace LF)
    rng = np.random.default_rng(seed)
    x_pts = rng.beta(0.5, 2.0, N_LF) * L
    t_pts = t_range[0] + rng.beta(0.5, 2.0, N_LF) * (T - t_range[0])

    x_pts = np.clip(x_pts, 1e-6, L - 1e-6)
    t_pts = np.clip(t_pts, t_range[0], T - 1e-6)

    points = np.column_stack([x_pts, t_pts])
    u_pts = interp(points)

    return {'x': x_pts, 't': t_pts, 'u': u_pts}


def compute_fd_lf_error(alpha, D=1.0, kappa=1.0, L=1.0, T=1.0,
                        bc_type='pulse', bc_params=None,
                        lf_Nx=20, lf_Nt=200,
                        hf_Nx=200, hf_Nt=4000,
                        N_test=100, seed=123):
    """
    Quantify FD LF solver error vs FD HF reference.

    Uses a fine FD grid as HF reference to measure coarse grid degradation.
    """
    bc_params = bc_params or {'a': 0.5}

    # HF FD solution
    x_hf, t_hf, u_hf = solve_fdr_fd(alpha, D, kappa, hf_Nx, hf_Nt, L, T,
                                      bc_type, bc_params)
    interp_hf = RegularGridInterpolator(
        (x_hf, t_hf), u_hf, method='linear',
        bounds_error=False, fill_value=0.0
    )

    # LF FD solution
    x_lf, t_lf, u_lf = solve_fdr_fd(alpha, D, kappa, lf_Nx, lf_Nt, L, T,
                                      bc_type, bc_params)
    interp_lf = RegularGridInterpolator(
        (x_lf, t_lf), u_lf, method='linear',
        bounds_error=False, fill_value=0.0
    )

    # Test points
    rng = np.random.default_rng(seed)
    x_test = rng.uniform(1e-3, L - 1e-3, N_test)
    t_test = rng.uniform(0.01, T - 0.01, N_test)
    pts = np.column_stack([x_test, t_test])

    u_hf_test = interp_hf(pts)
    u_lf_test = interp_lf(pts)

    diff = u_hf_test - u_lf_test
    norm_hf = np.linalg.norm(u_hf_test)

    if norm_hf < 1e-12:
        return {'l2_relative': 0.0, 'linf_relative': 0.0}

    l2_rel = np.linalg.norm(diff) / norm_hf
    linf_rel = np.max(np.abs(diff)) / np.max(np.abs(u_hf_test))

    return {
        'l2_relative': l2_rel,
        'linf_relative': linf_rel,
        'lf_Nx': lf_Nx,
        'lf_Nt': lf_Nt,
    }
