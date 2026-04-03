"""
Finite-difference solver for the time-fractional Burgers equation.

    ∂_t^α u + u ∂u/∂x = ν ∂²u/∂x²,   x ∈ (0,1), t ∈ (0,1)

    u(x, 0) = sin(πx)
    u(0, t) = 0,   u(1, t) = 0

Caputo fractional derivative via L1 scheme:
    ∂_t^α u(x, t_n) ≈ (Δt)^{-α}/Γ(2-α) Σ_{k=0}^{n-1} b_k [u^{n-k} - u^{n-k-1}]
    where b_k = (k+1)^{1-α} - k^{1-α}

IMEX time stepping: implicit backward Euler for diffusion, explicit for nonlinearity.
"""

import numpy as np
from math import gamma
from scipy.interpolate import RegularGridInterpolator


def solve_fractional_burgers(alpha, nu, Nx=400, Nt=4000, L=1.0, T=1.0):
    """
    Solve the time-fractional Burgers equation using L1 + IMEX.

    Parameters
    ----------
    alpha : float
        Fractional order α ∈ (0, 1].
    nu : float
        Viscosity coefficient.
    Nx : int
        Number of spatial intervals.
    Nt : int
        Number of time steps.
    L : float
        Domain length.
    T : float
        Final time.

    Returns
    -------
    x_grid : np.ndarray, shape (Nx+1,)
    t_grid : np.ndarray, shape (Nt+1,)
    u : np.ndarray, shape (Nx+1, Nt+1)
        Solution u(x_i, t_n).
    """
    dx = L / Nx
    dt = T / Nt
    x_grid = np.linspace(0, L, Nx + 1)
    t_grid = np.linspace(0, T, Nt + 1)

    # Solution array
    u = np.zeros((Nx + 1, Nt + 1))

    # Initial condition: u(x, 0) = sin(πx)
    u[:, 0] = np.sin(np.pi * x_grid)

    # BCs are u(0,t) = u(L,t) = 0, already zero-initialized

    if alpha >= 0.999:
        # Classical case: standard IMEX with backward Euler diffusion
        # u^{n+1} - u^n = dt * (nu * u_xx^{n+1} - u^n * u_x^n)
        r = nu * dt / dx**2

        # Tridiagonal coefficients for implicit diffusion (interior: 1..Nx-1)
        N_int = Nx - 1
        diag_main = np.ones(N_int) * (1.0 + 2.0 * r)
        diag_lower = np.ones(N_int - 1) * (-r)
        diag_upper = np.ones(N_int - 1) * (-r)

        # Thomas algorithm solver
        for n in range(Nt):
            u_curr = u[1:Nx, n]  # interior points at time n

            # Explicit nonlinear term: -u * u_x using central differences
            u_full = u[:, n]
            u_x = np.zeros(Nx + 1)
            u_x[1:Nx] = (u_full[2:Nx+1] - u_full[0:Nx-1]) / (2.0 * dx)

            nonlinear = -u_full[1:Nx] * u_x[1:Nx]

            # RHS: u^n + dt * nonlinear
            rhs = u_curr + dt * nonlinear

            # Solve tridiagonal system
            u[1:Nx, n+1] = _thomas_solve(diag_lower.copy(), diag_main.copy(),
                                          diag_upper.copy(), rhs)
            # BCs
            u[0, n+1] = 0.0
            u[Nx, n+1] = 0.0

    else:
        # Fractional case: L1 scheme for Caputo derivative
        # (Δt)^{-α}/Γ(2-α) Σ_{k=0}^{n-1} b_k [u^{n-k} - u^{n-k-1}]
        #   = ν u_xx^{n} (implicit on new time step) - u^n u_x^n (explicit)
        #
        # Rearrange: separate the k=0 term (which contains u^n):
        #   σ [b_0 u^{n+1} - b_0 u^n + Σ_{k=1}^{n-1} b_k (u^{n+1-k} - u^{n-k})]
        #     = ν u_xx^{n+1} - u^n u_x^n
        # Wait, let me be more careful. At time step n (computing u^{n+1}):
        #
        # The L1 approximation of ∂_t^α u at t_{n+1}:
        #   ∂_t^α u(t_{n+1}) ≈ σ Σ_{k=0}^{n} b_k [u^{n+1-k} - u^{n-k}]
        # where σ = (Δt)^{-α} / Γ(2-α)
        #
        # The k=0 term is: σ b_0 [u^{n+1} - u^n]
        # So: σ b_0 u^{n+1} + σ Σ_{k=1}^{n} b_k [u^{n+1-k} - u^{n-k}]
        #     = ν u_xx^{n+1} - u^n u_x^n
        #
        # => (σ b_0) u^{n+1} - ν u_xx^{n+1} = σ b_0 u^n
        #    - σ Σ_{k=1}^{n} b_k [u^{n+1-k} - u^{n-k}] - u^n u_x^n

        sigma = dt**(-alpha) / gamma(2 - alpha)

        # Precompute b_k coefficients (we need up to Nt of them)
        b = np.zeros(Nt + 1)
        for k in range(Nt + 1):
            b[k] = (k + 1)**(1 - alpha) - k**(1 - alpha)

        r = nu / dx**2
        N_int = Nx - 1

        for n in range(Nt):
            # We are computing u^{n+1} from u^{0..n}
            u_full = u[:, n]

            # Explicit nonlinear term at time n: -u * u_x
            u_x = np.zeros(Nx + 1)
            u_x[1:Nx] = (u_full[2:Nx+1] - u_full[0:Nx-1]) / (2.0 * dx)
            nonlinear = -u_full[1:Nx] * u_x[1:Nx]

            # History sum: Σ_{k=1}^{n} b_k [u^{n+1-k} - u^{n-k}]
            history = np.zeros(N_int)
            for k in range(1, n + 1):
                history += b[k] * (u[1:Nx, n + 1 - k] - u[1:Nx, n - k])

            # RHS for interior points
            rhs = sigma * b[0] * u[1:Nx, n] - sigma * history + nonlinear

            # Tridiagonal system: (σ b_0 + 2ν/dx²) u_i - ν/dx² (u_{i-1} + u_{i+1}) = rhs_i
            coeff = sigma * b[0]
            diag_main = np.ones(N_int) * (coeff + 2.0 * r)
            diag_lower = np.ones(N_int - 1) * (-r)
            diag_upper = np.ones(N_int - 1) * (-r)

            u[1:Nx, n+1] = _thomas_solve(diag_lower, diag_main, diag_upper, rhs)
            u[0, n+1] = 0.0
            u[Nx, n+1] = 0.0

    return x_grid, t_grid, u


def _thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system Ax = d using Thomas algorithm.

    Parameters
    ----------
    a : np.ndarray, shape (n-1,)
        Sub-diagonal.
    b : np.ndarray, shape (n,)
        Main diagonal.
    c : np.ndarray, shape (n-1,)
        Super-diagonal.
    d : np.ndarray, shape (n,)
        Right-hand side.

    Returns
    -------
    x : np.ndarray, shape (n,)
    """
    n = len(b)
    # Make copies to avoid modifying input
    c = c.copy()
    d = d.copy()
    b = b.copy()

    # Forward sweep
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x


def generate_reference_burgers(alpha, nu=0.1, Nx=400, Nt=4000, L=1.0, T=1.0,
                                save_path=None):
    """
    Generate high-fidelity reference solution for the fractional Burgers equation.

    Parameters
    ----------
    alpha : float
        Fractional order.
    nu : float
        Viscosity.
    Nx, Nt : int
        Grid resolution.
    L, T : float
        Domain and time extent.
    save_path : str, optional
        Path to save .npz file.

    Returns
    -------
    data : dict with keys 'x', 't', 'u', 'alpha', 'nu'
    """
    print(f"Generating HF reference: α={alpha}, ν={nu}, Nx={Nx}, Nt={Nt}")
    x_grid, t_grid, u = solve_fractional_burgers(alpha, nu, Nx, Nt, L, T)

    data = {
        'x': x_grid,
        't': t_grid,
        'u': u,
        'alpha': alpha,
        'nu': nu,
        'L': L,
        'T': T,
        'Nx': Nx,
        'Nt': Nt,
    }

    if save_path is not None:
        np.savez(save_path, **data)
        print(f"  Saved to {save_path}")

    print(f"  Solution shape: {u.shape}, max: {u.max():.4f}")
    return data


def generate_lf_data_burgers(alpha, nu=0.1, N_LF=50, seed=42,
                              lf_Nx=40, lf_Nt=400, L=1.0, T=1.0):
    """
    Generate low-fidelity data by solving on a coarse grid and interpolating.

    Parameters
    ----------
    alpha : float
        Fractional order.
    nu : float
        Viscosity.
    N_LF : int
        Number of low-fidelity sample points.
    seed : int
        Random seed.
    lf_Nx, lf_Nt : int
        Coarse grid resolution (10x coarser than HF default).
    L, T : float
        Domain extent.

    Returns
    -------
    data : dict with keys 'x', 't', 'u' (each shape (N_LF,))
    """
    print(f"Generating LF data: α={alpha}, ν={nu}, Nx={lf_Nx}, Nt={lf_Nt}, N_LF={N_LF}")

    # Solve on coarse grid
    x_lf, t_lf, u_lf = solve_fractional_burgers(alpha, nu, lf_Nx, lf_Nt, L, T)

    # Build interpolator from coarse solution
    interp = RegularGridInterpolator(
        (x_lf, t_lf), u_lf, method='linear', bounds_error=False, fill_value=0.0
    )

    # Sample random (x, t) points
    rng = np.random.default_rng(seed)
    # Beta-biased sampling (same pattern as FDR)
    x_pts = rng.beta(0.5, 2.0, N_LF) * L
    t_pts = 0.01 + rng.beta(0.5, 2.0, N_LF) * (T - 0.01)

    # Clip to domain
    x_pts = np.clip(x_pts, 1e-6, L - 1e-6)
    t_pts = np.clip(t_pts, 1e-6, T - 1e-6)

    # Interpolate LF solution at sampled points
    points = np.column_stack([x_pts, t_pts])
    u_pts = interp(points)

    print(f"  LF data: {N_LF} points, u range [{u_pts.min():.4f}, {u_pts.max():.4f}]")
    return {'x': x_pts, 't': t_pts, 'u': u_pts}


def compute_lf_error_burgers(alpha, nu=0.1, lf_Nx=40, lf_Nt=400,
                              hf_Nx=400, hf_Nt=4000, N_test=100,
                              L=1.0, T=1.0, seed=123):
    """
    Quantify LF solver error vs HF reference.

    Parameters
    ----------
    alpha : float
    nu : float
    lf_Nx, lf_Nt : int
        LF grid resolution.
    hf_Nx, hf_Nt : int
        HF grid resolution.
    N_test : int
        Number of random test points.
    L, T : float
    seed : int

    Returns
    -------
    l2_relative : float
        Relative L2 error between LF and HF solutions.
    """
    # Solve HF
    x_hf, t_hf, u_hf = solve_fractional_burgers(alpha, nu, hf_Nx, hf_Nt, L, T)
    interp_hf = RegularGridInterpolator(
        (x_hf, t_hf), u_hf, method='linear', bounds_error=False, fill_value=0.0
    )

    # Solve LF
    x_lf, t_lf, u_lf = solve_fractional_burgers(alpha, nu, lf_Nx, lf_Nt, L, T)
    interp_lf = RegularGridInterpolator(
        (x_lf, t_lf), u_lf, method='linear', bounds_error=False, fill_value=0.0
    )

    # Random test points
    rng = np.random.default_rng(seed)
    x_pts = rng.uniform(0.01, L - 0.01, N_test)
    t_pts = rng.uniform(0.01, T - 0.01, N_test)
    points = np.column_stack([x_pts, t_pts])

    u_hf_vals = interp_hf(points)
    u_lf_vals = interp_lf(points)

    diff = u_lf_vals - u_hf_vals
    hf_norm = np.linalg.norm(u_hf_vals)
    l2_rel = np.linalg.norm(diff) / max(hf_norm, 1e-15)

    return l2_rel
