"""
2D Laplace-domain solver for the fractional diffusion-reaction equation.

    ∂ᵗᵅ u = D(∂²u/∂x² + ∂²u/∂y²) − κ u,   (x,y) ∈ (0,Lx)×(0,Ly), t > 0

BCs: u(0,y,t) = g(t)  (uniform in y),  u(Lx,y,t) = 0
     u(x,0,t) = 0,  u(x,Ly,t) = 0
IC:  u(x,y,0) = 0

The uniform y-BC requires Fourier sine expansion:
    ū(x,y,s) = Σ_{n odd} (4/(nπ)) · ḡ(s) · sinh(λn(Lx−x))/sinh(λn·Lx) · sin(nπy/Ly)
    λn = √((s^α + κ)/D + (nπ/Ly)²)

This is genuinely 2D — multiple modes needed, hard BC cannot capture all y-structure.
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, pi, re, sinh, sqrt as mpsqrt, sin as mpsin

from .fdr_solver import valsa_brancik_weights


class FDRLaplaceSolver2D:
    """
    2D fractional diffusion-reaction solver via inverse Laplace transform.

    Uses Fourier sine expansion for uniform y-BC.

    Parameters
    ----------
    alpha : float
        Fractional order α ∈ (0, 1].
    D, kappa : float
        Diffusion coefficient and reaction rate.
    Lx, Ly : float
        Domain dimensions.
    n_modes : int
        Number of Fourier modes for y-expansion (odd modes only).
    bc_type : str
        'pulse' or 'step' for the temporal part g(t).
    bc_params : dict
    N_terms : int
        Number of Laplace inversion terms.
    precision : int
        mpmath decimal digits.
    """

    def __init__(self, alpha=0.5, D=1.0, kappa=1.0, Lx=1.0, Ly=1.0,
                 n_modes=15, bc_type='pulse', bc_params=None,
                 N_terms=50, precision=50):
        self.alpha = mpf(alpha)
        self.D = mpf(D)
        self.kappa = mpf(kappa)
        self.Lx = mpf(Lx)
        self.Ly = mpf(Ly)
        self.n_modes = n_modes
        self.bc_type = bc_type
        self.bc_params = bc_params or {}
        self.N_terms = N_terms
        self.precision = precision
        mp.dps = precision

        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5

        self.abscissa, self.weights = valsa_brancik_weights(N_terms, precision)

        # Precompute Fourier coefficients: c_n = 4/(nπ) for odd n
        self.modes = []  # (n, c_n) for odd n
        for k in range(n_modes):
            n = 2 * k + 1  # odd: 1, 3, 5, ...
            c_n = mpf(4) / (n * pi)
            self.modes.append((n, c_n))

    def _bc_laplace(self, s):
        """Laplace transform of g(t)."""
        if self.bc_type == 'step':
            return mpf(1) / s
        elif self.bc_type == 'pulse':
            a = mpf(self.bc_params['a'])
            omega = pi / a
            return omega * (1 + exp(-a * s)) / (s ** 2 + omega ** 2)
        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

    def _laplace_domain_solution(self, s, x, y):
        """
        ū(x,y,s) = Σ_{n odd} c_n · ḡ(s) · sinh(λn(Lx−x))/sinh(λn·Lx) · sin(nπy/Ly)
        """
        x = mpf(x)
        y = mpf(y)
        G_s = self._bc_laplace(s)

        u_bar = mpf(0)
        for n, c_n in self.modes:
            mu_n = (n * pi / self.Ly) ** 2
            lam_sq = (s ** self.alpha + self.kappa) / self.D + mu_n
            lam = mpsqrt(lam_sq)

            lam_Lx = lam * self.Lx
            lam_Lx_mx = lam * (self.Lx - x)

            x_part = sinh(lam_Lx_mx) / sinh(lam_Lx)
            y_part = mpsin(n * pi * y / self.Ly)

            u_bar += c_n * G_s * x_part * y_part

        return u_bar

    def evaluate(self, x, y, t):
        """Compute u(x, y, t) via numerical inverse Laplace transform."""
        if t <= 0:
            return 0.0

        mp.dps = self.precision
        t_mp = mpf(t)
        N = self.N_terms

        u_sum = mpf(0)
        for k in range(2 * N + 1):
            s_k = (self.abscissa + mpc(0, k * pi)) / t_mp
            u_bar = self._laplace_domain_solution(s_k, x, y)

            w_k = self.weights[k]
            if k == 0:
                u_sum += w_k * re(u_bar)
            else:
                sign = (-1) ** k
                u_sum += w_k * sign * re(u_bar)

        scale = exp(self.abscissa) / t_mp
        return float(scale * u_sum)

    def evaluate_grid(self, x_array, y_array, t_array, verbose=True):
        """Evaluate u on a grid of (x, y) at given times."""
        nx, ny, nt = len(x_array), len(y_array), len(t_array)
        u_grid = np.zeros((nx, ny, nt))

        total = nx * ny * nt
        count = 0
        for i, xi in enumerate(x_array):
            for j, yj in enumerate(y_array):
                for k, tk in enumerate(t_array):
                    u_grid[i, j, k] = self.evaluate(xi, yj, tk)
                    count += 1
                    if verbose and count % max(1, total // 10) == 0:
                        print(f"  {count}/{total}")

        return u_grid


def generate_sparse_training_data_2d(solver, N_LF=100, seed=42):
    """
    Generate sparse 2D Laplace-domain evaluations for Phase 1.

    Returns
    -------
    data : dict with keys 'x', 'y', 't', 'u'
    """
    rng = np.random.default_rng(seed)

    Lx = float(solver.Lx)
    Ly = float(solver.Ly)

    # Beta-distributed sampling
    x_points = 1e-6 + (Lx - 1e-6) * rng.beta(0.5, 2.0, N_LF)
    y_points = 1e-6 + (Ly - 1e-6) * rng.beta(2.0, 2.0, N_LF)
    t_points = 0.01 + 0.49 * rng.beta(0.5, 2.0, N_LF)

    u_values = np.zeros(N_LF)
    print(f"Generating {N_LF} sparse 2D training points...")
    for i in range(N_LF):
        u_values[i] = solver.evaluate(x_points[i], y_points[i], t_points[i])
        if (i + 1) % max(1, N_LF // 5) == 0:
            print(f"  {i+1}/{N_LF}")

    return {
        'x': x_points.astype(np.float32),
        'y': y_points.astype(np.float32),
        't': t_points.astype(np.float32),
        'u': u_values.astype(np.float32),
    }


def generate_low_fidelity_data_2d(alpha, D, kappa, Lx, Ly, bc_type, bc_params,
                                   N_LF=100, seed=42,
                                   lf_N_terms=8, lf_precision=20, lf_n_modes=5):
    """
    Generate GENUINELY low-fidelity 2D training data using a degraded solver.

    Creates a separate FDRLaplaceSolver2D with fewer Laplace inversion terms,
    lower precision, AND fewer Fourier modes, producing data that is
    systematically less accurate than the HF reference.

    Parameters
    ----------
    alpha, D, kappa, Lx, Ly : float
        Physical parameters.
    bc_type : str
        Boundary condition type.
    bc_params : dict
    N_LF : int
        Number of LF data points.
    seed : int
        Random seed.
    lf_N_terms : int
        Laplace inversion terms for LF solver (HF default: 40).
    lf_precision : int
        mpmath precision for LF solver (HF default: 40).
    lf_n_modes : int
        Fourier modes for LF solver (HF default: 15).
        Fewer modes = worse y-structure approximation.

    Returns
    -------
    data : dict with keys 'x', 'y', 't', 'u'
    """
    lf_solver = FDRLaplaceSolver2D(
        alpha=alpha, D=D, kappa=kappa, Lx=Lx, Ly=Ly,
        n_modes=lf_n_modes,
        bc_type=bc_type, bc_params=bc_params,
        N_terms=lf_N_terms, precision=lf_precision,
    )

    rng = np.random.default_rng(seed)

    # Beta-distributed sampling (same as HF version)
    x_points = 1e-6 + (Lx - 1e-6) * rng.beta(0.5, 2.0, N_LF)
    y_points = 1e-6 + (Ly - 1e-6) * rng.beta(2.0, 2.0, N_LF)
    t_points = 0.01 + 0.49 * rng.beta(0.5, 2.0, N_LF)

    u_values = np.zeros(N_LF)
    print(f"Generating {N_LF} LOW-FIDELITY 2D points "
          f"(N_terms={lf_N_terms}, precision={lf_precision}, n_modes={lf_n_modes})...")
    for i in range(N_LF):
        u_values[i] = lf_solver.evaluate(x_points[i], y_points[i], t_points[i])
        if (i + 1) % max(1, N_LF // 5) == 0:
            print(f"  {i+1}/{N_LF}")

    return {
        'x': x_points.astype(np.float32),
        'y': y_points.astype(np.float32),
        't': t_points.astype(np.float32),
        'u': u_values.astype(np.float32),
    }


def compute_lf_error_2d(hf_solver, lf_N_terms=8, lf_precision=20, lf_n_modes=5,
                         N_test=50, seed=123):
    """
    Quantify degradation between HF and LF 2D solvers.

    Parameters
    ----------
    hf_solver : FDRLaplaceSolver2D
        High-fidelity solver.
    lf_N_terms, lf_precision, lf_n_modes : int
        LF solver settings.
    N_test : int
        Number of random test points.
    seed : int

    Returns
    -------
    dict with 'l2_relative', 'linf_relative', etc.
    """
    lf_solver = FDRLaplaceSolver2D(
        alpha=float(hf_solver.alpha),
        D=float(hf_solver.D),
        kappa=float(hf_solver.kappa),
        Lx=float(hf_solver.Lx),
        Ly=float(hf_solver.Ly),
        n_modes=lf_n_modes,
        bc_type=hf_solver.bc_type,
        bc_params=hf_solver.bc_params,
        N_terms=lf_N_terms,
        precision=lf_precision,
    )

    rng = np.random.default_rng(seed)
    Lx, Ly = float(hf_solver.Lx), float(hf_solver.Ly)

    x_pts = 1e-6 + (Lx - 1e-6) * rng.uniform(size=N_test)
    y_pts = 1e-6 + (Ly - 1e-6) * rng.uniform(size=N_test)
    t_pts = 0.01 + 0.49 * rng.uniform(size=N_test)

    u_hf = np.zeros(N_test)
    u_lf = np.zeros(N_test)

    print(f"Computing 2D LF degradation ({N_test} test points)...")
    for i in range(N_test):
        u_hf[i] = hf_solver.evaluate(x_pts[i], y_pts[i], t_pts[i])
        u_lf[i] = lf_solver.evaluate(x_pts[i], y_pts[i], t_pts[i])
        if (i + 1) % max(1, N_test // 5) == 0:
            print(f"  {i+1}/{N_test}")

    diff = u_lf - u_hf
    hf_norm = np.linalg.norm(u_hf)

    l2_rel = np.linalg.norm(diff) / max(hf_norm, 1e-15)
    linf_rel = np.max(np.abs(diff)) / max(np.max(np.abs(u_hf)), 1e-15)

    print(f"  2D LF degradation: L2_rel = {l2_rel*100:.2f}%, "
          f"Linf_rel = {linf_rel*100:.2f}%")

    return {
        'l2_relative': float(l2_rel),
        'linf_relative': float(linf_rel),
        'max_abs_error': float(np.max(np.abs(diff))),
        'mean_abs_error': float(np.mean(np.abs(diff))),
    }
