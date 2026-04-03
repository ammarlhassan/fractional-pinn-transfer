"""
Laplace-domain solver for the fractional diffusion-reaction equation.

    ∂ᵗᵅ u = D ∂²u/∂x² − κ u,   x ∈ (0, L), t > 0

BCs: u(0, t) = g(t),  u(L, t) = 0
IC:  u(x, 0) = 0

Laplace-domain closed-form:
    ū(x, s) = ḡ(s) · sinh(λ(L − x)) / sinh(λ L)
    λ = √((s^α + κ) / D)

Numerical inversion via Valsa-Brancik (Durbin-with-Euler form).
"""

import numpy as np
from mpmath import mp, mpf, mpc, exp, pi, re, sinh, sqrt as mpsqrt


def valsa_brancik_weights(N=50, precision=50):
    """Compute Valsa-Brancik / Durbin inversion weights."""
    mp.dps = precision
    a = mpf(6)
    weights = []
    for k in range(2 * N + 1):
        weights.append(mpf(0.5) if k == 0 else mpf(1.0))
    return a, weights


class FDRLaplaceSolver:
    """
    Numerical inverse Laplace transform solver for the fractional
    diffusion-reaction equation on [0, L].

    Parameters
    ----------
    alpha : float
        Fractional order α ∈ (0, 1].
    D : float
        Diffusion coefficient.
    kappa : float
        Reaction rate.
    L : float
        Domain length.
    bc_type : str
        'pulse' : g(t) = sin(πt/a)·H(t)·H(a−t)
        'step'  : g(t) = H(t)  (unit step)
        'ramp'  : g(t) = 1 − exp(−b·t)
    bc_params : dict
        Parameters for the BC type (e.g. {'a': 0.5} for pulse).
    N_terms : int
        Number of Laplace inversion terms.
    precision : int
        mpmath decimal digits.
    """

    def __init__(self, alpha=0.5, D=1.0, kappa=1.0, L=1.0,
                 bc_type='pulse', bc_params=None,
                 N_terms=50, precision=50):
        self.alpha = mpf(alpha)
        self.D = mpf(D)
        self.kappa = mpf(kappa)
        self.L = mpf(L)
        self.bc_type = bc_type
        self.bc_params = bc_params or {}
        self.N_terms = N_terms
        self.precision = precision
        mp.dps = precision

        # Default BC parameters
        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5
        if bc_type == 'ramp' and 'b' not in self.bc_params:
            self.bc_params['b'] = 10.0

        self.abscissa, self.weights = valsa_brancik_weights(N_terms, precision)

    def _bc_laplace(self, s):
        """Laplace transform of the boundary condition g(t)."""
        if self.bc_type == 'step':
            return mpf(1) / s

        elif self.bc_type == 'pulse':
            a = mpf(self.bc_params['a'])
            omega = pi / a
            return omega * (1 + exp(-a * s)) / (s ** 2 + omega ** 2)

        elif self.bc_type == 'ramp':
            b = mpf(self.bc_params['b'])
            return b / (s * (s + b))

        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

    def _laplace_domain_solution(self, s, x):
        """
        Compute ū(x, s) = ḡ(s) · sinh(λ(L−x)) / sinh(λL).

        λ = √((s^α + κ) / D)
        """
        x = mpf(x)
        G_s = self._bc_laplace(s)

        lam_sq = (s ** self.alpha + self.kappa) / self.D
        lam = mpsqrt(lam_sq)

        # Protect against overflow for large |λL|
        lam_L = lam * self.L
        lam_Lmx = lam * (self.L - x)

        u_bar = G_s * sinh(lam_Lmx) / sinh(lam_L)
        return u_bar

    def evaluate(self, x, t):
        """
        Compute u(x, t) via numerical inverse Laplace transform.

        Parameters
        ----------
        x : float
            Spatial coordinate, x ∈ [0, L].
        t : float
            Time, t > 0.

        Returns
        -------
        u : float
        """
        if t <= 0:
            return 0.0

        mp.dps = self.precision
        t_mp = mpf(t)
        N = self.N_terms

        u_sum = mpf(0)

        for k in range(2 * N + 1):
            s_k = (self.abscissa + mpc(0, k * pi)) / t_mp
            u_bar = self._laplace_domain_solution(s_k, x)

            w_k = self.weights[k]
            if k == 0:
                u_sum += w_k * re(u_bar)
            else:
                sign = (-1) ** k
                u_sum += w_k * sign * re(u_bar)

        scale = exp(self.abscissa) / t_mp
        return float(scale * u_sum)

    def evaluate_grid(self, x_array, t_array, verbose=True):
        """
        Evaluate u on a grid of (x, t) points.

        Returns
        -------
        u_grid : np.ndarray, shape (len(x_array), len(t_array))
        """
        nx, nt = len(x_array), len(t_array)
        u_grid = np.zeros((nx, nt))

        total = nx * nt
        count = 0
        for i, x in enumerate(x_array):
            for j, t in enumerate(t_array):
                u_grid[i, j] = self.evaluate(x, t)
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

        return u_grid

    def generate_reference_data(self, nx=100,
                                t_values=(0.1, 0.3, 0.5, 0.8),
                                x_range=None, save_path=None):
        """
        Generate dense reference solution for evaluation.

        Parameters
        ----------
        nx : int
            Number of spatial points.
        t_values : tuple
            Time snapshots.
        x_range : tuple, optional
            (x_min, x_max). Defaults to (eps, L).
        save_path : str, optional
            Save as .npz.

        Returns
        -------
        data : dict
        """
        if x_range is None:
            x_range = (1e-6, float(self.L))

        x_array = np.linspace(x_range[0], x_range[1], nx)
        t_array = np.array(t_values)

        print(f"Generating reference data: {nx} x × {len(t_array)} t")
        print(f"  α={float(self.alpha)}, D={float(self.D)}, "
              f"κ={float(self.kappa)}, L={float(self.L)}")

        u_grid = self.evaluate_grid(x_array, t_array)

        data = {
            'x': x_array,
            't': t_array,
            'u': u_grid,
            'alpha': float(self.alpha),
            'D': float(self.D),
            'kappa': float(self.kappa),
            'L': float(self.L),
            'bc_type': self.bc_type,
        }

        if save_path is not None:
            np.savez(save_path, **data)
            print(f"  Saved to {save_path}")

        return data


def generate_sparse_training_data(solver, N_LF=50, x_range=None,
                                  t_range=(0.01, 1.0), seed=42):
    """
    Generate sparse Laplace-domain evaluations for Phase 1 pre-training.

    Uses Beta-distributed sampling biased towards x=0 and t=0.

    NOTE: This uses the SAME solver passed in. For genuine multi-fidelity
    experiments, use generate_low_fidelity_data() which creates a degraded
    solver with fewer Laplace terms and lower precision.

    Parameters
    ----------
    solver : FDRLaplaceSolver
    N_LF : int
        Number of low-fidelity data points.
    x_range : tuple
    t_range : tuple
    seed : int

    Returns
    -------
    data : dict with keys 'x', 't', 'u'
    """
    if x_range is None:
        x_range = (1e-6, float(solver.L))

    rng = np.random.default_rng(seed)

    # Beta(0.5, 2) biased towards x=0 where gradients are steepest
    x_points = x_range[0] + (x_range[1] - x_range[0]) * rng.beta(0.5, 2.0, N_LF)
    t_points = t_range[0] + (t_range[1] - t_range[0]) * rng.beta(0.5, 2.0, N_LF)

    x_points = np.maximum(x_points, 1e-6)
    t_points = np.maximum(t_points, 1e-6)

    u_vals = np.zeros(N_LF)

    print(f"Generating {N_LF} sparse training points...")
    for i in range(N_LF):
        u_vals[i] = solver.evaluate(x_points[i], t_points[i])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_LF}")

    return {'x': x_points, 't': t_points, 'u': u_vals}


def generate_low_fidelity_data(alpha, D, kappa, L, bc_type, bc_params,
                               N_LF=50, x_range=None, t_range=(0.01, 1.0),
                               seed=42, lf_N_terms=5, lf_precision=15):
    """
    Generate GENUINELY low-fidelity training data using a degraded solver.

    Creates a separate FDRLaplaceSolver with fewer Laplace inversion terms
    and lower mpmath precision, producing data that is systematically less
    accurate than the high-fidelity reference. This is essential for a
    genuine multi-fidelity setup.

    Parameters
    ----------
    alpha : float
        Fractional order.
    D, kappa, L : float
        Physical parameters.
    bc_type : str
        Boundary condition type.
    bc_params : dict
        BC parameters.
    N_LF : int
        Number of low-fidelity data points.
    x_range : tuple, optional
        Spatial range; defaults to (1e-6, L).
    t_range : tuple
        Temporal range.
    seed : int
        Random seed for reproducible point locations.
    lf_N_terms : int
        Number of Laplace inversion terms for the LF solver (default 5).
        Compare to HF default of 30. Fewer terms = less accurate inversion.
    lf_precision : int
        mpmath decimal precision for the LF solver (default 15).
        Compare to HF default of 30. Lower precision = more numerical error.

    Returns
    -------
    data : dict with keys 'x', 't', 'u'
    """
    if x_range is None:
        x_range = (1e-6, float(L))

    # Create a DEGRADED solver (fewer terms, lower precision)
    lf_solver = FDRLaplaceSolver(
        alpha=alpha, D=D, kappa=kappa, L=L,
        bc_type=bc_type, bc_params=bc_params,
        N_terms=lf_N_terms, precision=lf_precision,
    )

    rng = np.random.default_rng(seed)

    # Same Beta-distributed sampling as generate_sparse_training_data
    x_points = x_range[0] + (x_range[1] - x_range[0]) * rng.beta(0.5, 2.0, N_LF)
    t_points = t_range[0] + (t_range[1] - t_range[0]) * rng.beta(0.5, 2.0, N_LF)

    x_points = np.maximum(x_points, 1e-6)
    t_points = np.maximum(t_points, 1e-6)

    u_vals = np.zeros(N_LF)

    print(f"Generating {N_LF} LOW-FIDELITY training points "
          f"(N_terms={lf_N_terms}, precision={lf_precision})...")
    for i in range(N_LF):
        u_vals[i] = lf_solver.evaluate(x_points[i], t_points[i])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_LF}")

    return {'x': x_points, 't': t_points, 'u': u_vals}


def compute_lf_error(hf_solver, lf_N_terms=5, lf_precision=15, N_test=50,
                     x_range=None, t_range=(0.01, 1.0), seed=123):
    """
    Quantify the degradation between HF and LF solvers.

    Evaluates both solvers at the same random test points and computes
    the relative L2 error. This number should be reported in the paper
    to characterize the fidelity gap.

    Parameters
    ----------
    hf_solver : FDRLaplaceSolver
        High-fidelity solver (reference).
    lf_N_terms : int
        Number of Laplace terms for the LF solver.
    lf_precision : int
        mpmath precision for the LF solver.
    N_test : int
        Number of random test points.
    x_range : tuple, optional
    t_range : tuple
    seed : int

    Returns
    -------
    results : dict
        'l2_relative' : float  — relative L2 error ||u_LF - u_HF|| / ||u_HF||
        'linf_relative' : float — relative Linf error
        'max_abs_error' : float
        'mean_abs_error' : float
        'u_hf' : np.ndarray
        'u_lf' : np.ndarray
    """
    if x_range is None:
        x_range = (1e-6, float(hf_solver.L))

    # Create degraded solver with same physics but fewer terms / lower precision
    lf_solver = FDRLaplaceSolver(
        alpha=float(hf_solver.alpha),
        D=float(hf_solver.D),
        kappa=float(hf_solver.kappa),
        L=float(hf_solver.L),
        bc_type=hf_solver.bc_type,
        bc_params=hf_solver.bc_params,
        N_terms=lf_N_terms,
        precision=lf_precision,
    )

    rng = np.random.default_rng(seed)
    x_pts = x_range[0] + (x_range[1] - x_range[0]) * rng.uniform(size=N_test)
    t_pts = t_range[0] + (t_range[1] - t_range[0]) * rng.uniform(size=N_test)

    u_hf = np.zeros(N_test)
    u_lf = np.zeros(N_test)

    print(f"Computing LF degradation error ({N_test} test points)...")
    for i in range(N_test):
        u_hf[i] = hf_solver.evaluate(x_pts[i], t_pts[i])
        u_lf[i] = lf_solver.evaluate(x_pts[i], t_pts[i])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_test}")

    diff = u_lf - u_hf
    hf_norm = np.linalg.norm(u_hf)

    l2_rel = np.linalg.norm(diff) / max(hf_norm, 1e-15)
    linf_rel = np.max(np.abs(diff)) / max(np.max(np.abs(u_hf)), 1e-15)

    results = {
        'l2_relative': float(l2_rel),
        'linf_relative': float(linf_rel),
        'max_abs_error': float(np.max(np.abs(diff))),
        'mean_abs_error': float(np.mean(np.abs(diff))),
        'u_hf': u_hf,
        'u_lf': u_lf,
    }

    print(f"  LF degradation: L2_rel = {l2_rel:.4f} ({l2_rel*100:.2f}%), "
          f"Linf_rel = {linf_rel:.4f} ({linf_rel*100:.2f}%)")

    return results
