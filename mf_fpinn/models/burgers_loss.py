"""
Loss functions for the time-fractional Burgers PINN.

PDE residual: R = ∂_t^α u + u·∂u/∂x − ν·∂²u/∂x² = 0

For α = 1:  R = ∂u/∂t + u·u_x − ν·u_xx
For α < 1:  Caputo derivative via Gruenwald-Letnikov scheme.
"""

import math
import torch
import torch.nn as nn
import numpy as np


class BurgersDataLoss(nn.Module):
    """MSE between PINN predictions and reference data."""

    def __init__(self, u_scale=None):
        super().__init__()
        self.u_scale = u_scale

    def forward(self, u_pred, u_ref):
        mse = torch.mean((u_pred - u_ref) ** 2)
        if self.u_scale is not None and self.u_scale > 1e-8:
            mse = mse / (self.u_scale ** 2)
        return mse


class BurgersPDELoss(nn.Module):
    """
    PDE residual for ∂_t^α u + u·u_x = ν·u_xx.

    Parameters
    ----------
    alpha : float
        Fractional order.
    nu : float
        Viscosity coefficient.
    gl_h : float
        GL time step for fractional derivative.
    gl_N_memory : int
        GL memory length.
    """

    def __init__(self, alpha=1.0, nu=0.1,
                 gl_h=0.01, gl_N_memory=100):
        super().__init__()
        self.alpha = alpha
        self.nu = nu
        self.gl_h = gl_h
        self.gl_N_memory = gl_N_memory

        # Precompute GL weights
        if alpha < 0.999:
            weights = [1.0]
            for k in range(1, gl_N_memory + 1):
                w_k = (1.0 - (1.0 + alpha) / k) * weights[-1]
                weights.append(w_k)
            self.register_buffer('gl_weights',
                                 torch.tensor(weights, dtype=torch.float32))

    def _caputo_gl(self, model, x, t):
        """
        Compute ∂_t^α u via GL scheme with Caputo correction.

        GL approximates the RL derivative. For non-zero IC u(x,0) = sin(πx),
        the Caputo and RL derivatives differ by:
            D^α_C f(t) = D^α_RL f(t) - f(0) · t^{-α} / Γ(1-α)

        We compute the GL sum (≈ RL derivative) and subtract the correction.

        ∂_t^α u ≈ h^{−α} Σ_{k=0}^{N} w_k · u(x, t−k·h) - u(x,0)·t^{-α}/Γ(1-α)
        """
        h = self.gl_h
        N_mem = min(self.gl_N_memory, max(1, int(t.max().item() / h)))

        scale = 1.0 / (h ** self.alpha)
        N = x.shape[0]
        chunk_size = 20

        Df = torch.zeros(N, 1, device=x.device, dtype=x.dtype)

        for chunk_start in range(0, N_mem + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_mem + 1)
            chunk_len = chunk_end - chunk_start

            k_vals = torch.arange(chunk_start, chunk_end, device=x.device, dtype=x.dtype)
            t_shifts = t - k_vals.unsqueeze(0) * h  # (N, chunk_len)
            t_shifts = torch.clamp(t_shifts, min=0.0)

            x_batch = x.expand(-1, chunk_len).reshape(-1, 1)
            t_batch = t_shifts.reshape(-1, 1)

            u_batch = model(x_batch, t_batch).reshape(N, chunk_len)

            weights = self.gl_weights[chunk_start:chunk_end]
            Df = Df + (u_batch * weights.unsqueeze(0)).sum(dim=1, keepdim=True)

        rl_deriv = scale * Df

        # Caputo correction for non-zero IC: subtract u(x,0) * t^{-α} / Γ(1-α)
        # u(x,0) = sin(πx) by the hard BC ansatz
        if self.alpha < 1.0 - 1e-6:
            ic_val = torch.sin(np.pi * x)  # (N, 1)
            # Clamp t to avoid division by zero at t=0
            t_safe = torch.clamp(t, min=1e-6)
            correction = ic_val * t_safe.pow(-self.alpha) / math.gamma(1.0 - self.alpha)
            return rl_deriv - correction
        else:
            return rl_deriv

    def forward(self, model, x_col, t_col):
        """
        Compute PDE residual at collocation points.

        Returns
        -------
        loss : scalar
        residual_info : dict
        """
        x = x_col.requires_grad_(True)
        t = t_col.requires_grad_(True)

        u = model(x, t)

        # Spatial derivatives via autograd
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, grad_outputs=torch.ones_like(du_dx),
            create_graph=True, retain_graph=True
        )[0]

        if self.alpha >= 0.999:
            # Classical: ∂u/∂t + u·u_x − ν·u_xx = 0
            du_dt = torch.autograd.grad(
                u, t, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            R = du_dt + u * du_dx - self.nu * d2u_dx2
        else:
            # Fractional: ∂_t^α u + u·u_x − ν·u_xx = 0
            caputo_u = self._caputo_gl(model, x, t)
            R = caputo_u + u * du_dx - self.nu * d2u_dx2

        loss = torch.mean(R ** 2)
        return loss, {'R_pde': loss.item()}


class BurgersBCLoss(nn.Module):
    """
    Boundary condition residuals (monitoring only — hard BC enforces them).

    u(0,t) = 0,  u(L,t) = 0
    """

    def __init__(self, L=1.0):
        super().__init__()
        self.L = L

    def forward(self, model, t_bc):
        x_zero = torch.zeros_like(t_bc)
        x_L = torch.full_like(t_bc, self.L)

        u_0 = model(x_zero, t_bc)
        u_L = model(x_L, t_bc)

        loss = torch.mean(u_0 ** 2) + torch.mean(u_L ** 2)
        return loss


class BurgersICLoss(nn.Module):
    """
    Initial condition residual: u(x, 0) = sin(πx).
    Monitoring only since hard BC enforces this.
    """

    def __init__(self, L=1.0):
        super().__init__()
        self.L = L

    def forward(self, model, x_ic):
        t_zero = torch.zeros_like(x_ic)
        u_pred = model(x_ic, t_zero)
        u_exact = torch.sin(np.pi * x_ic / self.L)
        return torch.mean((u_pred - u_exact) ** 2)
