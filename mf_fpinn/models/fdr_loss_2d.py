"""
Loss functions for the 2D fractional diffusion-reaction PINN.

PDE residual: R = ∂ᵗᵅ u − D(∂²u/∂x² + ∂²u/∂y²) + κ u = 0
"""

import numpy as np
import torch
import torch.nn as nn


class FDRDataLoss2D(nn.Module):
    """MSE between PINN predictions and reference data."""

    def forward(self, u_pred, u_ref):
        return torch.mean((u_pred - u_ref) ** 2)


class FDRBCICLoss2D(nn.Module):
    """
    BC and IC soft penalty losses for 2D PINN (no hard BC).

    BCs:
        u(0, y, t) = g(t)     (non-homogeneous, x=0)
        u(Lx, y, t) = 0       (x=Lx)
        u(x, 0, t) = 0        (y=0)
        u(x, Ly, t) = 0       (y=Ly)
    IC:
        u(x, y, 0) = 0
    """

    def __init__(self, Lx=1.0, Ly=1.0, T=0.5, bc_type='pulse', bc_params=None,
                 N_bc=50):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.bc_type = bc_type
        self.bc_params = bc_params or {}
        self.N_bc = N_bc

        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5

    def _bc_function(self, t):
        """g(t) — temporal BC at x=0."""
        if self.bc_type == 'step':
            return torch.ones_like(t)
        elif self.bc_type == 'pulse':
            a = self.bc_params['a']
            omega = np.pi / a
            g = torch.sin(omega * t)
            mask = torch.sigmoid(100.0 * (a - t)) * torch.sigmoid(100.0 * t)
            return g * mask
        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

    def forward(self, model, device):
        """Sample boundary/IC points and compute MSE."""
        N = self.N_bc
        Lx, Ly, T = self.Lx, self.Ly, self.T
        losses = {}

        # x=0 boundary: u(0, y, t) = g(t)
        y_bc = torch.rand(N, 1, device=device) * Ly
        t_bc = torch.rand(N, 1, device=device) * T
        x_bc = torch.zeros(N, 1, device=device)
        u_pred = model(x_bc, y_bc, t_bc)
        g_t = self._bc_function(t_bc)
        losses['bc_x0'] = torch.mean((u_pred - g_t) ** 2)

        # x=Lx boundary: u(Lx, y, t) = 0
        x_bc = torch.full((N, 1), Lx, device=device)
        u_pred = model(x_bc, y_bc, t_bc)
        losses['bc_xL'] = torch.mean(u_pred ** 2)

        # y=0 boundary: u(x, 0, t) = 0
        x_bc = torch.rand(N, 1, device=device) * Lx
        t_bc = torch.rand(N, 1, device=device) * T
        y_bc = torch.zeros(N, 1, device=device)
        u_pred = model(x_bc, y_bc, t_bc)
        losses['bc_y0'] = torch.mean(u_pred ** 2)

        # y=Ly boundary: u(x, Ly, t) = 0
        y_bc = torch.full((N, 1), Ly, device=device)
        u_pred = model(x_bc, y_bc, t_bc)
        losses['bc_yL'] = torch.mean(u_pred ** 2)

        # IC: u(x, y, 0) = 0
        x_ic = torch.rand(N, 1, device=device) * Lx
        y_ic = torch.rand(N, 1, device=device) * Ly
        t_ic = torch.zeros(N, 1, device=device)
        u_pred = model(x_ic, y_ic, t_ic)
        losses['ic'] = torch.mean(u_pred ** 2)

        total = sum(losses.values())
        return total, losses


class FDRPDELoss2D(nn.Module):
    """
    2D PDE residual: ∂ᵗᵅ u = D(∂²u/∂x² + ∂²u/∂y²) − κ u.

    Parameters
    ----------
    alpha : float
        Fractional order.
    D : float
        Diffusion coefficient.
    kappa : float
        Reaction rate.
    gl_h : float
        GL time step.
    gl_N_memory : int
        GL memory length.
    """

    def __init__(self, alpha=1.0, D=1.0, kappa=1.0,
                 gl_h=0.01, gl_N_memory=100):
        super().__init__()
        self.alpha = alpha
        self.D = D
        self.kappa = kappa
        self.gl_h = gl_h
        self.gl_N_memory = gl_N_memory

        if alpha < 0.999:
            weights = [1.0]
            for k in range(1, gl_N_memory + 1):
                w_k = (1.0 - (1.0 + alpha) / k) * weights[-1]
                weights.append(w_k)
            self.register_buffer('gl_weights',
                                 torch.tensor(weights, dtype=torch.float32))

    def _caputo_gl(self, model, x, y, t):
        """GL approximation of ∂ᵗᵅ u for 2D model."""
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
            y_batch = y.expand(-1, chunk_len).reshape(-1, 1)
            t_batch = t_shifts.reshape(-1, 1)

            u_batch = model(x_batch, y_batch, t_batch).reshape(N, chunk_len)

            weights = self.gl_weights[chunk_start:chunk_end].to(x.device)
            Df = Df + (u_batch * weights.unsqueeze(0)).sum(dim=1, keepdim=True)

        return scale * Df

    def forward(self, model, x_col, y_col, t_col):
        """
        Compute 2D PDE residual at collocation points.

        Returns
        -------
        loss : scalar
        residual_info : dict
        """
        x = x_col.requires_grad_(True)
        y = y_col.requires_grad_(True)
        t = t_col.requires_grad_(True)

        u = model(x, y, t)

        # Spatial derivatives via autograd
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, grad_outputs=torch.ones_like(du_dx),
            create_graph=True, retain_graph=True
        )[0]

        du_dy = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        d2u_dy2 = torch.autograd.grad(
            du_dy, y, grad_outputs=torch.ones_like(du_dy),
            create_graph=True, retain_graph=True
        )[0]

        laplacian = d2u_dx2 + d2u_dy2

        if self.alpha >= 0.999:
            du_dt = torch.autograd.grad(
                u, t, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            R = du_dt - self.D * laplacian + self.kappa * u
        else:
            caputo_u = self._caputo_gl(model, x, y, t)
            R = caputo_u - self.D * laplacian + self.kappa * u

        loss = torch.mean(R ** 2)
        return loss, {'R_pde': loss.item()}
