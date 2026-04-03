"""
Loss functions for the fractional diffusion-reaction PINN.

PDE residual: R = ∂ᵗᵅ u − D ∂²u/∂x² + κ u = 0

For α = 1:  R = ∂u/∂t − D ∂²u/∂x² + κ u
For α < 1:  Caputo derivative via Grünwald-Letnikov scheme.
"""

import torch
import torch.nn as nn
import numpy as np


class FDRDataLoss(nn.Module):
    """MSE between PINN predictions and Laplace-domain reference data."""

    def __init__(self, u_scale=None):
        super().__init__()
        self.u_scale = u_scale

    def forward(self, u_pred, u_ref):
        mse = torch.mean((u_pred - u_ref) ** 2)
        if self.u_scale is not None and self.u_scale > 1e-8:
            mse = mse / (self.u_scale ** 2)
        return mse


class FDRPDELoss(nn.Module):
    """
    PDE residual for ∂ᵗᵅ u = D ∂²u/∂x² − κ u.

    Parameters
    ----------
    alpha : float
        Fractional order.
    D : float
        Diffusion coefficient.
    kappa : float
        Reaction rate.
    gl_h : float
        GL time step for fractional derivative.
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

        # Precompute GL weights (need gl_N_memory + 1 weights)
        if alpha < 0.999:
            weights = [1.0]
            for k in range(1, gl_N_memory + 1):
                w_k = (1.0 - (1.0 + alpha) / k) * weights[-1]
                weights.append(w_k)
            self.register_buffer('gl_weights',
                                 torch.tensor(weights, dtype=torch.float32))
        # Note: u(x, t-k*h) ≈ 0 for k*h > t due to hard BC u(x,0)=0,
        # so extra terms in the sum are harmless (contribute zero)

    def _caputo_gl(self, model, x, t):
        """
        Compute ∂ᵗᵅ u via GL approximation (chunked batched for GPU efficiency).

        Since u(x,0)=0 (enforced by hard BC), Caputo = Riemann-Liouville.
        ∂ᵗᵅ u ≈ h^{−α} Σ_{k=0}^{N} w_k · u(x, t−k·h)
        """
        h = self.gl_h
        N_mem = min(self.gl_N_memory, max(1, int(t.max().item() / h)))

        scale = 1.0 / (h ** self.alpha)
        N = x.shape[0]
        chunk_size = 20  # Process GL terms in chunks to avoid OOM

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

        return scale * Df

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
            # Classical: ∂u/∂t − D ∂²u/∂x² + κu = 0
            du_dt = torch.autograd.grad(
                u, t, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            R = du_dt - self.D * d2u_dx2 + self.kappa * u
        else:
            # Fractional: ∂ᵗᵅ u − D ∂²u/∂x² + κu = 0
            caputo_u = self._caputo_gl(model, x, t)
            R = caputo_u - self.D * d2u_dx2 + self.kappa * u

        loss = torch.mean(R ** 2)
        return loss, {'R_pde': loss.item()}


class DifferentiableFDRPDELoss(nn.Module):
    """
    PDE residual with differentiable GL weights — gradients flow through α.
    Used for inverse problems where α is a trainable parameter.

    Parameters
    ----------
    D : float
        Diffusion coefficient (fixed).
    gl_h : float
        GL time step.
    gl_N_memory : int
        GL memory length.
    """

    def __init__(self, D=1.0, gl_h=0.01, gl_N_memory=30):
        super().__init__()
        self.D = D
        self.gl_h = gl_h
        self.gl_N_memory = gl_N_memory

    def _compute_gl_weights(self, alpha, N_mem):
        """GL weights as differentiable function of alpha (tensor)."""
        weights = [torch.ones(1, device=alpha.device, dtype=alpha.dtype)]
        for k in range(1, N_mem + 1):
            w_k = (1.0 - (1.0 + alpha) / k) * weights[-1]
            weights.append(w_k)
        return weights

    def forward(self, model, x_col, t_col, alpha, kappa):
        """
        Parameters
        ----------
        model : nn.Module
        x_col, t_col : Tensor (N, 1)
        alpha : Tensor (scalar, requires_grad)
        kappa : Tensor (scalar, requires_grad)

        Returns
        -------
        loss, residual_info
        """
        x = x_col.requires_grad_(True)
        t = t_col.requires_grad_(True)

        u = model(x, t)

        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, grad_outputs=torch.ones_like(du_dx),
            create_graph=True, retain_graph=True
        )[0]

        # Differentiable Caputo via GL
        h = self.gl_h
        N_mem = min(self.gl_N_memory, max(1, int(t.max().item() / h)))
        gl_weights = self._compute_gl_weights(alpha, N_mem)

        # Scale-normalized residual to prevent α gradient pathology.
        # Original:  R = h^{-α} Σ w_k u_k  −  D u_xx  +  κ u
        #   → d(h^{-α})/dα = ln(1/h)·h^{-α} dominates, pushing α→0.
        # Fixed:     R = Σ w_k u_k  −  h^α (D u_xx − κ u)
        #   → dividing by h^{-α} removes the spurious scale gradient.
        log_h = torch.log(torch.tensor(h, device=alpha.device, dtype=alpha.dtype))
        h_alpha = torch.exp(alpha * log_h)  # h^α (small number)

        # Chunked batched GL evaluation for efficiency + memory safety
        N = x.shape[0]
        chunk_size = 20
        Df = torch.zeros(N, 1, device=x.device, dtype=x.dtype)

        for chunk_start in range(0, N_mem + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_mem + 1)
            chunk_len = chunk_end - chunk_start

            k_vals = torch.arange(chunk_start, chunk_end, device=x.device, dtype=x.dtype)
            t_shifts = t - k_vals.unsqueeze(0) * h
            t_shifts = torch.clamp(t_shifts, min=0.0)

            x_batch = x.expand(-1, chunk_len).reshape(-1, 1)
            t_batch = t_shifts.reshape(-1, 1)

            u_batch = model(x_batch, t_batch).reshape(N, chunk_len)

            chunk_weights = torch.stack(gl_weights[chunk_start:chunk_end]).squeeze()
            if chunk_weights.dim() == 0:
                chunk_weights = chunk_weights.unsqueeze(0)
            Df = Df + (u_batch * chunk_weights.unsqueeze(0)).sum(dim=1, keepdim=True)

        # Scale-normalized: Σw_k·u_k = h^α · (D·u_xx − κ·u)
        R = Df - h_alpha * (self.D * d2u_dx2 - kappa * u)
        loss = torch.mean(R ** 2)

        return loss, {'R_pde': loss.item()}


class FDRBCLoss(nn.Module):
    """
    Boundary condition residuals.

    x=0: u(0,t) = g(t)    (enforced by hard BC, monitor only)
    x=L: u(L,t) = 0       (enforced by hard BC, monitor only)
    """

    def __init__(self, L=1.0):
        super().__init__()
        self.L = L

    def forward(self, model, t_bc):
        device = t_bc.device
        x_zero = torch.zeros_like(t_bc)
        x_L = torch.full_like(t_bc, self.L)

        u_0 = model(x_zero, t_bc)
        u_L = model(x_L, t_bc)

        # g(t) is embedded in the model's hard BC
        # For monitoring: just check u(L,t) ≈ 0
        loss = torch.mean(u_L ** 2)
        return loss


class FDRICLoss(nn.Module):
    """Initial condition: u(x, 0) = 0."""

    def forward(self, model, x_ic):
        t_zero = torch.zeros_like(x_ic)
        u_0 = model(x_ic, t_zero)
        return torch.mean(u_0 ** 2)
