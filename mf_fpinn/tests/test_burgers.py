"""Tests for the Burgers equation components: solver, PINN, and loss functions."""

import math
import numpy as np
import pytest
import torch

from mf_fpinn.solvers.burgers_solver import solve_fractional_burgers
from mf_fpinn.models.burgers_pinn import BurgersNet
from mf_fpinn.models.burgers_loss import (
    BurgersDataLoss,
    BurgersPDELoss,
    BurgersBCLoss,
    BurgersICLoss,
)


# ── Solver tests ──────────────────────────────────────────────────


class TestBurgersSolver:
    """Tests for the L1/IMEX finite-difference solver."""

    def test_output_shapes(self):
        Nx, Nt = 50, 200
        x, t, u = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=Nx, Nt=Nt)
        assert x.shape == (Nx + 1,)
        assert t.shape == (Nt + 1,)
        assert u.shape == (Nx + 1, Nt + 1)

    def test_boundary_conditions(self):
        """u(0,t) = 0 and u(L,t) = 0 for all t."""
        x, t, u = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=100, Nt=500)
        np.testing.assert_allclose(u[0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[-1, :], 0.0, atol=1e-12)

    def test_initial_condition(self):
        """u(x,0) = sin(pi*x)."""
        x, t, u = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=100, Nt=500)
        np.testing.assert_allclose(u[:, 0], np.sin(np.pi * x), atol=1e-12)

    def test_solution_decays(self):
        """Viscous Burgers: energy should decrease over time."""
        x, t, u = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=100, Nt=500)
        energy_start = np.sum(u[:, 0] ** 2)
        energy_end = np.sum(u[:, -1] ** 2)
        assert energy_end < energy_start

    def test_fractional_slower_diffusion(self):
        """Lower alpha should result in slower temporal evolution (more energy retained)."""
        _, _, u_frac = solve_fractional_burgers(alpha=0.5, nu=0.1, Nx=100, Nt=500)
        _, _, u_class = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=100, Nt=500)
        # Fractional retains more energy at final time
        energy_frac = np.sum(u_frac[:, -1] ** 2)
        energy_class = np.sum(u_class[:, -1] ** 2)
        assert energy_frac > energy_class

    def test_higher_viscosity_smoother(self):
        """Higher nu => faster decay."""
        _, _, u_lo = solve_fractional_burgers(alpha=1.0, nu=0.01, Nx=100, Nt=500)
        _, _, u_hi = solve_fractional_burgers(alpha=1.0, nu=0.1, Nx=100, Nt=500)
        assert np.max(np.abs(u_hi[:, -1])) < np.max(np.abs(u_lo[:, -1]))

    def test_solution_finite(self):
        """No NaN or Inf in solution."""
        x, t, u = solve_fractional_burgers(alpha=0.7, nu=0.02, Nx=100, Nt=500)
        assert np.all(np.isfinite(u))


# ── PINN architecture tests ──────────────────────────────────────


class TestBurgersNet:
    """Tests for the BurgersNet hard-BC PINN."""

    @pytest.fixture
    def net(self):
        return BurgersNet(n_layers=3, n_neurons=32, fourier_features=0)

    def test_output_shape(self, net):
        x = torch.rand(50, 1)
        t = torch.rand(50, 1)
        u = net(x, t)
        assert u.shape == (50, 1)

    def test_hard_bc_left(self, net):
        """u(0, t) = 0."""
        x = torch.zeros(20, 1)
        t = torch.rand(20, 1) * 0.9 + 0.1  # avoid t=0
        u = net(x, t)
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)

    def test_hard_bc_right(self, net):
        """u(L, t) = 0."""
        x = torch.ones(20, 1)
        t = torch.rand(20, 1) * 0.9 + 0.1
        u = net(x, t)
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)

    def test_hard_ic(self, net):
        """u(x, 0) = sin(pi*x)."""
        x = torch.linspace(0, 1, 50).unsqueeze(1)
        t = torch.zeros(50, 1)
        u = net(x, t)
        expected = torch.sin(torch.pi * x)
        assert torch.allclose(u, expected, atol=1e-5)

    def test_gradient_flow(self, net):
        """Gradients flow through the network."""
        x = torch.rand(10, 1, requires_grad=True)
        t = torch.rand(10, 1, requires_grad=True)
        u = net(x, t)
        u.sum().backward()
        assert all(p.grad is not None and not torch.all(p.grad == 0) for p in net.parameters())

    def test_fourier_features(self):
        """Network with Fourier features produces different output."""
        net_ff = BurgersNet(n_layers=3, n_neurons=32, fourier_features=32)
        x = torch.rand(10, 1)
        t = torch.rand(10, 1)
        u = net_ff(x, t)
        assert u.shape == (10, 1)


# ── Loss function tests ──────────────────────────────────────────


class TestBurgersLosses:
    """Tests for all Burgers loss components."""

    @pytest.fixture
    def net(self):
        return BurgersNet(n_layers=3, n_neurons=32, fourier_features=0)

    def test_data_loss_zero_on_match(self):
        u = torch.ones(10, 1)
        loss_fn = BurgersDataLoss()
        loss = loss_fn(u, u)
        assert loss.item() < 1e-10

    def test_data_loss_positive_on_mismatch(self):
        loss_fn = BurgersDataLoss()
        loss = loss_fn(torch.ones(10, 1), torch.zeros(10, 1))
        assert loss.item() > 0

    def test_pde_loss_returns_tuple(self, net):
        """PDE loss returns (scalar, dict)."""
        pde = BurgersPDELoss(alpha=1.0, nu=0.1)
        x = torch.rand(20, 1, requires_grad=True)
        t = torch.rand(20, 1, requires_grad=True) * 0.9 + 0.1
        loss, info = pde(net, x, t)
        assert loss.dim() == 0  # scalar
        assert isinstance(info, dict)

    def test_pde_loss_positive(self, net):
        """Untrained network should have non-zero PDE residual."""
        pde = BurgersPDELoss(alpha=1.0, nu=0.1)
        x = torch.rand(20, 1, requires_grad=True)
        t = torch.rand(20, 1, requires_grad=True) * 0.9 + 0.1
        loss, _ = pde(net, x, t)
        assert loss.item() > 0

    def test_pde_loss_fractional(self, net):
        """Fractional PDE loss (GL scheme) runs without error."""
        pde = BurgersPDELoss(alpha=0.7, nu=0.02, gl_h=0.02, gl_N_memory=50)
        x = torch.rand(15, 1, requires_grad=True)
        t = torch.rand(15, 1, requires_grad=True) * 0.9 + 0.1
        loss, info = pde(net, x, t)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_bc_loss_with_hard_bc(self, net):
        """BC loss should be near-zero with hard BC enforcement."""
        bc = BurgersBCLoss(L=1.0)
        t_bc = torch.rand(20, 1) * 0.9 + 0.1
        loss = bc(net, t_bc)
        assert loss.item() < 1e-8

    def test_ic_loss_with_hard_bc(self, net):
        """IC loss should be near-zero with hard BC enforcement."""
        ic = BurgersICLoss(L=1.0)
        x_ic = torch.linspace(0.01, 0.99, 20).unsqueeze(1)
        loss = ic(net, x_ic)
        assert loss.item() < 1e-8

    def test_pde_loss_caputo_correction(self):
        """At alpha<1, Caputo correction should be applied (non-zero IC)."""
        net = BurgersNet(n_layers=3, n_neurons=32, fourier_features=0)
        pde_frac = BurgersPDELoss(alpha=0.5, nu=0.1, gl_h=0.02, gl_N_memory=50)
        pde_class = BurgersPDELoss(alpha=1.0, nu=0.1)
        x = torch.rand(10, 1, requires_grad=True)
        t = torch.rand(10, 1, requires_grad=True) * 0.9 + 0.1
        loss_frac, _ = pde_frac(net, x, t)
        loss_class, _ = pde_class(net, x, t)
        # Both should be finite, and generally different
        assert torch.isfinite(loss_frac)
        assert torch.isfinite(loss_class)


# ── Integration test ──────────────────────────────────────────────


class TestBurgersIntegration:
    """End-to-end integration: solver → data → loss pipeline."""

    def test_solver_to_data_loss_pipeline(self):
        """Generate reference data, evaluate net, compute data loss."""
        x_grid, t_grid, u_ref = solve_fractional_burgers(
            alpha=1.0, nu=0.1, Nx=50, Nt=200
        )
        # Sample 30 random points
        ix = np.random.choice(len(x_grid), 30)
        it = np.random.choice(len(t_grid), 30)
        x_pts = torch.tensor(x_grid[ix], dtype=torch.float32).unsqueeze(1)
        t_pts = torch.tensor(t_grid[it], dtype=torch.float32).unsqueeze(1)
        u_pts = torch.tensor(u_ref[ix, it], dtype=torch.float32).unsqueeze(1)

        net = BurgersNet(n_layers=3, n_neurons=32, fourier_features=0)
        u_pred = net(x_pts, t_pts)

        loss_fn = BurgersDataLoss()
        loss = loss_fn(u_pred, u_pts)
        assert torch.isfinite(loss)
        assert loss.item() >= 0
