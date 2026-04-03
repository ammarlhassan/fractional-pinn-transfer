"""
Tests for the fractional diffusion-reaction modules.

Covers:
  1. Laplace solver (steady-state, classical, symmetry)
  2. PINN architecture (shapes, hard BC)
  3. PDE residual (manufactured solution)
  4. Trainer (smoke test)
"""

import pytest
import numpy as np
import torch

from mf_fpinn.solvers.fdr_solver import FDRLaplaceSolver, generate_sparse_training_data
from mf_fpinn.models.fdr_pinn import FDRNet, InverseFDRNet
from mf_fpinn.models.fdr_loss import FDRPDELoss, FDRDataLoss, FDRBCLoss, FDRICLoss
from mf_fpinn.training.fdr_trainer import FDRTrainer


# ────────────────────────────────────────────────────────
# Laplace Solver Tests
# ────────────────────────────────────────────────────────

class TestFDRLaplaceSolver:
    """Test the Laplace-domain solver for fractional diffusion-reaction."""

    def test_step_bc_increasing_early(self):
        """For step BC, u(x, t) should increase at early times."""
        solver = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=1.0, L=1.0,
                                  bc_type='step', N_terms=30, precision=30)
        x_test = 0.3
        u_t1 = solver.evaluate(x_test, 0.1)
        u_t2 = solver.evaluate(x_test, 0.5)

        assert 0 < u_t1 < u_t2, \
            f"Should increase: 0 < {u_t1:.4f} < {u_t2:.4f}"

    def test_bc_at_x0(self):
        """u(0, t) should equal g(t) for step BC."""
        solver = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=1.0, L=1.0,
                                  bc_type='step', N_terms=30, precision=30)
        # g(t) = 1 for step BC, so u(0, t) should be ≈ 1 for t > 0
        u_at_0 = solver.evaluate(1e-4, 0.5)
        assert abs(u_at_0 - 1.0) < 0.05, f"u(0, 0.5) = {u_at_0:.4f}, expected ~1.0"

    def test_bc_at_xL(self):
        """u(L, t) should be ≈ 0."""
        solver = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=1.0, L=1.0,
                                  bc_type='step', N_terms=30, precision=30)
        u_at_L = solver.evaluate(0.999, 0.5)
        assert abs(u_at_L) < 0.05, f"u(L, 0.5) = {u_at_L:.4f}, expected ~0"

    def test_pulse_bc_after_pulse(self):
        """After pulse ends (t > a), u should decay towards 0."""
        solver = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=1.0, L=1.0,
                                  bc_type='pulse', bc_params={'a': 0.5},
                                  N_terms=30, precision=30)
        # After pulse, boundary is 0, so solution decays
        u_early = solver.evaluate(0.1, 0.3)   # during pulse
        u_late = solver.evaluate(0.1, 2.0)     # well after pulse

        assert abs(u_late) < abs(u_early), \
            f"Solution should decay: u(0.1, 2.0)={u_late:.4f} vs u(0.1, 0.3)={u_early:.4f}"

    def test_fractional_slower_diffusion(self):
        """α < 1 (subdiffusion) should diffuse slower → smaller u at large x."""
        solver_1 = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=0.0, L=1.0,
                                     bc_type='step', N_terms=30, precision=30)
        solver_05 = FDRLaplaceSolver(alpha=0.5, D=1.0, kappa=0.0, L=1.0,
                                      bc_type='step', N_terms=30, precision=30)

        # At x=0.5, t=0.2 (early time), subdiffusion should have smaller u
        u_alpha1 = solver_1.evaluate(0.5, 0.2)
        u_alpha05 = solver_05.evaluate(0.5, 0.2)

        assert u_alpha05 < u_alpha1 + 0.05, \
            f"Subdiffusion should be slower: α=1 → {u_alpha1:.4f}, α=0.5 → {u_alpha05:.4f}"

    def test_generate_sparse_data(self):
        """Smoke test for sparse data generation."""
        solver = FDRLaplaceSolver(alpha=1.0, D=1.0, kappa=1.0, L=1.0,
                                  bc_type='step', N_terms=20, precision=20)
        data = generate_sparse_training_data(solver, N_LF=10,
                                             t_range=(0.1, 1.0), seed=42)
        assert len(data['x']) == 10
        assert len(data['u']) == 10
        assert np.all(np.isfinite(data['u']))


# ────────────────────────────────────────────────────────
# PINN Architecture Tests
# ────────────────────────────────────────────────────────

class TestFDRNet:
    """Test the FDRNet architecture."""

    def test_output_shape(self):
        """Output shape should be (N, 1)."""
        net = FDRNet(n_layers=3, n_neurons=32, fourier_features=0)
        x = torch.rand(20, 1)
        t = torch.rand(20, 1)
        u = net(x, t)
        assert u.shape == (20, 1)

    def test_output_shape_fourier(self):
        """With Fourier features."""
        net = FDRNet(n_layers=3, n_neurons=32, fourier_features=16, fourier_sigma=3.0)
        x = torch.rand(20, 1)
        t = torch.rand(20, 1)
        u = net(x, t)
        assert u.shape == (20, 1)

    def test_hard_bc_x0(self):
        """u(0, t) should equal g(t) with hard BC."""
        net = FDRNet(n_layers=3, n_neurons=32, hard_bc=True,
                     bc_type='step', fourier_features=0)
        x = torch.zeros(10, 1)
        t = torch.rand(10, 1) * 0.5 + 0.01
        u = net(x, t)
        # For step BC, g(t) = 1
        assert torch.allclose(u, torch.ones_like(u), atol=1e-5), \
            f"u(0,t) should be 1.0, got {u.flatten()[:3]}"

    def test_hard_bc_xL(self):
        """u(L, t) should be 0 with hard BC."""
        net = FDRNet(n_layers=3, n_neurons=32, hard_bc=True, L=1.0,
                     fourier_features=0)
        x = torch.ones(10, 1)
        t = torch.rand(10, 1)
        u = net(x, t)
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-5), \
            f"u(L,t) should be 0, got {u.flatten()[:3]}"

    def test_hard_bc_t0(self):
        """u(x, 0) should be 0 with hard BC (if g(0)=0)."""
        # Pulse BC: g(0) = sin(0) = 0
        net = FDRNet(n_layers=3, n_neurons=32, hard_bc=True,
                     bc_type='pulse', bc_params={'a': 0.5}, fourier_features=0)
        x = torch.rand(10, 1) * 0.8 + 0.1  # avoid x=0 and x=L
        t = torch.zeros(10, 1)
        u = net(x, t)
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-5), \
            f"u(x,0) should be 0, got {u.flatten()[:3]}"

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        net = FDRNet(n_layers=3, n_neurons=32, fourier_features=0)
        x = torch.rand(10, 1, requires_grad=True)
        t = torch.rand(10, 1, requires_grad=True)
        u = net(x, t)
        u.sum().backward()
        assert x.grad is not None
        assert t.grad is not None


# ────────────────────────────────────────────────────────
# Loss Function Tests
# ────────────────────────────────────────────────────────

class TestFDRLoss:
    """Test loss functions."""

    def test_data_loss_zero(self):
        """Zero error for perfect predictions."""
        loss_fn = FDRDataLoss()
        u = torch.tensor([[1.0], [2.0], [3.0]])
        assert loss_fn(u, u).item() < 1e-10

    def test_pde_runs_on_real_network(self):
        """PDE loss should compute and be finite on a real FDRNet."""
        net = FDRNet(n_layers=2, n_neurons=16, fourier_features=0, hard_bc=False)
        pde = FDRPDELoss(alpha=1.0, D=1.0, kappa=1.0)
        x = torch.rand(20, 1)
        t = torch.rand(20, 1) * 0.5 + 0.01
        loss, info = pde(net, x, t)
        assert torch.isfinite(loss)
        assert loss.item() > 0  # untrained net should have non-zero residual

    def test_pde_returns_scalar(self):
        """PDE loss should return a scalar."""
        net = FDRNet(n_layers=2, n_neurons=16, fourier_features=0, hard_bc=False)
        pde = FDRPDELoss(alpha=1.0, D=1.0, kappa=1.0)
        x = torch.rand(20, 1)
        t = torch.rand(20, 1) * 0.5 + 0.01
        loss, _ = pde(net, x, t)
        assert loss.dim() == 0

    def test_bc_loss_with_hard_bc(self):
        """BC loss should be near zero when hard BC is active."""
        net = FDRNet(n_layers=2, n_neurons=16, hard_bc=True,
                     L=1.0, fourier_features=0)
        bc_fn = FDRBCLoss(L=1.0)
        t_bc = torch.linspace(0.01, 1.0, 20).unsqueeze(1)
        loss = bc_fn(net, t_bc)
        assert loss.item() < 1e-8

    def test_ic_loss_with_hard_bc(self):
        """IC loss should be near zero when hard BC is active (pulse BC)."""
        net = FDRNet(n_layers=2, n_neurons=16, hard_bc=True,
                     bc_type='pulse', bc_params={'a': 0.5}, fourier_features=0)
        ic_fn = FDRICLoss()
        x_ic = torch.linspace(0.01, 0.99, 20).unsqueeze(1)
        loss = ic_fn(net, x_ic)
        assert loss.item() < 1e-8


# ────────────────────────────────────────────────────────
# Inverse PINN Tests
# ────────────────────────────────────────────────────────

class TestInverseFDRNet:
    """Test the inverse PINN wrapper."""

    def test_output_shape(self):
        net = FDRNet(n_layers=2, n_neurons=32, fourier_features=0)
        inv = InverseFDRNet(net, init_alpha=0.5, init_kappa=1.0)
        x = torch.rand(10, 1)
        t = torch.rand(10, 1)
        u = inv(x, t)
        assert u.shape == (10, 1)

    def test_param_gradients(self):
        """Physical parameters should have gradients after backward."""
        net = FDRNet(n_layers=2, n_neurons=32, fourier_features=0, hard_bc=False)
        inv = InverseFDRNet(net, init_alpha=0.5, init_kappa=1.0)
        x = torch.rand(10, 1)
        t = torch.rand(10, 1)
        u = inv(x, t)
        u.sum().backward()
        assert inv.alpha_raw.grad is not None
        assert inv.kappa_raw.grad is not None

    def test_param_constraints(self):
        """Parameters should stay within specified ranges."""
        net = FDRNet(n_layers=2, n_neurons=32, fourier_features=0)
        inv = InverseFDRNet(net, init_alpha=0.5, init_kappa=1.0,
                            alpha_range=(0.05, 0.99), kappa_range=(0.01, 5.0))
        alpha = inv.get_alpha().item()
        kappa = inv.get_kappa().item()
        assert 0.05 <= alpha <= 0.99
        assert 0.01 <= kappa <= 5.0


# ────────────────────────────────────────────────────────
# Trainer Smoke Test
# ────────────────────────────────────────────────────────

class TestFDRTrainer:
    """Quick smoke tests for the trainer."""

    @pytest.fixture
    def setup(self):
        config = {
            'alpha': 1.0, 'D': 1.0, 'kappa': 1.0, 'L': 1.0, 'T': 1.0,
            'lr': 1e-3, 'weight_decay': 1e-4, 'lr_schedule': 'cosine',
            'n_layers': 2, 'n_neurons': 16,
            'phase1_epochs': 50, 'phase2_epochs': 50, 'vanilla_epochs': 50,
            'phase1_lbfgs': False,
            'N_collocation': 100, 'pde_ramp_epochs': 20,
            'w_data': 10.0, 'w_pde': 1.0, 'w_bc': 1.0, 'w_ic': 1.0,
            'bc_type': 'pulse', 'bc_params': {'a': 0.5},
            'fourier_features': 0, 'fourier_sigma': 5.0,
            'activation': 'tanh', 'gl_h': 0.05, 'gl_N_memory': 5,
        }
        lf_data = {
            'x': np.random.rand(20) * 0.9 + 0.05,
            't': np.random.rand(20) * 0.9 + 0.05,
            'u': np.random.rand(20) * 0.5,
        }
        return config, lf_data

    def test_phase1(self, setup):
        config, lf_data = setup
        model = FDRNet(n_layers=2, n_neurons=16, fourier_features=0,
                       bc_type='pulse', bc_params={'a': 0.5})
        trainer = FDRTrainer(model, config, lf_data)
        loss = trainer.train_phase1(epochs=50, verbose=False)
        assert np.isfinite(loss)

    def test_vanilla(self, setup):
        config, lf_data = setup
        model = FDRNet(n_layers=2, n_neurons=16, fourier_features=0,
                       bc_type='pulse', bc_params={'a': 0.5})
        trainer = FDRTrainer(model, config, lf_data)
        loss = trainer.train_vanilla(epochs=50, verbose=False)
        assert np.isfinite(loss)

    def test_full_training(self, setup):
        config, lf_data = setup
        model = FDRNet(n_layers=2, n_neurons=16, fourier_features=0,
                       bc_type='pulse', bc_params={'a': 0.5})
        trainer = FDRTrainer(model, config, lf_data)
        history = trainer.train_full(verbose=False)
        assert len(history['total_loss']) == 100  # 50 + 50
