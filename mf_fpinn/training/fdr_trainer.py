"""
Two-phase trainer for the fractional diffusion-reaction PINN.

Phase 1: Data fitting on sparse Laplace-domain evaluations.
Phase 2: Physics-informed fine-tuning (PDE + BC + IC + data).
Vanilla: PDE + BC + IC only (no Laplace data).

Supports the 2×2 factorial design:
  M1: Vanilla (no transfer, manual HPs)
  M2: Vanilla + HPO
  M3: MF-fPINN (transfer, manual HPs)
  M4: MF-fPINN + HPO
"""

import time
import torch
import numpy as np
from pathlib import Path

from ..models.fdr_loss import FDRDataLoss, FDRPDELoss, FDRBCLoss, FDRICLoss


class CollocationSampler:
    """Sample collocation points with Beta-biased distribution."""

    def __init__(self, N_col=2000, x_range=(0.0, 1.0), t_range=(0.01, 1.0),
                 device=torch.device('cpu'), resample_every=1000):
        self.N_col = N_col
        self.x_range = x_range
        self.t_range = t_range
        self.device = device
        self.resample_every = resample_every
        self._current = None

    def sample(self, epoch=0):
        if self._current is None or epoch % self.resample_every == 0:
            rng = np.random.default_rng(epoch)
            x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * \
                rng.uniform(0.0, 1.0, self.N_col)
            t = self.t_range[0] + (self.t_range[1] - self.t_range[0]) * \
                rng.beta(0.5, 2.0, self.N_col)
            self._current = (
                torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(1),
                torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(1),
            )
        return self._current


class FDRTrainer:
    """
    Two-phase MF-fPINN trainer for fractional diffusion-reaction.

    Parameters
    ----------
    model : FDRNet
    config : dict
        Training configuration.
    lf_data : dict
        Low-fidelity data {'x', 't', 'u'}.
    device : torch.device
    """

    def __init__(self, model, config, lf_data, device=torch.device('cpu')):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # LF data tensors
        self.x_lf = torch.tensor(
            lf_data['x'], dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.t_lf = torch.tensor(
            lf_data['t'], dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.u_lf = torch.tensor(
            lf_data['u'], dtype=torch.float32, device=device
        ).unsqueeze(1)

        # Loss functions
        u_scale = max(self.u_lf.abs().max().item(), 0.01)
        self.data_loss_fn = FDRDataLoss(u_scale=u_scale)

        alpha = config.get('alpha', 1.0)
        D = config.get('D', 1.0)
        kappa = config.get('kappa', 1.0)
        self.pde_loss_fn = FDRPDELoss(
            alpha=alpha, D=D, kappa=kappa,
            gl_h=config.get('gl_h', 0.01),
            gl_N_memory=config.get('gl_N_memory', 100),
        ).to(device)
        L = config.get('L', 1.0)
        self.bc_loss_fn = FDRBCLoss(L=L)
        self.ic_loss_fn = FDRICLoss()

        # Collocation sampler
        self.sampler = CollocationSampler(
            N_col=config.get('N_collocation', 2000),
            x_range=(0.0, L),
            t_range=(0.01, config.get('T', 1.0)),
            device=device,
        )

        # BC/IC evaluation points
        T = config.get('T', 1.0)
        self.t_bc = torch.linspace(0.01, T, 50, device=device).unsqueeze(1)
        self.x_ic = torch.linspace(0.01, L, 50, device=device).unsqueeze(1)

        # History
        self.history = {
            'phase': [], 'epoch': [], 'total_loss': [],
            'data_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'lr': [], 'wall_time': []
        }

    def _build_optimizer(self, lr=None):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr or self.config.get('lr', 1e-3),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

    def _build_scheduler(self, optimizer, total_epochs):
        schedule = self.config.get('lr_schedule', 'cosine')
        if schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=1e-6
            )
        elif schedule == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=total_epochs // 3, gamma=0.5
            )
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_epochs
        )

    def train_phase1(self, epochs=None, verbose=True):
        """Phase 1: Pure data fitting on Laplace-domain evaluations."""
        epochs = epochs or self.config.get('phase1_epochs', 5000)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PHASE 1: Data Pre-training ({epochs} epochs)")
            print(f"  Data points: {len(self.x_lf)}")
            print(f"{'='*60}", flush=True)

        self.model.train()
        start = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()
            u_pred = self.model(self.x_lf, self.t_lf)
            loss = self.data_loss_fn(u_pred, self.u_lf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            self._log(1, epoch, loss.item(), loss.item(), 0, 0, 0,
                      optimizer.param_groups[0]['lr'], start)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                      f"Loss: {loss.item():.6e}  |  "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # L-BFGS refinement with safeguard
        if self.config.get('phase1_lbfgs', True):
            n_steps = self.config.get('phase1_lbfgs_steps', 50)
            if verbose:
                print(f"  L-BFGS refinement ({n_steps} steps)...", flush=True)
            # Save state before L-BFGS in case it diverges
            with torch.no_grad():
                pre_lbfgs_loss = self.data_loss_fn(
                    self.model(self.x_lf, self.t_lf), self.u_lf
                ).item()
            pre_lbfgs_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            lbfgs = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0,
                max_iter=20, history_size=50,
                line_search_fn='strong_wolfe'
            )
            for _ in range(n_steps):
                def closure():
                    lbfgs.zero_grad()
                    pred = self.model(self.x_lf, self.t_lf)
                    l = self.data_loss_fn(pred, self.u_lf)
                    l.backward()
                    return l
                lbfgs.step(closure)

            # Check if L-BFGS improved; rollback if it diverged or created bad solution
            with torch.no_grad():
                post_lbfgs_loss = self.data_loss_fn(
                    self.model(self.x_lf, self.t_lf), self.u_lf
                ).item()
                # Also check model output on a coarse grid for wild oscillations
                x_check = torch.linspace(0, self.config['L'], 20,
                                         device=self.device).unsqueeze(1)
                t_check = torch.full_like(x_check, 0.25)
                u_check = self.model(x_check, t_check)
                u_max = u_check.abs().max().item()
                u_data_max = self.u_lf.abs().max().item()

            rollback = False
            if post_lbfgs_loss > pre_lbfgs_loss * 1.5:
                rollback = True
                reason = f"loss diverged ({post_lbfgs_loss:.4e} > {pre_lbfgs_loss:.4e})"
            elif u_max > 3 * max(u_data_max, 0.1):
                rollback = True
                reason = f"output unbounded (max |u|={u_max:.1f} vs data max={u_data_max:.3f})"

            if rollback:
                self.model.load_state_dict(pre_lbfgs_state)
                if verbose:
                    print(f"  L-BFGS safeguard: {reason}, rolled back", flush=True)

        with torch.no_grad():
            final = self.data_loss_fn(
                self.model(self.x_lf, self.t_lf), self.u_lf
            ).item()

        if verbose:
            print(f"  Phase 1 done in {time.time()-start:.1f}s  |  "
                  f"Final: {final:.6e}", flush=True)
        return final

    def train_phase2(self, epochs=None, verbose=True):
        """Phase 2: Physics-informed fine-tuning."""
        epochs = epochs or self.config.get('phase2_epochs', 10000)
        lr = self.config.get('phase2_lr', self.config.get('lr', 1e-3))
        optimizer = self._build_optimizer(lr=lr)
        scheduler = self._build_scheduler(optimizer, epochs)

        w_data = self.config.get('w_data', 10.0)
        w_pde = self.config.get('w_pde', 1.0)
        w_bc = self.config.get('w_bc', 1.0)
        w_ic = self.config.get('w_ic', 1.0)
        ramp = self.config.get('pde_ramp_epochs', 2000)
        # Adaptive data weight annealing: linearly decay w_data to zero
        # over w_data_anneal_epochs. 0 = no annealing (constant w_data).
        data_anneal = self.config.get('w_data_anneal_epochs', 0)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PHASE 2: Physics-Informed ({epochs} epochs)")
            print(f"  Weights: data={w_data}, pde={w_pde}, bc={w_bc}, ic={w_ic}")
            print(f"  PDE ramp: {ramp} epochs")
            if data_anneal > 0:
                print(f"  Data anneal: w_data -> 0 over {data_anneal} epochs")
            print(f"{'='*60}", flush=True)

        self.model.train()
        start = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Data loss
            u_pred = self.model(self.x_lf, self.t_lf)
            l_data = self.data_loss_fn(u_pred, self.u_lf)

            # PDE loss
            x_col, t_col = self.sampler.sample(epoch)
            l_pde, _ = self.pde_loss_fn(self.model, x_col, t_col)

            # BC/IC losses
            l_bc = self.bc_loss_fn(self.model, self.t_bc)
            l_ic = self.ic_loss_fn(self.model, self.x_ic)

            # PDE ramp-up
            pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0

            # Data weight annealing
            if data_anneal > 0:
                w_data_eff = w_data * max(0.0, 1.0 - epoch / data_anneal)
            else:
                w_data_eff = w_data

            total = (w_data_eff * l_data + w_pde * pde_scale * l_pde +
                     w_bc * l_bc + w_ic * l_ic)

            total.backward()
            clip = self.config.get('grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            self._log(2, epoch, total.item(), l_data.item(), l_pde.item(),
                      l_bc.item(), l_ic.item(),
                      optimizer.param_groups[0]['lr'], start)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                      f"Total: {total.item():.4e}  Data: {l_data.item():.4e}  "
                      f"PDE: {l_pde.item():.4e}  BC: {l_bc.item():.4e}  "
                      f"IC: {l_ic.item():.4e}", flush=True)

        final = total.item()
        if verbose:
            print(f"  Phase 2 done in {time.time()-start:.1f}s  |  "
                  f"Final: {final:.6e}", flush=True)
        return final

    def train_full(self, verbose=True):
        """Run Phase 1 + Phase 2."""
        p1 = self.train_phase1(verbose=verbose)
        p2 = self.train_phase2(verbose=verbose)
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE  |  P1: {p1:.6e}  P2: {p2:.6e}")
            print(f"  Wall time: {self.history['wall_time'][-1]:.1f}s")
            print(f"{'='*60}", flush=True)
        return self.history

    def train_vanilla(self, epochs=None, verbose=True):
        """Vanilla fPINN: PDE + BC + IC only, no Laplace data."""
        epochs = epochs or self.config.get('vanilla_epochs',
                     self.config.get('phase2_epochs', 20000))
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs)

        w_pde = self.config.get('w_pde', 1.0)
        w_bc = self.config.get('w_bc', 1.0)
        w_ic = self.config.get('w_ic', 1.0)
        ramp = self.config.get('pde_ramp_epochs', 2000)

        if verbose:
            print(f"\n{'='*60}")
            print(f"VANILLA fPINN ({epochs} epochs)  |  NO data")
            print(f"  Weights: pde={w_pde}, bc={w_bc}, ic={w_ic}")
            print(f"{'='*60}", flush=True)

        self.model.train()
        start = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()

            x_col, t_col = self.sampler.sample(epoch)
            l_pde, _ = self.pde_loss_fn(self.model, x_col, t_col)
            l_bc = self.bc_loss_fn(self.model, self.t_bc)
            l_ic = self.ic_loss_fn(self.model, self.x_ic)

            pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
            total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            self._log(0, epoch, total.item(), 0, l_pde.item(),
                      l_bc.item(), l_ic.item(),
                      optimizer.param_groups[0]['lr'], start)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                      f"Total: {total.item():.4e}  PDE: {l_pde.item():.4e}  "
                      f"BC: {l_bc.item():.4e}  IC: {l_ic.item():.4e}", flush=True)

        final = total.item()
        if verbose:
            print(f"  Vanilla done in {time.time()-start:.1f}s  |  "
                  f"Final: {final:.6e}", flush=True)
        return final

    def train_rar(self, epochs=None, verbose=True, rar_interval=2000,
                  rar_fraction=0.2, n_candidates=10000):
        """Vanilla fPINN with Residual-Adaptive Refinement (RAR).

        Every `rar_interval` epochs, evaluate PDE residual on `n_candidates`
        random points and replace `rar_fraction` of collocation points with
        those having the highest residual.
        """
        epochs = epochs or self.config.get('vanilla_epochs',
                     self.config.get('phase2_epochs', 20000))
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs)

        w_pde = self.config.get('w_pde', 1.0)
        w_bc = self.config.get('w_bc', 1.0)
        w_ic = self.config.get('w_ic', 1.0)
        ramp = self.config.get('pde_ramp_epochs', 2000)

        N_col = self.config.get('N_collocation', 2000)
        L = self.config.get('L', 1.0)
        T = self.config.get('T', 1.0)
        n_replace = int(N_col * rar_fraction)

        if verbose:
            print(f"\n{'='*60}")
            print(f"RAR fPINN ({epochs} epochs)  |  NO data")
            print(f"  RAR: every {rar_interval} epochs, replace {n_replace}/{N_col} pts")
            print(f"  Weights: pde={w_pde}, bc={w_bc}, ic={w_ic}")
            print(f"{'='*60}", flush=True)

        self.model.train()
        start = time.time()

        # Initial collocation points
        rng = np.random.default_rng(0)
        x_col = torch.tensor(rng.uniform(0.0, 1.0, N_col) * L, dtype=torch.float32,
                             device=self.device).unsqueeze(1)
        t_col = torch.tensor(0.01 + rng.beta(0.5, 2.0, N_col) * (T - 0.01),
                             dtype=torch.float32, device=self.device).unsqueeze(1)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # RAR: adaptively refine collocation points
            if epoch > 0 and epoch % rar_interval == 0:
                cand_rng = np.random.default_rng(epoch)
                x_cand = torch.tensor(
                    cand_rng.uniform(0, L, n_candidates),
                    dtype=torch.float32, device=self.device
                ).unsqueeze(1).requires_grad_(True)
                t_cand = torch.tensor(
                    cand_rng.uniform(0.01, T, n_candidates),
                    dtype=torch.float32, device=self.device
                ).unsqueeze(1).requires_grad_(True)

                with torch.enable_grad():
                    u = self.model(x_cand, t_cand)
                    du_dx = torch.autograd.grad(
                        u, x_cand, torch.ones_like(u),
                        create_graph=True, retain_graph=True)[0]
                    d2u_dx2 = torch.autograd.grad(
                        du_dx, x_cand, torch.ones_like(du_dx),
                        create_graph=False, retain_graph=True)[0]

                    if self.pde_loss_fn.alpha >= 0.999:
                        du_dt = torch.autograd.grad(
                            u, t_cand, torch.ones_like(u),
                            create_graph=False, retain_graph=True)[0]
                        R = du_dt - self.pde_loss_fn.D * d2u_dx2 + self.pde_loss_fn.kappa * u
                    else:
                        caputo_u = self.pde_loss_fn._caputo_gl(self.model, x_cand, t_cand)
                        R = caputo_u - self.pde_loss_fn.D * d2u_dx2 + self.pde_loss_fn.kappa * u

                res_abs = R.abs().detach().squeeze()
                x_cand_d = x_cand.detach()
                t_cand_d = t_cand.detach()

                topk_idx = torch.topk(res_abs, n_replace).indices
                replace_idx = torch.randperm(N_col, device=self.device)[:n_replace]
                x_col[replace_idx] = x_cand_d[topk_idx]
                t_col[replace_idx] = t_cand_d[topk_idx]

                if verbose:
                    print(f"  RAR epoch {epoch}: replaced {n_replace} pts "
                          f"(max res={res_abs.max():.4e})", flush=True)

            x_col = x_col.detach().requires_grad_(True)
            t_col = t_col.detach().requires_grad_(True)

            l_pde, _ = self.pde_loss_fn(self.model, x_col, t_col)
            l_bc = self.bc_loss_fn(self.model, self.t_bc)
            l_ic = self.ic_loss_fn(self.model, self.x_ic)

            pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
            total = w_pde * pde_scale * l_pde + w_bc * l_bc + w_ic * l_ic

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            x_col = x_col.detach()
            t_col = t_col.detach()

            self._log(0, epoch, total.item(), 0, l_pde.item(),
                      l_bc.item(), l_ic.item(),
                      optimizer.param_groups[0]['lr'], start)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                      f"Total: {total.item():.4e}  PDE: {l_pde.item():.4e}  "
                      f"BC: {l_bc.item():.4e}  IC: {l_ic.item():.4e}", flush=True)

        final = total.item()
        if verbose:
            print(f"  RAR done in {time.time()-start:.1f}s  |  "
                  f"Final: {final:.6e}", flush=True)
        return final

    def train_causal(self, epochs=None, verbose=True, epsilon=10.0, n_windows=20):
        """Causal training (Wang et al. 2022): weight PDE loss by temporal causality.

        Collocation points are sorted into temporal windows. The PDE residual
        in each window is weighted by exp(-ε * cumulative_residual_from_earlier_windows).
        This forces the network to satisfy the PDE at earlier times first.

        Parameters
        ----------
        epochs : int
            Training epochs (default: vanilla_epochs = 25000).
        epsilon : float
            Causality parameter. Larger = more focus on earlier times.
        n_windows : int
            Number of temporal bins for causal weighting.

        Fairness: Uses same architecture, epochs, LR, collocation budget as
        train_vanilla(). Only difference is the per-window causal weighting.
        """
        epochs = epochs or self.config.get('vanilla_epochs',
                     self.config.get('phase2_epochs', 20000))
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, epochs)

        w_pde = self.config.get('w_pde', 1.0)
        w_bc = self.config.get('w_bc', 1.0)
        w_ic = self.config.get('w_ic', 1.0)
        ramp = self.config.get('pde_ramp_epochs', 2000)

        T = self.config.get('T', 1.0)
        t_min = 0.01
        window_edges = torch.linspace(t_min, T, n_windows + 1, device=self.device)

        if verbose:
            print(f"\n{'='*60}")
            print(f"CAUSAL fPINN ({epochs} epochs, ε={epsilon}, {n_windows} windows)")
            print(f"  Weights: pde={w_pde}, bc={w_bc}, ic={w_ic}")
            print(f"{'='*60}", flush=True)

        self.model.train()
        start = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()

            x_col, t_col = self.sampler.sample(epoch)

            # --- Causal weighting ---
            # Compute per-point PDE residual (no reduction)
            x_c = x_col.requires_grad_(True)
            t_c = t_col.requires_grad_(True)
            u = self.model(x_c, t_c)

            du_dx = torch.autograd.grad(
                u, x_c, torch.ones_like(u), create_graph=True, retain_graph=True
            )[0]
            d2u_dx2 = torch.autograd.grad(
                du_dx, x_c, torch.ones_like(du_dx), create_graph=True, retain_graph=True
            )[0]

            if self.pde_loss_fn.alpha >= 0.999:
                du_dt = torch.autograd.grad(
                    u, t_c, torch.ones_like(u), create_graph=True, retain_graph=True
                )[0]
                R = du_dt - self.pde_loss_fn.D * d2u_dx2 + self.pde_loss_fn.kappa * u
            else:
                caputo_u = self.pde_loss_fn._caputo_gl(self.model, x_c, t_c)
                R = caputo_u - self.pde_loss_fn.D * d2u_dx2 + self.pde_loss_fn.kappa * u

            R_sq = R.squeeze() ** 2  # (N_col,)
            t_flat = t_col.squeeze()  # (N_col,)

            # Compute mean residual per window
            window_losses = []
            for w in range(n_windows):
                mask = (t_flat >= window_edges[w]) & (t_flat < window_edges[w + 1])
                if mask.any():
                    window_losses.append(R_sq[mask].mean())
                else:
                    window_losses.append(torch.zeros(1, device=self.device).squeeze())

            # Causal weights: w_i = exp(-ε * Σ_{j<i} L_j)
            cumsum = torch.zeros(1, device=self.device)
            causal_weights = []
            for w_loss in window_losses:
                causal_weights.append(torch.exp(-epsilon * cumsum.detach()))
                cumsum = cumsum + w_loss.detach()

            # Weighted PDE loss = Σ_i w_i * L_i
            l_pde_causal = torch.zeros(1, device=self.device)
            for cw, wl in zip(causal_weights, window_losses):
                l_pde_causal = l_pde_causal + cw * wl

            l_pde_causal = l_pde_causal / n_windows  # normalize

            l_bc = self.bc_loss_fn(self.model, self.t_bc)
            l_ic = self.ic_loss_fn(self.model, self.x_ic)

            pde_scale = min(1.0, epoch / ramp) if ramp > 0 else 1.0
            total = w_pde * pde_scale * l_pde_causal + w_bc * l_bc + w_ic * l_ic

            total.backward()
            clip = self.config.get('grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            self._log(0, epoch, total.item(), 0, l_pde_causal.item(),
                      l_bc.item(), l_ic.item(),
                      optimizer.param_groups[0]['lr'], start)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                # Show causality info
                min_cw = min(cw.item() for cw in causal_weights)
                print(f"  Epoch {epoch+1:>6d}/{epochs}  |  "
                      f"Total: {total.item():.4e}  PDE: {l_pde_causal.item():.4e}  "
                      f"BC: {l_bc.item():.4e}  IC: {l_ic.item():.4e}  "
                      f"min_cw: {min_cw:.4f}", flush=True)

        final = total.item()
        if verbose:
            print(f"  Causal done in {time.time()-start:.1f}s  |  "
                  f"Final: {final:.6e}", flush=True)
        return final

    def _log(self, phase, epoch, total, data, pde, bc, ic, lr, start):
        self.history['phase'].append(phase)
        self.history['epoch'].append(epoch)
        self.history['total_loss'].append(total)
        self.history['data_loss'].append(data)
        self.history['pde_loss'].append(pde)
        self.history['bc_loss'].append(bc)
        self.history['ic_loss'].append(ic)
        self.history['lr'].append(lr)
        self.history['wall_time'].append(time.time() - start)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'history' in ckpt:
            self.history = ckpt['history']
