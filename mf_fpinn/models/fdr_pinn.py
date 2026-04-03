"""
Single-output PINN for the fractional diffusion-reaction equation.

    ∂ᵗᵅ u = D ∂²u/∂x² − κ u

Architecture: Fourier feature embedding → shared MLP → single output head.
Hard BC enforcement: u(x,t) = g(t)·(L−x)/L + x·(L−x)·t·NN(x,t)
  → u(0,t) = g(t), u(L,t) = 0, u(x,0) = 0  automatically.
"""

import torch
import torch.nn as nn
import numpy as np


ACTIVATIONS = {
    'tanh': nn.Tanh,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
}


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier features: z → [sin(Bz), cos(Bz)]."""

    def __init__(self, in_dim=2, n_features=64, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FDRNet(nn.Module):
    """
    Single-output PINN for u(x, t).

    Parameters
    ----------
    n_layers : int
        Number of hidden layers.
    n_neurons : int
        Width of each hidden layer.
    activation : str
        'tanh', 'swish', or 'gelu'.
    L : float
        Domain length.
    T : float
        Time horizon.
    hard_bc : bool
        Enforce BCs via output transformation.
    bc_type : str
        'pulse', 'step', or 'ramp'.
    bc_params : dict
        BC parameters (e.g. {'a': 0.5} for pulse).
    fourier_features : int
        Number of Fourier features (0 = disabled).
    fourier_sigma : float
        Fourier feature frequency scale.
    """

    def __init__(self, n_layers=4, n_neurons=128, activation='tanh',
                 L=1.0, T=1.0, hard_bc=True,
                 bc_type='pulse', bc_params=None,
                 fourier_features=64, fourier_sigma=5.0):
        super().__init__()
        self.L = L
        self.T = T
        self.hard_bc = hard_bc
        self.bc_type = bc_type
        self.bc_params = bc_params or {}
        self.use_fourier = fourier_features > 0

        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5

        # Input normalization buffers
        self.register_buffer('x_scale', torch.tensor(2.0 / L, dtype=torch.float32))
        self.register_buffer('t_scale', torch.tensor(2.0 / T, dtype=torch.float32))

        act_class = ACTIVATIONS[activation]

        # Fourier features
        if self.use_fourier:
            self.fourier = FourierFeatureEmbedding(2, fourier_features, fourier_sigma)
            input_dim = 2 * fourier_features
        else:
            self.fourier = None
            input_dim = 2

        # MLP
        layers = [nn.Linear(input_dim, n_neurons), act_class()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act_class()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(n_neurons, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _bc_function(self, t):
        """Evaluate g(t) — the boundary condition at x=0."""
        if self.bc_type == 'step':
            return torch.ones_like(t)

        elif self.bc_type == 'pulse':
            a = self.bc_params['a']
            omega = np.pi / a
            g = torch.sin(omega * t)
            mask = torch.sigmoid(100.0 * (a - t)) * torch.sigmoid(100.0 * t)
            return g * mask

        elif self.bc_type == 'ramp':
            b = self.bc_params.get('b', 10.0)
            return 1.0 - torch.exp(-b * t)

        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : Tensor, shape (N, 1)
        t : Tensor, shape (N, 1)

        Returns
        -------
        u : Tensor, shape (N, 1)
        """
        # Normalize to [-1, 1]
        x_norm = x * self.x_scale - 1.0
        t_norm = t * self.t_scale - 1.0
        inp = torch.cat([x_norm, t_norm], dim=-1)

        if self.use_fourier:
            inp = self.fourier(inp)

        nn_out = self.head(self.trunk(inp))

        if self.hard_bc:
            # u = g(t)·(L−x)/L + x·(L−x)·t·NN(x,t)
            # Guarantees: u(0,t)=g(t), u(L,t)=0, u(x,0)=0 (if g(0)=0)
            L = self.L
            g_t = self._bc_function(t)
            u = g_t * (L - x) / L + x * (L - x) * t * nn_out
        else:
            u = nn_out

        return u


class InverseFDRNet(nn.Module):
    """
    Inverse PINN wrapper — jointly learns u(x,t) and physical parameters.

    Trainable parameters: α (fractional order) and/or κ (reaction rate).
    Uses FiLM conditioning so forward pass depends on (α, κ).

    Parameters
    ----------
    net : FDRNet
        Base forward network.
    init_alpha : float
        Initial guess for α.
    init_kappa : float
        Initial guess for κ.
    alpha_range : tuple
        Constraint range for α.
    kappa_range : tuple
        Constraint range for κ.
    fix_alpha, fix_kappa : bool
        Whether to hold each parameter fixed.
    """

    def __init__(self, net, init_alpha=0.5, init_kappa=1.0,
                 alpha_range=(0.05, 0.99), kappa_range=(0.01, 5.0),
                 fix_alpha=False, fix_kappa=False):
        super().__init__()
        self.net = net
        self.alpha_range = alpha_range
        self.kappa_range = kappa_range
        self.fix_alpha = fix_alpha
        self.fix_kappa = fix_kappa

        # Unconstrained parameters (sigmoid-mapped)
        a_min, a_max = alpha_range
        alpha_sig = np.clip((init_alpha - a_min) / (a_max - a_min), 0.01, 0.99)
        self.alpha_raw = nn.Parameter(
            torch.tensor(float(np.log(alpha_sig / (1 - alpha_sig))),
                         dtype=torch.float32),
            requires_grad=not fix_alpha
        )

        k_min, k_max = kappa_range
        kappa_sig = np.clip((init_kappa - k_min) / (k_max - k_min), 0.01, 0.99)
        self.kappa_raw = nn.Parameter(
            torch.tensor(float(np.log(kappa_sig / (1 - kappa_sig))),
                         dtype=torch.float32),
            requires_grad=not fix_kappa
        )

        # FiLM: condition trunk features on (α, κ)
        n_neurons = net.head.in_features
        self.cond_net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 2 * n_neurons)
        )
        nn.init.normal_(self.cond_net[-1].weight, std=0.01)
        nn.init.zeros_(self.cond_net[-1].bias)

    def get_alpha(self):
        a_min, a_max = self.alpha_range
        return a_min + (a_max - a_min) * torch.sigmoid(self.alpha_raw)

    def get_kappa(self):
        k_min, k_max = self.kappa_range
        return k_min + (k_max - k_min) * torch.sigmoid(self.kappa_raw)

    def forward(self, x, t):
        """Forward pass with FiLM conditioning on (α, κ)."""
        x_norm = x * self.net.x_scale - 1.0
        t_norm = t * self.net.t_scale - 1.0
        inp = torch.cat([x_norm, t_norm], dim=-1)

        if self.net.use_fourier:
            inp = self.net.fourier(inp)

        features = self.net.trunk(inp)

        # FiLM modulation
        alpha_n = 2.0 * (self.get_alpha() - self.alpha_range[0]) / \
                  (self.alpha_range[1] - self.alpha_range[0]) - 1.0
        kappa_n = 2.0 * (self.get_kappa() - self.kappa_range[0]) / \
                  (self.kappa_range[1] - self.kappa_range[0]) - 1.0
        param_in = torch.stack([alpha_n, kappa_n]).unsqueeze(0).expand(x.shape[0], -1)
        cond = self.cond_net(param_in)
        n = features.shape[-1]
        features = (1.0 + cond[..., :n]) * features + cond[..., n:]

        nn_out = self.net.head(features)

        if self.net.hard_bc:
            L = self.net.L
            g_t = self.net._bc_function(t)
            u = g_t * (L - x) / L + x * (L - x) * t * nn_out
        else:
            u = nn_out

        return u

    def network_params(self):
        yield from self.net.parameters()
        yield from self.cond_net.parameters()

    def physical_params(self):
        params = []
        if not self.fix_alpha:
            params.append(self.alpha_raw)
        if not self.fix_kappa:
            params.append(self.kappa_raw)
        return params

    def param_dict(self):
        return {'alpha': self.get_alpha().item(), 'kappa': self.get_kappa().item()}
