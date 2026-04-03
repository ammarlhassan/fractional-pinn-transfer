"""
2D PINN for the fractional diffusion-reaction equation.

    ∂ᵗᵅ u = D(∂²u/∂x² + ∂²u/∂y²) − κ u

Input: (x, y, t) → Output: u(x, y, t)

No hard BC — uses soft BC/IC losses instead.
The uniform x=0 BC (u(0,y,t) = g(t)) is incompatible with homogeneous
y-BCs (u(x,0,t) = u(x,Ly,t) = 0) at corners, creating singularities
that prevent hard BC enforcement from working.
"""

import torch
import torch.nn as nn
import numpy as np


ACTIVATIONS = {
    'tanh': nn.Tanh,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
}


class FourierFeatureEmbedding2D(nn.Module):
    """Random Fourier features for 3D input (x, y, t)."""

    def __init__(self, n_features=64, sigma=5.0):
        super().__init__()
        B = torch.randn(3, n_features) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FDRNet2D(nn.Module):
    """
    2D PINN for u(x, y, t).

    Parameters
    ----------
    n_layers : int
    n_neurons : int
    activation : str
    Lx, Ly : float
        Domain dimensions.
    T : float
        Time horizon.
    bc_type : str
        'pulse' or 'step'.
    bc_params : dict
    fourier_features : int
    fourier_sigma : float
    """

    def __init__(self, n_layers=4, n_neurons=128, activation='tanh',
                 Lx=1.0, Ly=1.0, T=0.5,
                 bc_type='pulse', bc_params=None,
                 fourier_features=64, fourier_sigma=5.0):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.bc_type = bc_type
        self.bc_params = bc_params or {}
        self.use_fourier = fourier_features > 0

        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5

        # Normalization
        self.register_buffer('x_scale', torch.tensor(2.0 / Lx, dtype=torch.float32))
        self.register_buffer('y_scale', torch.tensor(2.0 / Ly, dtype=torch.float32))
        self.register_buffer('t_scale', torch.tensor(2.0 / T, dtype=torch.float32))

        act_class = ACTIVATIONS[activation]

        if self.use_fourier:
            self.fourier = FourierFeatureEmbedding2D(fourier_features, fourier_sigma)
            input_dim = 2 * fourier_features
        else:
            self.fourier = None
            input_dim = 3

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
        """g(t) — temporal part of BC at x=0."""
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

    def forward(self, x, y, t):
        """
        Parameters
        ----------
        x, y, t : Tensor, shape (N, 1)

        Returns
        -------
        u : Tensor, shape (N, 1)
        """
        x_norm = x * self.x_scale - 1.0
        y_norm = y * self.y_scale - 1.0
        t_norm = t * self.t_scale - 1.0
        inp = torch.cat([x_norm, y_norm, t_norm], dim=-1)

        if self.use_fourier:
            inp = self.fourier(inp)

        return self.head(self.trunk(inp))
