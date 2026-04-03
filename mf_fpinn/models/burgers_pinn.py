"""
PINN model for the time-fractional Burgers equation.

    ∂_t^α u + u ∂u/∂x = ν ∂²u/∂x²
    u(x, 0) = sin(πx),  u(0, t) = u(1, t) = 0

Hard BC ansatz:
    û(x,t) = sin(πx)·exp(-t) + x·(1-x)·t·NN(x,t)

Guarantees: û(x,0) = sin(πx), û(0,t) = 0, û(1,t) = 0.
The exp(-t) decay provides an initial guess corrected by the NN.
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
    """Random Fourier features: z -> [sin(Bz), cos(Bz)]."""

    def __init__(self, in_dim=2, n_features=64, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class BurgersNet(nn.Module):
    """
    Single-output PINN for u(x, t) of the fractional Burgers equation.

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
        Enforce BCs/IC via output transformation.
    fourier_features : int
        Number of Fourier features (0 = disabled).
    fourier_sigma : float
        Fourier feature frequency scale.
    """

    def __init__(self, n_layers=5, n_neurons=128, activation='tanh',
                 L=1.0, T=1.0, hard_bc=True,
                 fourier_features=64, fourier_sigma=5.0):
        super().__init__()
        self.L = L
        self.T = T
        self.hard_bc = hard_bc
        self.use_fourier = fourier_features > 0

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
            # û(x,t) = sin(πx)·exp(-t) + x·(1-x)·t·NN(x,t)
            # Guarantees: û(x,0) = sin(πx), û(0,t) = 0, û(1,t) = 0
            L = self.L
            ic_term = torch.sin(np.pi * x / L) * torch.exp(-t)
            correction = x * (L - x) * t * nn_out
            u = ic_term + correction
        else:
            u = nn_out

        return u
