"""
Composite Neural Network for multi-fidelity learning (Meng & Karniadakis 2020).

Architecture:
    u_H(x,t) = NN_L(x,t) + NN_lin(NN_L(x,t); x,t) + NN_nl(NN_L(x,t); x,t)

Where:
    NN_L  : Low-fidelity network that learns the LF data distribution
    NN_lin: Linear (shallow) correction — takes [u_L, x, t] as input
    NN_nl : Nonlinear (deeper) correction — takes [u_L, x, t] as input

The composite output u_H is used in the PDE loss.

Reference:
    Meng, X. & Karniadakis, G.E. (2020). A composite neural network that
    learns from multi-fidelity data: Application to function approximation
    and inverse PDE problems. J. Comput. Phys., 401, 109020.
"""

import torch
import torch.nn as nn
import numpy as np

from .fdr_pinn import FourierFeatureEmbedding, ACTIVATIONS


class _CorrectionNet(nn.Module):
    """Small MLP that maps [u_L, x, t] -> correction scalar.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers.
    n_neurons : int
        Width of each hidden layer.
    activation : str
        Activation function name.
    """

    def __init__(self, n_layers=2, n_neurons=64, activation='tanh'):
        super().__init__()
        act_class = ACTIVATIONS[activation]
        # Input: [u_L, x, t] -> 3 features
        layers = [nn.Linear(3, n_neurons), act_class()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act_class()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize last layer small so corrections start near zero
        last = self.net[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, correction_input):
        """correction_input: (N, 3) = [u_L, x, t]."""
        return self.net(correction_input)


class CompositeMFNet(nn.Module):
    """
    Meng-Karniadakis composite NN for multi-fidelity learning.

    u_H(x,t) = hard_bc( raw_L + lin_corr + nl_corr )

    where raw_L is the raw NN_L output (before hard BC),
    lin_corr and nl_corr take [u_L_bc, x, t] as input.

    Parameters
    ----------
    n_layers_lf : int
        Hidden layers for NN_L (low-fidelity network).
    n_neurons_lf : int
        Width for NN_L.
    n_layers_lin : int
        Hidden layers for NN_lin (linear/shallow correction).
    n_neurons_lin : int
        Width for NN_lin.
    n_layers_nl : int
        Hidden layers for NN_nl (nonlinear/deeper correction).
    n_neurons_nl : int
        Width for NN_nl.
    activation : str
        Activation function ('tanh', 'swish', 'gelu').
    L : float
        Domain length.
    T : float
        Time horizon.
    hard_bc : bool
        Whether to enforce hard BCs via output transformation.
    bc_type : str
        BC type ('pulse', 'step', 'ramp').
    bc_params : dict
        BC parameters (e.g. {'a': 0.5} for pulse).
    fourier_features : int
        Number of Fourier features for NN_L (0 = disabled).
    fourier_sigma : float
        Fourier feature frequency scale.
    """

    def __init__(self, n_layers_lf=5, n_neurons_lf=128,
                 n_layers_lin=2, n_neurons_lin=64,
                 n_layers_nl=4, n_neurons_nl=128,
                 activation='tanh', L=1.0, T=1.0,
                 hard_bc=True, bc_type='pulse', bc_params=None,
                 fourier_features=64, fourier_sigma=5.0):
        super().__init__()
        self.L = L
        self.T = T
        self.hard_bc = hard_bc
        self.bc_type = bc_type
        self.bc_params = bc_params or {}

        if bc_type == 'pulse' and 'a' not in self.bc_params:
            self.bc_params['a'] = 0.5

        # Input normalization buffers
        self.register_buffer('x_scale', torch.tensor(2.0 / L, dtype=torch.float32))
        self.register_buffer('t_scale', torch.tensor(2.0 / T, dtype=torch.float32))

        act_class = ACTIVATIONS[activation]
        self.use_fourier = fourier_features > 0

        # ── NN_L: Low-fidelity network (same architecture as FDRNet) ──
        if self.use_fourier:
            self.fourier = FourierFeatureEmbedding(2, fourier_features, fourier_sigma)
            input_dim = 2 * fourier_features
        else:
            self.fourier = None
            input_dim = 2

        lf_layers = [nn.Linear(input_dim, n_neurons_lf), act_class()]
        for _ in range(n_layers_lf - 1):
            lf_layers += [nn.Linear(n_neurons_lf, n_neurons_lf), act_class()]
        self.lf_trunk = nn.Sequential(*lf_layers)
        self.lf_head = nn.Linear(n_neurons_lf, 1)

        # ── NN_lin: Shallow linear correction ──
        self.nn_lin = _CorrectionNet(n_layers_lin, n_neurons_lin, activation)

        # ── NN_nl: Deeper nonlinear correction ──
        self.nn_nl = _CorrectionNet(n_layers_nl, n_neurons_nl, activation)

        self._init_lf_weights()

    def _init_lf_weights(self):
        """Xavier init for the LF network."""
        for m in [self.lf_trunk, self.lf_head]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def _bc_function(self, t):
        """Evaluate g(t) -- the boundary condition at x=0."""
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

    def _apply_hard_bc(self, raw, x, t):
        """Hard BC: u = g(t)*(L-x)/L + x*(L-x)*t*raw."""
        L = self.L
        g_t = self._bc_function(t)
        return g_t * (L - x) / L + x * (L - x) * t * raw

    def _lf_raw(self, x, t):
        """Raw NN_L output (before hard BC)."""
        x_norm = x * self.x_scale - 1.0
        t_norm = t * self.t_scale - 1.0
        inp = torch.cat([x_norm, t_norm], dim=-1)
        if self.use_fourier:
            inp = self.fourier(inp)
        return self.lf_head(self.lf_trunk(inp))

    def forward(self, x, t):
        """
        Composite forward pass:
            u_H = hard_bc( raw_L + lin_corr + nl_corr )

        Parameters
        ----------
        x : Tensor, shape (N, 1)
        t : Tensor, shape (N, 1)

        Returns
        -------
        u : Tensor, shape (N, 1)
        """
        # Raw LF output (before hard BC)
        raw_L = self._lf_raw(x, t)

        # LF output with hard BC (used as input to correction nets)
        u_L_bc = self._apply_hard_bc(raw_L, x, t)

        # Correction input: [u_L_bc, x, t]
        correction_input = torch.cat([u_L_bc, x, t], dim=1)
        lin_corr = self.nn_lin(correction_input)
        nl_corr = self.nn_nl(correction_input)

        # Composite raw = raw_L + corrections, then apply hard BC
        raw_composite = raw_L + lin_corr + nl_corr

        if self.hard_bc:
            return self._apply_hard_bc(raw_composite, x, t)
        else:
            return raw_composite

    def forward_lf(self, x, t):
        """Return just the LF network output (for LF data loss).

        Parameters
        ----------
        x : Tensor, shape (N, 1)
        t : Tensor, shape (N, 1)

        Returns
        -------
        u_L : Tensor, shape (N, 1)
        """
        raw_L = self._lf_raw(x, t)
        if self.hard_bc:
            return self._apply_hard_bc(raw_L, x, t)
        else:
            return raw_L

    def lf_parameters(self):
        """Parameters of the LF network only."""
        yield from self.lf_trunk.parameters()
        yield from self.lf_head.parameters()
        if self.use_fourier:
            yield from self.fourier.parameters()

    def correction_parameters(self):
        """Parameters of the correction networks only."""
        yield from self.nn_lin.parameters()
        yield from self.nn_nl.parameters()

    def param_summary(self):
        """Return parameter counts for each sub-network."""
        n_lf = sum(p.numel() for p in self.lf_trunk.parameters()) + \
               sum(p.numel() for p in self.lf_head.parameters())
        if self.use_fourier:
            n_lf += sum(p.numel() for p in self.fourier.parameters())
        n_lin = sum(p.numel() for p in self.nn_lin.parameters())
        n_nl = sum(p.numel() for p in self.nn_nl.parameters())
        return {
            'nn_lf': n_lf,
            'nn_lin': n_lin,
            'nn_nl': n_nl,
            'total': n_lf + n_lin + n_nl,
        }
