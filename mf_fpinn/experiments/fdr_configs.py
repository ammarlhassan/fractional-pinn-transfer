"""
Experimental configurations for the fractional diffusion-reaction study.

Problem: ∂ᵗᵅ u = D ∂²u/∂x² − κ u  on [0, L] × [0, T]
BCs:     u(0,t) = g(t),  u(L,t) = 0
IC:      u(x,0) = 0

2×2 Factorial:
  M1: Vanilla fPINN (PDE+BC+IC)
  M2: Vanilla + HPO
  M3: MF-fPINN (Phase 1 Laplace data + Phase 2 physics)
  M4: MF-fPINN + HPO
"""

# Physical parameters
PHYSICS = {
    'D': 1.0,          # diffusion coefficient
    'kappa': 1.0,      # reaction rate
    'L': 1.0,          # domain length
    'T': 1.0,          # time horizon
}

# BC: sin pulse
BC = {
    'bc_type': 'pulse',
    'bc_params': {'a': 0.5},   # g(t) = sin(πt/0.5) for t∈[0, 0.5]
}

# Evaluation time snapshots (within pulse window t∈[0, 0.5] where solution is substantial)
EVAL_TIMES = [0.05, 0.15, 0.25, 0.4]

# ── Experimental Configs ──────────────────────────────────

CONFIG_A = {
    'name': 'Config A: Classical (α=1.0)',
    'alpha': 1.0,
    **PHYSICS, **BC,
    't_values': EVAL_TIMES,
}

CONFIG_B = {
    'name': 'Config B: Subdiffusion (α=0.5)',
    'alpha': 0.5,
    **PHYSICS, **BC,
    't_values': EVAL_TIMES,
}

CONFIG_C = {
    'name': 'Config C: Mild subdiffusion (α=0.75)',
    'alpha': 0.75,
    **PHYSICS, **BC,
    't_values': EVAL_TIMES,
}

CONFIG_D = {
    'name': 'Config D: Strong subdiffusion (α=0.25)',
    'alpha': 0.25,
    **PHYSICS, **BC,
    't_values': EVAL_TIMES,
}

# ── Method Configurations ─────────────────────────────────

METHODS = {
    'M1_vanilla': {
        'description': 'Vanilla fPINN (no transfer, manual HPs)',
        'use_transfer': False,
        'use_hpo': False,
    },
    'M2_vanilla_hpo': {
        'description': 'fPINN + HPO (no transfer, optimized HPs)',
        'use_transfer': False,
        'use_hpo': True,
    },
    'M3_mf': {
        'description': 'MF-fPINN (transfer, manual HPs)',
        'use_transfer': True,
        'use_hpo': False,
    },
    'M4_mf_hpo': {
        'description': 'MF-fPINN + HPO (transfer, optimized HPs)',
        'use_transfer': True,
        'use_hpo': True,
    },
}

# ── Default Hyperparameters (M1/M3 manual) ───────────────

MANUAL_HP = {
    'lr': 5e-4,
    'phase2_lr': 1e-4,
    'weight_decay': 1e-5,
    'lr_schedule': 'cosine',
    'n_layers': 5,
    'n_neurons': 128,
    'activation': 'tanh',
    'fourier_features': 64,
    'fourier_sigma': 5.0,
    'N_collocation': 2000,
    'phase1_epochs': 5000,
    'phase1_lbfgs': True,        # Part of MF method: fits 50 LF points in <1s
    'phase1_lbfgs_steps': 50,
    'phase2_epochs': 20000,
    'vanilla_epochs': 25000,
    'w_data': 10.0,
    'w_data_anneal_epochs': 10000,
    'w_pde': 1.0,
    'w_bc': 1.0,
    'w_ic': 1.0,
    'pde_ramp_epochs': 3000,
    'grad_clip': 5.0,
    'gl_h': 0.01,
    'gl_N_memory': 100,
}

# ── Reference / LF data settings ─────────────────────────

REFERENCE = {
    'nx': 100,
    'precision': 30,
    'N_terms': 30,
}

LF_DATA = {
    'N_LF_default': 50,
    'N_LF_values': [5, 10, 20, 30, 50, 100],
    # Degraded solver settings for genuine multi-fidelity data
    'lf_N_terms': 12,      # vs HF 30 — fewer Laplace inversion terms (~17% L2 error)
    'lf_precision': 20,    # vs HF 30 — lower mpmath decimal precision
}

# ── HPO settings ──────────────────────────────────────────

HPO = {
    'n_trials': 50,
    'n_seeds_final': 10,
}

# ── Transfer efficiency study ─────────────────────────────

TRANSFER_STUDY = {
    'N_LF_values': [5, 10, 20, 30, 50, 100],
    'alpha_values': [0.25, 0.5, 0.75, 1.0],
    'n_seeds': 5,
}
