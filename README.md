# MF-fPINN: Multi-Fidelity Transfer for Fractional PINNs

**Paper:** "Laplace-Domain Multi-Fidelity Transfer for Fractional Physics-Informed Neural Networks in the Low-Collocation Regime"

## Overview

This repository contains the code, data, and paper for a systematic study of when and why multi-fidelity (MF) data transfer improves fractional PINN (fPINN) convergence. The study covers the fractional diffusion-reaction equation

```
∂ᵗᵅ u = D ∂²u/∂x² − κu,   α ∈ (0, 1],   x ∈ (0,1), t > 0
```

using a two-phase training protocol: Phase 1 pre-trains on Laplace-domain low-fidelity (LF) data; Phase 2 refines with the full fractional PDE loss.

**Key finding:** The Grünwald–Letnikov (GL) fractional derivative evaluates the model at K_eff(α) past time steps per collocation point — providing implicit temporal regularization that makes vanilla fPINN already effective at low α. MF transfer is most valuable where GL memory is weakest (α → 1, low collocation budget, high-quality LF data).

## Repository Structure

```
mf_fpinn/               # Core library
├── solvers/            #   Laplace-domain FDR solver (Valsa-Brancik), FD solver
├── models/             #   FDRNet (hard BC), loss functions (GL scheme), Burgers PINN
├── training/           #   Two-phase trainer, HPO (Optuna)
├── experiments/        #   Configs, hyperparameters
└── evaluation/         #   Metrics, plotting

paper/                  # LaTeX source (main.tex + supplementary.tex)
figures/                # Publication-ready PDF figures
reproducibility/        # Curated JSON data for every paper table
scripts/experiments/    # Rerun scripts for all experiments
```

## Quick Start

```bash
pip install -r requirements.txt
python3 -m pytest mf_fpinn/tests/ -v   # 23 tests, all should pass

# Main 1D experiment (M1 vanilla vs M3 MF-fPINN)
python3 run_fdr_comparison.py --alpha 1.0 --device cuda --seeds 5

# Collocation budget study (key figure)
python3 run_fdr_ncol_sweep.py --alpha 1.0 --device cuda --seeds 5
```

## Reproducibility

Every number in every paper table traces to a JSON file in `reproducibility/data/`.  
See [`reproducibility/TABLE_TO_SOURCE.txt`](reproducibility/TABLE_TO_SOURCE.txt) for the complete mapping.

To rerun all experiments (4 GPU batches, can run in parallel):
```bash
python3 scripts/experiments/rerun_all_v2.py --gpu-batch 0 --device cuda:0
python3 scripts/experiments/rerun_all_v2.py --gpu-batch 1 --device cuda:1
python3 scripts/experiments/rerun_all_v2.py --gpu-batch 2 --device cuda:2
python3 scripts/experiments/rerun_all_v2.py --gpu-batch 3 --device cuda:3
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- See `requirements.txt` for full dependencies

## Citation

If you use this code, please cite the accompanying paper (preprint forthcoming).
