#!/usr/bin/env python3
"""Lean per-shard worker: trains one (alpha, role, nterms, seed-subset) and saves
per-timepoint L2 errors. Reuses the validated build_config / build_model /
evaluate_model from run_fdr_fidelity_timepoints.py so the training/eval path is
identical to the canonical fidelity sweep."""
import argparse, json, time, numpy as np, torch, importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "base", "scripts/experiments/run_fdr_fidelity_timepoints.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from mf_fpinn.solvers.fdr_solver import (
    FDRLaplaceSolver, generate_low_fidelity_data, compute_lf_error)
from mf_fpinn.training.fdr_trainer import FDRTrainer
from mf_fpinn.experiments.fdr_configs import PHYSICS, BC, REFERENCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device")
    p.add_argument("--alpha", type=float)
    p.add_argument("--role")               # "vanilla" or "mf"
    p.add_argument("--nterms", type=int, default=0)
    p.add_argument("--seeds")              # comma-separated, e.g. "0,1,2,3,4"
    p.add_argument("--out")
    a = p.parse_args()

    dev = torch.device(a.device)
    alpha = a.alpha
    seeds = [int(s) for s in a.seeds.split(",")]
    ref = dict(np.load(f"results/fdr/reference_alpha{alpha}.npz", allow_pickle=True))

    solver = FDRLaplaceSolver(
        alpha=alpha, D=PHYSICS["D"], kappa=PHYSICS["kappa"], L=PHYSICS["L"],
        bc_type=BC["bc_type"], bc_params=BC["bc_params"],
        N_terms=REFERENCE["N_terms"], precision=REFERENCE["precision"])

    lf_nt = 12 if a.role == "vanilla" else a.nterms
    lf = generate_low_fidelity_data(
        alpha=alpha, D=PHYSICS["D"], kappa=PHYSICS["kappa"], L=PHYSICS["L"],
        bc_type=BC["bc_type"], bc_params=BC["bc_params"],
        N_LF=50, seed=42, lf_N_terms=lf_nt, lf_precision=20)

    lf_err = None
    if a.role == "mf":
        lf_err = compute_lf_error(
            solver, lf_N_terms=a.nterms, lf_precision=20, N_test=50)["l2_relative"] * 100

    l2, tp = [], []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        cfg = base.build_config(alpha, n_col=100)
        model = base.build_model(cfg)
        tr = FDRTrainer(model, cfg, lf, device=dev)
        t0 = time.time()
        (tr.train_vanilla if a.role == "vanilla" else tr.train_full)(verbose=False)
        m, tps = base.evaluate_model(model, ref, dev)
        l2.append(m * 100)
        tp.append(tps)
        print(f"{a.role} a{alpha} N{a.nterms} seed{s}: {m*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

    out = {"alpha": alpha, "role": a.role, "nterms": a.nterms, "seeds": seeds,
           "l2_values": l2, "tp_values": tp, "tp_times": [0.05, 0.15, 0.25, 0.4],
           "lf_err": lf_err}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"SHARD DONE -> {a.out}", flush=True)


main()
