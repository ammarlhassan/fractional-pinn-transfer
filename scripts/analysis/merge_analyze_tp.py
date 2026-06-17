#!/usr/bin/env python3
"""Merge per-timepoint shards into per-alpha JSONs and analyze the time-localization
of the MF advantage. Run after pulling results/fdr/tp_shards/ locally."""
import json, glob, sys
from collections import defaultdict
import numpy as np
from scipy import stats

SHARD_DIR = sys.argv[1] if len(sys.argv) > 1 else "tp_shards_pulled"
TLAB = ["t=0.05", "t=0.15", "t=0.25", "t=0.40"]

# group shards by (alpha, role, nterms); merge the two 5-seed halves in seed order
buckets = defaultdict(list)
for f in glob.glob(f"{SHARD_DIR}/*.json"):
    d = json.load(open(f))
    buckets[(d["alpha"], d["role"], d["nterms"])].append(d)

merged = {}  # (alpha, role, nterms) -> dict(l2[10], tp[10][4], lf_err)
for key, shards in buckets.items():
    pairs = []
    for sh in shards:
        for s, l2, tp in zip(sh["seeds"], sh["l2_values"], sh["tp_values"]):
            pairs.append((s, l2, tp))
    pairs.sort(key=lambda x: x[0])  # seed order 0..9
    merged[key] = {
        "l2": [p[1] for p in pairs],
        "tp": [p[2] for p in pairs],
        "lf_err": shards[0]["lf_err"],
    }

for alpha in (0.5, 1.0):
    van = merged[(alpha, "vanilla", 0)]
    van_l2 = np.array(van["l2"]); van_tp = np.array(van["tp"])  # (10,4)
    print("=" * 70)
    print(f"ALPHA = {alpha}   (vanilla mean {van_l2.mean():.2f}% +/- {van_l2.std():.2f}%)")
    print("=" * 70)

    # build merged JSON matching canonical format + tp
    out = {"vanilla": {"l2_values": van["l2"], "l2_mean": float(van_l2.mean()),
                       "l2_std": float(van_l2.std()), "tp_values": van["tp"],
                       "tp_times": [0.05, 0.15, 0.25, 0.4]}}

    nts = sorted(nt for (a, r, nt) in merged if a == alpha and r == "mf")
    print(f"\n{'Nterms':>6}{'LF%':>7}{'MF%':>8}{'ratio':>7}{'wilcox_p':>9}   per-timepoint MF/Van ratio")
    for nt in nts:
        mf = merged[(alpha, "mf", nt)]
        mf_l2 = np.array(mf["l2"]); mf_tp = np.array(mf["tp"])  # (10,4)
        ratio = van_l2.mean() / mf_l2.mean()
        try:
            _, p = stats.wilcoxon(mf_l2, van_l2)
        except ValueError:
            p = float("nan")
        # per-timepoint ratios (mean across seeds)
        tp_ratio = van_tp.mean(0) / mf_tp.mean(0)
        rstr = "  ".join(f"{TLAB[i].split('=')[1]}:{tp_ratio[i]:.2f}" for i in range(4))
        print(f"{nt:>6}{mf['lf_err']:>7.1f}{mf_l2.mean():>8.2f}{ratio:>7.2f}{p:>9.4f}   {rstr}")
        out[f"MF_Nterms{nt}"] = {"l2_values": mf["l2"], "l2_mean": float(mf_l2.mean()),
                                 "l2_std": float(mf_l2.std()), "lf_err": mf["lf_err"],
                                 "ratio": float(ratio), "wilcoxon_p": float(p),
                                 "tp_values": mf["tp"], "tp_times": [0.05, 0.15, 0.25, 0.4]}

    # KEY ANALYSIS: time-localization at the best LF config
    best = max(nts)
    mf_tp = np.array(merged[(alpha, "mf", best)]["tp"])
    print(f"\n  --- Time-localization (best LF, N_terms={best}) ---")
    print(f"  {'time':8}{'Van%':>8}{'MF%':>8}{'ratio':>8}")
    for i in range(4):
        v = van_tp[:, i].mean(); m = mf_tp[:, i].mean()
        print(f"  {TLAB[i]:8}{v:8.2f}{m:8.2f}{v/m:8.2f}")
    # per-seed: MF worse than Vanilla at late time?
    late_worse = int(np.sum(mf_tp[:, 3] > van_tp[:, 3]))
    early_better = int(np.sum(mf_tp[:, 0] < van_tp[:, 0]))
    print(f"  MF worse than Vanilla at LATE time (t=0.40): {late_worse}/10 seeds")
    print(f"  MF better than Vanilla at EARLY time (t=0.05): {early_better}/10 seeds")

    json.dump(out, open(f"fidelity_10seed_timepoints_alpha{alpha}_MERGED.json", "w"), indent=2)
    print(f"\n  merged JSON -> fidelity_10seed_timepoints_alpha{alpha}_MERGED.json\n")
