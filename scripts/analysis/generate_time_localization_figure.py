#!/usr/bin/env python3
"""Generate the time-localization figure showing per-timepoint MF vs Vanilla
error for both alpha values, demonstrating the late-time reversal at alpha=1.0
that GL memory eliminates at alpha=0.5."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

TLAB = ["t = 0.05", "t = 0.15", "t = 0.25", "t = 0.40"]
TIMES = [0.05, 0.15, 0.25, 0.40]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

for ax, alpha in zip(axes, [0.5, 1.0]):
    d = json.load(open(f"fidelity_10seed_timepoints_alpha{alpha}_MERGED.json"))
    van_tp = np.array(d["vanilla"]["tp_values"])             # (10, 4)
    best = max(int(k.replace("MF_Nterms", ""))
               for k in d if k.startswith("MF_Nterms"))
    mf = d[f"MF_Nterms{best}"]
    mf_tp = np.array(mf["tp_values"])                         # (10, 4)
    van_mean = van_tp.mean(0); van_std = van_tp.std(0)
    mf_mean  = mf_tp.mean(0);  mf_std  = mf_tp.std(0)

    x = np.arange(4)
    w = 0.36
    bv = ax.bar(x - w/2, van_mean, w, yerr=van_std, capsize=3,
                color="#888888", edgecolor="black", linewidth=0.6,
                label="Vanilla")
    bm = ax.bar(x + w/2, mf_mean, w, yerr=mf_std, capsize=3,
                color="#1F77B4", edgecolor="black", linewidth=0.6,
                label=f"MF-fPINN ($N_{{\\mathrm{{terms}}}}={best}$)")

    # ratio annotations over each pair
    for i in range(4):
        r = van_mean[i] / mf_mean[i]
        ymax = max(van_mean[i] + van_std[i], mf_mean[i] + mf_std[i])
        clr = "#0B6E2D" if r > 1.0 else "#A1232B"
        ax.text(i, ymax * 1.04, f"{r:.1f}×", ha="center", va="bottom",
                fontsize=9, color=clr, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TLAB)
    ax.set_xlabel("evaluation time")
    ax.set_ylabel(r"relative $L^2$ error (\%)")
    title_extra = ("GL memory propagates corrections backward"
                   if alpha == 0.5
                   else "no GL memory: late-time error grows")
    ax.set_title(rf"(${chr(97 + (0 if alpha==0.5 else 1)) }$)  $\alpha = {alpha}$ — {title_extra}",
                 loc="left", fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right" if alpha == 1.0 else "upper left", frameon=False, fontsize=9)

fig.suptitle("Per-timepoint relative $L^2$ error: the MF advantage is time-localized",
             fontsize=11.5, y=1.02)
fig.tight_layout()
out = "figures/time_localization.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"WROTE {out}")
