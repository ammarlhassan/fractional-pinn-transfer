#!/usr/bin/env python3
"""Fan-out dispatcher: build 24 shards (12 configs x two 5-seed halves), assign
3 shards per GPU across 8 GPUs, write per-GPU driver scripts, and nohup-launch them."""
import os, stat, subprocess
from pathlib import Path

REPO = "/home/skyvision/Ammar/fractional-pinn-transfer"
SHARD_DIR = "results/fdr/tp_shards"
LOG_DIR = "logs_timepoints/fanout"

configs = [("vanilla", 0.5, 0), ("vanilla", 1.0, 0)]
configs += [("mf", 0.5, n) for n in (8, 12, 15, 20)]
configs += [("mf", 1.0, n) for n in (5, 8, 10, 12, 15, 20)]

seed_halves = ["0,1,2,3,4", "5,6,7,8,9"]

# 24 shards
shards = []
for role, alpha, nt in configs:
    for sh in seed_halves:
        first = sh.split(",")[0]
        out = f"{SHARD_DIR}/{role}_a{alpha}_N{nt}_s{first}.json"
        shards.append((role, alpha, nt, sh, out))

assert len(shards) == 24, len(shards)

Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# assign 3 shards per GPU (gpu = i // 3)
drivers = []
for gpu in range(8):
    lines = ["#!/bin/bash", f"export PYTHONPATH={REPO}", f"cd {REPO}"]
    for i in range(gpu * 3, gpu * 3 + 3):
        role, alpha, nt, sh, out = shards[i]
        lines.append(
            f'python3 scripts/experiments/tp_worker.py --device cuda:{gpu} '
            f'--alpha {alpha} --role {role} --nterms {nt} --seeds "{sh}" --out {out}')
    lines.append(f'echo "GPU {gpu} ALL SHARDS DONE"')
    dpath = f"{LOG_DIR}/driver_gpu{gpu}.sh"
    with open(dpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(dpath, 0o755)
    drivers.append(dpath)

# launch each driver detached
pids = []
for gpu, dpath in enumerate(drivers):
    log = f"{LOG_DIR}/gpu{gpu}.log"
    p = subprocess.Popen(["nohup", "bash", dpath],
                         stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    pids.append((gpu, p.pid))

print("LAUNCHED 8 drivers (3 shards each, 24 total):")
for gpu, pid in pids:
    print(f"  GPU {gpu}: PID {pid}")
print(f"Shards -> {SHARD_DIR}/  Logs -> {LOG_DIR}/")
