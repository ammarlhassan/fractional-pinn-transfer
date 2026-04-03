#!/usr/bin/env python3
"""Generate ALL new table values from rerun JSONs for paper update.

Produces exact LaTeX rows ready to paste, plus identifies text values that need changing.
"""
import json, numpy as np, sys, os
from pathlib import Path
from scipy.stats import wilcoxon

RERUN = Path('results/rerun')

def load(name):
    p = RERUN / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def s(vals, ddof=1):
    """mean ± std"""
    return np.mean(vals), np.std(vals, ddof=ddof)

def fmt(m, sd, bold=False):
    """Format as $m \\pm sd$"""
    if bold:
        return f"$\\mathbf{{{m:.2f} \\pm {sd:.2f}}}$"
    return f"${m:.2f} \\pm {sd:.2f}$"

def ratio_p_d(van, mf):
    """Compute ratio, p-value, Cohen's d"""
    vm, vs = s(van)
    mm, ms = s(mf)
    r = vm / mm if mm > 0 else float('inf')
    try:
        _, p = wilcoxon(van, mf)
    except:
        p = float('nan')
    # Cohen's dz (paired) 
    diffs = np.array(van) - np.array(mf)
    dz = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0
    return r, p, dz

def bold_if(val, cond):
    if cond:
        return f"\\mathbf{{{val}}}"
    return val

# ─── FIDELITY SWEEP α=0.5 (tab:fidelity_sweep_a05) ───
print("="*80)
print("tab:fidelity_sweep_a05 (α=0.5, N_col=100, 10 seeds)")
print("="*80)
d = load('fidelity_10seed_alpha0.5.json')
if d:
    van = d['vanilla']['l2_values']
    vm, vs = s(van)
    print(f"% Vanilla: {vm:.2f}±{vs:.2f}% (n={len(van)})")
    for nt in [8, 12, 15, 20]:
        k = f'MF_Nterms{nt}'
        mf = d[k]['l2_values']
        mm, ms = s(mf)
        r, p, dz = ratio_p_d(van, mf)
        sig = p < 0.05
        rstr = f"{r:.2f}\\times" 
        pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
        row = f"{nt} & ${mm:.2f} \\pm {ms:.2f}$ & ${vm:.2f} \\pm {vs:.2f}$ & "
        if sig:
            row += f"$\\mathbf{{{rstr}}}$ & $\\mathbf{{{pstr}}}$"
        else:
            row += f"${rstr}$ & ${pstr}$"
        row += " \\\\"
        print(row)
    print()

# ─── FIDELITY SWEEP α=1.0 (tab:fidelity_sweep) ───
print("="*80)
print("tab:fidelity_sweep (α=1.0, N_col=100, 10 seeds)")
print("="*80)
d = load('fidelity_10seed_alpha1.0.json')
if d:
    van = d['vanilla']['l2_values']
    vm, vs = s(van)
    print(f"% Vanilla: {vm:.2f}±{vs:.2f}% (n={len(van)})")
    for nt in [5, 8, 10, 12, 15, 20]:
        k = f'MF_Nterms{nt}'
        mf = d[k]['l2_values']
        mm, ms = s(mf)
        r, p, dz = ratio_p_d(van, mf)
        
        # Get LF error from data if available, else skip
        lf_err = d[k].get('lf_error', '?')
        
        sig = p < 0.05
        rstr = f"{r:.2f}\\times"
        pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
        row = f"{nt} & ${mm:.2f} \\pm {ms:.2f}$ & ${vm:.2f} \\pm {vs:.2f}$ & "
        if sig:
            row += f"$\\mathbf{{{rstr}}}$ & $\\mathbf{{{pstr}}}$"
        else:
            row += f"${rstr}$ & ${pstr}$"
        row += " \\\\"
        print(row)
    print()

# ─── FIDELITY CROSS (tab:fidelity_cross) ───
print("="*80)
print("tab:fidelity_cross (N_col=100)")
print("="*80)
d05 = load('fidelity_10seed_alpha0.5.json')
d10 = load('fidelity_10seed_alpha1.0.json')
if d05 and d10:
    for alpha_label, d in [('0.5', d05), ('1.0', d10)]:
        van = d['vanilla']['l2_values']
        vm, vs = s(van)
        for nt, ntk in [(8, 'MF_Nterms8'), (12, 'MF_Nterms12')]:
            mf = d[ntk]['l2_values']
            mm, ms = s(mf)
            r = vm / mm if mm > 0 else float('inf')
            print(f"α={alpha_label}, N_terms={nt}: MF ${mm:.2f} \\pm {ms:.2f}$ & Van ${vm:.2f} \\pm {vs:.2f}$ & ${r:.2f}\\times$")
    print()

# ─── N_COL BUDGET (tab:ncol) ───
print("="*80)
print("tab:ncol")
print("="*80)
for alpha, fname in [('0.5', 'ncol_alpha0.5_nterms12.json'),
                      ('0.7', 'ncol_alpha0.7_nterms12.json'),
                      ('1.0', 'ncol_alpha1.0_nterms12.json')]:
    d = load(fname)
    if not d:
        print(f"% α={alpha}: NOT YET AVAILABLE")
        continue
    for nc in [100, 200, 500]:
        k = f'N_col={nc}'
        if k not in d:
            continue
        van = d[k]['vanilla']['l2_values']
        mf = d[k]['mf']['l2_values']
        vp = d[k]['van50']['l2_values']
        vm, vs = s(van)
        mm, ms = s(mf)
        vpm, vps = s(vp)
        r, p, dz = ratio_p_d(van, mf)
        sig = p < 0.05
        n = len(van)
        rstr = f"{r:.2f}\\times"
        pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
        row = f"% α={alpha}, N_col={nc}: "
        row += f"Van {vm:.2f}±{vs:.2f} | MF {mm:.2f}±{ms:.2f} | Van+50 {vpm:.2f}±{vps:.2f} | {rstr} p={pstr} (n={n})"
        print(row)
    print()

# ─── GL ABLATION (tab:gl_ablation) ───
print("="*80)
print("tab:gl_ablation (α=0.5, N_col=100, N_terms=15, 10 seeds)")
print("="*80)
d = load('gl_ablation_alpha0.5.json')
if d:
    for gl in [2, 5, 10, 20, 50, 100]:
        k = f'gl_mem={gl}'
        van = d[k]['vanilla']['l2_values']
        mf = d[k]['mf']['l2_values']
        vm, vs = s(van)
        mm, ms = s(mf)
        r, p, dz = ratio_p_d(van, mf)
        sig = p < 0.05
        rstr = f"{r:.2f}\\times"
        pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
        row = f"{gl} & ${vm:.2f} \\pm {vs:.2f}$ & ${mm:.2f} \\pm {ms:.2f}$ & "
        if sig:
            row += f"$\\mathbf{{{rstr}}}$ & $\\mathbf{{{pstr}}}$ & ${dz:.2f}$"
        else:
            row += f"${rstr}$ & ${pstr}$ & ${dz:.2f}$"
        row += " \\\\"
        print(row)
    print()

# ─── NOISE (tab:noise) ───
print("="*80)
print("tab:noise (α=1.0, N_col=100, N_terms=12, 5 seeds)")
print("="*80)
d = load('noise_alpha1.0.json')
if d:
    van = d['vanilla']['l2_values']
    vm, vs = s(van)
    n = len(van)
    print(f"% Vanilla: {vm:.2f}±{vs:.2f}% (n={n})")
    for pct in [0, 1, 5, 10, 20]:
        k = f'noise_{pct}pct'
        mf = d[k]['l2_values']
        mm, ms = s(mf)
        # Use matching number of seeds
        van_use = van[:len(mf)]
        r = np.mean(van_use) / mm if mm > 0 else float('inf')
        try:
            _, p = wilcoxon(van_use, mf)
        except:
            p = float('nan')
        rstr = f"{r:.2f}\\times"
        pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
        row = f"{pct} & ${mm:.2f} \\pm {ms:.2f}$ & ${np.mean(van_use):.2f} \\pm {np.std(van_use, ddof=1):.2f}$ & ${rstr}$"
        row += " \\\\"
        print(row)
    print()

# ─── ALPHA SWEEP (tab:alpha_sweep = N_col=100 from ncol files) ───
print("="*80)
print("tab:alpha_sweep (N_col=100, N_terms=12)")
print("="*80)
for alpha, fname in [('0.5', 'ncol_alpha0.5_nterms12.json'),
                      ('0.7', 'ncol_alpha0.7_nterms12.json'),
                      ('1.0', 'ncol_alpha1.0_nterms12.json')]:
    d = load(fname)
    if not d:
        print(f"% α={alpha}: NOT YET AVAILABLE")
        continue
    k = 'N_col=100'
    van = d[k]['vanilla']['l2_values']
    mf = d[k]['mf']['l2_values']
    vm, vs = s(van)
    mm, ms = s(mf)
    r, p, dz = ratio_p_d(van, mf)
    sig = p < 0.05
    rstr = f"{r:.2f}\\times"
    pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
    row = f"{alpha} & ${mm:.2f} \\pm {ms:.2f}$ & ${vm:.2f} \\pm {vs:.2f}$ & "
    if sig:
        row += f"$\\mathbf{{{rstr}}}$ & $\\mathbf{{{pstr}}}$ & ${dz:.2f}$"
    else:
        row += f"${rstr}$ & ${pstr}$ & ${dz:.2f}$"
    row += " \\\\"
    print(row)
print()

# ─── KEY TEXT VALUES ───
print("="*80)
print("KEY TEXT VALUES THAT NEED UPDATING")
print("="*80)

# Load all
f05 = load('fidelity_10seed_alpha0.5.json')
f10 = load('fidelity_10seed_alpha1.0.json')
n05 = load('ncol_alpha0.5_nterms12.json')
n10 = load('ncol_alpha1.0_nterms12.json')
gl = load('gl_ablation_alpha0.5.json')
noise = load('noise_alpha1.0.json')

if f05:
    van = f05['vanilla']['l2_values']
    vm, _ = s(van)
    mf20 = f05['MF_Nterms20']['l2_values']
    mm20, _ = s(mf20)
    r20 = vm / mm20
    _, p20 = wilcoxon(van, mf20)
    print(f"α=0.5 fidelity N_terms=20: {r20:.2f}× (was 3.11×), p={p20:.4f}")
    
    mf15 = f05['MF_Nterms15']['l2_values']
    mm15, _ = s(mf15)
    r15 = vm / mm15
    _, p15 = wilcoxon(van, mf15)
    print(f"α=0.5 fidelity N_terms=15: {r15:.2f}× (was 1.74×), p={p15:.4f}")
    
    mf12 = f05['MF_Nterms12']['l2_values']
    r12 = vm / np.mean(mf12)
    _, p12 = wilcoxon(van, mf12)
    print(f"α=0.5 fidelity N_terms=12: {r12:.2f}× (was 0.95×), p={p12:.4f}")

if f10:
    van = f10['vanilla']['l2_values']
    vm, _ = s(van)
    # Find peak
    best_r, best_nt = 0, 0
    for nt in [5, 8, 10, 12, 15, 20]:
        mf = f10[f'MF_Nterms{nt}']['l2_values']
        r = vm / np.mean(mf) if np.mean(mf) > 0 else 0
        if r > best_r:
            best_r = r
            best_nt = nt
    print(f"α=1.0 fidelity peak: N_terms={best_nt}, {best_r:.2f}× (was peak at N_terms=10, 1.65×)")

if n05:
    van100 = n05['N_col=100']['vanilla']['l2_values']
    vm100, _ = s(van100)
    print(f"α=0.5 N_col=100 Vanilla: {vm100:.2f}% (was 11.39%)")

if n10:
    van100 = n10['N_col=100']['vanilla']['l2_values']
    vm100, _ = s(van100)
    print(f"α=1.0 N_col=100 Vanilla: {vm100:.2f}% (was 30.87%)")

if gl:
    print(f"\nGL ablation trend:")
    for g in [2, 5, 10, 20, 50, 100]:
        k = f'gl_mem={g}'
        van = gl[k]['vanilla']['l2_values']
        mf = gl[k]['mf']['l2_values']
        r = np.mean(van) / np.mean(mf)
        print(f"  gl={g}: {r:.2f}× (was: 2.73, 2.47, 1.65, 1.73, 1.62, 1.74)")

if noise:
    print(f"\nNoise robustness changes:")
    van = noise['vanilla']['l2_values']
    for pct in [0, 1, 5, 10, 20]:
        mf = noise[f'noise_{pct}pct']['l2_values']
        r = np.mean(van[:len(mf)]) / np.mean(mf)
        print(f"  {pct}% noise: {r:.2f}× (was: 1.21, 1.30, 1.09, 0.94, 0.81)")
