#!/usr/bin/env python3
"""
Noise robustness sweep for the inverse problem.

Run:  python3 run_fdr_noise_sweep.py [--device cuda] [--seeds 5]
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from run_fdr_inverse import run_inverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--true-alpha', type=float, default=0.7)
    parser.add_argument('--true-kappa', type=float, default=1.5)
    parser.add_argument('--pretrain', type=int, default=5000)
    parser.add_argument('--inverse-epochs', type=int, default=20000)
    parser.add_argument('--noise-levels', type=float, nargs='+',
                        default=[0.01, 0.02, 0.05, 0.10])
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'#'*70}")
    print(f"  Noise Robustness Sweep")
    print(f"  True: α={args.true_alpha}, κ={args.true_kappa}")
    print(f"  Noise levels: {args.noise_levels}")
    print(f"  Seeds: {args.seeds}")
    print(f"{'#'*70}\n")

    out_dir = Path('results/fdr/inverse')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_noise_results = {}

    for noise in args.noise_levels:
        print(f"\n{'='*60}")
        print(f"  Noise level: {noise*100:.0f}%")
        print(f"{'='*60}")

        noise_results = []
        for seed in range(args.seeds):
            print(f"\n  --- Seed {seed} ---")
            result = run_inverse(
                true_alpha=args.true_alpha,
                true_kappa=args.true_kappa,
                D=1.0, L=1.0, T=1.0,
                bc_type='pulse', bc_params={'a': 0.5},
                init_alpha=0.5, init_kappa=1.0,
                noise_level=noise,
                device=device, seed=seed,
                n_lf=50,
                pretrain_epochs=args.pretrain,
                inverse_epochs=args.inverse_epochs,
                verbose=False,
            )
            noise_results.append({k: v for k, v in result.items() if k != 'history'})
            print(f"    α={result['est_alpha']:.4f} (err={result['alpha_error_pct']:.2f}%)  "
                  f"κ={result['est_kappa']:.4f} (err={result['kappa_error_pct']:.2f}%)")

        all_noise_results[str(noise)] = noise_results

        # Noise level summary
        alpha_errs = [r['alpha_error_pct'] for r in noise_results]
        kappa_errs = [r['kappa_error_pct'] for r in noise_results]
        print(f"\n  Summary ({noise*100:.0f}% noise):")
        print(f"    α error: {np.mean(alpha_errs):.2f}% ± {np.std(alpha_errs):.2f}%")
        print(f"    κ error: {np.mean(kappa_errs):.2f}% ± {np.std(kappa_errs):.2f}%")

    # Final summary table
    print(f"\n\n{'#'*70}")
    print(f"  NOISE ROBUSTNESS SUMMARY")
    print(f"{'#'*70}")
    print(f"\n  {'Noise':>8s}  {'α error (%)':>15s}  {'κ error (%)':>15s}")
    print(f"  {'-'*45}")
    for noise in args.noise_levels:
        results = all_noise_results[str(noise)]
        ae = [r['alpha_error_pct'] for r in results]
        ke = [r['kappa_error_pct'] for r in results]
        print(f"  {noise*100:7.0f}%  {np.mean(ae):7.2f} ± {np.std(ae):5.2f}  "
              f"{np.mean(ke):7.2f} ± {np.std(ke):5.2f}")

    # Save
    json_path = out_dir / f'noise_sweep_alpha{args.true_alpha}.json'
    with open(json_path, 'w') as f:
        json.dump(all_noise_results, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
