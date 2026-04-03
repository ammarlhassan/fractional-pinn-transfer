#!/usr/bin/env python3
"""
Expanded inverse problem: test multiple (α, κ) pairs to show generalizability.

Uses the scale-normalized GL residual fix.
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
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Test cases: different (α, κ) pairs spanning the parameter space
    test_cases = [
        {'true_alpha': 0.3, 'true_kappa': 0.5, 'init_alpha': 0.5, 'init_kappa': 1.0},
        {'true_alpha': 0.5, 'true_kappa': 1.0, 'init_alpha': 0.7, 'init_kappa': 1.5},
        {'true_alpha': 0.7, 'true_kappa': 1.5, 'init_alpha': 0.5, 'init_kappa': 1.0},
        {'true_alpha': 0.9, 'true_kappa': 2.0, 'init_alpha': 0.5, 'init_kappa': 1.0},
    ]

    out_dir = Path('results/fdr/inverse')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_case_results = {}

    for case in test_cases:
        ta, tk = case['true_alpha'], case['true_kappa']
        ia, ik = case['init_alpha'], case['init_kappa']

        print(f"\n{'#'*70}")
        print(f"  TEST CASE: α={ta}, κ={tk}  (init: α={ia}, κ={ik})")
        print(f"  Noise: {args.noise*100:.0f}%, Seeds: {args.seeds}")
        print(f"{'#'*70}")

        case_results = []
        for seed in range(args.seeds):
            result = run_inverse(
                true_alpha=ta, true_kappa=tk,
                D=1.0, L=1.0, T=1.0,
                bc_type='pulse', bc_params={'a': 0.5},
                init_alpha=ia, init_kappa=ik,
                noise_level=args.noise,
                device=device, seed=seed,
                n_lf=50, pretrain_epochs=5000,
                inverse_epochs=20000,
                verbose=(seed == 0),
            )
            case_results.append(result)
            print(f"  Seed {seed}: α={result['est_alpha']:.4f} "
                  f"({result['alpha_error_pct']:.1f}%), "
                  f"κ={result['est_kappa']:.4f} "
                  f"({result['kappa_error_pct']:.1f}%)")

        alphas = [r['est_alpha'] for r in case_results]
        kappas = [r['est_kappa'] for r in case_results]

        key = f'alpha{ta}_kappa{tk}'
        all_case_results[key] = {
            'true_alpha': ta, 'true_kappa': tk,
            'est_alpha_mean': float(np.mean(alphas)),
            'est_alpha_std': float(np.std(alphas)),
            'est_kappa_mean': float(np.mean(kappas)),
            'est_kappa_std': float(np.std(kappas)),
            'alpha_err_pct': float(np.mean([r['alpha_error_pct'] for r in case_results])),
            'kappa_err_pct': float(np.mean([r['kappa_error_pct'] for r in case_results])),
            'results': [{k: v for k, v in r.items() if k != 'history'}
                       for r in case_results],
        }

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  EXPANDED INVERSE PROBLEM SUMMARY")
    print(f"  Noise: {args.noise*100:.0f}%, Seeds: {args.seeds}")
    print(f"{'='*70}")
    print(f"  {'True α':>8} {'True κ':>8} {'Est α':>12} {'Est κ':>12} {'α err%':>8} {'κ err%':>8}")
    print(f"  {'-'*60}")
    for key, res in all_case_results.items():
        print(f"  {res['true_alpha']:>8.1f} {res['true_kappa']:>8.1f} "
              f"{res['est_alpha_mean']:>8.3f}±{res['est_alpha_std']:.3f} "
              f"{res['est_kappa_mean']:>8.3f}±{res['est_kappa_std']:.3f} "
              f"{res['alpha_err_pct']:>7.1f}% {res['kappa_err_pct']:>7.1f}%")

    save_path = out_dir / f'inverse_sweep_noise{args.noise}.json'
    with open(save_path, 'w') as f:
        json.dump(all_case_results, f, indent=2)
    print(f"\n  Saved to {save_path}")


if __name__ == '__main__':
    main()
