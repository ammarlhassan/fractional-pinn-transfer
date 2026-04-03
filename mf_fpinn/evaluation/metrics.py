"""
Evaluation metrics for comparing methods.

Implements:
- L2 relative error
- L∞ error
- Optimal value accuracy
- Convergence speed (epochs to threshold)
"""

import numpy as np
from scipy import stats


def compute_l2_relative_error(predicted, reference):
    """
    L2 relative error: ||pred - ref||_2 / ||ref||_2

    Parameters
    ----------
    predicted : np.ndarray
    reference : np.ndarray

    Returns
    -------
    float
    """
    pred = np.asarray(predicted).flatten()
    ref = np.asarray(reference).flatten()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-15:
        return np.linalg.norm(pred - ref)
    return np.linalg.norm(pred - ref) / ref_norm


def compute_linf_error(predicted, reference):
    """
    L∞ error: max|pred - ref|

    Parameters
    ----------
    predicted, reference : np.ndarray

    Returns
    -------
    float
    """
    return np.max(np.abs(np.asarray(predicted).flatten() -
                         np.asarray(reference).flatten()))


def compute_optimal_value_accuracy(predicted, reference):
    """
    Relative error of the global optimum.

    |max(pred) - max(ref)| / |max(ref)|

    Parameters
    ----------
    predicted, reference : np.ndarray

    Returns
    -------
    float
    """
    pred_opt = np.max(np.abs(np.asarray(predicted).flatten()))
    ref_opt = np.max(np.abs(np.asarray(reference).flatten()))
    if abs(ref_opt) < 1e-15:
        return abs(pred_opt - ref_opt)
    return abs(pred_opt - ref_opt) / abs(ref_opt)


def convergence_speed(history, threshold=0.01, metric_key='data_loss'):
    """
    Number of epochs to reach a given error threshold.

    Parameters
    ----------
    history : dict
        Training history with loss values.
    threshold : float
        Error threshold.
    metric_key : str
        Which metric to check.

    Returns
    -------
    int or None
        Epoch at which threshold was first reached, or None if never reached.
    """
    values = history.get(metric_key, [])
    for i, v in enumerate(values):
        if v <= threshold:
            return i
    return None


def compute_all_metrics(predicted, reference, field_name='theta'):
    """
    Compute all metrics for a single field.

    Parameters
    ----------
    predicted, reference : np.ndarray
    field_name : str

    Returns
    -------
    dict
    """
    return {
        f'{field_name}_l2_rel': compute_l2_relative_error(predicted, reference),
        f'{field_name}_linf': compute_linf_error(predicted, reference),
        f'{field_name}_opt_acc': compute_optimal_value_accuracy(predicted, reference),
    }


def wilcoxon_comparison(results_a, results_b, metric='l2_theta', alpha=0.05):
    """
    Wilcoxon signed-rank test for paired comparison.

    Parameters
    ----------
    results_a, results_b : list of dict
        Per-seed results from two methods.
    metric : str
        Which metric to compare.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: statistic, p_value, significant, effect_size
    """
    a = np.array([r[metric] for r in results_a])
    b = np.array([r[metric] for r in results_b])

    stat, p_value = stats.wilcoxon(a, b, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n = len(a)
    effect_size = 1 - (2 * stat) / (n * (n + 1))

    return {
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
        'std_a': np.std(a),
        'std_b': np.std(b),
    }


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list of float
    alpha : float
        Family-wise error rate.

    Returns
    -------
    corrected_alpha : float
    significant : list of bool
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    significant = [p < corrected_alpha for p in p_values]
    return corrected_alpha, significant


def transfer_efficiency_curve(lf_data_generator, ref_data, model_factory,
                              trainer_factory, N_LF_values, n_seeds=5):
    """
    Compute accuracy vs N_LF (number of Laplace data points) to quantify
    how much transfer learning benefit comes from how many data points.

    Parameters
    ----------
    lf_data_generator : callable(N_LF) -> dict
        Function that generates LF data with given number of points.
    ref_data : dict
        Reference data for evaluation.
    model_factory : callable -> MultiOutputPINN
        Creates a fresh model.
    trainer_factory : callable(model, lf_data) -> Trainer
        Creates a trainer.
    N_LF_values : list of int
        Values of N_LF to test.
    n_seeds : int
        Seeds per N_LF value.

    Returns
    -------
    results : dict mapping N_LF -> list of L2 errors
    """
    results = {}
    for n_lf in N_LF_values:
        print(f"\n--- N_LF = {n_lf} ---")
        errors = []
        for seed in range(n_seeds):
            lf_data = lf_data_generator(n_lf, seed=seed)
            model = model_factory()
            trainer = trainer_factory(model, lf_data)
            trainer.train_full(verbose=False)

            model.eval()
            import torch
            with torch.no_grad():
                x_ref = torch.tensor(
                    ref_data['x'], dtype=torch.float32
                ).unsqueeze(1)
                t_ref = torch.tensor(
                    ref_data['t_eval'], dtype=torch.float32
                ).unsqueeze(1)
                theta_pred, _, _ = model(x_ref, t_ref)

            l2 = compute_l2_relative_error(
                theta_pred.numpy().flatten(), ref_data['theta_eval']
            )
            errors.append(l2)

        results[n_lf] = errors
        print(f"  L2 error: {np.mean(errors):.4e} ± {np.std(errors):.4e}")

    return results
