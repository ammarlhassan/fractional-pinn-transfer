"""
Optuna HPO for the fractional diffusion-reaction PINN.

Searches over architecture, optimizer, and loss weight hyperparameters.
Supports both vanilla (M2) and MF-fPINN (M4) configurations.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from pathlib import Path

from ..models.fdr_pinn import FDRNet
from ..evaluation.metrics import compute_l2_relative_error
from .fdr_trainer import FDRTrainer


def sample_hyperparameters(trial, use_transfer=True):
    """Sample hyperparameters from the search space."""
    config = {
        # Optimizer
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'beta1': 0.9, 'beta2': 0.999,

        # Architecture
        'n_layers': trial.suggest_int('n_layers', 3, 6),
        'n_neurons': trial.suggest_categorical('n_neurons', [64, 128, 256]),
        'activation': trial.suggest_categorical('activation', ['tanh', 'swish', 'gelu']),
        'fourier_features': 64,
        'fourier_sigma': trial.suggest_float('fourier_sigma', 2.0, 10.0),

        # Training
        'N_collocation': trial.suggest_categorical('N_collocation', [1000, 2000, 3000]),
        'lr_schedule': 'cosine',
        'grad_clip': trial.suggest_float('grad_clip', 1.0, 10.0),
        'phase1_lbfgs': True,
        'phase1_lbfgs_steps': 50,

        # GL parameters for fractional derivative
        'gl_h': 0.01,
        'gl_N_memory': 100,
    }

    if use_transfer:
        config['phase1_epochs'] = trial.suggest_int('phase1_epochs', 3000, 10000, step=1000)
        config['phase2_epochs'] = trial.suggest_int('phase2_epochs', 5000, 20000, step=1000)
        config['phase2_lr'] = trial.suggest_float('phase2_lr', 1e-5, 5e-4, log=True)
        config['w_data'] = trial.suggest_float('w_data', 5.0, 200.0, log=True)
        config['w_pde'] = trial.suggest_float('w_pde', 0.1, 10.0, log=True)
        config['w_bc'] = trial.suggest_float('w_bc', 0.1, 10.0, log=True)
        config['w_ic'] = trial.suggest_float('w_ic', 0.1, 10.0, log=True)
        config['pde_ramp_epochs'] = trial.suggest_int('pde_ramp_epochs', 1000, 5000, step=500)
    else:
        config['vanilla_epochs'] = trial.suggest_int('vanilla_epochs', 10000, 30000, step=2000)
        config['w_pde'] = trial.suggest_float('w_pde', 0.1, 10.0, log=True)
        config['w_bc'] = trial.suggest_float('w_bc', 0.1, 10.0, log=True)
        config['w_ic'] = trial.suggest_float('w_ic', 0.1, 10.0, log=True)
        config['pde_ramp_epochs'] = trial.suggest_int('pde_ramp_epochs', 2000, 8000, step=1000)

    return config


def create_objective(lf_data, ref_data, physics, bc_cfg,
                     use_transfer=True, device=torch.device('cpu')):
    """
    Create an Optuna objective function.

    Parameters
    ----------
    lf_data : dict
        Low-fidelity training data {'x', 't', 'u'}.
    ref_data : dict
        Reference data with 'x', 'u', 't' for evaluation.
    physics : dict
        {'alpha', 'D', 'kappa', 'L', 'T'}.
    bc_cfg : dict
        {'bc_type', 'bc_params'}.
    use_transfer : bool
    device : torch.device

    Returns
    -------
    objective : callable
    """
    # Pre-compute evaluation tensors
    x_eval = torch.tensor(ref_data['x'], dtype=torch.float32, device=device).unsqueeze(1)
    # Use middle time snapshot for evaluation
    t_idx = len(ref_data['t']) // 2
    t_eval_val = float(ref_data['t'][t_idx])
    t_eval = torch.full_like(x_eval, t_eval_val)
    u_eval_ref = ref_data['u'][:, t_idx]

    def objective(trial):
        config = sample_hyperparameters(trial, use_transfer=use_transfer)
        config.update(physics)
        config.update(bc_cfg)

        model = FDRNet(
            n_layers=config['n_layers'],
            n_neurons=config['n_neurons'],
            activation=config['activation'],
            L=config['L'], T=config['T'],
            hard_bc=True,
            bc_type=config['bc_type'],
            bc_params=config['bc_params'],
            fourier_features=config['fourier_features'],
            fourier_sigma=config['fourier_sigma'],
        )

        trainer = FDRTrainer(model, config, lf_data, device=device)

        try:
            if use_transfer:
                trainer.train_full(verbose=False)
            else:
                trainer.train_vanilla(verbose=False)
        except RuntimeError as e:
            if 'nan' in str(e).lower() or 'out of memory' in str(e).lower():
                raise optuna.TrialPruned()
            raise

        model.eval()
        with torch.no_grad():
            u_pred = model(x_eval, t_eval).cpu().numpy().flatten()

        l2_error = compute_l2_relative_error(u_pred, u_eval_ref)

        trial.report(l2_error, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return l2_error

    return objective


class FDRHPORunner:
    """
    Orchestrates HPO for one experimental configuration.

    Parameters
    ----------
    lf_data : dict
    ref_data : dict
    physics : dict
        {'alpha', 'D', 'kappa', 'L', 'T'}.
    bc_cfg : dict
    device : torch.device
    study_name : str
    """

    def __init__(self, lf_data, ref_data, physics, bc_cfg,
                 device=torch.device('cpu'), study_name='fdr_hpo'):
        self.lf_data = lf_data
        self.ref_data = ref_data
        self.physics = physics
        self.bc_cfg = bc_cfg
        self.device = device
        self.study_name = study_name

    def run_study(self, n_trials=50, use_transfer=True, seed=42):
        """Run an Optuna HPO study."""
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )

        objective = create_objective(
            self.lf_data, self.ref_data, self.physics, self.bc_cfg,
            use_transfer=use_transfer, device=self.device,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nHPO Complete: {self.study_name}")
        print(f"  Best trial: #{study.best_trial.number}")
        print(f"  Best L2 error: {study.best_value:.6e}")
        print(f"  Best params:")
        for key, val in study.best_params.items():
            print(f"    {key}: {val}")

        return study

    def analyze_importance(self, study):
        """Compute hyperparameter importance via fANOVA."""
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nHyperparameter Importance (fANOVA):")
            for param, imp in sorted(importance.items(),
                                      key=lambda x: x[1], reverse=True):
                bar = '█' * int(imp * 50)
                print(f"  {param:20s}: {imp:.4f}  {bar}")
            return importance
        except Exception as e:
            print(f"  Could not compute importance: {e}")
            return {}

    def run_best_multi_seed(self, study, n_seeds=10, use_transfer=True):
        """Retrain best config with multiple seeds for statistical evaluation."""
        best_params = study.best_params
        config = sample_hyperparameters(study.best_trial, use_transfer=use_transfer)
        config.update(self.physics)
        config.update(self.bc_cfg)

        results = []
        x_eval = torch.tensor(
            self.ref_data['x'], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FDRNet(
                n_layers=config['n_layers'],
                n_neurons=config['n_neurons'],
                activation=config['activation'],
                L=config['L'], T=config['T'],
                hard_bc=True,
                bc_type=config['bc_type'],
                bc_params=config['bc_params'],
                fourier_features=config['fourier_features'],
                fourier_sigma=config['fourier_sigma'],
            )

            trainer = FDRTrainer(model, config, self.lf_data, self.device)
            if use_transfer:
                trainer.train_full(verbose=False)
            else:
                trainer.train_vanilla(verbose=False)

            model.eval()
            seed_results = {'seed': seed, 'wall_time': trainer.history['wall_time'][-1]}

            with torch.no_grad():
                for j, t_val in enumerate(self.ref_data['t']):
                    t_t = torch.full_like(x_eval, float(t_val))
                    u_pred = model(x_eval, t_t).cpu().numpy().flatten()
                    u_ref = self.ref_data['u'][:, j]
                    l2 = compute_l2_relative_error(u_pred, u_ref)
                    seed_results[f't{float(t_val):.2f}_u_l2'] = l2

            results.append(seed_results)
            l2_avg = np.mean([v for k, v in seed_results.items() if k.endswith('_l2')])
            print(f"  Seed {seed}: avg L2 = {l2_avg:.4f}")

        # Summary
        all_l2 = []
        for r in results:
            all_l2.extend([v for k, v in r.items() if k.endswith('_l2')])
        print(f"\nSummary ({n_seeds} seeds): "
              f"L2 = {np.mean(all_l2):.4f} ± {np.std(all_l2):.4f}")

        return results
