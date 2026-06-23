"""
Multi-objective Optuna objectives for autoencoderRNN.

Optimizes two objectives simultaneously:
    1. Poisson AICr (minimize) — predictive accuracy penalized for complexity
    2. Loss-LL correlation (maximize) — generalization consistency

Optuna finds the Pareto front: configs where you can't improve one
metric without worsening the other. You pick from that front.

Drop these into optuna_core.py to replace the single-objective versions,
or import them alongside.
"""

import os
import time
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime

import optuna
from run_training import kfold_evaluate


def get_criterion(loss_name):
    from train import MSELoss, smooth_MSELoss
    if loss_name == 'smooth':
        return smooth_MSELoss(alpha=0.05)
    return MSELoss()


# ----------------------------------------------------------------
# Shared: extract both metrics from info_criteria
# ----------------------------------------------------------------

def _extract_metrics(info_criteria, corr_metric='poisson'):
    """
    Extract normalized AICr and loss-LL correlation from info_criteria.
 
    Args:
        info_criteria: dict from kfold_evaluate
        corr_metric: 'poisson' or 'gaussian'
 
    Returns:
        aicr_norm: float, AICr per observation (lower is better)
        loss_ll_corr: float, correlation between fold loss and fold LL
    """
    aicr = info_criteria.get('poisson_aicr', float('nan'))
    if np.isnan(aicr):
        aicr = info_criteria.get('poisson_aic', float('nan'))
 
    n_obs = info_criteria.get('n_observations', 1)
    aicr_norm = aicr / n_obs if not np.isnan(aicr) else float('inf')
 
    if corr_metric == 'gaussian':
        loss_ll_corr = info_criteria.get('loss_gaussian_corr', float('nan'))
    else:
        loss_ll_corr = info_criteria.get('loss_poisson_corr', float('nan'))
 
    return aicr_norm, loss_ll_corr


# ----------------------------------------------------------------
# Single-taste multi-objective
# ----------------------------------------------------------------

def create_multiobjective(prep, device, taste_ind, artifacts_dir,
                          loo_train_steps, loo_patience, corr_metric='poisson'):
    """
    Multi-objective for a single taste.
    Returns (aicr_normalized, -loss_ll_corr).
    Optuna minimizes both, so we negate the correlation.
    """
    def objective(trial):
        from optuna_sweep import define_search_space

        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])
        trial_start = time.time()

        try:
            info_criteria = kfold_evaluate(
                inputs_tensor=prep['inputs_tensor'],
                labels_tensor=prep['labels_tensor'],
                input_size=prep['input_size'],
                hidden_size=params['hidden_size'],
                output_size=prep['output_size'],
                device=device,
                criterion=criterion,
                train_steps=loo_train_steps,
                patience=loo_patience,
                lr=params['lr'],
                rnn_layers=params['rnn_layers'],
                dropout=params['dropout'],
                scaler=prep['scaler'],
                pca_obj=prep['pca_obj'],
                raw_labels_tensor=prep.get('raw_labels_tensor'),
                n_folds=15, # we need to ensure that we cut through the noise-- so this goes up at the cost of time...
                seed=42,
                verbose=False,
                taste_ind=taste_ind,
            )
        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {e}")
            return float('inf'), float('inf')

        aicr_norm, loss_ll_corr = _extract_metrics(info_criteria, corr_metric)
        elapsed = time.time() - trial_start

        # Store everything for later analysis
        trial.set_user_attr('poisson_aicr_norm', aicr_norm)
        trial.set_user_attr('loss_ll_corr', loss_ll_corr)
        trial.set_user_attr('loss_poisson_corr', info_criteria.get('loss_poisson_corr', float('nan')))
        trial.set_user_attr('loss_gaussian_corr', info_criteria.get('loss_gaussian_corr', float('nan')))
        trial.set_user_attr('poisson_aicr_raw', info_criteria.get('poisson_aicr', float('nan')))
        trial.set_user_attr('poisson_aic', info_criteria.get('poisson_aic', float('nan')))
        trial.set_user_attr('gaussian_aicr', info_criteria.get('gaussian_aicr', float('nan')))
        trial.set_user_attr('n_params', info_criteria.get('n_params', 0))
        trial.set_user_attr('n_observations', info_criteria.get('n_observations', 0))
        trial.set_user_attr('time_s', elapsed)

        corr_label = 'Poisson' if corr_metric == 'poisson' else 'Gaussian'
        print(f"  Trial {trial.number}: "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f} | "
              f"AICr/obs={aicr_norm:.6f}, r({corr_label})={loss_ll_corr:.3f} | "
              f"{elapsed:.1f}s")

        # Optuna minimizes both: lower AICr is better, lower -r is better (= higher r)
        neg_corr = -loss_ll_corr if not np.isnan(loss_ll_corr) else float('inf')
        return aicr_norm, neg_corr

    return objective


# ----------------------------------------------------------------
# Multi-taste multi-objective
# ----------------------------------------------------------------

def create_multitaste_multiobjective(all_preps, device, artifacts_dir,
                                     loo_train_steps, loo_patience, corr_metric='poisson'):
    """
    Multi-objective across all tastes for a single dataset.
    Returns (mean_aicr_normalized, mean_neg_correlation).
    """
    def objective(trial):
        from optuna_sweep import define_search_space
        print(f"    DEBUG: info_criteria keys = {list(info_criteria.keys())}")
        print(f"    DEBUG: loss_poisson_corr = {info_criteria.get('loss_poisson_corr', 'MISSING')}")

        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])
        trial_start = time.time()

        taste_aicrs = []
        taste_corrs = []

        for taste_ind in sorted(all_preps):
            prep = all_preps[taste_ind]
            try:
                info_criteria = kfold_evaluate(
                    inputs_tensor=prep['inputs_tensor'],
                    labels_tensor=prep['labels_tensor'],
                    input_size=prep['input_size'],
                    hidden_size=params['hidden_size'],
                    output_size=prep['output_size'],
                    device=device,
                    criterion=criterion,
                    train_steps=loo_train_steps,
                    patience=loo_patience,
                    lr=params['lr'],
                    rnn_layers=params['rnn_layers'],
                    dropout=params['dropout'],
                    scaler=prep['scaler'],
                    pca_obj=prep['pca_obj'],
                    raw_labels_tensor=prep.get('raw_labels_tensor'),
                    n_folds=15,
                    seed=42,
                    verbose=False,
                    taste_ind=taste_ind,
                )
            except Exception as e:
                print(f"  Trial {trial.number} taste {taste_ind} FAILED: {e}")
                return float('inf'), float('inf')

            aicr_norm, loss_ll_corr = _extract_metrics(info_criteria, corr_metric)
            taste_aicrs.append(aicr_norm)
            taste_corrs.append(loss_ll_corr)

            print(f"    Taste {taste_ind}: AICr/obs={aicr_norm:.6f}, r={loss_ll_corr:.3f}")

        mean_aicr = np.mean(taste_aicrs)
        valid_corrs = [c for c in taste_corrs if not np.isnan(c)]
        mean_corr = np.mean(valid_corrs) if valid_corrs else float('nan')
        elapsed = time.time() - trial_start

        trial.set_user_attr('per_taste_aicr_norm', taste_aicrs)
        trial.set_user_attr('per_taste_corr', taste_corrs)
        trial.set_user_attr('mean_aicr_norm', mean_aicr)
        trial.set_user_attr('mean_corr', mean_corr)
        trial.set_user_attr('time_s', elapsed)

        corr_label = 'Poisson' if corr_metric == 'poisson' else 'Gaussian'
        print(f"  Trial {trial.number}: "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f} | "
              f"Mean AICr/obs={mean_aicr:.6f}, Mean r({corr_label})={mean_corr:.3f} | "
              f"{elapsed:.1f}s")

        neg_corr = -mean_corr if not np.isnan(mean_corr) else float('inf')
        return mean_aicr, neg_corr

    return objective


# ----------------------------------------------------------------
# Multi-dataset multi-objective
# ----------------------------------------------------------------
def create_multidataset_multiobjective(all_preps_by_bin, device, artifacts_dir,
                                       loo_train_steps, loo_patience,
                                       corr_metric='poisson'):
    """
    Multi-objective across multiple datasets and bin sizes.

    all_preps_by_bin: dict {bin_size: {ds_name: {taste_ind: prep}}}
        Pre-computed preprocessing for each candidate bin size.
        The trial's sampled bin_size selects which prep set is used.

    Returns (mean_aicr_normalized, mean_neg_correlation).
    """
    def objective(trial):
        from optuna_sweep import define_search_space

        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])

        # Select the pre-binned data matching this trial's bin_size
        bin_size = params['bin_size']
        all_dataset_preps = all_preps_by_bin[bin_size]

        trial_start = time.time()

        all_aicrs = []
        all_corrs = []

        for ds_name in sorted(all_dataset_preps):
            taste_preps = all_dataset_preps[ds_name]
            ds_aicrs = []
            ds_corrs = []

            for taste_ind in sorted(taste_preps):
                prep = taste_preps[taste_ind]
                try:
                    info_criteria = kfold_evaluate(
                        inputs_tensor=prep['inputs_tensor'],
                        labels_tensor=prep['labels_tensor'],
                        input_size=prep['input_size'],
                        hidden_size=params['hidden_size'],
                        output_size=prep['output_size'],
                        device=device,
                        criterion=criterion,
                        train_steps=loo_train_steps,
                        patience=loo_patience,
                        lr=params['lr'],
                        rnn_layers=params['rnn_layers'],
                        dropout=params['dropout'],
                        scaler=prep['scaler'],
                        pca_obj=prep['pca_obj'],
                        raw_labels_tensor=prep.get('raw_labels_tensor'),
                        n_folds=15,
                        seed=42,
                        verbose=False,
                        taste_ind=taste_ind,
                    )
                except Exception as e:
                    print(f"  Trial {trial.number} {ds_name} taste {taste_ind} FAILED: {e}")
                    return float('inf'), float('inf')

                aicr_norm, loss_ll_corr = _extract_metrics(info_criteria, corr_metric)
                ds_aicrs.append(aicr_norm)
                ds_corrs.append(loss_ll_corr)

            ds_mean_aicr = np.mean(ds_aicrs)
            valid_corrs = [c for c in ds_corrs if not np.isnan(c)]
            ds_mean_corr = np.mean(valid_corrs) if valid_corrs else float('nan')
            all_aicrs.append(ds_mean_aicr)
            all_corrs.append(ds_mean_corr)

            print(f"    {ds_name}: AICr/obs={ds_mean_aicr:.6f}, r={ds_mean_corr:.3f}")

        overall_aicr = np.mean(all_aicrs)
        valid_all_corrs = [c for c in all_corrs if not np.isnan(c)]
        overall_corr = np.mean(valid_all_corrs) if valid_all_corrs else float('nan')
        elapsed = time.time() - trial_start

        trial.set_user_attr('bin_size', bin_size)
        trial.set_user_attr('per_dataset_aicr_norm', all_aicrs)
        trial.set_user_attr('per_dataset_corr', all_corrs)
        trial.set_user_attr('mean_aicr_norm', overall_aicr)
        trial.set_user_attr('mean_corr', overall_corr)
        trial.set_user_attr('time_s', elapsed)

        corr_label = 'Poisson' if corr_metric == 'poisson' else 'Gaussian'
        print(f"  Trial {trial.number}: bin={bin_size}, "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f} | "
              f"Overall AICr/obs={overall_aicr:.6f}, "
              f"Overall r({corr_label})={overall_corr:.3f} | {elapsed:.1f}s")

        neg_corr = -overall_corr if not np.isnan(overall_corr) else float('inf')
        return overall_aicr, neg_corr

    return objective


# ----------------------------------------------------------------
# Pareto front visualization
# ----------------------------------------------------------------

def plot_pareto_front(study, taste_label, output_dir, corr_metric='poisson'):
    """
    Plot the Pareto front from a multi-objective study.

    Figure 1: Pareto front (AICr/obs vs correlation)
    Figure 2: Best trials by each objective with parameter comparison
    """
    trials = [t for t in study.trials
              if t.values is not None
              and all(v < float('inf') for v in t.values)]
    if len(trials) < 2:
        print("  Not enough valid trials for Pareto plot.")
        return

    aicrs = [t.values[0] for t in trials]
    neg_corrs = [t.values[1] for t in trials]
    corrs = [-nc for nc in neg_corrs]  # flip back to positive
    trial_nums = [t.number for t in trials]

    # Identify Pareto-optimal trials
    pareto_mask = np.zeros(len(trials), dtype=bool)
    for i in range(len(trials)):
        dominated = False
        for j in range(len(trials)):
            if i == j:
                continue
            # j dominates i if j is <= on both objectives and < on at least one
            if (aicrs[j] <= aicrs[i] and neg_corrs[j] <= neg_corrs[i] and
                    (aicrs[j] < aicrs[i] or neg_corrs[j] < neg_corrs[i])):
                dominated = True
                break
        if not dominated:
            pareto_mask[i] = True

    pareto_aicrs = [aicrs[i] for i in range(len(trials)) if pareto_mask[i]]
    pareto_corrs = [corrs[i] for i in range(len(trials)) if pareto_mask[i]]
    pareto_nums = [trial_nums[i] for i in range(len(trials)) if pareto_mask[i]]

    # Sort Pareto front by AICr for the connecting line
    pareto_order = np.argsort(pareto_aicrs)
    pareto_aicrs_sorted = [pareto_aicrs[i] for i in pareto_order]
    pareto_corrs_sorted = [pareto_corrs[i] for i in pareto_order]

    # --- Figure 1: Pareto front ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # All trials
    sc = ax.scatter(aicrs, corrs, c=trial_nums, cmap='viridis',
                    edgecolors='k', s=40, alpha=0.5, label='All trials')
    plt.colorbar(sc, ax=ax, label='Trial #')

    # Pareto front
    ax.scatter(pareto_aicrs, pareto_corrs, c='red', s=120, marker='*',
               zorder=5, label=f'Pareto front ({sum(pareto_mask)} trials)')
    ax.plot(pareto_aicrs_sorted, pareto_corrs_sorted, 'r--', alpha=0.5, zorder=4)

    # Label Pareto trials
    for i in range(len(pareto_aicrs)):
        ax.annotate(f'#{pareto_nums[i]}', (pareto_aicrs[i], pareto_corrs[i]),
                    fontsize=7, ha='left', va='bottom', color='red')

    corr_label = 'Poisson' if corr_metric == 'poisson' else 'Gaussian'
    ax.set_ylabel(f'Loss vs {corr_label} LL correlation (higher = more consistent)')
    ax.set_title(f'Pareto Front ({corr_label} corr) - {taste_label}')
    # ax.set_title(f'Pareto Front - {taste_label}')
    ax.legend(fontsize=9)

    # Add quadrant annotations
    mid_x = np.median(aicrs)
    mid_y = np.median(corrs)
    ax.axvline(mid_x, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(mid_y, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'pareto_front_{taste_label}.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    # --- Figure 2: Pareto trial parameters ---
    if sum(pareto_mask) >= 2:
        pareto_trials = [trials[i] for i in range(len(trials)) if pareto_mask[i]]
        param_names = ['hidden_size', 'rnn_layers', 'dropout', 'lr']

        fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 5))
        for ax, param in zip(axes, param_names):
            p_vals = [t.params.get(param) for t in pareto_trials]
            p_corrs_local = [corrs[i] for i in range(len(trials)) if pareto_mask[i]]
            ax.scatter(p_vals, p_corrs_local, c='red', s=80, marker='*')
            ax.set_xlabel(param)
            ax.set_ylabel('Loss-LL correlation')
            ax.set_title(f'{param} on Pareto front')

        fig.suptitle(f'Pareto Front Parameters - {taste_label}', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f'pareto_params_{taste_label}.png'),
                    bbox_inches='tight', dpi=200)
        plt.close(fig)

    print(f"  Pareto plots saved to {output_dir}")
    print(f"  Pareto-optimal trials: {pareto_nums}")

    return pareto_mask


def save_multiobjective_results(study, taste_label, output_dir):
    """Save results from a multi-objective study."""
    trials = [t for t in study.trials
              if t.values is not None
              and all(v < float('inf') for v in t.values)]

    txt_path = os.path.join(output_dir, f'optuna_results_{taste_label}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Multi-Objective Optuna Results\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Taste/scope:     {taste_label}\n")
        f.write(f"Objectives:      Poisson AICr/obs (minimize), Loss-LL r (maximize)\n")
        f.write(f"Total trials:    {len(study.trials)}\n")
        f.write(f"Valid trials:    {len(trials)}\n")
        f.write(f"{'=' * 70}\n\n")

        # Best by each objective
        if trials:
            best_aicr = min(trials, key=lambda t: t.values[0])
            best_corr = min(trials, key=lambda t: t.values[1])  # min of -corr = max corr

            f.write(f"BEST BY AICr/obs: Trial #{best_aicr.number}\n")
            f.write(f"  AICr/obs = {best_aicr.values[0]:.6f}, "
                    f"r = {-best_aicr.values[1]:.3f}\n")
            for k, v in best_aicr.params.items():
                f.write(f"  {k:<20s} = {v}\n")

            f.write(f"\nBEST BY CORRELATION: Trial #{best_corr.number}\n")
            f.write(f"  AICr/obs = {best_corr.values[0]:.6f}, "
                    f"r = {-best_corr.values[1]:.3f}\n")
            for k, v in best_corr.params.items():
                f.write(f"  {k:<20s} = {v}\n")
        f.write(f"\n\nFULL TRIAL LOG:\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"{'#':>4s} {'AICr/obs':>12s} {'r':>8s} {'bin':>5s} "
                f"{'hidden':>7s} {'layers':>7s} {'k':>7s} {'time':>7s}\n")
        f.write(f"{'-' * 70}\n")
        for t in trials:
            ua = t.user_attrs
            # bin_size may be in params (searched) or user_attrs (set in objective)
            bin_size = t.params.get('bin_size', ua.get('bin_size', '?'))
            f.write(f"{t.number:>4d} "
                    f"{t.values[0]:>12.6f} "
                    f"{-t.values[1]:>8.3f} "
                    f"{bin_size:>5} "
                    f"{t.params.get('hidden_size', '?'):>7} "
                    f"{t.params.get('rnn_layers', '?'):>7} "
                    f"{ua.get('n_params', '?'):>7} "
                    f"{ua.get('time_s', 0):>6.1f}s\n")
    print(f"  Saved: {txt_path}")

    # JSON
    json_path = os.path.join(output_dir, f'optuna_results_{taste_label}.json')
    json_data = {
        'objectives': ['poisson_aicr_per_obs', 'neg_loss_ll_correlation'],
        'n_trials': len(study.trials),
        'n_valid': len(trials),
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'aicr_per_obs': t.values[0],
                'loss_ll_corr': -t.values[1],
                'user_attrs': {
                    k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                    for k, v in t.user_attrs.items()
                },
            }
            for t in trials
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")