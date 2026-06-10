"""
Staged hyperparameter sweep via LOO cross-validation.

Sweeps one parameter at a time, fixes the best value, moves to the next.
I am not doing the full grid search, as with this as it stands this will be probably an hour of training per taste. 
eg: 
5 hidden sizes + 3 rnn_layers + 5 dropouts + 3 lrs + 2 losses = 18 configs. Each runs 30-fold LOO at ~6s/fold = ~3 min per config. Total: ~54 minutes per taste.
With a full grid search: (lmao iykyk)
5 hidden sizes * 3 rnn_layers * 5 dropouts * 3 lrs * 2 losses = 450 configs. 
around 6 sec per LOO * 30 LOO and the final training run, ~13,500 trainings per run or ~68,000 seconds. FYI that's ~19 hours. PER TASTE!

Uses LOO log-likelihood (Gaussian or Poisson) as the optimization target.

Lower AIC = better model accounting for complexity.

Usage:
    python sweep.py

Configure by editing the SWEEP CONFIG section below, or by modifying
the JSON config path.

Outputs:
    - Per-stage AIC/BIC vs parameter plots
    - optimized_params.txt with final best values and full sweep log
    - sweep_results.json with all numeric results for later analysis
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# SWEEP CONFIG — edit these
# ----------------------------------------------------------------

# Path to your standard blechrnn config (for paths, data loading, etc.)
CONFIG_PATH = '/home/vincent/Senior thesis work/blechRNN-master/config/blechrnn_config.json'

# Which taste to sweep (0, 1, 2, 3, or 'all')
SWEEP_TASTE = 0

# Which LL to optimize: 'gaussian' or 'poisson'
OPTIMIZE_LL = 'gaussian'

# LOO fold settings (kept short for speed during sweep)
LOO_TRAIN_STEPS = 3000
LOO_PATIENCE = 15

# Parameters to sweep, in order. Each stage fixes the best value
# from the previous stage before sweeping the next.
# Edit the lists to add/remove values.

# NOTE: Figure out if any one of thse (or combination of these...) will run me plumb out of VRAM
# similarly, are any of these unnessecary? hmmmmmmmmmmmmm....
SWEEP_PARAMS = {
    'hidden_size':  [4, 8, 16, 32, 64],
    'rnn_layers':   [1, 2, 3],
    'dropout':      [0.0, 0.1, 0.2, 0.3, 0.5],
    'lr':           [0.01, 0.001, 0.0001],
    'loss_name':    ['mse', 'smooth'],
}

# Defaults — these are used as the starting point before any stage
# has found a best value. After each stage, the best value replaces
# the default for subsequent stages.
DEFAULTS = {
    'hidden_size':  8,
    'rnn_layers':   2,
    'dropout':      0.2,
    'lr':           0.001,
    'loss_name':    'mse',
}

# ----------------------------------------------------------------
# End of config
# ----------------------------------------------------------------

from config_loader import load_config
from preprocessing import preprocess_taste
from run_training import loo_then_train
from training import MSELoss, smooth_MSELoss
from ephys_data import ephys_data


def get_criterion(loss_name):
    """Return loss function by name."""
    if loss_name == 'smooth':
        return smooth_MSELoss(alpha=0.05)
    return MSELoss()


def extract_metric(info_criteria, metric='gaussian'):
    """
    Extract the target AIC from info_criteria.

    Args:
        info_criteria: dict from loo_then_train
        metric: 'gaussian' or 'poisson'

    Returns:
        aic: float (lower is better)
        bic: float
        ll: float (higher is better)
    """
    if metric == 'poisson':
        aic = info_criteria.get('poisson_aic', float('nan'))
        bic = info_criteria.get('poisson_bic', float('nan'))
        ll = info_criteria.get('poisson_log_likelihood', float('nan'))
    else:
        aic = info_criteria.get('aic', float('nan'))
        bic = info_criteria.get('bic', float('nan'))
        ll = info_criteria.get('log_likelihood', float('nan'))
    return aic, bic, ll


def run_single_config(prep, params_dict, device, taste_ind, artifacts_dir):
    """
    Run LOO for a single parameter configuration.

    Args:
        prep: dict from preprocess_taste
        params_dict: dict with hidden_size, rnn_layers, dropout, lr, loss_name
        device: torch device
        taste_ind: int
        artifacts_dir: str

    Returns:
        info_criteria: dict from loo_then_train
    """
    criterion = get_criterion(params_dict['loss_name'])

    # loo_then_train returns (net, loss, cross_val_loss, info_criteria)
    # We only need info_criteria for the sweep — discard the rest
    _, _, _, info_criteria = loo_then_train(
        inputs_tensor=prep['inputs_tensor'],
        labels_tensor=prep['labels_tensor'],
        input_size=prep['input_size'],
        hidden_size=params_dict['hidden_size'],
        output_size=prep['output_size'],
        device=device,
        criterion=criterion,
        train_steps=LOO_TRAIN_STEPS,   # use LOO settings for all folds + final
        patience=LOO_PATIENCE,
        lr=params_dict['lr'],
        rnn_layers=params_dict['rnn_layers'],
        dropout=params_dict['dropout'],
        retrain=True,
        model_save_path=None,  # don't save models during sweep
        artifacts_dir=artifacts_dir,
        taste_ind=taste_ind,
        verbose=False,
        loo_train_steps=LOO_TRAIN_STEPS,
        loo_patience=LOO_PATIENCE,
        scaler=prep['scaler'],
        pca_obj=prep['pca_obj'],
        raw_labels_tensor=prep.get('raw_labels_tensor'),
    )
    return info_criteria


def run_staged_sweep(prep, device, taste_ind, output_dir):
    """
    Run the full staged sweep.

    Returns:
        best_params: dict with optimal values
        all_results: dict with full sweep data
    """
    artifacts_dir = os.path.join(output_dir, 'sweep_artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    current_best = dict(DEFAULTS)
    all_results = {}
    stage_order = list(SWEEP_PARAMS.keys())

    total_configs = sum(len(v) for v in SWEEP_PARAMS.values())
    print(f"\n{'=' * 60}")
    print(f"STAGED SWEEP — Taste {taste_ind}")
    print(f"Optimizing: {OPTIMIZE_LL} AIC")
    print(f"Stages: {stage_order}")
    print(f"Total configs to evaluate: {total_configs}")
    print(f"{'=' * 60}\n")

    sweep_start = time.time()

    for stage_idx, param_name in enumerate(stage_order):
        values = SWEEP_PARAMS[param_name]
        print(f"\n--- Stage {stage_idx + 1}/{len(stage_order)}: "
              f"Sweeping {param_name} over {values} ---")
        print(f"    Current best: {current_best}")

        stage_results = []

        for val_idx, val in enumerate(values):
            # Build config for this run
            run_params = dict(current_best)
            run_params[param_name] = val

            print(f"\n  [{val_idx + 1}/{len(values)}] {param_name}={val}")
            run_start = time.time()

            info_criteria = run_single_config(
                prep, run_params, device, taste_ind, artifacts_dir
            )

            aic, bic, ll = extract_metric(info_criteria, OPTIMIZE_LL)
            run_elapsed = time.time() - run_start

            result = dict(
                param_name=param_name,
                param_value=val,
                full_params=dict(run_params),
                aic=aic,
                bic=bic,
                log_likelihood=ll,
                gaussian_aic=info_criteria.get('aic', float('nan')),
                gaussian_bic=info_criteria.get('bic', float('nan')),
                gaussian_ll=info_criteria.get('log_likelihood', float('nan')),
                poisson_aic=info_criteria.get('poisson_aic', float('nan')),
                poisson_bic=info_criteria.get('poisson_bic', float('nan')),
                poisson_ll=info_criteria.get('poisson_log_likelihood', float('nan')),
                n_params=info_criteria.get('n_params', 0),
                n_trials=info_criteria.get('n_trials', 0),
                time_s=run_elapsed,
            )
            stage_results.append(result)

            print(f"    AIC={aic:.2f}  BIC={bic:.2f}  LL={ll:.2f}  "
                  f"n_params={result['n_params']}  ({run_elapsed:.1f}s)")

        # Find best value for this stage
        valid_results = [r for r in stage_results if not np.isnan(r['aic'])]
        if valid_results:
            best_result = min(valid_results, key=lambda r: r['aic'])
            best_val = best_result['param_value']
            current_best[param_name] = best_val
            print(f"\n  >>> Best {param_name} = {best_val} "
                  f"(AIC={best_result['aic']:.2f})")
        else:
            print(f"\n  >>> WARNING: No valid results for {param_name}, "
                  f"keeping default {current_best[param_name]}")

        all_results[param_name] = stage_results

        # Plot this stage
        _plot_stage(stage_results, param_name, taste_ind,
                    current_best[param_name], output_dir)

    total_elapsed = time.time() - sweep_start
    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE — {total_elapsed:.1f}s total")
    print(f"Best params: {current_best}")
    print(f"{'=' * 60}\n")

    return current_best, all_results


def _plot_stage(stage_results, param_name, taste_ind, best_val, output_dir):
    """Plot AIC/BIC vs parameter value for one sweep stage."""
    values = [r['param_value'] for r in stage_results]
    aics = [r['aic'] for r in stage_results]
    bics = [r['bic'] for r in stage_results]
    n_params_list = [r['n_params'] for r in stage_results]

    # Handle string values (loss_name)
    if isinstance(values[0], str):
        x = np.arange(len(values))
        x_labels = values
    else:
        x = np.array(values, dtype=float)
        x_labels = None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: AIC and BIC
    ax = axes[0]
    if x_labels is not None:
        ax.bar(x - 0.175, aics, 0.35, label='AIC', color='steelblue')
        ax.bar(x + 0.175, bics, 0.35, label='BIC', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    else:
        ax.plot(x, aics, 'o-', label='AIC', color='steelblue', markersize=8)
        ax.plot(x, bics, 's-', label='BIC', color='coral', markersize=8)
        # Mark best
        best_mask = np.array(values) == best_val
        if np.any(best_mask):
            ax.axvline(best_val, color='green', linestyle='--', alpha=0.5,
                       label=f'Best: {best_val}')
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'{OPTIMIZE_LL.capitalize()} Information Criterion')
    ax.set_title(f'AIC / BIC vs {param_name}')
    ax.legend(fontsize=8)

    # Panel 2: Log-likelihood
    ax = axes[1]
    lls = [r['log_likelihood'] for r in stage_results]
    if x_labels is not None:
        ax.bar(x, lls, color='seagreen')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    else:
        ax.plot(x, lls, 'o-', color='seagreen', markersize=8)
        if np.any(best_mask):
            ax.axvline(best_val, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'{OPTIMIZE_LL.capitalize()} Log-Likelihood')
    ax.set_title(f'LL vs {param_name}')

    # Panel 3: Parameter count
    ax = axes[2]
    if x_labels is not None:
        ax.bar(x, n_params_list, color='slategray')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    else:
        ax.plot(x, n_params_list, 'o-', color='slategray', markersize=8)
        if np.any(best_mask):
            ax.axvline(best_val, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Trainable Parameters')
    ax.set_title(f'Model Size vs {param_name}')

    fig.suptitle(f'Sweep: {param_name} — Taste {taste_ind} ({OPTIMIZE_LL})',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f'sweep_{param_name}_taste_{taste_ind}.png'),
        bbox_inches='tight', dpi=200
    )
    plt.close(fig)


def save_results(best_params, all_results, taste_ind, output_dir):
    """Save optimized params and full sweep data."""
    # --- optimized_params.txt ---
    txt_path = os.path.join(output_dir, f'optimized_params_taste_{taste_ind}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Staged Hyperparameter Sweep Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Taste:           {taste_ind}\n")
        f.write(f"Optimized for:   {OPTIMIZE_LL} AIC\n")
        f.write(f"LOO fold steps:  {LOO_TRAIN_STEPS}\n")
        f.write(f"LOO patience:    {LOO_PATIENCE}\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"BEST PARAMETERS:\n")
        f.write(f"{'-' * 40}\n")
        for k, v in best_params.items():
            f.write(f"  {k:<20s} = {v}\n")
        f.write(f"\n")

        f.write(f"SWEEP ORDER AND RANGES:\n")
        f.write(f"{'-' * 40}\n")
        for param_name, values in SWEEP_PARAMS.items():
            f.write(f"  {param_name:<20s}: {values}\n")
        f.write(f"\n")

        f.write(f"DEFAULTS (starting point):\n")
        f.write(f"{'-' * 40}\n")
        for k, v in DEFAULTS.items():
            f.write(f"  {k:<20s} = {v}\n")
        f.write(f"\n")

        f.write(f"DETAILED RESULTS PER STAGE:\n")
        f.write(f"{'=' * 60}\n\n")
        for param_name, results in all_results.items():
            f.write(f"Stage: {param_name}\n")
            f.write(f"{'-' * 60}\n")
            f.write(f"  {'Value':<12s} {'AIC':>12s} {'BIC':>12s} "
                    f"{'LL':>12s} {'n_params':>10s} {'Time':>8s}\n")
            f.write(f"  {'-' * 66}\n")
            for r in results:
                val_str = str(r['param_value'])
                f.write(f"  {val_str:<12s} {r['aic']:>12.2f} {r['bic']:>12.2f} "
                        f"{r['log_likelihood']:>12.2f} {r['n_params']:>10d} "
                        f"{r['time_s']:>7.1f}s\n")
            # Mark best
            valid = [r for r in results if not np.isnan(r['aic'])]
            if valid:
                best = min(valid, key=lambda r: r['aic'])
                f.write(f"  >>> Best: {param_name}={best['param_value']} "
                        f"(AIC={best['aic']:.2f})\n")
            f.write(f"\n")

        # Also log both Gaussian and Poisson for each run
        f.write(f"\nFULL METRICS (Gaussian + Poisson):\n")
        f.write(f"{'=' * 60}\n")
        for param_name, results in all_results.items():
            f.write(f"\nStage: {param_name}\n")
            f.write(f"  {'Value':<12s} {'G_AIC':>10s} {'G_BIC':>10s} "
                    f"{'G_LL':>10s} {'P_AIC':>10s} {'P_BIC':>10s} {'P_LL':>10s}\n")
            f.write(f"  {'-' * 74}\n")
            for r in results:
                val_str = str(r['param_value'])
                f.write(f"  {val_str:<12s} "
                        f"{r['gaussian_aic']:>10.1f} {r['gaussian_bic']:>10.1f} "
                        f"{r['gaussian_ll']:>10.1f} "
                        f"{r['poisson_aic']:>10.1f} {r['poisson_bic']:>10.1f} "
                        f"{r['poisson_ll']:>10.1f}\n")

    print(f"  Saved: {txt_path}")

    # --- sweep_results.json ---
    json_path = os.path.join(output_dir, f'sweep_results_taste_{taste_ind}.json')
    # Convert all_results to JSON-serializable format
    json_data = {
        'best_params': best_params,
        'optimize_ll': OPTIMIZE_LL,
        'defaults': DEFAULTS,
        'sweep_params': {k: [str(v) if not isinstance(v, (int, float)) else v
                              for v in vals]
                         for k, vals in SWEEP_PARAMS.items()},
        'stages': {},
    }
    for param_name, results in all_results.items():
        json_data['stages'][param_name] = [
            {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
             for k, v in r.items()}
            for r in results
        ]
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    config, paths, params, _ = load_config(CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Find the first dataset (or modify to loop)
    for subdir in sorted(os.listdir(paths['h5_dir'])):
        full_subdir_path = os.path.join(paths['h5_dir'], subdir)
        if not os.path.isdir(full_subdir_path):
            continue
        h5_files = [f for f in os.listdir(full_subdir_path) if f.endswith(".h5")]
        if len(h5_files) != 1:
            continue

        dataset_name = os.path.splitext(h5_files[0])[0]
        print(f"Dataset: {dataset_name}")

        data = ephys_data(full_subdir_path)
        data.get_spikes()
        spike_array = np.stack(data.spikes)

        # Output directory
        output_dir = os.path.join(paths['output_base_dir'], dataset_name,
                                  'sweep_results')
        os.makedirs(output_dir, exist_ok=True)

        # Determine which tastes to sweep
        if SWEEP_TASTE == 'all':
            taste_indices = list(range(len(spike_array)))
        else:
            taste_indices = [SWEEP_TASTE]

        for taste_ind in taste_indices:
            print(f"\n{'#' * 60}")
            print(f"# Taste {taste_ind}")
            print(f"{'#' * 60}")

            taste_spikes = spike_array[taste_ind]
            taste_spikes = taste_spikes[...,
                                        params['time_lims'][0]:params['time_lims'][1]]

            prep = preprocess_taste(
                taste_spikes,
                bin_size=params['bin_size'],
                stim_time_val=paths['stim_time_val'],
                use_pca=params['use_pca'],
            )

            best_params, all_results = run_staged_sweep(
                prep, device, taste_ind, output_dir
            )

            save_results(best_params, all_results, taste_ind, output_dir)

        # Only process first dataset — remove break to process all
        break


if __name__ == '__main__':
    main()