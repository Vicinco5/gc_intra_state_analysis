"""
Core functions for Optuna hyperparameter optimization.
Imported by optuna_sweep.py — do not run directly.
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import optuna

from run_training import loo_then_train, run_prediction, kfold_evaluate
from postprocessing import reconstruct_firing
from train import MSELoss, smooth_MSELoss
from visualizations import (
    plot_inputs, plot_loss_curves, plot_firing_overview,
    plot_mean_firing, plot_latent_factors, plot_trial_latents,
    plot_individual_neurons, plot_mean_neurons_across_tastes,
    plot_pred_vs_true_neurons, plot_aic_bic_summary,
    plot_loo_diagnostics,
)
from save_outputs import save_to_hdf5, save_latents_parquet, save_firing_parquet
from neuron_eval import evaluate_neurons
from preprocessing import preprocess_taste
from ephys_data import ephys_data
from fixedpoint import run_fixed_point_analysis


def get_criterion(loss_name):
    if loss_name == 'smooth':
        return smooth_MSELoss(alpha=0.05)
    return MSELoss()


# ----------------------------------------------------------------
# Objective
# ----------------------------------------------------------------

def create_objective(prep, device, taste_ind, artifacts_dir,
                     loo_train_steps, loo_patience):
    """Create Optuna objective closed over the data."""
    def objective(trial):
        # Import here to avoid circular — define_search_space lives in optuna_sweep.py
        from optuna_sweep import define_search_space
        kfold = True # internal flag that makes it so that the optuna objective is informed by 
        # the kfold results
        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])
        trial_start = time.time()
        try:
            if kfold: 
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
                    n_folds=5,
                    seed=42,
                    verbose=False,
                    taste_ind=taste_ind,
                )
            else:
                _, _, _, info_criteria = loo_then_train(
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
                    retrain=True,
                    model_save_path=None,
                    artifacts_dir=artifacts_dir,
                    taste_ind=taste_ind,
                    verbose=False,
                    loo_train_steps=loo_train_steps,
                    loo_patience=loo_patience,
                    scaler=prep['scaler'],
                    pca_obj=prep['pca_obj'],
                    raw_labels_tensor=prep.get('raw_labels_tensor'),
                )
        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {e}")
            return float('inf')
        # Now computing for poisson aicr:
        poisson_aic = info_criteria.get('poisson_aicr', float('nan'))
        if np.isnan(poisson_aic):
            print(f"  Trial {trial.number}: Poisson AICr is nan, "
                  f"falling back to poisson AIC")
            poisson_aic = info_criteria.get('poisson_aic', float('nan'))
            if np.isnan(poisson_aic):
                print(f"  Trial {trial.number}: Poisson AIC is nan, "
                    f"falling back to Gaussian")
                poisson_aic = info_criteria.get('aic', float('inf'))

        elapsed = time.time() - trial_start

        trial.set_user_attr('gaussian_aic', info_criteria.get('aic', float('nan')))
        trial.set_user_attr('gaussian_aicr', info_criteria.get('aicr', float('nan')))
        trial.set_user_attr('gaussian_bic', info_criteria.get('bic', float('nan')))
        trial.set_user_attr('gaussian_ll', info_criteria.get('log_likelihood', float('nan')))
        trial.set_user_attr('poisson_aic', info_criteria.get('poisson_aic', float('nan')))
        trial.set_user_attr('poisson_aicr', info_criteria.get('poisson_aicr', float('nan')))
        trial.set_user_attr('poisson_bic', info_criteria.get('poisson_bic', float('nan')))
        trial.set_user_attr('poisson_ll', info_criteria.get('poisson_log_likelihood', float('nan')))
        trial.set_user_attr('n_params', info_criteria.get('n_params', 0))
        trial.set_user_attr('time_s', elapsed)

        print(f"  Trial {trial.number}: "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f}, "
              f"loss={params['loss_name']} | "
              f"Poisson AICr={poisson_aic:.2f} | {elapsed:.1f}s")

        return poisson_aic

    return objective


# ----------------------------------------------------------------
# Results saving (AICr-aware)
# ----------------------------------------------------------------

def export_best_params(study, taste_ind, output_dir, loo_train_steps, loo_patience):
    """Save best params as a timestamped JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best = study.best_trial
    export = {
        'timestamp': timestamp,
        'study_name': study.study_name,
        'taste_ind': taste_ind,
        'objective': 'poisson_aicr',
        'best_poisson_aicr': best.value,
        'best_trial_number': best.number,
        'parameters': best.params,
        'metrics': {k: v for k, v in best.user_attrs.items()},
        'loo_train_steps': loo_train_steps,
        'loo_patience': loo_patience,
        'n_trials_total': len(study.trials),
    }
    path = os.path.join(output_dir, f'optuna_params_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"  Saved best params: {path}")
    return path


def save_study_results(study, taste_ind, output_dir):
    """Save human-readable and machine-readable results."""
    best = study.best_trial

    # --- Text report ---
    txt_path = os.path.join(output_dir, f'optuna_results_taste_{taste_ind}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Optuna Hyperparameter Optimization Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Study name:      {study.study_name}\n")
        f.write(f"Taste:           {taste_ind}\n")
        f.write(f"Objective:       Poisson AICr (minimize)\n")
        f.write(f"                 (DelSole & Tippett 2021, Random-X corrected)\n")
        f.write(f"Total trials:    {len(study.trials)}\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"BEST TRIAL: #{best.number}\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"  Poisson AICr:   {best.value:.2f}\n")
        for k, v in best.params.items():
            f.write(f"  {k:<20s} = {v}\n")
        f.write(f"\n  Additional metrics:\n")
        for k, v in best.user_attrs.items():
            if isinstance(v, float):
                f.write(f"    {k:<20s} = {v:.4f}\n")
            else:
                f.write(f"    {k:<20s} = {v}\n")

        f.write(f"\n\nTOP 10 TRIALS:\n")
        f.write(f"{'=' * 60}\n")
        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value
        )
        for t in sorted_trials[:10]:
            f.write(f"\n  Trial #{t.number}  |  Poisson AICr = {t.value:.2f}\n")
            for k, v in t.params.items():
                f.write(f"    {k:<20s} = {v}\n")
            ua = t.user_attrs
            # Show both AIC and AICr for comparison
            p_aic = ua.get('poisson_aic', float('nan'))
            p_aicr = ua.get('poisson_aicr', t.value)
            n_p = ua.get('n_params', '?')
            f.write(f"    {'poisson_aic':<20s} = {p_aic:.2f}\n")
            f.write(f"    {'poisson_aicr':<20s} = {p_aicr:.2f}\n")
            f.write(f"    {'n_params':<20s} = {n_p}\n")

        f.write(f"\n\nFULL TRIAL LOG:\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"{'#':>4s} {'P_AICr':>10s} {'P_AIC':>10s} {'P_LL':>10s} "
                f"{'G_AICr':>10s} {'G_AIC':>10s} {'hidden':>7s} {'layers':>7s} "
                f"{'dropout':>8s} {'lr':>10s} {'loss':>7s} {'k':>7s} "
                f"{'time':>7s}\n")
        f.write(f"{'-' * 110}\n")
        for t in study.trials:
            if t.value is None:
                continue
            ua = t.user_attrs
            f.write(f"{t.number:>4d} "
                    f"{t.value:>10.1f} "
                    f"{ua.get('poisson_aic', float('nan')):>10.1f} "
                    f"{ua.get('poisson_ll', float('nan')):>10.1f} "
                    f"{ua.get('gaussian_aicr', float('nan')):>10.1f} "
                    f"{ua.get('gaussian_aic', float('nan')):>10.1f} "
                    f"{t.params.get('hidden_size', '?'):>7} "
                    f"{t.params.get('rnn_layers', '?'):>7} "
                    f"{t.params.get('dropout', '?'):>8.2f} "
                    f"{t.params.get('lr', '?'):>10.5f} "
                    f"{t.params.get('loss_name', '?'):>7s} "
                    f"{ua.get('n_params', '?'):>7} "
                    f"{ua.get('time_s', 0):>6.1f}s\n")
    print(f"  Saved: {txt_path}")

    # --- JSON ---
    json_path = os.path.join(output_dir, f'optuna_results_taste_{taste_ind}.json')
    json_data = {
        'objective': 'poisson_aicr',
        'best_params': best.params,
        'best_poisson_aicr': best.value,
        'best_trial_number': best.number,
        'best_user_attrs': {
            k: (v if not isinstance(v, float) or not np.isnan(v) else None)
            for k, v in best.user_attrs.items()
        },
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'poisson_aicr': t.value,
                'user_attrs': {
                    k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                    for k, v in t.user_attrs.items()
                },
            }
            for t in study.trials if t.value is not None
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")


# ----------------------------------------------------------------
# Plots (AICr-aware)
# ----------------------------------------------------------------

def plot_study_results(study, taste_ind, output_dir):
    """Custom plots for Optuna results."""
    trials = [t for t in study.trials
              if t.value is not None and t.value < float('inf')]
    if len(trials) < 2:
        print("  Not enough valid trials to plot.")
        return

    trial_nums = [t.number for t in trials]
    aics = [t.value for t in trials]
    best_so_far = np.minimum.accumulate(aics)

    # --- Figure 1: Optimization history ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_nums, aics, c='steelblue', edgecolors='k',
               s=30, alpha=0.6, label='Trial AICr')
    ax.plot(trial_nums, best_so_far, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Poisson AICr')
    ax.set_title(f'Optimization History - Taste {taste_ind}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'optuna_history_taste_{taste_ind}.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    # --- Figure 2: AICr vs each parameter ---
    param_names = ['hidden_size', 'rnn_layers', 'dropout', 'lr', 'loss_name']
    n_cols = 3
    n_rows = int(np.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        vals = [t.params.get(param) for t in trials]
        trial_order = np.arange(len(trials))

        if isinstance(vals[0], str):
            categories = sorted(set(vals))
            grouped = {c: [aics[j] for j, v in enumerate(vals) if v == c]
                       for c in categories}
            bp = ax.boxplot([grouped[c] for c in categories],
                            positions=range(len(categories)),
                            widths=0.6, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories)
        else:
            sc = ax.scatter(vals, aics, c=trial_order, cmap='viridis',
                            edgecolors='k', s=40, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Trial #')
            best_idx = np.argmin(aics)
            ax.scatter(vals[best_idx], aics[best_idx], c='red', s=100,
                       marker='*', zorder=5, label='Best')
            ax.legend(fontsize=8)

        ax.set_xlabel(param)
        ax.set_ylabel('Poisson AICr')
        ax.set_title(f'AICr vs {param}')

    for i in range(len(param_names), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Parameter Exploration (AICr) - Taste {taste_ind}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'optuna_params_taste_{taste_ind}.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    # --- Figure 3: Parameter importance ---
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = {}
    for param in param_names:
        vals = [t.params.get(param) for t in trials]
        if isinstance(vals[0], str):
            categories = sorted(set(vals))
            vals_numeric = [categories.index(v) for v in vals]
        else:
            vals_numeric = vals
        try:
            corr = np.corrcoef(vals_numeric, aics)[0, 1]
            importances[param] = abs(corr)
        except Exception:
            importances[param] = 0.0

    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]
    bars = ax.barh(names, values, color='steelblue', edgecolor='k')
    ax.set_xlabel('|Correlation| with Poisson AICr')
    ax.set_title(f'Parameter Importance (AICr) - Taste {taste_ind}')
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'optuna_importance_taste_{taste_ind}.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Plots saved to {output_dir}")


# ----------------------------------------------------------------
# Run full pipeline with optimized params
# ----------------------------------------------------------------

def run_optimized_pipeline(best_params, paths, params,
                           dataset_name, full_subdir_path, output_base,
                           loo_train_steps, loo_patience):
    """
    Run the full run_rnn.py pipeline using the optimized parameters.
    Results go to output_base/optuna_optimization/<dataset>/
    """
    best_params = {
        'hidden_size': 8,
        'rnn_layers': 2,
        'dropout': 0.2,
        'lr': 0.001,
        'loss_name': 'mse',
        **best_params,   # searched params override the defaults
    }
    print(f"\n{'=' * 60}")
    print(f"Running full pipeline with optimized params")
    print(f"{'=' * 60}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = get_criterion(best_params.get('loss_name', 'mse'))

    opt_output = os.path.join(output_base, 'optuna_optimization', dataset_name)
    plots_dir = os.path.join(opt_output, 'plots')
    artifacts_dir = os.path.join(opt_output, 'artifacts')
    model_eval_dir = os.path.join(opt_output, 'model_eval')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(model_eval_dir, exist_ok=True)

    with open(os.path.join(opt_output, 'optimized_params_used.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    data = ephys_data(full_subdir_path)
    data.get_spikes()
    spike_array = np.stack(data.spikes)

    pred_firing_list = []
    latent_out_list = []
    binned_spikes_list = []
    conv_rate_list = []
    conv_x_list = []
    info_criteria_all = {}

    for taste_ind, taste_spikes_raw in enumerate(spike_array):
        print(f"\n  Taste {taste_ind}")
        taste_spikes = taste_spikes_raw[...,
                                        params['time_lims'][0]:params['time_lims'][1]]

        prep = preprocess_taste(
            taste_spikes,
            bin_size=params['bin_size'],
            stim_time_val=paths['stim_time_val'],
            use_pca=params['use_pca'],
        )

        plot_inputs(prep['inputs_plus_context'], dataset_name, taste_ind, plots_dir)

        model_name = (f'taste_{taste_ind}_hidden_{best_params["hidden_size"]}'
                      f'_loss_{best_params["loss_name"]}_optimized')
        model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')

        net, loss, cross_val_loss, info_criteria = loo_then_train(
            inputs_tensor=prep['inputs_tensor'],
            labels_tensor=prep['labels_tensor'],
            input_size=prep['input_size'],
            hidden_size=best_params['hidden_size'],
            output_size=prep['output_size'],
            device=device,
            criterion=criterion,
            train_steps=params['train_steps'],
            patience=params['patience'],
            lr=best_params['lr'],
            rnn_layers=best_params['rnn_layers'],
            dropout=best_params['dropout'],
            retrain=True,
            model_save_path=model_save_path,
            artifacts_dir=artifacts_dir,
            taste_ind=taste_ind,
            verbose=True,
            loo_train_steps=loo_train_steps,
            loo_patience=loo_patience,
            scaler=prep['scaler'],
            pca_obj=prep['pca_obj'],
            raw_labels_tensor=prep.get('raw_labels_tensor'),
        )
        info_criteria_all[taste_ind] = info_criteria

        plot_loo_diagnostics(info_criteria, dataset_name, taste_ind, model_eval_dir)

        outs, latent_outs = run_prediction(net, prep['inputs_tensor'], device)
        latent_out_list.append(latent_outs)

        pred_firing = reconstruct_firing(
            outs, scaler=prep['scaler'], pca_obj=prep['pca_obj'],
            num_neurons=prep['num_neurons'], use_pca=params['use_pca'],
        )
        pred_firing_list.append(pred_firing)
        binned_spikes_list.append(prep['binned_spikes'])

        conv_kern = np.ones(250) / 250
        conv_rate = np.apply_along_axis(
            lambda m: np.convolve(m, conv_kern, mode='valid'),
            axis=-1, arr=taste_spikes
        ) * params['bin_size']
        conv_x = np.convolve(
            np.arange(taste_spikes.shape[-1]), conv_kern, mode='valid'
        )
        conv_rate_list.append(conv_rate)
        conv_x_list.append(conv_x)

        plot_loss_curves(loss, cross_val_loss, dataset_name, taste_ind, plots_dir)
        plot_firing_overview(pred_firing, prep['binned_spikes'],
                            dataset_name, taste_ind, plots_dir)
        plot_mean_firing(pred_firing, prep['binned_spikes'],
                        dataset_name, taste_ind, plots_dir)
        plot_latent_factors(latent_outs, dataset_name, taste_ind, plots_dir)
        plot_trial_latents(latent_outs, dataset_name, taste_ind, plots_dir)
        plot_individual_neurons(
            taste_spikes, prep['binned_spikes'], pred_firing,
            conv_rate, conv_x, params['bin_size'],
            paths['stim_time_val'], dataset_name, taste_ind, plots_dir
        )

        evaluate_neurons(
            net=net, inputs_tensor=prep['inputs_tensor'],
            labels_tensor=prep['labels_tensor'],
            raw_labels_tensor=prep.get('raw_labels_tensor'),
            scaler=prep['scaler'], pca_obj=prep['pca_obj'],
            binned_spikes=prep['binned_spikes'],
            dataset_name=dataset_name, taste_ind=taste_ind,
            output_dir=model_eval_dir, device=device,
        )
        fp_dir = os.path.join(model_eval_dir, 'model_eval')
        run_fixed_point_analysis(
            net=net,
            prep=prep,
            latent_outs=latent_outs,
            dataset_name=dataset_name,
            taste_ind=taste_ind,
            output_dir=fp_dir,
            device=device,
            # NOTE: gotta figure out timing as this shit is in ms 
            # and concatenation plus whatever else 
            time_indices=None,
        )

    plot_mean_neurons_across_tastes(
        spike_array, pred_firing_list, binned_spikes_list,
        conv_rate_list, conv_x_list, params['bin_size'],
        paths['stim_time_val'], params['time_lims'],
        dataset_name, plots_dir
    )
    plot_pred_vs_true_neurons(pred_firing_list, binned_spikes_list, plots_dir)
    plot_aic_bic_summary(info_criteria_all, dataset_name, model_eval_dir)

    save_to_hdf5(data.hdf5_path, pred_firing_list, latent_out_list,
                 params['bin_size'])
    save_latents_parquet(latent_out_list, dataset_name, artifacts_dir,
                         paths['pred_lat_dir'])
    save_firing_parquet(pred_firing_list, dataset_name, artifacts_dir,
                        paths['pred_fr_dir'])

    print(f"\n  Optimized pipeline complete: {opt_output}")


# but the thing is, what is good for one taste may not be ideal for another. Soooooo... we do a multi-taste sweep
# and we do that sweep somewhat intelligently so that things don't take too long. 

def create_multitaste_objective(all_preps, device, artifacts_dir,
                                loo_train_steps, loo_patience):
    """
    Objective that evaluates across all tastes.
    Returns mean Poisson AIC. Prunes early if a taste is catastrophically bad.

    We're gonna implement kfold here to make the optuna sweep go faster. 
    An internal flag is to be used to switch this on and off. 
    """
    def objective(trial):
        from optuna_sweep import define_search_space

        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])
        trial_start = time.time()

        taste_aics = []


        # k-fold eval makes this all a bit quicker, at the cost of introducing some noise. 
        kfold = True

        # to avoid data saving conflicts, we may consider saving data for each optuna trial: 
        trial_root = os.path.join(artifacts_dir, f"trial_{trial.number:05d}")
        os.makedirs(trial_root, exist_ok=True)
        # iterating in a fixed order for maximum comparability here:
        for taste_ind in sorted(all_preps):
            prep = all_preps[taste_ind]
            # and also for taste: 
            taste_dir = os.path.join(trial_root, f"taste_{taste_ind}")
            os.makedirs(taste_dir, exist_ok=True)
        #for taste_ind, prep in all_preps.items():
            try:
                if kfold:

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
                        n_folds=5,
                        seed=42,
                        verbose=False,
                        taste_ind=taste_ind,
                    )
                else:
                    _, _, _, info_criteria = loo_then_train(
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
                        retrain=True,
                        model_save_path=None,
                        artifacts_dir=artifacts_dir, # change to taste_dir if we want to avoid conflicts too much....
                        taste_ind=taste_ind,
                        verbose=False,
                        loo_train_steps=loo_train_steps,
                        loo_patience=loo_patience,
                        scaler=prep['scaler'],
                        pca_obj=prep['pca_obj'],
                        raw_labels_tensor=prep.get('raw_labels_tensor'),
                    )
            except Exception as e:
                print(f"  Trial {trial.number} taste {taste_ind} FAILED: {e}")
                return float('inf')

                # Now computing for poisson aicr:
            poisson_aic = info_criteria.get('poisson_aicr', float('nan'))
            if np.isnan(poisson_aic):
                print(f"  Trial {trial.number}: Poisson AICr is nan, "
                    f"falling back to poisson AIC")
                poisson_aic = info_criteria.get('poisson_aic', float('nan'))
                if np.isnan(poisson_aic):
                    print(f"  Trial {trial.number}: Poisson AIC is nan, "
                        f"falling back to Gaussian")
                    poisson_aic = info_criteria.get('aic', float('inf'))
            taste_aics.append(poisson_aic)
            # Report intermediate value for pruning
            running_mean = np.mean(taste_aics)
            trial.report(running_mean, step=taste_ind)
            if trial.should_prune():
                print(f"  Trial {trial.number} PRUNED after taste {taste_ind} "
                      f"(running mean AICr={running_mean:.2f})")
                raise optuna.TrialPruned()
            print(f"    Taste {taste_ind}: Poisson AICr={poisson_aic:.2f}")

        elapsed = time.time() - trial_start
        mean_aic = np.mean(taste_aics)

        trial.set_user_attr('per_taste_aic', taste_aics)
        trial.set_user_attr('mean_aic', mean_aic)
        trial.set_user_attr('std_aic', float(np.std(taste_aics)))
        trial.set_user_attr('time_s', elapsed)


        #    aic = info_criteria.get('poisson_aic', float('nan'))
        #    if np.isnan(aic):
        #        aic = info_criteria.get('aic', float('inf'))
        #    taste_aics.append(aic)

            # Report intermediate value for pruning
        #    running_mean = np.mean(taste_aics)
        #    trial.report(running_mean, step=taste_ind)
        #    if trial.should_prune():
        #        print(f"  Trial {trial.number} PRUNED after taste {taste_ind} "
        #              f"(running mean AIC={running_mean:.2f})")
        #        raise optuna.TrialPruned()

        #    print(f"    Taste {taste_ind}: Poisson AIC={aic:.2f}")

        #mean_aic = np.mean(taste_aics)
        #elapsed = time.time() - trial_start

        #trial.set_user_attr('per_taste_aic', taste_aics)
        #trial.set_user_attr('mean_aic', mean_aic)
        #trial.set_user_attr('std_aic', float(np.std(taste_aics)))
        #trial.set_user_attr('time_s', elapsed)

        #trial.set_user_attr('raw_mse', float(np.mean((pred_long - raw_long)**2)))
        #trial.set_user_attr('raw_corr', float(np.corrcoef(pred_long.flatten(), 
        #                                             raw_long.flatten())[0,1]))

        print(f"  Trial {trial.number}: "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f}, "
              f"loss={params['loss_name']} | "
              f"Mean AICr={mean_aic:.2f} +/- {np.std(taste_aics):.2f} | "
              f"{elapsed:.1f}s")

        return mean_aic

    return objective


def create_multidataset_objective(all_dataset_preps, device, artifacts_dir,
                                   loo_train_steps, loo_patience):
    """
    Objective that evaluates across multiple datasets and tastes.
    
    all_dataset_preps: dict of {dataset_name: {taste_ind: prep_dict}}
    Returns mean Poisson AICr across all datasets and tastes.
    """
    def objective(trial):
        from optuna_sweep import define_search_space
        params = define_search_space(trial)
        criterion = get_criterion(params['loss_name'])
        trial_start = time.time()

        dataset_aics = []

        for ds_idx, (ds_name, taste_preps) in enumerate(
                sorted(all_dataset_preps.items())):
            taste_aics = []

            for taste_ind in sorted(taste_preps):
                prep = taste_preps[taste_ind]
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
                    n_folds=5,
                    seed=42,
                    verbose=False,
                    taste_ind=taste_ind,
                )
                # NOTE: have to account for the fact that different datasets are wildly differing 
                # in their size. As such, we need a way to ensure that none are disproportionately 
                # pushing the results one way or another.
                aic = info_criteria.get('poisson_aicr', float('nan'))
                if np.isnan(aic):
                    aic = info_criteria.get('poisson_aic', float('inf'))
                n_obs = info_criteria.get('n_observations', 1)
                # we normalize here to number of params so we can control things. 
                taste_aics.append(aic / n_obs)

            ds_mean = np.mean(taste_aics)
            dataset_aics.append(ds_mean)

            # Report for pruning: step = dataset index
            running_mean = np.mean(dataset_aics)
            trial.report(running_mean, step=ds_idx)
            if trial.should_prune():
                print(f"  Trial {trial.number} PRUNED after {ds_name}")
                raise optuna.TrialPruned()

            print(f"    {ds_name}: mean AICr={ds_mean:.2f}")

        overall_mean = np.mean(dataset_aics)
        elapsed = time.time() - trial_start

        trial.set_user_attr('per_dataset_aicr_normalized', dataset_aics)
        trial.set_user_attr('mean_aicr_normalized', overall_mean)
        trial.set_user_attr('std_aic', float(np.std(dataset_aics)))
        trial.set_user_attr('time_s', elapsed)

        print(f"  Trial {trial.number}: "
              f"hidden={params['hidden_size']}, layers={params['rnn_layers']}, "
              f"dropout={params['dropout']:.2f}, lr={params['lr']:.4f} | "
              f"Mean AICr={overall_mean:.2f} +/- {np.std(dataset_aics):.2f} | "
              f"{elapsed:.1f}s")

        return overall_mean

    return objective