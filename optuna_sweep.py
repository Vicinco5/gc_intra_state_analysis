"""
Optuna hyperparameter optimization for autoencoderRNN.

Uses Bayesian optimization (TPE sampler) to search the joint hyperparameter
space, targeting Poisson AIC as the objective.

Target metric: Poisson AIC (lower is better)
    AIC = 2k - 2LL
    Minimizing AIC selects the model with the best predictive accuracy
    relative to its complexity.

Requirements:
    pip install optuna

Usage:
    python optuna_sweep.py

Configure by editing the CONFIG section below.

Outputs:
    - optuna_study_taste_<N>.db                  Resumable study database
    - optuna_results_taste_<N>.txt               Human-readable report
    - optuna_results_taste_<N>.json              Machine-readable results
    - optuna_params_<datetime>.json              Best params (timestamped)
    - optuna_history_taste_<N>.png               AIC over trials
    - optuna_params_taste_<N>.png                AIC vs each parameter
    - optuna_importance_taste_<N>.png            Parameter importance
    - optuna_optimization/<dataset>/...          Full pipeline results
                                                  (if RUN_OPTIMIZED=True)

 SOME ADDITIONLS 
 Multi-objective Bayesian optimization (TPE sampler) targeting:
    1. Poisson AICr / observation (minimize) — predictive accuracy
    2. Loss-LL correlation (maximize) — generalization consistency
 
Finds Pareto front: configs where you can't improve one metric
without worsening the other.


"""

#############################################################################
# NOTE: On what dataset we're ultimately optimizing for here
# each dataset is going to be different. Ultimately, they will likely all want slightly different fits. 
# because we want to be able to compare across all of them, my logic is that I think I'd want to train on the *best* dataset to train on 
# this means a dataset that isn't boring. iirc either 101430 or even 105700 (tbh I think the second one)-- as it contains a multitude of data. 

# I think the DS that I try to optimize params off of really, REALLY matters. 

# For science reasons (abu and I have discussed), MSE error is probably what we need to be sticking with. 


import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from optuna.trial import FixedTrial
try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)
 
# ================================================================
# CONFIG — edit everything here
# ================================================================
 
CONFIG_PATH = '/home/vincent/Senior thesis work/blechRNN-master/src/rnn_v2/blechrnn_config.json'
 
# Which taste to optimize (0, 1, 2, 3) — only used when ALL_TASTES=False
SWEEP_TASTE = 0
 
# Number of Optuna trials
N_TRIALS = 37
 
# LOO fold settings
LOO_TRAIN_STEPS = 12000
LOO_PATIENCE = 40
 
# Study name
STUDY_NAME = 'blechrnn_poisson_aic_optim_binsize_v15_multidata_one_layer_only'
 
# Resume a previous study?
RESUME = True
 
# After optimization, run the full pipeline with best params?
RUN_OPTIMIZED = True
 
# Optimize across all tastes jointly?
ALL_TASTES = True
 
# Optimize across multiple datasets?
MULTI_DATASET = True
 
DATASET_SUBDIRS = [
    'AM12_4Tastes_191105_083246_repacked',
    'AM26_4Tastes_200826_101430_repacked',
    'AM35_4Tastes_201231_105700_repacked_repacked',
]
 
# Which correlation metric for the second objective?
# 'poisson' = loss vs held-out Poisson LL (raw count space)
# 'gaussian' = loss vs held-out Gaussian LL (z-scored space)
CORRELATION_METRIC = 'poisson'
# WHY I BIAS TOWARDS CORRELATION: 
# AICr is great and all, but correlation is the informative metric with respect to how well the model generalizes to the held-out data 
# This generalization consistency is extremely important for the repeatability of the dynamics we observe. 
# in general, I am (very) happy with this balance
# reason being, much more extreme bias to correlation and I begin to encourage the model to train on noise. 
# AICr is about parameter efficiency (which would imply structure beyond just noise), so it is important to guard against noise. 
PARETO_AICR_WEIGHT = 0.4 # what combination of metric we're optimizing for
# on run 13, we had a trial that was the best r correlate and effectively tied for the best AICr (1.61 (16) vs 1.60)
# trial 16: 
#  16    AICr/obs 1.610216  r:  0.795    bin size 25    hidden_state  10      layers 3       ?  355.3s

# see: 
# 0.0 = pure correlation, 1.0 = pure AICr, 0.5 = balanced
 
# ================================================================
# SEARCH SPACE
# ================================================================
 
def define_search_space(trial):
    """defines the search space-- espeically with respect to bins. I may want to elim some of these as bin is so important"""
    return {
        # 'bin_size': 25 # fixed bin size as that is clearly what is best
        'bin_size':    trial.suggest_categorical('bin_size', [25, 30, 35, 40, 45, 50]),
        'hidden_size': trial.suggest_int('hidden_size', 8, 50, log=True),
        #'rnn_layers':  trial.suggest_int('rnn_layers', 1, 4),
        'rnn_layers': 1, # literally fixing the RNN to 1 layer
        # fixing these for now to keep the search more reasonable: 
        #'dropout':     0.2,
        'dropout':     0.0, # with one layer dropout is 0
        #'lr':          0.001,
        'loss_name':   'mse',
        #'dropout':     trial.suggest_float('dropout', 0.1, 0.3, step=0.05),
        'lr':          trial.suggest_float('lr', 5e-4, 4e-3, log=True),
        #'loss_name':   trial.suggest_categorical('loss_name', ['mse']),
    }
 
# ================================================================
# End of config
# ================================================================
 
from config_loader import load_config
from preprocessing import preprocess_taste
from ephys_data import ephys_data
from optuna_core import (
    run_optimized_pipeline,
)
from optuna_multiobjective import (
    create_multiobjective,
    create_multitaste_multiobjective,
    create_multidataset_multiobjective,
    plot_pareto_front,
    save_multiobjective_results,
)

def _select_from_pareto(study, aicr_weight=0.4):
    """Select best trial from Pareto front using weighted combination."""
    pareto = study.best_trials
    if len(pareto) == 1:
        return pareto[0]

    aicrs = [t.values[0] for t in pareto]
    corrs = [-t.values[1] for t in pareto]

    aicr_min, aicr_max = min(aicrs), max(aicrs)
    corr_min, corr_max = min(corrs), max(corrs)
    aicr_range = aicr_max - aicr_min + 1e-10
    corr_range = corr_max - corr_min + 1e-10

    best_score = -1
    best_trial = pareto[0]
    for t, a, c in zip(pareto, aicrs, corrs):
        a_norm = 1 - (a - aicr_min) / aicr_range
        c_norm = (c - corr_min) / corr_range
        score = aicr_weight * a_norm + (1 - aicr_weight) * c_norm
        if score > best_score:
            best_score = score
            best_trial = t

    print(f"  Selected trial #{best_trial.number} from Pareto front "
          f"(AICr/obs={best_trial.values[0]:.6f}, "
          f"r={-best_trial.values[1]:.3f}, "
          f"combined score={best_score:.3f}, "
          f"weight={aicr_weight})")
    return best_trial
 
def _print_pareto_results(study, corr_label):
    """Print Pareto front summary."""
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Pareto-optimal trials: {len(study.best_trials)}")
    for t in study.best_trials:
        print(f"  #{t.number}: AICr/obs={t.values[0]:.6f}, "
              f"r({corr_label})={-t.values[1]:.3f} | {t.params}")
    print(f"{'=' * 60}\n")
 
 
def _export_best_params(study, taste_label, output_dir):
    """Export best params from Pareto front using weighted selection."""
    best_trial = _select_from_pareto(study, aicr_weight=PARETO_AICR_WEIGHT)
    full_params = _full_params(best_trial)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export = {
        'timestamp': timestamp,
        'study_name': STUDY_NAME,
        'taste_label': taste_label,
        'objectives': ['poisson_aicr_per_obs', f'neg_loss_{CORRELATION_METRIC}_corr'],
        'selection_criterion': f'weighted_pareto (aicr_weight={PARETO_AICR_WEIGHT})',
        'selected_trial': best_trial.number,
        'selected_aicr_per_obs': best_trial.values[0],
        'selected_correlation': -best_trial.values[1],
        'parameters': full_params, # now explicitly getting the full params 
        #'parameters': {**best_trial.params, 'dropout': 0.2, 'lr': 0.001, 'loss_name': 'mse'},
        'n_pareto_trials': len(study.best_trials),
        'n_trials_total': len(study.trials),
    }
    export_path = os.path.join(output_dir, f'optuna_params_{timestamp}.json')
    with open(export_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"  Saved best params: {export_path}")
    return best_trial
 
def _full_params(best_trial):
    """Complete param dict including fixed (non-suggested) values, by replaying
    define_search_space with the trial's recorded suggested params."""
    return define_search_space(FixedTrial(best_trial.params))

def _run_optimized_on_datasets(study, dataset_subdirs, paths, params):
    best_trial = _select_from_pareto(study, aicr_weight=PARETO_AICR_WEIGHT)
    corr_label = 'Poisson' if CORRELATION_METRIC == 'poisson' else 'Gaussian'
    full_params = _full_params(best_trial)          # includes rnn_layers=1, dropout=0.0, loss_name

    # Override bin_size with the selected value so the final pipeline re-bins correctly
    run_params = dict(params)
    if 'bin_size' in best_trial.params:
        run_params['bin_size'] = best_trial.params['bin_size']
        print(f"  Final runs will use bin_size={run_params['bin_size']}")

    print(f"\n  Using trial #{best_trial.number} for optimized runs "
          f"(r({corr_label})={-best_trial.values[1]:.3f}, "
          f"AICr/obs={best_trial.values[0]:.6f})")

    for subdir in dataset_subdirs:
        full_path = os.path.join(paths['h5_dir'], subdir)
        h5_files = [f for f in os.listdir(full_path) if f.endswith('.h5')]
        if not h5_files:
            print(f"  Skipping {subdir} — no .h5 files")
            continue
        ds_name = os.path.splitext(h5_files[0])[0]
        print(f"\n  Running optimized pipeline for {ds_name}...")
        run_optimized_pipeline(
            #best_params=best_trial.params,
            best_params=full_params, # forcing to use params it had found
            paths=paths, 
            params=run_params,     # <-- Now passing in the optimization for bins specifically 
            dataset_name=ds_name,
            full_subdir_path=full_path,
            output_base=paths['output_base_dir'],
            loo_train_steps=LOO_TRAIN_STEPS,
            loo_patience=LOO_PATIENCE,
        )
 
 
def main():
    config, paths, params, _ = load_config(CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    corr_label = 'Poisson' if CORRELATION_METRIC == 'poisson' else 'Gaussian'
 
    # =============================================================
    # MULTI-DATASET path (independent of per-dataset loop)
    # =============================================================
    if MULTI_DATASET:
            output_dir = os.path.join(paths['output_base_dir'], 'optuna_multidataset')
            os.makedirs(output_dir, exist_ok=True)
            artifacts_dir = os.path.join(output_dir, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)

            # Bin sizes to test — MUST match the categorical in define_search_space
            BIN_SIZES = [25, 30, 35, 40, 45, 50]

            # Load raw spikes once per dataset, reuse across bin sizes
            raw_spikes_by_ds = {}
            for subdir in DATASET_SUBDIRS:
                full_path = os.path.join(paths['h5_dir'], subdir)
                h5_files = [f for f in os.listdir(full_path) if f.endswith('.h5')]
                ds_name = os.path.splitext(h5_files[0])[0]
                data = ephys_data(full_path)
                data.get_spikes()
                raw_spikes_by_ds[ds_name] = np.stack(data.spikes)
                print(f"  Loaded spikes for {ds_name}")

            # Pre-compute preprocessing for each bin size
            all_preps_by_bin = {}
            for bs in BIN_SIZES:
                all_dataset_preps = {}
                for ds_name, spike_array in raw_spikes_by_ds.items():
                    taste_preps = {}
                    for ti in range(len(spike_array)):
                        t_spikes = spike_array[ti][...,
                            params['time_lims'][0]:params['time_lims'][1]]
                        taste_preps[ti] = preprocess_taste(
                            t_spikes, bin_size=bs,
                            stim_time_val=paths['stim_time_val'],
                            use_pca=params['use_pca'],
                        )
                    all_dataset_preps[ds_name] = taste_preps
                all_preps_by_bin[bs] = all_dataset_preps
                print(f"  Preprocessed all datasets at bin_size={bs}")

            # Multi-objective over bin size + model params
            objective = create_multidataset_multiobjective(
                all_preps_by_bin, device, artifacts_dir,
                LOO_TRAIN_STEPS, LOO_PATIENCE,
                corr_metric=CORRELATION_METRIC,
            )

            taste_label = 'multidataset'
            db_path = os.path.join(output_dir, f'optuna_study_{taste_label}.db')
            storage = f'sqlite:///{db_path}'

            study = optuna.create_study(
                study_name=STUDY_NAME, storage=storage,
                directions=['minimize', 'minimize'],
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=RESUME,
            )

            print(f"\n{'=' * 60}")
            print(f"Optuna multi-dataset study: {N_TRIALS} trials")
            print(f"Datasets: {list(raw_spikes_by_ds.keys())}")
            print(f"Bin sizes: {BIN_SIZES}")
            print(f"Objectives: Poisson AICr/obs + {corr_label} loss-LL correlation")
            print(f"{'=' * 60}\n")

            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

            _print_pareto_results(study, corr_label)
            save_multiobjective_results(study, taste_label, output_dir)
            plot_pareto_front(study, taste_label, output_dir, corr_metric=CORRELATION_METRIC)
            _export_best_params(study, taste_label, output_dir)

            if RUN_OPTIMIZED:
                _run_optimized_on_datasets(study, DATASET_SUBDIRS, paths, params)

            return # no per dataset loop
 
    # =============================================================
    # PER-DATASET loop (MULTI_DATASET is False)
    # =============================================================
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
 
        output_dir = os.path.join(paths['output_base_dir'], dataset_name,
                                  'optuna_results')
        os.makedirs(output_dir, exist_ok=True)
        artifacts_dir = os.path.join(output_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
 
        if ALL_TASTES:
            # --- All tastes, single dataset ---
            all_preps = {}
            for ti in range(len(spike_array)):
                t_spikes = spike_array[ti][...,
                    params['time_lims'][0]:params['time_lims'][1]]
                all_preps[ti] = preprocess_taste(
                    t_spikes,
                    bin_size=params['bin_size'],
                    stim_time_val=paths['stim_time_val'],
                    use_pca=params['use_pca'],
                )
            taste_label = 'all'
 
            objective = create_multitaste_multiobjective(
                all_preps, device, artifacts_dir,
                LOO_TRAIN_STEPS, LOO_PATIENCE,
                corr_metric=CORRELATION_METRIC,
            )
 
            db_path = os.path.join(output_dir,
                                   f'optuna_study_taste_{taste_label}.db')
            storage = f'sqlite:///{db_path}'
 
            study = optuna.create_study(
                study_name=STUDY_NAME, storage=storage,
                directions=['minimize', 'minimize'],
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=RESUME,
            )
 
        else:
            # --- Single taste ---
            taste_ind = SWEEP_TASTE
            taste_spikes = spike_array[taste_ind]
            taste_spikes = taste_spikes[...,
                                        params['time_lims'][0]:params['time_lims'][1]]
            prep = preprocess_taste(
                taste_spikes,
                bin_size=params['bin_size'],
                stim_time_val=paths['stim_time_val'],
                use_pca=params['use_pca'],
            )
            taste_label = str(taste_ind)
 
            objective = create_multiobjective(
                prep, device, taste_ind, artifacts_dir,
                LOO_TRAIN_STEPS, LOO_PATIENCE,
                corr_metric=CORRELATION_METRIC,
            )
 
            db_path = os.path.join(output_dir,
                                   f'optuna_study_taste_{taste_ind}.db')
            storage = f'sqlite:///{db_path}'
 
            study = optuna.create_study(
                study_name=STUDY_NAME, storage=storage,
                directions=['minimize', 'minimize'],
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=RESUME,
            )
 
        print(f"\n{'=' * 60}")
        print(f"Optuna study: {N_TRIALS} trials, taste={taste_label}")
        print(f"Objectives: Poisson AICr/obs + {corr_label} loss-LL correlation")
        print(f"{'=' * 60}\n")
 
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
 
        _print_pareto_results(study, corr_label)
        save_multiobjective_results(study, taste_label, output_dir)
        plot_pareto_front(study, taste_label, output_dir, corr_metric=CORRELATION_METRIC)
        best_trial = _export_best_params(study, taste_label, output_dir)
 
        if RUN_OPTIMIZED:
            print(f"\n  Using trial #{best_trial.number} for optimized run "
                  f"(r({corr_label})={-best_trial.values[1]:.3f})")
            run_optimized_pipeline(
                best_params=best_trial.params,
                paths=paths, params=params,
                dataset_name=dataset_name,
                full_subdir_path=full_subdir_path,
                output_base=paths['output_base_dir'],
                loo_train_steps=LOO_TRAIN_STEPS,
                loo_patience=LOO_PATIENCE,
            )
 
        # Only process first dataset
        break
 
 
if __name__ == '__main__':
    main()