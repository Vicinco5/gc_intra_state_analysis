"""
Part of a concerted effort to clean this script up in general to make it that much 
easier to do model comparisons, etc

run_rnn.py — Main orchestration script.

Loops over H5 datasets and tastes, calling modular functions for:
    - Config loading
    - Preprocessing
    - Training / loading
    - Postprocessing (inverse PCA/scaling)
    - Visualization
    - Saving outputs

NOTE: On the params json: 
Config parameter "validation_mode":
    - "split" (default): standard train/test split
    - "loo": leave-one-out CV for AIC/BIC, then retrain on all trials

"""
import os
import numpy as np
import torch

from config_loader import load_config
from preprocessing import preprocess_taste, train_test_split_trials
from run_training import train_or_load, run_prediction, loo_then_train
from postprocessing import reconstruct_firing
from visualizations import (
    plot_inputs, plot_loss_curves, plot_firing_overview,
    plot_mean_firing, plot_latent_factors, plot_trial_latents,
    plot_individual_neurons, plot_mean_neurons_across_tastes,
    plot_pred_vs_true_neurons, plot_aic_bic_summary, plot_loo_diagnostics,
)
from save_outputs import save_to_hdf5, save_latents_parquet, save_firing_parquet
from neuron_eval import evaluate_neurons

from fixedpoint import run_fixed_point_analysis

from train import MSELoss, smooth_MSELoss

def get_criterion(loss_name):
    if loss_name == 'smooth':
        return smooth_MSELoss(alpha=0.05)
    return MSELoss()

import json
# load in the configs: 
config_path = '/home/vincent/Senior thesis work/blechRNN-master/config/blechrnn_config.json'
# ----------------------------------------------------------------
# Optuna override (set to True to use optimized params from Optuna)
# NOTE: you really need to know which ones you want to use. Also, be aware that this applies to ALL datasets (hehe). 
# ----------------------------------------------------------------
USE_OPTUNA_PARAMS = False
OPTUNA_PARAMS_PATH = '/home/vincent/Senior thesis work/blechRNN-master/jan2026validationR1_testing/optuna_optimization/AM26_4Tastes_200826_101430_repacked/optimized_params_used.json'
config, paths, params, criterion = load_config(config_path)

if USE_OPTUNA_PARAMS:
    with open(OPTUNA_PARAMS_PATH, 'r') as f:
        optuna_data = json.load(f)
    # Handle both formats: nested (timestamped export) or flat (optimized_params_used)
    if 'parameters' in optuna_data:
        optuna_params = optuna_data['parameters']
    else:
        optuna_params = optuna_data
    print(f"  [INFO] Overriding params with Optuna results from {OPTUNA_PARAMS_PATH}")
    if 'best_trial_number' in optuna_data:
        print(f"         Trial #{optuna_data['best_trial_number']}, "
              f"Poisson AIC={optuna_data.get('best_poisson_aic', '?')}")
    for key in ['hidden_size', 'rnn_layers', 'dropout', 'lr', 'loss_name']:
        if key in optuna_params:
            old_val = params.get(key)
            params[key] = optuna_params[key]
            print(f"         {key}: {old_val} -> {params[key]}")
    criterion = get_criterion(params['loss_name'])
    print(f"  [INFO] Final params: { {k: params[k] for k in ['hidden_size', 'rnn_layers', 'dropout', 'lr', 'loss_name']} }")

from ephys_data import ephys_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
validation_mode = params.get('validation_mode', 'split') # note 'split' is the default if nothing is specified. 
print(f"Validation mode: {validation_mode}")
# ----------------------------------------------------------------
# Loop over datasets
# ----------------------------------------------------------------

#if not USE_OPTUNA_PARAMS:
#    # original params
#    lr=0.001
#    rnn_layers=2
#    dropout=0.2
#    loss_name = 'mse'
#    rnn_layers = 2

for subdir in sorted(os.listdir(paths['h5_dir'])):
    full_subdir_path = os.path.join(paths['h5_dir'], subdir)
    if not os.path.isdir(full_subdir_path):
        continue

    h5_files = [f for f in os.listdir(full_subdir_path) if f.endswith(".h5")]
    if len(h5_files) != 1:
        print(f"Skipping {subdir} — expected 1 .h5 file, found {len(h5_files)}")
        continue

    dataset_name = os.path.splitext(h5_files[0])[0]
    print(f"\nProcessing: {dataset_name}")

    # --- Output directories ---
    output_path = os.path.join(paths['output_base_dir'], dataset_name)
    plots_dir = os.path.join(output_path, 'plots')
    artifacts_dir = os.path.join(output_path, 'artifacts')
    model_eval_dir = os.path.join(output_path, 'model_eval') # specicifically for the LOO stuff
    os.makedirs(model_eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Load spikes ---
    data = ephys_data(full_subdir_path)
    data.get_spikes()
    spike_array = np.stack(data.spikes)

    # --- Accumulators ---
    pred_firing_list = []
    latent_out_list = []
    binned_spikes_list = []
    conv_rate_list = []
    conv_x_list = []
    info_criteria_all = {}

    # ----------------------------------------------------------------
    # Loop over tastes
    # ----------------------------------------------------------------
    for taste_ind, taste_spikes in enumerate(spike_array):
        print(f"\n  Taste {taste_ind}")

        # --- Time slice ---
        taste_spikes = taste_spikes[..., params['time_lims'][0]:params['time_lims'][1]]

        # --- Preprocess ---
        prep = preprocess_taste(
            taste_spikes,
            bin_size=params['bin_size'],
            stim_time_val=paths['stim_time_val'],
            use_pca=params['use_pca'],
        )

        # --- Debug ---
        print(f"    Input shape: {prep['inputs_plus_context'].shape}")
        print(f"    input_size={prep['input_size']}, output_size={prep['output_size']}")

        # --- Plot inputs ---
        plot_inputs(
            prep['inputs_plus_context'], dataset_name, taste_ind, plots_dir
        )

        # --- Model path ---
        model_name = (f'taste_{taste_ind}_hidden_{params["hidden_size"]}'
                      f'_loss_{params["loss_name"]}')
        model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')

        # ==============================================================
        # Branch on validation mode
        # ==============================================================
        if validation_mode == 'loo':
            # LOO: no manual split needed — loo_then_train handles it
            net, loss, cross_val_loss, info_criteria = loo_then_train(
                inputs_tensor=prep['inputs_tensor'],
                labels_tensor=prep['labels_tensor'],
                input_size=prep['input_size'],
                hidden_size=params['hidden_size'],
                output_size=prep['output_size'],
                device=device,
                criterion=criterion,
                train_steps=params['train_steps'],
                patience=params['patience'],
                retrain=params['retrain'],
                model_save_path=model_save_path,
                artifacts_dir=artifacts_dir,
                taste_ind=taste_ind,
                loo_train_steps = params.get('loo_train_steps'),
                loo_patience=params.get('loo_patience'), 
                scaler=prep['scaler'],
                pca_obj=prep['pca_obj'], 
                raw_labels_tensor=prep['raw_labels_tensor'],
            )

        else:
            # Standard split
            (train_inputs, train_labels,
             test_inputs, test_labels,
             train_inds, test_inds) = train_test_split_trials(
                prep['inputs_tensor'], prep['labels_tensor'],
                split_ratio=params['train_test_split'],
            )

            net, loss, cross_val_loss, info_criteria = train_or_load(
                input_size=prep['input_size'],
                hidden_size=params['hidden_size'],
                output_size=prep['output_size'],
                train_inputs=train_inputs.to(device),
                train_labels=train_labels.to(device),
                test_inputs=test_inputs.to(device),
                test_labels=test_labels.to(device),
                device=device,
                criterion=criterion,
                train_steps=params['train_steps'],
                patience=params['patience'],
                retrain=params['retrain'],
                model_save_path=model_save_path,
                artifacts_dir=artifacts_dir,
                taste_ind=taste_ind,
            )

        info_criteria_all[taste_ind] = info_criteria
        
        # --- LOO diagnostics (only in LOO mode) ---
        if validation_mode == 'loo':
            plot_loo_diagnostics(info_criteria, dataset_name, taste_ind, model_eval_dir)

        # --- Predict on full data ---
        outs, latent_outs = run_prediction(
            net, prep['inputs_tensor'], device
        )
        latent_out_list.append(latent_outs)

        # --- Reconstruct firing rates ---
        pred_firing = reconstruct_firing(
            outs,
            scaler=prep['scaler'],
            pca_obj=prep['pca_obj'],
            num_neurons=prep['num_neurons'],
            use_pca=params['use_pca'],
        )
        pred_firing_list.append(pred_firing)
        binned_spikes_list.append(prep['binned_spikes'])
        # now also running a quick and cheeky evaluate neurons to figure out if the model fits ok: 
        evaluate_neurons(
            net=net,
            inputs_tensor=prep['inputs_tensor'],
            labels_tensor=prep['labels_tensor'],
            raw_labels_tensor=prep['raw_labels_tensor'],
            scaler=prep['scaler'],
            pca_obj=prep['pca_obj'],
            binned_spikes=prep['binned_spikes'],
            dataset_name=dataset_name,
            taste_ind=taste_ind,
            output_dir=model_eval_dir,
            device=device,
        )

        # --- Convolved firing rate (for comparison plots) ---
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

        # --- Per-taste plots ---
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
        # also do some fixed point tomfoolery: 
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

    # ----------------------------------------------------------------
    # Post-taste-loop: cross-taste plots and saving
    # ----------------------------------------------------------------
    plot_mean_neurons_across_tastes(
        spike_array, pred_firing_list, binned_spikes_list,
        conv_rate_list, conv_x_list, params['bin_size'],
        paths['stim_time_val'], params['time_lims'],
        dataset_name, plots_dir
    )
    plot_pred_vs_true_neurons(pred_firing_list, binned_spikes_list, plots_dir)
    plot_aic_bic_summary(info_criteria_all, dataset_name, model_eval_dir)

    # --- Save outputs ---
    save_to_hdf5(
        data.hdf5_path, pred_firing_list, latent_out_list,
        params['bin_size']
    )
    save_latents_parquet(
        latent_out_list, dataset_name, artifacts_dir,
        paths['pred_lat_dir']
    )
    save_firing_parquet(
        pred_firing_list, dataset_name, artifacts_dir,
        paths['pred_fr_dir']
    )

    print(f"\n  Done: {dataset_name}")