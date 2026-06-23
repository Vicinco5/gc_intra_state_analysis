"""
Output saving: HDF5 and Parquet.
"""

import os
import numpy as np
import polars as pl
import tables
import json

def save_to_hdf5(hdf5_path, pred_firing_list, latent_out_list, bin_size):
    """
    Save predicted firing rates and latent outputs to the source HDF5 file.

    Args:
        hdf5_path: str, path to HDF5 file
        pred_firing_list: list of arrays, one per taste, shape (trials, neurons, time)
        latent_out_list: list of arrays, one per taste, shape (time, trials, hidden)
        bin_size: int
    """
    with tables.open_file(hdf5_path, 'r+') as hf5:
        if '/rnn_output' not in hf5:
            hf5.create_group('/', 'rnn_output', 'RNN Output')
        rnn_output = hf5.get_node('/rnn_output')

        for taste_ind, (pred, latent) in enumerate(
                zip(pred_firing_list, latent_out_list)):
            group_name = f'taste_{taste_ind}'
            if hasattr(rnn_output, group_name):
                hf5.remove_node(rnn_output, group_name, recursive=True)
            taste_group = hf5.create_group(
                rnn_output, group_name, f"Taste {taste_ind}"
            )
            hf5.create_array(taste_group, 'pred_firing', pred)
            hf5.create_array(taste_group, 'latent_out', latent)
            hf5.create_array(
                taste_group, 'pred_x',
                np.arange(pred.shape[-1]) * bin_size
            )


def save_latents_parquet(
        latent_out_list, dataset_name, artifacts_dir, pred_lat_dir):
    """
    Save latent vectors to parquet (local artifacts + shared directory).

    Args:
        latent_out_list: list of arrays (time, trials, hidden) per taste
        dataset_name: str
        artifacts_dir: str
        pred_lat_dir: str
    """
    all_latents = []
    for taste_ind, latent in enumerate(latent_out_list):
        num_trials, num_time, num_latent = latent.shape
        df = pl.DataFrame(
            latent.reshape(-1, num_latent),
            schema=[f'latent_dim_{i}' for i in range(num_latent)]
        )
        df = df.with_columns([
            pl.Series("taste", [taste_ind] * len(df)),
            pl.Series("trial", np.repeat(np.arange(num_trials), num_time)),
            pl.Series("time", np.tile(np.arange(num_time), num_trials)),
        ])
        all_latents.append(df)

    combined_df = pl.concat(all_latents)

    path1 = os.path.join(artifacts_dir, f'{dataset_name}_raw_latent_vectors.parquet')
    combined_df.write_parquet(path1)
    print(f"  Saved latent outputs: {path1}")

    path2 = os.path.join(pred_lat_dir, f'{dataset_name}_raw_latent_vectors.parquet')
    combined_df.write_parquet(path2)
    print(f"  Also saved to: {path2}")


def save_firing_parquet(
        pred_firing_list, dataset_name, artifacts_dir, pred_fr_dir):
    """
    Save predicted firing rates to parquet.

    Args:
        pred_firing_list: list of arrays (trials, neurons, time) per taste
        dataset_name: str
        artifacts_dir: str
        pred_fr_dir: str
    """
    all_firing = []
    neuron_counts = []

    for taste_ind, pred_firing in enumerate(pred_firing_list):
        num_trials, num_neurons, num_time = pred_firing.shape
        neuron_counts.append(num_neurons)

        reshaped = pred_firing.transpose(0, 2, 1).reshape(-1, num_neurons)
        df = pl.DataFrame(
            reshaped,
            schema=[f'neuron_{i}' for i in range(num_neurons)]
        )
        df = df.with_columns([
            pl.Series("taste", [taste_ind] * len(df)),
            pl.Series("trial", np.repeat(np.arange(num_trials), num_time)),
            pl.Series("time", np.tile(np.arange(num_time), num_trials)),
        ])
        all_firing.append(df)

    if len(set(neuron_counts)) != 1:
        raise ValueError(
            f"[ERROR] Neuron counts differ across tastes: {neuron_counts}"
        )

    combined_df = pl.concat(all_firing)

    path1 = os.path.join(artifacts_dir, f"{dataset_name}_raw_predicted_firing.parquet")
    combined_df.write_parquet(path1)
    print(f"  Saved predicted firing rates: {path1}")

    path2 = os.path.join(pred_fr_dir, f"{dataset_name}_raw_predicted_firing.parquet")
    combined_df.write_parquet(path2)
    print(f"  Also saved to: {path2}")

# helper to save metrics that I generate as a result of these various trainings: 
def save_metrics_json(info_criteria_all, dataset_name, params, output_dir,
                      master_path=None):
    """
    Extract scalar fit metrics per taste and write to JSON for supplementals.
    THIS IS GOING TO BE IMPORTANT WHEN I NEED THIS DATA TO JUSTIFY MY MODEL FITS 
    Writes a per-dataset JSON, and optionally appends/updates a master
    JSON keyed by dataset name so all datasets accumulate in one file.
    """
    # Scalar fields worth keeping (skip the big per-fold arrays)
    scalar_keys = [
        'log_likelihood', 'aic', 'bic', 'aicr',                 # Gaussian
        'poisson_log_likelihood', 'poisson_aic', 'poisson_bic', # Poisson
        'poisson_aicr',
        'loss_gaussian_corr', 'loss_poisson_corr',              # Model generalization consistency keys right here are pretty important 
        'n_params', 'n_observations', 'n_trials',
        'final_train_loss',
    ]

    def _clean(v):
        # Make numpy scalars JSON-safe; pass through None/str/float
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    dataset_metrics = {
        'dataset': dataset_name,
        'hyperparameters': {
            k: params.get(k) for k in
            ['bin_size', 'hidden_size', 'rnn_layers', 'dropout', 'lr',
             'loss_name', 'validation_mode', 'use_pca']
        },
        'per_taste': {},
    }

    for taste_ind, ic in info_criteria_all.items():
        dataset_metrics['per_taste'][str(taste_ind)] = {
            k: _clean(ic.get(k)) for k in scalar_keys if k in ic
        }

    # --- Per-dataset file ---
    per_ds_path = os.path.join(output_dir, f'fit_metrics_{dataset_name}.json')
    with open(per_ds_path, 'w') as f:
        json.dump(dataset_metrics, f, indent=2)
    print(f"  [INFO] Saved fit metrics: {per_ds_path}")

    # --- Optional master aggregate (keyed by dataset)---
    # depsite being optional, you will definitely want this 
    if master_path is not None:
        master = {}
        if os.path.exists(master_path):
            try:
                with open(master_path, 'r') as f:
                    master = json.load(f)
            except Exception:
                master = {}
        master[dataset_name] = dataset_metrics
        with open(master_path, 'w') as f:
            json.dump(master, f, indent=2)
        print(f"  [INFO] Updated master metrics: {master_path}")

    return dataset_metrics