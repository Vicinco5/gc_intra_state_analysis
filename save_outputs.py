"""
Output saving: HDF5 and Parquet.
"""

import os
import numpy as np
import polars as pl
import tables


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