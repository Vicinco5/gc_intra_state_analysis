#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:19:49 2025

@author: vincentcalia-bogan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:07:31 2024

@author: vincentcalia-bogan
"""

" Spike sorting with Pytau state transitions "

## NON-VINCENT MODULE IMPORTS ##
# 17 datasets 365 neurons total
import os, os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import ttest_ind
import polars as pl
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from matplotlib_venn import venn3
from scipy.interpolate import interp1d
import math
import matplotlib.lines as mlines

## VINCENT MODULE IMPORTS aka VIMPORTS##
# TODO: Fix these file imports later so not as hard coded, but that's a later project #
# MODULE DIRECTORIES
project_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work"
submodule_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions"
# Check if paths already exist in sys.path
sys.path.append(submodule_dir)
sys.path.append(project_dir)
# NOTE: DO NOT RUN TWO FUNCTIONS TO ONE DIR. WILL END POORLY
# Importing Vinmodules
from spike_train_to_npz import find_h5_files, extract_to_npz
from extract_npz import extract_from_npz
from unpkl_generator import unpickle_changepoints
from unpkl_generator import extract_valid_changepoints

# from interpolation_xr import interpolate_and_average_all_tastes, interpolate_firing_rates # this is actually a deprecated func now
from calc_firing_rate_with_states import calc_fr_states

# new firing rate class with related calls:
from calc_fr_class_war_unwar import CalcFRStates


from generate_parquet_sig_all import (
    sig_neurons_hz,
    sig_neurons_ttest,
    consolidate_all_neuron_data,
)

from generate_parquet_sig_all import (
    sig_neurons_war_hz,
    sig_neurons_war_ttest,
    consolidate_all_neuron_war_data,
)
from read_parquets import read_parquet_files_into_dict, all_nrns_to_df
from serialize_overlap import serialized_neuron_df, create_overlap_dataframes

from find_extract_info import find_copy_h5info, process_info_files, modify_tastes

# plotting funcs that are currently unused
# from current_plotting_funcs import plt_inter_cpfr_avg_nrn_line, plt_inter_cpfr_avg_tr_line, plt_changepoint_hist, plt_changepoints_scat
# from current_plotting_funcs import plt_single_neuron_firing_unwarped_heatmap, plt_single_neuron_firing_unwarped_line, plt_single_neuron_firing_warped_line

# PCA Plotting funcs -- currently deprecated
# from taste_sep_pca_plotting_funcs import plot_averaged_pca_trajectories_sep_taste, plot_averaged_pca_heatmaps_all_taste
# from taste_sep_pca_plotting_funcs import plot_averaged_pca_trajectories_thresholded, plot_pca_scatter, plot_pca_scatter_3d, calculate_pca_distances, plot_euclidean_distances_bar, plot_explained_variance_combined_by_taste

# other RNN-type things:
# this is a work in progress class that does not currently work
#   from RNNLatentprocessing import RNNLatentProcessor


# single-neuron plotting functions --

## FILE PATHS FOR DATA WRANGLING -- OTHERWISE WILL HAVE TO SPECIFY EVERY TIME ##
file_path = "/Volumes/T7 Shield/spikesorting"
spike_trains_path = "/spike_trains"
npz_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train npz data"
pkl_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/pkl files/"
info_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train info data"
# dir paths for plotting unwarped and warped data
output_dir_unwarped = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/uw-s-tr-th-1hz-line"
output_dir_warped = ""
sig_ttest_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet"
sig_war_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped"
sig_hz_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet"
all_nrns_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet"
all_war_nrns_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped"

sig_ttest_parquet_path_w = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped"
all_nrns_parquet_path_w = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped"

# this dir contains every single inferred firing rate in parquet form for each of the datasets. Here's to hoping to god that the data lines up
# reasonably well

# Notes on RNN:
rnn_inf_fr_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/newRNN/infer_fr_parquet"
# so it seems, the rnn files are of dim (something , 279) wherein the 279 bit appears to be time. This is not going to be trivial to slice up.

# modifying the names of tastes for legibility on graphs. The string "modified tastes" contains
# the tastes modified from "dataset_tastes".
taste_replacements = {
    "nacl": "NaCl",
    "suc": "Sucrose",
    "ca": "Citric Acid",
    "qhcl": "Quinine",
}
epoch_labels = ("Identification", "Palatability", "Decision", "2000 ms Post-Stimulus")

## NECESARY PARAMETERS ##
window_length = 250
step_size = 25
alpha = 0.05  # alpha for any null hypothesis tests
threshold_hz = 2.0  # threshold firing rate freuqency for various funcs

# this is a pretty janky work-around; fix this later plz
for data in extract_from_npz(npz_path):
    if isinstance(data, tuple):
        spike_array, dataset_num, index, key = data
        print(f"Dataset number: {dataset_num}")
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(
            pkl_path, [(spike_array, dataset_num, index, key)]
        )

## CHECKING IF INTERMEDIATE FILES EXIST AND IF SO DO NOT REGENERATE THEM ## (saves time lol)
npz_files_exist = any(file.endswith(".npz") for file in os.listdir(npz_path))
if npz_files_exist:
    print(
        ".npz files containing spike trains already exist in npz_path. Skipping extraction from h5 files."
    )
    # Perform certain actions when npz files already exist (pass certain functions)
else:
    print("No .npz files found in npz_path. Running functions to generate npz files.")
    # Run functions to generate npz files
    h5_files = find_h5_files(file_path)  # pulling h5 file paths
    save_data = extract_to_npz(
        h5_files, file_path, spike_trains_path, npz_path
    )  # saving npz files to save location
# checking for info files
info_files_exist = any(file.endswith(".info") for file in os.listdir(info_path))
if info_files_exist:
    print(".info files already exist in info_path; skipping re-copying them")
else:
    print("No .info files found in info_path. Extracting info_files")
    info_files = find_copy_h5info(file_path, info_path)
    print(f"Found and copied {len(info_files)} .info files.")
# checking for sig_parquet files -- unwarped
sig_parquet_exist = any(
    file.endswith(".parquet") for file in os.listdir(sig_ttest_parquet_path)
)
if sig_parquet_exist:
    print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
else:
    print("No .parquet files found in sig_parquet_path. Extracting info_files")
    sig_ttest = sig_neurons_ttest(
        npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size
    )
    sig_2hz = sig_neurons_hz(
        npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size
    )
    # print(f"Found and copied {len(sig_ttest)} .parquet files.")
    # print(f"Found and copied {len(sig_2hz)} .parquet files.")
# checking for parquet file with all neuron data
consolidated_parquet_exist = any(
    file.endswith(".parquet") for file in os.listdir(all_nrns_parquet_path)
)
if consolidated_parquet_exist:
    print(
        ".parquet files already exist in all_nrns_parquet_path; skipping re-copying them"
    )
else:
    print("No .info files found in all_nrns_parquet_path. Extracting info_files")
    consolidated_data = consolidate_all_neuron_data(
        npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size
    )
    # print(f"Found and copied {len(consolidated_data)} .parquet files.")

# unwarped data
# reading data back from parquets
sig_nrns_dict = read_parquet_files_into_dict(sig_ttest_parquet_path)

# dataframe of all neuron data
all_data_df = all_nrns_to_df(all_nrns_parquet_path)
serialized_nrns_df = serialized_neuron_df(
    npz_path, pkl_path
)  # just counts serialized nrns
sig_overlap = create_overlap_dataframes(
    serialized_nrns_df, sig_nrns_dict
)  # overlap of nrns


#####
# for warped data # not working quite yet
sig_parquet_exist_w = any(
    file.endswith(".parquet") for file in os.listdir(sig_ttest_parquet_path_w)
)
if sig_parquet_exist_w:
    print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
else:
    print("No .parquet files found in sig_parquet_path. Extracting info_files")
    sig_ttest_w = sig_neurons_war_ttest(
        npz_path, pkl_path, sig_ttest_parquet_path_w, alpha, window_length, step_size
    )
    sig_2hz_w = sig_neurons_war_hz(
        npz_path,
        pkl_path,
        sig_ttest_parquet_path_w,
        threshold_hz,
        window_length,
        step_size,
    )
    # print(f"Found and copied {len(sig_ttest)} .parquet files.")
    # print(f"Found and copied {len(sig_2hz)} .parquet files.")
# checking for parquet file with all neuron data
consolidated_parquet_exist_w = any(
    file.endswith(".parquet") for file in os.listdir(all_nrns_parquet_path_w)
)
if consolidated_parquet_exist_w:
    print(
        ".parquet files already exist in all_nrns_parquet_path; skipping re-copying them"
    )
else:
    print("No .info files found in all_nrns_parquet_path. Extracting info_files")
    consolidated_data_w = consolidate_all_neuron_war_data(
        npz_path, pkl_path, all_nrns_parquet_path_w, window_length, step_size
    )
    # print(f"Found and copied {len(consolidated_data)} .parquet files.")

# reading data back from parquets
sig_nrns_dict_w = read_parquet_files_into_dict(sig_war_parquet_path)
# dataframe of all neuron data
all_data_w_df = all_nrns_to_df(all_war_nrns_parquet_path)
serialized_w_nrns_df = serialized_neuron_df(
    npz_path, pkl_path
)  # just counts serialized nrns
sig_w_overlap = create_overlap_dataframes(
    serialized_w_nrns_df, sig_nrns_dict_w
)  # overlap of nrns


###ONCE THE .NPZ FILES ARE GENERATED, YOU CAN WORK OFF THE LOCAL STORAGE###
# lol jk have to get the rnn stuff

###  Inferred latents ### All about the infrence of latents

latent_path = "/Volumes/T7 Shield/RNN CODE FROM BIG PC/output/latent_parquet"

latent_dict_rnn = read_parquet_files_into_dict(latent_path)

# THIS CLASS now finally works and I'm adding shit to it as we speak. Make life easier.
from RNNLatentprocessing import RNNLatentProcessor

# init processor
processor = RNNLatentProcessor(
    parquet_dir="/Volumes/T7 Shield/octRNN/parquert",
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements=taste_replacements,
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/RNN_PROCESSING_PARQUETS",  # <-- NEW
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500,
)
# run full RNN analysis pipeline
epoch_dataframes_dict, robust_pca_95, robust_pca_full, first_derivs, second_derivs = (
    processor.full_pipeline(
        variance_threshold=95.0,
        compute_first_derivative=True,
        compute_second_derivative=True,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=True,  # <-- triggers automatic parquet saving!
    )
)

standardized_changepoints_dict = {
    key.replace("dataset_", "").split("_repacked.npz")[0]: value
    for key, value in processor.changepoints_dict.items()
}

# # Old, original loops that will load me up with the epoch dataframes that I need -- the above class now works well
# keeping it around as I'm not sure I trust the above class completely yet lol
# taste_latent = read_parquet_files_into_dict('/Volumes/T7 Shield/octRNN/parquert')
# # # taste_sep_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir'

# # # generating arrays I can work with:
# # # Assume `dataframes_dict` is the dictionary containing dataframes loaded from the Parquet files
# # processed_data = {}
# for dataset_name, df in taste_latent.items():
# #     # Get unique tastes and trials
# #     unique_tastes = df['taste'].unique()
# #     unique_trials = df['trial'].unique()

# #     # Identify latent dimension columns explicitly
# #     latent_columns = [col for col in df.columns if col.startswith("latent_dim_")]
# #     latent_dim = len(latent_columns)

# #     # Determine the number of time steps by counting entries for the first taste and trial
# #     num_time_steps = len(df.filter(pl.col("trial") == unique_trials[0])
# #                             .filter(pl.col("taste") == unique_tastes[0]))

# #     # Initialize array to store the result in shape (taste, trial, latent_dim, time)
# #     dataset_array = np.empty((len(unique_tastes), len(unique_trials), latent_dim, num_time_steps))

# #     for taste_idx, taste in enumerate(unique_tastes):
# #         for trial_idx, trial in enumerate(unique_trials):
# #             # Filter data for this taste and trial
# #             trial_data = df.filter((pl.col("taste") == taste) & (pl.col("trial") == trial))

# #             # Extract latent dimensions and reshape to (latent_dim, time)
# #             latent_values = trial_data.select(latent_columns).to_numpy().T  # Shape: (latent_dim, time)

# #             # Insert into the appropriate location in the dataset array
# #             dataset_array[taste_idx, trial_idx, :, :] = latent_values

# #     # Store the result for this dataset
# #     processed_data[dataset_name] = dataset_array

# # # storing changepoints
# # changepoints_dict = {}

# # for data in extract_from_npz(npz_path):
# #     if isinstance(data, tuple):
# #         spike_array, dataset_num, index, key = data
# #         print(f'Dataset number: {dataset_num}')

# #         # Derive dataset name or adjust this to match your naming convention
# #         dataset_name = f"dataset_{dataset_num}"

# #         # Process taste data
# #         dataset_tastes = process_info_files(info_path, dataset_num)
# #         modified_tastes = modify_tastes(dataset_tastes, taste_replacements)

# #         # Extract and process changepoints for the dataset
# #         extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)

# #         # Check if extracted_pkl is not None and has the expected shape
# #         if extracted_pkl is not None:
# #             try:
# #                 changepoints = extracted_pkl[:, 3]  # Select the relevant column for changepoints
# #                 changepoints_dict[dataset_name] = changepoints
# #             except IndexError:
# #                 print(f"Index error with dataset {dataset_name}: extracted_pkl shape is {extracted_pkl.shape}")
# #         else:
# #             print(f"No valid changepoints found for dataset {dataset_name}")
# # #processing in CP's now


# # # Constants for RNN timing and binning
# # BIN_SIZE_MS = 25  # Each bin represents 25 ms
# # START_TIME_MS = 1500  # Start time in ms
# # MAX_TIME_MS = 4500 # how long I let the RNN run... do post stim later ig

# epoch_dataframes_dict = {}

#  # Standardize changepoints_dict keys to core names without prefixes/suffixes
# standardized_changepoints_dict = {
# #     key.replace('dataset_', '').split('_repacked.npz')[0]: value
# #     for key, value in changepoints_dict.items()
# # }

# for dataset_name, data_array in processed_data.items():
# #     # Extract the core name from dataset_name
# #     core_dataset_name = dataset_name.split('_repacked_raw_latent_vectors')[0]

# #     # Check if core_dataset_name exists in the standardized changepoints dictionary
# #     if core_dataset_name in standardized_changepoints_dict:
# #         changepoints = standardized_changepoints_dict[core_dataset_name]

# #         # Get the number of tastes, time steps, latent dimensions, and trials
# #         num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape
# #         all_epochs = []  # Store epochs as DataFrames for concatenation later

# #         # Convert RNN time steps to ms
# #         time_in_ms = [(t * BIN_SIZE_MS) + START_TIME_MS for t in range(num_time_steps)]

# #         # Check if changepoints dimensions align with data_array dimensions
# #         if len(changepoints) < num_tastes:
# #             print(f"Mismatch in taste dimension for dataset {core_dataset_name}")
# #             continue

# #         # Iterate over each taste and trial
# #         for taste_idx in range(num_tastes):
# #             if len(changepoints[taste_idx]) < num_trials:
# #                 print(f"Mismatch in trial dimension for dataset {core_dataset_name}, taste index {taste_idx}")
# #                 continue

# #             for trial_idx in range(num_trials):
# #                 # Retrieve the changepoints for this taste and trial
# #                 trial_changepoints = changepoints[taste_idx][trial_idx]

# #                 # Split data into sub-arrays (epochs) based on ms-based changepoints
# #                 start_idx = 0
# #                 for cp_idx, changepoint_ms in enumerate(trial_changepoints):
# #                     # Find the closest RNN time index that matches or exceeds the changepoint in ms
# #                     end_idx = next((i for i, t in enumerate(time_in_ms) if t >= changepoint_ms), num_time_steps)

# #                     # Slice the data array from start_idx to end_idx
# #                     epoch_data = data_array[taste_idx, start_idx:end_idx, :, trial_idx]

# #                     # Create a DataFrame for this epoch
# #                     epoch_df = pl.DataFrame(
# #                         epoch_data,  # No need to transpose as time is now the first dimension
# #                         schema=[f'latent_dim_{i}' for i in range(latent_dim)]
# #                     )

# #                     # Add metadata columns for taste, trial, changepoint index, and time in ms
# #                     epoch_df = epoch_df.with_columns([
# #                         pl.Series("taste", [taste_idx] * len(epoch_df)),
# #                         pl.Series("trial", [trial_idx] * len(epoch_df)),
# #                         pl.Series("changepoint", [cp_idx] * len(epoch_df)),
# #                         pl.Series("time", time_in_ms[start_idx:end_idx])
# #                     ])

# #                     # Append the epoch DataFrame to the list
# #                     all_epochs.append(epoch_df)

# #                     # Update start_idx to the current changepoint for the next epoch
# #                     start_idx = end_idx

# #                 # Handle any remaining time after the last changepoint as the final epoch
# #                 if start_idx < num_time_steps:
# #                     epoch_data = data_array[taste_idx, start_idx:num_time_steps, :, trial_idx]

# #                     epoch_df = pl.DataFrame(
# #                         epoch_data,
# #                         schema=[f'latent_dim_{i}' for i in range(latent_dim)]
# #                     )
# #                     epoch_df = epoch_df.with_columns([
# #                         pl.Series("taste", [taste_idx] * len(epoch_df)),
# #                         pl.Series("trial", [trial_idx] * len(epoch_df)),
# #                         pl.Series("changepoint", [len(trial_changepoints)] * len(epoch_df)),
# #                         pl.Series("time", time_in_ms[start_idx:num_time_steps])
# #                     ])
# #                     all_epochs.append(epoch_df)

# #         # Concatenate all epochs for the current dataset
# #         dataset_epoch_df = pl.concat(all_epochs)

# #         # Store the concatenated DataFrame in the dictionary
# #         epoch_dataframes_dict[dataset_name] = dataset_epoch_df


# Run everything here to reload stuff
# %% cell dividing line

# firing rate stuff-- from april as well

from class_plot_fr_line_heat_april import FiringRatePlotter


def process_and_plot_fr_datasets(
    npz_path,
    pkl_path,
    base_output_dir,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000,
):
    """
    CLI-based pipeline to generate firing rate heatmaps and line plots.
    """
    # Prompt user
    mode, is_warped, alignment, sort_by_length = FiringRatePlotter.cli_options()

    filename = os.path.basename(npz_path)
    base_name = filename.split("_repacked.npz")[0]

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple) and len(data) == 4:
            spike_array, dataset_num, index, key = data

            dataset_num_str = str(dataset_num)
            dataset_num_clean = dataset_num_str.split("_repacked.npz")[0]
            core_dataset_name = dataset_num_clean

            print(f"Processing dataset number: {core_dataset_name}")

            extracted_pkl = extract_valid_changepoints(
                pkl_path, spike_array, dataset_num_clean, index, key
            )

            if extracted_pkl is not None:
                try:
                    changepoints = extracted_pkl[:, 3]

                    calc = CalcFRStates(
                        spike_array=spike_array,
                        changepoints=changepoints,
                        window_length=window_length,
                        step_size=step_size,
                        compute_unwarped_spike_arrays=False,
                        compute_unwarped_firing_rates=True,
                        compute_warped_spike_arrays=False,
                        compute_warped_firing_rates=True,
                        fixed_warp_duration=fixed_warp_duration,
                    )

                    fr_unwarped, _, fr_warped, _ = calc.run()
                    fr_data = fr_warped if is_warped else fr_unwarped

                    dataset_dir = os.path.join(
                        base_output_dir, f"dataset_{dataset_num_clean}"
                    )
                    os.makedirs(dataset_dir, exist_ok=True)

                    plotter = FiringRatePlotter(
                        fr_data_array=fr_data,
                        dataset_dir=dataset_dir,
                        dataset_name=core_dataset_name,
                        base_name=base_name,
                        is_warped=is_warped,
                        mode=mode,
                        alignment=alignment,
                        sort_by_length=sort_by_length,
                        changepoints_dict=standardized_changepoints_dict,
                    )

                    plotter.plot_all()

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Error processing dataset {core_dataset_name}: {e}")
            else:
                print(
                    f"No valid changepoints for dataset {core_dataset_name}; skipping..."
                )

    print("All datasets processed.")


# new 4/14/25 working with warped data now too
# SEM single-neuron population funcs that are to be run every so often
# transferred to own file-- see april_single_nrn_sem_war_unwar_plot.py


# func that runs the calcFRstaes stuff for further use:
def process_and_return_fr_datasets(
    npz_path, pkl_path, window_length=250, step_size=25, fixed_warp_duration=1000
):
    """
    Returns the firing rate data (unwarped and warped) for each valid dataset
    found in the npz file using the CalcFRStates pipeline.

    Returns:
        dict: {
            dataset_name: {
                'fr_unwarped': xarray.DataArray,
                'fr_warped': xarray.DataArray
            }, ...
        }
    """
    results = {}
    filename = os.path.basename(npz_path)
    base_name = filename.split("_repacked.npz")[0]

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple) and len(data) == 4:
            spike_array, dataset_num, index, key = data

            dataset_num_str = str(dataset_num)
            dataset_num_clean = dataset_num_str.split("_repacked.npz")[0]
            core_dataset_name = dataset_num_clean

            print(f"Processing dataset number: {core_dataset_name}")

            extracted_pkl = extract_valid_changepoints(
                pkl_path, spike_array, dataset_num_clean, index, key
            )

            if extracted_pkl is not None:
                try:
                    changepoints = extracted_pkl[:, 3]

                    calc = CalcFRStates(
                        spike_array=spike_array,
                        changepoints=changepoints,
                        window_length=window_length,
                        step_size=step_size,
                        compute_unwarped_spike_arrays=False,
                        compute_unwarped_firing_rates=True,
                        compute_warped_spike_arrays=False,
                        compute_warped_firing_rates=True,
                        fixed_warp_duration=fixed_warp_duration,
                    )

                    fr_unwarped, _, fr_warped, _ = calc.run()

                    results[core_dataset_name] = {
                        "fr_unwarped": fr_unwarped,
                        "fr_warped": fr_warped,
                    }

                except Exception as e:
                    print(f"Error processing {core_dataset_name}: {e}")
                    continue

    return results


fr_dict = process_and_return_fr_datasets(npz_path, pkl_path)
###### New RNN Latent stuff ######


# plot space for this hooey-- separated heat maps:
rnn_taste_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir"


def plot_inferred_data_heatmaps(epoch_dataframes_dict, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each DataFrame in the dictionary
    for df_name, df in epoch_dataframes_dict.items():
        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes and changepoints from the DataFrame
        unique_tastes = df["taste"].unique().to_list()
        unique_changepoints = df["changepoint"].unique().to_list()

        # Loop over each taste and changepoint
        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                # Filter for the current taste and changepoint
                taste_changepoint_df = df.filter(
                    (pl.col("taste") == taste) & (pl.col("changepoint") == changepoint)
                )

                # Extract unique trials for the current taste and changepoint
                unique_trials = taste_changepoint_df["trial"].unique().to_list()

                # Initialize plot counter for tracking figures
                plot_counter = 0
                fig_num = 1

                # Loop over trials in chunks of 12 for the 4x3 grid
                for trial_chunk in range(0, len(unique_trials), 12):
                    # Create a new figure with a 4x3 grid layout
                    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
                    fig.suptitle(
                        f"Taste {taste}, Changepoint {changepoint} - {df_name}",
                        fontsize=16,
                    )

                    for i, trial in enumerate(
                        unique_trials[trial_chunk : trial_chunk + 12]
                    ):
                        # Filter the DataFrame for the current trial
                        trial_df = taste_changepoint_df.filter(pl.col("trial") == trial)

                        # Extract latent values as a numpy array
                        latent_values = np.stack(
                            [trial_df[f"latent_dim_{j}"].to_numpy() for j in range(8)],
                            axis=0,
                        )

                        # Get start and end times for the changepoint segment and calculate duration
                        start_time = trial_df["time"].min()
                        end_time = trial_df["time"].max()
                        duration = end_time - start_time

                        # Plot on the appropriate subplot
                        ax = axs[i // 3, i % 3]
                        heatmap = ax.imshow(
                            latent_values,
                            aspect="auto",
                            cmap="coolwarm",
                            interpolation="none",
                        )
                        ax.set_title(
                            f"Trial {trial} | Start: {start_time}, End: {end_time}, Duration: {duration}"
                        )
                        ax.set_xlabel("Time(ms)")
                        ax.set_ylabel("Latent Dimensions")
                        # Calculate time in ms for each time bin, setting ticks every 250 ms
                        time_bins_ms = np.arange(start_time, end_time + 1, BIN_SIZE_MS)
                        tick_locations = np.arange(
                            0, len(time_bins_ms), 250 // BIN_SIZE_MS
                        )
                        tick_labels = time_bins_ms[:: 250 // BIN_SIZE_MS]

                        # Set x-ticks and labels
                        ax.set_xticks(tick_locations)
                        ax.set_xticklabels(tick_labels)

                        # Set y-ticks for latent dimensions
                        ax.set_yticks(np.arange(latent_values.shape[0]))
                        ax.set_yticklabels(
                            [f"Dim {j + 1}" for j in range(latent_values.shape[0])]
                        )

                    # Adjust layout and add colorbar
                    plt.tight_layout(
                        rect=[0, 0, 1, 0.95]
                    )  # Leave space for the main title
                    fig.colorbar(
                        heatmap,
                        ax=axs,
                        orientation="horizontal",
                        fraction=0.05,
                        pad=0.05,
                    )

                    # Save the figure
                    output_file = os.path.join(
                        df_output_dir,
                        f"hm_taste_{taste}_changepoint_{changepoint}_fig_{fig_num}.png",
                    )
                    plt.savefig(output_file)
                    print(f"Saved heatmap figure: {output_file}")

                    plt.close(fig)
                    fig_num += 1


# line plots, separated:
line_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/line_plots_epoch_sep"

# Line plots that are separated by dark lines at CP's -- not separate plots themselves

# dir specifically for these plots
# this is the plotting that we like
rnn_taste_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/line_plots_don"


def plot_all_rnns_with_changepoints(
    epoch_dataframes_dict,
    standardized_changepoints_dict,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots RNN latent dimensions with changepoints, allowing manual control over the start and end time.

    Parameters:
    - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Extract the core name from df_name to match with standardized_changepoints_dict
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            # Plot 4 trials per figure
            for trial_chunk in range(0, len(unique_trials), 4):
                fig, axs = plt.subplots(4, 1, figsize=(15, 15))
                fig.suptitle(
                    f"RNN Plots for taste: {modified_tastes[taste_idx]} - {df_name}",
                    fontsize=16,
                )

                for i, trial in enumerate(unique_trials[trial_chunk : trial_chunk + 4]):
                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    time_values = trial_df["time"].to_numpy()
                    latent_values = [
                        trial_df[f"latent_dim_{j}"].to_numpy() for j in range(8)
                    ]

                    # Apply time filtering
                    if start_time is not None:
                        time_mask = time_values >= start_time
                    else:
                        time_mask = np.ones_like(time_values, dtype=bool)

                    if end_time is not None:
                        time_mask &= time_values <= end_time

                    time_values = time_values[time_mask]
                    latent_values = [lv[time_mask] for lv in latent_values]

                    ax = (
                        axs[i] if len(unique_trials) > 1 else axs
                    )  # Handle single subplot case

                    # Plot each latent dimension as a line
                    for j, latent_series in enumerate(latent_values):
                        ax.plot(time_values, latent_series, label=f"Latent Dim {j+1}")

                    # Plot changepoints within the filtered range
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(
                                changepoint_time,
                                color="black",
                                linestyle="--",
                                linewidth=2,
                                label="Changepoint",
                            )

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(
                            2000,
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Stimulus Delivery",
                        )

                    # Set plot title and labels
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Latent Value")

                # Add a shared legend outside the subplots
                fig.legend(
                    [f"Latent Dim {j+1}" for j in range(8)]
                    + ["Changepoint", "Stimulus Delivery (dotted)"],
                    loc="lower center",
                    ncol=5,
                    fontsize="small",
                    frameon=False,
                )

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"rnn_plots_taste_{taste}_chunk_{trial_chunk // 4 + 1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")

                plt.close(fig)


# APRIL NEW
# the above but reborn for firing rate PCA plots:
def plot_pca_of_firing_rates(
    fr_unwarped: xr.DataArray,
    changepoints_dict: dict,
    dataset_name: str,
    output_dir: str,
    modified_tastes: list,
    start_time: int = int,
    end_time: int = int,
):
    """
    Plots PCA of firing rate vectors for each trial (first 8 PCs), preserving the structure
    of the original RNN plotting function, with changepoints and stimulus time.

    Parameters:
    - fr_unwarped: xarray.DataArray with dims (taste, trial, segment, neuron, time_bin)
    - changepoints_dict: dict of changepoints indexed by dataset name
    - dataset_name: name of dataset (used to match in changepoint dict and output naming)
    - output_dir: directory to save the figures
    - modified_tastes: list of taste labels
    - start_time: optional time window start in ms
    - end_time: optional time window end in ms
    """
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name not in changepoints_dict:
        print(f"Changepoints not found for {dataset_name}. Skipping...")
        return

    changepoints = changepoints_dict[dataset_name]
    fr_data = fr_unwarped  # Do NOT select a single segment
    for taste_idx in fr_data.coords["taste"].values:
        taste_name = modified_tastes[taste_idx]
        taste_data = fr_data.sel(taste=taste_idx)
        trials = taste_data.coords["trial"].values

        # Stack trials for PCA input: (trials × time) × neurons
        trial_mats = []
        valid_trial_lengths = []

        for trial_idx in trials:
            # Concatenate segments in time for this trial
            trial_seg_stack = []
            for seg_idx in fr_data.coords["segment"].values:
                segment_data = fr_data.sel(
                    taste=taste_idx, trial=trial_idx, segment=seg_idx
                ).values  # shape: (neurons, time)
                if np.isnan(segment_data).all():
                    continue
                segment_data = segment_data[
                    :, ~np.isnan(segment_data[0])
                ]  # Drop NaN-padding
                trial_seg_stack.append(segment_data)

            if not trial_seg_stack:
                continue

            full_trial_mat = np.concatenate(
                trial_seg_stack, axis=1
            )  # shape: (neurons, total_time)
            trial_mats.append(full_trial_mat)
            valid_trial_lengths.append(full_trial_mat.shape[1])

        if not trial_mats:
            print(f"No valid trials for taste {taste_name} in {dataset_name}.")
            continue

        # Concatenate across trials
        full_mat = np.concatenate(trial_mats, axis=1).T  # shape: (total_time, neurons)

        # Standardize and apply PCA
        scaler = StandardScaler()
        full_mat_std = scaler.fit_transform(full_mat)

        pca = PCA(whiten=False)
        pcs = pca.fit_transform(full_mat_std)  # shape: (total_time, 8)

        # Unconcatenate back into trial segments
        trial_pcs = []
        cursor = 0
        for length in valid_trial_lengths:
            trial_pcs.append(
                pcs[cursor : cursor + length].T
            )  # Each trial: shape (8, time)
            cursor += length

        # Plotting 4 trials per figure
        for chunk_start in range(0, len(trial_pcs), 4):
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))
            fig.suptitle(f"PCA on FR - {taste_name} - {dataset_name}", fontsize=16)

            for i, (trial_idx, trial_pc) in enumerate(
                zip(
                    trials[chunk_start : chunk_start + 4],
                    trial_pcs[chunk_start : chunk_start + 4],
                )
            ):
                ax = axs[i] if isinstance(axs, np.ndarray) else axs
                num_timepoints = trial_pc.shape[1]
                time_vals = np.arange(num_timepoints) * 25 + 1500  # Default time base

                if start_time is not None:
                    mask = time_vals >= start_time
                else:
                    mask = np.ones_like(time_vals, dtype=bool)
                if end_time is not None:
                    mask &= time_vals <= end_time

                time_vals = time_vals[mask]
                trial_pc = trial_pc[:, mask]

                for dim in range(8):
                    ax.plot(time_vals, trial_pc[dim], label=f"PC {dim+1}")

                # Plot changepoints
                if taste_idx < len(changepoints) and trial_idx < len(
                    changepoints[taste_idx]
                ):
                    for cp in changepoints[taste_idx][trial_idx]:
                        if start_time is None or (start_time <= cp <= end_time):
                            ax.axvline(cp, color="black", linestyle="--", linewidth=2)

                # Stimulus at 2000ms
                if (start_time is None or start_time <= 2000) and (
                    end_time is None or 2000 <= end_time
                ):
                    ax.axvline(2000, color="black", linestyle=":", linewidth=2)

                ax.set_title(f"Trial {trial_idx}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("PCA Value")

            # Add legend
            fig.legend(
                [f"PC {i+1}" for i in range(8)] + ["Changepoint", "Stimulus"],
                loc="lower center",
                ncol=5,
                fontsize="small",
                frameon=False,
            )

            # Save and close
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fname = (
                f"pca_firingrates_taste_{taste_idx}_chunk_{chunk_start // 4 + 1}.png"
            )
            plt.savefig(os.path.join(output_dir, fname))
            plt.close(fig)
            # Save explained variance info for this taste
            var_percent = pca.explained_variance_ratio_ * 100
            cumulative_var = np.cumsum(var_percent)

            # var_percent = pca.explained_variance_ratio_[:] * 100
            var_lines = [f"PC{i+1}: {vp:.2f}%" for i, vp in enumerate(var_percent)]
            var_text = "\n".join(var_lines)

            ev_file = os.path.join(
                output_dir,
                f"explained_variance_taste_{taste_idx}_{taste_name.replace(' ', '_')}.txt",
            )
            with open(ev_file, "w") as f:
                f.write(
                    f"Explained Variance for Taste {taste_name} (Dataset: {dataset_name})\n"
                )
                f.write(f"{'-'*60}\n")
                f.write(var_text)
                f.write("\n")
                # Explained variance values
            #  cumulative_var = np.cumsum(pca.explained_variance_ratio_[:] * 100)

            fig, ax = plt.subplots(figsize=(5, 7))
            x = np.arange(1, len(var_percent) + 1)

            # Bar plot for individual PC explained variance
            bars = ax.bar(
                x, var_percent, color="blue", label="Explained Variance (per PC)"
            )

            # Line plot for cumulative explained variance
            ax.plot(
                x, cumulative_var, color="red", marker="o", label="Cumulative Variance"
            )

            # Labels and styling
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance (%)")
            ax.set_title(f"Explained Variance for {taste_name} - {dataset_name}")
            ax.set_xticks(x)
            ax.set_ylim(0, max(100, cumulative_var[-1] + 5))
            ax.legend(loc="upper left")

            # Save plot
            ev_plot_path = os.path.join(
                output_dir,
                f"explained_variance_plot_taste_{taste_idx}_{taste_name.replace(' ', '_')}.png",
            )
            plt.tight_layout()
            plt.savefig(ev_plot_path)
            plt.close()
            print(f"Saved explained variance plot: {ev_plot_path}")

            print(f"Saved explained variance to: {ev_file}")
            print(f"Saved PCA plot figure: {fname}")


# below is a mess lol
# def plot_pca_of_firing_rates(fr_unwarped: xr.DataArray,
#                               changepoints_dict: dict,
#                               dataset_name: str,
#                               output_dir: str,
#                               modified_tastes: list,
#                               start_time: int = None,
#                               end_time: int = None):
#     """
#     Plots PCA of firing rate vectors for each trial (first 8 PCs), preserving the structure
#     of the original RNN plotting function, with changepoints and stimulus time.

#     Parameters:
#     - fr_unwarped: xarray.DataArray with dims (taste, trial, segment, neuron, time_bin)
#     - changepoints_dict: dict of changepoints indexed by dataset name
#     - dataset_name: name of dataset (used to match in changepoint dict and output naming)
#     - output_dir: directory to save the figures
#     - modified_tastes: list of taste labels
#     - start_time: optional time window start in ms
#     - end_time: optional time window end in ms
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     if dataset_name not in changepoints_dict:
#         print(f"Changepoints not found for {dataset_name}. Skipping...")
#         return

#     changepoints = changepoints_dict[dataset_name]
#     fr_data = fr_unwarped

#     for taste_idx in fr_data.coords["taste"].values:
#         taste_name = modified_tastes[taste_idx]
#         taste_data = fr_data.sel(taste=taste_idx)
#         trials = taste_data.coords["trial"].values

#         trial_mats = []
#         trial_timevals = []
#         valid_trial_lengths = []

#         for trial_idx in trials:
#             trial_seg_stack = []
#             time_vals = []

#             for seg_idx in fr_data.coords["segment"].values:
#                 segment_data = fr_data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx).values
#                 if np.isnan(segment_data).all():
#                     continue
#                 segment_data = segment_data[:, ~np.isnan(segment_data[0])]
#                 if segment_data.shape[1] == 0:
#                     continue

#                 segment_time_vals = np.arange(segment_data.shape[1]) * 25
#                 segment_offset = 0 if not time_vals else time_vals[-1][-1] + 25
#                 segment_time_vals += 1500 + segment_offset

#                 trial_seg_stack.append(segment_data)
#                 time_vals.append(segment_time_vals)

#             if not trial_seg_stack:
#                 continue

#             full_trial_mat = np.concatenate(trial_seg_stack, axis=1)
#             full_time_vals = np.concatenate(time_vals)

#             # Apply time filter BEFORE PCA
#             time_mask = np.ones_like(full_time_vals, dtype=bool)
#             if start_time is not None:
#                 time_mask &= full_time_vals >= start_time
#             if end_time is not None:
#                 time_mask &= full_time_vals <= end_time

#             full_trial_mat = full_trial_mat[:, time_mask]
#             full_time_vals = full_time_vals[time_mask]

#             if full_trial_mat.shape[1] == 0:
#                 continue

#             trial_mats.append(full_trial_mat)
#             valid_trial_lengths.append(full_trial_mat.shape[1])
#             trial_timevals.append(full_time_vals)

#         if not trial_mats:
#             print(f"No valid trials for taste {taste_name} in {dataset_name}.")
#             continue

#         full_mat = np.concatenate(trial_mats, axis=1).T  # (total_time, neurons)
#         scaler = StandardScaler()
#         full_mat_std = scaler.fit_transform(full_mat)

#         pca = PCA(n_components=8, whiten=True)
#         pcs = pca.fit_transform(full_mat_std)  # (total_time, 8)

#         # Unconcatenate PCA back into trials
#         trial_pcs = []
#         cursor = 0
#         for length in valid_trial_lengths:
#             trial_pcs.append(pcs[cursor:cursor + length].T)  # (8, time)
#             cursor += length

#         # Plotting
#         for chunk_start in range(0, len(trial_pcs), 4):
#             fig, axs = plt.subplots(4, 1, figsize=(15, 15))
#             fig.suptitle(f"PCA on FR - {taste_name} - {dataset_name}", fontsize=16)

#             for i, (trial_idx, trial_pc) in enumerate(zip(trials[chunk_start:chunk_start+4], trial_pcs[chunk_start:chunk_start+4])):
#                 if i >= len(axs):
#                     break
#                 ax = axs[i]
#                 time_vals = trial_timevals[chunk_start + i]

#                 for dim in range(8):
#                     ax.plot(time_vals, trial_pc[dim], label=f'PC {dim+1}')

#                 # Plot changepoints
#                 if taste_idx < len(changepoints) and trial_idx < len(changepoints[taste_idx]):
#                     for cp in changepoints[taste_idx][trial_idx]:
#                         if start_time is None or (start_time <= cp <= end_time):
#                             ax.axvline(cp, color='blue', linestyle='--', linewidth=2)

#                 # Stimulus at 2000 ms
#                 if (start_time is None or start_time <= 2000) and (end_time is None or 2000 <= end_time):
#                     ax.axvline(2000, color='red', linestyle=':', linewidth=2)

#                 ax.set_title(f"Trial {trial_idx}")
#                 ax.set_xlabel("Time (ms)")
#                 ax.set_ylabel("PCA Value")

#             fig.legend([f'PC {i+1}' for i in range(8)] + ['Changepoint', 'Stimulus'],
#                        loc='lower center', ncol=5, fontsize='small', frameon=False)

#             plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#             fname = f"pca_firingrates_taste_{taste_idx}_chunk_{chunk_start // 4 + 1}.png"
#             plt.savefig(os.path.join(output_dir, fname))
#             plt.close(fig)
#             print(f"Saved PCA plot figure: {fname}")


# running this mess:
for dataset_name, fr_data in fr_dict.items():
    fr_unwarped = fr_data["fr_unwarped"]

    plot_pca_of_firing_rates(
        fr_unwarped=fr_unwarped,
        changepoints_dict=standardized_changepoints_dict,
        dataset_name=dataset_name,
        output_dir=os.path.join(
            "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_ON_RNN_FULL_TRIAL/PCA_NONWHITENED_ON_FR_DATA_FULL_FRAME",
            dataset_name,
        ),
        modified_tastes=modified_tastes,
        start_time=1500,
        end_time=4500,
    )


# plotting latant graphs-- RNN
# also want to plot these data but instead of the lines, plot the variance
def plot_thin_thick_rnn_latants(
    epoch_dataframes_dict,
    standardized_changepoints_dict,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots RNN latent dimensions and averages across all trials for each taste.

    Parameters:
    - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Extract the core name from df_name to match with standardized_changepoints_dict
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_df = df.filter(pl.col("taste") == taste)

            # Prepare two figures for each taste (4 latent dimensions per figure)
            for fig_num, latent_dims in enumerate([range(4), range(4, 8)]):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                axs = axs.flatten()
                fig.suptitle(
                    f"Averaged RNN latents for taste: {modified_tastes[taste_idx]} - {df_name}",
                    fontsize=16,
                )

                for ax, latent_idx in zip(axs, latent_dims):
                    time_values = taste_df["time"].to_numpy()
                    latent_values = taste_df[f"latent_dim_{latent_idx}"].to_numpy()

                    # Apply time filtering
                    if start_time is not None:
                        time_mask = (time_values >= start_time) & (
                            time_values <= end_time
                        )
                        time_values = time_values[time_mask]
                    else:
                        time_mask = np.ones_like(time_values, dtype=bool)

                    # Split latent values by trial
                    latent_trials = []
                    for trial in taste_df["trial"].unique():
                        trial_mask = (taste_df["trial"] == trial).to_numpy() & time_mask
                        latent_trial = latent_values[trial_mask]
                        if len(latent_trial) > 0:
                            latent_trials.append(latent_trial)

                    for trial_latent in latent_trials:
                        ax.plot(
                            time_values[: len(trial_latent)],
                            trial_latent,
                            color="gray",
                            alpha=0.5,
                            linewidth=0.5,
                        )

                    # Compute and plot the average latent dimension over all trials
                    avg_latent = np.mean(np.stack(latent_trials), axis=0)
                    ax.plot(
                        time_values[: len(avg_latent)],
                        avg_latent,
                        color="blue",
                        linewidth=2,
                        label=f"Avg Latent Dim {latent_idx+1}",
                    )

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(
                            2000,
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Stimulus Delivery",
                        )

                    # Set plot title and labels
                    ax.set_title(f"Latent Dimension {latent_idx+1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Latent Value")

                # Add a shared legend outside the subplots
                fig.legend(loc="lower center", ncol=5, fontsize="small", frameon=False)

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"averaged_rnn_plots_taste_{taste}_fig_{fig_num + 1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved averaged RNN plot figure: {output_file}")

                plt.close(fig)


# this but variance:
thin_thick_output = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_Thin_Thick_Var"


def plot_thin_thick_rnn_latants(
    epoch_dataframes_dict,
    standardized_changepoints_dict,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots the variance of RNN latent dimensions across all trials for each taste.

    Parameters:
    - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Extract the core name from df_name to match with standardized_changepoints_dict
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_df = df.filter(pl.col("taste") == taste)

            # Prepare two figures for each taste (4 latent dimensions per figure)
            for fig_num, latent_dims in enumerate([range(4), range(4, 8)]):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                axs = axs.flatten()
                fig.suptitle(
                    f"Variance of RNN latents for taste: {modified_tastes[taste_idx]} - {df_name}",
                    fontsize=16,
                )

                for ax, latent_idx in zip(axs, latent_dims):
                    time_values = taste_df["time"].to_numpy()
                    latent_values = taste_df[f"latent_dim_{latent_idx}"].to_numpy()

                    # Apply time filtering
                    if start_time is not None:
                        time_mask = (time_values >= start_time) & (
                            time_values <= end_time
                        )
                        time_values = time_values[time_mask]
                    else:
                        time_mask = np.ones_like(time_values, dtype=bool)

                    # Split latent values by trial
                    latent_trials = []
                    for trial in taste_df["trial"].unique():
                        trial_mask = (taste_df["trial"] == trial).to_numpy() & time_mask
                        latent_trial = latent_values[trial_mask]
                        if len(latent_trial) > 0:
                            latent_trials.append(latent_trial)

                    # Compute and plot the variance of latent dimensions over all trials
                    if latent_trials:
                        stacked_trials = np.stack(latent_trials)
                        variance_latent = np.var(stacked_trials, axis=0)
                        ax.plot(
                            time_values[: len(variance_latent)],
                            variance_latent,
                            color="red",
                            linewidth=2,
                            label=f"Variance Latent Dim {latent_idx+1}",
                        )

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(
                            2000,
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Stimulus Delivery",
                        )

                    # Set plot title and labels
                    ax.set_title(f"Variance of Latent Dimension {latent_idx+1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Variance")

                # Add a shared legend outside the subplots
                fig.legend(loc="lower center", ncol=5, fontsize="small", frameon=False)

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"variance_rnn_plots_taste_{taste}_fig_{fig_num + 1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved variance RNN plot figure: {output_file}")

                plt.close(fig)


# attempting interpolation of the data on an epoch by epoch basis -- don't think this is working-- look more into


## I want to update this to warp to a standard 1000 ms. This will become part of a class which will do warping, epoch alignmnet, std.dev and SEM, as well as variance across time. (with an average as well ofc)
def plot_RNN_epochs_with_interpolation(
    epoch_dataframes_dict,
    standardized_changepoints_dict,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots RNN latent dimensions for each epoch using interpolation to align trial durations.

    Parameters:
    - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Extract the core name from df_name to match with standardized_changepoints_dict
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_df = df.filter(pl.col("taste") == taste)

            latent_dims = [col for col in df.columns if col.startswith("latent_dim_")]

            for epoch_idx in range(len(changepoints[0]) + 1):
                # Define epoch boundaries
                if epoch_idx == 0:
                    epoch_start = 2000
                else:
                    epoch_start = changepoints[epoch_idx - 1] if epoch_idx > 0 else 2000

                if epoch_idx == len(changepoints[0]):
                    epoch_end = end_time
                else:
                    epoch_end = (
                        changepoints[epoch_idx]
                        if epoch_idx < len(changepoints)
                        else end_time
                    )

                # Find the longest duration for this epoch across all trials
                epoch_durations = []
                for trial_idx in range(len(changepoints)):
                    if epoch_idx == 0:
                        duration = changepoints[0] - 2000
                    elif epoch_idx == len(changepoints[0]):
                        duration = end_time - changepoints[-1]
                    else:
                        duration = changepoints[epoch_idx] - changepoints[epoch_idx - 1]
                    epoch_durations.append(duration)

                max_epoch_duration = max(np.array(epoch_durations).flatten())

                # Interpolate all trials for this epoch to the longest duration
                interpolated_trials = {dim: [] for dim in latent_dims}

                for trial_idx in range(changepoints.shape[0]):
                    trial_start = epoch_start
                    trial_end = epoch_end

                    trial_mask = (taste_df["time"] >= trial_start) & (
                        taste_df["time"] <= trial_end
                    )
                    trial_df = taste_df.filter(trial_mask)

                    for dim in latent_dims:
                        trial_latent = trial_df[dim].to_numpy()
                        if len(trial_latent) > 0:
                            f_interp = interp1d(
                                np.linspace(0, 1, len(trial_latent)),
                                trial_latent,
                                kind="nearest",
                                fill_value="extrapolate",
                            )
                            interpolated_trials[dim].append(
                                f_interp(np.linspace(0, 1, max_epoch_duration))
                            )

                # Plot each latent dimension individually
                for dim_idx, dim in enumerate(latent_dims):
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot all trials in light gray
                    for trial_latent in interpolated_trials[dim]:
                        ax.plot(
                            np.linspace(epoch_start, epoch_end, max_epoch_duration),
                            trial_latent,
                            color="gray",
                            alpha=0.5,
                            linewidth=0.5,
                        )

                    # Compute and plot the average latent dimension
                    avg_latent = np.mean(interpolated_trials[dim], axis=0)
                    ax.plot(
                        np.linspace(epoch_start, epoch_end, max_epoch_duration),
                        avg_latent,
                        color="blue",
                        linewidth=2,
                        label=f"Avg Latent Dim {dim_idx + 1}",
                    )

                    # Set plot title and labels
                    ax.set_title(
                        f"Taste: {taste}, Epoch: {epoch_idx + 1}, Latent: {dim}"
                    )
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Latent Value")
                    ax.legend()

                    # Save the figure
                    output_file = os.path.join(
                        df_output_dir,
                        f"taste_{taste}_epoch_{epoch_idx + 1}_latent_{dim_idx + 1}.png",
                    )
                    plt.tight_layout()
                    plt.savefig(output_file)
                    print(f"Saved plot: {output_file}")
                    plt.close(fig)


## New version of the above but as a class

## fully working class


class RNNWarpingIntraAroundCP:
    """
    Pipeline for warping RNN latent trajectories around changepoints,
    then plotting (warped, unwarped, or windowed) segments around each CP.
    """

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        return [
            c for c in df.columns if c.startswith("PC_") or c.startswith("latent_dim_")
        ]

    def __init__(
        self,
        standardized_changepoints_dict: dict,
        output_dir: str,
        mode: str = "unwarped",
        warp_length: int = 1000,
        window_size: int = None,
        start_time: int = 1500,
        end_time: int = 4500,
        plot_average: bool = False,
    ):
        if mode == "windowed" and window_size is None:
            raise ValueError("`window_size` must be set when mode='windowed'")
        self.changepoints = standardized_changepoints_dict
        self.output_dir = output_dir
        self.mode = mode
        self.warp_length = warp_length
        self.window_size = window_size
        self.start_time = start_time
        self.end_time = end_time
        self.plot_average = plot_average
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_changepoints(self, core_name: str):
        cp_list = self.changepoints.get(core_name)
        if cp_list is None:
            print(f"Changepoints not found for {core_name}. Skipping.")
            return None
        return [
            np.clip(np.array(arr), self.start_time, self.end_time) for arr in cp_list
        ]

    def align_trials(self, df: pl.DataFrame):
        aligned = {}
        for taste, trial in df.select(["taste", "trial"]).unique().rows():
            grp = df.filter((pl.col("taste") == taste) & (pl.col("trial") == trial))
            aligned[(taste, trial)] = grp.with_columns(
                (pl.col("time") - self.start_time).alias("time")
            )
        return aligned

    def _plot_non_windowed(
        self, dataset_name: str, df: pl.DataFrame, dims: list, cp_list: list
    ):
        tastes = df["taste"].unique().to_list()
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()
            trials_cps = cp_list[taste_idx]
            n_cps = trials_cps.shape[1]
            num_epochs = min(n_cps + 1, 4)
            fig, axes = plt.subplots(
                len(dims),
                num_epochs,
                figsize=(5 * num_epochs, 4 * len(dims)),
                sharex=(self.mode != "unwarped"),
            )
            axes = np.atleast_2d(axes)
            for row, dim in enumerate(dims):
                for eid in range(num_epochs):
                    ax = axes[row, eid]
                    epoch_vals = []
                    for trial in trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        cps = trials_cps[trial]
                        if eid == 0:
                            start, end = self.start_time, cps[0]
                        elif eid < n_cps:
                            start, end = cps[eid - 1], cps[eid]
                        else:
                            start, end = cps[-1], self.end_time
                        seg = trial_df.filter(
                            (pl.col("time") >= start) & (pl.col("time") <= end)
                        )
                        t = seg["time"].to_numpy() - start
                        v = seg[dim].to_numpy()
                        if t.size > 0:
                            if self.mode == "unwarped":
                                ax.plot(t, v, alpha=0.7, linewidth=0.5)
                                epoch_vals.append((t.astype(int), v))
                            else:
                                src = np.linspace(0, 1, t.size)
                                tgt = np.linspace(0, 1, self.warp_length)
                                f = interp1d(
                                    src, v, kind="linear", fill_value="extrapolate"
                                )
                                # play with this a lot-- as I may want to figure out which interpolation method is best.
                                # there are a lot fo types of interpolation methods that could work here; the idea is to find the right one.
                                # nearest looks quite blocky. Linear could do , but is clearly generating outliers in the extreme.
                                wv = f(tgt)
                                ta = np.linspace(0, self.warp_length, self.warp_length)
                                ax.plot(ta, wv, alpha=0.7, linewidth=0.5)
                                epoch_vals.append((ta, wv))
                    # if self.plot_average and epoch_vals:
                    #     grid=epoch_vals[0][0]
                    #     arr=np.vstack([np.interp(grid,tt,vv) for tt,vv in epoch_vals])
                    #     ax.plot(grid,np.nanmean(arr,axis=0),linewidth=2, alpha = 0.8, color='blue') # issue with unwarped; disspaearing
                    # fix below:
                    if self.plot_average and epoch_vals:
                        # Create a unified time grid from all trial times
                        grid = np.unique(np.concatenate([t for t, _ in epoch_vals]))
                        # Interpolate each trial onto the common grid
                        mat = np.vstack(
                            [
                                np.interp(grid, t, v, left=np.nan, right=np.nan)
                                for t, v in epoch_vals
                            ]
                        )
                        # Plot the mean across trials
                        ax.plot(
                            grid,
                            np.nanmean(mat, axis=0),
                            linewidth=2,
                            alpha=1,
                            color="gray",
                        )
                    ax.set_title(f"Epoch {eid}\n{dim}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("PC magnitude")
            fig.suptitle(
                f"Mode: {self.mode}, Taste: {taste}, Dataset: {dataset_name} (4 epochs)"
            )
            ann = (
                "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
                f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end. \nStart and End times are T = {self.start_time} and T = {self.end_time} respecitvely. Average in gray"
            )
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.06)
            fig.text(0.5, 0.02, ann, ha="center", va="top", fontsize=16)
            od = os.path.join(self.output_dir, dataset_name, f"taste_{taste}")
            os.makedirs(od, exist_ok=True)
            fig.savefig(
                os.path.join(od, f"{self.mode}_taste_{taste}_{dataset_name}.png")
            )
            plt.close(fig)

    def _plot_windowed(
        self, dataset_name: str, df: pl.DataFrame, dims: list, cp_list: list
    ):
        # Plot mean and individual PC trajectories around stimulus (CP0) and 3 changepoints
        tastes = df["taste"].unique().to_list()
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()
            trials_cps = cp_list[taste_idx]
            # Always plot 4 columns: CP0 (stimulus at 2000 ms) + up to 3 changepoints
            cp_cols = 1 + min(trials_cps.shape[1], 3)
            pc_columns = dims[:8]
            pc_count = len(pc_columns)

            fig, axs = plt.subplots(
                pc_count, cp_cols, figsize=(5 * cp_cols, 3 * pc_count), squeeze=False
            )
            fig.suptitle(
                f"RNN PCs Around Changepoints (window-aligned): Taste {taste}, Dataset {dataset_name}",
                fontsize=16,
            )

            for col in range(cp_cols):
                for row, pc in enumerate(pc_columns):
                    ax = axs[row, col]
                    trial_vals = []
                    for trial in trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        if col == 0:
                            cp_time = 2000
                        else:
                            cp_time = trials_cps[trial][col - 1]
                        mask = (trial_df["time"] >= cp_time - self.window_size) & (
                            trial_df["time"] <= cp_time + self.window_size
                        )
                        seg = trial_df.filter(mask)
                        t = seg["time"].to_numpy() - cp_time
                        v = seg[pc].to_numpy()
                        if t.size > 0:
                            ax.plot(
                                t, v, linewidth=1
                            )  # change this to a rainbow-esque thing later on
                            trial_vals.append((t, v))
                    if self.plot_average and trial_vals:
                        grid = trial_vals[0][0]
                        arr = np.vstack([np.interp(grid, t, v) for t, v in trial_vals])
                        ax.plot(
                            grid,
                            np.nanmean(arr, axis=0),
                            color="gray",
                            alpha=1,
                            linewidth=2,
                        )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    label = "Stimulus" if col == 0 else f"CP {col}"
                    ax.set_title(f"{pc} — {label}")
                    ax.set_xlabel("Relative Time (ms)")
                    ax.set_ylabel("PC Value")

            # Annotation
            note = (
                "Stimulus delivery (2000 ms); CP1-3 = first three changepoints of the dataset.\n "
                "Time is relative to each CP, which is set to T=0. Data aligned to changepoint.\n average in light blue."
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.text(0.5, 0.02, note, ha="center", va="top", fontsize=16)

            out_dir = os.path.join(self.output_dir, dataset_name, f"taste_{taste}")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"windowed_taste_{taste}_{dataset_name}.png"
            fig.savefig(os.path.join(out_dir, fname))
            plt.close(fig)

    def _analyze_dataset(self, dataset_name: str, df: pl.DataFrame):
        core = dataset_name.split("_repacked_raw_latent_vectors")[0]
        cp_list = self.extract_changepoints(core)
        if cp_list is None:
            return
        dims = self.get_data_columns(df)
        if self.mode == "windowed":
            self._plot_windowed(dataset_name, df, dims, cp_list)
        else:
            self._plot_non_windowed(dataset_name, df, dims, cp_list)

    def run(self, dataset_dict: dict):
        for name, df in tqdm(dataset_dict.items(), desc="Processing datasets"):
            if not isinstance(df, pl.DataFrame):
                raise ValueError(f"Data under {name} is not a Polars DataFrame")
            self._analyze_dataset(name, df)


pipeline = RNNWarpingIntraAroundCP(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/INTRA_PCA_RNN_WINDOW_REF",
    mode="windowed",  # 'warped', 'unwarped', or 'windowed'
    warp_length=1000,  # used if mode='warped'
    window_size=300,  # half-window size if mode='windowed'
    start_time=1500,  # global data start (ms)
    end_time=4500,  # global data end (ms)
    plot_average=True,  # whether to overlay the mean trajectory
)
pipeline.run(robust_pca_95)

# test mode:
pipeline = RNNWarpingIntraAroundCP(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/test",
    mode="warped",  # 'warped', 'unwarped', or 'windowed'
    warp_length=1000,  # used if mode='warped'
    window_size=300,  # half-window size if mode='windowed'
    start_time=1500,  # global data start (ms)
    end_time=4500,  # global data end (ms)
    plot_average=True,  # whether to overlay the mean trajectory
)
pipeline.run(robust_pca_95)
# windowed is still a problem somehow.

### MOOOVE all of this to its own class eventually (post-thesis defense!)
# plotting individual latant (newer stuff):
ind_latant_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/lines_ind_latant"


def plot_individual_latents(
    epoch_dataframes_dict,
    standardized_changepoints_dict,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots individual latent vectors for each trial, taste, and epoch.

    Parameters:
    - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            for trial in unique_trials:
                trial_df = taste_df.filter(pl.col("trial") == trial)
                time_values = trial_df["time"].to_numpy()
                latent_values = [
                    trial_df[f"latent_dim_{j}"].to_numpy() for j in range(8)
                ]

                if start_time is not None:
                    time_mask = time_values >= start_time
                else:
                    time_mask = np.ones_like(time_values, dtype=bool)

                if end_time is not None:
                    time_mask &= time_values <= end_time

                time_values = time_values[time_mask]
                latent_values = [lv[time_mask] for lv in latent_values]

                # Create a figure with 8 subplots
                fig, axs = plt.subplots(8, 1, figsize=(15, 20))
                fig.suptitle(
                    f"Latent Vectors for Trial {trial}, Taste {modified_tastes[taste_idx]} - {df_name}",
                    fontsize=16,
                )

                for i, latent_series in enumerate(latent_values):
                    ax = axs[i]
                    ax.plot(time_values, latent_series, label=f"Latent Dim {i+1}")

                    # Plot changepoints for this trial
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(
                                changepoint_time,
                                color="black",
                                linestyle="--",
                                linewidth=2,
                                label="Changepoint",
                            )

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(
                            2000,
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Stimulus Delivery",
                        )

                    ax.set_title(f"Latent Dim {i+1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Value")
                    ax.legend(loc="upper right", fontsize="small", frameon=False)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                output_file = os.path.join(
                    df_output_dir, f"latents_taste_{taste}_trial_{trial}.png"
                )
                plt.savefig(output_file)
                print(f"Saved latent vector plot figure: {output_file}")
                plt.close(fig)


def get_data_columns(df: pl.DataFrame):
    # Return PC_x or latent_dim_x columns from a Polars DataFrame.
    # Useful for looping across dimensions.
    return [
        col
        for col in df.columns
        if col.startswith("PC_") or col.startswith("latent_dim_")
    ]


# Plotting these data:
diff_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/DERIV_PCA_RNN/SECOND_DERIV"
output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/CP_window_plt/PCA_RNN_t_robust_full"


def plot_pca_rnn_with_changepoints(
    pca_lat_dict,
    standardized_changepoints_dict,
    modified_tastes,
    output_dir,
    start_time=None,
    end_time=None,
):
    """
    Plots RNN principal components with changepoints, allowing manual control over the start and end time.

    Parameters:
    - epoch_dataframes_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - start_time: Manual start time for plotting (ms).
    - end_time: Manual end time for plotting (ms).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        # Extract the core name from df_name to match with standardized_changepoints_dict
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            # Plot 4 trials per figure
            for trial_chunk in range(0, len(unique_trials), 4):
                fig, axs = plt.subplots(4, 1, figsize=(15, 15))
                fig.suptitle(
                    f"Second_deriv_PCA_trial_RNN Plots for Taste {taste_name} - {df_name}",
                    fontsize=16,
                )

                for i, trial in enumerate(unique_trials[trial_chunk : trial_chunk + 4]):
                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    time_values = trial_df["time"].to_numpy()

                    # Dynamically detect PC columns
                    pc_columns = [
                        col
                        for col in df.columns
                        if col.startswith("PC_") or col.startswith("latent_dim_")
                    ]
                    pc_values = [trial_df[col].to_numpy() for col in pc_columns]

                    # Apply time filtering
                    if start_time is not None:
                        time_mask = time_values >= start_time
                    else:
                        time_mask = np.ones_like(time_values, dtype=bool)

                    if end_time is not None:
                        time_mask &= time_values <= end_time

                    time_values = time_values[time_mask]
                    pc_values = [pc[time_mask] for pc in pc_values]

                    ax = (
                        axs[i] if len(unique_trials) > 1 else axs
                    )  # Handle single subplot case

                    # Plot each principal component as a line
                    for j, pc_series in enumerate(pc_values):
                        ax.plot(time_values, pc_series, label=f"{pc_columns[j]}")

                    # Plot changepoints within the filtered range
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(
                                changepoint_time,
                                color="black",
                                linestyle="--",
                                linewidth=2,
                                label="Changepoint",
                            )

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(
                            2000,
                            color="black",
                            linestyle=":",
                            linewidth=2,
                            label="Stimulus Delivery",
                        )

                    # Set plot title and labels
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Principal Component Value")

                # Add a shared legend outside the subplots
                fig.legend(
                    [f"{col}" for col in pc_columns]
                    + ["Changepoint"]
                    + ["Stimulus Delivery (dotted)"],
                    loc="lower center",
                    ncol=5,
                    fontsize="small",
                    frameon=False,
                )

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"rnn_plots_taste_{taste}_chunk_{trial_chunk // 4 + 1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")

                plt.close(fig)


# plotting more specific amounts of time-- what's happening within a latent as well as around changepoints
transition_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/CP_window_plt/PCA_RNN_t_robust_full"


def plot_pca_rnn_around_changepoints(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components within +/- 250 ms of each changepoint,
    organizing figures by taste and changepoint index, with each trial in a separate subplot.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_changepoints = len(changepoints[taste_idx][unique_trials[0]])

            for changepoint_idx in range(num_changepoints):
                fig, axs = plt.subplots(6, 5, figsize=(20, 24))
                fig.suptitle(
                    f"PCA_robust_95pct_RNN_for {taste_name} - Changepoint {changepoint_idx+1}",
                    fontsize=16,
                )

                for i, trial in enumerate(unique_trials):
                    if i >= 30:
                        break  # Limit to 30 trials per figure

                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    trial_changepoints = changepoints[taste_idx][trial]

                    changepoint_time = trial_changepoints[changepoint_idx]
                    time_window_mask = (trial_df["time"] >= changepoint_time - 250) & (
                        trial_df["time"] <= changepoint_time + 250
                    )
                    time_values = trial_df.filter(time_window_mask)["time"].to_numpy()

                    pc_columns = [
                        col for col in trial_df.columns if col.startswith("PC_")
                    ]
                    pc_values = [
                        trial_df.filter(time_window_mask)[col].to_numpy()
                        for col in pc_columns
                    ]

                    ax = axs[i // 5, i % 5]
                    for j, pc_series in enumerate(pc_values):
                        ax.plot(time_values, pc_series, label=f"{pc_columns[j]}")

                    ax.axvline(
                        changepoint_time,
                        color="black",
                        linestyle="--",
                        linewidth=2,
                        label="Changepoint",
                    )
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Principal Component Value")

                fig.legend(
                    [f"{col}" for col in pc_columns] + ["Changepoint"],
                    loc="lower center",
                    ncol=5,
                    fontsize="small",
                    frameon=False,
                )

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"rnn_plots_taste_{taste_name}_changepoint_{changepoint_idx+1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")
                plt.close(fig)


# Modification of this that produces a plot that gives one latent across all trials in one plot, times however many transitions.

transition_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/CP_window_plt/FIR_DIFF_PCA_RNN_trial_wise_rob_95"


def plot_mean_vectors_rnn_around_changepoints(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots the mean and individual RNN principal component vectors within +/- 250 ms of each changepoint.
    For each taste, a figure is generated with 8 rows (one per latent vector, up to 8) and 3 columns (one per changepoint).
    In each subplot, every trial’s PC vector (within the time window) is plotted as a light gray line,
    and the mean (across trials) is plotted as a thicker blue line.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each dataset in the dictionary.
    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        # Create an output folder for the dataset.
        dataset_output_dir = os.path.join(output_dir, core_dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        # Loop over tastes in the dataset.
        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )
            # Filter the DataFrame to only include rows for the current taste.
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            # Determine the number of changepoints to plot.
            num_changepoints = len(changepoints[taste_idx][unique_trials[0]])
            cp_count = min(num_changepoints, 3)  # Only plot up to 3 changepoints.

            # Get the list of principal component columns (up to 8 latent vectors).
            pc_columns = [col for col in taste_df.columns if col.startswith("PC_")][:8]
            num_pcs = len(pc_columns)

            # Create the figure with a grid of subplots (rows: latent vectors, cols: changepoints).
            fig, axs = plt.subplots(
                num_pcs, cp_count, figsize=(5 * cp_count, 3 * num_pcs), squeeze=False
            )
            fig.suptitle(
                f"First_diff_Mean RNN PCA for {core_dataset_name} - {taste_name}",
                fontsize=16,
            )

            # Loop over each changepoint (column in the subplot grid).
            for cp_idx in range(cp_count):
                # Loop over each PC (row in the subplot grid).
                for pc_idx, pc_col in enumerate(pc_columns):
                    trial_values_list = []  # To collect PC vectors for each trial.
                    time_axis = None  # To hold the relative time axis.

                    # Loop over each trial.
                    for trial in unique_trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        # If the current trial does not have this changepoint, skip.
                        if cp_idx >= len(trial_changepoints):
                            continue

                        changepoint_time = trial_changepoints[cp_idx]
                        # Create a mask for the time window around the changepoint.
                        time_window_mask = (
                            trial_df["time"] >= changepoint_time - 250
                        ) & (trial_df["time"] <= changepoint_time + 250)

                        trial_time = trial_df.filter(time_window_mask)[
                            "time"
                        ].to_numpy()
                        trial_pc = trial_df.filter(time_window_mask)[pc_col].to_numpy()

                        # Compute relative time (centered on the changepoint).
                        relative_time = trial_time - changepoint_time

                        # Store the time axis from the first valid trial.
                        if time_axis is None:
                            time_axis = relative_time

                        trial_values_list.append(trial_pc)

                        # Plot the individual trial's PC values in light gray.
                        axs[pc_idx, cp_idx].plot(
                            relative_time, trial_pc, color="lightgray", linewidth=1
                        )

                    # # Compute and plot the mean PC values (if any trials were processed).
                    # if trial_values_list:
                    #     trial_values_array = np.array(trial_values_list)  # Shape: (num_trials, num_time_points)
                    #     mean_values = np.mean(trial_values_array, axis=0)
                    #     axs[pc_idx, cp_idx].plot(time_axis, mean_values, color='blue', linewidth=2)

                    # Draw a vertical line at time 0 (the changepoint).
                    axs[pc_idx, cp_idx].axvline(
                        0, color="black", linestyle="--", linewidth=1
                    )
                    axs[pc_idx, cp_idx].set_title(f"{pc_col}, Changepoint {cp_idx+1}")
                    axs[pc_idx, cp_idx].set_xlabel("Relative Time (ms)")
                    axs[pc_idx, cp_idx].set_ylabel("PC Value")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_file = os.path.join(
                dataset_output_dir, f"{core_dataset_name}_{taste_name}.png"
            )
            plt.savefig(output_file)
            print(f"Saved mean RNN plot figure: {output_file}")
            plt.close(fig)


# version just for the raw_rnn_data
def plot_rnn_around_changepoints(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components within +/- 250 ms of each changepoint,
    organizing figures by taste and changepoint index, with each trial in a separate subplot.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_changepoints = len(changepoints[taste_idx][unique_trials[0]])

            for changepoint_idx in range(num_changepoints):
                fig, axs = plt.subplots(6, 5, figsize=(20, 24))
                fig.suptitle(
                    f"RNN for {taste_name} - Changepoint {changepoint_idx+1}",
                    fontsize=16,
                )

                for i, trial in enumerate(unique_trials):
                    if i >= 30:
                        break  # Limit to 30 trials per figure

                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    trial_changepoints = changepoints[taste_idx][trial]

                    changepoint_time = trial_changepoints[changepoint_idx]
                    time_window_mask = (trial_df["time"] >= changepoint_time - 250) & (
                        trial_df["time"] <= changepoint_time + 250
                    )
                    time_values = trial_df.filter(time_window_mask)["time"].to_numpy()

                    pc_columns = [
                        col for col in trial_df.columns if col.startswith("latent_")
                    ]
                    pc_values = [
                        trial_df.filter(time_window_mask)[col].to_numpy()
                        for col in pc_columns
                    ]

                    ax = axs[i // 5, i % 5]
                    for j, pc_series in enumerate(pc_values):
                        ax.plot(time_values, pc_series, label=f"{pc_columns[j]}")

                    ax.axvline(
                        changepoint_time,
                        color="black",
                        linestyle="--",
                        linewidth=2,
                        label="Changepoint",
                    )
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("RNN Latent")

                fig.legend(
                    [f"{col}" for col in pc_columns] + ["Changepoint"],
                    loc="lower center",
                    ncol=5,
                    fontsize="small",
                    frameon=False,
                )

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(
                    df_output_dir,
                    f"rnn_plots_taste_{taste_name}_changepoint_{changepoint_idx+1}.png",
                )
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")
                plt.close(fig)


# now plotting the activity intra-state
intra_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/INTRA_PCA_RNN/RNN_only"


def plot_pca_rnn_between_changepoints(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components from the start of one changepoint to the end of the next,
    organizing figures by taste and changepoint index, with each trial in a separate subplot.
    Each figure contains up to 16 plots (4x4 grid), creating multiple figures if necessary.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_changepoints = (
                len(changepoints[taste_idx][unique_trials[0]]) - 1
            )  # Define segments between changepoints

            for changepoint_idx in range(num_changepoints):
                trial_chunks = [
                    unique_trials[i : i + 16] for i in range(0, len(unique_trials), 16)
                ]

                for chunk_idx, trial_chunk in enumerate(trial_chunks):
                    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
                    fig.suptitle(
                        f"RNN for {taste_name} - Epoch {changepoint_idx+1} - Chunk {chunk_idx+1}",
                        fontsize=16,
                    )

                    for i, trial in enumerate(trial_chunk):
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        start_time = trial_changepoints[changepoint_idx]
                        end_time = trial_changepoints[changepoint_idx + 1]
                        time_window_mask = (trial_df["time"] >= start_time) & (
                            trial_df["time"] <= end_time
                        )
                        time_values = trial_df.filter(time_window_mask)[
                            "time"
                        ].to_numpy()

                        pc_columns = [
                            col for col in trial_df.columns if col.startswith("latent_")
                        ]
                        pc_values = [
                            trial_df.filter(time_window_mask)[col].to_numpy()
                            for col in pc_columns
                        ]

                        ax = axs[i // 4, i % 4]
                        for j, pc_series in enumerate(pc_values):
                            ax.plot(time_values, pc_series, label=f"{pc_columns[j]}")

                        ax.axvline(
                            start_time,
                            color="black",
                            linestyle="--",
                            linewidth=2,
                            label="Start Changepoint",
                        )
                        ax.axvline(
                            end_time,
                            color="black",
                            linestyle="--",
                            linewidth=2,
                            label="End Changepoint",
                        )
                        ax.set_title(f"Trial {trial}")
                        ax.set_xlabel("Time (ms)")
                        ax.set_ylabel("Principal Component Value")

                    fig.legend(
                        [f"{col}" for col in pc_columns]
                        + ["Start Changepoint", "End Changepoint"],
                        loc="lower center",
                        ncol=5,
                        fontsize="small",
                        frameon=False,
                    )

                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                    output_file = os.path.join(
                        df_output_dir,
                        f"rnn_plots_taste_{taste_name}_segment_{changepoint_idx+1}_chunk_{chunk_idx+1}.png",
                    )
                    plt.savefig(output_file)
                    print(f"Saved RNN plot figure: {output_file}")
                    plt.close(fig)


## all of this to its own class later

# now heat maps with latants, time-- seeing about changepoint alignmnet


## 2/16-- now figs where latents are plotted on one subplot (several subplots) -- this is going to have some issues for sure

pca_rnn_lat_plots_path_unal = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS"


def plot_pca_rnn_all_trials_per_transition(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components for all trials across each transition, organizing figures by taste.
    Each figure has 8x3 subplots: 8 latent vectors, 3 transitions.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_transitions = (
                len(changepoints[taste_idx][unique_trials[0]]) - 1
            )  # Number of transitions between changepoints

            fig, axs = plt.subplots(
                8, 3, figsize=(24, 32)
            )  # 8 latent vectors x 3 transitions
            fig.suptitle(
                f"PCA Latent Vectors for {taste_name} - Dataset {core_dataset_name}",
                fontsize=20,
            )

            pc_columns = [col for col in taste_df.columns if col.startswith("PC_")][
                :8
            ]  # Maximum 8 latent vectors

            for row_idx, pc_col in enumerate(pc_columns):
                for col_idx in range(min(num_transitions, 3)):  # Max 3 transitions
                    ax = axs[row_idx, col_idx]

                    end_times = []
                    for trial in unique_trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        start_time = trial_changepoints[col_idx]
                        end_time = trial_changepoints[col_idx + 1]
                        time_window_mask = (trial_df["time"] >= start_time) & (
                            trial_df["time"] <= end_time
                        )

                        time_values = trial_df.filter(time_window_mask)[
                            "time"
                        ].to_numpy()
                        pc_values = trial_df.filter(time_window_mask)[pc_col].to_numpy()

                        if len(time_values) > 0:
                            end_times.append(time_values[-1])
                            ax.plot(time_values, pc_values, alpha=0.5)

                    if (
                        end_times
                    ):  # adding in percentile lines for when each trial ends-- decently easy to visualize what's what
                        percentiles = np.percentile(end_times, [25, 50, 75])
                        ax.axvline(
                            percentiles[0],
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="25% Trials End",
                        )
                        ax.axvline(
                            percentiles[1],
                            color="blue",
                            linestyle="-.",
                            linewidth=2,
                            label="50% Trials End",
                        )
                        ax.axvline(
                            percentiles[2],
                            color="green",
                            linestyle=":",
                            linewidth=2,
                            label="75% Trials End",
                        )

                    ax.set_title(f"{pc_col} - Epoch {col_idx + 1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Principal Component Value")

            fig.legend(
                ["25% Trials End", "50% Trials End", "75% Trials End"],
                loc="lower center",
                ncol=3,
                fontsize="small",
                frameon=False,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            output_file = os.path.join(
                df_output_dir,
                f"pca_latent_vectors_{core_dataset_name}_{taste_name}.png",
            )
            plt.savefig(output_file)
            print(f"Saved PCA latent vectors plot: {output_file}")
            plt.close(fig)


## Now similar to the above but with an extra bit that aligns start times to the beginning of the frame for each plot

trans_aligned_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS/aligned_unwarped"


def plot_pca_rnn_all_trials_per_transition_aligned(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components for all trials across each transition, organizing figures by taste.
    Each figure has 8x3 subplots: 8 latent vectors, 3 transitions.
    Each subplot aligns trials to the mean start time and includes vertical lines marking where 25%, 50%, and 75% of trials end.
    The mean start time is noted in the figure title.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_transitions = (
                len(changepoints[taste_idx][unique_trials[0]]) - 1
            )  # Number of transitions between changepoints

            fig, axs = plt.subplots(
                8, 3, figsize=(24, 32)
            )  # 8 latent vectors x 3 transitions

            pc_columns = [col for col in taste_df.columns if col.startswith("PC_")][
                :8
            ]  # Maximum 8 latent vectors

            for row_idx, pc_col in enumerate(pc_columns):
                for col_idx in range(min(num_transitions, 3)):  # Max 3 transitions
                    ax = axs[row_idx, col_idx]

                    start_times = []
                    end_times = []
                    trial_curves = []

                    for trial in unique_trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        start_time = trial_changepoints[col_idx]
                        end_time = trial_changepoints[col_idx + 1]
                        start_times.append(start_time)

                        time_window_mask = (trial_df["time"] >= start_time) & (
                            trial_df["time"] <= end_time
                        )

                        time_values = trial_df.filter(time_window_mask)[
                            "time"
                        ].to_numpy()
                        pc_values = trial_df.filter(time_window_mask)[pc_col].to_numpy()

                        if len(time_values) > 0:
                            end_times.append(time_values[-1])
                            trial_curves.append(
                                (time_values - start_time, pc_values)
                            )  # Align start to 0 ms

                    if len(start_times) > 0:
                        mean_start_time = np.mean(start_times)
                        for time_values, pc_values in trial_curves:
                            ax.plot(time_values, pc_values, alpha=0.5)

                        if end_times:
                            percentiles = np.percentile(end_times, [25, 50, 75])
                            ax.axvline(
                                percentiles[0] - mean_start_time,
                                color="red",
                                linestyle="--",
                                linewidth=2,
                                label="25% Trials End",
                            )
                            ax.axvline(
                                percentiles[1] - mean_start_time,
                                color="blue",
                                linestyle="-.",
                                linewidth=2,
                                label="50% Trials End",
                            )
                            ax.axvline(
                                percentiles[2] - mean_start_time,
                                color="green",
                                linestyle=":",
                                linewidth=2,
                                label="75% Trials End",
                            )

                    ax.set_title(f"{pc_col} - Epoch {col_idx + 1}")
                    ax.set_xlabel("Time from Mean Start (ms)")
                    ax.set_ylabel("Principal Component Value")

            fig.suptitle(
                f"PCA Latent Vectors for {taste_name} - Dataset {core_dataset_name} (Aligned Start Time)",
                fontsize=20,
            )
            fig.legend(
                ["25% Trials End", "50% Trials End", "75% Trials End"],
                loc="lower center",
                ncol=3,
                fontsize="small",
                frameon=False,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            output_file = os.path.join(
                df_output_dir,
                f"pca_latent_vectors_{core_dataset_name}_{taste_name}_aligned.png",
            )
            plt.savefig(output_file)
            print(f"Saved PCA latent vectors plot: {output_file}")
            plt.close(fig)


## doing the same as above except this time we're interpolating trials to all be the same len
# 2/24 note with these-- there's a lot of uhhhh...garbage? being produced by this analysis.
# really have to think hard on what might be producing that garbage-- must double check that the correct dir is passed through
interpolated_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS/second_deriv_interpol"


def plot_pca_rnn_all_trials_interpolated(
    pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes
):
    """
    Plots RNN principal components for all trials across each transition, interpolated to the mean trial length.
    Each figure has 8x3 subplots: 8 latent vectors, 3 transitions.
    Each subplot aligns trials to the mean start time and interpolates them to the mean trial length.
    The mean start time and interpolation are noted in the figure title.

    Parameters:
    - pca_lat_dict: Dictionary of PCA-transformed DataFrames for each dataset.
    - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
    - output_dir: Directory to save the output plots.
    - modified_tastes: List of taste names corresponding to taste indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in pca_lat_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = (
                modified_tastes[taste_idx]
                if taste_idx < len(modified_tastes)
                else f"Unknown_Taste_{taste_idx}"
            )

            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()
            num_transitions = (
                len(changepoints[taste_idx][unique_trials[0]]) - 1
            )  # Number of transitions between changepoints

            fig, axs = plt.subplots(
                8, 3, figsize=(24, 32)
            )  # 8 latent vectors x 3 transitions

            pc_columns = [col for col in taste_df.columns if col.startswith("PC_")][
                :8
            ]  # Maximum 8 latent vectors
            # think about if there's a better way to optimize this-- as there's a lot of wasted graph space-- for future thought.

            for row_idx, pc_col in enumerate(pc_columns):
                for col_idx in range(min(num_transitions, 3)):  # Max 3 transitions
                    ax = axs[row_idx, col_idx]

                    start_times = []
                    end_times = []
                    trial_curves = []
                    trial_lengths = []

                    for trial in unique_trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        start_time = trial_changepoints[col_idx]
                        end_time = trial_changepoints[col_idx + 1]
                        start_times.append(start_time)

                        time_window_mask = (trial_df["time"] >= start_time) & (
                            trial_df["time"] <= end_time
                        )

                        time_values = trial_df.filter(time_window_mask)[
                            "time"
                        ].to_numpy()
                        pc_values = trial_df.filter(time_window_mask)[pc_col].to_numpy()

                        if len(time_values) > 1:
                            end_times.append(time_values[-1])
                            trial_lengths.append(time_values[-1] - time_values[0])
                            trial_curves.append(
                                (time_values - start_time, pc_values)
                            )  # Align start to 0 ms

                    if len(trial_lengths) > 0:
                        max_length = int(np.max(time_values[-1]))
                        # mean_length = int(np.mean(trial_lengths)) # mean len -- this was an issue with infrence and extrapolating values unnesecarily
                        mean_start_time = np.mean(start_times)
                        interpolated_times = np.linspace(0, max_length, max_length)

                        for time_values, pc_values in trial_curves:
                            # Interpolate each trial to the mean length
                            interpolator = interp1d(
                                time_values,
                                pc_values,
                                kind="nearest",
                                bounds_error="True",
                            )  # eliminate any sort of extrpolation of data
                            # issue to do with the shortest units causing severe outliers -- look into different types of interpolation
                            interpolated_values = interpolator(
                                np.linspace(time_values[0], time_values[-1], max_length)
                            )
                            ax.plot(interpolated_times, interpolated_values, alpha=0.5)

                        if end_times:
                            percentiles = np.percentile(end_times, [25, 50, 75])
                            ax.axvline(
                                np.percentile(interpolated_times, 25),
                                color="red",
                                linestyle="--",
                                linewidth=2,
                                label="25% Trials End",
                            )
                            ax.axvline(
                                np.percentile(interpolated_times, 50),
                                color="blue",
                                linestyle="-.",
                                linewidth=2,
                                label="50% Trials End",
                            )
                            ax.axvline(
                                np.percentile(interpolated_times, 75),
                                color="green",
                                linestyle=":",
                                linewidth=2,
                                label="75% Trials End",
                            )

                    ax.set_title(f"{pc_col} - Epoch {col_idx + 1}")
                    ax.set_xlabel("Interpolated Time (ms)")
                    ax.set_ylabel("Principal Component Value")

            fig.suptitle(
                f"Second Derivative PCA Latent Vectors for {taste_name} - Dataset {core_dataset_name} (Interpolated to Mean Trial Length)",
                fontsize=20,
            )
            fig.legend(
                ["25% Trials End", "50% Trials End", "75% Trials End"],
                loc="lower center",
                ncol=3,
                fontsize="small",
                frameon=False,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            output_file = os.path.join(
                df_output_dir,
                f"sec_deriv_pca_latent_vectors_{core_dataset_name}_{taste_name}_interpolated.png",
            )
            plt.savefig(output_file)
            print(f"Saved PCA latent vectors plot: {output_file}")
            plt.close(fig)


## QUANTIFICATION OF SOME OF THIS MESS: shuffling data and testing against it:

# new func:

# this should be working and I am not sure why it's not-- gotta talk to abu what's going on

import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, ks_2samp, kruskal


def get_data_columns(df: pl.DataFrame):
    # Return PC_x or latent_dim_x columns from a Polars DataFrame.
    # Useful for looping across dimensions.
    return [
        col
        for col in df.columns
        if col.startswith("PC_") or col.startswith("latent_dim_")
    ]


class SignificanceTester:
    def __init__(
        self,
        data_dict: dict,
        rng: np.random.Generator,
        save_dir: str,
        n_permutations: int = 10000,
    ):
        """
        Initialize the SignificanceTester class.

        Parameters
        ----------
        data_dict : dict
            Dictionary of dataset_name -> Polars DataFrames.
        rng : np.random.Generator
            Numpy random generator for reproducibility.
        save_dir : str
            Directory where text files will be saved.
        n_permutations : int
            Number of permutations for permutation testing.
        """
        self.data_dict = data_dict
        self.rng = rng
        self.save_dir = Path(save_dir)
        self.n_permutations = n_permutations

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        """
        Return PC_x or latent_dim_x columns from a Polars DataFrame.
        Useful for looping across latent dimensions.
        """
        return [
            col
            for col in df.columns
            if col.startswith("PC_") or col.startswith("latent_dim_")
        ]

    @staticmethod
    def permutation_test(actual_data, shuffled_data, rng, n_permutations):
        """
        Permutation test using controlled random generator (rng.permutation).
        """
        combined = np.concatenate([actual_data, shuffled_data])
        real_statistic = np.mean(actual_data) - np.mean(shuffled_data)

        n = len(actual_data)
        perm_stats = np.empty(n_permutations)

        for i in range(n_permutations):
            permuted = rng.permutation(
                combined
            )  # Use rng.permutation instead of np.random.shuffle
            perm_group1 = permuted[:n]
            perm_group2 = permuted[n:]
            perm_stats[i] = np.mean(perm_group1) - np.mean(perm_group2)

        p_val = np.mean(np.abs(perm_stats) >= np.abs(real_statistic))

        return p_val, real_statistic

    @staticmethod
    def colored_significance(is_significant):
        if is_significant:
            return "\033[92mSignificant\033[0m"  # Green
        else:
            return "\033[91mNot Significant\033[0m"  # Red

    @staticmethod
    def summarize_tests_console(p_vals_and_significance):
        print("\nSummary of significance across tests:")
        for test_name, results in p_vals_and_significance.items():
            p_val = results["p_val"]
            is_sig = results["significant"]
            color = "\033[92m" if is_sig else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {test_name}: {color}{'Significant' if is_sig else 'Not Significant'}{reset} (p = {p_val:.4f})"
            )
        print("-" * 50)

    def run(self):
        """
        Main function to run significance tests on all datasets.
        Returns
        -------
        dict
            Dataset_name -> List of result dicts.
        """
        results_summary = {}

        for dataset_name, df in self.data_dict.items():
            print(f"\nProcessing dataset: {dataset_name}")
            output_file = self.save_dir / f"{dataset_name}_significance_results.txt"
            sig_output_file = self.save_dir / f"{dataset_name}_significant_only.txt"

            all_lines = []
            sig_lines = []

            latent_cols = get_data_columns(df)
            unique_tastes = sorted(df["taste"].unique().to_list())
            unique_cps = sorted(df["changepoint"].unique().to_list())

            dataset_results = []

            for taste in unique_tastes:
                for cp in unique_cps:
                    sub_df = df.filter(
                        (pl.col("taste") == taste) & (pl.col("changepoint") == cp)
                    )

                    if sub_df.height == 0:
                        continue

                    result_entry = {
                        "taste": taste,
                        "changepoint": cp,
                        "results_per_latent": {},
                    }

                    for col in latent_cols:
                        trial_arrays = []

                        unique_trials = sorted(sub_df["trial"].unique().to_list())
                        for trial_id in unique_trials:
                            trial_df = sub_df.filter(pl.col("trial") == trial_id).sort(
                                "time"
                            )
                            trial_array = trial_df[col].to_numpy()
                            trial_arrays.append(trial_array)

                        if len(trial_arrays) < 2:
                            continue

                        actual_data = np.concatenate(trial_arrays)
                        shuffled_data = actual_data.copy()
                        self.rng.shuffle(shuffled_data)

                        # Statistical tests
                        t_stat, p_val_t = ttest_ind(
                            actual_data,
                            shuffled_data,
                            alternative="two-sided",
                            equal_var=False,
                        )
                        ks_stat, p_val_ks = ks_2samp(
                            actual_data, shuffled_data, alternative="two-sided"
                        )
                        h_stat, p_val_kruskal = kruskal(actual_data, shuffled_data)
                        p_val_perm, real_statistic = self.permutation_test(
                            actual_data, shuffled_data, self.rng, self.n_permutations
                        )
                        sig_ttest = p_val_t < 0.05
                        sig_ks = p_val_ks < 0.05
                        sig_kruskal = p_val_kruskal < 0.05
                        sig_perm = p_val_perm < 0.05

                        result_entry["results_per_latent"][col] = {
                            "t-test": {
                                "p_val": p_val_t,
                                "stat": t_stat,
                                "significant": sig_ttest,
                            },
                            "ks-test": {
                                "p_val": p_val_ks,
                                "stat": ks_stat,
                                "significant": sig_ks,
                            },
                            "kruskal": {
                                "p_val": p_val_kruskal,
                                "stat": h_stat,
                                "significant": sig_kruskal,
                            },
                            "permutation": {
                                "p_val": p_val_perm,
                                "real_stat": real_statistic,
                                "significant": sig_perm,
                            },
                        }

                        block = f"Taste {taste}, Changepoint {cp}, Latent {col}:\n"
                        block += f"  Welch's t-test: p = {p_val_t:.4f}, t = {t_stat:.4f} --> {'Significant' if sig_ttest else 'Not Significant'}\n"
                        block += f"  Kolmogorov-Smirnov test: p = {p_val_ks:.4f}, D = {ks_stat:.4f} --> {'Significant' if sig_ks else 'Not Significant'}\n"
                        block += f"  Kruskal-Wallis test: p = {p_val_kruskal:.4f}, H = {h_stat:.4f} --> {'Significant' if sig_kruskal else 'Not Significant'}\n"
                        block += f"  Permutation test: p = {p_val_perm:.4f}, diff = {real_statistic:.4f} --> {'Significant' if sig_perm else 'Not Significant'}\n\n"

                        all_lines.append(block)

                        if any([sig_ttest, sig_ks, sig_kruskal, sig_perm]):
                            sig_lines.append(block)

                        p_vals_and_significance = {
                            "Welch's t-test": {
                                "p_val": p_val_t,
                                "significant": sig_ttest,
                            },
                            "Kolmogorov-Smirnov test": {
                                "p_val": p_val_ks,
                                "significant": sig_ks,
                            },
                            "Kruskal-Wallis test": {
                                "p_val": p_val_kruskal,
                                "significant": sig_kruskal,
                            },
                            "Permutation test": {
                                "p_val": p_val_perm,
                                "significant": sig_perm,
                            },
                        }
                        self.summarize_tests_console(p_vals_and_significance)

                    dataset_results.append(result_entry)

            with open(output_file, "w") as f:
                f.writelines(all_lines)

            with open(sig_output_file, "w") as f:
                if sig_lines:
                    f.write(f"Significant Results for {dataset_name}\n\n")
                    f.writelines(sig_lines)
                else:
                    f.write(f"No significant results found for {dataset_name}.\n")

            results_summary[dataset_name] = dataset_results

        return results_summary


from numpy.random import default_rng

rng = default_rng(42)  # keeping a random seed to be working with -- for future use

tester = SignificanceTester(
    data_dict=robust_pca_95,  # your dictionary of polars DataFrames
    rng=rng,
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS/significance_testing/rob_95_pca_on_rnn",
    n_permutations=10000,
)

results_summary = tester.run()


# the below sucks
def analyze_significant_epochs(
    df: pl.DataFrame, dataset_name: str, rng: np.random.Generator
) -> pl.DataFrame:
    """
    For a given Polars DataFrame that contains latent dimension columns along with
    metadata columns ("taste", "trial", "changepoint", "time"), this function:

      1. For each (taste, changepoint) combination, groups the data by trial.
      2. For each latent column, stacks the time-series data of each trial (sorted by time)
         to form the 'actual' distribution.
      3. Creates a surrogate ('shuffled') distribution by shuffling the order of the trial arrays
         before stacking them.
      4. Runs a two-sample t-test between the actual and shuffled distributions.
      5. Prints a message if p < 0.05.
      6. Records for each latent column a tuple (p-value, t-statistic).

    The output DataFrame has one row per unique (taste, changepoint) combination, with
    the same column structure as the original (metadata fields "trial" and "time" set to None).
    """
    meta_cols = ["taste", "trial", "changepoint", "time"]
    latent_cols = [col for col in df.columns if col not in meta_cols]

    # Get unique taste and changepoint values.
    unique_tastes = sorted(df["taste"].unique().to_list())
    unique_cps = sorted(df["changepoint"].unique().to_list())

    rows = []

    for taste in unique_tastes:
        for cp in unique_cps:
            # Filter the data for this (taste, changepoint) combination.
            sub_df = df.filter(
                (pl.col("taste") == taste) & (pl.col("changepoint") == cp)
            )
            if sub_df.height == 0:
                continue  # Skip if no data exists for this combination.

            # Get unique trial identifiers.
            trial_ids = sorted(sub_df["trial"].unique().to_list())
            result_row = {"taste": taste, "changepoint": cp}

            for col in latent_cols:
                # For each trial, extract and sort data by time, then store the 1D array.
                trial_arrays = []
                for trial in trial_ids:
                    trial_df = sub_df.filter(pl.col("trial") == trial).sort("time")
                    trial_data = trial_df[col].to_numpy()
                    trial_arrays.append(trial_data)

                # Stack trials in natural order to form the actual data.
                actual_data = np.concatenate(trial_arrays)

                # For the shuffled distribution, shuffle the list of trial arrays before stacking.
                shuffled_trials = trial_arrays.copy()
                rng.shuffle(shuffled_trials)
                shuffled_data = np.concatenate(shuffled_trials)

                # Run the two-sample t-test between actual and shuffled distributions.
                t_stat, p_val = ttest_ind(
                    actual_data, shuffled_data, alternative="two-sided", equal_var=False
                )

                if p_val < 0.05:
                    print(
                        f"For dataset {dataset_name}, taste {taste} epoch {cp}, latent '{col}' is significant (p = {p_val:.3f})."
                    )

                result_row[col] = (p_val, t_stat)

            # For metadata fields that no longer apply after aggregation.
            result_row["trial"] = None
            result_row["time"] = None
            rows.append(result_row)

    # Construct the DataFrame with latent columns first, then metadata.
    new_columns_order = latent_cols + ["taste", "trial", "changepoint", "time"]
    result_df = pl.DataFrame(rows)[new_columns_order]
    return result_df


def quantify_significant_epochs(data_dict: dict, rng: np.random.Generator) -> dict:
    """
    Processes each DataFrame in the input dictionary (e.g., first_deriv_dict) and returns a dictionary
    (sig_epochs_marker) where each key is the same as in the input, and the corresponding value is a DataFrame
    with t-test results (p-value and t-statistic) for each (taste, changepoint) combination.

    Parameters:
        data_dict (dict): Dictionary of Polars DataFrames.
        rng (np.random.Generator): A NumPy random generator for shuffling.

    Returns:
        dict: Dictionary where keys are dataset names and values are marker DataFrames.
    """
    sig_epochs_marker = {}

    for dataset_name, df in data_dict.items():
        marker_df = analyze_significant_epochs(df, dataset_name, rng)
        sig_epochs_marker[dataset_name] = marker_df

    return sig_epochs_marker


# Example usage:
# Assume first_deriv_dict is your input dictionary of Polars DataFrames.
rng = np.random.default_rng()
sig_epochs_marker = quantify_significant_epochs(first_der_dict, rng)

# Now, sig_epochs_marker is a dict of DataFrames with t-test results.


### cheekly little spectrogram thrown in there:


# because we have space spect in this household
def safe_spectrogram(data, fs=1000.0):
    """
    Computes a spectrogram on 'data', automatically handling the case where
    len(data) < default nperseg (256). Also ensures noverlap < nperseg.
    """
    # Pick your base nperseg (e.g., 256). If data is shorter, reduce nperseg accordingly:
    nperseg = min(16, len(data))  # this is very changable; calibrate to enviornent

    # Choose noverlap so that it’s strictly less than nperseg
    # e.g. 50% overlap -> noverlap = nperseg // 2
    noverlap = (
        nperseg // 3
    )  # or int(0.75 * nperseg), etc. # this is also very changable and should be calibrated

    # If the data is extremely short or empty, handle gracefully:
    if len(data) < 2:
        return None, None, None  # or handle however you wish

    f_spect, t_spect, Sxx = spectrogram(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    return f_spect, t_spect, Sxx


spect_output = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/SPECTROGRAM"


def plot_spectrograms_for_latents(
    epoch_dataframes_dict, standardized_changepoints_dict, output_dir
):
    """
    For each dataset's DataFrame -> each taste -> each latent dimension,
    chunk trials in groups of 5. Each figure has up to 5 subplots
    (one trial per subplot). Overlays changepoints in ms.
    The x-axis is in ms (no longer seconds), and the frequency axis is 0–25 Hz.
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        if core_dataset_name not in standardized_changepoints_dict:
            print(f"No changepoints found for '{core_dataset_name}'. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        dataset_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for this taste
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            n_dims = 8  # Adjust if you have more or fewer dims
            for dim_idx in range(n_dims):
                dim_col = f"latent_dim_{dim_idx}"
                if dim_col not in taste_df.columns:
                    continue  # skip if this dimension doesn't exist

                # Break trials into chunks of 5 (so each figure has up to 5 subplots)
                trial_chunks = [
                    unique_trials[i : i + 5] for i in range(0, len(unique_trials), 5)
                ]

                for chunk_i, chunk_trials in enumerate(trial_chunks, start=1):
                    fig = plt.figure(figsize=(12, 16))
                    fig.suptitle(
                        f"Spectrograms (ms)\n"
                        f"Dataset={df_name}, Taste={taste}, Dim={dim_idx}, "
                        f"Trials {chunk_trials}",
                        fontsize=14,
                    )

                    for subplot_idx, trial in enumerate(chunk_trials):
                        ax = fig.add_subplot(5, 1, subplot_idx + 1)
                        trial_df = taste_df.filter(pl.col("trial") == trial)

                        # Grab the times (ms)
                        trial_times = trial_df["time"].to_numpy()
                        if len(trial_times) < 2:
                            continue  # too short or empty trial

                        min_time_ms = trial_times[0]
                        max_time_ms = trial_times[-1]
                        data = trial_df[dim_col].to_numpy()

                        # Check how many samples we have
                        # Debug print example:
                        # print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}, "
                        #       f"time_min={min_time_ms}, time_max={max_time_ms}")

                        # Compute spectrogram with fs=1000
                        f_spect, t_spect, Sxx = safe_spectrogram(data, fs=1000.0)
                        print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}")
                        if f_spect is None:
                            # Data too short
                            continue

                        # Convert the spectrogram's time axis from seconds to ms
                        t_spect_ms = t_spect * 1000.0

                        # Shift so that time=0 in the spectrogram lines up with the trial's min_time_ms
                        t_spect_total = t_spect_ms + min_time_ms

                        # Plot the spectrogram in dB
                        cmesh = ax.pcolormesh(
                            t_spect_total, f_spect, 10 * np.log10(Sxx), shading="auto"
                        )

                        # Plot changepoints (they're presumably in ms already)
                        trial_changepoints = []
                        if (
                            taste_idx < len(changepoints)
                            and trial in changepoints[taste_idx]
                        ):
                            trial_changepoints = changepoints[taste_idx][trial]

                        # Overlay each changepoint in ms
                        for cp_ms in trial_changepoints:
                            ax.axvline(
                                cp_ms, linestyle="--", color="green", linewidth=2
                            )

                        # Frequency limit 0–25 Hz
                        ax.set_ylim(0, 15)

                        # X-axis in ms
                        ax.set_xlim(min_time_ms, max_time_ms)

                        ax.set_title(f"Trial {trial}", fontsize=12)
                        if subplot_idx == len(chunk_trials) - 1:
                            ax.set_xlabel("Time (ms)")
                        ax.set_ylabel("Freq (Hz)")

                    # Adjust layout
                    fig.tight_layout(rect=[0, 0, 1, 0.94])

                    # Single colorbar for the figure
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(cmesh, cax=cbar_ax, label="Power (dB)")

                    # Save figure
                    output_filename = f"spect_{core_dataset_name}_taste_{taste}_dim_{dim_idx}_chunk_{chunk_i}.png"
                    output_path = os.path.join(dataset_output_dir, output_filename)
                    plt.savefig(output_path, dpi=150)
                    plt.close(fig)

                    print(f"Saved spectrogram figure: {output_path}")

    print("Finished generating all spectrogram plots.")


# trying a new thing out.... idk if this will work


def uniform_spectrogram(data, start_ms, end_ms):
    """
    1) Uniformly space 'data' between start_ms and end_ms.
    2) Compute an effective fs so that we treat the entire data
       as one uniform chunk from 0 -> (end_ms - start_ms) in ms.
    3) Return (f_spect, t_spect, Sxx), with t_spect in seconds
       from 0 -> total_duration_s. We'll still shift it afterward
       to [start_ms, end_ms].
    """
    # If data is extremely short, abort
    if len(data) < 2:
        return None, None, None

    range_ms = end_ms - start_ms
    if range_ms < 1:
        # No real range -> can't do a meaningful spectrogram
        return None, None, None

    # Create a uniform time array for these samples: 0..range_ms
    # (We won't actually pass this time array to spectrogram, but
    #  we need to figure out an effective fs that matches this layout.)
    n_samples = len(data)

    # Effective sample frequency (in Hz) so that n_samples spans 'range_ms' in milliseconds
    # total duration in seconds = range_ms / 1000
    # n_samples - 1 intervals in that duration:
    fs_eff = (n_samples - 1) / (range_ms / 1000.0)

    # Now run spectrogram with that effective fs
    # We'll do smaller nperseg if data is short
    nperseg = min(48, n_samples)
    noverlap = nperseg // 2
    f_spect, t_spect, Sxx = spectrogram(
        data,
        fs=fs_eff,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    return f_spect, t_spect, Sxx, fs_eff


def plot_spectrograms_for_latents(
    epoch_dataframes_dict, standardized_changepoints_dict, output_dir
):
    """
    For each dataset's DataFrame -> each taste -> each latent dimension,
    chunk trials in groups of 5. Each figure has up to 5 subplots
    (one trial per subplot). Overlays changepoints in ms, converted to seconds.

    Frequency axis (y-axis) is truncated at 25 Hz; time axis goes from the actual
    earliest to latest ms of each trial (converted to s).
    """
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Extract base dataset name
        core_dataset_name = df_name.split("_repacked_raw_latent_vectors")[0]

        # Skip if no changepoints
        if core_dataset_name not in standardized_changepoints_dict:
            print(f"No changepoints found for '{core_dataset_name}'. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create output folder for this dataset
        dataset_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Unique tastes
        unique_tastes = df["taste"].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for this taste
            taste_df = df.filter(pl.col("taste") == taste)
            unique_trials = taste_df["trial"].unique().to_list()

            # We'll create one figure per dimension for every chunk of 5 trials
            n_dims = 8  # or however many latent dims you have

            for dim_idx in range(n_dims):
                dim_col = f"latent_dim_{dim_idx}"
                if dim_col not in taste_df.columns:
                    # Skip if this dimension doesn't exist in the data
                    continue

                # Break the trials into chunks of 5
                trial_chunks = [
                    unique_trials[i : i + 5] for i in range(0, len(unique_trials), 5)
                ]

                for chunk_i, chunk_trials in enumerate(trial_chunks, start=1):
                    # Create a figure that will have up to 5 subplots (one for each trial)
                    fig = plt.figure(figsize=(10, 16))
                    fig.suptitle(
                        f"Spectrograms\n"
                        f"Dataset={df_name}, Taste={taste}, Dim={dim_idx}, "
                        f"Trials {chunk_trials}",
                        fontsize=14,
                    )

                    # For each trial in this chunk
                    for subplot_idx, trial in enumerate(chunk_trials):
                        # Subplot for this trial
                        ax = fig.add_subplot(5, 1, subplot_idx + 1)

                        # Filter data for this trial
                        trial_df = taste_df.filter(pl.col("trial") == trial)

                        # Grab this trial's array of times (ms)
                        trial_times = trial_df["time"].to_numpy()
                        if trial_times.size == 0:
                            continue

                        min_time_ms = trial_times[0]
                        max_time_ms = trial_times[-1]

                        # Grab the latent dimension data
                        data = trial_df[dim_col].to_numpy()

                        # Compute spectrogram
                        f_spect, t_spect, Sxx = safe_spectrogram(data, fs=1000.0)
                        print(f"trial={trial}, dim={dim_idx}, data_size={len(data)}")
                        if f_spect is None:
                            # too short or empty
                            print("Skipping spectrogram: data is too short.")
                            continue

                        # Shift spectrogram's time axis so that 0 corresponds to min_time_ms
                        # i.e. each spectrogram time bin = t_spect + (start_time_in_s)
                        t_spect_shifted = t_spect + (min_time_ms / 1000.0)

                        # Plot the spectrogram in dB
                        cmesh = ax.pcolormesh(
                            t_spect_shifted, f_spect, 10 * np.log10(Sxx), shading="auto"
                        )

                        # Overlay changepoints (ms -> s)
                        trial_changepoints = []
                        if (
                            taste_idx < len(changepoints)
                            and trial in changepoints[taste_idx]
                        ):
                            trial_changepoints = changepoints[taste_idx][trial]

                        for cp_time_ms in trial_changepoints:
                            ax.axvline(
                                cp_time_ms / 1000.0,
                                linestyle="--",
                                color="white",
                                linewidth=2,
                            )

                        # Limit freq axis from 0 to 25 Hz -- chagne this as needed
                        # ax.set_ylim(0, 25)

                        # Set time axis from min_time_ms -> max_time_ms (in s)
                        ax.set_xlim(min_time_ms / 1000.0, max_time_ms / 1000.0)

                        # Add subplot title
                        ax.set_title(f"Trial {trial}", fontsize=12)

                        # Add labels only for the bottom subplot or so
                        if subplot_idx == len(chunk_trials) - 1:
                            ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Freq (Hz)")

                    # Adjust layout, add colorbar
                    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
                    # Add a single colorbar for the figure, right side:
                    # (to do that gracefully, can do fig.colorbar(..., ax=axes, location='right')
                    # but since we have subplots, a quick approach is to just add it to the last subplot)
                    #
                    # Alternatively, you could individually create colorbars in each subplot if you prefer.
                    # For a single colorbar across all subplots:
                    cbar_ax = fig.add_axes(
                        [0.92, 0.15, 0.02, 0.7]
                    )  # x, y, width, height
                    fig.colorbar(cmesh, cax=cbar_ax, label="Power (dB)")

                    # Save figure
                    output_filename = f"spect_{core_dataset_name}_taste_{taste}_dim_{dim_idx}_chunk_{chunk_i}.png"
                    output_path = os.path.join(dataset_output_dir, output_filename)
                    plt.savefig(output_path, dpi=150)
                    plt.close(fig)

                    print(f"Saved spectrogram figure: {output_path}")

    print("Finished generating all spectrogram plots.")


## A periodogram as well


### NOW FOR THE FFT-- studying the oscillatory stuff
## all the above is hooey-- this is where we're at now:

import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy.signal import find_peaks, periodogram, spectrogram, lombscargle
from scipy.fft import fft, fftfreq
from tqdm import tqdm


# definitely going to have to break this up by changepoint...not looking forward (the data rates will be unbelieveable)


class FrequencyAnalysisPipeline:
    def __init__(
        self,
        tld,
        standardized_changepoints_dict,
        modified_tastes,
        peak_height_threshold=0.05,
        spectrogram_nperseg=32,
        min_freq=1,
        max_freq=np.inf,
        start_time=1500,
        end_time=4500,
        do_fft=True,
        do_periodogram=True,
        do_peaks=True,
        do_lombscargle=True,
        do_full_trials=True,
        do_spectrogram=True,
        do_individual_plots=True,
        smart_skip=False,
        min_amplitude_threshold=0.0,
    ):
        self.tld = tld
        self.standardized_changepoints_dict = standardized_changepoints_dict
        self.modified_tastes = modified_tastes
        self.peak_height_threshold = peak_height_threshold
        self.spectrogram_nperseg = spectrogram_nperseg
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.start_time = start_time
        self.end_time = end_time

        self.do_fft = do_fft
        self.do_periodogram = do_periodogram
        self.do_peaks = do_peaks
        self.do_lombscargle = do_lombscargle
        self.do_full_trials = do_full_trials
        self.do_spectrogram = do_spectrogram
        self.do_individual_plots = do_individual_plots
        self.smart_skip = smart_skip
        self.min_amplitude_threshold = min_amplitude_threshold

        os.makedirs(tld, exist_ok=True)

        print("Frequency Analysis Pipeline Settings:")
        print(
            f"FFT: {self.do_fft}, Periodogram: {self.do_periodogram}, Peaks: {self.do_peaks}, Lomb-Scargle: {self.do_lombscargle}"
        )
        print(
            f"Full Trials: {self.do_full_trials}, Spectrogram: {self.do_spectrogram}, Individual Plots: {self.do_individual_plots}, Smart Skip: {self.smart_skip}"
        )
        print(
            f"Min Frequency: {self.min_freq} Hz, Max Frequency: {self.max_freq} Hz, Min Amplitude Threshold: {self.min_amplitude_threshold}"
        )

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        return [
            col
            for col in df.columns
            if col.startswith("PC_") or col.startswith("latent_dim_")
        ]

    def run(self, dataset_dict):
        for dataset_name, df in tqdm(dataset_dict.items(), desc="Processing datasets"):
            self._analyze_dataset(dataset_name, df)

    def _analyze_dataset(self, dataset_name, df):
        print(f"Processing dataset: {dataset_name}")
        data_cols = self.get_data_columns(df)
        tastes = df["taste"].unique().to_list()
        core_dataset_name = dataset_name.split("_repacked_raw_latent_vectors")[0]

        dataset_dir = os.path.join(self.tld, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        for taste_idx, taste in enumerate(tastes):
            taste_dir = os.path.join(dataset_dir, f"taste_{taste}")
            analysis_dirs = {
                "fft": os.path.join(taste_dir, "fft"),
                "peaks": os.path.join(taste_dir, "peaks"),
                "periodogram": os.path.join(taste_dir, "periodogram"),
                "spectrogram": os.path.join(taste_dir, "spectrogram"),
                "lombscargle_periodogram": os.path.join(
                    taste_dir, "lombscargle_periodogram"
                ),
                "individual_pc_lombscargle": os.path.join(
                    taste_dir, "lombscargle_periodogram", "individual-pc"
                ),
                "individual_pc_periodogram": os.path.join(
                    taste_dir, "periodogram", "individual-pc"
                ),
                "individual_pc_fft": os.path.join(taste_dir, "fft", "individual-pc"),
                "individual_pc_peaks": os.path.join(taste_dir, "peaks", "individual"),
                "full_trial_plots": os.path.join(taste_dir, "full_trial_plots"),
            }

            # Smart skip check BEFORE creating directories
            skip_flags = {}
            for analysis_type in [
                "fft",
                "periodogram",
                "peaks",
                "lombscargle_periodogram",
                "spectrogram",
            ]:
                path = analysis_dirs[analysis_type]
                skip_flags[analysis_type] = (
                    self.smart_skip and os.path.exists(path) and os.listdir(path)
                )

            for analysis_type, should_skip in skip_flags.items():
                if should_skip:
                    print(
                        f"[Smart Skip] Skipping {analysis_type.replace('_', ' ').title()} analysis for dataset {dataset_name}, taste {taste}."
                    )

            # Create the directories if they do not exist
            for path in analysis_dirs.values():
                os.makedirs(path, exist_ok=True)

            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()

            for trial in trials:
                trial_df = taste_df.filter(pl.col("trial") == trial)
                time = trial_df["time"].to_numpy() / 1000.0

                if self.do_fft and not skip_flags["fft"]:
                    combined_fig_fft, combined_ax_fft = plt.subplots(figsize=(8, 5))
                if self.do_periodogram and not skip_flags["periodogram"]:
                    combined_fig_periodogram, combined_ax_periodogram = plt.subplots(
                        figsize=(8, 5)
                    )
                if self.do_peaks and not skip_flags["peaks"]:
                    combined_fig_peaks, combined_ax_peaks = plt.subplots(figsize=(8, 5))
                if self.do_lombscargle and not skip_flags["lombscargle_periodogram"]:
                    combined_fig_lombscargle, combined_ax_lombscargle = plt.subplots(
                        figsize=(8, 5)
                    )

                for col in data_cols:
                    signal = trial_df[col].to_numpy()
                    trial_prefix = f"trial_{trial}_{col}"

                    if (self.do_fft or self.do_peaks) and not skip_flags["fft"]:
                        freqs, fft_magnitude = self.compute_fft(signal, time * 1000)
                        mask = (
                            (freqs >= self.min_freq)
                            & (freqs <= self.max_freq)
                            & (fft_magnitude >= self.min_amplitude_threshold)
                        )

                    if self.do_fft and not skip_flags["fft"]:
                        if self.do_individual_plots:
                            fft_txt_path = os.path.join(
                                analysis_dirs["individual_pc_fft"],
                                f"{trial_prefix}_fft - {core_dataset_name}.txt",
                            )
                            fft_plot_path = os.path.join(
                                analysis_dirs["individual_pc_fft"],
                                f"{trial_prefix}_fft_peaks.png",
                            )
                            if not (self.smart_skip and os.path.exists(fft_txt_path)):
                                self.save_fft(
                                    freqs[mask],
                                    fft_magnitude[mask],
                                    analysis_dirs["individual_pc_fft"],
                                    trial_prefix,
                                    core_dataset_name,
                                )
                            if not (self.smart_skip and os.path.exists(fft_plot_path)):
                                self.plot_individual_fft(
                                    freqs[mask],
                                    fft_magnitude[mask],
                                    [],
                                    analysis_dirs["individual_pc_fft"],
                                    trial_prefix,
                                    core_dataset_name,
                                )
                        combined_ax_fft.plot(
                            freqs[mask], fft_magnitude[mask], label=col
                        )

                    if self.do_peaks and not skip_flags["peaks"]:
                        peaks, properties = find_peaks(
                            fft_magnitude, height=self.peak_height_threshold
                        )
                        if self.do_individual_plots:
                            peaks_txt_path = os.path.join(
                                analysis_dirs["individual_pc_peaks"],
                                f"{trial_prefix}_peaks - {core_dataset_name}.txt",
                            )
                            if not (self.smart_skip and os.path.exists(peaks_txt_path)):
                                self.save_peaks(
                                    freqs,
                                    fft_magnitude,
                                    peaks,
                                    properties,
                                    analysis_dirs["individual_pc_peaks"],
                                    trial_prefix,
                                    core_dataset_name,
                                )
                        combined_ax_peaks.plot(
                            freqs[peaks], fft_magnitude[peaks], label=col
                        )

                    if self.do_periodogram and not skip_flags["periodogram"]:
                        if self.do_individual_plots:
                            periodogram_plot_path = os.path.join(
                                analysis_dirs["individual_pc_periodogram"],
                                f"{trial_prefix}_periodogram.png",
                            )
                            if not (
                                self.smart_skip
                                and os.path.exists(periodogram_plot_path)
                            ):
                                self.plot_individual_periodogram(
                                    signal,
                                    time * 1000,
                                    analysis_dirs["individual_pc_periodogram"],
                                    trial_prefix,
                                    core_dataset_name,
                                )
                        fs = 1000.0 / np.mean(np.diff(time) * 1000)
                        freqs_comb, pxx_comb = periodogram(signal, fs=fs)
                        mask_comb = (freqs_comb >= self.min_freq) & (
                            freqs_comb <= self.max_freq
                        )  # << ONLY freq filtering
                        combined_ax_periodogram.semilogy(
                            freqs_comb[mask_comb], pxx_comb[mask_comb], label=col
                        )

                    # if self.do_periodogram and not skip_flags['periodogram']:
                    #     if self.do_individual_plots:
                    #         periodogram_plot_path = os.path.join(analysis_dirs['individual_pc_periodogram'], f"{trial_prefix}_periodogram.png")
                    #         if not (self.smart_skip and os.path.exists(periodogram_plot_path)):
                    #             self.plot_individual_periodogram(signal, time * 1000, analysis_dirs['individual_pc_periodogram'], trial_prefix, core_dataset_name)
                    #     fs = 1000.0 / np.mean(np.diff(time) * 1000)
                    #     freqs_comb, pxx_comb = periodogram(signal, fs=fs)
                    #     mask_comb = (freqs_comb >= self.min_freq) & (freqs_comb <= self.max_freq) & (pxx_comb >= self.min_amplitude_threshold)
                    #     combined_ax_periodogram.semilogy(freqs_comb[mask_comb], pxx_comb[mask_comb], label=col)

                    if (
                        self.do_lombscargle
                        and not skip_flags["lombscargle_periodogram"]
                    ):
                        if self.do_individual_plots:
                            lombscargle_plot_path = os.path.join(
                                analysis_dirs["individual_pc_lombscargle"],
                                f"{trial_prefix}_lombscargle.png",
                            )
                            if not (
                                self.smart_skip
                                and os.path.exists(lombscargle_plot_path)
                            ):
                                self.plot_lombscargle_periodogram(
                                    signal,
                                    time,
                                    analysis_dirs["individual_pc_lombscargle"],
                                    trial_prefix,
                                    core_dataset_name,
                                )
                        freqs_lomb = np.linspace(self.min_freq, self.max_freq, 1000)
                        angular_freqs = 2 * np.pi * freqs_lomb
                        power = lombscargle(time, signal, angular_freqs)
                        mask_lomb = power >= self.min_amplitude_threshold
                        combined_ax_lombscargle.plot(
                            freqs_lomb[mask_lomb], power[mask_lomb], label=col
                        )

                    if self.do_spectrogram and not skip_flags["spectrogram"]:
                        peaks, _ = find_peaks(signal)
                        if len(peaks) > 0 and self.do_individual_plots:
                            spectrogram_path = os.path.join(
                                analysis_dirs["spectrogram"],
                                f"{trial_prefix}_spectrogram.png",
                            )
                            if not (
                                self.smart_skip and os.path.exists(spectrogram_path)
                            ):
                                self.plot_spectrogram(
                                    signal,
                                    time * 1000,
                                    analysis_dirs["spectrogram"],
                                    trial_prefix,
                                    core_dataset_name,
                                )

                if self.do_fft and not skip_flags["fft"]:
                    combined_ax_fft.set_title(
                        f"Combined FFT - {core_dataset_name} - Taste {taste} - Trial {trial}"
                    )
                    combined_ax_fft.set_xlabel("Frequency (Hz)")
                    combined_ax_fft.set_ylabel("Magnitude")
                    combined_ax_fft.legend(fontsize="small")
                    combined_fig_fft.tight_layout()
                    combined_fig_fft.savefig(
                        os.path.join(
                            analysis_dirs["fft"], f"trial_{trial}_combined_fft.png"
                        )
                    )
                    plt.close(combined_fig_fft)

                if self.do_periodogram and not skip_flags["periodogram"]:
                    combined_ax_periodogram.set_title(
                        f"Combined Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}"
                    )
                    combined_ax_periodogram.set_xlabel("Frequency (Hz)")
                    combined_ax_periodogram.set_ylabel("Power Spectral Density")
                    combined_ax_periodogram.legend(fontsize="small")
                    combined_fig_periodogram.tight_layout()
                    combined_fig_periodogram.savefig(
                        os.path.join(
                            analysis_dirs["periodogram"],
                            f"trial_{trial}_combined_periodogram.png",
                        )
                    )
                    plt.close(combined_fig_periodogram)

                if self.do_peaks and not skip_flags["peaks"]:
                    combined_ax_peaks.set_title(
                        f"Combined Peaks - {core_dataset_name} - Taste {taste} - Trial {trial}"
                    )
                    combined_ax_peaks.set_xlabel("Frequency (Hz)")
                    combined_ax_peaks.set_ylabel("Magnitude")
                    combined_ax_peaks.legend(fontsize="small")
                    combined_fig_peaks.tight_layout()
                    combined_fig_peaks.savefig(
                        os.path.join(
                            analysis_dirs["peaks"], f"trial_{trial}_combined_peaks.png"
                        )
                    )
                    plt.close(combined_fig_peaks)

                if self.do_lombscargle and not skip_flags["lombscargle_periodogram"]:
                    combined_ax_lombscargle.set_title(
                        f"Combined Lomb-Scargle Periodogram - {core_dataset_name} - Taste {taste} - Trial {trial}"
                    )
                    combined_ax_lombscargle.set_xlabel("Frequency (Hz)")
                    combined_ax_lombscargle.set_ylabel("Power")
                    combined_ax_lombscargle.legend(fontsize="small")
                    combined_fig_lombscargle.tight_layout()
                    combined_fig_lombscargle.savefig(
                        os.path.join(
                            analysis_dirs["lombscargle_periodogram"],
                            f"trial_{trial}_combined_lombscargle.png",
                        )
                    )
                    plt.close(combined_fig_lombscargle)

            if self.do_full_trials:
                self._plot_full_trials_for_taste(
                    core_dataset_name,
                    taste_idx,
                    taste_df,
                    analysis_dirs["full_trial_plots"],
                )

    def _plot_full_trials_for_taste(
        self, core_dataset_name, taste_idx, taste_df, output_dir
    ):
        unique_trials = taste_df["trial"].unique().to_list()
        pc_columns = self.get_data_columns(taste_df)

        if core_dataset_name not in self.standardized_changepoints_dict:
            print(
                f"Changepoints not found for {core_dataset_name}. Skipping full trial plots."
            )
            return

        changepoints = self.standardized_changepoints_dict[core_dataset_name]

        for trial_chunk in tqdm(
            range(0, len(unique_trials), 4), desc=f"Taste {taste_idx} full trial plots"
        ):
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))
            fig.suptitle(
                f"Full Trial RNN Plots for {core_dataset_name} - Taste {taste_idx}",
                fontsize=16,
            )

            for i, trial in enumerate(unique_trials[trial_chunk : trial_chunk + 4]):
                trial_df = taste_df.filter(pl.col("trial") == trial)
                time_values = trial_df["time"].to_numpy()
                pc_values = [trial_df[col].to_numpy() for col in pc_columns]

                mask = (time_values >= self.start_time) & (time_values <= self.end_time)
                time_values = time_values[mask]
                pc_values = [pc[mask] for pc in pc_values]

                ax = axs[i] if len(unique_trials) > 1 else axs

                colors = plt.cm.tab10.colors  # Get default matplotlib color cycle
                legend_entries = []

                for j, pc_series in enumerate(pc_values):
                    color = colors[j % len(colors)]
                    ax.plot(
                        time_values, pc_series, label=f"{pc_columns[j]}", color=color
                    )
                    peaks, _ = find_peaks(pc_series)
                    ax.plot(
                        time_values[peaks],
                        pc_series[peaks],
                        "x",
                        color=color,
                        label=f"{pc_columns[j]}_peak",
                    )

                trial_changepoints = changepoints[taste_idx][trial]
                for changepoint_time in trial_changepoints:
                    if self.start_time <= changepoint_time <= self.end_time:
                        ax.axvline(
                            changepoint_time,
                            color="black",
                            linestyle="--",
                            linewidth=2,
                            label="Changepoint",
                        )

                if self.start_time <= 2000 <= self.end_time:
                    ax.axvline(
                        2000,
                        color="black",
                        linestyle=":",
                        linewidth=2,
                        label="Stimulus Delivery",
                    )

                ax.set_title(f"Trial {trial}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Principal Component Value")

            handles, labels = axs[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                ncol=5,
                fontsize="small",
                frameon=False,
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            output_file = os.path.join(
                output_dir,
                f"rnn_plots_taste_{taste_idx}_chunk_{trial_chunk // 4 + 1}.png",
            )
            plt.savefig(output_file)
            plt.close(fig)

    def compute_fft(self, signal, time):
        n = len(signal)
        dt = np.mean(np.diff(time)) / 1000.0
        fft_vals = fft(signal)
        fft_magnitude = np.abs(fft_vals)[: n // 2]
        freqs = fftfreq(n, dt)[: n // 2]
        return freqs, fft_magnitude

    def save_fft(self, freqs, fft_magnitude, base_dir, name, dataset_name):
        out_path = os.path.join(base_dir, f"{name}_fft - {dataset_name}.txt")
        np.savetxt(
            out_path,
            np.column_stack((freqs, fft_magnitude)),
            header="Frequency(Hz) FFT_Magnitude",
            fmt="%0.6f",
        )

    def save_peaks(
        self, freqs, fft_magnitude, peaks, properties, base_dir, name, dataset_name
    ):
        out_path = os.path.join(base_dir, f"{name}_peaks - {dataset_name}.txt")
        with open(out_path, "w") as f:
            for peak_idx in peaks:
                if self.min_freq <= freqs[peak_idx] <= self.max_freq:
                    f.write(
                        f"Freq={freqs[peak_idx]:.3f} Hz, Height={properties['peak_heights'][np.where(peaks == peak_idx)[0][0]]:.3f}\n"
                    )

    def plot_individual_fft(
        self, freqs, fft_magnitude, peaks, base_dir, name, dataset_name
    ):
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[mask], fft_magnitude[mask], label="FFT Magnitude")
        plt.plot(freqs[peaks], fft_magnitude[peaks], label="Peaks")
        plt.title(f"Individual FFT {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_fft_peaks.png"))
        plt.close()

    def plot_individual_periodogram(self, signal, time, base_dir, name, dataset_name):
        fs = 1000.0 / np.mean(np.diff(time))
        freqs, pxx = periodogram(signal, fs=fs)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        plt.figure(figsize=(8, 4))
        plt.semilogy(freqs[mask], pxx[mask])
        plt.title(f"Individual Periodogram {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_periodogram.png"))
        plt.close()

    def plot_lombscargle_periodogram(self, signal, time, base_dir, name, dataset_name):
        freqs = np.linspace(self.min_freq, self.max_freq, 1000)
        angular_freqs = 2 * np.pi * freqs
        power = lombscargle(time, signal, angular_freqs)
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, power)
        plt.title(f"Individual Lomb-Scargle Periodogram {name} - {dataset_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_lombscargle.png"))
        plt.close()

    def plot_spectrogram(self, signal, time, base_dir, name, dataset_name):
        fs = 1000.0 / np.mean(np.diff(time))
        nperseg = min(self.spectrogram_nperseg, len(signal))
        noverlap = nperseg // 2

        f_spect, t_spect, Sxx = spectrogram(
            signal,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        t_spect_ms = t_spect * 1000
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(t_spect_ms, f_spect, 10 * np.log10(Sxx), shading="auto")
        plt.colorbar(label="Power (dB)")
        plt.title("Spectrogram - {dataset_name}")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"{name}_spectrogram.png"))
        plt.close()

    # this should be a familiar plot lol


tld = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/FFT_FREQUENCY_SUITE"
pipeline = FrequencyAnalysisPipeline(
    tld=tld,
    standardized_changepoints_dict=standardized_changepoints_dict,
    modified_tastes=modified_tastes,
    peak_height_threshold=0.25,
    spectrogram_nperseg=8,
    min_freq=2,  # You can set this!
    max_freq=15,
    do_fft=True,
    do_periodogram=True,
    do_peaks=True,
    do_lombscargle=True,
    do_full_trials=True,
    do_spectrogram=True,  # And this too!
    do_individual_plots=True,
    smart_skip=True,
    min_amplitude_threshold=0.05,
)

# dataset_dict = {dataset_name: polars_dataframe}
pipeline.run(first_derivs)

# further improvments for this:
# a text file in the dataset dir that will tell me what's signficant and where
# batching plots on figs so I don't have a billion files
# RUNNING THIS ON AN EPOCH-BY-EPOCH BASIS:
# only do if the periodograms come back with some level of high-power oscillatory significance
# as otherwise this would take practically forever
# figure out quantificaiton of freuqencies as well


#### 4/15: Optimized version of this code to deal in alignment and warping and stuff--
# (defining a class for this business as needed-- this is taking a lot of time for not a whole lot of return).

# class moved to its own module; func that calls the plotting below:
import os
from spike_raster_class_plot_april import (
    SpikeRasterPlotter,
)  # assumes the class is in this module


# new one:
def process_and_plot_datasets(
    npz_path,
    pkl_path,
    base_output_dir,
    standardized_changepoints_dict=None,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000,
):
    """
    CLI-enhanced pipeline for generating raster plots based on user options.
    Now avoids saving CalcFRStates unless debug mode is selected.
    """

    # Initial CLI prompt (just to check for debug mode)
    mode, is_warped, alignment, show_markers, sort_by_length = (
        SpikeRasterPlotter.cli_options()
    )

    filename = os.path.basename(npz_path)
    base_name = filename.split("_repacked.npz")[0]
    all_calc_objs = {}

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple) and len(data) == 4:
            spike_array, dataset_num, index, key = data

            dataset_num_str = str(dataset_num)
            dataset_num_clean = dataset_num_str.split("_repacked.npz")[0]
            core_dataset_name = dataset_num_clean

            print(f"Processing dataset number: {dataset_num_clean}")

            extracted_pkl = extract_valid_changepoints(
                pkl_path, spike_array, dataset_num_clean, index, key
            )

            if extracted_pkl is not None:
                try:
                    changepoints = extracted_pkl[:, 3]

                    calc = CalcFRStates(
                        spike_array=spike_array,
                        changepoints=changepoints,
                        window_length=window_length,
                        step_size=step_size,
                        compute_unwarped_spike_arrays=True,
                        compute_unwarped_firing_rates=False,
                        compute_warped_spike_arrays=True,
                        compute_warped_firing_rates=False,
                        fixed_warp_duration=fixed_warp_duration,
                    )

                    fr_unwarped, spike_unwarped, fr_warped, spike_warped = calc.run()

                    if mode == "debug":
                        calc.spike_arrays_unwarped = spike_unwarped
                        calc.spike_arrays_warped = spike_warped
                        all_calc_objs[core_dataset_name] = calc

                    else:
                        spike_data = spike_warped if is_warped else spike_unwarped
                        dataset_dir = os.path.join(
                            base_output_dir, f"dataset_{dataset_num_clean}"
                        )
                        os.makedirs(dataset_dir, exist_ok=True)

                        plotter = SpikeRasterPlotter(
                            spike_data_array=spike_data,
                            dataset_dir=dataset_dir,
                            dataset_name=core_dataset_name,
                            base_name=base_name,
                            is_warped=is_warped,
                            mode=mode,
                            alignment=alignment,
                            show_markers=show_markers,
                            sort_by_length=sort_by_length,
                            changepoints_dict=standardized_changepoints_dict,
                        )

                        plotter.plot_all()

                except Exception as e:
                    print(f"Error processing dataset {dataset_num_clean}: {e}")
            else:
                print(
                    f"No valid changepoints for dataset {dataset_num_clean}; skipping..."
                )

    if mode == "debug":
        if not all_calc_objs:
            print("No datasets were successfully loaded. Cannot enter debug mode.")
            return
        SpikeRasterPlotter.debug_extract_from_calc(all_calc_objs)

    print("All datasets processed.")


### all manner of spike train hooey ###


## also for warping etc.

###### this is very old code that I'll have to clean up at a later date #### note as of 4/18

## CHECKING VIABILITY FOR CHANGING THE NUMBER OF LATANTS IN EACH PLOT -- old buiz (gotta do major housecleaning)
# using PCA to optimize for the number of latants I'm inferring
txt_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/PCA_EXP_VAR"


def run_pca_analysis(epoch_dataframes_dict, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for dataset_name, df in epoch_dataframes_dict.items():
        # Extract unique tastes and changepoints from the dataset
        unique_tastes = df["taste"].unique().to_list()
        unique_changepoints = df["changepoint"].unique().to_list()

        # Initialize a string to store the PCA results
        pca_results_text = f"Dataset: {dataset_name}\n\n"

        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                # Filter the DataFrame for the specific taste and changepoint
                filtered_df = df.filter(
                    (pl.col("taste") == taste) & (pl.col("changepoint") == changepoint)
                )

                # Check if there is sufficient data for PCA
                if filtered_df.is_empty():
                    continue

                # Extract latent dimensions for PCA
                latent_columns = [
                    col for col in filtered_df.columns if "latent_dim_" in col
                ]
                latent_data = filtered_df.select(latent_columns).to_numpy()

                if latent_data.shape[0] < 2:
                    # Skip if there's insufficient data for PCA
                    continue

                # Run PCA
                pca = PCA()
                pca.fit(latent_data)

                # Get explained variance percentage for each PC
                explained_variance = pca.explained_variance_ratio_ * 100
                explained_variance_text = ", ".join(
                    [f"{var:.2f}%" for var in explained_variance]
                )

                # Append the results to the text
                pca_results_text += f"Taste {taste}, Epoch {changepoint}:\n"
                pca_results_text += f"Explained Variance: {explained_variance_text}\n\n"

        # Write the results to a text file
        output_file = os.path.join(output_directory, f"pca_8_rnn_{dataset_name}.txt")
        with open(output_file, "w") as file:
            file.write(pca_results_text)


def run_pca_analysis_detailed(epoch_dataframes_dict, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for dataset_name, df in epoch_dataframes_dict.items():
        # Extract unique tastes and changepoints from the dataset
        unique_tastes = df["taste"].unique().to_list()
        unique_changepoints = df["changepoint"].unique().to_list()

        # Initialize a string to store the PCA results
        pca_results_text = f"Dataset: {dataset_name}\n\n"
        pca_results_text += "Average PC Variance:\n"

        # Dictionary to store cumulative explained variance for averages
        avg_variance = {
            taste: {epoch: [] for epoch in unique_changepoints}
            for taste in unique_tastes
        }

        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                # Filter the DataFrame for the specific taste and changepoint
                filtered_df = df.filter(
                    (pl.col("taste") == taste) & (pl.col("changepoint") == changepoint)
                )

                if filtered_df.is_empty():
                    continue

                # Extract unique trials
                unique_trials = filtered_df["trial"].unique().to_list()
                trial_variance_text = ""
                trial_explained_variance = []

                for trial in unique_trials:
                    # Filter data for the specific trial
                    trial_df = filtered_df.filter(pl.col("trial") == trial)

                    # Extract latent dimensions for PCA
                    latent_columns = [
                        col for col in trial_df.columns if "latent_dim_" in col
                    ]
                    latent_data = trial_df.select(latent_columns).to_numpy()

                    if latent_data.shape[0] < 2:
                        # Skip if there's insufficient data for PCA
                        continue

                    # Run PCA
                    pca = PCA()
                    pca.fit(latent_data)

                    # Get explained variance percentage for each PC
                    explained_variance = pca.explained_variance_ratio_ * 100
                    trial_explained_variance.append(explained_variance)

                    # Calculate the number of PCs capturing >95% variance
                    cumulative_variance = 0
                    num_pcs_95 = 0
                    for idx, variance in enumerate(explained_variance, start=1):
                        cumulative_variance += variance
                        if cumulative_variance > 95:
                            num_pcs_95 = idx
                            break

                    # Add trial-level details
                    trial_variance_text += f"  Trial {trial}:\n"
                    trial_variance_text += f"    Explained Variance: {', '.join([f'{var:.2f}%' for var in explained_variance])}\n"
                    trial_variance_text += f"    >95% trial variance is captured with {num_pcs_95} Principal Components\n\n"

                # Pad trial explained variance to the same length
                if trial_explained_variance:
                    max_length = max(len(var) for var in trial_explained_variance)
                    trial_explained_variance = np.array(
                        [
                            np.pad(var, (0, max_length - len(var)), constant_values=0)
                            for var in trial_explained_variance
                        ]
                    )

                    # Calculate average explained variance across trials
                    mean_explained_variance = trial_explained_variance.mean(axis=0)
                    avg_variance[taste][changepoint].append(mean_explained_variance)

                    # Add averages to the main text
                    pca_results_text += f"Taste {taste}, Epoch {changepoint}:\n"
                    pca_results_text += f"  Average Explained Variance: {', '.join([f'{var:.2f}%' for var in mean_explained_variance])}\n"

                    cumulative_variance = 0
                    num_pcs_95 = 0
                    for idx, variance in enumerate(mean_explained_variance, start=1):
                        cumulative_variance += variance
                        if cumulative_variance > 95:
                            num_pcs_95 = idx
                            break

                    pca_results_text += f"  >95% average variance is captured with {num_pcs_95} Principal Components\n\n"
                    pca_results_text += "  Individual Trial Variances:\n"
                    pca_results_text += trial_variance_text

        # Write the results to a text file
        output_file = os.path.join(output_directory, f"pca_rnn_{dataset_name}.txt")
        with open(output_file, "w") as file:
            file.write(pca_results_text)


# OLD BUSINESS BELOW
#### DOING HEATMAPS BUT WITH RNN LATENTS AS WE WANT TO BE COMPARING THESE TWO ITEMS###

rnn_plt_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/rnn_latant_first_8_comp"
alt_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/rnn_latant_hmap"


def plot_inferred_data_heatmaps(data_dict, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each DataFrame in the dictionary
    for df_name, df in data_dict.items():
        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique trials from the DataFrame
        unique_trials = df["trial"].unique().to_list()

        # Iterate over each trial and plot its heatmap
        for trial in unique_trials:
            # Filter the DataFrame for the current trial
            trial_df = df.filter(pl.col("trial") == trial)

            # Extract latent values as a numpy array
            latent_values = np.stack(trial_df["latent_values"].to_list(), axis=0)

            # Plot the heatmap with correct axes and ticks
            plt.figure(figsize=(10, 5))
            plt.imshow(
                latent_values, aspect="auto", cmap="coolwarm", interpolation="none"
            )
            plt.colorbar(label="Latent Value")
            plt.title(f"Latent Heatmap for {df_name}, Trial {trial}")
            plt.xlabel("Time Bins")
            plt.ylabel("Latent Dimensions")
            # Set x-ticks every 25 time bins
            plt.xticks(
                ticks=np.arange(0, latent_values.shape[1], 25),
                labels=np.arange(0, latent_values.shape[1], 25),
            )
            plt.yticks(
                ticks=np.arange(latent_values.shape[0]),
                labels=[f"Dim {i + 1}" for i in range(latent_values.shape[0])],
            )

            # Save the heatmap
            plt.savefig(
                os.path.join(df_output_dir, f"trial_{trial}_latent_heatmap.png")
            )
            print(f"RNN Heatmap results saved to {output_dir}")

            plt.close()


# NOW SEM - Unwarped data for the RNN:


def plot_averaged_epochs_with_sem(epoch_dataframes_dict, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for df_name, df in epoch_dataframes_dict.items():
        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes and changepoints
        unique_tastes = df["taste"].unique().to_list()
        unique_changepoints = df["changepoint"].unique().to_list()

        # Initialize figure number for saving
        fig_num = 1

        # Create a figure for each dataset with a 4x4 grid layout
        fig, axs = plt.subplots(4, 4, figsize=(20, 15))
        fig.suptitle(f"Averaged Trials with SEM - {df_name}", fontsize=18)

        for taste_idx, taste in enumerate(unique_tastes):
            for epoch_idx, changepoint in enumerate(unique_changepoints):
                # Filter for the current taste and changepoint
                taste_epoch_df = df.filter(
                    (pl.col("taste") == taste) & (pl.col("changepoint") == changepoint)
                )

                # Extract unique trials for this taste and changepoint
                unique_trials = taste_epoch_df["trial"].unique().to_list()

                # Find the longest epoch in terms of time bins
                max_time_length = max(
                    [
                        len(taste_epoch_df.filter(pl.col("trial") == trial)["time"])
                        for trial in unique_trials
                    ]
                )

                # Collect latent dimension data for all trials, padding with NaNs
                padded_data = []
                for trial in unique_trials:
                    trial_df = taste_epoch_df.filter(pl.col("trial") == trial)
                    latent_values = np.stack(
                        [trial_df[f"latent_dim_{j}"].to_numpy() for j in range(8)],
                        axis=0,
                    )
                    # Pad with NaNs to match the max_time_length
                    padded_latent_values = np.full((8, max_time_length), np.nan)
                    padded_latent_values[:, : latent_values.shape[1]] = latent_values
                    padded_data.append(padded_latent_values)

                # Convert to a 3D array (trials x latent_dim x time) and compute nanmean and SEM
                padded_data_array = np.stack(padded_data, axis=0)
                mean_values = np.nanmean(padded_data_array, axis=0)
                sem_values = np.nanstd(padded_data_array, axis=0) / np.sqrt(
                    np.sum(~np.isnan(padded_data_array), axis=0)
                )

                # Plot the mean line and SEM for each latent dimension
                ax = axs[taste_idx, epoch_idx]
                time_values = np.arange(max_time_length) * BIN_SIZE_MS + START_TIME_MS
                for j in range(8):  # Plot each latent dimension
                    ax.plot(time_values, mean_values[j], label=f"Latent Dim {j+1}")
                    ax.fill_between(
                        time_values,
                        mean_values[j] - sem_values[j],
                        mean_values[j] + sem_values[j],
                        alpha=0.2,
                    )

                # Set plot title and labels
                ax.set_title(f"Taste {taste}, Epoch {changepoint}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Latent Value")

                if taste_idx == 0 and epoch_idx == 0:  # Add legend only once
                    ax.legend(loc="upper right", fontsize="small")

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        output_file = os.path.join(df_output_dir, f"averaged_plot_fig_{fig_num}.png")
        plt.savefig(output_file)
        print(f"Saved averaged plot figure: {output_file}")

        plt.close(fig)
        fig_num += 1


###########


def plot_averaged_pca_trajectories_thresholded_SEM(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
    def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
        # Filter the DataFrame for the specific dataset, taste, and changepoint
        filtered_df = pca_results_df.filter(
            (pl.col("dataset") == dataset)
            & (pl.col("taste") == taste)
            & (pl.col("changepoint") == changepoint)
        )

        # Assuming there's only one matching row, extract the pca_result and trial_lengths
        row = filtered_df.row(0, named=True)
        pca_result = np.array(row["pca_result"])
        trial_lengths = row["trial_lengths"]

        # Unconcatenate the PCA results using the trial lengths
        start_idx = 0
        split_pca_result = []
        for length in trial_lengths:
            trial_pca_result = pca_result[start_idx : start_idx + length, :]
            split_pca_result.append(trial_pca_result)
            start_idx += length
        return split_pca_result

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()
    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes
    num_pc_avg = 1

    # Create one folder named for the number of principal components being averaged
    output_dir = os.path.join(pca_output_dir, f"averaged_{num_pc_avg}_pc_line_SEM")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"PCA Trajectories, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)
                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("Time (ms)")
                    if cp_idx == 0:
                        ax.set_ylabel(f"PC {num_pc_avg} Values")

                    # Averaging the specified number of principal components
                    averaged_pca_results = [
                        np.mean(trial_pca[:, :num_pc_avg], axis=1)
                        for trial_pca in split_pca_result
                    ]

                    trial_lengths = []
                    trial_data = []

                    for trial_idx, averaged_pca in enumerate(averaged_pca_results):
                        num_time_bins = averaged_pca.shape[0]
                        trial_lengths.append(num_time_bins)
                        trial_data.append(averaged_pca)

                    # Calculate the mean and SEM at each time point across all trials
                    max_time_bins = max(trial_lengths)
                    all_data = np.full((num_trials, max_time_bins), np.nan)
                    for i, data in enumerate(trial_data):
                        all_data[i, : len(data)] = data

                    mean_data = np.nanmean(all_data, axis=0)
                    sem_data = np.nanstd(all_data, axis=0) / np.sqrt(
                        np.sum(~np.isnan(all_data), axis=0)
                    )
                    mean_time_bins = np.arange(max_time_bins) * step_size
                    ax.plot(
                        mean_time_bins,
                        mean_data,
                        color="black",
                        linewidth=1,
                        label="Trial Average",
                    )
                    ax.fill_between(
                        mean_time_bins,
                        mean_data - sem_data,
                        mean_data + sem_data,
                        color="gray",
                        alpha=0.5,
                        label="SEM",
                    )
                    ax.legend()

                    avg_duration = np.mean(trial_lengths) * step_size

                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                    )

                except Exception as e:
                    print(
                        f"Could not plot PCA trajectories for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )
        plt.tight_layout()
        neuron_fig_path = os.path.join(output_dir, f"{dataset}_pca_trajectories.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


# loop for going through the thresholded neurons; doing the whole PCA analysis pipeline on them too
# all these plots are now in a separate file
base_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/pca_thresholded_w_poststim"
os.makedirs(base_dir, exist_ok=True)
# Iterate over each DataFrame in sig_nrns_dict
for key, df in sig_nrns_dict.items():
    # Create a subdirectory for the current DataFrame
    sub_dir = os.path.join(base_dir, key)
    os.makedirs(sub_dir, exist_ok=True)
    # Apply the PCA function to the current DataFrame
    pca_results_df = whitened_pca_all_nrns_sep_taste(df, sub_dir)
    plot_averaged_pca_trajectories_thresholded_SEM(
        pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size
    )
# plot_averaged_pca_heatmaps_all_taste(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_pca_scatter(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_pca_scatter_3d(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
# distance_df = calculate_pca_distances(pca_results_df)
# plot_euclidean_distances_bar(distance_df, sub_dir, modified_tastes, epoch_labels, step_size)
# plot_explained_variance_combined_by_taste(pca_results_df, sub_dir, modified_tastes, epoch_labels)


# actual generator that runs through ALL .npz files and datasets; just kinda doing everything
# creating a dataframe for ALL significant neurons

# all under generator that I'm increasingly not using anymore in favor of newer, faster code
for data in extract_from_npz(npz_path):
    if isinstance(data, tuple):
        spike_array, dataset_num, index, key = data
        print(f"Dataset number: {dataset_num}")
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(
            pkl_path, [(spike_array, dataset_num, index, key)]
        )
        if extracted_pkl is not None:
            # Process the extracted_pkl data as needed
            print("Processed .pkl data:")
        else:
            print("Failed to process .pkl data")
        try:
            changepoints = extracted_pkl[
                :, 3
            ]  # Giving 4 changepoint arrays, 1 for each taste-- for each bit of data
            window_length = 250  # make 25 for just straight binning, no sliding window lol (might work); for sliding make 250
            step_size = 25  # for sliding make 25
            alpha = 0.05  # for the t-test being run

            # start_bin = 000/25 # not needed as specified inside function now
            # end_bin = 7000/25 # not needed as specified inside function now
            # Calculate firing rates with changepoints
            state_firing_rates_all_trials, state_spike_arrays_all_trials = (
                calc_fr_states(spike_array, changepoints, window_length, step_size)
            )
            # Process firing rate data as needed
            print(
                f"Firing rate with changepoints calculated for dataset {dataset_num}:"
            )
            all_tastes_averaged, all_interpolated_states = (
                interpolate_and_average_all_tastes(state_firing_rates_all_trials)
            )

            ### plotting functions ###

            ## Warped Plots ##
            # plt_inter_cpfr_avg_nrn_line(all_interpolated_states, modified_tastes, dataset_num)
            # plt_inter_cpfr_avg_tr_line(all_interpolated_states, modified_tastes, dataset_num)
            # plt_single_neuron_firing_warped_line(all_interpolated_states, dataset_num, modified_tastes, output_dir_warped)

            ## Unwarped Plots ##
            # plt_single_neuron_firing_unwarped_line(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)
        # plt_single_neuron_firing_unwarped_heatmap(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)

        ## Changepoint plots ##
        # plt_changepoints_scat(changepoints, modified_tastes, dataset_num)
        # plt_changepoint_hist(changepoints, modified_tastes, dataset_num)

        ## Unwarped plots with significance dictated via t-test or 1 hz firing threshold## -- old
        # thresh_firing_rates_unwarped = thresh_ttest_unwarped_hz(state_firing_rates_all_trials, alpha) # significance via ttest
        # thresh_firing_rates_unwarped = thresh_min_fr_unwarped_hz(state_firing_rates_all_trials, threshold_hz) # signifigance via 1 hz firing rate
        # plt_single_neuron_firing_unwarped_heatmap_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)
        # plt_single_neuron_firing_unwarped_line_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)

        # appending individual significant neurons to one big happy dataset
        except TypeError as e:
            if str(e) == "'float' object is not iterable":
                print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
            else:
                raise
                # error handling for one dataset whose data type seems to be irritating
    else:
        spike_array, index, key = data
        dataset_num = "Unknown Dataset"  # as in for some reason we can't get a number
        print(f"Dataset number: {dataset_num}")
