#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:07:31 2024

At this point, this script is just kinda where I'm building all sorts of stuff...

@author: vincentcalia-bogan
"""
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

## VINCENT MODULE IMPORTS aka VIMPORTS## Init the environment 
# TODO: Fix these file imports later so not as hard coded, but that's a later project # 
# MODULE DIRECTORIES 
project_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work' 
submodule_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions'
# Check if paths already exist in sys.path
sys.path.append(submodule_dir)
sys.path.append(project_dir)
# Importing Vinmodules 
from spike_train_to_npz import find_h5_files, extract_to_npz
from extract_npz import extract_from_npz
from unpkl_generator import unpickle_changepoints
from unpkl_generator import extract_valid_changepoints
# from interpolation_xr import interpolate_and_average_all_tastes, interpolate_firing_rates # this is actually a deprecated func now
from calc_firing_rate_with_states import calc_fr_states # a new class 
# new firing rate class with related calls: 
from calc_fr_class_war_unwar import CalcFRStates
from generate_parquet_sig_all import sig_neurons_hz, sig_neurons_ttest, consolidate_all_neuron_data
from generate_parquet_sig_all import sig_neurons_war_hz, sig_neurons_war_ttest, consolidate_all_neuron_war_data
from read_parquets import read_parquet_files_into_dict, all_nrns_to_df
from serialize_overlap import serialized_neuron_df, create_overlap_dataframes
from find_extract_info import find_copy_h5info, process_info_files, modify_tastes

## FILE PATHS FOR DATA WRANGLING -- OTHERWISE WILL HAVE TO SPECIFY EVERY TIME ## 
file_path = '/Volumes/T7 Shield/spikesorting'
spike_trains_path = '/spike_trains'
npz_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train npz data'
pkl_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/pkl files/'
info_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Spike train info data'
# dir paths for plotting unwarped and warped data
output_dir_unwarped = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/uw-s-tr-th-1hz-line'
output_dir_warped = ''
sig_ttest_parquet_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet'
sig_war_parquet_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped'
sig_hz_parquet_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet'
all_nrns_parquet_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet'
all_war_nrns_parquet_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped'

sig_ttest_parquet_path_w = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet_warped'
all_nrns_parquet_path_w = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet_warped'

# Notes on RNN: 
rnn_inf_fr_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/newRNN/infer_fr_parquet'

# refrence name replacements 
taste_replacements = {
    'nacl': 'NaCl',
    'suc': 'Sucrose', 
    'ca' : 'Citric Acid',
    'qhcl': 'Quinine'
}
epoch_labels = ('Identification', 'Palatability', 'Decision', '2000 ms Post-Stimulus')
## NECESARY PARAMETERS ## 
window_length = 250
step_size = 25
alpha = 0.05 # alpha for any null hypothesis tests 
threshold_hz = 2.0 # threshold firing rate freuqency for various funcs
# this is a pretty janky work-around; fix this later plz 

################################################## INIT INTERMEDIATE FILES ####
for data in extract_from_npz(npz_path):
    if isinstance(data, tuple): 
        spike_array, dataset_num, index, key = data
        print(f'Dataset number: {dataset_num}')
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
npz_files_exist = any(file.endswith('.npz') for file in os.listdir(npz_path)) # save time with a check 
if npz_files_exist:
    print(".npz files containing spike trains already exist in npz_path. Skipping extraction from h5 files.")
    # Perform certain actions when npz files already exist (pass certain functions)
else:
    print("No .npz files found in npz_path. Running functions to generate npz files.")
    # Run functions to generate npz files
    h5_files = find_h5_files(file_path) # pulling h5 file paths
    save_data = extract_to_npz(h5_files, file_path, spike_trains_path, npz_path) # saving npz files to save location 
# checking for info files 
info_files_exist = any(file.endswith('.info') for file in os.listdir(info_path))
if info_files_exist: 
    print(".info files already exist in info_path; skipping re-copying them")
else: 
    print("No .info files found in info_path. Extracting info_files")
    info_files = find_copy_h5info(file_path, info_path)
    print(f"Found and copied {len(info_files)} .info files.")
# checking for sig_parquet files -- unwarped
sig_parquet_exist = any(file.endswith('.parquet') for file in os.listdir(sig_ttest_parquet_path))
if sig_parquet_exist: 
    print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
else: 
    print("No .parquet files found in sig_parquet_path. Extracting info_files")
    sig_ttest = sig_neurons_ttest(npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size)
    sig_2hz = sig_neurons_hz(npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size)
    #print(f"Found and copied {len(sig_ttest)} .parquet files.")
    #print(f"Found and copied {len(sig_2hz)} .parquet files.")
# checking for parquet file with all neuron data 
consolidated_parquet_exist = any(file.endswith('.parquet') for file in os.listdir(all_nrns_parquet_path))
if consolidated_parquet_exist: 
    print(".parquet files already exist in all_nrns_parquet_path; skipping re-copying them")
else: 
    print("No .info files found in all_nrns_parquet_path. Extracting info_files")
    consolidated_data = consolidate_all_neuron_data(npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size)
    #print(f"Found and copied {len(consolidated_data)} .parquet files.")
# unwarped data
# reading data back from parquets 
sig_nrns_dict = read_parquet_files_into_dict(sig_ttest_parquet_path)
# dataframe of all neuron data
all_data_df = all_nrns_to_df(all_nrns_parquet_path)
serialized_nrns_df = serialized_neuron_df(npz_path, pkl_path) # just counts serialized nrns
sig_overlap = create_overlap_dataframes(serialized_nrns_df, sig_nrns_dict) # overlap of nrns 
sig_parquet_exist_w = any(file.endswith('.parquet') for file in os.listdir(sig_ttest_parquet_path_w))
if sig_parquet_exist_w: 
    print(".parquet files already exist in sig_parquet_path; skipping re-copying them")
else: 
    print("No .parquet files found in sig_parquet_path. Extracting info_files")
    sig_ttest_w= sig_neurons_war_ttest(npz_path, pkl_path, sig_ttest_parquet_path_w, alpha, window_length, step_size)
    sig_2hz_w = sig_neurons_war_hz(npz_path, pkl_path, sig_ttest_parquet_path_w, threshold_hz, window_length, step_size)
    #print(f"Found and copied {len(sig_ttest)} .parquet files.")
    #print(f"Found and copied {len(sig_2hz)} .parquet files.")
# checking for parquet file with all neuron data 
consolidated_parquet_exist_w = any(file.endswith('.parquet') for file in os.listdir(all_nrns_parquet_path_w))
if consolidated_parquet_exist_w: 
    print(".parquet files already exist in all_nrns_parquet_path; skipping re-copying them")
else: 
    print("No .info files found in all_nrns_parquet_path. Extracting info_files")
    consolidated_data_w = consolidate_all_neuron_war_data(npz_path, pkl_path, all_nrns_parquet_path_w, window_length, step_size)
    #print(f"Found and copied {len(consolidated_data)} .parquet files.")
# reading data back from parquets 
sig_nrns_dict_w = read_parquet_files_into_dict(sig_war_parquet_path)
# dataframe of all neuron data
all_data_w_df = all_nrns_to_df(all_war_nrns_parquet_path)
serialized_w_nrns_df = serialized_neuron_df(npz_path, pkl_path) # just counts serialized nrns
sig_w_overlap = create_overlap_dataframes(serialized_w_nrns_df, sig_nrns_dict_w) # overlap of nrns 

# note: Go over some of these (likely now antequated) dependencies -- as I may or may not need some of this data...

latent_path = '/Volumes/T7 Shield/RNN CODE FROM BIG PC/output/latent_parquet'
latent_dict_rnn = read_parquet_files_into_dict(latent_path)
# THIS CLASS now finally works and I'm adding shit to it as we speak. Make life easier. 
from RNNLatentprocessing import RNNLatentProcessor
# init processor 
processor = RNNLatentProcessor(
    parquet_dir='/Volumes/T7 Shield/octRNN/parquert',
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements=taste_replacements,
    save_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/RNN_PROCESSING_PARQUETS',  # <-- NEW
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500
)
# run full RNN analysis pipeline 
epoch_dataframes_dict, robust_pca_95, robust_pca_full, first_derivs, second_derivs = processor.full_pipeline(
    variance_threshold=95.0,
    compute_first_derivative=True,
    compute_second_derivative=True,
    derivative_source="threshold",
    return_derivatives=True,
    save_outputs=True  # <-- triggers automatic parquet saving!
)
standardized_changepoints_dict = {
    key.replace('dataset_', '').split('_repacked.npz')[0]: value
    for key, value in processor.changepoints_dict.items()
}

# Run everything here to reload stuff 
#%% cell dividing line 

# firing rate stuff-- from april as well 
from class_plot_fr_line_heat_april import FiringRatePlotter 
# function that wraps this class for it to be run later. 
def process_and_plot_fr_datasets(
    npz_path,
    pkl_path,
    base_output_dir,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000
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
                pkl_path,
                spike_array,
                dataset_num_clean,
                index,
                key
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
                        fixed_warp_duration=fixed_warp_duration
                    )

                    fr_unwarped, _, fr_warped, _ = calc.run()
                    fr_data = fr_warped if is_warped else fr_unwarped

                    dataset_dir = os.path.join(base_output_dir, f"dataset_{dataset_num_clean}")
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
                        changepoints_dict=standardized_changepoints_dict
                    )

                    plotter.plot_all()


                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing dataset {core_dataset_name}: {e}")
            else:
                print(f"No valid changepoints for dataset {core_dataset_name}; skipping...")

    print("All datasets processed.")
# the above is a wrapper that explicitly plots firing rate -- below is somethign that just returns firing rate. 
# 6/10 TODO: add the new plots abu was talking about to the plotting class

# new 4/14/25 working with warped data now too 
# SEM single-neuron population funcs that are to be run every so often
# transferred to own file-- see april_single_nrn_sem_war_unwar_plot.py

# func that runs the calcFRstaes stuff for further use: 
def process_and_return_fr_datasets(
    npz_path,
    pkl_path,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000
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
                pkl_path,
                spike_array,
                dataset_num_clean,
                index,
                key
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
                        fixed_warp_duration=fixed_warp_duration
                    )

                    fr_unwarped, _, fr_warped, _ = calc.run()

                    results[core_dataset_name] = {
                        'fr_unwarped': fr_unwarped,
                        'fr_warped': fr_warped
                    }

                except Exception as e:
                    print(f"Error processing {core_dataset_name}: {e}")
                    continue

    return results
fr_dict = process_and_return_fr_datasets(npz_path, pkl_path)
###### New RNN Latent stuff ###### 



# plot space for this hooey-- separated heat maps: 
rnn_taste_output_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir'
def plot_inferred_data_heatmaps(epoch_dataframes_dict, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each DataFrame in the dictionary
    for df_name, df in epoch_dataframes_dict.items():
        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes and changepoints from the DataFrame
        unique_tastes = df['taste'].unique().to_list()
        unique_changepoints = df['changepoint'].unique().to_list()

        # Loop over each taste and changepoint
        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                # Filter for the current taste and changepoint
                taste_changepoint_df = df.filter((pl.col('taste') == taste) & (pl.col('changepoint') == changepoint))
                
                # Extract unique trials for the current taste and changepoint
                unique_trials = taste_changepoint_df['trial'].unique().to_list()
                
                # Initialize plot counter for tracking figures
                plot_counter = 0
                fig_num = 1

                # Loop over trials in chunks of 12 for the 4x3 grid
                for trial_chunk in range(0, len(unique_trials), 12):
                    # Create a new figure with a 4x3 grid layout
                    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
                    fig.suptitle(f"Taste {taste}, Changepoint {changepoint} - {df_name}", fontsize=16)

                    for i, trial in enumerate(unique_trials[trial_chunk:trial_chunk + 12]):
                        # Filter the DataFrame for the current trial
                        trial_df = taste_changepoint_df.filter(pl.col('trial') == trial)
                        
                        # Extract latent values as a numpy array
                        latent_values = np.stack([trial_df[f'latent_dim_{j}'].to_numpy() for j in range(8)], axis=0)
                        
                        # Get start and end times for the changepoint segment and calculate duration
                        start_time = trial_df['time'].min()
                        end_time = trial_df['time'].max()
                        duration = end_time - start_time
                        
                        # Plot on the appropriate subplot
                        ax = axs[i // 3, i % 3]
                        heatmap = ax.imshow(latent_values, aspect='auto', cmap='coolwarm', interpolation='none')
                        ax.set_title(f"Trial {trial} | Start: {start_time}, End: {end_time}, Duration: {duration}")
                        ax.set_xlabel("Time(ms)")
                        ax.set_ylabel("Latent Dimensions")
                        # Calculate time in ms for each time bin, setting ticks every 250 ms
                        time_bins_ms = np.arange(start_time, end_time + 1, BIN_SIZE_MS)
                        tick_locations = np.arange(0, len(time_bins_ms), 250 // BIN_SIZE_MS)
                        tick_labels = time_bins_ms[::250 // BIN_SIZE_MS]
                        
                        # Set x-ticks and labels
                        ax.set_xticks(tick_locations)
                        ax.set_xticklabels(tick_labels)
                        
                        # Set y-ticks for latent dimensions
                        ax.set_yticks(np.arange(latent_values.shape[0]))
                        ax.set_yticklabels([f"Dim {j + 1}" for j in range(latent_values.shape[0])])

                    # Adjust layout and add colorbar
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
                    fig.colorbar(heatmap, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)

                    # Save the figure
                    output_file = os.path.join(df_output_dir, f"hm_taste_{taste}_changepoint_{changepoint}_fig_{fig_num}.png")
                    plt.savefig(output_file)
                    print(f"Saved heatmap figure: {output_file}")

                    plt.close(fig)
                    fig_num += 1

# line plots, separated:
line_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/line_plots_epoch_sep'

# Line plots that are separated by dark lines at CP's -- not separate plots themselves 

# dir specifically for these plots 
# this is the plotting that we like 
rnn_taste_output_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/line_plots_don'

def plot_all_rnns_with_changepoints(epoch_dataframes_dict, standardized_changepoints_dict, output_dir, start_time=None, end_time=None):
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
        core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]
        
        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df['taste'].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_df = df.filter(pl.col('taste') == taste)
            unique_trials = taste_df['trial'].unique().to_list()

            # Plot 4 trials per figure
            for trial_chunk in range(0, len(unique_trials), 4):
                fig, axs = plt.subplots(4, 1, figsize=(15, 15))
                fig.suptitle(f"RNN Plots for taste: {modified_tastes[taste_idx]} - {df_name}", fontsize=16)

                for i, trial in enumerate(unique_trials[trial_chunk:trial_chunk + 4]):
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time_values = trial_df['time'].to_numpy()
                    latent_values = [trial_df[f'latent_dim_{j}'].to_numpy() for j in range(8)]

                    # Apply time filtering
                    if start_time is not None:
                        time_mask = time_values >= start_time
                    else:
                        time_mask = np.ones_like(time_values, dtype=bool)

                    if end_time is not None:
                        time_mask &= time_values <= end_time

                    time_values = time_values[time_mask]
                    latent_values = [lv[time_mask] for lv in latent_values]

                    ax = axs[i] if len(unique_trials) > 1 else axs  # Handle single subplot case

                    # Plot each latent dimension as a line
                    for j, latent_series in enumerate(latent_values):
                        ax.plot(time_values, latent_series, label=f'Latent Dim {j+1}')

                    # Plot changepoints within the filtered range
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(changepoint_time, color='black', linestyle='--', linewidth=2, label='Changepoint')

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

                    # Set plot title and labels
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Latent Value")

                # Add a shared legend outside the subplots
                fig.legend([f'Latent Dim {j+1}' for j in range(8)] + ['Changepoint', 'Stimulus Delivery (dotted)'],
                           loc="lower center", ncol=5, fontsize='small', frameon=False)
                
                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(df_output_dir, f"rnn_plots_taste_{taste}_chunk_{trial_chunk // 4 + 1}.png")
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")

                plt.close(fig)
                
                
# APRIL NEW 
# the above but reborn for firing rate PCA plots: 
def plot_pca_of_firing_rates(fr_unwarped: xr.DataArray, 
                              changepoints_dict: dict,
                              dataset_name: str, 
                              output_dir: str,
                              modified_tastes: list,
                              start_time: int = int,
                              end_time: int = int):
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
                segment_data = fr_data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx).values  # shape: (neurons, time)
                if np.isnan(segment_data).all():
                    continue
                segment_data = segment_data[:, ~np.isnan(segment_data[0])]  # Drop NaN-padding
                trial_seg_stack.append(segment_data)
        
            if not trial_seg_stack:
                continue
        
            full_trial_mat = np.concatenate(trial_seg_stack, axis=1)  # shape: (neurons, total_time)
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
            trial_pcs.append(pcs[cursor:cursor + length].T)  # Each trial: shape (8, time)
            cursor += length

        # Plotting 4 trials per figure
        for chunk_start in range(0, len(trial_pcs), 4):
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))
            fig.suptitle(f"PCA on FR - {taste_name} - {dataset_name}", fontsize=16)

            for i, (trial_idx, trial_pc) in enumerate(zip(trials[chunk_start:chunk_start+4], trial_pcs[chunk_start:chunk_start+4])):
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
                    ax.plot(time_vals, trial_pc[dim], label=f'PC {dim+1}')

                # Plot changepoints
                if taste_idx < len(changepoints) and trial_idx < len(changepoints[taste_idx]):
                    for cp in changepoints[taste_idx][trial_idx]:
                        if start_time is None or (start_time <= cp <= end_time):
                            ax.axvline(cp, color='black', linestyle='--', linewidth=2)

                # Stimulus at 2000ms
                if (start_time is None or start_time <= 2000) and (end_time is None or 2000 <= end_time):
                    ax.axvline(2000, color='black', linestyle=':', linewidth=2)

                ax.set_title(f"Trial {trial_idx}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("PCA Value")

            # Add legend
            fig.legend([f'PC {i+1}' for i in range(8)] + ['Changepoint', 'Stimulus'],
                       loc='lower center', ncol=5, fontsize='small', frameon=False)

            # Save and close
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fname = f"pca_firingrates_taste_{taste_idx}_chunk_{chunk_start // 4 + 1}.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close(fig)
            # Save explained variance info for this taste
            var_percent = pca.explained_variance_ratio_ * 100
            cumulative_var = np.cumsum(var_percent)

           # var_percent = pca.explained_variance_ratio_[:] * 100
            var_lines = [f"PC{i+1}: {vp:.2f}%" for i, vp in enumerate(var_percent)]
            var_text = "\n".join(var_lines)
            
            ev_file = os.path.join(output_dir, f"explained_variance_taste_{taste_idx}_{taste_name.replace(' ', '_')}.txt")
            with open(ev_file, 'w') as f:
                f.write(f"Explained Variance for Taste {taste_name} (Dataset: {dataset_name})\n")
                f.write(f"{'-'*60}\n")
                f.write(var_text)
                f.write("\n")
                # Explained variance values
          #  cumulative_var = np.cumsum(pca.explained_variance_ratio_[:] * 100)
            
            fig, ax = plt.subplots(figsize=(5, 7))
            x = np.arange(1, len(var_percent) + 1)
            
            # Bar plot for individual PC explained variance
            bars = ax.bar(x, var_percent, color='blue', label='Explained Variance (per PC)')
            
            # Line plot for cumulative explained variance
            ax.plot(x, cumulative_var, color='red', marker='o', label='Cumulative Variance')
            
            # Labels and styling
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance (%)')
            ax.set_title(f'Explained Variance for {taste_name} - {dataset_name}')
            ax.set_xticks(x)
            ax.set_ylim(0, max(100, cumulative_var[-1] + 5))
            ax.legend(loc='upper left')
            
            # Save plot
            ev_plot_path = os.path.join(output_dir, f"explained_variance_plot_taste_{taste_idx}_{taste_name.replace(' ', '_')}.png")
            plt.tight_layout()
            plt.savefig(ev_plot_path)
            plt.close()
            print(f"Saved explained variance plot: {ev_plot_path}")
            
            print(f"Saved explained variance to: {ev_file}")
            print(f"Saved PCA plot figure: {fname}")

# running this mess: THIS IS A VERY IMPORTANT FUNCTION OF MINE THAT HAS TO STAY FOR NOW fr tho
for dataset_name, fr_data in fr_dict.items():
    fr_unwarped = fr_data['fr_unwarped']

    plot_pca_of_firing_rates(
        fr_unwarped=fr_unwarped,
        changepoints_dict=standardized_changepoints_dict,
        dataset_name=dataset_name,
        output_dir=os.path.join('/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_ON_RNN_FULL_TRIAL/PCA_NONWHITENED_ON_FR_DATA_FULL_FRAME', dataset_name),
        modified_tastes=modified_tastes,
        start_time=1500,
        end_time=4500
    )

## INTRA-STATE PLOTS NOW 
from RNN_Intrastate_war_unwar_plotting_class import RNNWarpingIntraAroundCP # own file now

pipeline = RNNWarpingIntraAroundCP(
    standardized_changepoints_dict=standardized_changepoints_dict,
    output_dir='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/INTRA_PCA_RNN_WINDOW_REF',
    mode='windowed',             # 'warped', 'unwarped', or 'windowed'
    warp_length=1000,          # used if mode='warped'
    window_size=300,           # half-window size if mode='windowed'
    start_time=1500,           # global data start (ms)
    end_time=4500,             # global data end (ms)
    plot_average=True         # whether to overlay the mean trajectory
)
pipeline.run(robust_pca_95)


### MOOOVE all of this to its own class eventually (post-thesis defense!)

ind_latant_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir/lines_ind_latant'
def plot_individual_latents(epoch_dataframes_dict, standardized_changepoints_dict, output_dir, start_time=None, end_time=None):
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
        core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]
        
        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df['taste'].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_df = df.filter(pl.col('taste') == taste)
            unique_trials = taste_df['trial'].unique().to_list()

            for trial in unique_trials:
                trial_df = taste_df.filter(pl.col('trial') == trial)
                time_values = trial_df['time'].to_numpy()
                latent_values = [trial_df[f'latent_dim_{j}'].to_numpy() for j in range(8)]

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
                fig.suptitle(f"Latent Vectors for Trial {trial}, Taste {modified_tastes[taste_idx]} - {df_name}", fontsize=16)

                for i, latent_series in enumerate(latent_values):
                    ax = axs[i]
                    ax.plot(time_values, latent_series, label=f'Latent Dim {i+1}')
                    
                    # Plot changepoints for this trial
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(changepoint_time, color='black', linestyle='--', linewidth=2, label='Changepoint')
                    
                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

                    ax.set_title(f"Latent Dim {i+1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Value")
                    ax.legend(loc="upper right", fontsize='small', frameon=False)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                output_file = os.path.join(df_output_dir, f"latents_taste_{taste}_trial_{trial}.png")
                plt.savefig(output_file)
                print(f"Saved latent vector plot figure: {output_file}")
                plt.close(fig)



def get_data_columns(df: pl.DataFrame):
   # Return PC_x or latent_dim_x columns from a Polars DataFrame.
   # Useful for looping across dimensions.
    return [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]
# Plotting these data: 
diff_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/DERIV_PCA_RNN/SECOND_DERIV'
output_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/CP_window_plt/PCA_RNN_t_robust_full'
def plot_pca_rnn_with_changepoints(pca_lat_dict, standardized_changepoints_dict, modified_tastes, output_dir, start_time=None, end_time=None):
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
        core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]
        
        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        # Get the changepoint data for the dataset
        changepoints = standardized_changepoints_dict[core_dataset_name]

        # Create a subdirectory for the current DataFrame
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        # Extract unique tastes
        unique_tastes = df['taste'].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            # Filter for the current taste
            taste_name = modified_tastes[taste_idx] if taste_idx < len(modified_tastes) else f"Unknown_Taste_{taste_idx}"
            taste_df = df.filter(pl.col('taste') == taste)
            unique_trials = taste_df['trial'].unique().to_list()

            # Plot 4 trials per figure
            for trial_chunk in range(0, len(unique_trials), 4):
                fig, axs = plt.subplots(4, 1, figsize=(15, 15))
                fig.suptitle(f"Second_deriv_PCA_trial_RNN Plots for Taste {taste_name} - {df_name}", fontsize=16)

                for i, trial in enumerate(unique_trials[trial_chunk:trial_chunk + 4]):
                    trial_df = taste_df.filter(pl.col('trial') == trial)
                    time_values = trial_df['time'].to_numpy()
                    
                    # Dynamically detect PC columns
                    pc_columns = [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]
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

                    ax = axs[i] if len(unique_trials) > 1 else axs  # Handle single subplot case

                    # Plot each principal component as a line
                    for j, pc_series in enumerate(pc_values):
                        ax.plot(time_values, pc_series, label=f'{pc_columns[j]}')

                    # Plot changepoints within the filtered range
                    trial_changepoints = changepoints[taste_idx][trial]
                    for changepoint_time in trial_changepoints:
                        if start_time <= changepoint_time <= end_time:
                            ax.axvline(changepoint_time, color='black', linestyle='--', linewidth=2, label='Changepoint')

                    # Add stimulus delivery line
                    if start_time <= 2000 <= end_time:
                        ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

                    # Set plot title and labels
                    ax.set_title(f"Trial {trial}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Principal Component Value")

                # Add a shared legend outside the subplots
                fig.legend([f'{col}' for col in pc_columns] + ['Changepoint'] + [ 'Stimulus Delivery (dotted)'],
                           loc="lower center", ncol=5, fontsize='small', frameon=False)
                
                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_file = os.path.join(df_output_dir, f"rnn_plots_taste_{taste}_chunk_{trial_chunk // 4 + 1}.png")
                plt.savefig(output_file)
                print(f"Saved RNN plot figure: {output_file}")

                plt.close(fig)

## 2/16-- now figs where latents are plotted on one subplot (several subplots) -- this is going to have some issues for sure

pca_rnn_lat_plots_path_unal = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS'
def plot_pca_rnn_all_trials_per_transition(pca_lat_dict, standardized_changepoints_dict, output_dir, modified_tastes):
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
        core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]
        
        if core_dataset_name not in standardized_changepoints_dict:
            print(f"Changepoints not found for {core_dataset_name}. Skipping...")
            continue

        changepoints = standardized_changepoints_dict[core_dataset_name]
        df_output_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_output_dir, exist_ok=True)

        unique_tastes = df['taste'].unique().to_list()

        for taste_idx, taste in enumerate(unique_tastes):
            taste_name = modified_tastes[taste_idx] if taste_idx < len(modified_tastes) else f"Unknown_Taste_{taste_idx}"
            
            taste_df = df.filter(pl.col('taste') == taste)
            unique_trials = taste_df['trial'].unique().to_list()
            num_transitions = len(changepoints[taste_idx][unique_trials[0]]) - 1  # Number of transitions between changepoints
            
            fig, axs = plt.subplots(8, 3, figsize=(24, 32))  # 8 latent vectors x 3 transitions
            fig.suptitle(f"PCA Latent Vectors for {taste_name} - Dataset {core_dataset_name}", fontsize=20)

            pc_columns = [col for col in taste_df.columns if col.startswith('PC_')][:8]  # Maximum 8 latent vectors

            for row_idx, pc_col in enumerate(pc_columns):
                for col_idx in range(min(num_transitions, 3)):  # Max 3 transitions
                    ax = axs[row_idx, col_idx]
                    
                    end_times = []
                    for trial in unique_trials:
                        trial_df = taste_df.filter(pl.col('trial') == trial)
                        trial_changepoints = changepoints[taste_idx][trial]

                        start_time = trial_changepoints[col_idx]
                        end_time = trial_changepoints[col_idx + 1]
                        time_window_mask = (trial_df['time'] >= start_time) & (trial_df['time'] <= end_time)

                        time_values = trial_df.filter(time_window_mask)['time'].to_numpy()
                        pc_values = trial_df.filter(time_window_mask)[pc_col].to_numpy()

                        if len(time_values) > 0:
                            end_times.append(time_values[-1])
                            ax.plot(time_values, pc_values, alpha=0.5)

                    if end_times: # adding in percentile lines for when each trial ends-- decently easy to visualize what's what
                        percentiles = np.percentile(end_times, [25, 50, 75])
                        ax.axvline(percentiles[0], color='red', linestyle='--', linewidth=2, label='25% Trials End')
                        ax.axvline(percentiles[1], color='blue', linestyle='-.', linewidth=2, label='50% Trials End')
                        ax.axvline(percentiles[2], color='green', linestyle=':', linewidth=2, label='75% Trials End')

                    ax.set_title(f"{pc_col} - Epoch {col_idx + 1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Principal Component Value")

            fig.legend(['25% Trials End', '50% Trials End', '75% Trials End'], loc="lower center", ncol=3, fontsize='small', frameon=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            output_file = os.path.join(df_output_dir, f"pca_latent_vectors_{core_dataset_name}_{taste_name}.png")
            plt.savefig(output_file)
            print(f"Saved PCA latent vectors plot: {output_file}")
            plt.close(fig)

## QUANTIFICATION OF SOME OF THIS MESS: shuffling data and testing against it: 

from sig_testing_class import SignificanceTester

from numpy.random import default_rng

rng = default_rng(42) # keeping a random seed to be working with -- for future use 

tester = SignificanceTester(
    data_dict=robust_pca_95,   # your dictionary of polars DataFrames
    rng=rng,
    save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/PCA_RNN_LAT_PLOTS/significance_testing/rob_95_pca_on_rnn",
    n_permutations=10000
)
results_summary = tester.run() # this kinda sucks and to find out why, we'll have to get into that later on, chuck


### NOW FOR THE FFT-- studying the oscillatory stuff
## all the above is hooey-- this is where we're at now: 
    
from freuqency_analysis_suite_rnn_class import FrequencyAnalysisPipeline # again now its own class 
tld = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/FFT_FREQUENCY_SUITE'
pipeline = FrequencyAnalysisPipeline(
    tld=tld,
    standardized_changepoints_dict= standardized_changepoints_dict,
    modified_tastes=modified_tastes,
    peak_height_threshold=0.25,
    spectrogram_nperseg=8,
    min_freq=2,        # You can set this!
    max_freq=15,     
    do_fft=True,
    do_periodogram=True,
    do_peaks=True,
    do_lombscargle=True,
    do_full_trials=True,
    do_spectrogram=True, # And this too!
    do_individual_plots = True,
    smart_skip = True, 
    min_amplitude_threshold = 0.05
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
from spike_raster_class_plot_april import SpikeRasterPlotter  # assumes the class is in this module
# new one: 
def process_and_plot_datasets(
    npz_path,
    pkl_path,
    base_output_dir,
    standardized_changepoints_dict=None,
    window_length=250,
    step_size=25,
    fixed_warp_duration=1000
):
    """
    CLI-enhanced pipeline for generating raster plots based on user options.
    Now avoids saving CalcFRStates unless debug mode is selected.
    """

    # Initial CLI prompt (just to check for debug mode)
    mode, is_warped, alignment, show_markers, sort_by_length = SpikeRasterPlotter.cli_options()

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
                pkl_path,
                spike_array,
                dataset_num_clean,
                index,
                key
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
                        fixed_warp_duration=fixed_warp_duration
                    )

                    fr_unwarped, spike_unwarped, fr_warped, spike_warped = calc.run()

                    if mode == "debug":
                        calc.spike_arrays_unwarped = spike_unwarped
                        calc.spike_arrays_warped = spike_warped
                        all_calc_objs[core_dataset_name] = calc

                    else:
                        spike_data = spike_warped if is_warped else spike_unwarped
                        dataset_dir = os.path.join(base_output_dir, f"dataset_{dataset_num_clean}")
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
                            changepoints_dict=standardized_changepoints_dict
                        )

                        plotter.plot_all()

                except Exception as e:
                    print(f"Error processing dataset {dataset_num_clean}: {e}")
            else:
                print(f"No valid changepoints for dataset {dataset_num_clean}; skipping...")

    if mode == "debug":
        if not all_calc_objs:
            print("No datasets were successfully loaded. Cannot enter debug mode.")
            return
        SpikeRasterPlotter.debug_extract_from_calc(all_calc_objs)

    print("All datasets processed.")
### all manner of spike train hooey above from a class that is called ### 


## also for warping etc. 

###### this is very old code that I'll have to clean up at a later date #### note as of 4/18

## CHECKING VIABILITY FOR CHANGING THE NUMBER OF LATANTS IN EACH PLOT -- old buiz (gotta do major housecleaning)
# using PCA to optimize for the number of latants I'm inferring 


# loop for going through the thresholded neurons; doing the whole PCA analysis pipeline on them too 
# all these plots are now in a separate file 
base_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/pca_thresholded_w_poststim'
os.makedirs(base_dir, exist_ok=True)
# Iterate over each DataFrame in sig_nrns_dict
for key, df in sig_nrns_dict.items():
    # Create a subdirectory for the current DataFrame
    sub_dir = os.path.join(base_dir, key)
    os.makedirs(sub_dir, exist_ok=True)
    # Apply the PCA function to the current DataFrame
    pca_results_df = whitened_pca_all_nrns_sep_taste(df, sub_dir)
    plot_averaged_pca_trajectories_thresholded_SEM(pca_results_df, sub_dir, modified_tastes, epoch_labels, step_size)
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
        print(f'Dataset number: {dataset_num}')
        dataset_tastes = process_info_files(info_path, dataset_num)
        modified_tastes = modify_tastes(dataset_tastes, taste_replacements)
        # Call unpickle_changepoints to process the .pkl files for the current dataset_num
        extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
        if extracted_pkl is not None:
            # Process the extracted_pkl data as needed
            print("Processed .pkl data:")
        else:
            print("Failed to process .pkl data")
        try: 
            changepoints = extracted_pkl[:, 3] # Giving 4 changepoint arrays, 1 for each taste-- for each bit of data
            window_length = 250 # make 25 for just straight binning, no sliding window lol (might work); for sliding make 250
            step_size = 25 # for sliding make 25
            alpha = 0.05 # for the t-test being run 

            #start_bin = 000/25 # not needed as specified inside function now 
            #end_bin = 7000/25 # not needed as specified inside function now
            # Calculate firing rates with changepoints
            state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)
            # Process firing rate data as needed
            print(f"Firing rate with changepoints calculated for dataset {dataset_num}:")
            all_tastes_averaged, all_interpolated_states = interpolate_and_average_all_tastes(state_firing_rates_all_trials)
            
            ### plotting functions ###
            
            ## Warped Plots ## 
            #plt_inter_cpfr_avg_nrn_line(all_interpolated_states, modified_tastes, dataset_num)           
            #plt_inter_cpfr_avg_tr_line(all_interpolated_states, modified_tastes, dataset_num)
            #plt_single_neuron_firing_warped_line(all_interpolated_states, dataset_num, modified_tastes, output_dir_warped)
            
            ## Unwarped Plots ##
            #plt_single_neuron_firing_unwarped_line(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)
           # plt_single_neuron_firing_unwarped_heatmap(state_firing_rates_all_trials, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)
          
            ## Changepoint plots ##
            #plt_changepoints_scat(changepoints, modified_tastes, dataset_num)
            #plt_changepoint_hist(changepoints, modified_tastes, dataset_num)
         
            ## Unwarped plots with significance dictated via t-test or 1 hz firing threshold## -- old 
            #thresh_firing_rates_unwarped = thresh_ttest_unwarped_hz(state_firing_rates_all_trials, alpha) # significance via ttest
            #thresh_firing_rates_unwarped = thresh_min_fr_unwarped_hz(state_firing_rates_all_trials, threshold_hz) # signifigance via 1 hz firing rate 
            #plt_single_neuron_firing_unwarped_heatmap_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels)
            #plt_single_neuron_firing_unwarped_line_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, modified_tastes, epoch_labels, step_size)

            # appending individual significant neurons to one big happy dataset
        except TypeError as e: 
            if str(e) == "'float' object is not iterable":
                print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
            else: 
                raise
                # error handling for one dataset whose data type seems to be irritating 
    else: 
        spike_array, index, key = data
        dataset_num = "Unknown Dataset" # as in for some reason we can't get a number
        print(f'Dataset number: {dataset_num}')  



    
    
    
