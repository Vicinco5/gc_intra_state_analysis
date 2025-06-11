#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:37:45 2025

more plots; much ado about plots 
all sorts of old plot funcs that are archived, loosely grouped 

Unwarped firing rate plots; vintange: scifest and beyond 

@author: vincentcalia-bogan
"""
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import polars as pl

# Warped / Unwarped firing rates


# working test 
# this is the above but in a heatmap that is sorted according to ascending unwarped firing rate 
# modified to work with the thresholding stuff-- I really need to get better at swithcing which 
def plt_single_neuron_firing_unwarped_heatmap_thresholded(thresh_firing_rates_unwarped, dataset_num, output_dir_unwarped, window_length, step_size, modified_tastes, epoch_labels):
    num_tastes = len(thresh_firing_rates_unwarped)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    # Create the output directory if it doesn't exist
    dataset_dir = os.path.join(output_dir_unwarped, f'{dataset_num}_single-neuron-firing')
    os.makedirs(dataset_dir, exist_ok=True)
    num_neurons = thresh_firing_rates_unwarped[0][0].shape[1]  # Assuming consistent neuron count across tastes and changepoints
    num_trials = thresh_firing_rates_unwarped[0].shape[0]  # Assuming consistent trial count across tastes and changepoints
    # Loop through each neuron
    for neuron_idx in range(num_neurons):
        fig, axes = plt.subplots(num_tastes, num_changepoints, figsize=(15, 5 * num_tastes), sharex=False, sharey=False)
        fig.suptitle(f'Unwarped Single Neuron Firing Rates, Threshold: 1 hz, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}', fontsize=16, y=0.995)
        # Ensure axes is 2D
        if num_tastes == 1:
            axes = np.expand_dims(axes, 0)
        for taste_idx in range(num_tastes):
            start_time = 2000
            for cp_idx in range(num_changepoints):
                num_non0_trials = 0
                ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                ax.set_title(f'Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}')
                ax.set_xlabel('Time Duration of State')
                # Collect data for heatmap
                trial_data = []
                trial_lengths = []
                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = thresh_firing_rates_unwarped[taste_idx][trial_idx][cp_idx][neuron_idx].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if valid_data.size > 0:  # Only include valid (non-NaN) values
                        trial_lengths.append(valid_data.shape[0])
                        trial_data.append(valid_data)
                        num_non0_trials = num_non0_trials + 1
                if trial_data:
                    # Sort trials by length
                    sorted_indices = np.argsort(trial_lengths)
                    sorted_data = [trial_data[i] for i in sorted_indices]
                    sorted_lengths = sorted(trial_lengths)
                    avg_duration = (np.mean(trial_lengths) * step_size) # onset of stimulus 
                    # Create heatmap
                    max_length = max(trial_lengths)
                    heatmap_data = np.zeros((num_non0_trials, max_length)) * np.nan
                    for i, data in enumerate(sorted_data):
                        heatmap_data[i, :len(data)] = data
                    # Plot heatmap
                    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower', extent=[0, max_length * step_size, 0, num_non0_trials])
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Firing Rate (Hz)')
                    # Add vertical percentile lines
                    percentiles = np.percentile(sorted_lengths, [0, 25, 50, 75, 90])
                    labels = ['0th', '25th', '50th', '75th', '90th']
                    for i, percentile in enumerate(percentiles):
                        line_position = percentile * step_size
                        ax.axvline(line_position, linestyle=':', color='m', alpha=0.7)
                        ax.text(line_position, num_non0_trials - 1, f'{labels[i]} percentile', rotation=90, verticalalignment='top', color='c', alpha=1, fontsize = 10)

                    # Calculate the end time of the current state using max_time_bins
                    end_time = start_time + (max_length * step_size)
                    avg_duration = avg_duration
                    # Update the title to include the average duration
                    ax.set_title(f'Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms')
                    ax.set_xlabel('Time (ms)')
                    if cp_idx == 0:  # Only add Y label to the first column
                        ax.set_ylabel('Trial Index')
                    # Update start time for the next state
                    start_time = end_time
        plt.tight_layout()
        neuron_fig_path = os.path.join(dataset_dir, f'neuron_{neuron_idx + 1}.png')
        plt.savefig(neuron_fig_path)
        plt.close(fig)
  
# into one function 
# good, working version of this lol # 
# check to make sure we're not missing some significant neurons via t-test-- but then, given the fraction of neurons 
# that are significant, this may make some sense... especailly as this is being calculated a slightly different way? 
# check back on the old functions too though 

# added black line for averages

def plt_single_neuron_firing_unwarped_line_thresholded(sig_nrns_dict, output_dir, modified_tastes, epoch_labels, step_size):
    num_tastes = len(modified_tastes)
    num_changepoints = 4  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    
    for df_name, df in sig_nrns_dict.items():
        df_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_dir, exist_ok=True)
        
        unique_datasets = df['dataset'].unique()
        threshold_col_name = df.columns[-1]  # Get the name of the 7th column
        
        for dataset_num in unique_datasets:
            dataset_dir = os.path.join(df_dir, f'dataset_{dataset_num}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_df = df.filter(pl.col('dataset') == dataset_num)
            unique_neurons = dataset_df['neuron'].unique(maintain_order=True).to_list()
            num_trials = dataset_df['trial'].max() + 1
            
            colormap = cm.get_cmap('turbo', num_trials)  # Generate a colormap with `num_trials` colors
            
            # Calculate mean threshold value for each neuron
            neuron_threshold_df = dataset_df.group_by('neuron').agg(pl.col(threshold_col_name).mean().alias(threshold_col_name))
            neuron_threshold_dict = {row[0]: row[1] for row in neuron_threshold_df.rows()}
            
            neuron_threshold_values = np.array(list(neuron_threshold_dict.values()))
            neuron_percentiles = np.percentile(neuron_threshold_values, [25, 50, 75, 90])
            
            for neuron_idx in unique_neurons:
                if neuron_idx in neuron_threshold_dict:
                    neuron_value = neuron_threshold_dict[neuron_idx]
                    
                    percentile_score = None
                    if neuron_value <= neuron_percentiles[0]:
                        percentile_score = '25th'
                    elif neuron_value <= neuron_percentiles[1]:
                        percentile_score = '50th'
                    elif neuron_value <= neuron_percentiles[2]:
                        percentile_score = '75th'
                    else:
                        percentile_score = '90th'
                else:
                    neuron_value = None
                    percentile_score = 'NA'
                
                fig, axes = plt.subplots(num_tastes, num_changepoints, figsize=(15, 5 * num_tastes), sharex=False, sharey=False)
                if neuron_value is not None:
                    fig.suptitle(f'Warped Single Neuron Firing Rates, Threshold: 2Hz, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}, {threshold_col_name}: {neuron_value:.2f}', fontsize=14, y=0.995)
                else: #change between warped and unwarped as needed-- there should be a better way to do this but hey eh
                    fig.suptitle(f'Warped Single Neuron Firing Rates, Threshold: 2Hz, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}', fontsize=14, y=0.995)
                
                # Ensure axes is 2D
                if num_tastes == 1:
                    axes = np.expand_dims(axes, 0)
                
                for taste_idx in range(num_tastes):
                    for cp_idx in range(num_changepoints):
                        num_non0_trials = 0
                        ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                        ax.set_title(f'Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}')
                        ax.set_xlabel('Time Duration of State')
                        if cp_idx == 0:  # Only add Y label to the first column
                            ax.set_ylabel('Firing Rate (Hz)')
                        
                        trial_lengths = []
                        trial_data = []
                        
                        # Filter dataframe for the current taste, changepoint, and neuron
                        filtered_df = dataset_df.filter(
                            (pl.col('taste') == taste_idx) &
                            (pl.col('changepoint') == cp_idx) &
                            (pl.col('neuron') == neuron_idx)
                        )
                        
                        unique_trials = filtered_df['trial'].unique()
                        
                        for trial_idx in unique_trials:
                            trial_row = filtered_df.filter(pl.col('trial') == trial_idx)
                            if not trial_row.is_empty():
                                trial_array = np.array(trial_row['trial_data'][0])
                                valid_data = trial_array[~np.isnan(trial_array)]
                                if valid_data.size > 0:  # Only plot if there are valid (non-NaN) values
                                    trial_lengths.append(valid_data.shape[0])
                                    trial_data.append(valid_data)
                                    num_non0_trials += 1
                        
                        if trial_data:
                            sorted_indices = np.argsort(trial_lengths)
                            sorted_data = [trial_data[i] for i in sorted_indices]
                            sorted_lengths = sorted(trial_lengths)
                            avg_duration = (np.mean(sorted_lengths) * step_size)  # Onset of stimulus

                            for trial_idx, valid_data in enumerate(sorted_data):
                                trial_color = colormap(trial_idx)  # Get a unique color for each trial
                                num_time_bins = valid_data.shape[0]
                                x_ax = np.arange(num_time_bins) * step_size  # Convert to milliseconds
                                ax.plot(x_ax, valid_data, color=trial_color, alpha=1, linewidth = 1.5)

                            # Calculate the mean at each time point across all trials
                            max_time_bins = max(trial_lengths)
                            all_data = np.full((num_non0_trials, max_time_bins), np.nan)
                            for i, data in enumerate(sorted_data):
                                all_data[i, :len(data)] = data

                            mean_data = np.nanmean(all_data, axis=0)
                            mean_time_bins = np.arange(max_time_bins) * step_size
                            #ax.plot(mean_time_bins, mean_data, color='black', linewidth=1.5, label='Average')
                            #ax.legend()

                            trial_lengths = np.array(sorted_lengths)
                            duration_percentiles = np.percentile(trial_lengths, [0, 25, 50, 75, 90])

                            # the gray lines will all look exactly the same across neurons-- this is expected and ok 
                            labels = ['0th', '25th', '50th', '75th', '90th']
                            for i, percentile in enumerate(duration_percentiles): 
                                ax.axvline(percentile * step_size, linestyle = ':', color = 'grey', alpha = 0.6)
                                ax.text(percentile * step_size, ax.get_ylim()[1] * 0.9, f'{labels[i]} percentile', rotation=90, verticalalignment='top', color='grey', alpha=1)
                
                            # Update the title to include the average duration
                            ax.set_title(f'Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms')
                            ax.set_xlabel('Time (ms)')
                            if cp_idx == 0:  # Only add Y label to the first column
                                ax.set_ylabel('Firing Rate (Hz)')
                
                plt.tight_layout()
                neuron_fig_path = os.path.join(dataset_dir, f'{percentile_score}_pct_neuron_{neuron_idx + 1}.png')
                plt.savefig(neuron_fig_path)
                plt.close(fig)

# new plot that is similar to Don's ppth that he showed me-- IDK if this is correct or not: 
    # this is, but not in the figure structure he was looking for-- this is a good start though. 



def plot_neuron_mean_with_sem(sig_nrns_dict, output_dir, modified_tastes, epoch_labels, step_size):
    num_tastes = len(modified_tastes)
    num_changepoints = 4  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    
    for df_name, df in sig_nrns_dict.items():
        df_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_dir, exist_ok=True)
        
        unique_datasets = df['dataset'].unique()
        threshold_col_name = df.columns[-1]  # Get the name of the 7th column
        
        for dataset_num in unique_datasets:
            dataset_dir = os.path.join(df_dir, f'dataset_{dataset_num}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_df = df.filter(pl.col('dataset') == dataset_num)
            unique_neurons = dataset_df['neuron'].unique(maintain_order=True).to_list()
            
            # Calculate mean threshold value for each neuron
            neuron_threshold_df = dataset_df.group_by('neuron').agg(pl.col(threshold_col_name).mean().alias(threshold_col_name))
            neuron_threshold_dict = {row[0]: row[1] for row in neuron_threshold_df.rows()}
            
            for neuron_idx in unique_neurons:
                if neuron_idx in neuron_threshold_dict:
                    neuron_value = neuron_threshold_dict[neuron_idx]
                
                fig, axes = plt.subplots(num_tastes, num_changepoints, figsize=(15, 5 * num_tastes), sharex=False, sharey=False)
                fig.suptitle(f'Mean Firing Rates with SEM, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}', fontsize=14, y=0.995)
                
                # Ensure axes is 2D
                if num_tastes == 1:
                    axes = np.expand_dims(axes, 0)
                
                for taste_idx in range(num_tastes):
                    for cp_idx in range(num_changepoints):
                        ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                        ax.set_title(f'Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}')
                        ax.set_xlabel('Time Duration of State')
                        if cp_idx == 0:  # Only add Y label to the first column
                            ax.set_ylabel('Firing Rate (Hz)')
                        
                        trial_lengths = []
                        trial_data = []
                        
                        # Filter dataframe for the current taste, changepoint, and neuron
                        filtered_df = dataset_df.filter(
                            (pl.col('taste') == taste_idx) &
                            (pl.col('changepoint') == cp_idx) &
                            (pl.col('neuron') == neuron_idx)
                        )
                        
                        unique_trials = filtered_df['trial'].unique()
                        
                        for trial_idx in unique_trials:
                            trial_row = filtered_df.filter(pl.col('trial') == trial_idx)
                            if not trial_row.is_empty():
                                trial_array = np.array(trial_row['trial_data'][0])
                                valid_data = trial_array[~np.isnan(trial_array)]
                                if valid_data.size > 0:  # Only plot if there are valid (non-NaN) values
                                    trial_lengths.append(valid_data.shape[0])
                                    trial_data.append(valid_data)
                        
                        if trial_data:
                            sorted_lengths = sorted(trial_lengths)
                            max_time_bins = max(sorted_lengths)
                            
                            # Calculate the mean and standard error of the mean (SEM)
                            all_data = np.full((len(trial_data), max_time_bins), np.nan)
                            for i, data in enumerate(trial_data):
                                all_data[i, :len(data)] = data
                            
                            mean_data = np.nanmean(all_data, axis=0)
                            sem_data = np.nanstd(all_data, axis=0) / np.sqrt(np.sum(~np.isnan(all_data), axis=0))
                            mean_time_bins = np.arange(max_time_bins) * step_size
                            
                            # Plot the mean line
                            ax.plot(mean_time_bins, mean_data, color='black', linewidth=2, label='Average')
                            # Plot the SEM as a shaded region
                            ax.fill_between(mean_time_bins, mean_data - sem_data, mean_data + sem_data, color='gray', alpha=0.5, label='SEM')
                            ax.legend()
                
                plt.tight_layout()
                neuron_fig_path = os.path.join(dataset_dir, f'mean_with_sem_neuron_{neuron_idx + 1}.png')
                plt.savefig(neuron_fig_path)
                plt.close(fig)
                

# new figure with the specifications that Don was looking for (code below); new output dir

# unwarped SEM neurons; plotting such that neurons can be directly compared on the figures. Important. Do warped data next. 
def plot_neuron_mean_with_sem_grid(sig_nrns_dict, output_dir, modified_tastes, epoch_labels, step_size):
    num_tastes = len(modified_tastes)
    num_changepoints = 4  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    neurons_per_figure = 16  # 4x4 grid
    
    for df_name, df in sig_nrns_dict.items():
        df_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_dir, exist_ok=True)
        
        unique_datasets = df['dataset'].unique()
        threshold_col_name = df.columns[-1]  # Get the name of the 7th column
        
        for dataset_num in unique_datasets:
            dataset_dir = os.path.join(df_dir, f'dataset_{dataset_num}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_df = df.filter(pl.col('dataset') == dataset_num)
            unique_neurons = dataset_df['neuron'].unique(maintain_order=True).to_list()
            
            # Calculate mean threshold value for each neuron
            neuron_threshold_df = dataset_df.group_by('neuron').agg(pl.col(threshold_col_name).mean().alias(threshold_col_name))
            neuron_threshold_dict = {row[0]: row[1] for row in neuron_threshold_df.rows()}
            
            for taste_idx in range(num_tastes):
                for cp_idx in range(num_changepoints):
                    fig_num = 1
                    for neuron_start in range(0, len(unique_neurons), neurons_per_figure):
                        fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=False, sharey=False)
                        fig.suptitle(f'Mean Unwarped Firing Rates with SEM, Epoch: {epoch_labels[cp_idx]}, Taste: {taste_labels[taste_idx]}, Dataset: {dataset_num}', fontsize=18, y=0.995)
                        
                        for i, neuron_idx in enumerate(unique_neurons[neuron_start:neuron_start + neurons_per_figure]):
                            row = i // 4
                            col = i % 4
                            ax = axes[row, col]
                            
                            if neuron_idx in neuron_threshold_dict:
                                neuron_value = neuron_threshold_dict[neuron_idx]
                            
                            ax.set_title(f'Neuron {neuron_idx + 1}')
                            ax.set_xlabel('Time Duration of State')
                            if col == 0:  # Only add Y label to the first column
                                ax.set_ylabel('Firing Rate (Hz)')
                            
                            trial_lengths = []
                            trial_data = []
                            
                            # Filter dataframe for the current taste, changepoint, and neuron
                            filtered_df = dataset_df.filter(
                                (pl.col('taste') == taste_idx) &
                                (pl.col('changepoint') == cp_idx) &
                                (pl.col('neuron') == neuron_idx)
                            )
                            
                            unique_trials = filtered_df['trial'].unique()
                            
                            for trial_idx in unique_trials:
                                trial_row = filtered_df.filter(pl.col('trial') == trial_idx)
                                if not trial_row.is_empty():
                                    trial_array = np.array(trial_row['trial_data'][0])
                                    valid_data = trial_array[~np.isnan(trial_array)]
                                    if valid_data.size > 0:  # Only plot if there are valid (non-NaN) values
                                        trial_lengths.append(valid_data.shape[0])
                                        trial_data.append(valid_data)
                            
                            if trial_data:
                                sorted_lengths = sorted(trial_lengths)
                                max_time_bins = max(sorted_lengths)
                                
                                # Calculate the mean and standard error of the mean (SEM)
                                all_data = np.full((len(trial_data), max_time_bins), np.nan)
                                for j, data in enumerate(trial_data):
                                    all_data[j, :len(data)] = data
                                
                                mean_data = np.nanmean(all_data, axis=0)
                                sem_data = np.nanstd(all_data, axis=0) / np.sqrt(np.sum(~np.isnan(all_data), axis=0))
                                mean_time_bins = np.arange(max_time_bins) * step_size
                                
                                # Plot the mean line
                                ax.plot(mean_time_bins, mean_data, color='black', linewidth=2, label='Average')
                                # Plot the SEM as a shaded region
                                ax.fill_between(mean_time_bins, mean_data - sem_data, mean_data + sem_data, color='gray', alpha=0.5, label='SEM')
                                ax.legend()
                        
                        plt.tight_layout()
                        neuron_fig_path = os.path.join(dataset_dir, f'mean_with_sem_epoch_{cp_idx}_taste_{taste_idx}_fig_{fig_num}.png')
                        plt.savefig(neuron_fig_path)
                        plt.close(fig)
                        fig_num += 1
                        
#### NEW 4/14/25: UPDATED CLASS WITH FIRING RATE DATA ######

def plot_neuron_mean_with_sem_grid_from_class(
    sig_nrns_dict,
    output_dir,
    modified_tastes,
    epoch_labels,
    step_size,
    firing_rate_data,
    use_warped=True
):
    """
    Plots neuron firing rate averages with SEM in a 4x4 grid per dataset/taste/epoch,
    using firing rate data produced by the CalcFRStates class.

    Parameters
    ----------
    sig_nrns_dict : dict of polars.DataFrame
        Dictionary containing only neuron metadata and filtering info.
    output_dir : str
        Where to save figures.
    modified_tastes : list of str
        Taste names used for plotting.
    epoch_labels : list of str
        Names of each changepoint epoch (length should match num_changepoints).
    step_size : int
        Step size used in CalcFRStates, to convert bins to time (in ms).
    firing_rate_data : xr.DataArray
        Output from CalcFRStates. Either unwarped or warped firing rates.
    use_warped : bool, optional
        If True, expects firing_rate_data to be warped.
    """

    num_tastes = len(modified_tastes)
    num_changepoints = firing_rate_data.sizes['segment']
    neurons_per_figure = 16

    for df_name, df in sig_nrns_dict.items():
        df_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_dir, exist_ok=True)

        unique_datasets = df['dataset'].unique()

        for dataset_num in unique_datasets:
            dataset_dir = os.path.join(df_dir, f'dataset_{dataset_num}')
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_df = df.filter(pl.col('dataset') == dataset_num)
            unique_neurons = dataset_df['neuron'].unique(maintain_order=True).to_list()

            for taste_idx in range(num_tastes):
                for cp_idx in range(num_changepoints):
                    fig_num = 1
                    for neuron_start in range(0, len(unique_neurons), neurons_per_figure):
                        fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=False, sharey=False)
                        fig.suptitle(
                            f'Mean {"Warped" if use_warped else "Unwarped"} Firing Rates with SEM, '
                            f'Epoch: {epoch_labels[cp_idx]}, Taste: {modified_tastes[taste_idx]}, '
                            f'Dataset: {dataset_num}', fontsize=18, y=0.995
                        )

                        for i, neuron_idx in enumerate(unique_neurons[neuron_start:neuron_start + neurons_per_figure]):
                            row, col = divmod(i, 4)
                            ax = axes[row, col]
                            ax.set_title(f'Neuron {neuron_idx + 1}')
                            ax.set_xlabel('Time Duration of State (ms)')
                            if col == 0:
                                ax.set_ylabel('Firing Rate (Hz)')

                            # Select relevant trials from xarray
                            try:
                                trials_data = firing_rate_data.sel(
                                    taste=taste_idx,
                                    segment=cp_idx,
                                    neuron=neuron_idx
                                ).values  # Shape: (num_trials, time_bins)

                                # Filter out completely NaN trials
                                valid_trials = [row[~np.isnan(row)] for row in trials_data if not np.all(np.isnan(row))]
                                if valid_trials:
                                    max_bins = max(len(row) for row in valid_trials)
                                    all_data = np.full((len(valid_trials), max_bins), np.nan)
                                    for j, row in enumerate(valid_trials):
                                        all_data[j, :len(row)] = row

                                    mean_data = np.nanmean(all_data, axis=0)
                                    sem_data = np.nanstd(all_data, axis=0) / np.sqrt(np.sum(~np.isnan(all_data), axis=0))
                                    time_axis = np.arange(max_bins) * step_size

                                    ax.plot(time_axis, mean_data, color='black', linewidth=2, label='Average')
                                    ax.fill_between(time_axis, mean_data - sem_data, mean_data + sem_data,
                                                    color='gray', alpha=0.5, label='SEM')
                                    ax.legend()
                            except KeyError:
                                continue

                        plt.tight_layout()
                        neuron_fig_path = os.path.join(
                            dataset_dir,
                            f'mean_with_sem_epoch_{cp_idx}_taste_{taste_idx}_fig_{fig_num}.png'
                        )
                        plt.savefig(neuron_fig_path)
                        plt.close(fig)
                        fig_num += 1





# quick plot that will show the population activity to support claims I'm making in the scifest poster. 
def plot_population_activity(sig_nrns_dict, output_dir, modified_tastes, epoch_labels, step_size):
    num_tastes = len(modified_tastes)
    num_changepoints = 4  # Updated number of changepoints to include poststimulus activity
    
    for df_name, df in sig_nrns_dict.items():
        df_dir = os.path.join(output_dir, df_name)
        os.makedirs(df_dir, exist_ok=True)
        
        unique_datasets = df['dataset'].unique()
        
        for dataset_num in unique_datasets:
            dataset_df = df.filter(pl.col('dataset') == dataset_num)
            unique_neurons = dataset_df['neuron'].unique(maintain_order=True).to_list()
            
            fig, axes = plt.subplots(num_tastes, num_changepoints, figsize=(15, 20), sharex=False, sharey=False)
            fig.suptitle(f'Population Activity, Dataset: {dataset_num}', fontsize=14, y=0.995)
            
            for taste_idx in range(num_tastes):
                for cp_idx in range(num_changepoints):
                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(f'Taste: {modified_tastes[taste_idx]}, Epoch: {epoch_labels[cp_idx]}')
                    ax.set_xlabel('Time Duration of State')
                    if cp_idx == 0:  # Only add Y label to the first column
                        ax.set_ylabel('Firing Rate (Hz)')
                    
                    all_neuron_data = []
                    
                    for neuron_idx in unique_neurons:
                        trial_lengths = []
                        trial_data = []
                        
                        # Filter dataframe for the current taste, changepoint, and neuron
                        filtered_df = dataset_df.filter(
                            (pl.col('taste') == taste_idx) &
                            (pl.col('changepoint') == cp_idx) &
                            (pl.col('neuron') == neuron_idx)
                        )
                        
                        unique_trials = filtered_df['trial'].unique()
                        
                        for trial_idx in unique_trials:
                            trial_row = filtered_df.filter(pl.col('trial') == trial_idx)
                            if not trial_row.is_empty():
                                trial_array = np.array(trial_row['trial_data'][0])
                                valid_data = trial_array[~np.isnan(trial_array)]
                                if valid_data.size > 0:  # Only plot if there are valid (non-NaN) values
                                    trial_lengths.append(valid_data.shape[0])
                                    trial_data.append(valid_data)
                        
                        if trial_data:
                            sorted_lengths = sorted(trial_lengths)
                            max_time_bins = max(sorted_lengths)
                            
                            # Calculate the mean of all trials for the neuron
                            all_data = np.full((len(trial_data), max_time_bins), np.nan)
                            for i, data in enumerate(trial_data):
                                all_data[i, :len(data)] = data
                            
                            mean_trial_data = np.nanmean(all_data, axis=0)
                            all_neuron_data.append(mean_trial_data)
                    
                    if all_neuron_data:
                        max_time_bins = max(len(data) for data in all_neuron_data)
                        all_data = np.full((len(all_neuron_data), max_time_bins), np.nan)
                        for i, data in enumerate(all_neuron_data):
                            all_data[i, :len(data)] = data
                        
                        mean_neuron_data = np.nanmean(all_data, axis=0)
                        sem_neuron_data = np.nanstd(all_data, axis=0) / np.sqrt(np.sum(~np.isnan(all_data), axis=0))
                        mean_time_bins = np.arange(max_time_bins) * step_size
                        
                        # Plot the mean neuron data
                        ax.plot(mean_time_bins, mean_neuron_data, color='black', linewidth=2, label='Average')
                        # Plot the SEM as a shaded region
                        ax.fill_between(mean_time_bins, mean_neuron_data - sem_neuron_data, mean_neuron_data + sem_neuron_data, color='gray', alpha=0.5, label='SEM')
                        ax.legend()
            
            plt.tight_layout()
            neuron_fig_path = os.path.join(df_dir, f'en_fr_{dataset_num}.png')
            plt.savefig(neuron_fig_path)
            plt.close(fig)