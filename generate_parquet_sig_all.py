#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:38:16 2024

@author: vincentcalia-bogan
"""

# significance testing neurons -- using parquet files and polars dataarrays 
import os, os.path
import numpy as np
from scipy.stats import ttest_rel
import polars as pl
# calling vinports 
from extract_npz import extract_from_npz
from unpkl_generator import unpickle_changepoints
from unpkl_generator import extract_valid_changepoints
from calc_firing_rate_with_states import calc_fr_states
# importing the script from the dead that warps the firing rates too 
from interpolation_xr import interpolate_and_average_all_tastes, interpolate_firing_rates


def sig_neurons_ttest(npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size):
    file_name = "sig_neurons_ttest.parquet"
    file_path = os.path.join(sig_ttest_parquet_path, file_name)
    all_sig_neuron_dataframes = []
    recorded_neurons = set()

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple):
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try:
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_all_trials, state_spike_arrays_all_trials, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    
                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]
                    
                    significant_neurons = set()  # Track neurons found significant in any epoch
                    pvals_for_neurons = {}  # Dictionary to track p-values for significant neurons
                    
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                if (dataset_num, neuron_idx) in recorded_neurons:
                                    continue  # Skip already recorded neurons
                                
                                trial_means_first_half = []
                                trial_means_second_half = []
                                
                                for trial_idx in range(num_trials):
                                    trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    valid_data = trial_array[~np.isnan(trial_array)]
                                    if valid_data.size > 0:
                                        half_point = len(valid_data) // 2
                                        mean_first_half = np.mean(valid_data[:half_point])
                                        mean_second_half = np.mean(valid_data[half_point:])
                                        trial_means_first_half.append(mean_first_half)
                                        trial_means_second_half.append(mean_second_half)
                                    
                                if trial_means_first_half and trial_means_second_half:
                                    tstat, pval = ttest_rel(trial_means_first_half, trial_means_second_half)
                                    if pval < alpha and not np.isnan(pval):
                                        recorded_neurons.add((dataset_num, neuron_idx))
                                        significant_neurons.add(neuron_idx)
                                        pvals_for_neurons[neuron_idx] = pval  # Record p-values

                    # Collect and save all data for significant neurons
                    for neuron_idx in significant_neurons:
                        for taste_idx in range(num_tastes):
                            for cp_idx in range(num_changepoints):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
                                    all_sig_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
                                        'p_value': pvals_for_neurons[neuron_idx]
                                    })

                    if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
                        if 'consolidated_ttest_df' in locals():
                            consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
                        else:
                            consolidated_ttest_df = temp_df
                        all_sig_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset
    
    # Process any remaining data
    if all_sig_neuron_dataframes:
        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
        if 'consolidated_ttest_df' in locals():
            consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
        else:
            consolidated_ttest_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_ttest_df' in locals():
        consolidated_ttest_df.write_parquet(file_path)
        print(f"Saved significant neuron data to '{sig_ttest_parquet_path}'.")
        
# generating warped one as well 
def sig_neurons_war_ttest(npz_path, pkl_path, sig_ttest_parquet_path_w, alpha, window_length, step_size):
    file_name = "sig_neurons_ttest_warped.parquet"
    file_path = os.path.join(sig_ttest_parquet_path_w, file_name)
    all_sig_neuron_dataframes = []
    recorded_neurons = set()

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple):
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try:
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_uw_all_trials, state_spike_arrays_all_trial, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    state_firing_rates_all_trials = state_firing_rates_warped
                #    state_firing_rates_all_trials, all_tastes_averaged = interpolate_firing_rates(state_firing_rates_uw_all_trials, state_spike_arrays_all_trials)
# trying new things 
             #       all_tastes_averaged, state_firing_rates_all_trials = interpolate_and_average_all_tastes(state_firing_rates_uw_all_trials)
                    
                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]
                    
                    significant_neurons = set()  # Track neurons found significant in any epoch
                    pvals_for_neurons = {}  # Dictionary to track p-values for significant neurons
                    
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                if (dataset_num, neuron_idx) in recorded_neurons:
                                    continue  # Skip already recorded neurons
                                
                                trial_means_first_half = []
                                trial_means_second_half = []
                                
                                for trial_idx in range(num_trials):
                                    trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    valid_data = trial_array[~np.isnan(trial_array)]
                                    if valid_data.size > 0:
                                        half_point = len(valid_data) // 2
                                        mean_first_half = np.mean(valid_data[:half_point])
                                        mean_second_half = np.mean(valid_data[half_point:])
                                        trial_means_first_half.append(mean_first_half)
                                        trial_means_second_half.append(mean_second_half)
                                    
                                if trial_means_first_half and trial_means_second_half:
                                    tstat, pval = ttest_rel(trial_means_first_half, trial_means_second_half)
                                    if pval < alpha and not np.isnan(pval):
                                        recorded_neurons.add((dataset_num, neuron_idx))
                                        significant_neurons.add(neuron_idx)
                                        pvals_for_neurons[neuron_idx] = pval  # Record p-values

                    # Collect and save all data for significant neurons
                    for neuron_idx in significant_neurons:
                        for taste_idx in range(num_tastes):
                            for cp_idx in range(num_changepoints):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
                                    all_sig_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
                                        'p_value': pvals_for_neurons[neuron_idx]
                                    })

                    if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
                        if 'consolidated_ttest_df' in locals():
                            consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
                        else:
                            consolidated_ttest_df = temp_df
                        all_sig_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset
    
    # Process any remaining data
    if all_sig_neuron_dataframes:
        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
        if 'consolidated_ttest_df' in locals():
            consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
        else:
            consolidated_ttest_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_ttest_df' in locals():
        consolidated_ttest_df.write_parquet(file_path)
        print(f"Saved significant neuron data to '{sig_ttest_parquet_path_w}'.")

# thresholding according to t-test results -- this has error handling built in; now built into the 
# function that extracts valid changepoints 


# def sig_neurons_ttest(npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size):
#     file_name = "sig_neurons_ttest.parquet"
#     file_path = os.path.join(sig_ttest_parquet_path, file_name)
#     all_sig_neuron_dataframes = []
#     recorded_neurons = set()
    
#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple):
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#                 try:
#                     changepoints = extracted_pkl[:, 3]
#                     state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    
#                     num_tastes = len(state_firing_rates_all_trials)
#                     num_changepoints = len(changepoints[1][1])
#                     num_neurons = state_firing_rates_all_trials.shape[3]
#                     num_trials = state_firing_rates_all_trials.shape[1]
                    
#                     significant_neurons = set()  # Track neurons found significant in any epoch
#                     pvals_for_neurons = {}  # Dictionary to track p-values for significant neurons
                    
#                     for taste_idx in range(num_tastes):
#                         for cp_idx in range(num_changepoints):
#                             for neuron_idx in range(num_neurons):
#                                 if (dataset_num, neuron_idx) in recorded_neurons:
#                                     continue  # Skip already recorded neurons
                                
#                                 trial_means_first_half = []
#                                 trial_means_second_half = []
                                
#                                 for trial_idx in range(num_trials):
#                                     trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                     valid_data = trial_array[~np.isnan(trial_array)]
#                                     if valid_data.size > 0:
#                                         half_point = len(valid_data) // 2
#                                         mean_first_half = np.mean(valid_data[:half_point])
#                                         mean_second_half = np.mean(valid_data[half_point:])
#                                         trial_means_first_half.append(mean_first_half)
#                                         trial_means_second_half.append(mean_second_half)
                                    
#                                 if trial_means_first_half and trial_means_second_half:
#                                     tstat, pval = ttest_rel(trial_means_first_half, trial_means_second_half)
#                                     if pval < alpha and not np.isnan(pval):
#                                         recorded_neurons.add((dataset_num, neuron_idx))
#                                         significant_neurons.add(neuron_idx)
#                                         pvals_for_neurons[neuron_idx] = pval  # Record p-values

#                     # Collect and save all data for significant neurons
#                     for neuron_idx in significant_neurons:
#                         for taste_idx in range(num_tastes):
#                             for cp_idx in range(num_changepoints):
#                                 for trial_idx in range(num_trials):
#                                     trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                     trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
#                                     all_sig_neuron_dataframes.append({
#                                         'dataset': dataset_num,
#                                         'taste': taste_idx,
#                                         'changepoint': cp_idx,
#                                         'neuron': neuron_idx,
#                                         'trial': trial_idx,
#                                         'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
#                                         'p_value': pvals_for_neurons[neuron_idx]
#                                     })

#                     if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#                         if 'consolidated_ttest_df' in locals():
#                             consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
#                         else:
#                             consolidated_ttest_df = temp_df
#                         all_sig_neuron_dataframes = []

#                 except TypeError as e:
#                     if str(e) == "'float' object is not iterable":
#                         print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                     else:
#                         raise
#             else:
#                 print("Failed to process .pkl data")
#                 continue  # Skip to the next dataset
    
#     # Process any remaining data
#     if all_sig_neuron_dataframes:
#         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#         if 'consolidated_ttest_df' in locals():
#             consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
#         else:
#             consolidated_ttest_df = temp_df

#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_ttest_df' in locals():
#         consolidated_ttest_df.write_parquet(file_path)
#         print(f"Saved significant neuron data to '{sig_ttest_parquet_path}'.")

### for older, simple extract pkl without adding cp's ### 

# def sig_neurons_ttest(npz_path, pkl_path, sig_ttest_parquet_path, alpha, window_length, step_size):
#     file_name = "sig_neurons_ttest.parquet"
#     file_path = os.path.join(sig_ttest_parquet_path, file_name)
#     all_sig_neuron_dataframes = []
#     recorded_neurons = set()
    
#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple):
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#             else:
#                 print("Failed to process .pkl data")
#             try:
#                 changepoints = extracted_pkl[:, 3]
#                 state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)
                
#                 num_tastes = len(state_firing_rates_all_trials)
#                 num_changepoints = 4
#                 num_neurons = state_firing_rates_all_trials.shape[3]
#                 num_trials = state_firing_rates_all_trials.shape[1]
                
#                 significant_neurons = set()  # Track neurons found significant in any epoch
#                 pvals_for_neurons = {}  # Dictionary to track p-values for significant neurons
                
#                 for taste_idx in range(num_tastes):
#                     for cp_idx in range(num_changepoints):
#                         for neuron_idx in range(num_neurons):
#                             if (dataset_num, neuron_idx) in recorded_neurons:
#                                 continue  # Skip already recorded neurons
                            
#                             trial_means_first_half = []
#                             trial_means_second_half = []
                            
#                             for trial_idx in range(num_trials):
#                                 trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                 valid_data = trial_array[~np.isnan(trial_array)]
#                                 if valid_data.size > 0:
#                                     half_point = len(valid_data) // 2
#                                     mean_first_half = np.mean(valid_data[:half_point])
#                                     mean_second_half = np.mean(valid_data[half_point:])
#                                     trial_means_first_half.append(mean_first_half)
#                                     trial_means_second_half.append(mean_second_half)
# # note this t-test method is slightly different in that means are involved rather than doing so 
# # on a trial by trial basis. This eliminates some of the detail retained in the individual trials, but 
# # probably is more robust? maybe? 
#                             if trial_means_first_half and trial_means_second_half:
#                                 tstat, pval = ttest_rel(trial_means_first_half, trial_means_second_half)
#                                 if pval < alpha and not np.isnan(pval):
#                                     recorded_neurons.add((dataset_num, neuron_idx))
#                                     significant_neurons.add(neuron_idx)
#                                     pvals_for_neurons[neuron_idx] = pval  # Record p-values

#                 # Collect and save all data for significant neurons
#                 for neuron_idx in significant_neurons:
#                     for taste_idx in range(num_tastes):
#                         for cp_idx in range(num_changepoints):
#                             for trial_idx in range(num_trials):
#                                 trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                 trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
#                                 all_sig_neuron_dataframes.append({
#                                     'dataset': dataset_num,
#                                     'taste': taste_idx,
#                                     'changepoint': cp_idx,
#                                     'neuron': neuron_idx,
#                                     'trial': trial_idx,
#                                     'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
#                                     'p_value': pvals_for_neurons[neuron_idx]
#                                 })

#                 if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                     temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#                     if 'consolidated_ttest_df' in locals():
#                         consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
#                     else:
#                         consolidated_ttest_df = temp_df
#                     all_sig_neuron_dataframes = []

#             except TypeError as e:
#                 if str(e) == "'float' object is not iterable":
#                     print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                 else:
#                     raise
    
#     # Process any remaining data
#     if all_sig_neuron_dataframes:
#         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#         if 'consolidated_ttest_df' in locals():
#             consolidated_ttest_df = pl.concat([consolidated_ttest_df, temp_df], rechunk=True)
#         else:
#             consolidated_ttest_df = temp_df

#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_ttest_df' in locals():
#         consolidated_ttest_df.write_parquet(file_path)
#         print(f"Saved significant neuron data to '{sig_ttest_parquet_path}'.")
        
# thresholding according to a minimum frequency         
# commented function is designed for the simpler pkl extraction methods 


def sig_neurons_hz(npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size):
    file_name = "sig_neurons_2hz_thresh.parquet"
    file_path = os.path.join(sig_hz_parquet_path, file_name)
    all_sig_neuron_dataframes = []

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple): 
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try: 
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_all_trials, state_spike_arrays_all_trials, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    
                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]

                    significant_neurons = set()  # Track neurons found significant in any epoch
                    avg_freq_nrn = {}  # Dictionary to track frequencies above threshold
                    
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                trial_avg_fr = []
                                for trial_idx in range(num_trials):
                                    trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    valid_data = trial_array[~np.isnan(trial_array)]
                                    if valid_data.size > 0:
                                        avg_fr_trial = np.nanmean(valid_data)
                                        trial_avg_fr.append(avg_fr_trial)
                                
                                if trial_avg_fr:
                                    overall_avg_fr = np.mean(trial_avg_fr)
                                    if overall_avg_fr >= threshold_hz:
                                        significant_neurons.add(neuron_idx)
                                        avg_freq_nrn[neuron_idx] = overall_avg_fr

                    # Collect and save all data for significant neurons
                    for neuron_idx in significant_neurons:
                        for taste_idx in range(num_tastes):
                            for cp_idx in range(num_changepoints):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
                                    all_sig_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
                                        'average_freq': avg_freq_nrn[neuron_idx]
                                    })

                    if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
                        if 'consolidated_hz_df' in locals():
                            consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
                        else:
                            consolidated_hz_df = temp_df
                        all_sig_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset

    # Process any remaining data
    if all_sig_neuron_dataframes:
        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
        if 'consolidated_hz_df' in locals():
            consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
        else:
            consolidated_hz_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_hz_df' in locals():
        consolidated_hz_df.write_parquet(file_path)
        print(f"Saved significant neuron data to '{sig_hz_parquet_path}'.")
        
# same deal but for warped data now: 
def sig_neurons_war_hz(npz_path, pkl_path, sig_hz_parquet_path_w, threshold_hz, window_length, step_size):
    file_name = "sig_neurons_2hz_thresh_warped.parquet"
    file_path = os.path.join(sig_hz_parquet_path_w, file_name)
    all_sig_neuron_dataframes = []

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple): 
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try: 
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_uw_all_trials, state_spike_arrays_all_trial, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    state_firing_rates_all_trials = state_firing_rates_warped
                #    state_firing_rates_all_trials, all_tastes_averaged = interpolate_firing_rates(state_firing_rates_uw_all_trials, state_spike_arrays_all_trials)
# trying new things 
             #       all_tastes_averaged, state_firing_rates_all_trials = interpolate_and_average_all_tastes(state_firing_rates_uw_all_trials)
                    
                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]

                    significant_neurons = set()  # Track neurons found significant in any epoch
                    avg_freq_nrn = {}  # Dictionary to track frequencies above threshold
                    
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                trial_avg_fr = []
                                for trial_idx in range(num_trials):
                                    trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    valid_data = trial_array[~np.isnan(trial_array)]
                                    if valid_data.size > 0:
                                        avg_fr_trial = np.nanmean(valid_data)
                                        trial_avg_fr.append(avg_fr_trial)
                                
                                if trial_avg_fr:
                                    overall_avg_fr = np.mean(trial_avg_fr)
                                    if overall_avg_fr >= threshold_hz:
                                        significant_neurons.add(neuron_idx)
                                        avg_freq_nrn[neuron_idx] = overall_avg_fr

                    # Collect and save all data for significant neurons
                    for neuron_idx in significant_neurons:
                        for taste_idx in range(num_tastes):
                            for cp_idx in range(num_changepoints):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
                                    all_sig_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
                                        'average_freq': avg_freq_nrn[neuron_idx]
                                    })

                    if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
                        if 'consolidated_hz_df' in locals():
                            consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
                        else:
                            consolidated_hz_df = temp_df
                        all_sig_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset

    # Process any remaining data
    if all_sig_neuron_dataframes:
        temp_df = pl.DataFrame(all_sig_neuron_dataframes)
        if 'consolidated_hz_df' in locals():
            consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
        else:
            consolidated_hz_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_hz_df' in locals():
        consolidated_hz_df.write_parquet(file_path)
        print(f"Saved significant neuron data to '{sig_hz_parquet_path_w}'.")


# this function has error handling built in rather than divided responsibility (back on the extraction func now)
# as such still works with the core extract pkl func but we like this less 

# def sig_neurons_hz(npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size):
#     file_name = "sig_neurons_2hz_thresh.parquet"
#     file_path = os.path.join(sig_hz_parquet_path, file_name)
#     all_sig_neuron_dataframes = []

#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple): 
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#                 try: 
#                     changepoints = extracted_pkl[:, 3]
#                     state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    
#                     num_tastes = len(state_firing_rates_all_trials)
#                     num_changepoints = len(changepoints[1][1])
#                     num_neurons = state_firing_rates_all_trials.shape[3]
#                     num_trials = state_firing_rates_all_trials.shape[1]

#                     significant_neurons = set()  # Track neurons found significant in any epoch
#                     avg_freq_nrn = {}  # Dictionary to track frequencies above threshold
                    
#                     for taste_idx in range(num_tastes):
#                         for cp_idx in range(num_changepoints):
#                             for neuron_idx in range(num_neurons):
#                                 trial_avg_fr = []
#                                 for trial_idx in range(num_trials):
#                                     trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                     valid_data = trial_array[~np.isnan(trial_array)]
#                                     if valid_data.size > 0:
#                                         avg_fr_trial = np.nanmean(valid_data)
#                                         trial_avg_fr.append(avg_fr_trial)
                                
#                                 if trial_avg_fr:
#                                     overall_avg_fr = np.mean(trial_avg_fr)
#                                     if overall_avg_fr >= threshold_hz:
#                                         significant_neurons.add(neuron_idx)
#                                         avg_freq_nrn[neuron_idx] = overall_avg_fr

#                     # Collect and save all data for significant neurons
#                     for neuron_idx in significant_neurons:
#                         for taste_idx in range(num_tastes):
#                             for cp_idx in range(num_changepoints):
#                                 for trial_idx in range(num_trials):
#                                     trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                     trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
#                                     all_sig_neuron_dataframes.append({
#                                         'dataset': dataset_num,
#                                         'taste': taste_idx,
#                                         'changepoint': cp_idx,
#                                         'neuron': neuron_idx,
#                                         'trial': trial_idx,
#                                         'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
#                                         'average_freq': avg_freq_nrn[neuron_idx]
#                                     })

#                     if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#                         if 'consolidated_hz_df' in locals():
#                             consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
#                         else:
#                             consolidated_hz_df = temp_df
#                         all_sig_neuron_dataframes = []

#                 except TypeError as e:
#                     if str(e) == "'float' object is not iterable":
#                         print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                     else:
#                         raise
#             else:
#                 print("Failed to process .pkl data")
#                 continue  # Skip to the next dataset

#     # Process any remaining data
#     if all_sig_neuron_dataframes:
#         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#         if 'consolidated_hz_df' in locals():
#             consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
#         else:
#             consolidated_hz_df = temp_df

#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_hz_df' in locals():
#         consolidated_hz_df.write_parquet(file_path)
#         print(f"Saved significant neuron data to '{sig_hz_parquet_path}'.")

# simpler, for when we're not adding changepoints excess 

# def sig_neurons_hz(npz_path, pkl_path, sig_hz_parquet_path, threshold_hz, window_length, step_size):
#     file_name = "sig_neurons_2hz_thresh.parquet"
#     file_path = os.path.join(sig_hz_parquet_path, file_name)
#     all_sig_neuron_dataframes = []

#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple): 
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#             else:
#                 print("Failed to process .pkl data")
#             try: 
#                 changepoints = extracted_pkl[:, 3]
#                 state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)
                
#                 num_tastes = len(state_firing_rates_all_trials)
#                 num_changepoints = len(changepoints)
#                 num_neurons = state_firing_rates_all_trials.shape[3]
#                 num_trials = state_firing_rates_all_trials.shape[1]

#                 significant_neurons = set()  # Track neurons found significant in any epoch
#                 avg_freq_nrn = {}  # Dictionary to track frequencies above threshold
                
#                 for taste_idx in range(num_tastes):
#                     for cp_idx in range(num_changepoints):
#                         for neuron_idx in range(num_neurons):
#                             trial_avg_fr = []
#                             for trial_idx in range(num_trials):
#                                 trial_array = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                 valid_data = trial_array[~np.isnan(trial_array)]
#                                 if valid_data.size > 0:
#                                     avg_fr_trial = np.nanmean(valid_data)
#                                     trial_avg_fr.append(avg_fr_trial)
                            
#                             if trial_avg_fr:
#                                 overall_avg_fr = np.mean(trial_avg_fr)
#                                 if overall_avg_fr >= threshold_hz:
#                                     significant_neurons.add(neuron_idx)
#                                     avg_freq_nrn[neuron_idx] = overall_avg_fr

#                 # Collect and save all data for significant neurons
#                 for neuron_idx in significant_neurons:
#                     for taste_idx in range(num_tastes):
#                         for cp_idx in range(num_changepoints):
#                             for trial_idx in range(num_trials):
#                                 trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                 trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # Remove NaN values
#                                 all_sig_neuron_dataframes.append({
#                                     'dataset': dataset_num,
#                                     'taste': taste_idx,
#                                     'changepoint': cp_idx,
#                                     'neuron': neuron_idx,
#                                     'trial': trial_idx,
#                                     'trial_data': trial_data_cleaned.tolist(),  # Convert to list for Polars
#                                     'average_freq': avg_freq_nrn[neuron_idx]
#                                 })

#                 if len(all_sig_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                     temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#                     if 'consolidated_hz_df' in locals():
#                         consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
#                     else:
#                         consolidated_hz_df = temp_df
#                     all_sig_neuron_dataframes = []

#             except TypeError as e:
#                 if str(e) == "'float' object is not iterable":
#                     print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                 else:
#                     raise

#     # Process any remaining data
#     if all_sig_neuron_dataframes:
#         temp_df = pl.DataFrame(all_sig_neuron_dataframes)
#         if 'consolidated_hz_df' in locals():
#             consolidated_hz_df = pl.concat([consolidated_hz_df, temp_df], rechunk=True)
#         else:
#             consolidated_hz_df = temp_df

#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_hz_df' in locals():
#         consolidated_hz_df.write_parquet(file_path)
#         print(f"Saved significant neuron data to '{sig_hz_parquet_path}'.")



# just throwing all neuron data into one big dataframe 
# commented functions are from when pkl extraction was simpler; these have extra data-handling to account for comre complexity
 

def consolidate_all_neuron_data(npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size):
    file_name = "consolidated_neurons.parquet"
    file_path = os.path.join(all_nrns_parquet_path, file_name)
    all_neuron_dataframes = []

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple):
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try:
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_all_trials, state_spike_arrays_all_trials, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)

                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]
                    time_bins = state_firing_rates_all_trials.shape[4]

                    # Collect data for all neurons
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # mask off the nan values
                                    all_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist()  # convert to list for Polars
                                    })

                    if len(all_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_neuron_dataframes)
                        if 'consolidated_df' in locals():
                            consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
                        else:
                            consolidated_df = temp_df
                        all_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset

    # Process any remaining data
    if all_neuron_dataframes:
        temp_df = pl.DataFrame(all_neuron_dataframes)
        if 'consolidated_df' in locals():
            consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
        else:
            consolidated_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_df' in locals():
        consolidated_df.write_parquet(file_path)
        print(f"Saved consolidated neuron data to '{all_nrns_parquet_path}'.")
        
# again same deal but with warped data too 
def consolidate_all_neuron_war_data(npz_path, pkl_path, all_nrns_parquet_path_w, window_length, step_size):
    file_name = "consolidated_neurons_warped.parquet"
    file_path = os.path.join(all_nrns_parquet_path_w, file_name)
    all_neuron_dataframes = []

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple):
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try:
                    changepoints = extracted_pkl[:, 3]
                    state_firing_rates_uw_all_trials, state_spike_arrays_all_trial, state_firing_rates_warped, state_spike_arrays_warped = calc_fr_states(spike_array, changepoints, window_length, step_size)
                    state_firing_rates_all_trials = state_firing_rates_warped
                #    state_firing_rates_all_trials, all_tastes_averaged = interpolate_firing_rates(state_firing_rates_uw_all_trials, state_spike_arrays_all_trials)
# trying new things 
             #       all_tastes_averaged, state_firing_rates_all_trials = interpolate_and_average_all_tastes(state_firing_rates_uw_all_trials)
                    
                    num_tastes = len(state_firing_rates_all_trials)
                    num_changepoints = len(changepoints[1][1])
                    num_neurons = state_firing_rates_all_trials.shape[3]
                    num_trials = state_firing_rates_all_trials.shape[1]
                    time_bins = state_firing_rates_all_trials.shape[4]

                    # Collect data for all neurons
                    for taste_idx in range(num_tastes):
                        for cp_idx in range(num_changepoints):
                            for neuron_idx in range(num_neurons):
                                for trial_idx in range(num_trials):
                                    trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
                                    trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # mask off the nan values
                                    all_neuron_dataframes.append({
                                        'dataset': dataset_num,
                                        'taste': taste_idx,
                                        'changepoint': cp_idx,
                                        'neuron': neuron_idx,
                                        'trial': trial_idx,
                                        'trial_data': trial_data_cleaned.tolist()  # convert to list for Polars
                                    })

                    if len(all_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
                        temp_df = pl.DataFrame(all_neuron_dataframes)
                        if 'consolidated_df' in locals():
                            consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
                        else:
                            consolidated_df = temp_df
                        all_neuron_dataframes = []

                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset

    # Process any remaining data
    if all_neuron_dataframes:
        temp_df = pl.DataFrame(all_neuron_dataframes)
        if 'consolidated_df' in locals():
            consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
        else:
            consolidated_df = temp_df

    # Save the consolidated dataframe as .parquet
    if 'consolidated_df' in locals():
        consolidated_df.write_parquet(file_path)
        print(f"Saved consolidated neuron data to '{all_nrns_parquet_path_w}'.")

# has error handling built in-- dividion of responsibility has since been implemented. 

# def consolidate_all_neuron_data(npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size):
#     file_name = "consolidated_neurons.parquet"
#     file_path = os.path.join(all_nrns_parquet_path, file_name)
#     all_neuron_dataframes = []

#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple):
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#                 try:
#                     changepoints = extracted_pkl[:, 3]
#                     state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)

#                     num_tastes = len(state_firing_rates_all_trials)
#                     num_changepoints = len(changepoints[1][1])
#                     num_neurons = state_firing_rates_all_trials.shape[3]
#                     num_trials = state_firing_rates_all_trials.shape[1]
#                     time_bins = state_firing_rates_all_trials.shape[4]

#                     # Collect data for all neurons
#                     for taste_idx in range(num_tastes):
#                         for cp_idx in range(num_changepoints):
#                             for neuron_idx in range(num_neurons):
#                                 for trial_idx in range(num_trials):
#                                     trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                     trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # mask off the nan values
#                                     all_neuron_dataframes.append({
#                                         'dataset': dataset_num,
#                                         'taste': taste_idx,
#                                         'changepoint': cp_idx,
#                                         'neuron': neuron_idx,
#                                         'trial': trial_idx,
#                                         'trial_data': trial_data_cleaned.tolist()  # convert to list for Polars
#                                     })

#                     if len(all_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                         temp_df = pl.DataFrame(all_neuron_dataframes)
#                         if 'consolidated_df' in locals():
#                             consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
#                         else:
#                             consolidated_df = temp_df
#                         all_neuron_dataframes = []

#                 except TypeError as e:
#                     if str(e) == "'float' object is not iterable":
#                         print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                     else:
#                         raise
#             else:
#                 print("Failed to process .pkl data")
#                 continue  # Skip to the next dataset

#     # Process any remaining data
#     if all_neuron_dataframes:
#         temp_df = pl.DataFrame(all_neuron_dataframes)
#         if 'consolidated_df' in locals():
#             consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
#         else:
#             consolidated_df = temp_df

#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_df' in locals():
#         consolidated_df.write_parquet(file_path)
#         print(f"Saved consolidated neuron data to '{all_nrns_parquet_path}'.")

# def consolidate_all_neuron_data(npz_path, pkl_path, all_nrns_parquet_path, window_length, step_size):
#     file_name = "consolidated_neurons.parquet"
#     file_path = os.path.join(all_nrns_parquet_path, file_name)
#     all_neuron_dataframes = []

#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple):
#             spike_array, dataset_num, index, key = data
#             print(f'Dataset number: {dataset_num}')
#             extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#             if extracted_pkl is not None:
#                 print("Processed .pkl data:")
#             else:
#                 print("Failed to process .pkl data")
#             try:
#                 changepoints = extracted_pkl[:, 3]
#                 state_firing_rates_all_trials, state_spike_arrays_all_trials = calc_fr_states(spike_array, changepoints, window_length, step_size)

#                 num_tastes = len(state_firing_rates_all_trials)
#                 num_changepoints = len(changepoints)
#                 num_neurons = state_firing_rates_all_trials.shape[3]
#                 num_trials = state_firing_rates_all_trials.shape[1]
#                 time_bins = state_firing_rates_all_trials.shape[4]

#                 # Collect data for all neurons
#                 for taste_idx in range(num_tastes):
#                     for cp_idx in range(num_changepoints):
#                         for neuron_idx in range(num_neurons):
#                             for trial_idx in range(num_trials):
#                                 trial_data = state_firing_rates_all_trials[taste_idx, trial_idx, cp_idx, neuron_idx, :].values
#                                 trial_data_cleaned = trial_data[~np.isnan(trial_data)]  # mask off the nan values
#                                 all_neuron_dataframes.append({
#                                     'dataset': dataset_num,
#                                     'taste': taste_idx,
#                                     'changepoint': cp_idx,
#                                     'neuron': neuron_idx,
#                                     'trial': trial_idx,
#                                     'trial_data': trial_data_cleaned.tolist()  # convert to list for Polars
#                                 })
# # for opening back up, convert these firing rates back to an array 
# # note for polars: it doesn't like to save arrays to a parquet; those have to be a list. 
# # also, eliminated the padding with nan values. if needed can be added later on. 
#                 if len(all_neuron_dataframes) > 10000:  # Arbitrary large chunk size to prevent memory issues
#                     temp_df = pl.DataFrame(all_neuron_dataframes)
#                     if 'consolidated_df' in locals():
#                         consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
#                     else:
#                         consolidated_df = temp_df
#                     all_neuron_dataframes = []

#             except TypeError as e:
#                 if str(e) == "'float' object is not iterable":
#                     print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
#                 else:
#                     raise
#     # Process any remaining data
#     if all_neuron_dataframes:
#         temp_df = pl.DataFrame(all_neuron_dataframes)
#         if 'consolidated_df' in locals():
#             consolidated_df = pl.concat([consolidated_df, temp_df], rechunk=True)
#         else:
#             consolidated_df = temp_df
#     # Save the consolidated dataframe as .parquet
#     if 'consolidated_df' in locals():
#         consolidated_df.write_parquet(file_path)
#         print(f"Saved consolidated neuron data to '{all_nrns_parquet_path}'.")
