#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:37:43 2024

@author: vincentcalia-bogan
"""

## doing things with parquets (serizalized neuron data frames, etc)


import polars as pl

from extract_npz import extract_from_npz

from unpkl_generator import extract_valid_changepoints

def serialized_neuron_df(npz_path, pkl_path):
    all_serialized_nrns = []
    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple):
            spike_array, dataset_num, index, key = data
            print(f'Dataset number: {dataset_num}')
            extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)
            if extracted_pkl is not None:
                print("Processed .pkl data:")
                try:
                    changepoints = extracted_pkl[:, 3]
                    num_tastes, num_trials, num_neurons, _ = spike_array.shape
                    for taste_idx in range(num_tastes):
                        for neuron_idx in range(num_neurons):
                            for cp_idx in range(len(changepoints)):
                                all_serialized_nrns.append({
                                    'dataset': dataset_num,
                                    'taste': taste_idx,
                                    'changepoint': cp_idx,
                                    'neuron': neuron_idx
                                })
                except TypeError as e:
                    if str(e) == "'float' object is not iterable":
                        print(f"Skipping dataset {dataset_num} due to TypeError: {e}")
                    else:
                        raise
            else:
                print("Failed to process .pkl data")
                continue  # Skip to the next dataset

    # Convert all_serialized_nrns to a Polars DataFrame and return
    if all_serialized_nrns:
        all_serialized_nrns_df = pl.DataFrame(all_serialized_nrns)
    else:
        all_serialized_nrns_df = pl.DataFrame([])  # Return an empty DataFrame if no data is processed
    return all_serialized_nrns_df

def create_overlap_dataframes(serialized_nrns_df, sig_nrns_dict):
    overlap_nrns_dict = {}    
    # Iterate over each DataFrame in the dictionary
    for df_name, df in sig_nrns_dict.items():
        # Extract unique (dataset, neuron) combinations
        unique_combinations = df.select(["dataset", "neuron"]).unique()
        uni_comb_df = pl.DataFrame(unique_combinations, schema = ["dataset", "neuron"])
        overlap_name = f"overlap_{df_name}"
        overlap_nrns_dict[overlap_name] = uni_comb_df
        print(f"Created overlap DataFrame '{overlap_name}'.")
    return overlap_nrns_dict