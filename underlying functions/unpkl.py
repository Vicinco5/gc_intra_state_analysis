#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:58:41 2024

@author: vincentcalia-bogan
"""

# UNPICKLE-ING ALL THE THINGS

import os, os.path
import numpy as np
import pandas as pd


def unpickle_changepoints(pkl_path, save_path):
    def truncate_name(name):
        # Extract numbers from parts of the name separated by underscores
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for i, part in enumerate(parts):
            if any(char.isdigit() for char in part):
                # If the part contains a number, add it to numbers and truncated_parts
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    extracted_pkl = None
    processed_files = set()
    processed_base_names = set()
    try:
        pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
        save_file_names = [
            truncate_name(os.path.splitext(os.path.basename(f))[0])
            for f in os.listdir(save_path)
        ]
        extracted_pkl = []
        for pkl_file in pkl_files:  # Iterate through each .pkl file
            if pkl_file in processed_files:
                continue  # Skip files that have already been processed
            pkl_file_path = os.path.join(pkl_path, pkl_file)
            try:
                df = pd.read_pickle(pkl_file_path)  # Open the pickle dataframe
                for index, row in df.iterrows():
                    row_basename = os.path.basename(row.iloc[0])
                    truncated_basename = truncate_name(row_basename)
                    if truncated_basename in save_file_names:
                        if (
                            truncated_basename in processed_base_names
                        ):  # Skip rows that have already been processed
                            continue
                        # Append the matching row to the extracted_pkl list
                        extracted_row = row.values
                        extracted_pkl.append(extracted_row)
                        processed_base_names.add(
                            truncated_basename
                        )  # Mark the base name as processed
                processed_files.add(pkl_file)
            except Exception as e:
                print(f"Error processing file {pkl_file}: {e}")
        extracted_pkl = np.array(extracted_pkl)
        return extracted_pkl
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    # note: df.columns = Index(['basename', 'taste_num', 'pkl_path', 'tau', 'present', 'tau_std'], dtype='object')
    # what we want specifically from this is the basename and 'tau', as the basename is the basename (we already have that)
    # and tau are each of the changepoints for every trial of that particular neuron -- we want col 0 and 4 for this
