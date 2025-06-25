#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:17:38 2024

@author: vincentcalia-bogan
"""
import os
import pandas as pd
import numpy as np

# UNPLK-ING THINGS, DESIGNED SPECIFICALLY FOR A GENERATOR ##
# this code is no longer skipping tastes-- as I aggressively optimized code too much.
# this particular function adds a 2000 ms post-stim col to the changepoints that I can then use further-- though this
# will be somewhat complex as I need to correct a whole whack of downstream funcs


# new func that should work just as well
def extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key):
    def truncate_name(name):
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for part in parts:
            if any(char.isdigit() for char in part):
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    try:
        truncated_dataset_num = truncate_name(dataset_num)
        print(f"Processing dataset number: {truncated_dataset_num}")
        raw_pkl = []
        pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
        for pkl_file in pkl_files:
            pkl_file_path = os.path.join(pkl_path, pkl_file)
            try:
                df = pd.read_pickle(pkl_file_path)
                for idx, row in df.iterrows():
                    row_basename = os.path.basename(row.iloc[0])
                    truncated_basename = truncate_name(row_basename)
                    if truncated_basename == truncated_dataset_num:
                        extracted_row = row.values
                        if isinstance(extracted_row[3], np.ndarray):
                            raw_pkl.append(extracted_row)
                        else:
                            print(
                                f"Skipping dataset {truncated_dataset_num} due to invalid data structure at index {idx}: {extracted_row[3]}"
                            )
                            raw_pkl = []
                            break
            except Exception as e:
                print(f"Error processing file {pkl_file}: {e}")
                raw_pkl = []
                break

        if not raw_pkl:
            print(
                f"Skipping dataset {truncated_dataset_num} due to invalid data structures."
            )
            return None

        print(
            f"Finished processing dataset {truncated_dataset_num}. Starting to process raw_pkl."
        )
        raw_pkl = np.array(raw_pkl, dtype=object)

        # Create extracted_pkl by ensuring correct length and adding the fourth number
        extracted_pkl = []
        for i in range(len(raw_pkl)):
            new_row = list(raw_pkl[i])
            new_sub_array = []
            for j in range(len(new_row[3])):
                row = new_row[3][j]
                if isinstance(row, (list, np.ndarray)) and len(row) == 3:
                    row = np.append(row, row[2] + 2000)
                elif isinstance(row, (list, np.ndarray)) and len(row) < 3:
                    padding = [np.nan] * (3 - len(row))
                    row = np.append(row, padding)
                    row = np.append(row, row[2] + 2000)
                else:
                    print(
                        f"Unexpected data structure for row {j} in new_row[{i}]: {row}"
                    )
                    continue
                new_sub_array.append(row)
            new_row[3] = np.array(new_sub_array)
            extracted_pkl.append(new_row)

        extracted_pkl = np.array(extracted_pkl, dtype=object)
        print(f"Finished processing valid dataset {truncated_dataset_num}.")
        return extracted_pkl if len(extracted_pkl) > 0 else None
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def unpickle_changepoints(pkl_path, generator):
    def truncate_name(name):
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for part in parts:
            if any(char.isdigit() for char in part):
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    all_extracted_pkl = []

    try:
        print("Starting to process pkl files.")
        for spike_array, dataset_num, index, key in generator:
            truncated_dataset_num = truncate_name(dataset_num)
            print(f"Processing dataset number: {truncated_dataset_num}")
            raw_pkl = []
            pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
            for pkl_file in pkl_files:
                pkl_file_path = os.path.join(pkl_path, pkl_file)
                try:
                    df = pd.read_pickle(pkl_file_path)
                    for idx, row in df.iterrows():
                        row_basename = os.path.basename(row.iloc[0])
                        truncated_basename = truncate_name(row_basename)
                        if truncated_basename == truncated_dataset_num:
                            extracted_row = row.values
                            if isinstance(extracted_row[3], np.ndarray):
                                raw_pkl.append(extracted_row)
                            else:
                                print(
                                    f"Skipping dataset {truncated_dataset_num} due to invalid data structure at index {idx}: {extracted_row[3]}"
                                )
                                raw_pkl = []
                                break
                except Exception as e:
                    print(f"Error processing file {pkl_file}: {e}")
                    raw_pkl = []
                    break

            if not raw_pkl:
                print(
                    f"Skipping dataset {truncated_dataset_num} due to invalid data structures."
                )
                continue

            print(
                f"Finished processing dataset {truncated_dataset_num}. Starting to process raw_pkl."
            )
            raw_pkl = np.array(raw_pkl, dtype=object)

            # Create extracted_pkl by ensuring correct length and adding the fourth number
            extracted_pkl = []
            for i in range(len(raw_pkl)):
                new_row = list(raw_pkl[i])
                new_sub_array = []
                for j in range(len(new_row[3])):
                    row = new_row[3][j]
                    if isinstance(row, (list, np.ndarray)) and len(row) == 3:
                        row = np.append(row, row[2] + 2000)
                    elif isinstance(row, (list, np.ndarray)) and len(row) < 3:
                        padding = [np.nan] * (3 - len(row))
                        row = np.append(row, padding)
                        row = np.append(row, row[2] + 2000)
                    else:
                        print(
                            f"Unexpected data structure for row {j} in new_row[{i}]: {row}"
                        )
                        continue
                    new_sub_array.append(row)
                new_row[3] = np.array(new_sub_array)
                extracted_pkl.append(new_row)

            extracted_pkl = np.array(extracted_pkl, dtype=object)
            all_extracted_pkl.extend(extracted_pkl)
            print(f"Finished processing valid dataset {truncated_dataset_num}.")

        all_extracted_pkl = np.array(all_extracted_pkl, dtype=object)
        return all_extracted_pkl if len(all_extracted_pkl) > 0 else None
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


### BELOW WORKS FOR WHEN THERE'RE AREN'T EXTRA CHANGEPOINTS WE'RE ADDING

# def unpickle_changepoints(pkl_path, generator):
#     def truncate_name(name):
#         numbers = []
#         parts = name.split('_')
#         truncated_parts = []
#         for part in parts:
#             if any(char.isdigit() for char in part):
#                 numbers.extend(char for char in part if char.isdigit())
#                 truncated_parts.append(part)
#         return '_'.join(truncated_parts)

#     try:
#         raw_pkl = []
#         for spike_array, dataset_num, index, key in generator:
#             truncated_dataset_num = truncate_name(dataset_num)
#             pkl_files = [f for f in os.listdir(pkl_path) if f.endswith('.pkl')]
#             for pkl_file in pkl_files:
#                 pkl_file_path = os.path.join(pkl_path, pkl_file)
#                 try:
#                     df = pd.read_pickle(pkl_file_path)
#                     for index, row in df.iterrows():
#                         row_basename = os.path.basename(row.iloc[0])
#                         truncated_basename = truncate_name(row_basename)
#                         if truncated_basename == truncated_dataset_num:
#                             extracted_row = row.values
#                             raw_pkl.append(extracted_row)
#                 except Exception as e:
#                     print(f"Error processing file {pkl_file}: {e}")

#         raw_pkl = np.array(raw_pkl, dtype=object)
#         # Create extracted_pkl by ensuring correct length and adding the fourth number
#         extracted_pkl = []
#         for i in range(len(raw_pkl)):
#             new_row = list(raw_pkl[i])
#             new_sub_array = []
#             for j in range(len(new_row[3])):
#                 row = new_row[3][j]
#                 if len(row) == 3:
#                     row = np.append(row, row[2] + 2000)
#                 elif len(row) < 3:
#                     padding = [np.nan] * (3 - len(row))
#                     row = np.append(row, padding)
#                     row = np.append(row, row[2] + 2000)
#                 new_sub_array.append(row)
#             new_row[3] = np.array(new_sub_array)
#             extracted_pkl.append(new_row)

#         extracted_pkl = np.array(extracted_pkl, dtype=object)
#         return extracted_pkl
#     except FileNotFoundError:
#         print(f"Error: Directory {pkl_path} not found.")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# def unpickle_changepoints(pkl_path, generator):
#     def truncate_name(name):
#         numbers = []
#         parts = name.split('_')
#         truncated_parts = []
#         for part in parts:
#             if any(char.isdigit() for char in part):
#                 numbers.extend(char for char in part if char.isdigit())
#                 truncated_parts.append(part)
#         return '_'.join(truncated_parts)

#     try:
#         extracted_pkl = []
#         for spike_array, dataset_num, index, key in generator:
#             truncated_dataset_num = truncate_name(dataset_num)
#             pkl_files = [f for f in os.listdir(pkl_path) if f.endswith('.pkl')]
#             for pkl_file in pkl_files:
#                 pkl_file_path = os.path.join(pkl_path, pkl_file)
#                 try:
#                     df = pd.read_pickle(pkl_file_path)
#                     #print(df.columns) # in the future, pull things out according to these names in the pkl
#                     for index, row in df.iterrows():
#                         row_basename = os.path.basename(row.iloc[0])
#                         truncated_basename = truncate_name(row_basename)
#                         if truncated_basename == truncated_dataset_num:
#                             extracted_row = row.values
#                             extracted_pkl.append(extracted_row)
#                 except Exception as e:
#                     print(f"Error processing file {pkl_file}: {e}")
#         extracted_pkl = np.array(extracted_pkl)
#         return extracted_pkl
#     except FileNotFoundError:
#         print(f"Error: Directory {pkl_path} not found.")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# changes from previous funciton: Turns out I optimized files too hard, removed sets that kept .pkl from being opened again
# and again

# gotta fix this:
#     from glob import glob

#     pkl_file_path = os.path.join(pkl_path, "*.pkl")

#     pkl_file_path
#     Out[46]: '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/pkl files/*.pkl'

#     pkl_file_path = glob(os.path.join(pkl_path, "*.pkl"))

#     pkl_file_path
#     Out[48]: ['/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/pkl files/tau_frame.pkl']

#     pkl_file_path = glob(os.path.join(pkl_path, "*.pkl"))[0]
# df = pd.read_pickle(pkl_file_path)
# Look up how to better traverse pandas dataframes!!! very important.
# the issue with the code below appeared to be that I was prematurely filtering out files based on what had been run
# see below-- comments and all
#
# def unpickle_changepoints(pkl_path, generator):
#     def truncate_name(name):
#         numbers = []
#         parts = name.split('_')
#         truncated_parts = []
#         for i, part in enumerate(parts):
#             if any(char.isdigit() for char in part):
#                 numbers.extend(char for char in part if char.isdigit())
#                 truncated_parts.append(part)
#         return '_'.join(truncated_parts)
#     processed_files = set() # prematurely filtering things
#     processed_base_names = set() # premature filtering again
#     try:
#         extracted_pkl = []
#         for spike_array, dataset_num, index, key in generator:
#             truncated_dataset_num = truncate_name(dataset_num)
#             pkl_files = [f for f in os.listdir(pkl_path) if f.endswith('.pkl')]
#             for pkl_file in pkl_files:
#                 if pkl_file in processed_files:
#                     continue
#                 pkl_file_path = os.path.join(pkl_path, pkl_file)
#                 try:
#                     df = pd.read_pickle(pkl_file_path)
#                     for index, row in df.iterrows():
#                         row_basename = os.path.basename(row.iloc[0])
#                         truncated_basename = truncate_name(row_basename)
#                         if truncated_basename == truncated_dataset_num:
#                             if truncated_basename in processed_base_names:
#                                 continue
#                             extracted_row = row.values
#                             extracted_pkl.append(extracted_row)
#                             processed_base_names.add(truncated_basename)
#                     processed_files.add(pkl_file)
#                 except Exception as e:
#                     print(f"Error processing file {pkl_file}: {e}")
#         extracted_pkl = np.array(extracted_pkl)
#         return extracted_pkl
#     except FileNotFoundError:
#         print(f"Error: Directory {pkl_path} not found.")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# df.columns
# Out[52]: Index(['basename', 'taste_num', 'pkl_path', 'tau', 'present', 'tau_std'], dtype='object')

# # CALLING IT FROM THE GENERATOR##
# for data in extract_from_npz(save_path): # i for indexing number of .npz files
#     if isinstance(data, tuple):
#         spike_array, dataset_num, index, key = data
#         print(f'Dataset number: {dataset_num}')

#         # Call unpickle_changepoints to process the .pkl files for the current dataset_num
#         extracted_pkl = unpickle_changepoints(pkl_path, [(spike_array, dataset_num, index, key)])
#         if extracted_pkl is not None:
#             # Process the extracted_pkl data as needed
#             print("Processed .pkl data:")
#         else:
#             print("Failed to process .pkl data")

#     else:
#         spike_array, index, key = data
#         dataset_num = "Unknown Dataset" # as in for some reason we can't get a number
#         print(f'Dataset number: {dataset_num}')
