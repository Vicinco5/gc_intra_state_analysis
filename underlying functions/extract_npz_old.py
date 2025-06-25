#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:02:02 2024

@author: vincentcalia-bogan
"""

## MAKING A GENERATOR FOR THE NPZ FILES FOR FURTHER OPERATIONS ##
# TODO: Figure out if I actually want to be using a generator or just a big ol list
import os
import numpy as np


def extract_from_npz(save_path=None):
    if save_path is None:
        save_path = input("Enter the save path: ")

    # Check if save_path exists
    if not os.path.exists(save_path):
        print(f"Error: Save path '{save_path}' does not exist.")
        return

    # Check if save_path is a directory
    if not os.path.isdir(save_path):
        print(f"Error: '{save_path}' is not a directory.")
        return

    # Iterate over .npz files in save_path
    npz_files = [f for f in os.listdir(save_path) if f.endswith(".npz")]
    if not npz_files:
        print(f"No .npz files found in '{save_path}'.")
        return

    for npz_file in npz_files:
        file_path = os.path.join(save_path, npz_file)
        try:
            npz_data = np.load(file_path)
            for key in npz_data.keys():
                if key.startswith("spike_array_"):
                    spike_array = npz_data[key]
                    yield spike_array, npz_file
        except Exception as e:
            print(f"Error processing '{npz_file}': {e}")
            continue


# Example usage:
# for spike_array, npz_file in extract_from_npz('/path/to/npz_files'):
#     print(spike_array, npz_file)


# opening and iterating over npz files -- this is a tricky bit of code due to the use of yeild and generators
# def extract_from_npz(save_path = None):
#     # if save_path is None:
#     #   save_path = input("Enter the save path: ")
#     npz_files = os.listdir(save_path)
#     for npz_file in npz_files:
#         if npz_file.endswith('.npz'): # filter out other potential files
#             file_path = os.path.join(save_path, npz_file)
#             npz_data = np.load(file_path) # load data, create a dictionary
#             for key in npz_data.keys():
#                 if key.startswith('spike_array_'): # indexing data correctly using a dictionary; the key command
#                     spike_array = npz_data[key] # create a generator for a given spike array
#                     yield spike_array, npz_file # understand this command-- is it worth using a generator like this?

### ONCE THE .NPZ FILES ARE GENERATED, YOU CAN WORK OFF THE LOCAL STORAGE SO LONG AS YOU DON'T CALL ANY OF THE ABOVE ###
