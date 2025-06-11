#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:54:40 2024

@author: vincentcalia-bogan
"""

import os
import tables 
import numpy as np

## ITERATING THROUGH H5 FILES AND SAVING SPIKE ARRAYS TO NPZ FILE ## 

# iterate through all files, finding the .h5 files and saving the path to a list 
def find_h5_files(file_path= None):
    if file_path is None:
      file_path = input("Enter the pathname for h5 files: ")

    h5_files = [] # init list 
    for entry in os.scandir(file_path):
        if entry.is_dir():
            subdir_h5_files = find_h5_files(entry.path)
            if subdir_h5_files:
                h5_files.extend(subdir_h5_files)
            else:
                pass  # If no .h5 files found in the subdirectory, do nothing
        elif entry.is_file() and entry.name.endswith('.h5'):
            h5_files.append(entry.path)
    return h5_files


# now, for each .h5 file, saving locations of spike trains and associated data to several 4-D arrays 
def extract_to_npz(h5_files, file_path, spike_trains_path = None, save_path = None):
    if spike_trains_path is None:
      spike_trains_path = '/spike_trains'
    if save_path is None:
      save_path = input("Enter the save path: ")
    
    num_h5_files = len(h5_files)
    
    for i, file_path in enumerate(h5_files): # i for number of files 
        with tables.open_file(file_path, mode='r') as f: #same as before
            try:
               dig_ins = f.list_nodes(spike_trains_path) # if the .h5 file hasn't been processed to generate spike trains
            except tables.NoSuchNodeError:
                   raise Exception(f"Spike trains not found at {spike_trains_path} for file {file_path}, skipping file.") # raise the exception 
            spike_trains = [x.spike_array[:] for x in dig_ins]
        spike_array = np.stack(spike_trains)
        
        # Create a dictionary to hold the spike arrays with specific names
        spike_arrays_dict = {f"spike_array {num_h5_files + i + 1}": spike_array} # + 1 to ensure we start at number 1 
        
        # Construct file name for saving
        file_name = f"spike_array {i + 1}.npz" # numbering 
        file_path = os.path.join(save_path, file_name)
        
        # Save the spike arrays to the file
        np.savez(file_path, **spike_arrays_dict) # ** for arbitrary number of arrays 
    
    print(f"Data saved successfully at {save_path}")
