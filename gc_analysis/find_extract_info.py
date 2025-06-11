#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:47:40 2024

@author: vincentcalia-bogan
"""

# handling .info files 

import os, os.path
import shutil
import json
import re

def find_copy_h5info(source_path=None, destination_path=None):
    if source_path is None:
        source_path = input("Enter the source directory path for .info files: ")
    if destination_path is None:
        destination_path = input("Enter the destination directory path for copying .info files: ")
    
    # Ensure destination directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    info_files = []  # Initialize list to store .info file paths
    for entry in os.scandir(source_path):
        if entry.is_dir():
            subdir_info_files = find_copy_h5info(entry.path, destination_path)
            if subdir_info_files:
                info_files.extend(subdir_info_files)
        elif entry.is_file() and entry.name.endswith('.info'):
            info_files.append(entry.path)
            # Copy file to the destination directory
            shutil.copy(entry.path, destination_path)
    
    return info_files

def process_info_files(info_path, dataset_num):
    def find_matching_info_file():
        dataset_numbers = re.findall(r'\d{6}', dataset_num)
        if len(dataset_numbers) < 2:
            raise ValueError("The dataset_num should contain at least two 6-digit numbers.")
        
        for entry in os.scandir(info_path):
            if entry.is_file() and entry.name.endswith('.info'):
                file_numbers = re.findall(r'\d{6}', entry.name)
                if len(file_numbers) >= 2 and file_numbers[:2] == dataset_numbers[:2]:
                    return entry.path
        return None
    def extract_tastes_from_info(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                # print("JSON content:", data)  # Debug print to inspect the JSON content
                if "taste_params" in data and "tastes" in data["taste_params"]:
                    return data["taste_params"]["tastes"]
                else:
                    print("No 'tastes' key found in the 'taste params' key of the JSON data.")  # Debug print if key is missing
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
        return []
    matching_info_file = find_matching_info_file()
    if matching_info_file:
        dataset_tastes = extract_tastes_from_info(matching_info_file)
        return dataset_tastes
    else:
        raise FileNotFoundError(f"No matching .info file found for dataset_num: {dataset_num}")
        
def modify_tastes(dataset_tastes, replacements):
    modified_tastes = []
    for taste in dataset_tastes:
        if taste in replacements:
            modified_tastes.append(replacements[taste])
        else:
            modified_tastes.append(taste)
    return modified_tastes


