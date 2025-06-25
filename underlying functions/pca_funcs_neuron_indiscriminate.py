#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:14:58 2024

@author: vincentcalia-bogan
"""

## these are all old PCA functions that work
## but had to clean up the main script big time

## specifically indiscriminant of taste PCAs
## NON-VINCENT MODULE IMPORTS ##
# 17 datasets 365 neurons total
import sys
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


## VINCENT MODULE IMPORTS aka VIMPORTS##
# TODO: Fix these file imports later so not as hard coded, but that's a later project #
# MODULE DIRECTORIES
project_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work"
submodule_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/underlying functions"
# Check if paths already exist in sys.path
sys.path.append(submodule_dir)
sys.path.append(project_dir)

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
sig_hz_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/sig_parquet"
all_nrns_parquet_path = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/all_parquet"
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

## WHOLE NEURON PCA FUNCS ##


def whitened_pca_all_nrns_all_taste(all_data_df):
    all_pca_results = []

    unique_datasets = all_data_df["dataset"].unique().to_list()

    for dataset in unique_datasets:
        dataset_df = all_data_df.filter(pl.col("dataset") == dataset)

        unique_changepoints = dataset_df["changepoint"].unique().to_list()
        unique_neurons = dataset_df["neuron"].unique().to_list()

        for changepoint in unique_changepoints:
            concatenated_data_list = []
            trial_lengths = []  # To store the original lengths of each trial

            for neuron in unique_neurons:
                neuron_df = dataset_df.filter(
                    (pl.col("changepoint") == changepoint)
                    & (pl.col("neuron") == neuron)
                )
                trials_data = neuron_df["trial_data"].to_list()

                # Ensure trial data is 2D and store original lengths
                trials_data = [
                    (
                        np.array(trial).reshape(1, -1)
                        if np.array(trial).ndim == 1
                        else np.array(trial)
                    )
                    for trial in trials_data
                ]

                if not trial_lengths:
                    trial_lengths = [trial.shape[1] for trial in trials_data]

                # Concatenate trial data along axis 1 (each trial is a row)
                concatenated_trials = np.concatenate(trials_data, axis=1)

                # Append the concatenated trials to the data list
                concatenated_data_list.append(concatenated_trials)

            # Create a Polars DataFrame where each column is a neuron and each row is concatenated trial data
            data_dict = {
                str(neuron): concatenated_data_list[i].flatten()
                for i, neuron in enumerate(unique_neurons)
            }
            neuron_df = pl.DataFrame(data_dict)

            # Convert the dataframe to a numpy array for scaling and PCA
            all_data = neuron_df.to_numpy()

            # Standardize the data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(all_data)

            # Check for sufficient variation in the data
            if np.any(np.std(standardized_data, axis=0) > 0):
                pca = PCA(whiten=True)
                pca_result = pca.fit_transform(standardized_data)
                explained_variance = pca.explained_variance_ratio_
                singular_values = pca.singular_values_

                # Save PCA results to the list
                all_pca_results.append(
                    (
                        dataset,
                        changepoint,
                        pca_result,
                        explained_variance,
                        singular_values,
                        trial_lengths,
                    )
                )
            else:
                print(
                    f"Skipping PCA for dataset {dataset}, changepoint {changepoint} due to lack of variation."
                )

    # Convert results to a Polars DataFrame
    pca_results_df = pl.DataFrame(
        {
            "dataset": [result[0] for result in all_pca_results],
            "changepoint": [result[1] for result in all_pca_results],
            "pca_result": [
                result[2].tolist() for result in all_pca_results
            ],  # Store as list
            "explained_variance_ratio": [
                result[3].tolist() for result in all_pca_results
            ],
            "singular_values": [result[4].tolist() for result in all_pca_results],
            "trial_lengths": [
                result[5] for result in all_pca_results
            ],  # Store trial lengths
        }
    )

    return pca_results_df


pca_results_df = whitened_pca_all_nrns_all_taste(all_data_df)


def unconcatenate_pca_results_all(pca_results_df, dataset, changepoint):
    # Filter the DataFrame for the specific dataset, taste, and changepoint
    filtered_df = pca_results_df.filter(
        (pl.col("dataset") == dataset) & (pl.col("changepoint") == changepoint)
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


# unconcantenating PCA results outside of the main PCA function-- I can go either way on this one; designed to
# avoid dealing with odd data types in the polars dataframe
# this short function is called inside the other function


def plot_averaged_pca_trajectories(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract PCA results and trial lengths
                pca_result = np.array(result[2])
                trial_lengths = result[5]

                # Unconcatenate PCA results
                split_pca_result = unconcatenate_pca_results_all(
                    pca_result, trial_lengths
                )

                # Average each PCA result array along neurons
                averaged_pca_results = [
                    np.mean(trial_pca, axis=1) for trial_pca in split_pca_result
                ]

                # Create plot for each averaged PCA result
                fig, ax = plt.subplots(figsize=(12, 6))

                for i, averaged_pca in enumerate(averaged_pca_results):
                    ax.plot(averaged_pca, label=f"Trial {i+1}")

                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Averaged PCA Values")
                ax.set_title(
                    f"Averaged PCA Trajectories for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                ax.legend()

                plt.show()
            except Exception as e:
                print(
                    f"Could not plot PCA trajectories for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# modifying to plot along the first 3 principal components -- can modify further as needed
def plot_averaged_pca_trajectories(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract PCA results and trial lengths
                pca_result = np.array(result[2])
                trial_lengths = result[5]

                # Unconcatenate PCA results
                split_pca_result = unconcatenate_pca_results_all(
                    pca_results_df, dataset, changepoint
                )

                # Average the first 3 principal components for each trial
                averaged_pca_results = [
                    np.mean(trial_pca[:, :1], axis=1) for trial_pca in split_pca_result
                ]

                # Create plot for each averaged PCA result
                fig, ax = plt.subplots(figsize=(12, 6))

                for i, averaged_pca in enumerate(averaged_pca_results):
                    ax.plot(averaged_pca, label=f"Trial {i+1}")

                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Averaged PCA Values (First 3 Components)")
                ax.set_title(
                    f"Averaged PCA Trajectories for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                # ax.legend()

                plt.show()
            except Exception as e:
                print(
                    f"Could not plot PCA trajectories for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# try a trial averaged PCA -- see if we can see recognizable dynamcis that way first for the trial avg--
# single trial stuff for the data we're feeding in the first place -- first 3 compoents how bout
# then do the timescale-- go from there-- what we want to do is do PCA first and then average the PCA trajectories
# that's actually not that bad


# other less useful PCA stuff


# this is a 2-D plot
def plot_all_pca_scatter(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract PCA results
                pca_result = np.array(result[2])

                # Create scatter plot of the first two principal components
                plt.figure(figsize=(10, 6))
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title(
                    f"PCA Scatter Plot for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                plt.show()
            except Exception as e:
                print(
                    f"Could not plot PCA scatter for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# 2-D plot with labels for the various soruces of variance
def plot_all_pca_biplot(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract PCA results and singular values
                pca_result = np.array(result[2])
                singular_values = np.array(result[4])

                # Calculate loadings (coefficients of the principal components)
                loadings = (
                    pca_result * singular_values / np.sqrt(pca_result.shape[0] - 1)
                )

                # Create scatter plot of the first two principal components
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    pca_result[:, 0], pca_result[:, 1], alpha=0.7, label="Samples"
                )

                # Add loading vectors
                for i, vector in enumerate(loadings[:2].T):
                    plt.arrow(0, 0, vector[0], vector[1], color="r", alpha=0.5)
                    plt.text(
                        vector[0] * 1.2,
                        vector[1] * 1.2,
                        f"Var{i+1}",
                        color="r",
                        ha="center",
                        va="center",
                    )

                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title(
                    f"PCA Biplot for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                plt.legend()
                plt.show()
            except Exception as e:
                print(
                    f"Could not plot PCA biplot for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# 3-D pca plot-- interesting, but not sure how useful-- as it doesn't much elucidate things beyond


def plot_all_pca_scatter_3d(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract PCA results
                pca_result = np.array(result[2])

                # Create 3D scatter plot of the first three principal components
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.7
                )
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_zlabel("Principal Component 3")
                ax.set_title(
                    f"PCA 3D Scatter Plot for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                plt.show()
            except Exception as e:
                print(
                    f"Could not plot PCA 3D scatter for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# and with explaiened variance ratio-- perhaps this can help elucidate this some more:
# what this ultimately is showing is that the first 2-3 PC's account for a solid proportion of the explained variance
# but not all of it-- shows that the data is probably solidly non-linear
# this really what we want
# this is a good plot to start with for sure -- as it shows that PCA may not be the best reduction technique outright
def plot_explained_variance_combined(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract explained variance ratio
                explained_variance_ratio = np.array(result[3])

                # Principal components
                pcs = np.arange(len(explained_variance_ratio))

                # Create combined plot for the explained variance ratio per principal component
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Bar plot for explained variance ratio
                ax1.bar(pcs, explained_variance_ratio, color="b", alpha=0.6)
                ax1.set_xlabel("Principal Components")
                ax1.set_ylabel("Explained Variance Ratio", color="b")
                ax1.set_xticks(pcs)
                ax1.set_xticklabels([f"PC{i+1}" for i in pcs])

                # Line plot for cumulative explained variance ratio
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                ax2 = ax1.twinx()
                ax2.plot(
                    pcs,
                    cumulative_explained_variance,
                    label="Cumulative Explained Variance",
                    color="r",
                    marker="o",
                )
                ax2.set_ylabel("Cumulative Explained Variance Ratio", color="r")
                ax2.set_ylim(0, 1.1)
                ax2.legend(loc="upper left")

                plt.title(
                    f"Explained Variance Ratio per PC for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                plt.show()
            except Exception as e:
                print(
                    f"Could not plot explained variance ratio for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# scaled so that the y axis for both plots is on the same scale
def plot_explained_variance_combined(pca_results_df):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for changepoint in unique_changepoints:
            try:
                # Filter the PCA results for the given dataset and changepoint
                result = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("changepoint") == changepoint)
                ).row(0)

                # Extract explained variance ratio
                explained_variance_ratio = np.array(result[3])

                # Principal components
                pcs = np.arange(len(explained_variance_ratio))

                # Create combined plot for the explained variance ratio per principal component
                fig, ax1 = plt.subplots(figsize=(12, 8))

                # Bar plot for explained variance ratio
                ax1.bar(
                    pcs,
                    explained_variance_ratio,
                    color="b",
                    alpha=0.6,
                    label="Explained Variance Ratio",
                )
                ax1.set_xlabel("Principal Components")
                ax1.set_ylabel("Explained Variance Ratio")
                ax1.set_xticks(pcs)
                ax1.set_xticklabels([f"PC{i+1}" for i in pcs])

                # Line plot for cumulative explained variance ratio
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                ax1.plot(
                    pcs,
                    cumulative_explained_variance,
                    label="Cumulative Explained Variance",
                    color="r",
                    marker="o",
                )
                ax1.set_ylabel("Variance Ratio")
                ax1.set_ylim(0, 1.1)
                ax1.legend(loc="upper left")

                plt.title(
                    f"Explained Variance Ratio per PC for Dataset: {dataset}, Changepoint: {changepoint}"
                )
                plt.show()
            except Exception as e:
                print(
                    f"Could not plot explained variance ratio for dataset {dataset}, changepoint {changepoint}: {e}"
                )


# might be for a slightly deprecated function at this point
