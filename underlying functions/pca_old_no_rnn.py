#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:42:43 2025

PCA figs, without having run RNN-- also some PCA processing code


@author: vincentcalia-bogan
"""
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

## PCA THINGS NOW ##
# function that does PCA transform on the *FIRING RATE* that we've inferred -- from "all data df"
## now separated according to taste: #### for all nrns
pca_output_dir = "/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/pca_first_8_comp"


def whitened_pca_all_nrns_sep_taste(all_data_df, pca_output_dir):
    all_pca_results = []
    unique_datasets = all_data_df["dataset"].unique().to_list()
    unique_tastes = all_data_df["taste"].unique().to_list()

    for dataset in unique_datasets:
        dataset_df = all_data_df.filter(pl.col("dataset") == dataset)

        for taste in unique_tastes:
            taste_df = dataset_df.filter(pl.col("taste") == taste)
            unique_changepoints = taste_df["changepoint"].unique().to_list()
            unique_neurons = taste_df["neuron"].unique().to_list()

            for changepoint in unique_changepoints:
                concatenated_data_list = []
                trial_lengths = []  # To store the original lengths of each trial

                for neuron in unique_neurons:
                    neuron_df = taste_df.filter(
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
                            taste,
                            changepoint,
                            pca_result,
                            explained_variance,
                            singular_values,
                            trial_lengths,
                        )
                    )
                else:
                    print(
                        f"Skipping PCA for dataset {dataset}, taste {taste}, changepoint {changepoint} due to lack of variation."
                    )

    # Convert results to a Polars DataFrame
    pca_results_df = pl.DataFrame(
        {
            "dataset": [result[0] for result in all_pca_results],
            "taste": [result[1] for result in all_pca_results],
            "changepoint": [result[2] for result in all_pca_results],
            "pca_result": [result[3].tolist() for result in all_pca_results],
            "explained_variance_ratio": [
                result[4].tolist() for result in all_pca_results
            ],
            "singular_values": [result[5].tolist() for result in all_pca_results],
            "trial_lengths": [
                result[6] for result in all_pca_results
            ],  # Store trial lengths
        }
    )
    os.makedirs(pca_output_dir, exist_ok=True)
    file_path = os.path.join(pca_output_dir, "pca_results.parquet")
    pca_results_df.write_parquet(file_path)
    print(f"PCA results saved to {file_path}")
    return pca_results_df


pca_results_df = whitened_pca_all_nrns_sep_taste(all_data_df, pca_output_dir)
# function designed for other functions that splits up the PCA results


def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
    # Filter the DataFrame for the specific dataset, taste, and changepoint
    filtered_df = pca_results_df.filter(
        (pl.col("dataset") == dataset)
        & (pl.col("taste") == taste)
        & (pl.col("changepoint") == changepoint)
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


# post splitting by cp, plotting:


# testing a function that plots the first 8 PCs


def plot_pca_components_by_trial(pca_results_df, pca_output_dir):
    def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
        # Filter the DataFrame for the specific dataset, taste, and changepoint
        filtered_df = pca_results_df.filter(
            (pl.col("dataset") == dataset)
            & (pl.col("taste") == taste)
            & (pl.col("changepoint") == changepoint)
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

    # Extract unique datasets, tastes, and changepoints from the DataFrame
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    # Iterate over datasets, tastes, and trials
    for dataset in unique_datasets:
        dataset_output_dir = os.path.join(pca_output_dir, f"{dataset}_pca_aligned")
        os.makedirs(dataset_output_dir, exist_ok=True)

        for taste in unique_tastes:
            concatenated_trials = []

            # Collect data across changepoints for each trial
            for changepoint in unique_changepoints:
                split_pca_results = unconcatenate_pca_results_taste(
                    pca_results_df, dataset, taste, changepoint
                )

                # For each trial, concatenate epochs
                for trial_idx, trial_data in enumerate(split_pca_results):
                    if len(concatenated_trials) <= trial_idx:
                        concatenated_trials.append([])
                    concatenated_trials[trial_idx].append(trial_data)

            # Concatenate epochs within each trial and standardize trial lengths
            final_trial_data = []
            max_trial_length = 0
            for trial_data in concatenated_trials:
                concatenated_data = np.concatenate(trial_data, axis=0)
                max_trial_length = max(max_trial_length, concatenated_data.shape[0])
                final_trial_data.append(concatenated_data)

            # Adjust all trials to the same length (trimming or padding as needed)
            for idx, trial_data in enumerate(final_trial_data):
                if trial_data.shape[0] < max_trial_length:
                    # Pad with NaNs or some appropriate method
                    padding = np.full(
                        (max_trial_length - trial_data.shape[0], trial_data.shape[1]),
                        np.nan,
                    )
                    final_trial_data[idx] = np.concatenate(
                        (trial_data, padding), axis=0
                    )
                elif trial_data.shape[0] > max_trial_length:
                    # Trim to match the max length
                    final_trial_data[idx] = trial_data[:max_trial_length, :]

            # Plot the first 8 principal components for each trial
            for trial_idx, trial_data in enumerate(final_trial_data):
                plt.figure(figsize=(12, 6))
                for pc_idx in range(8):  # Plot the first 8 components
                    plt.plot(trial_data[:, pc_idx], label=f"PC {pc_idx + 1}")

                plt.title(
                    f"PCA Components for {dataset}, Taste: {taste}, Trial {trial_idx + 1}"
                )
                plt.xlabel("Time Bins")
                plt.ylabel("PCA Value")
                plt.legend(loc="upper right", fontsize="small")

                # Save the figure
                plt.savefig(
                    os.path.join(
                        dataset_output_dir, f"{taste}_trial_{trial_idx + 1}_pca.png"
                    )
                )
                plt.close()


# attempting to align PCA components by warping: (experimental)


## EXPERIMENTAL: TRYING TO ALIGN THE ARRAYS TO A COMMON SIZE/LEN VIA INTERPOLATION SO I CAN AVERAGE DOWN TASTE
# idk about this one tbh
def plot_pca_components_averaged_over_tastes(pca_results_df, pca_output_dir):
    def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
        # Filter the DataFrame for the specific dataset, taste, and changepoint
        filtered_df = pca_results_df.filter(
            (pl.col("dataset") == dataset)
            & (pl.col("taste") == taste)
            & (pl.col("changepoint") == changepoint)
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

    def interpolate_to_length(data, target_length):
        # Interpolate data to the target length
        original_length = data.shape[0]
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolated_data = np.zeros((target_length, data.shape[1]))

        for i in range(
            data.shape[1]
        ):  # Interpolate each principal component separately
            f = interp1d(
                x_original, data[:, i], kind="linear", fill_value="extrapolate"
            )
            interpolated_data[:, i] = f(x_target)

        return interpolated_data

    # Extract unique datasets, changepoints, and tastes
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    # Iterate over datasets
    for dataset in unique_datasets:
        dataset_trials = []

        # Collect data across changepoints and tastes for averaging
        for changepoint in unique_changepoints:
            # Retrieve split PCA results for all tastes
            taste_pca_data = []
            for taste in unique_tastes:
                split_pca_results = unconcatenate_pca_results_taste(
                    pca_results_df, dataset, taste, changepoint
                )
                taste_pca_data.append(split_pca_results)

            # Average PCA results over tastes for each trial
            num_trials = len(
                taste_pca_data[0]
            )  # Assuming all tastes have the same number of trials
            max_trial_length = max(
                max(len(trial) for trial in trials) for trials in taste_pca_data
            )
            averaged_trials = []

            for trial_idx in range(num_trials):
                # Interpolate each trial for all tastes to the maximum length
                interpolated_tastes = [
                    interpolate_to_length(
                        taste_pca_data[t][trial_idx], max_trial_length
                    )
                    for t in range(len(taste_pca_data))
                ]

                # Stack interpolated taste arrays and calculate the mean across tastes
                taste_stacks = np.stack(interpolated_tastes, axis=0)
                averaged_trial = np.nanmean(taste_stacks, axis=0)
                averaged_trials.append(averaged_trial)

            dataset_trials.append(averaged_trials)

        # Concatenate averaged results across changepoints for each trial
        final_trial_data = []
        max_trial_length = 0
        for trial_idx in range(
            len(dataset_trials[0])
        ):  # Assuming consistent number of trials across changepoints
            concatenated_data = np.concatenate(
                [
                    dataset_trials[cp_idx][trial_idx]
                    for cp_idx in range(len(unique_changepoints))
                ],
                axis=0,
            )
            max_trial_length = max(max_trial_length, concatenated_data.shape[0])
            final_trial_data.append(concatenated_data)

        # Create a directory for the dataset
        dataset_output_dir = os.path.join(pca_output_dir, f"{dataset}_pca_aligned")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Plot the first 8 principal components for each trial
        for trial_idx, trial_data in enumerate(final_trial_data):
            plt.figure(figsize=(12, 6))
            for pc_idx in range(8):  # Plot the first 8 components
                plt.plot(trial_data[:, pc_idx], label=f"PC {pc_idx + 1}")

            plt.title(
                f"PCA Components Averaged Over Tastes for {dataset}, Trial {trial_idx + 1}"
            )
            plt.xlabel("Time Bins")
            plt.ylabel("PCA Value")
            plt.legend(loc="upper right", fontsize="small")

            # Save the figure
            plt.savefig(
                os.path.join(dataset_output_dir, f"trial_{trial_idx + 1}_pca.png")
            )
            plt.close()


def plot_pca_components_averaged_over_tastes(pca_results_df, pca_output_dir):
    def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
        # Filter the DataFrame for the specific dataset, taste, and changepoint
        filtered_df = pca_results_df.filter(
            (pl.col("dataset") == dataset)
            & (pl.col("taste") == taste)
            & (pl.col("changepoint") == changepoint)
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

    def interpolate_to_length(data, target_length):
        # Interpolate data to the target length
        original_length = data.shape[0]
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolated_data = np.zeros((target_length, data.shape[1]))

        for i in range(
            data.shape[1]
        ):  # Interpolate each principal component separately
            f = interp1d(
                x_original, data[:, i], kind="linear", fill_value="extrapolate"
            )
            interpolated_data[:, i] = f(x_target)

        return interpolated_data

    # Extract unique datasets, changepoints, and tastes
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    # Iterate over datasets
    for dataset in unique_datasets:
        dataset_trials = []

        # Collect data across changepoints and tastes for averaging
        for changepoint in unique_changepoints:
            # Retrieve split PCA results for all tastes
            taste_pca_data = []
            for taste in unique_tastes:
                split_pca_results = unconcatenate_pca_results_taste(
                    pca_results_df, dataset, taste, changepoint
                )
                taste_pca_data.append(split_pca_results)

            # Average PCA results over tastes for each trial
            num_trials = len(
                taste_pca_data[0]
            )  # Assuming all tastes have the same number of trials
            max_trial_length = max(
                max(len(trial) for trial in trials) for trials in taste_pca_data
            )
            averaged_trials = []

            for trial_idx in range(num_trials):
                # Interpolate each trial for all tastes to the maximum length
                interpolated_tastes = []
                for t in range(len(taste_pca_data)):
                    if (
                        len(taste_pca_data[t][trial_idx]) > 0
                    ):  # Ensure the trial is non-empty
                        interpolated_tastes.append(
                            interpolate_to_length(
                                taste_pca_data[t][trial_idx], max_trial_length
                            )
                        )

                # Check if we have any valid interpolated arrays before proceeding
                if interpolated_tastes:
                    # Stack interpolated taste arrays and calculate the mean across tastes
                    try:
                        taste_stacks = np.stack(interpolated_tastes, axis=0)
                        averaged_trial = np.nanmean(taste_stacks, axis=0)
                        averaged_trials.append(averaged_trial)
                    except ValueError as e:
                        print(f"Skipping trial {trial_idx} due to shape mismatch: {e}")
                        continue

            dataset_trials.append(averaged_trials)

        # Concatenate averaged results across changepoints for each trial
        final_trial_data = []
        max_trial_length = 0
        for trial_idx in range(
            len(dataset_trials[0])
        ):  # Assuming consistent number of trials across changepoints
            concatenated_data = np.concatenate(
                [
                    dataset_trials[cp_idx][trial_idx]
                    for cp_idx in range(len(unique_changepoints))
                ],
                axis=0,
            )
            max_trial_length = max(max_trial_length, concatenated_data.shape[0])
            final_trial_data.append(concatenated_data)

        # Create a directory for the dataset
        dataset_output_dir = os.path.join(pca_output_dir, f"{dataset}_pca_aligned")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Plot the first 8 principal components for each trial
        for trial_idx, trial_data in enumerate(final_trial_data):
            plt.figure(figsize=(10, 5))
            for pc_idx in range(8):  # Plot the first 8 components
                plt.plot(trial_data[:, pc_idx], label=f"PC {pc_idx + 1}")

            plt.title(
                f"PCA Components Averaged Over Tastes for {dataset}, Trial {trial_idx + 1}"
            )
            plt.xlabel("Time Bins")
            plt.ylabel("PCA Value")
            plt.legend(loc="upper right", fontsize="small")

            # Save the figure
            plt.savefig(
                os.path.join(dataset_output_dir, f"trial_{trial_idx + 1}_pca.png")
            )
            plt.close()


######### STILL PCA BUT DIFFERENT NOW #########
### testing with a heatmap-- this is a big, ugly piece of code that I'll fix later:
def plot_pca_components_with_heatmap(pca_results_df, pca_output_dir):
    def unconcatenate_pca_results_taste(pca_results_df, dataset, taste, changepoint):
        # Filter the DataFrame for the specific dataset, taste, and changepoint
        filtered_df = pca_results_df.filter(
            (pl.col("dataset") == dataset)
            & (pl.col("taste") == taste)
            & (pl.col("changepoint") == changepoint)
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

    def interpolate_to_length(data, target_length):
        # Interpolate data to the target length
        original_length = data.shape[0]
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolated_data = np.zeros((target_length, data.shape[1]))

        for i in range(
            data.shape[1]
        ):  # Interpolate each principal component separately
            f = interp1d(
                x_original, data[:, i], kind="linear", fill_value="extrapolate"
            )
            interpolated_data[:, i] = f(x_target)

        return interpolated_data

    # Extract unique datasets, changepoints, and tastes
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    # Iterate over datasets
    for dataset in unique_datasets:
        dataset_trials = []

        # Collect data across changepoints and tastes for averaging
        for changepoint in unique_changepoints:
            # Retrieve split PCA results for all tastes
            taste_pca_data = []
            for taste in unique_tastes:
                split_pca_results = unconcatenate_pca_results_taste(
                    pca_results_df, dataset, taste, changepoint
                )
                taste_pca_data.append(split_pca_results)

            # Average PCA results over tastes for each trial
            num_trials = len(
                taste_pca_data[0]
            )  # Assuming all tastes have the same number of trials
            max_trial_length = max(
                max(len(trial) for trial in trials) for trials in taste_pca_data
            )
            averaged_trials = []

            for trial_idx in range(num_trials):
                # Interpolate each trial for all tastes to the maximum length
                interpolated_tastes = []
                for t in range(len(taste_pca_data)):
                    if (
                        len(taste_pca_data[t][trial_idx]) > 0
                    ):  # Ensure the trial is non-empty
                        interpolated_tastes.append(
                            interpolate_to_length(
                                taste_pca_data[t][trial_idx], max_trial_length
                            )
                        )

                # Check if we have any valid interpolated arrays before proceeding
                if interpolated_tastes:
                    try:
                        taste_stacks = np.stack(interpolated_tastes, axis=0)
                        averaged_trial = np.nanmean(taste_stacks, axis=0)
                        averaged_trials.append(averaged_trial)
                    except ValueError as e:
                        print(f"Skipping trial {trial_idx} due to shape mismatch: {e}")
                        continue

            dataset_trials.append(averaged_trials)

        # Concatenate averaged results across changepoints for each trial
        final_trial_data = []
        max_trial_length = 0
        for trial_idx in range(
            len(dataset_trials[0])
        ):  # Assuming consistent number of trials across changepoints
            concatenated_data = np.concatenate(
                [
                    dataset_trials[cp_idx][trial_idx]
                    for cp_idx in range(len(unique_changepoints))
                ],
                axis=0,
            )
            max_trial_length = max(max_trial_length, concatenated_data.shape[0])
            final_trial_data.append(concatenated_data)

        # Create a directory for the dataset
        dataset_output_dir = os.path.join(pca_output_dir, f"{dataset}_pca_aligned")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Plot the first 8 principal components for each trial as line plots and heatmaps
        for trial_idx, trial_data in enumerate(final_trial_data):
            # Line plot
            plt.figure(figsize=(10, 5))
            for pc_idx in range(8):  # Plot the first 8 components
                plt.plot(trial_data[:, pc_idx], label=f"PC {pc_idx + 1}")

            plt.title(
                f"PCA Components Averaged Over Tastes for {dataset}, Trial {trial_idx + 1}"
            )
            plt.xlabel("Time Bins")
            plt.ylabel("PCA Value")
            plt.legend(loc="upper right", fontsize="small")

            # Save the line plot
            plt.savefig(
                os.path.join(dataset_output_dir, f"trial_{trial_idx + 1}_pca.png")
            )
            plt.close()

            # Heatmap plot
            plt.figure(figsize=(10, 5))
            plt.imshow(
                trial_data[:, :8].T,
                aspect="auto",
                cmap="coolwarm",
                interpolation="none",
            )
            plt.colorbar(label="PCA Value")
            plt.title(
                f"PCA Heatmap Averaged Over Tastes for {dataset}, Trial {trial_idx + 1}"
            )
            plt.xlabel("Time Bins")
            plt.ylabel("Principal Components")
            plt.yticks(ticks=np.arange(8), labels=[f"PC {i + 1}" for i in range(8)])

            # Save the heatmap
            plt.savefig(
                os.path.join(
                    dataset_output_dir, f"trial_{trial_idx + 1}_pca_heatmap.png"
                )
            )
            plt.close()
