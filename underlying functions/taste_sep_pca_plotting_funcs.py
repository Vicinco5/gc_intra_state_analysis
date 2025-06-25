#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:39:15 2024

@author: vincentcalia-bogan

This file is for the collection of all the PCA plotting functions used
"""
# this is a relatively useless function
# compared to the rest

import os, os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import polars as pl
from scipy.spatial.distance import euclidean

## PLOTTING FUNCS FOR THE TASTE SEPARATED PCA ##


def plot_averaged_pca_trajectories_sep_taste(pca_results_df):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                try:
                    # Filter the PCA results for the given dataset, taste, and changepoint
                    result = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    ).row(0)

                    # Extract PCA results and trial lengths
                    pca_result = np.array(result[3])
                    trial_lengths = result[6]

                    # Unconcatenate PCA results
                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    # Average the first 3 principal components for each trial-- in this case just one
                    averaged_pca_results = [
                        np.mean(trial_pca[:, :1], axis=1)
                        for trial_pca in split_pca_result
                    ]

                    # Create plot for each averaged PCA result
                    fig, ax = plt.subplots(figsize=(12, 6))

                    for i, averaged_pca in enumerate(averaged_pca_results):
                        ax.plot(averaged_pca, label=f"Trial {i+1}")

                    ax.set_xlabel("Time Bins")
                    ax.set_ylabel("Averaged PCA Values (First 3 Components)")
                    ax.set_title(
                        f"Averaged PCA Trajectories for Dataset: {dataset}, Taste: {taste}, Changepoint: {changepoint}"
                    )
                    # ax.legend()

                    plt.show()
                except Exception as e:
                    print(
                        f"Could not plot PCA trajectories for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )


# more combined figure:
# this is actually very useful this one
# not actually certain that this figure is strictly useful
# gotta go back and revisit what PCA is really useful for after all this
# this is without the SEM stuff-  this has got all sorts of colors to it
def plot_averaged_pca_trajectories_thresholded(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()
    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes
    num_pc_avg = 1

    # Create one folder named for the number of principal components being averaged
    output_dir = os.path.join(pca_output_dir, f"averaged_{num_pc_avg}_pc_line")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"PCA Trajectories, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)
                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("Time (ms)")
                    if cp_idx == 0:
                        ax.set_ylabel(f"PC {num_pc_avg} Values")

                    # Averaging the specified number of principal components
                    averaged_pca_results = [
                        np.mean(trial_pca[:, :num_pc_avg], axis=1)
                        for trial_pca in split_pca_result
                    ]

                    trial_lengths = []
                    trial_data = []

                    for trial_idx, averaged_pca in enumerate(averaged_pca_results):
                        trial_color = cm.get_cmap("rainbow", len(averaged_pca_results))(
                            trial_idx
                        )
                        num_time_bins = averaged_pca.shape[0]
                        x_ax = np.arange(num_time_bins) * step_size
                        ax.plot(x_ax, averaged_pca, color=trial_color, alpha=0.5)

                        trial_lengths.append(num_time_bins)
                        trial_data.append(averaged_pca)

                    # Calculate the mean at each time point across all trials
                    max_time_bins = max(trial_lengths)
                    all_data = np.full((num_trials, max_time_bins), np.nan)
                    for i, data in enumerate(trial_data):
                        all_data[i, : len(data)] = data

                    mean_data = np.nanmean(all_data, axis=0)
                    mean_time_bins = np.arange(max_time_bins) * step_size
                    ax.plot(
                        mean_time_bins,
                        mean_data,
                        color="black",
                        linewidth=1,
                        label="Trial Average",
                    )
                    ax.legend()

                    sorted_lengths = sorted(trial_lengths)
                    avg_duration = np.mean(sorted_lengths) * step_size

                    trial_lengths = np.array(sorted_lengths)
                    duration_percentiles = np.percentile(
                        trial_lengths, [0, 25, 50, 75, 90]
                    )

                    labels = ["0th", "25th", "50th", "75th", "90th"]
                    for i, percentile in enumerate(duration_percentiles):
                        ax.axvline(
                            percentile * step_size,
                            linestyle=":",
                            color="grey",
                            alpha=0.6,
                        )
                        ax.text(
                            percentile * step_size,
                            ax.get_ylim()[1] * 0.9,
                            f"{labels[i]} percentile",
                            rotation=90,
                            verticalalignment="top",
                            color="grey",
                            alpha=1,
                        )

                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                    )

                except Exception as e:
                    print(
                        f"Could not plot PCA trajectories for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )
        plt.tight_layout()
        neuron_fig_path = os.path.join(output_dir, f"{dataset}_pca_trajectories.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


## SAME BUT WITH SEM


def plot_averaged_pca_trajectories_thresholded_SEM(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()
    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes
    num_pc_avg = 1

    # Create one folder named for the number of principal components being averaged
    output_dir = os.path.join(pca_output_dir, f"averaged_{num_pc_avg}_pc_line_SEM")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"PCA Trajectories, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)
                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("Time (ms)")
                    if cp_idx == 0:
                        ax.set_ylabel(f"PC {num_pc_avg} Values")

                    # Averaging the specified number of principal components
                    averaged_pca_results = [
                        np.mean(trial_pca[:, :num_pc_avg], axis=1)
                        for trial_pca in split_pca_result
                    ]

                    trial_lengths = []
                    trial_data = []

                    for trial_idx, averaged_pca in enumerate(averaged_pca_results):
                        num_time_bins = averaged_pca.shape[0]
                        trial_lengths.append(num_time_bins)
                        trial_data.append(averaged_pca)

                    # Calculate the mean and SEM at each time point across all trials
                    max_time_bins = max(trial_lengths)
                    all_data = np.full((num_trials, max_time_bins), np.nan)
                    for i, data in enumerate(trial_data):
                        all_data[i, : len(data)] = data

                    mean_data = np.nanmean(all_data, axis=0)
                    sem_data = np.nanstd(all_data, axis=0) / np.sqrt(
                        np.sum(~np.isnan(all_data), axis=0)
                    )
                    mean_time_bins = np.arange(max_time_bins) * step_size
                    ax.plot(
                        mean_time_bins,
                        mean_data,
                        color="black",
                        linewidth=1,
                        label="Trial Average",
                    )
                    ax.fill_between(
                        mean_time_bins,
                        mean_data - sem_data,
                        mean_data + sem_data,
                        color="gray",
                        alpha=0.5,
                        label="SEM",
                    )
                    ax.legend()

                    avg_duration = np.mean(trial_lengths) * step_size

                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                    )

                except Exception as e:
                    print(
                        f"Could not plot PCA trajectories for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )
        plt.tight_layout()
        neuron_fig_path = os.path.join(output_dir, f"{dataset}_pca_trajectories.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


def plot_averaged_pca_heatmaps_all_taste(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes
    num_pc_avg = 1

    # Create one folder named for the number of principal components being averaged
    output_dir = os.path.join(pca_output_dir, f"averaged_{num_pc_avg}_pc_heatmap")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"PCA Heatmaps, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)

                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    # trial_lengths = [step_size * x for x in trial_length]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("Time (ms)")
                    if cp_idx == 0:
                        ax.set_ylabel(f"Averaged PC {num_pc_avg} Values")

                    # Averaging the specified number of principal components
                    averaged_pca_results = [
                        np.mean(trial_pca[:, :num_pc_avg], axis=1)
                        for trial_pca in split_pca_result
                    ]

                    sorted_indices = np.argsort(trial_lengths)
                    sorted_averaged_pca_results = [
                        averaged_pca_results[i] for i in sorted_indices
                    ]

                    # Pad the trials to the same length
                    max_length = max(
                        len(trial) for trial in sorted_averaged_pca_results
                    )
                    padded_pca_results = [
                        np.pad(
                            trial,
                            (0, max_length - len(trial)),
                            "constant",
                            constant_values=np.nan,
                        )
                        for trial in sorted_averaged_pca_results
                    ]

                    # Create a 2D array for the heatmap
                    heatmap_data = np.vstack(padded_pca_results)

                    # Plot the heatmap
                    cax = ax.imshow(
                        heatmap_data,
                        aspect="auto",
                        cmap="viridis",
                        interpolation="nearest",
                    )
                    fig.colorbar(cax, ax=ax)

                    sorted_lengths = sorted(trial_lengths)
                    avg_duration = np.mean(sorted_lengths) * step_size

                    trial_lengths = np.array(sorted_lengths)
                    duration_percentiles = np.percentile(
                        trial_lengths, [0, 25, 50, 75, 90]
                    )

                    labels = ["0th", "25th", "50th", "75th", "90th"]
                    for i, percentile in enumerate(duration_percentiles):
                        ax.axvline(percentile, linestyle=":", color="grey", alpha=0.6)
                        ax.text(
                            percentile,
                            ax.get_ylim()[1] * 0.9,
                            f"{labels[i]} percentile",
                            rotation=90,
                            verticalalignment="top",
                            color="grey",
                            alpha=1,
                        )

                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                    )

                except Exception as e:
                    print(
                        f"Could not plot PCA heatmaps for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )

        plt.tight_layout()
        neuron_fig_path = os.path.join(output_dir, f"{dataset}_pca_heatmaps.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


def plot_pca_scatter(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes

    # Create one folder for the scatter plots
    output_dir = os.path.join(pca_output_dir, "pca_scatter")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"PCA Scatter Plots, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)

                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("PC 1")
                    if cp_idx == 0:
                        ax.set_ylabel("PC 2")

                    colormap = cm.get_cmap("rainbow", num_trials)
                    for trial_idx, trial_pca in enumerate(split_pca_result):
                        trial_color = colormap(trial_idx)
                        ax.scatter(
                            trial_pca[:, 0],
                            trial_pca[:, 1],
                            color=trial_color,
                            alpha=0.5,
                            label=f"Trial {trial_idx + 1}",
                        )

                    if taste_idx == num_tastes - 1 and cp_idx == num_changepoints - 1:
                        ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1))

                except Exception as e:
                    print(
                        f"Could not plot PCA scatter plots for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )

        plt.tight_layout()
        scatter_fig_path = os.path.join(output_dir, f"{dataset}_pca_scatter.png")
        plt.savefig(scatter_fig_path)
        plt.close(fig)


def plot_pca_scatter_3d(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
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

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes

    # Create one folder for the scatter plots
    output_dir = os.path.join(pca_output_dir, "pca_scatter_3d")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig = plt.figure(figsize=(15, 5 * num_tastes))
        fig.suptitle(f"PCA 3D Scatter Plots, Dataset: {dataset}", fontsize=14, y=0.995)
        axes = [
            [
                fig.add_subplot(
                    num_tastes,
                    num_changepoints,
                    i * num_changepoints + j + 1,
                    projection="3d",
                )
                for j in range(num_changepoints)
            ]
            for i in range(num_tastes)
        ]

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    result = filtered_df.row(0)

                    pca_result = np.array(result[3])
                    trial_lengths = result[6]
                    num_trials = len(trial_lengths)

                    split_pca_result = unconcatenate_pca_results_taste(
                        pca_results_df, dataset, taste, changepoint
                    )

                    ax = axes[taste_idx][cp_idx]
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel("PC 2")
                    ax.set_zlabel("PC 3")

                    colormap = cm.get_cmap("rainbow", num_trials)
                    for trial_idx, trial_pca in enumerate(split_pca_result):
                        trial_color = colormap(trial_idx)
                        ax.scatter(
                            trial_pca[:, 0],
                            trial_pca[:, 1],
                            trial_pca[:, 2],
                            color=trial_color,
                            alpha=0.5,
                            label=f"Trial {trial_idx + 1}",
                        )
                        ax.plot(
                            trial_pca[:, 0],
                            trial_pca[:, 1],
                            trial_pca[:, 2],
                            color=trial_color,
                            alpha=0.5,
                        )

                    if taste_idx == num_tastes - 1 and cp_idx == num_changepoints - 1:
                        ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1))

                except Exception as e:
                    print(
                        f"Could not plot PCA 3D scatter plots for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )

        plt.tight_layout()
        scatter_fig_path = os.path.join(output_dir, f"{dataset}_pca_scatter_3d.png")
        plt.savefig(scatter_fig_path)
        plt.close(fig)


# euchlidean distances now: -- want to see if whatever dynamic behavior PCA is able to isolate is being
# retained trial to trial; tryna measure this with euchlidean distance
def calculate_pca_distances(pca_results_df):
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

    data = {
        "dataset": [],
        "taste": [],
        "changepoint": [],
        "num_datapoints": [],
        "euclidean_distances": [],
        "mean_distance": [],
        "std_distance": [],
        "var_distance": [],
        "coords": [],
    }

    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    for dataset in unique_datasets:
        for taste in unique_tastes:
            for changepoint in unique_changepoints:
                filtered_df = pca_results_df.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("taste") == taste)
                    & (pl.col("changepoint") == changepoint)
                )
                if filtered_df.shape[0] == 0:
                    continue

                result = filtered_df.row(0)
                pca_result = np.array(result[3])
                split_pca_result = unconcatenate_pca_results_taste(
                    pca_results_df, dataset, taste, changepoint
                )

                for trial_pca in split_pca_result:
                    num_points = trial_pca.shape[0]
                    distances = []

                    if num_points > 1:
                        for i in range(num_points - 1):
                            distances.append(
                                euclidean(trial_pca[i, :3], trial_pca[i + 1, :3])
                            )

                        mean_distance = np.mean(distances)
                        std_distance = np.std(distances)
                        var_distance = np.var(distances)
                    else:
                        distances = [np.nan]
                        mean_distance = std_distance = var_distance = np.nan

                    data["dataset"].append(dataset)
                    data["taste"].append(taste)
                    data["changepoint"].append(changepoint)
                    data["num_datapoints"].append(num_points)
                    data["euclidean_distances"].append(distances)
                    data["mean_distance"].append(mean_distance)
                    data["std_distance"].append(std_distance)
                    data["var_distance"].append(var_distance)
                    data["coords"].append(trial_pca[:, :3].tolist())

    distance_df = pl.DataFrame(data)
    return distance_df


# now plotting a bar plot of the euclidean distances for trials-- hoping to see relatively tight characterization
def plot_euclidean_distances_bar(
    distance_df, pca_output_dir, modified_tastes, epoch_labels, step_size
):
    distance_df = calculate_pca_distances(pca_results_df)
    unique_datasets = distance_df["dataset"].unique().to_list()
    unique_tastes = distance_df["taste"].unique().to_list()
    unique_changepoints = distance_df["changepoint"].unique().to_list()

    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes

    # Create one folder for the bar plots
    bar_output_dir = os.path.join(pca_output_dir, "euclidean_distances_bar")
    os.makedirs(bar_output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(
            f"Euclidean Distances (dist = 0 counts single points only), Dataset: {dataset}",
            fontsize=14,
            y=0.995,
        )

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    filtered_df = distance_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    euclidean_distances_list = filtered_df[
                        "euclidean_distances"
                    ].to_list()

                    # Flatten the list of lists
                    euclidean_distances = [
                        item for sublist in euclidean_distances_list for item in sublist
                    ]

                    # Treat NaN values as 0 for plotting
                    distances = np.nan_to_num(euclidean_distances)

                    # Calculate mean and standard deviation ignoring NaN values
                    valid_distances = np.array(
                        [x for x in euclidean_distances if not np.isnan(x)]
                    )
                    mean_distance = (
                        np.mean(valid_distances) if valid_distances.size > 0 else np.nan
                    )
                    std_distance = (
                        np.std(valid_distances) if valid_distances.size > 0 else np.nan
                    )

                    ax = axes[taste_idx, cp_idx]
                    ax.hist(distances, bins=30, color="blue", alpha=0.7, rwidth=0.85)
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nMean: {mean_distance:.2f}, Std: {std_distance:.2f}"
                    )
                    ax.set_xlabel("Euclidean Distance")
                    ax.set_ylabel("Count")

                except Exception as e:
                    print(
                        f"Could not plot Euclidean distances for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )

        plt.tight_layout()
        bar_fig_path = os.path.join(
            bar_output_dir, f"{dataset}_euclidean_distances_bar.png"
        )
        plt.savefig(bar_fig_path)
        plt.close(fig)


# now explained vairance for each PC for each segmentation of stuff- this will likely tell if this
# makes or breaks the whole ass analysis
# if we see poor explained variance then that's the story there bub
# should have done this earlier
def plot_explained_variance_combined_by_taste(
    pca_results_df, pca_output_dir, modified_tastes, epoch_labels
):
    unique_datasets = pca_results_df["dataset"].unique().to_list()
    unique_tastes = pca_results_df["taste"].unique().to_list()
    unique_changepoints = pca_results_df["changepoint"].unique().to_list()

    num_tastes = len(unique_tastes)
    num_changepoints = len(unique_changepoints)
    taste_labels = modified_tastes

    # Create one folder for the explained variance plots
    explained_variance_output_dir = os.path.join(pca_output_dir, "explained_variance")
    os.makedirs(explained_variance_output_dir, exist_ok=True)

    for dataset in unique_datasets:
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(f"Explained Variance, Dataset: {dataset}", fontsize=14, y=0.995)

        for taste_idx, taste in enumerate(unique_tastes):
            for cp_idx, changepoint in enumerate(unique_changepoints):
                try:
                    # Filter the PCA results for the given dataset, taste, and changepoint
                    filtered_df = pca_results_df.filter(
                        (pl.col("dataset") == dataset)
                        & (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )
                    if filtered_df.shape[0] == 0:
                        print(
                            f"No data for dataset {dataset}, taste {taste}, changepoint {changepoint}"
                        )
                        continue

                    explained_variance_ratio = filtered_df["explained_variance_ratio"][
                        0
                    ].to_numpy()

                    # Principal components
                    pcs = np.arange(len(explained_variance_ratio))

                    ax = axes[taste_idx, cp_idx]
                    ax.bar(
                        pcs,
                        explained_variance_ratio,
                        color="b",
                        alpha=0.6,
                        label="Explained Variance Ratio",
                    )
                    ax.set_xlabel("Principal Components")
                    ax.set_ylabel("Explained Variance Ratio")
                    ax.set_xticks(pcs)
                    ax.set_xticklabels([f"PC{i+1}" for i in pcs])

                    # Line plot for cumulative explained variance ratio
                    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                    ax.plot(
                        pcs,
                        cumulative_explained_variance,
                        label="Cumulative Explained Variance",
                        color="r",
                        marker="o",
                    )
                    ax.set_ylabel("Variance Ratio")
                    ax.set_ylim(0, 1.1)
                    ax.legend(loc="upper left")

                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                    )

                except Exception as e:
                    print(
                        f"Could not plot explained variance ratio for dataset {dataset}, taste {taste}, changepoint {changepoint}: {e}"
                    )

        plt.tight_layout()
        explained_variance_fig_path = os.path.join(
            explained_variance_output_dir, f"{dataset}_explained_variance.png"
        )
        plt.savefig(explained_variance_fig_path)
        plt.close(fig)
