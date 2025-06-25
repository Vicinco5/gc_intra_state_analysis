#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:20:13 2024

@author: vincentcalia-bogan
"""


import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# this plots a line plot of the warped stated averaged down across neuron such that each individual trial is
# plotted. This is less useful.
def plt_inter_cpfr_avg_nrn_line(all_interpolated_states, modified_tastes, dataset_num):
    num_tastes = len(all_interpolated_states)
    num_changepoints = 3  # number of changepoints; hardcded but will fix down the line
    colors = ["red", "green", "blue", "purple"]  # Adjust as necessary for more taste
    taste_labels = modified_tastes
    fig, axes = plt.subplots(
        num_tastes,
        num_changepoints,
        figsize=(15, 5 * num_tastes),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"Interpolated Firing Rates Averaged Across Neurons, Dataset: {dataset_num}",
        fontsize=16,
        y=0.995,
    )
    # Ensure axes is 2D array for consistent indexing when num_tastes is 1
    if num_tastes == 1:
        axes = np.expand_dims(axes, 0)
    for taste_idx in range(num_tastes):
        for cp_idx in range(num_changepoints):
            ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
            ax.set_title(f"Taste:{taste_labels[taste_idx]}, Changepoint {cp_idx + 1}")
            ax.set_xlabel("Time Duration of State")
            if cp_idx == 0:  # Only add Y label to the first column
                ax.set_ylabel("Firing Rate (Hz)")
            # using np.nanmean to specifically ignore nan values when doing this graphing
            # Plot each trial in faint lines for the current changepoint and taste
            first_trial = True  # flag for the purposes of the legend
            all_valid_trials = []
            for trial_array in all_interpolated_states[taste_idx][cp_idx]:
                valid_data = np.array(trial_array[~np.isnan(trial_array).all(axis=1)])
                if (
                    valid_data.size > 0
                ):  # if there aren't only nan values in the original data
                    avg_fr_nrns = np.nanmean(valid_data, axis=0)
                    avg_firing_rate_across_neurons = avg_fr_nrns[~np.isnan(avg_fr_nrns)]
                    all_valid_trials.append(avg_firing_rate_across_neurons)
                    # Calculate and plot the average firing rate across neurons for each trial
                    num_time_bins = avg_firing_rate_across_neurons.shape[0]
                    x_ax = np.linspace(0, 1, num=num_time_bins)
                    if first_trial:
                        ax.plot(
                            x_ax,
                            avg_firing_rate_across_neurons,
                            color=colors[taste_idx],
                            alpha=0.2,
                            label="Trial",
                        )
                        ax.legend()
                        first_trial = False
                    else:
                        ax.plot(
                            x_ax,
                            avg_firing_rate_across_neurons,
                            color=colors[taste_idx],
                            alpha=0.2,
                        )
            # Plot the average of all trials for this taste and changepoint as a thicker line
            if all_valid_trials:
                max_len = max(trial.shape[0] for trial in all_valid_trials)
                align_trials = np.array(
                    [
                        np.pad(
                            trial, (0, max_len - trial.shape[0]), constant_values=np.nan
                        )
                        for trial in all_valid_trials
                    ]
                )
                # again this padding should no longer be needed but retaining it anyhow
                avg_fr_nrns = np.nanmean(align_trials, axis=0)
                num_time_bins = avg_fr_nrns.shape[0]
                x_ax = np.linspace(0, 1, num=num_time_bins)
                ax.plot(
                    x_ax,
                    avg_fr_nrns,
                    color=colors[taste_idx],
                    linewidth=2,
                    label="Average",
                )
                ax.legend()
    plt.tight_layout()
    plt.show()


# Same style of line plot as above, expect this time we're averaging down across trials such that individual neuron's
# firing rates are preserved. Less of a measure of population but as we're working with warped data still
# somewhat non-ideal
def plt_inter_cpfr_avg_tr_line(all_interpolated_states, modified_tastes, dataset_num):
    num_tastes = len(all_interpolated_states)
    num_changepoints = 3  # number of changepoints; hardcoded but will fix down the line
    colors = ["red", "green", "blue", "purple"]  # Adjust as necessary for more tastes
    taste_labels = modified_tastes
    fig, axes = plt.subplots(
        num_tastes,
        num_changepoints,
        figsize=(15, 5 * num_tastes),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"Interpolated Firing Rates Averaged Across Trials, Dataset: {dataset_num}",
        fontsize=16,
        y=0.995,
    )
    # Ensure axes is 2D
    if num_tastes == 1:
        axes = np.expand_dims(axes, 0)
    for taste_idx in range(num_tastes):
        for cp_idx in range(num_changepoints):
            ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
            ax.set_title(f"Taste: {taste_labels[taste_idx]}, Changepoint {cp_idx + 1}")
            ax.set_xlabel("Time Duration of State")
            if cp_idx == 0:  # Only add Y label to the first column
                ax.set_ylabel("Firing Rate (Hz)")
            all_neuron_data = []
            first_trial = True
            for neuron_idx in range(
                all_interpolated_states[taste_idx].shape[2]
            ):  # Loop through neurons
                neuron_trials = []  # init neuron trials
                for trial_idx in range(
                    all_interpolated_states[taste_idx].shape[1]
                ):  # Loop through trials
                    trial_array = all_interpolated_states[taste_idx][cp_idx][trial_idx][
                        neuron_idx
                    ].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if (
                        valid_data.size > 0
                    ):  # if there are only nan values in the original data
                        neuron_trials.append(valid_data)  # append non-nan values
                if neuron_trials:
                    stacked_trials = np.stack(neuron_trials, axis=0)
                    avg_fr_neuron = np.nanmean(stacked_trials, axis=0)
                    all_neuron_data.append(avg_fr_neuron)
                    num_time_bins = avg_fr_neuron.shape[0]
                    x_ax = np.linspace(0, 1, num=num_time_bins)
                    if first_trial:
                        ax.plot(
                            x_ax,
                            avg_fr_neuron,
                            color=colors[taste_idx],
                            alpha=0.2,
                            label="Neuron",
                        )
                        ax.legend()
                        first_trial = False
                    else:
                        ax.plot(x_ax, avg_fr_neuron, color=colors[taste_idx], alpha=0.2)
            if all_neuron_data:
                all_neuron_data = np.array(all_neuron_data)
                avg_fr_across_neurons = np.nanmean(all_neuron_data, axis=0)
                num_time_bins = avg_fr_across_neurons.shape[0]
                x_ax = np.linspace(0, 1, num=num_time_bins)
                ax.plot(
                    x_ax,
                    avg_fr_across_neurons,
                    color=colors[taste_idx],
                    linewidth=2,
                    label="Neuron Avg",
                )
                ax.legend()
    plt.tight_layout()
    plt.show()


# plots warped firing rates of single neurons (all trials per state); generates several hundred plots in the output
# directory specified. Still a line plot
def plt_single_neuron_firing_warped_line(
    all_interpolated_states, dataset_num, modified_tastes, output_dir_warped
):
    num_tastes = len(all_interpolated_states)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    # Create the output directory if it doesn't exist
    dataset_dir = os.path.join(output_dir_warped, f"{dataset_num}_single-neuron-firing")
    os.makedirs(dataset_dir, exist_ok=True)

    num_neurons = all_interpolated_states[0][0].shape[
        1
    ]  # Assuming consistent neuron count across tastes and changepoints
    num_trials = all_interpolated_states[0][0].shape[
        0
    ]  # Assuming consistent trial count across tastes and changepoints
    colormap = cm.get_cmap(
        "rainbow", num_trials
    )  # Generate a colormap with `num_trials` colors

    # Loop through each neuron
    for neuron_idx in range(num_neurons):
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(
            f"Single Neuron Firing Rates, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}",
            fontsize=16,
            y=0.995,
        )

        # Ensure axes is 2D
        if num_tastes == 1:
            axes = np.expand_dims(axes, 0)

        for taste_idx in range(num_tastes):
            for cp_idx in range(num_changepoints):
                ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                ax.set_title(
                    f"Taste: {taste_labels[taste_idx]}, Changepoint {cp_idx + 1}"
                )
                ax.set_xlabel("Time Duration of State")
                if cp_idx == 0:  # Only add Y label to the first column
                    ax.set_ylabel("Firing Rate (Hz)")

                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = all_interpolated_states[taste_idx][cp_idx][trial_idx][
                        neuron_idx
                    ].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if (
                        valid_data.size > 0
                    ):  # Only plot if there are valid (non-NaN) values
                        trial_color = colormap(
                            trial_idx
                        )  # Get a unique color for each trial
                        num_time_bins = valid_data.shape[0]
                        x_ax = np.linspace(0, 1, num=num_time_bins)
                        ax.plot(x_ax, valid_data, color=trial_color, alpha=0.3)
        plt.tight_layout()
        neuron_fig_path = os.path.join(dataset_dir, f"neuron_{neuron_idx + 1}.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


# still the line plot; added some lines and other logic to calculate percentiles as to when and where firing ends
# Now doing the same but for the unwarped data-- plotting every trial of every neuron
# note: update the name of the actual states for each plot; also check with abu about how many trials there are
def plt_single_neuron_firing_unwarped_line(
    state_firing_rates_all_trials,
    dataset_num,
    output_dir_unwarped,
    modified_tastes,
    epoch_labels,
    step_size,
):
    num_tastes = len(state_firing_rates_all_trials)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    # label the changepoints better too
    # Create the output directory if it doesn't exist
    dataset_dir = os.path.join(
        output_dir_unwarped, f"{dataset_num}_single-neuron-firing"
    )
    os.makedirs(dataset_dir, exist_ok=True)
    num_neurons = state_firing_rates_all_trials[0][0].shape[
        1
    ]  # Assuming consistent neuron count across tastes and changepoints
    num_trials = state_firing_rates_all_trials[0].shape[
        0
    ]  # Assuming consistent trial count across tastes and changepoints
    colormap = cm.get_cmap(
        "rainbow", num_trials
    )  # Generate a colormap with `num_trials` colors
    # Loop through each neuron
    for neuron_idx in range(num_neurons):
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(
            f"Unwarped Single Neuron Firing Rates, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}",
            fontsize=16,
            y=0.995,
        )
        # Ensure axes is 2D
        if num_tastes == 1:
            axes = np.expand_dims(axes, 0)
        for taste_idx in range(num_tastes):
            start_time = 2000
            for cp_idx in range(num_changepoints):
                ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                ax.set_title(
                    f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                )
                ax.set_xlabel("Time Duration of State")
                if cp_idx == 0:  # Only add Y label to the first column
                    ax.set_ylabel("Firing Rate (Hz)")
                trial_lengths = []
                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = state_firing_rates_all_trials[taste_idx][trial_idx][
                        cp_idx
                    ][neuron_idx].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if (
                        valid_data.size > 0
                    ):  # Only plot if there are valid (non-NaN) values
                        trial_color = colormap(
                            trial_idx
                        )  # Get a unique color for each trial
                        num_time_bins = valid_data.shape[
                            0
                        ]  # appears that we're getting repeated lengths?
                        x_ax = (
                            np.arange(num_time_bins) * step_size
                        )  # Convert to milliseconds
                        ax.plot(x_ax, valid_data, color=trial_color, alpha=0.5)
                        trial_lengths.append(
                            num_time_bins
                        )  # record how long a trial is
                if trial_lengths:
                    avg_duration = (
                        np.mean(trial_lengths) * step_size
                    )  # onset of stimulus
                    trial_lengths = sorted(trial_lengths)
                    trial_lengths = np.array(trial_lengths)
                    percentiles = np.percentile(trial_lengths, [0, 25, 50, 75, 90])
                    # the gray lines will all look exactly the same across neurons-- this is expected and ok
                    max_time_bins = max(trial_lengths)
                    labels = ["0th", "25th", "50th", "75th", "90th"]
                    for i, percentile in enumerate(percentiles):
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
                end_time = start_time + (max_time_bins * step_size)
                start_time = end_time  # not actually sure these are needed here
                avg_duration = (
                    avg_duration  # I don't actually think start/end times are needed
                )
                # Update the title to include the average duration
                ax.set_title(
                    f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                )
                ax.set_xlabel("Time (ms)")
                if cp_idx == 0:  # Only add Y label to the first column
                    ax.set_ylabel("Firing Rate (Hz)")
        plt.tight_layout()
        neuron_fig_path = os.path.join(dataset_dir, f"neuron_{neuron_idx + 1}.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


# this is the above but in a heatmap that is sorted according to ascending unwarped firing rate
def plt_single_neuron_firing_unwarped_heatmap(
    state_firing_rates_all_trials,
    dataset_num,
    output_dir_unwarped,
    window_length,
    step_size,
    modified_tastes,
    epoch_labels,
):
    num_tastes = len(state_firing_rates_all_trials)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    taste_labels = modified_tastes
    # Create the output directory if it doesn't exist
    dataset_dir = os.path.join(
        output_dir_unwarped, f"{dataset_num}_single-neuron-firing"
    )
    os.makedirs(dataset_dir, exist_ok=True)
    num_neurons = state_firing_rates_all_trials[0][0].shape[
        1
    ]  # Assuming consistent neuron count across tastes and changepoints
    num_trials = state_firing_rates_all_trials[0].shape[
        0
    ]  # Assuming consistent trial count across tastes and changepoints
    # Loop through each neuron
    for neuron_idx in range(num_neurons):
        fig, axes = plt.subplots(
            num_tastes,
            num_changepoints,
            figsize=(15, 5 * num_tastes),
            sharex=False,
            sharey=False,
        )
        fig.suptitle(
            f"Unwarped Single Neuron Firing Rates, Neuron: {neuron_idx + 1}, Dataset: {dataset_num}",
            fontsize=16,
            y=0.995,
        )
        # Ensure axes is 2D
        if num_tastes == 1:
            axes = np.expand_dims(axes, 0)
        for taste_idx in range(num_tastes):
            start_time = 2000
            for cp_idx in range(num_changepoints):
                ax = axes[taste_idx, cp_idx] if num_tastes > 1 else axes[cp_idx]
                ax.set_title(
                    f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}"
                )
                ax.set_xlabel("Time Duration of State")
                # Collect data for heatmap
                trial_data = []
                trial_lengths = []
                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = state_firing_rates_all_trials[taste_idx][trial_idx][
                        cp_idx
                    ][neuron_idx].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if valid_data.size > 0:  # Only include valid (non-NaN) values
                        trial_lengths.append(valid_data.shape[0])
                        trial_data.append(valid_data)
                if trial_data:
                    # Sort trials by length
                    sorted_indices = np.argsort(trial_lengths)
                    sorted_data = [trial_data[i] for i in sorted_indices]
                    sorted_lengths = sorted(trial_lengths)
                    avg_duration = (
                        np.mean(trial_lengths) * step_size
                    )  # onset of stimulus
                    # Create heatmap
                    max_length = max(trial_lengths)
                    heatmap_data = np.zeros((num_trials, max_length)) * np.nan
                    for i, data in enumerate(sorted_data):
                        heatmap_data[i, : len(data)] = data
                    # Plot heatmap
                    im = ax.imshow(
                        heatmap_data,
                        aspect="auto",
                        cmap="viridis",
                        origin="lower",
                        extent=[0, max_length * step_size, 0, num_trials],
                    )
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Firing Rate (Hz)")
                    # Add vertical percentile lines
                    percentiles = np.percentile(sorted_lengths, [0, 25, 50, 75, 90])
                    labels = ["0th", "25th", "50th", "75th", "90th"]
                    for i, percentile in enumerate(percentiles):
                        line_position = percentile * step_size
                        ax.axvline(line_position, linestyle=":", color="m", alpha=0.7)
                        ax.text(
                            line_position,
                            num_trials - 1,
                            f"{labels[i]} percentile",
                            rotation=90,
                            verticalalignment="top",
                            color="c",
                            alpha=1,
                            fontsize=12,
                        )

                    # Calculate the end time of the current state using max_time_bins
                    end_time = start_time + (max_length * step_size)
                    avg_duration = avg_duration
                    # Update the title to include the average duration
                    ax.set_title(
                        f"Taste: {taste_labels[taste_idx]}, Epoch: {epoch_labels[cp_idx]}\nAverage Epoch Duration: {avg_duration:.2f} ms"
                    )
                    ax.set_xlabel("Time (ms)")
                    if cp_idx == 0:  # Only add Y label to the first column
                        ax.set_ylabel("Trial Index")
                    # Update start time for the next state
                    start_time = end_time
        plt.tight_layout()
        neuron_fig_path = os.path.join(dataset_dir, f"neuron_{neuron_idx + 1}.png")
        plt.savefig(neuron_fig_path)
        plt.close(fig)


# plotting changepoints in a scatter plot(mainly to see transition times)
def plt_changepoints_scat(changepoints, modified_tastes, dataset_num):
    taste_labels = modified_tastes
    num_tastes = len(changepoints)
    fig, axes = plt.subplots(num_tastes, 1, figsize=(5, 15), sharey=True)
    fig.suptitle(
        f"Changepoints for each Taste, Dataset: {dataset_num}", fontsize=9, y=0.993
    )
    changepoint_colors = ["blue", "red", "green"]  # Colors for each changepoint
    start_time = 2000  # start time for graphing -- upon stim. delivery
    # Ensure axes is a list for consistent indexing when num_tastes is 1
    if num_tastes == 1:
        axes = [axes]
    for taste_idx in range(num_tastes):
        ax = axes[taste_idx]
        ax.set_title(taste_labels[taste_idx])
        ax.set_xlabel("Time in ms post stimulus delivery")
        if taste_idx == 0:
            ax.set_ylabel("Trial")
        # sorting cp's using a cool lambda function
        sorted_changepoints = sorted(changepoints[taste_idx], key=lambda x: x[0])

        # Loop through each trial for the current taste
        for trial_idx, changepoint in enumerate(sorted_changepoints):
            # Plot each changepoint as a dot with the corresponding color
            for cp_idx, cp_time in enumerate(changepoint):
                adjusted_cp_time = cp_time - start_time
                ax.scatter(
                    adjusted_cp_time,
                    trial_idx,
                    color=changepoint_colors[cp_idx],
                    s=10,
                    label=f"CP{cp_idx+1}" if trial_idx == 0 else "",
                )
            # Add a thin black line separating each trial
            ax.axhline(y=trial_idx, color="black", linewidth=0.5)
        # ax.text(changepoints[taste_idx].min(), trial_idx, str(trial_idx), verticalalignment='center', fontsize=8)
        # ax.text(changepoints[taste_idx].max(), trial_idx, str(trial_idx), verticalalignment='center', fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
    plt.tight_layout()
    plt.show()


# plotting changepoints in a relatively quick and dirty histogram
def plt_changepoint_hist(changepoints, modified_tastes, dataset_num):
    taste_labels = modified_tastes
    num_tastes = len(changepoints)
    fig, axes = plt.subplots(num_tastes, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(
        f"Changepoint Durations for each Taste, Dataset: {dataset_num}",
        fontsize=9,
        y=0.993,
    )
    changepoint_colors = ["blue", "red", "green"]  # Colors for each changepoint
    start_time = 2000  # Start time for the first changepoint
    # Ensure axes is a list for consistent indexing when num_tastes is 1
    if num_tastes == 1:
        axes = [axes]
    for taste_idx in range(num_tastes):
        ax = axes[taste_idx]
        ax.set_title(taste_labels[taste_idx])
        ax.set_xlabel("Duration in ms")
        if taste_idx == 0:
            ax.set_ylabel("Frequency")
        # Collect durations for each changepoint
        cp_durations = [[] for _ in range(len(changepoint_colors))]
        for trial_changepoints in changepoints[taste_idx]:
            # Calculate durations between changepoints
            previous_cp_time = start_time
            for cp_idx, cp_time in enumerate(trial_changepoints):
                duration = cp_time - previous_cp_time
                cp_durations[cp_idx].append(duration)
                previous_cp_time = cp_time
        # Plot histograms for each changepoint
        for cp_idx, durations in enumerate(cp_durations):
            ax.hist(
                durations,
                bins=30,
                color=changepoint_colors[cp_idx],
                alpha=0.5,
                label=f"CP{cp_idx+1}",
            )
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
