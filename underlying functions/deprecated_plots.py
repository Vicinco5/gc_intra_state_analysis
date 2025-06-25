#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:57:32 2024

@author: vincentcalia-bogan

Unused and/or deprecated plotting functions
"""


def plot_firing_rates(all_interpolated_states, firing_rate_arrays, taste_idx=0):
    """
    Plots interpolated and actual firing rates for all neurons and trials of a given taste.
    The idea of this exploratory function is to plot, for one taste, the same neuron at the same
    trial with both interpolated and uniterpolated states near each other so we can
    analyze any rhyme or reason for states being cut early.
    """
    # Determine number of neurons based on the shape of the first trial of the first state
    num_neurons = firing_rate_arrays[taste_idx][0][0].shape[0]
    num_trials = len(firing_rate_arrays[taste_idx])

    for neuron_idx in range(num_neurons):
        for trial_idx in range(num_trials):
            # Create a new figure for each neuron and trial with 4 sets of plots
            fig, axs = plt.subplots(
                2, 3, figsize=(15, 10), constrained_layout=True
            )  # 2 rows, 3 columns per set
            fig.suptitle(f"Taste {taste_idx}, Neuron {neuron_idx}, Trial {trial_idx}")

        for state_idx in range(3):  # Assuming 3 states as per the setup
            # Correctly index interpolated data for a given state across all trials
            interpolated_data = np.array(
                [
                    trial[neuron_idx]
                    for trial in all_interpolated_states[taste_idx][state_idx]
                ]
            )
            median_length = interpolated_data.shape[
                2
            ]  # Assuming time is the third dimension
            time_interpolated = np.linspace(0, median_length - 1, median_length)

            # Plot interpolated firing rates for this neuron across all trials
            for interp_trial in interpolated_data:
                axs[0, state_idx].plot(
                    time_interpolated, interp_trial[trial_idx], alpha=0.5
                )

            axs[0, state_idx].set_title(f"Interpolated State {state_idx + 1}")
            axs[0, state_idx].set_xlabel("Time")
            axs[0, state_idx].set_ylabel("Firing Rate")

            # Plot actual firing rates for this neuron and trial
            actual_data = firing_rate_arrays[taste_idx][trial_idx][state_idx][
                neuron_idx, :
            ]
            time_actual = np.arange(actual_data.shape[0])
            axs[1, state_idx].plot(time_actual, actual_data)
            axs[1, state_idx].set_title(f"Actual State {state_idx + 1}")
            axs[1, state_idx].set_xlabel("Time")
            axs[1, state_idx].set_ylabel("Firing Rate")

        plt.show()


# These plots aren't super useful as there are a lot and we need to process them further with linspace


def plot_firing_rates(all_interpolated_states, state_firing_rates, taste_idx=0):
    """
    Plots interpolated and actual firing rates for all neurons and trials of a given taste.
    Plots are generated for each neuron and trial, comparing interpolated and actual firing rates.
    """
    # Extract the specific taste data
    interpolated_taste_data = all_interpolated_states[taste_idx]
    actual_taste_data = state_firing_rates[taste_idx]

    num_neurons = interpolated_taste_data.sizes["neuron"]
    num_trials = interpolated_taste_data.sizes["trial"]
    num_segments = interpolated_taste_data.sizes[
        "segment"
    ]  # Assuming 3 states as per the setup

    # Iterate over each neuron and trial
    for neuron_idx in range(num_neurons):
        for trial_idx in range(num_trials):
            # Create a new figure for each neuron and trial with 2 rows, num_segments columns
            fig, axs = plt.subplots(
                2, num_segments, figsize=(15, 10), constrained_layout=True
            )
            fig.suptitle(f"Taste {taste_idx}, Neuron {neuron_idx}, Trial {trial_idx}")

            for state_idx in range(num_segments):
                # Get interpolated data for this state
                interpolated_data = interpolated_taste_data.sel(
                    segment=state_idx, neuron=neuron_idx, trial=trial_idx
                )
                median_length = interpolated_data.sizes["time_bin"]
                time_interpolated = np.linspace(0, median_length - 1, median_length)

                # Plot interpolated firing rates
                axs[0, state_idx].plot(time_interpolated, interpolated_data, alpha=0.5)
                axs[0, state_idx].set_title(f"Interpolated State {state_idx + 1}")
                axs[0, state_idx].set_xlabel("Time (bins)")
                axs[0, state_idx].set_ylabel("Firing Rate (Hz)")

                # Get actual data for this state
                actual_data = actual_taste_data.sel(
                    segment=state_idx, neuron=neuron_idx, trial=trial_idx
                )
                time_actual = np.arange(actual_data.sizes["time_bin"])

                # Plot actual firing rates
                axs[1, state_idx].plot(time_actual, actual_data)
                axs[1, state_idx].set_title(f"Actual State {state_idx + 1}")
                axs[1, state_idx].set_xlabel("Time (bins)")
                axs[1, state_idx].set_ylabel("Firing Rate (Hz)")

            plt.show()


# interesting one in that the graph is continueous-- no seperate plots. Work through this one after


# this works lol
def plot_changepoints_spiketrain(
    dataset_num, state_spike_arrays_all_trials, window_length, step_size
):
    num_trials = len(state_spike_arrays_all_trials)

    # Determine the number of figures needed
    num_figures = (num_trials + 4) // 5  # Round up to the nearest multiple of 5

    for fig_idx in range(num_figures):
        fig, axs = plt.subplots(5, 3, figsize=(15, 20))
        fig.suptitle(
            f"Dataset {dataset_num} - Trials {fig_idx*5+1}-{min((fig_idx+1)*5, num_trials)}"
        )

        for i in range(5):
            if fig_idx * 5 + i >= num_trials:
                break
            trial_idx = fig_idx * 5 + i
            trial_spike_arrays = state_spike_arrays_all_trials[trial_idx]

            for cp_idx, state_spike_array in enumerate(trial_spike_arrays):
                ax = axs[i, cp_idx]

                # Create a binary mask where 0 indicates absence of spike
                spike_mask = state_spike_array[0] == 1

                # Plot the spike train using heatmap
                ax.imshow(spike_mask, cmap="binary", aspect="auto")
                ax.set_title(f"Trial {trial_idx + 1}, Changepoint {cp_idx + 1}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Neuron")

        plt.tight_layout()
        plt.show()


# fixes the x-axis but runs into some other issues
def plot_changepoints_spiketrain(
    dataset_num, state_spike_arrays_all_trials, window_length, step_size
):
    num_trials = len(state_spike_arrays_all_trials)

    # Determine the number of figures needed
    num_figures = (num_trials + 4) // 5  # Round up to the nearest multiple of 5

    for fig_idx in range(num_figures):
        fig, axs = plt.subplots(5, 3, figsize=(15, 20))
        fig.suptitle(
            f"Dataset {dataset_num} - Trials {fig_idx*5+1}-{min((fig_idx+1)*5, num_trials)}"
        )

        for i in range(5):
            if fig_idx * 5 + i >= num_trials:
                break
            trial_idx = fig_idx * 5 + i
            trial_spike_arrays = state_spike_arrays_all_trials[trial_idx]

            prev_end_time = 0  # Initialize previous end time

            for cp_idx, state_spike_array in enumerate(trial_spike_arrays):
                ax = axs[i, cp_idx]

                # Create a binary mask where 0 indicates absence of spike
                spike_mask = state_spike_array[0] == 1

                # Plot the spike train using heatmap
                ax.imshow(spike_mask, cmap="binary", aspect="auto")
                ax.set_title(f"Trial {trial_idx + 1}, Changepoint {cp_idx + 1}")
                ax.set_ylabel("Neuron")

                # Calculate the x-axis limits based on the total time span
                start_time = prev_end_time
                end_time = start_time + state_spike_array.shape[1]
                ax.set_xlim(0, state_spike_array.shape[1])  # Set x-axis limit

                # Set the x-axis tick labels
                num_ticks = state_spike_array.shape[1]
                tick_labels = [
                    f"{start_time + t * step_size:.0f}" for t in range(num_ticks)
                ]
                ax.set_xticks(range(num_ticks))  # Set ticks
                ax.set_xticklabels(tick_labels)  # Set tick labels
                ax.set_xlabel("Time")

                prev_end_time += (
                    state_spike_array.shape[1] + 1
                )  # Update previous end time with padding

        plt.tight_layout()
        plt.show()
