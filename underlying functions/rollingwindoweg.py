#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:22:29 2024

@author: vincentcalia-bogan
"""


import numpy as np
import matplotlib.pyplot as plt


# Function to calculate firing rate using sliding window
def calc_fr_rr(state_spike_array, window_length, step_size):
    num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
    nrn_num, time_num = state_spike_array.shape
    firing_rate = np.zeros((nrn_num, num_bins))
    for bin_ini in range(num_bins):
        start_bin = bin_ini * step_size
        end_bin = min(start_bin + window_length, time_num)
        num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
        firing_rate[:, bin_ini] = num_spikes / (window_length)  # Convert to Hz
    return firing_rate


# Generate a spike train with three different states
np.random.seed(42)
total_time = 600
changepoints = [200, 400]
spike_train = np.zeros((1, total_time))  # Single neuron spike train

# State 1: Low firing rate
spike_train[0, : changepoints[0]] = np.random.binomial(1, 0.1, changepoints[0])

# State 2: High firing rate
spike_train[0, changepoints[0] : changepoints[1]] = np.random.binomial(
    1, 0.5, changepoints[1] - changepoints[0]
)

# State 3: Moderate firing rate
spike_train[0, changepoints[1] :] = np.random.binomial(
    1, 0.3, total_time - changepoints[1]
)

# Calculate sliding window firing rate
window_length = 20
step_size = 5
firing_rate = calc_fr_rr(spike_train, window_length, step_size)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(15, 3), sharex=True)

# Plot the firing rate
ax[0].plot(
    np.arange(firing_rate.shape[1]) * step_size + window_length / 2,
    firing_rate[0],
    color="black",
)
ax[0].set_ylabel("Firing Rate", fontsize=16)
# ax[0].set_title('Sliding-Window Firing Rate and Spike Train with Changepoints')

# Highlight state changes
colors = ["#FC6ECB", "#BC7FBC", "#E8BBDF", "#C7CEE7"]

# Shade the first epoch
ax[0].axvspan(0, changepoints[0], color=colors[0], alpha=0.3)
ax[1].axvspan(0, changepoints[0], color=colors[0], alpha=0.3)

# Shade subsequent epochs
for i, cp in enumerate(changepoints):
    ax[0].axvspan(
        cp,
        changepoints[i + 1] if i + 1 < len(changepoints) else total_time,
        color=colors[i + 1],
        alpha=0.3,
    )
    ax[1].axvspan(
        cp,
        changepoints[i + 1] if i + 1 < len(changepoints) else total_time,
        color=colors[i + 1],
        alpha=0.3,
    )

# Plot the spike train
ax[1].eventplot(
    np.where(spike_train[0] == 1)[0], orientation="horizontal", colors="black"
)
ax[1].set_xlabel("Time (ms)", fontsize=18)
ax[1].set_ylabel("Spike Train", fontsize=16)

# Indicate changepoints with vertical lines
for cp in changepoints:
    ax[0].axvline(cp, color="#D00000", linestyle="--")
    ax[1].axvline(cp, color="#D00000", linestyle="--")

# Adjust the margins to remove the white space
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
ax[0].margins(x=0)
ax[1].margins(x=0)

plt.tight_layout()
plt.show()


# eg warping fig.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Function to calculate firing rate using sliding window
def calc_fr_rr(state_spike_array, window_length, step_size):
    num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
    nrn_num, time_num = state_spike_array.shape
    firing_rate = np.zeros((nrn_num, num_bins))
    for bin_ini in range(num_bins):
        start_bin = bin_ini * step_size
        end_bin = min(start_bin + window_length, time_num)
        num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
        firing_rate[:, bin_ini] = num_spikes / (
            window_length * 0.001
        )  # Convert to Hz (window in ms)
    return firing_rate


# -------------------------
# Generate 4 neurons with different durations
np.random.seed(42)
neuron_durations = [500, 700, 600, 800]  # Different lengths (ms)

neurons = []
target_firing_rates = np.random.uniform(
    4, 7, size=4
)  # Random target FR between 3 and 7 Hz

print("Target firing rates (Hz) per neuron:", target_firing_rates)

for dur, fr_target in zip(neuron_durations, target_firing_rates):
    # Approximate spike probability per ms
    spike_prob_per_ms = fr_target / 1000  # (Hz is spikes/sec, ms is 1/1000 sec)
    spikes = np.random.binomial(1, spike_prob_per_ms, dur)
    neurons.append(spikes)

# Make them into a ragged list first
# Then pad them with NaN to the longest
max_length = max(len(nrn) for nrn in neurons)
neurons_padded = []

for nrn in neurons:
    padded = np.full(max_length, np.nan)
    padded[: len(nrn)] = nrn
    neurons_padded.append(padded)

neurons_padded = np.vstack(neurons_padded)  # Shape: (4 neurons, max_length)

# -------------------------
# Calculate firing rates
window_length = 250  # ms window (adjusted for smoothing, better for 3–7 Hz)
step_size = 25  # ms step
firing_rates = []

for nrn_idx in range(neurons_padded.shape[0]):
    valid_spikes = neurons_padded[nrn_idx][~np.isnan(neurons_padded[nrn_idx])]
    valid_spikes = valid_spikes.reshape(1, -1)  # calc_fr_rr expects (n_neurons, time)
    fr = calc_fr_rr(valid_spikes, window_length, step_size)
    # Now pad back to common length
    padded_fr = np.full(
        (fr.shape[0], (max_length - window_length) // step_size + 1), np.nan
    )
    padded_fr[:, : fr.shape[1]] = fr
    firing_rates.append(padded_fr[0])

firing_rates = np.vstack(firing_rates)  # Shape: (4, num_bins)

# Calculate average firing rate (ignoring NaNs)
avg_firing_rate = np.nanmean(firing_rates, axis=0)

# -------------------------
# Plot 1: Original firing rates (padded)
time_vector = np.arange(firing_rates.shape[1]) * step_size + window_length / 2

fig, ax = plt.subplots(figsize=(15, 5))

for idx in range(firing_rates.shape[0]):
    ax.plot(time_vector, firing_rates[idx], label=f"Neuron {idx+1}", alpha=0.7)

ax.plot(
    time_vector,
    avg_firing_rate,
    color="black",
    linewidth=2.5,
    label="Average firing rate",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Raw (non-stretched) Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()

# -------------------------
# Stretching: interpolate each neuron to same length
interp_len = 300  # Define a common stretched length
new_time = np.linspace(0, time_vector[-1], interp_len)

firing_rates_stretched = []

for idx in range(firing_rates.shape[0]):
    valid_mask = ~np.isnan(firing_rates[idx])
    old_time_valid = time_vector[valid_mask]
    fr_valid = firing_rates[idx][valid_mask]
    if len(old_time_valid) < 2:
        stretched = np.full(interp_len, np.nan)
    else:
        interpolator = interp1d(
            old_time_valid, fr_valid, kind="nearest", fill_value="extrapolate"
        )
        stretched = interpolator(new_time)
    firing_rates_stretched.append(stretched)

firing_rates_stretched = np.vstack(firing_rates_stretched)

avg_firing_rate_stretched = np.nanmean(firing_rates_stretched, axis=0)

# -------------------------
# Plot 2: Stretched firing rates
fig, ax = plt.subplots(figsize=(15, 5))

for idx in range(firing_rates_stretched.shape[0]):
    ax.plot(new_time, firing_rates_stretched[idx], label=f"Neuron {idx+1}", alpha=0.7)

ax.plot(
    new_time,
    avg_firing_rate_stretched,
    color="black",
    linewidth=2.5,
    label="Average firing rate (stretched)",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Stretched Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Function to calculate firing rate using sliding window
def calc_fr_rr(state_spike_array, window_length, step_size):
    num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
    nrn_num, time_num = state_spike_array.shape
    firing_rate = np.zeros((nrn_num, num_bins))
    for bin_ini in range(num_bins):
        start_bin = bin_ini * step_size
        end_bin = min(start_bin + window_length, time_num)
        num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
        firing_rate[:, bin_ini] = num_spikes / (
            window_length * 0.001
        )  # Convert to Hz (window in ms)
    return firing_rate


# -------------------------
# Generate 4 neurons with different durations and controlled dynamics
np.random.seed(42)
neuron_durations = [550, 720, 600, 800]  # Different lengths (ms)

neurons = []
base_firing_rates = np.random.uniform(3, 7, size=4)  # Base target FR between 3 and 7 Hz

print("Base firing rates (Hz) per neuron:", base_firing_rates)

for dur, fr_target in zip(neuron_durations, base_firing_rates):
    spike_prob_per_ms = fr_target / 1000  # Convert Hz to prob per ms

    # Create structured modulation:
    time = np.linspace(0, 1, dur)
    modulation = (
        -0.5 * np.exp(-((time - 0.33) ** 2) / (2 * 0.01))  # Dip around 1/3
        + 1.3 * np.exp(-((time - 0.66) ** 2) / (2 * 0.01))  # Rise/peak around 2/3
        - 0.3 * np.exp(-((time - 0.9) ** 2) / (2 * 0.005))  # Small settling near end
    )
    modulation += 1.9 * np.random.randn(dur)  # Add small random noise

    # Final dynamic firing probability
    dynamic_spike_prob = np.clip(
        spike_prob_per_ms * (1 + modulation), 0.0005, 0.05
    )  # Cap at realistic spike probs

    # Sample spikes
    spikes = np.random.binomial(1, dynamic_spike_prob)
    neurons.append(spikes)

# -------------------------
# Pad shorter neurons with NaNs to match the longest
max_length = max(len(nrn) for nrn in neurons)
neurons_padded = []

for nrn in neurons:
    padded = np.full(max_length, np.nan)
    padded[: len(nrn)] = nrn
    neurons_padded.append(padded)

neurons_padded = np.vstack(neurons_padded)  # Shape: (4 neurons, max_length)

# -------------------------
# Calculate firing rates
window_length = 180  # ms window
step_size = 20  # ms step
firing_rates = []

for nrn_idx in range(neurons_padded.shape[0]):
    valid_spikes = neurons_padded[nrn_idx][~np.isnan(neurons_padded[nrn_idx])]
    valid_spikes = valid_spikes.reshape(1, -1)  # calc_fr_rr expects (n_neurons, time)
    fr = calc_fr_rr(valid_spikes, window_length, step_size)
    # Now pad back to common length
    padded_fr = np.full(
        (fr.shape[0], (max_length - window_length) // step_size + 1), np.nan
    )
    padded_fr[:, : fr.shape[1]] = fr
    firing_rates.append(padded_fr[0])

firing_rates = np.vstack(firing_rates)  # Shape: (4, num_bins)

# Calculate average firing rate (ignoring NaNs)
avg_firing_rate = np.nanmean(firing_rates, axis=0)

# -------------------------
# Plot 1: Original firing rates (padded)
time_vector = np.arange(firing_rates.shape[1]) * step_size + window_length / 2

fig, ax = plt.subplots(figsize=(7, 5))

for idx in range(firing_rates.shape[0]):
    ax.plot(time_vector, firing_rates[idx], label=f"Neuron {idx+1}", alpha=0.7)

ax.plot(
    time_vector,
    avg_firing_rate,
    color="black",
    linewidth=2.5,
    label="Average firing rate",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Raw (non-stretched) Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()

# -------------------------
# Stretching: interpolate each neuron to same length
interp_len = 300  # Define a common stretched length
new_time = np.linspace(0, time_vector[-1], interp_len)

firing_rates_stretched = []

for idx in range(firing_rates.shape[0]):
    valid_mask = ~np.isnan(firing_rates[idx])
    old_time_valid = time_vector[valid_mask]
    fr_valid = firing_rates[idx][valid_mask]
    if len(old_time_valid) < 2:
        stretched = np.full(interp_len, np.nan)
    else:
        interpolator = interp1d(
            old_time_valid, fr_valid, kind="nearest", fill_value="extrapolate"
        )
        stretched = interpolator(new_time)
    firing_rates_stretched.append(stretched)

firing_rates_stretched = np.vstack(firing_rates_stretched)

avg_firing_rate_stretched = np.nanmean(firing_rates_stretched, axis=0)

# -------------------------
# Plot 2: Stretched firing rates
fig, ax = plt.subplots(figsize=(7, 5))

for idx in range(firing_rates_stretched.shape[0]):
    ax.plot(new_time, firing_rates_stretched[idx], label=f"Neuron {idx+1}", alpha=0.7)

ax.plot(
    new_time,
    avg_firing_rate_stretched,
    color="black",
    linewidth=2.5,
    label="Average firing rate (stretched)",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Stretched Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Function to calculate firing rate using sliding window
def calc_fr_rr(state_spike_array, window_length, step_size):
    num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
    nrn_num, time_num = state_spike_array.shape
    firing_rate = np.zeros((nrn_num, num_bins))
    for bin_ini in range(num_bins):
        start_bin = bin_ini * step_size
        end_bin = min(start_bin + window_length, time_num)
        num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
        firing_rate[:, bin_ini] = num_spikes / (
            window_length * 0.001
        )  # Convert to Hz (window in ms)
    return firing_rate


# -------------------------
# Generate 4 neurons with different durations and controlled dynamics
np.random.seed(42)
neuron_durations = [550, 720, 600, 800]  # Different lengths (ms)

neurons = []
base_firing_rates = np.random.uniform(3, 7, size=4)  # Base target FR between 3 and 7 Hz

print("Base firing rates (Hz) per neuron:", base_firing_rates)

for dur, fr_target in zip(neuron_durations, base_firing_rates):
    spike_prob_per_ms = fr_target / 1000  # Convert Hz to prob per ms

    # Create structured modulation:
    time = np.linspace(0, 1, dur)
    modulation = (
        -0.5 * np.exp(-((time - 0.33) ** 2) / (2 * 0.01))  # Dip around 1/3
        + 1.3 * np.exp(-((time - 0.66) ** 2) / (2 * 0.01))  # Rise/peak around 2/3
        - 0.3 * np.exp(-((time - 0.9) ** 2) / (2 * 0.005))  # Small settling near end
    )
    modulation += 1.9 * np.random.randn(dur)  # Add big random noise

    # Final dynamic firing probability
    dynamic_spike_prob = np.clip(
        spike_prob_per_ms * (1 + modulation), 0.0005, 0.05
    )  # Cap at realistic spike probs

    # Sample spikes
    spikes = np.random.binomial(1, dynamic_spike_prob)
    neurons.append(spikes)

# -------------------------
# Pad shorter neurons with NaNs to match the longest
max_length = max(len(nrn) for nrn in neurons)
neurons_padded = []

for nrn in neurons:
    padded = np.full(max_length, np.nan)
    padded[: len(nrn)] = nrn
    neurons_padded.append(padded)

neurons_padded = np.vstack(neurons_padded)  # Shape: (4 neurons, max_length)

# -------------------------
# Calculate firing rates
window_length = 180  # ms window
step_size = 20  # ms step
firing_rates = []

for nrn_idx in range(neurons_padded.shape[0]):
    valid_spikes = neurons_padded[nrn_idx][~np.isnan(neurons_padded[nrn_idx])]
    valid_spikes = valid_spikes.reshape(1, -1)  # calc_fr_rr expects (n_neurons, time)
    fr = calc_fr_rr(valid_spikes, window_length, step_size)
    # Now pad back to common length
    padded_fr = np.full(
        (fr.shape[0], (max_length - window_length) // step_size + 1), np.nan
    )
    padded_fr[:, : fr.shape[1]] = fr
    firing_rates.append(padded_fr[0])

firing_rates = np.vstack(firing_rates)  # Shape: (4, num_bins)

# Calculate average firing rate (ignoring NaNs)
avg_firing_rate = np.nanmean(firing_rates, axis=0)

# -------------------------
# Plot 1: Original firing rates (padded) with vertical dashed lines
time_vector = np.arange(firing_rates.shape[1]) * step_size + window_length / 2

fig, ax = plt.subplots(figsize=(7, 5))

colors = plt.cm.tab10.colors  # Grab a standard colormap (tab10)

for idx in range(firing_rates.shape[0]):
    (line,) = ax.plot(
        time_vector,
        firing_rates[idx],
        label=f"Neuron {idx+1}",
        alpha=0.7,
        color=colors[idx % len(colors)],
    )

    # Find the index where NaNs start
    valid_mask = ~np.isnan(firing_rates[idx])
    if np.any(valid_mask):
        last_valid_idx = np.where(valid_mask)[0][-1]
        last_valid_time = time_vector[last_valid_idx]
        ax.axvline(
            x=last_valid_time,
            color=colors[idx % len(colors)],
            linestyle="--",
            alpha=0.7,
        )

# Plot average
ax.plot(
    time_vector,
    avg_firing_rate,
    color="black",
    linewidth=2.5,
    label="Average firing rate",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Raw (non-warped) Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()

# -------------------------
# Stretching: interpolate each neuron to same length
interp_len = 300  # Define a common stretched length
new_time = np.linspace(0, time_vector[-1], interp_len)

firing_rates_stretched = []

for idx in range(firing_rates.shape[0]):
    valid_mask = ~np.isnan(firing_rates[idx])
    old_time_valid = time_vector[valid_mask]
    fr_valid = firing_rates[idx][valid_mask]
    if len(old_time_valid) < 2:
        stretched = np.full(interp_len, np.nan)
    else:
        interpolator = interp1d(
            old_time_valid, fr_valid, kind="nearest", fill_value="extrapolate"
        )
        stretched = interpolator(new_time)
    firing_rates_stretched.append(stretched)

firing_rates_stretched = np.vstack(firing_rates_stretched)

avg_firing_rate_stretched = np.nanmean(firing_rates_stretched, axis=0)

# -------------------------
# Plot 2: Stretched firing rates
fig, ax = plt.subplots(figsize=(7, 5))

for idx in range(firing_rates_stretched.shape[0]):
    ax.plot(
        new_time,
        firing_rates_stretched[idx],
        label=f"Neuron {idx+1}",
        alpha=0.7,
        color=colors[idx % len(colors)],
    )

ax.plot(
    new_time,
    avg_firing_rate_stretched,
    color="black",
    linewidth=2.5,
    label="Average firing rate (warped)",
)

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("Warped Firing Rates", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()


# t-test fig:

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu


# Function to calculate firing rate using sliding window
def calc_fr_rr(state_spike_array, window_length, step_size):
    num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
    nrn_num, time_num = state_spike_array.shape
    firing_rate = np.zeros((nrn_num, num_bins))
    for bin_ini in range(num_bins):
        start_bin = bin_ini * step_size
        end_bin = min(start_bin + window_length, time_num)
        num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
        firing_rate[:, bin_ini] = num_spikes / (
            window_length * 0.001
        )  # Convert to Hz (window in ms)
    return firing_rate


# -------------------------
# Settings
np.random.seed(42)
n_neurons = 10
total_duration = 600  # ms per neuron
window_length = 180  # ms window
step_size = 20  # ms step

# Define first and second half durations
first_half_duration = total_duration // 2
second_half_duration = total_duration - first_half_duration

# Pre-allocate
all_firing_rates = []

# -------------------------
# Generate spike trains for each neuron
for neuron_idx in range(n_neurons):
    # Base firing probabilities
    first_half_base_rate = 3 / 1000  # 3 Hz
    second_half_base_rate = 7 / 1000  # 7 Hz

    # Structured modulation + noise
    time_first = np.linspace(0, 1, first_half_duration)
    time_second = np.linspace(0, 1, second_half_duration)

    first_half_modulation = 0.2 * np.sin(
        2 * np.pi * 2 * time_first
    ) + 0.5 * np.random.randn(first_half_duration)
    second_half_modulation = 0.2 * np.sin(
        2 * np.pi * 3 * time_second
    ) + 0.5 * np.random.randn(second_half_duration)

    first_half_spike_prob = np.clip(
        first_half_base_rate * (1 + first_half_modulation), 0.0005, 0.05
    )
    second_half_spike_prob = np.clip(
        second_half_base_rate * (1 + second_half_modulation), 0.0005, 0.05
    )

    # Generate spikes
    first_half_spikes = np.random.binomial(1, first_half_spike_prob)
    second_half_spikes = np.random.binomial(1, second_half_spike_prob)

    spike_train = np.concatenate([first_half_spikes, second_half_spikes])

    # Calculate firing rate for this neuron
    spike_train_reshaped = spike_train.reshape(1, -1)
    firing_rate = calc_fr_rr(spike_train_reshaped, window_length, step_size)
    firing_rate = firing_rate[0]  # (drop neuron axis)

    all_firing_rates.append(firing_rate)

# Stack into an array
all_firing_rates = np.vstack(all_firing_rates)  # Shape: (n_neurons, time_bins)

# Average firing rate across neurons
avg_firing_rate = np.nanmean(all_firing_rates, axis=0)

# -------------------------
# Time vector
time_vector = np.arange(avg_firing_rate.shape[0]) * step_size + window_length / 2

# -------------------------
# Statistical test: first half vs second half of the average
split_idx = avg_firing_rate.shape[0] // 2
first_half_fr = avg_firing_rate[:split_idx]
second_half_fr = avg_firing_rate[split_idx:]

u_stat, p_value = mannwhitneyu(first_half_fr, second_half_fr, alternative="two-sided")

print(f"Mann-Whitney U test (average): U = {u_stat:.3f}, p = {p_value:.5f}")
if p_value < 0.05:
    print(
        "✅ Statistically significant difference between halves (average firing rate)."
    )
else:
    print("❌ No significant difference between halves.")

# -------------------------
# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(time_vector, avg_firing_rate, color="black", linewidth=2.5, label="Firing Rate")

# Shade halves
ax.axvspan(time_vector[0], 300, color="#FC6ECB", alpha=0.3, label="First Half")
ax.axvspan(300, time_vector[-1], color="#C7CEE7", alpha=0.3, label="Second Half")

# Line at split
ax.axvline(300, color="#D00000", linestyle="--", label="Split")

ax.set_xlabel("Time (ms)", fontsize=16)
ax.set_ylabel("Firing Rate (Hz)", fontsize=16)
ax.set_title("", fontsize=18)
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.show()

# colors = ['#FC6ECB', '#BC7FBC', '#E8BBDF', '#C7CEE7']
