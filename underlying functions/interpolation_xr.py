#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:10:54 2024

@author: vincentcalia-bogan
"""
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

# interpolation code, last updated 5/29/24
# new type of interpolation using xarray for the firing rate stuff -- notate all this stuff
# this one works properly (eliminates nan values)-- watch it with the type of interpolation

# return exactly the same datatype as calc_fr_states


def interpolate_and_average_all_tastes(state_firing_rates):
    # Initialize lists to store the final results for all tastes and segments
    all_tastes_averaged = []
    all_interpolated_states = []

    # Iterate through each taste
    for taste_idx in range(state_firing_rates.sizes["taste"]):
        taste_data = state_firing_rates.sel(taste=taste_idx)
        changepoint_averages = []
        taste_interpolated_states = []

        # Iterate through each segment (changepoint)
        for state_idx in range(taste_data.sizes["segment"]):
            segment_data = taste_data.sel(segment=state_idx)

            # Collect lengths of all trials and calculate the median length for interpolation
            lengths = segment_data.notnull().sum(dim=["time_bin", "neuron"])
            median_length = int(lengths.median())

            # Container for storing interpolated arrays for each trial
            interpolated_states = []

            # Interpolate each trial to the median length
            for trial_idx in range(segment_data.sizes["trial"]):
                trial_data = segment_data.sel(trial=trial_idx)
                interpolated_array = np.full(
                    (trial_data.sizes["neuron"], median_length), np.nan
                )  # Prepare container with NaNs

                # Interpolate the trial to the median length
                for nrn_idx in range(trial_data.sizes["neuron"]):
                    neuron_data = trial_data.sel(neuron=nrn_idx).values
                    valid_mask = ~np.isnan(neuron_data)

                    if valid_mask.sum() > 1:  # Need at least 2 points to interpolate
                        valid_time = np.arange(len(neuron_data))[valid_mask]
                        valid_data = neuron_data[valid_mask]
                        interp_func = interp1d(
                            valid_time,
                            valid_data,
                            kind="nearest",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        interpolated_array[nrn_idx] = interp_func(
                            np.linspace(0, len(neuron_data) - 1, median_length)
                        )

                interpolated_states.append(interpolated_array)

            # Convert the interpolated states to an xarray DataArray
            interpolated_xr = xr.DataArray(
                interpolated_states,
                dims=["trial", "neuron", "time_bin"],
                coords={
                    "trial": range(segment_data.sizes["trial"]),
                    "neuron": range(trial_data.sizes["neuron"]),
                    "time_bin": range(median_length),
                },
            )

            # Calculate the average across trials
            averaged_state = interpolated_xr.mean(dim="trial", skipna=True)
            changepoint_averages.append(averaged_state)
            taste_interpolated_states.append(interpolated_xr)

        # Concatenate all segments for this taste
        all_tastes_averaged.append(xr.concat(changepoint_averages, dim="segment"))
        all_interpolated_states.append(
            xr.concat(taste_interpolated_states, dim="segment")
        )

    # Concatenate all tastes into the final DataArrays
    final_firing_rates = xr.concat(all_tastes_averaged, dim="taste")
    final_spike_arrays = xr.concat(all_interpolated_states, dim="taste")

    return final_firing_rates, final_spike_arrays


## Completely new attempt-- as the above is just fucked and I don't super know why

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


def interpolate_firing_rates(state_firing_rates, state_spike_arrays):
    taste_num, trial_num, segment_num, nrn_num, max_bins = state_firing_rates.shape

    # Create new xarray DataArrays to store the interpolated firing rates and spike arrays
    interpolated_firing_rates = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time_bin"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(segment_num),
            "neuron": range(nrn_num),
            "time_bin": range(max_bins),
        },
    )

    interpolated_spike_arrays = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(segment_num),
            "neuron": range(nrn_num),
            "time": range(max_bins),
        },
    )

    # Interpolation process
    for taste_idx in range(taste_num):
        for seg_idx in range(segment_num):
            for nrn_idx in range(nrn_num):
                # Collect all non-nan firing rate vectors for this neuron, taste, and segment
                valid_firing_rates = []
                for trial_idx in range(trial_num):
                    fr_vector = state_firing_rates[
                        taste_idx, trial_idx, seg_idx, nrn_idx, :
                    ].values
                    if not np.isnan(fr_vector).all():  # Skip if all values are nan
                        valid_firing_rates.append(fr_vector[~np.isnan(fr_vector)])

                if len(valid_firing_rates) == 0:
                    continue  # Skip if no valid firing rates

                # Determine the maximum length among the valid firing rate vectors
                max_length = max(len(fr) for fr in valid_firing_rates)

                # Interpolate each firing rate vector to the maximum length
                for trial_idx in range(trial_num):
                    fr_vector = state_firing_rates[
                        taste_idx, trial_idx, seg_idx, nrn_idx, :
                    ].values
                    if not np.isnan(fr_vector).all():  # Skip if all values are nan
                        original_length = len(fr_vector[~np.isnan(fr_vector)])
                        if original_length < 2:
                            # If there's not enough data for interpolation, skip this trial
                            continue
                        time_original = np.linspace(0, 1, original_length)
                        time_new = np.linspace(0, 1, max_length)
                        interpolator = interp1d(
                            time_original,
                            fr_vector[~np.isnan(fr_vector)],
                            kind="linear",
                        )
                        interpolated_vector = interpolator(time_new)

                        # Ensure that the interpolated vector has the correct length
                        if len(interpolated_vector) > max_bins:
                            interpolated_vector = interpolated_vector[:max_bins]
                        elif len(interpolated_vector) < max_bins:
                            interpolated_vector = np.pad(
                                interpolated_vector,
                                (0, max_bins - len(interpolated_vector)),
                                mode="constant",
                                constant_values=np.nan,
                            )

                        # Store the interpolated vector in the new xarray
                        interpolated_firing_rates.loc[
                            taste_idx, trial_idx, seg_idx, nrn_idx, :
                        ] = interpolated_vector

    return interpolated_firing_rates, interpolated_spike_arrays


# 7/7-- this datatype appears to be bugged; reliant on a previous iteration of code. updated datatype is above.
# def interpolate_and_average_all_tastes(state_firing_rates):
#     # Handle the xarray DataArray directly
#     all_tastes_averaged = []
#     all_interpolated_states = []
#     for taste_idx in range(state_firing_rates.sizes['taste']):
#         taste_data = state_firing_rates.sel(taste=taste_idx)
#         changepoint_averages = []
#         taste_interpolated_states = []
#         # segment = changepoint
#         for state_idx in range(taste_data.sizes['segment']):
#             segment_data = taste_data.sel(segment=state_idx)
#             # Collect lengths and calculate median length for interpolation
#             lengths = segment_data.notnull().sum(dim='time_bin')
#             median_length = int(lengths.median())
#             # Container for storing interpolated arrays for each trial
#             interpolated_states = []
#             # Interpolate each trial to this median length
#             for trial_idx in range(segment_data.sizes['trial']):
#                 trial_data = segment_data.sel(trial=trial_idx)
#                 nrn_num = trial_data.sizes['neuron']
#                 #new_time = np.linspace(0, median_length - 1, median_length)
#                 interpolated_array = np.full((nrn_num, median_length), np.nan)  # Prepare container with NaNs
#                 # Interpolate all trials for each neuron-- use 'nearest' type as this is highly non-linear data
#                 # so nearest is the best we can do
#                 for nrn_idx in range(nrn_num):
#                     neuron_data = trial_data.sel(neuron=nrn_idx).values
#                     valid_mask = ~np.isnan(neuron_data)
#                     if valid_mask.sum() > 1:  # Need at least 2 points to interpolate
#                         valid_time = np.arange(len(neuron_data))[valid_mask]
#                         valid_data = neuron_data[valid_mask]
#                         interp_func = interp1d(valid_time, valid_data, kind='nearest', bounds_error=False, fill_value='extrapolate')
#                         interpolated_array[nrn_idx] = interp_func(np.linspace(0, median_length - 1, median_length))
#                 interpolated_states.append(interpolated_array)
#             # Create xarray for the interpolated states
#             interpolated_xr = xr.DataArray(interpolated_states,
#                                            dims=['trial', 'neuron', 'time_bin'],
#                                            coords={'trial': range(len(interpolated_states)), 'neuron': range(nrn_num), 'time_bin': range(median_length)})
#             averaged_state = interpolated_xr.mean(dim='trial', skipna=True)
#             changepoint_averages.append(averaged_state)
#             taste_interpolated_states.append(interpolated_xr)  # Store xarray of interpolated states
#         all_tastes_averaged.append(xr.concat(changepoint_averages, dim='segment'))
#         all_interpolated_states.append(xr.concat(taste_interpolated_states, dim='segment'))
#     return all_tastes_averaged, all_interpolated_states
