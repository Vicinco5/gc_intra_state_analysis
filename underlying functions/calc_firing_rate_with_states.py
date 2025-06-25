#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:40:07 2024

@author: vincentcalia-bogan
"""

# firing rate calculation script
# NOTE: This may very well change (a lot)
# up to date as of 6/21/2024
import numpy as np
import xarray as xr

# def calc_fr_states(spike_array, changepoints, window_length, step_size):
#     def calc_fr_rr(state_spike_array, window_length, step_size):
#         num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
#         nrn_num, time_num = state_spike_array.shape
#         firing_rate = np.zeros((nrn_num, num_bins))
#         for bin_ini in range(num_bins):
#             start_bin = bin_ini * step_size
#             end_bin = min(start_bin + window_length, time_num)
#             num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
#             firing_rate[:, bin_ini] = num_spikes / (window_length / 1000)  # Convert to Hz
#         return firing_rate
#     # highly possible that as this gets more complex firing rate should become it's own file
#     # and not a function within a function
#     taste_num, trial_num, nrn_num, time_num = spike_array.shape
#     # calc max num of time bins for init x-arrays
#     max_bins = (time_num - window_length) // step_size + 1
#     # Creating xarray DataArrays for easier datamanagmnet
#     state_firing_rates = xr.DataArray(
#         data=np.nan,
#         dims=["taste", "trial", "segment", "neuron", "time_bin"],
#         coords={
#             "taste": range(taste_num),
#             "trial": range(trial_num),
#             "segment": range(max(len(cps) for cps_list in changepoints for cps in cps_list)),
#             "neuron": range(nrn_num),
#             "time_bin": range(max_bins)  # set maximum possible bins
#         }
#     )
#     # segment is analagous to changepoint
#     state_spike_arrays = xr.DataArray(
#         data=np.nan,
#         dims=["taste", "trial", "segment", "neuron", "time"],
#         coords={
#             "taste": range(taste_num),
#             "trial": range(trial_num),
#             "segment": range(max(len(cps) for cps_list in changepoints for cps in cps_list)),
#             "neuron": range(nrn_num),
#             "time": range(time_num)
#         }
#     )
#     for taste_idx in range(taste_num):
#         for trial_idx in range(trial_num):
#             trial_spike_array = spike_array[taste_idx, trial_idx, :, :]
#             trial_changepoints = changepoints[taste_idx][trial_idx] # Ensuring the last segment captures all remaining data
#             if not isinstance(changepoints[taste_idx], np.ndarray) or np.isnan(changepoints[taste_idx]).any():
#                 print(f"Invalid changepoints for taste {taste_idx}, trial {trial_idx}. Skipping...")
#                 continue
#             start_time = 2000 # start 2000 ms into the rec-- always change to 2000
#             for cp_idx, changepoint in enumerate(trial_changepoints):
#                 end_time = changepoint if cp_idx < len(trial_changepoints) - 1 else (changepoint + 000) # reset to + 0
#                 # start before stim delivery and then end at cp 3 rather than going to very end of rec
#                 if end_time <= start_time:  # Checking if the segment length would be zero or negative
#                     print(f"No data to process for segment starting at {start_time} and ending at {end_time}. Skipping...")
#                     continue  # Skip processing this segment
#                 state_spike_array = trial_spike_array[:, start_time:end_time]
#                 firing_rate = calc_fr_rr(state_spike_array, window_length, step_size)

#                 if firing_rate is not None:
#                     num_bins = firing_rate.shape[1]
#                     # Ensure padding matches the xarray size exactly
#                     padded_firing_rate = np.pad(firing_rate, ((0, 0), (0, max_bins - num_bins)), mode='constant', constant_values=np.nan)
#                     # what the above line is doing is making all arrays the same length array by
#                     # padding extra entries with nan values. This is done for xarray and storage efficency
#                     # Interpolation function eliminates the padding. Padding to the maximum amount of time bins
#                     # Directly use all the `time_bin` indices since padded to `max_bins`
#                     state_firing_rates.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_firing_rate
#                     state_spike_arrays.loc[taste_idx, trial_idx, cp_idx, :, :end_time-start_time - 1] = state_spike_array
#                 start_time = end_time
#     return state_firing_rates, state_spike_arrays


# MODIFIED FUNCTION TO BAKE IN THE INTERPOLATION / WARPING IN FUNCTION. EXPERIMENTAL.

from scipy.interpolate import interp1d


def calc_fr_states(spike_array, changepoints, window_length, step_size):
    def calc_fr_rr(state_spike_array, window_length, step_size):
        num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
        nrn_num, time_num = state_spike_array.shape
        firing_rate = np.zeros((nrn_num, num_bins))
        for bin_ini in range(num_bins):
            start_bin = bin_ini * step_size
            end_bin = min(start_bin + window_length, time_num)
            num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
            firing_rate[:, bin_ini] = num_spikes / (
                window_length / 1000
            )  # Convert to Hz CHECK MATH !!! -- this may explain some of what I'm seeing
            # in terms of 8's and 4's rather than 1's and 2's. YIKES. Fix this. maybe?
        return firing_rate

    taste_num, trial_num, nrn_num, time_num = spike_array.shape
    # this is not a good way to be definign max_bins (?)
    # it's also an incorrect way to be defining max_bins

    max_bins = (time_num - window_length) // step_size + 1

    state_firing_rates = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time_bin"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(
                max(len(cps) for cps_list in changepoints for cps in cps_list)
            ),
            # gives a range of 0 to howevermany rather than hardcoding
            "neuron": range(nrn_num),
            "time_bin": range(max_bins),
        },
    )

    state_firing_rates_warped = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time_bin"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(
                max(len(cps) for cps_list in changepoints for cps in cps_list)
            ),
            "neuron": range(nrn_num),
            "time_bin": range(max_bins),
        },
    )

    state_spike_arrays = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(
                max(len(cps) for cps_list in changepoints for cps in cps_list)
            ),
            "neuron": range(nrn_num),
            "time": range(time_num),
        },
    )

    state_spike_arrays_warped = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time"],
        coords={
            "taste": range(taste_num),
            "trial": range(trial_num),
            "segment": range(
                max(len(cps) for cps_list in changepoints for cps in cps_list)
            ),
            "neuron": range(nrn_num),
            "time": range(time_num),
        },
    )

    for taste_idx in range(taste_num):
        # for the purposes of warping to the desired len this is what we need to be doing
        # also I think it might be better to warp the spike array rather than the firing rate...
        # supposedly warping is now working properly, finally
        war_cp = changepoints[taste_idx]
        first_val = war_cp[:, 0]
        sec_val = war_cp[:, 1]
        third_val = war_cp[:, 2]
        fourth_val = war_cp[:, 3]
        max_first = np.max(first_val)
        max_second = np.max(sec_val)
        max_third = np.max(third_val)
        max_fourth = np.max(fourth_val)
        largest_cps = np.array(
            [max_first, max_second, max_third, max_fourth]
        )  # pulling these out of all trials
        for trial_idx in range(trial_num):
            trial_spike_array = spike_array[taste_idx, trial_idx, :, :]
            trial_changepoints = changepoints[taste_idx][trial_idx]
            if (
                not isinstance(changepoints[taste_idx], np.ndarray)
                or np.isnan(changepoints[taste_idx]).any()
            ):
                print(
                    f"Invalid changepoints for taste {taste_idx}, trial {trial_idx}. Skipping..."
                )
                continue
            start_time = 2000
            # note: no separate start time for warping, as we want to start at each trial's original time and warp it to the end
            for cp_idx, changepoint in enumerate(trial_changepoints):
                end_time = (
                    changepoint
                    if cp_idx < len(trial_changepoints) - 1
                    else (changepoint + 0)
                )
                if end_time <= start_time:
                    print(
                        f"No data to process for segment starting at {start_time} and ending at {end_time}. Skipping..."
                    )
                    continue
                war_end = largest_cps[
                    cp_idx
                ]  # warped end_time rather than regular end_time

                # unwarped -- straightforward handling of spike array
                state_spike_array = trial_spike_array[:, start_time:end_time]
                firing_rate = calc_fr_rr(state_spike_array, window_length, step_size)
                # building warping into spike array: WANT TO DO WARPING ON THE SPIKE TRAINS NOT THE CALCULATED FIRING RATE
                # also make sure that TRIALS are being warped to overall STATE lens -- make sure that the right thing is being warped
                if state_spike_array.shape[1] < 2:
                    print(
                        f"Not enough data points for interpolation. Skipping segment {cp_idx}..."
                    )
                    continue  # forgot to add this
                # issue here with the construction of war_state_spike array-- getting odd lenghs
                original_len = np.linspace(
                    start_time, end_time, state_spike_array.shape[1]
                )  # linearly spaced vectors for warping
                # must ensure we have the right len vector for war_len
                num_war_pts = int(war_end - start_time)
                war_len = np.linspace(start_time, war_end, num_war_pts)
                war_state_spike_array = np.zeros(
                    (state_spike_array.shape[0], len(war_len))
                )

                for neuron_idx in range(state_spike_array.shape[0]):
                    f = interp1d(
                        original_len,
                        state_spike_array[neuron_idx, :],
                        kind="nearest",
                        fill_value="extrapolate",
                    )
                    war_state_spike_array[neuron_idx, :] = f(war_len)
                war_firing_rate = calc_fr_rr(
                    war_state_spike_array, window_length, step_size
                )
                # note that even though we're warping things, I'm still padding-- so that storage in arrays is easier
                num_war_bins = war_firing_rate.shape[1]
                padded_war_fr = np.pad(
                    war_firing_rate,
                    ((0, 0), (0, max_bins - num_war_bins)),
                    mode="constant",
                    constant_values=np.nan,
                )
                state_firing_rates_warped.loc[taste_idx, trial_idx, cp_idx, :, :] = (
                    padded_war_fr
                )
                state_spike_arrays_warped.loc[
                    taste_idx, trial_idx, cp_idx, :, : war_end - start_time - 1
                ] = war_state_spike_array

                if firing_rate is not None:
                    # padding with nans
                    num_bins = firing_rate.shape[1]
                    padded_firing_rate = np.pad(
                        firing_rate,
                        ((0, 0), (0, max_bins - num_bins)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    state_firing_rates.loc[taste_idx, trial_idx, cp_idx, :, :] = (
                        padded_firing_rate
                    )
                    state_spike_arrays.loc[
                        taste_idx, trial_idx, cp_idx, :, : end_time - start_time - 1
                    ] = state_spike_array

                start_time = end_time

    return (
        state_firing_rates,
        state_spike_arrays,
        state_firing_rates_warped,
        state_spike_arrays_warped,
    )


# deprecated, not working anyways firing rate warping:
# also adding interpolation into this-- I don't think it's needed to index over neurons

# deprecated warping
# # Interpolation
# for nrn_idx in range(nrn_num):
#     fr_vector = firing_rate[nrn_idx, :]
#     if not np.isnan(fr_vector).all():
#         original_length = len(fr_vector)
#         if original_length < 2:
#             continue
#         time_original = np.linspace(0, 1, original_length)
#         time_new = np.linspace(0, 1, max_bins)
#         interpolator = interp1d(time_original, fr_vector[:original_length], kind='nearest')
#         interpolated_vector = interpolator(time_new)
#         state_firing_rates_interpolated.loc[taste_idx, trial_idx, cp_idx, nrn_idx, :] = interpolated_vector
