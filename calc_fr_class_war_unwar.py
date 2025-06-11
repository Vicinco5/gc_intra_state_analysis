#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:22:14 2025

Optimizing spike train and firing rate calculations-- calling a class and setting flags for true/false

@author: vincentcalia-bogan
"""
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


class CalcFRStates:
    """
    A class version of calc_fr_states that can selectively compute:
       1) Unwarped spike arrays
       2) Unwarped firing rates
       3) Warped spike arrays
       4) Warped firing rates
    
    By default, it computes all four (to serve as a drop-in replacement).
    You can disable any subset to skip unneeded computations.

    Usage:
        calc = CalcFRStates(
            spike_array=..., 
            changepoints=..., 
            window_length=..., 
            step_size=..., 
            compute_unwarped_spike_arrays=True, 
            compute_unwarped_firing_rates=True,
            compute_warped_spike_arrays=True,
            compute_warped_firing_rates=True
        )
        
        sfr, ssa, sfrw, ssaw = calc.run()
        # returns the same four outputs as the original calc_fr_states, in the same order
    """

    def __init__(
        self,
        spike_array,
        changepoints,
        window_length,
        step_size,
        compute_unwarped_spike_arrays=True,
        compute_unwarped_firing_rates=True,
        compute_warped_spike_arrays=True,
        compute_warped_firing_rates=True,
        fixed_warp_duration=None  # in ms, e.g., 1000
    ):
        """
        Parameters
        ----------
        spike_array : np.ndarray
            Shape: (taste_num, trial_num, nrn_num, time_num)
        changepoints : list or array-like
            Changepoints for each taste/trial
        window_length : int
        step_size : int

        compute_unwarped_spike_arrays : bool
            Whether to compute and store unwarped spike arrays.
        compute_unwarped_firing_rates : bool
            Whether to compute and store unwarped firing rates.
        compute_warped_spike_arrays : bool
            Whether to compute and store warped spike arrays.
        compute_warped_firing_rates : bool
            Whether to compute and store warped firing rates.
        """

        self.spike_array = spike_array
        self.changepoints = changepoints
        self.window_length = window_length
        self.step_size = step_size

        self.compute_unwarped_spike_arrays = compute_unwarped_spike_arrays
        self.compute_unwarped_firing_rates = compute_unwarped_firing_rates
        self.compute_warped_spike_arrays = compute_warped_spike_arrays
        self.compute_warped_firing_rates = compute_warped_firing_rates

        # Outputs (initialized to None, filled upon .run())
        self.fixed_warp_duration = fixed_warp_duration
        self.state_firing_rates = None
        self.state_spike_arrays = None
        self.state_firing_rates_warped = None
        self.state_spike_arrays_warped = None

    def calc_fr_rr(self, state_spike_array, window_length, step_size):
        """
        Identical logic to the original nested calc_fr_rr function.
        Computes the firing rate in Hz for each neuron using a sliding window.
        """
        num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
        nrn_num, time_num = state_spike_array.shape
        firing_rate = np.zeros((nrn_num, num_bins))

        for bin_ini in range(num_bins):
            start_bin = bin_ini * step_size
            end_bin = min(start_bin + window_length, time_num)
            num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
            # Convert to Hz
            firing_rate[:, bin_ini] = num_spikes / (window_length / 1000.0)

        return firing_rate

    def run(self):
        """
        Replicates the logic of the original calc_fr_states, but now you can
        selectively enable/disable unwarped or warped spike arrays and firing rates.
        
        Returns (in the same order as the original function):
        ----------------------------------------------------
        state_firing_rates           (unwarped),
        state_spike_arrays           (unwarped),
        state_firing_rates_warped    (warped),
        state_spike_arrays_warped    (warped).

        Any that were disabled will be None.
        """
        # Extract parameters
        spike_array   = self.spike_array
        changepoints  = self.changepoints
        window_length = self.window_length
        step_size     = self.step_size

        # Dimensions
        taste_num, trial_num, nrn_num, time_num = spike_array.shape
        max_bins = (time_num - window_length) // step_size + 1
        if self.fixed_warp_duration is not None:
            max_bins = (self.fixed_warp_duration - window_length) // step_size + 1


        # Figure out the maximum number of segments across all tastes/trials
        max_segments = max(len(cps) for cps_list in changepoints for cps in cps_list)

        # -------------
        # Build empty xarray templates only for data we intend to compute.
        # -------------

        # Unwarped firing rates
        if self.compute_unwarped_firing_rates:
            self.state_firing_rates = xr.DataArray(
                data=np.nan,
                dims=["taste", "trial", "segment", "neuron", "time_bin"],
                coords={
                    "taste": range(taste_num),
                    "trial": range(trial_num),
                    "segment": range(max_segments),
                    "neuron": range(nrn_num),
                    "time_bin": range(max_bins)
                }
            )

        # Unwarped spike arrays
        if self.compute_unwarped_spike_arrays:
            self.state_spike_arrays = xr.DataArray(
                data=np.nan,
                dims=["taste", "trial", "segment", "neuron", "time"],
                coords={
                    "taste": range(taste_num),
                    "trial": range(trial_num),
                    "segment": range(max_segments),
                    "neuron": range(nrn_num),
                    "time": range(time_num)
                }
            )

        # Warped firing rates
        if self.compute_warped_firing_rates:
            self.state_firing_rates_warped = xr.DataArray(
                data=np.nan,
                dims=["taste", "trial", "segment", "neuron", "time_bin"],
                coords={
                    "taste": range(taste_num),
                    "trial": range(trial_num),
                    "segment": range(max_segments),
                    "neuron": range(nrn_num),
                    "time_bin": range(max_bins)
                }
            )

        # Warped spike arrays
        if self.compute_warped_spike_arrays:
            self.state_spike_arrays_warped = xr.DataArray(
                data=np.nan,
                dims=["taste", "trial", "segment", "neuron", "time"],
                coords={
                    "taste": range(taste_num),
                    "trial": range(trial_num),
                    "segment": range(max_segments),
                    "neuron": range(nrn_num),
                    "time": range(time_num)
                }
            )

        # -------------
        # Main loop over tastes, trials, changepoints
        # -------------
        for taste_idx in range(taste_num):
            # Identify largest CP across trials for warping
            war_cp = changepoints[taste_idx]
            if not isinstance(war_cp, np.ndarray):
                war_cp = np.array(war_cp)

            # Skip taste if invalid
            if np.isnan(war_cp).any():
                continue

            # For each segment, find max across all trials (assuming exactly 4 CPs):
            first_val  = war_cp[:, 0]
            second_val = war_cp[:, 1]
            third_val  = war_cp[:, 2]
            fourth_val = war_cp[:, 3]

            max_first  = np.max(first_val)
            max_second = np.max(second_val)
            max_third  = np.max(third_val)
            max_fourth = np.max(fourth_val)
            largest_cps = np.array([max_first, max_second, max_third, max_fourth])

            for trial_idx in range(trial_num):
                trial_spike_array = spike_array[taste_idx, trial_idx, :, :]
                trial_changepoints = war_cp[trial_idx]

                if not isinstance(trial_changepoints, np.ndarray) or np.isnan(trial_changepoints).any():
                    print(f"Invalid changepoints for taste {taste_idx}, trial {trial_idx}. Skipping...")
                    continue

                start_time = 1500 # 2000 is stim delivery--- change to 1500 for time prior to
                for cp_idx, changepoint in enumerate(trial_changepoints):
                    # if last CP, it's effectively the same as "changepoint + 0"
                    end_time = changepoint if cp_idx < len(trial_changepoints) - 1 else changepoint

                    if end_time <= start_time:
                        print(f"No data for segment from {start_time} to {end_time}. Skipping...")
                        continue

                    war_end = largest_cps[cp_idx]

                    # Extract the unwarped segment (if relevant)
                    segment_unwarped = trial_spike_array[:, start_time:end_time]

                    # Possibly compute unwarped FR
                    if self.compute_unwarped_firing_rates:
                        if segment_unwarped.shape[1] > 0:  # non-empty
                            unwarped_fr = self.calc_fr_rr(segment_unwarped, window_length, step_size)
                        else:
                            unwarped_fr = None
                    else:
                        unwarped_fr = None

                    # Prepare for warp if needed
                    if self.compute_warped_spike_arrays or self.compute_warped_firing_rates:
                        if segment_unwarped.shape[1] < 2:
                            # Not enough points for interpolation
                            war_segment = None
                            war_fr = None
                            print(f"Not enough points to interpolate for segment {cp_idx}. Skipping warp...")
                        else:
                            original_len = np.linspace(0, 1, segment_unwarped.shape[1])
                            
                            if self.fixed_warp_duration is not None:
                                num_war_pts = self.fixed_warp_duration  # e.g., 1000 ms
                            else:
                                num_war_pts = int(war_end - start_time)
                            
                            war_len = np.linspace(0, 1, num_war_pts)
                            
                            war_segment = np.zeros((segment_unwarped.shape[0], len(war_len)))

                            # Interpolate each neuron with nearest
                            for neuron_idx in range(segment_unwarped.shape[0]):
                                f = interp1d(
                                    original_len,
                                    segment_unwarped[neuron_idx, :],
                                    kind='nearest',
                                    bounds_error=False,
                                    fill_value='extrapolate'
                                )
                                war_segment[neuron_idx, :] = f(war_len)
# Think about interpolating to a standard value across all the hings 
                            # Possibly compute FR for warped segment
                            if self.compute_warped_firing_rates:
                                war_fr = self.calc_fr_rr(war_segment, window_length, step_size)
                            else:
                                war_fr = None
                    else:
                        # Warping disabled
                        war_segment = None
                        war_fr = None

                    # --- Store unwarped results (remove mismatch bug) ---
                    if self.compute_unwarped_spike_arrays and segment_unwarped.shape[1] > 0:
                        # segment_unwarped.shape[1] == end_time - start_time
                        self.state_spike_arrays.loc[taste_idx, trial_idx, cp_idx, :, :end_time - start_time-1] = (
                            segment_unwarped
                        )
# max bins is not an attribute -- no self., just max_bins
                    if self.compute_unwarped_firing_rates and (unwarped_fr is not None):
                        num_bins = unwarped_fr.shape[1]
                        pad_width = max(max_bins - num_bins, 0)  # safe padding only
                    
                        if pad_width > 0:
                            padded_fr = np.pad(
                                unwarped_fr,
                                ((0, 0), (0, pad_width)),
                                mode='constant',
                                constant_values=np.nan
                            )
                        else:
                            # Trial is already at or longer than max_bins — use as is (or truncate if needed)
                            padded_fr = unwarped_fr[:, :max_bins]  # truncate if it's too long
                    
                        self.state_firing_rates.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_fr

                    # bugged method below-- padding indexing 
                    # if self.compute_unwarped_firing_rates and (unwarped_fr is not None):
                    #     num_bins = unwarped_fr.shape[1]
                    #     padded_fr = np.pad(
                    #         unwarped_fr,
                    #         ((0, 0), (0, max_bins - num_bins)),
                    #         mode='constant',
                    #         constant_values=np.nan
                    #     )
                    #     self.state_firing_rates.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_fr

                    # --- Store warped results (remove the "-1" bug) ---
                    if self.compute_warped_spike_arrays and (war_segment is not None):
                        # war_segment.shape[1] == war_end - start_time
                        self.state_spike_arrays_warped.loc[taste_idx, trial_idx, cp_idx, :, :war_segment.shape[1]-1] = (
                            war_segment
                        )

                    if self.compute_warped_firing_rates and (war_fr is not None):
                        num_war_bins = war_fr.shape[1]
                        padded_war_fr = np.pad(
                            war_fr,
                            ((0, 0), (0, max_bins - num_war_bins)),
                            mode='constant',
                            constant_values=np.nan
                        )
                        self.state_firing_rates_warped.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_war_fr

                    start_time = end_time

        # Return in the same order as the original function
        return (
            self.state_firing_rates,         # unwarped firing rates
            self.state_spike_arrays,         # unwarped spike arrays
            self.state_firing_rates_warped,  # warped firing rates
            self.state_spike_arrays_warped   # warped spike arrays
        )

# calc = CalcFRStates(
#     spike_array=spike_array,
#     changepoints=changepoints,
#     window_length=window_length,
#     step_size=step_size,
#     compute_unwarped_spike_arrays=True, 
#     compute_unwarped_firing_rates=False,
#     compute_warped_spike_arrays=False,
#     compute_warped_firing_rates=False
#     fixed_warp_duration=1000  # <-- Warp all segments to 1000 ms
# )
# sfr, ssa, sfrw, ssaw = calc.run()
# # 'sfr' = None, 'ssa' = unwarped arrays, 'sfrw' = None, 'ssaw' = None
# setting true/false will result in non/not none for various outputs. 
# If compute_warped=False, you'll get None for the last two return values.
# If compute_unwarped=False, you'll get None for the first two return values.
# Otherwise, you get the original four arrays as if from calc_fr_states.

# pre-warping fix-- old methodology that didn't quite work as intended: 
    
# class CalcFRStates:
#     """
#     A class version of calc_fr_states that can selectively compute:
#        1) Unwarped spike arrays
#        2) Unwarped firing rates
#        3) Warped spike arrays
#        4) Warped firing rates
    
#     By default, it computes all four (to serve as a drop-in replacement).
#     You can disable any subset to skip unneeded computations.

#     Usage:
#         calc = CalcFRStates(
#             spike_array=..., 
#             changepoints=..., 
#             window_length=..., 
#             step_size=..., 
#             compute_unwarped_spike_arrays=True, 
#             compute_unwarped_firing_rates=True,
#             compute_warped_spike_arrays=True,
#             compute_warped_firing_rates=True
#         )
        
#         sfr, ssa, sfrw, ssaw = calc.run()
#         # returns the same four outputs as the original calc_fr_states, in the same order
#     """

#     def __init__(
#         self,
#         spike_array,
#         changepoints,
#         window_length,
#         step_size,
#         compute_unwarped_spike_arrays=True,
#         compute_unwarped_firing_rates=True,
#         compute_warped_spike_arrays=True,
#         compute_warped_firing_rates=True,
#         fixed_warp_duration=None  # in ms, e.g., 1000
#     ):
#         """
#         Parameters
#         ----------
#         spike_array : np.ndarray
#             Shape: (taste_num, trial_num, nrn_num, time_num)
#         changepoints : list or array-like
#             Changepoints for each taste/trial
#         window_length : int
#         step_size : int

#         compute_unwarped_spike_arrays : bool
#             Whether to compute and store unwarped spike arrays.
#         compute_unwarped_firing_rates : bool
#             Whether to compute and store unwarped firing rates.
#         compute_warped_spike_arrays : bool
#             Whether to compute and store warped spike arrays.
#         compute_warped_firing_rates : bool
#             Whether to compute and store warped firing rates.
#         """

#         self.spike_array = spike_array
#         self.changepoints = changepoints
#         self.window_length = window_length
#         self.step_size = step_size

#         self.compute_unwarped_spike_arrays = compute_unwarped_spike_arrays
#         self.compute_unwarped_firing_rates = compute_unwarped_firing_rates
#         self.compute_warped_spike_arrays = compute_warped_spike_arrays
#         self.compute_warped_firing_rates = compute_warped_firing_rates

#         # Outputs (initialized to None, filled upon .run())
#         self.fixed_warp_duration = fixed_warp_duration
#         self.state_firing_rates = None
#         self.state_spike_arrays = None
#         self.state_firing_rates_warped = None
#         self.state_spike_arrays_warped = None

#     def calc_fr_rr(self, state_spike_array, window_length, step_size):
#         """
#         Identical logic to the original nested calc_fr_rr function.
#         Computes the firing rate in Hz for each neuron using a sliding window.
#         """
#         num_bins = max((state_spike_array.shape[1] - window_length) // step_size + 1, 1)
#         nrn_num, time_num = state_spike_array.shape
#         firing_rate = np.zeros((nrn_num, num_bins))

#         for bin_ini in range(num_bins):
#             start_bin = bin_ini * step_size
#             end_bin = min(start_bin + window_length, time_num)
#             num_spikes = np.sum(state_spike_array[:, start_bin:end_bin], axis=-1)
#             # Convert to Hz
#             firing_rate[:, bin_ini] = num_spikes / (window_length / 1000.0)

#         return firing_rate

#     def run(self):
#         """
#         Replicates the logic of the original calc_fr_states, but now you can
#         selectively enable/disable unwarped or warped spike arrays and firing rates.
        
#         Returns (in the same order as the original function):
#         ----------------------------------------------------
#         state_firing_rates           (unwarped),
#         state_spike_arrays           (unwarped),
#         state_firing_rates_warped    (warped),
#         state_spike_arrays_warped    (warped).

#         Any that were disabled will be None.
#         """
#         # Extract parameters
#         spike_array   = self.spike_array
#         changepoints  = self.changepoints
#         window_length = self.window_length
#         step_size     = self.step_size

#         # Dimensions
#         taste_num, trial_num, nrn_num, time_num = spike_array.shape
#         max_bins = (time_num - window_length) // step_size + 1
#         if self.fixed_warp_duration is not None:
#             max_bins = (self.fixed_warp_duration - window_length) // step_size + 1


#         # Figure out the maximum number of segments across all tastes/trials
#         max_segments = max(len(cps) for cps_list in changepoints for cps in cps_list)

#         # -------------
#         # Build empty xarray templates only for data we intend to compute.
#         # -------------

#         # Unwarped firing rates
#         if self.compute_unwarped_firing_rates:
#             self.state_firing_rates = xr.DataArray(
#                 data=np.nan,
#                 dims=["taste", "trial", "segment", "neuron", "time_bin"],
#                 coords={
#                     "taste": range(taste_num),
#                     "trial": range(trial_num),
#                     "segment": range(max_segments),
#                     "neuron": range(nrn_num),
#                     "time_bin": range(max_bins)
#                 }
#             )

#         # Unwarped spike arrays
#         if self.compute_unwarped_spike_arrays:
#             self.state_spike_arrays = xr.DataArray(
#                 data=np.nan,
#                 dims=["taste", "trial", "segment", "neuron", "time"],
#                 coords={
#                     "taste": range(taste_num),
#                     "trial": range(trial_num),
#                     "segment": range(max_segments),
#                     "neuron": range(nrn_num),
#                     "time": range(time_num)
#                 }
#             )

#         # Warped firing rates
#         if self.compute_warped_firing_rates:
#             self.state_firing_rates_warped = xr.DataArray(
#                 data=np.nan,
#                 dims=["taste", "trial", "segment", "neuron", "time_bin"],
#                 coords={
#                     "taste": range(taste_num),
#                     "trial": range(trial_num),
#                     "segment": range(max_segments),
#                     "neuron": range(nrn_num),
#                     "time_bin": range(max_bins)
#                 }
#             )

#         # Warped spike arrays
#         if self.compute_warped_spike_arrays:
#             self.state_spike_arrays_warped = xr.DataArray(
#                 data=np.nan,
#                 dims=["taste", "trial", "segment", "neuron", "time"],
#                 coords={
#                     "taste": range(taste_num),
#                     "trial": range(trial_num),
#                     "segment": range(max_segments),
#                     "neuron": range(nrn_num),
#                     "time": range(time_num)
#                 }
#             )

#         # -------------
#         # Main loop over tastes, trials, changepoints
#         # -------------
#         for taste_idx in range(taste_num):
#             # Identify largest CP across trials for warping
#             war_cp = changepoints[taste_idx]
#             if not isinstance(war_cp, np.ndarray):
#                 war_cp = np.array(war_cp)

#             # Skip taste if invalid
#             if np.isnan(war_cp).any():
#                 continue

#             # For each segment, find max across all trials (assuming exactly 4 CPs):
#             first_val  = war_cp[:, 0]
#             second_val = war_cp[:, 1]
#             third_val  = war_cp[:, 2]
#             fourth_val = war_cp[:, 3]

#             max_first  = np.max(first_val)
#             max_second = np.max(second_val)
#             max_third  = np.max(third_val)
#             max_fourth = np.max(fourth_val)
#             largest_cps = np.array([max_first, max_second, max_third, max_fourth])

#             for trial_idx in range(trial_num):
#                 trial_spike_array = spike_array[taste_idx, trial_idx, :, :]
#                 trial_changepoints = war_cp[trial_idx]

#                 if not isinstance(trial_changepoints, np.ndarray) or np.isnan(trial_changepoints).any():
#                     print(f"Invalid changepoints for taste {taste_idx}, trial {trial_idx}. Skipping...")
#                     continue

#                 start_time = 2000
#                 for cp_idx, changepoint in enumerate(trial_changepoints):
#                     # if last CP, it's effectively the same as "changepoint + 0"
#                     end_time = changepoint if cp_idx < len(trial_changepoints) - 1 else changepoint

#                     if end_time <= start_time:
#                         print(f"No data for segment from {start_time} to {end_time}. Skipping...")
#                         continue

#                     war_end = largest_cps[cp_idx]

#                     # Extract the unwarped segment (if relevant)
#                     segment_unwarped = trial_spike_array[:, start_time:end_time]

#                     # Possibly compute unwarped FR
#                     if self.compute_unwarped_firing_rates:
#                         if segment_unwarped.shape[1] > 0:  # non-empty
#                             unwarped_fr = self.calc_fr_rr(segment_unwarped, window_length, step_size)
#                         else:
#                             unwarped_fr = None
#                     else:
#                         unwarped_fr = None

#                     # Prepare for warp if needed
#                     if self.compute_warped_spike_arrays or self.compute_warped_firing_rates:
#                         if segment_unwarped.shape[1] < 2:
#                             # Not enough points for interpolation
#                             war_segment = None
#                             war_fr = None
#                             print(f"Not enough points to interpolate for segment {cp_idx}. Skipping warp...")
#                         else:
#                             original_len = np.linspace(start_time, end_time, segment_unwarped.shape[1])
#                             if self.fixed_warp_duration is not None:
#                                 num_war_pts = self.fixed_warp_duration
#                                 war_end_actual = start_time + self.fixed_warp_duration
#                             else:
#                                 num_war_pts = int(war_end - start_time)
#                                 war_end_actual = war_end
                            
#                             war_len = np.linspace(start_time, war_end_actual, num_war_pts)
#                             war_segment = np.zeros((segment_unwarped.shape[0], len(war_len)))

#                             # Interpolate each neuron with linear
#                             for neuron_idx in range(segment_unwarped.shape[0]):
#                                 f = interp1d(
#                                     original_len,
#                                     segment_unwarped[neuron_idx, :],
#                                     kind='linear',
#                                     bounds_error=False,
#                                     fill_value='extrapolate'
#                                 )
#                                 war_segment[neuron_idx, :] = f(war_len)
# # Think about interpolating to a standard value across all the hings 
#                             # Possibly compute FR for warped segment
#                             if self.compute_warped_firing_rates:
#                                 war_fr = self.calc_fr_rr(war_segment, window_length, step_size)
#                             else:
#                                 war_fr = None
#                     else:
#                         # Warping disabled
#                         war_segment = None
#                         war_fr = None

#                     # --- Store unwarped results (remove mismatch bug) ---
#                     if self.compute_unwarped_spike_arrays and segment_unwarped.shape[1] > 0:
#                         # segment_unwarped.shape[1] == end_time - start_time
#                         self.state_spike_arrays.loc[taste_idx, trial_idx, cp_idx, :, :end_time - start_time-1] = (
#                             segment_unwarped
#                         )

#                     if self.compute_unwarped_firing_rates and (unwarped_fr is not None):
#                         num_bins = unwarped_fr.shape[1]
#                         padded_fr = np.pad(
#                             unwarped_fr,
#                             ((0, 0), (0, max_bins - num_bins)),
#                             mode='constant',
#                             constant_values=np.nan
#                         )
#                         self.state_firing_rates.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_fr

#                     # --- Store warped results (remove the "-1" bug) ---
#                     if self.compute_warped_spike_arrays and (war_segment is not None):
#                         # war_segment.shape[1] == war_end - start_time
#                         self.state_spike_arrays_warped.loc[taste_idx, trial_idx, cp_idx, :, :war_segment.shape[1]-1] = (
#                             war_segment
#                         )

#                     if self.compute_warped_firing_rates and (war_fr is not None):
#                         num_war_bins = war_fr.shape[1]
#                         padded_war_fr = np.pad(
#                             war_fr,
#                             ((0, 0), (0, max_bins - num_war_bins)),
#                             mode='constant',
#                             constant_values=np.nan
#                         )
#                         self.state_firing_rates_warped.loc[taste_idx, trial_idx, cp_idx, :, :] = padded_war_fr

#                     start_time = end_time

#         # Return in the same order as the original function
#         return (
#             self.state_firing_rates,         # unwarped firing rates
#             self.state_spike_arrays,         # unwarped spike arrays
#             self.state_firing_rates_warped,  # warped firing rates
#             self.state_spike_arrays_warped   # warped spike arrays
#         )
