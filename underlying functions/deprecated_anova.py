#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:59:54 2024

@author: vincentcalia-bogan
"""

# anova code


## THIS WILL NOT WORK FOR A VERY GOOD REASON (SEE BELOW)
def calc_anova_cp(
    nrn,
    state_spike_arrays_all_trials,
    state_firing_rates_all_trials,
    window_length,
    step_size,
):
    appended_p_values = []
    for trial_idx in range(len(state_firing_rates_all_trials)):
        trial_spike_array = state_spike_arrays_all_trials[trial_idx]
        trial_firing_array = state_firing_rates_all_trials[trial_idx]
        p_values = []
        for changepoint_idx in range(len(trial_spike_array)):
            state_spike_array = trial_spike_array[changepoint_idx]
            state_firing_array = trial_firing_array[changepoint_idx]
            num_bins = (state_spike_array.shape[2] - window_length) // step_size + 1
            for neuron in range(nrn):
                neuron_p_values = []
                for time_bin in range(num_bins):
                    taste1 = state_firing_array[0, neuron, time_bin]
                    taste2 = state_firing_array[1, neuron, time_bin]
                    taste3 = state_firing_array[2, neuron, time_bin]
                    taste4 = state_firing_array[3, neuron, time_bin]
                    f_stat, p_value = f_oneway(taste1, taste2, taste3, taste4)
                    neuron_p_values.append((time_bin, p_value, f_stat))
                p_values.append(neuron_p_values)
            appended_p_values.append(p_values)
    return np.array(appended_p_values)


# note to Vincent: we can't do ANOVA the way we would have because we're now splitting up on an individual trial basis
# This is because of changepoint schenanigans. As such, we've got to figure out another way around this
# Things to address with the whole changepoint business: How to handle not doing individual trials, and how to handle
# the different tastes-- as these are both critical before we may proceed.
