#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:03:41 2024

@author: vincentcalia-bogan
"""

## NON-VINCENT MODULE IMPORTS ## 
# 17 datasets 365 neurons total 
import os, os.path
import numpy as np
import xarray as xr
from scipy.stats import ttest_rel

# single trial thresholding-- likely less useful 
 
# t-test to record signifigant neurons on a trial-by-trial basis

# also a working test # 
# sorting according to minimum firing rate-- at least 1 hz over the entire trial? Hmm # 
def thresh_min_fr_unwarped_hz(state_firing_rates_all_trials, threshold_hz):
    num_tastes = len(state_firing_rates_all_trials)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    num_neurons = state_firing_rates_all_trials[0][0].shape[1]
    num_trials = state_firing_rates_all_trials[0].shape[0]    
    # create identical xarray object for storing firing rate data 
    thresh_hz_state_firing_rates = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time_bin"],
        coords={
            "taste": range(num_tastes),
            "trial": range(num_trials),
            "segment": range(num_changepoints),
            "neuron": range(num_neurons),
            "time_bin": range(state_firing_rates_all_trials[0][0].shape[-1])  # maximum possible bins
        }
    )
    # Loop through each neuron-- thresholding for neurons that are exhibitng minimum firing activity
    for neuron_idx in range(num_neurons):
        for taste_idx in range(num_tastes):
            for cp_idx in range(num_changepoints):
                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = state_firing_rates_all_trials[taste_idx][trial_idx][cp_idx][neuron_idx].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if valid_data.size > 0:  # Only process if there are valid (non-NaN) values
                        avg_fr = np.nanmean(valid_data)
                        if avg_fr >= threshold_hz:  # threshold firing rate of 1 hz over the entire recording 
                            thresh_hz_state_firing_rates[taste_idx, trial_idx, cp_idx, neuron_idx, :] = state_firing_rates_all_trials[taste_idx][trial_idx][cp_idx][neuron_idx]
   
    return thresh_hz_state_firing_rates

# same thing but now thresholding via t-test-- see if this works at all 
def thresh_ttest_unwarped_hz(state_firing_rates_all_trials, alpha):
    num_tastes = len(state_firing_rates_all_trials)
    num_changepoints = 3  # Number of changepoints; hardcoded but can be made flexible
    num_neurons = state_firing_rates_all_trials[0][0].shape[1]
    num_trials = state_firing_rates_all_trials[0].shape[0]    
    # create identical xarray object for storing firing rate data 
    thresh_ttest_state_firing_rates = xr.DataArray(
        data=np.nan,
        dims=["taste", "trial", "segment", "neuron", "time_bin"],
        coords={
            "taste": range(num_tastes),
            "trial": range(num_trials),
            "segment": range(num_changepoints),
            "neuron": range(num_neurons),
            "time_bin": range(state_firing_rates_all_trials[0][0].shape[-1])  # maximum possible bins
        }
    )
    # Loop through each neuron-- thresholding for neurons that are exhibitng minimum firing activity
    for neuron_idx in range(num_neurons):
        for taste_idx in range(num_tastes):
            for cp_idx in range(num_changepoints):
                for trial_idx in range(num_trials):  # Loop through trials
                    trial_array = state_firing_rates_all_trials[taste_idx][trial_idx][cp_idx][neuron_idx].values
                    valid_data = trial_array[~np.isnan(trial_array)]
                    if valid_data.size > 0:  # Only process if there are valid (non-NaN) values
                        num_tbin_ttest = round((valid_data.shape[0]/2))
                        ttest_a = valid_data[:num_tbin_ttest]
                        ttest_b = valid_data[num_tbin_ttest:num_tbin_ttest + len(ttest_a)]
                        if len(ttest_a) == len(ttest_b):
                            tstat, pval = ttest_rel(ttest_a, ttest_b)
                            if pval < alpha and not np.isnan(pval): # thresholding for significance of firing via t-test
                                thresh_ttest_state_firing_rates[taste_idx, trial_idx, cp_idx, neuron_idx, :] = state_firing_rates_all_trials[taste_idx][trial_idx][cp_idx][neuron_idx]
   
    return thresh_ttest_state_firing_rates