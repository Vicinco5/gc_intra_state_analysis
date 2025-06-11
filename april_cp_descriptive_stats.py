#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:13:52 2025


# 4/15 scripts on analyzing the descriptive stats of the changepoints themselves 

@author: vincentcalia-bogan
"""

import os, os.path
import numpy as np

# 4/14 changepoint descrptive stats: 
save_path = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/CP_descriptive_stats/cp_stats.txt'
def analyze_changepoints_to_file(standardized_changepoints_dict, save_path, stim_time=2000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_cp = [[] for _ in range(4)]
    all_durations = [[] for _ in range(4)]

    with open(save_path, 'w') as f:
        for dataset_name, taste_array_list in standardized_changepoints_dict.items():
            f.write(f"\n=== Dataset: {dataset_name} ===\n")
            for taste_idx, trial_array in enumerate(taste_array_list):
                f.write(f"\n  Taste {taste_idx}:\n")
                cp_array = np.array(trial_array)  # Shape: (num_trials, 4)

                cp0, cp1, cp2, cp3 = cp_array.T
                cp_list = [cp0, cp1, cp2, cp3]

                for i, cp in enumerate(cp_list):
                    rel_cp = cp# - stim_time
                    all_cp[i].extend(cp)

                    mean_cp = np.mean(rel_cp)
                    std_cp = np.std(cp)
                    var_cp = np.var(cp)

                    deviations = rel_cp - mean_cp
                    max_idx = np.argmax(deviations)
                    min_idx = np.argmin(deviations)

                    max_val = cp[max_idx]
                    max_dev = deviations[max_idx]

                    min_val = cp[min_idx]
                    min_dev = deviations[min_idx]

                    f.write(f"    CP{i+1} stats (relative to stimulus at {stim_time} ms):\n")
                    f.write(f"      Mean:  {mean_cp:.2f} ms\n")
                    f.write(f"      Var:   {var_cp:.2f}\n")
                    f.write(f"      Std:   {std_cp:.2f}\n")
                    f.write(f"      Most extreme high: {max_val} ms (deviation: {max_dev:+.2f})\n")
                    f.write(f"      Most extreme low:  {min_val} ms (deviation: {min_dev:+.2f})\n")

                durations = [
                    cp0 - stim_time,     # Duration 0
                    cp1 - cp0,           # Duration 1
                    cp2 - cp1,           # Duration 2
                    cp3 - cp2            # Duration 3
                ]

                for i, dur in enumerate(durations):
                    all_durations[i].extend(dur)

                    mean_dur = np.mean(dur)
                    std_dur = np.std(dur)
                    min_dur = np.min(dur)
                    max_dur = np.max(dur)

                    f.write(f"    Duration {i} stats:\n")
                    f.write(f"      Mean:  {mean_dur:.2f} ms\n")
                    f.write(f"      Std:   {std_dur:.2f} ms\n")
                    f.write(f"      Min:   {min_dur:.2f} ms\n")
                    f.write(f"      Max:   {max_dur:.2f} ms\n")

        # Global stats
        f.write(f"\n=== Global Statistics Across All Datasets and Tastes ===\n")

        for i, cp in enumerate(all_cp):
            cp = np.array(cp)
            rel_cp = cp# - stim_time
            mean_cp = np.mean(rel_cp)
            std_cp = np.std(cp)
            var_cp = np.var(cp)

            deviations = rel_cp - mean_cp
            max_idx = np.argmax(deviations)
            min_idx = np.argmin(deviations)

            max_val = cp[max_idx]
            max_dev = deviations[max_idx]

            min_val = cp[min_idx]
            min_dev = deviations[min_idx]

            f.write(f"\n  Global CP{i+1} stats (relative to stimulus):\n")
            f.write(f"    Mean:  {mean_cp:.2f} ms\n")
            f.write(f"    Var:   {var_cp:.2f}\n")
            f.write(f"    Std:   {std_cp:.2f}\n")
            f.write(f"    Most extreme high: {max_val} ms (deviation: {max_dev:+.2f})\n")
            f.write(f"    Most extreme low:  {min_val} ms (deviation: {min_dev:+.2f})\n")

        for i, dur in enumerate(all_durations):
            dur = np.array(dur)
            mean_dur = np.mean(dur)
            std_dur = np.std(dur)
            min_dur = np.min(dur)
            max_dur = np.max(dur)

            f.write(f"\n  Global Duration {i} stats:\n")
            f.write(f"    Mean:  {mean_dur:.2f} ms\n")
            f.write(f"    Std:   {std_dur:.2f} ms\n")
            f.write(f"    Min:   {min_dur:.2f} ms\n")
            f.write(f"    Max:   {max_dur:.2f} ms\n")

    print(f"Stats written to {save_path}")

