#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:37:47 2025

@author: vincentcalia-bogan

firing_rate_plotter
"""


# new one: 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import matplotlib.lines as mlines

class FiringRatePlotter:
    """
Plot firing-rate heatmaps and line plots from an xarray DataArray of precomputed firing rates.

Parameters
----------
fr_data_array : xr.DataArray
    A 4D DataArray indexed by
    (‘taste’, ‘trial’, ‘segment’, ‘neuron’) containing firing-rate vectors (in Hz).
dataset_dir : str
    Path to the directory where all figures will be saved. Subfolders will be created
    automatically per taste (e.g. “taste_0”, “taste_1”, …).
dataset_name : str
    A short identifier for this dataset (used in figure titles and filenames).
base_name : str
    A base filename prefix (unused internally in the current implementation,
    but reserved for future extensions).
is_warped : Union[bool, str]
    – True      : firing-rate vectors are already time-warped/padded to equal length  
    – False     : unwarped vectors of variable length (can be aligned/padded)  
    – 'whole-data' : concatenate all segments end-to-end to plot the full trial,
                     ignoring segmentation
mode : str
    – 'cell-wise'  : one subplot per neuron (rows of heatmap = trials)  
    – 'trial-wise' : one subplot per trial (rows of heatmap = neurons)  
    – 'debug'      : reserved (currently no output)
alignment : Optional[str], default=None
    When `is_warped is False`, how to pad unwarped vectors to equal length:
    – 'start' : pad NaNs at end so that all vectors begin at t=0  
    – 'end'   : pad NaNs at beginning so that all vectors end at the same time  
    – None    : no padding
sort_by_length : bool, default=False
    When `is_warped is False`, whether to sort heatmap rows
    by descending non-NaN length (longest first).
changepoints_dict : dict, default=None
    Nested dict mapping
    `{ dataset_name → { taste_idx → { trial_idx → [cp1, cp2, …] } } }`.
    If provided, vertical lines at each changepoint will be overlaid.

Plot types generated
--------------------
1. Segmented Cell-Wise Heatmaps (`_plot_segmented_cellwise`):
   • One figure per (taste, segment)  
   • Subplots arranged in a grid of up to 5 columns, each subplot = one neuron  
   • Rows = trials, heatmap of firing-rate vs. time  
   • Optional alignment, sorting, stim-line (dotted), changepoints (dashed)  

2. Segmented Trial-Wise Heatmaps (`_plot_segmented_trialwise`):
   • One figure per (taste, segment)  
   • Subplots arranged in a grid of up to 5 columns, each subplot = one trial  
   • Rows = neurons, heatmap of firing-rate vs. time  
   • Optional alignment, sorting, stim-line, changepoints  

3. Whole-Data Cell-Wise Heatmaps (`_plot_wholedata_cellwise`):
   • One figure per taste  
   • 4 subplots (neurons 0–3) stacked vertically  
   • Heatmap of full-trial firing rates (concatenated segments)  
   • Stimulus and changepoint lines  

4. Whole-Data Trial-Wise Heatmaps (`_plot_wholedata_trialwise`):
   • One figure per taste, in chunks of 4 trials  
   • 4 subplots stacked vertically (each = one trial)  
   • Heatmap of full-trial firing rates (concatenated segments)  
   • x-axis in ms (with time step and start offset)  
   • Stimulus and changepoint lines  

All figures are saved as both PNG and SVG with filenames:
    `<dataset_dir>/taste_<taste_idx>/…_<dataset_name>.(png|svg)`
"""
    def __init__(
        self,
        fr_data_array: xr.DataArray,
        dataset_dir: str,
        dataset_name: str,
        base_name: str,
        is_warped: str,  # can be True, False, or 'whole-data'
        mode: str,       # 'cell-wise', 'trial-wise', or 'debug'
        alignment: str = None,
        sort_by_length: bool = False,
        changepoints_dict: dict = None
    ):
        self.data = fr_data_array
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.base_name = base_name
        self.is_warped = is_warped
        self.mode = mode
        self.alignment = alignment
        self.sort_by_length = sort_by_length
        self.changepoints_dict = changepoints_dict

        os.makedirs(self.dataset_dir, exist_ok=True)

    @staticmethod
    def cli_options():
        print("\n--- Firing Rate Plotting Configuration ---")
        mode = input("Select plotting mode ('cell-wise' or 'trial-wise'): ").strip().lower()
        while mode not in ['cell-wise', 'trial-wise']:
            mode = input("Invalid input. Choose 'cell-wise' or 'trial-wise': ").strip().lower()

        is_whole_data = input("Use whole-data mode (no padding, full time)? (y/n): ").strip().lower() == 'y'
        is_warped = 'whole-data' if is_whole_data else input("Plot warped data? (y/n): ").strip().lower() == 'y'

        alignment = None
        sort_by_length = False

        if not is_whole_data and not is_warped:
            alignment = input("Select alignment for unwarped data ('start', 'end', or 'none'): ").strip().lower()
            while alignment not in ['start', 'end', 'none']:
                alignment = input("Invalid input. Choose 'start', 'end', or 'none': ").strip().lower()
            if alignment == 'none':
                alignment = None

            sort_by_length = input("Sort vectors by length (only for unwarped)? (y/n): ").strip().lower() == 'y'

        print("--- Configuration complete. ---\n")
        return mode, is_warped, alignment, sort_by_length

    def plot_all(self):
        if self.mode == "debug":
            print("Debug mode not implemented.")
            return

        if self.is_warped == "whole-data":
            if self.mode == "cell-wise":
                self._plot_wholedata_cellwise()
            elif self.mode == "trial-wise":
                self._plot_wholedata_trialwise()
        elif self.mode == "cell-wise":
            self._plot_segmented_cellwise()
        elif self.mode == "trial-wise":
            self._plot_segmented_trialwise()

    # Utilities ------------------------------------------------------------
    def _pad_align(self, vectors, align_to):
        max_len = max(len(v) for v in vectors)
        aligned = []
        for v in vectors:
            pad_size = max_len - len(v)
            if align_to == 'start':
                padded = np.concatenate([v, np.full(pad_size, np.nan)])
            elif align_to == 'end':
                padded = np.concatenate([np.full(pad_size, np.nan), v])
            else:
                padded = v
            aligned.append(padded)
        return np.array(aligned)

    def _sort_by_length(self, vectors):
        lengths = [np.sum(~np.isnan(v)) for v in vectors]
        return np.argsort(lengths)[::-1]

    def _draw_changepoints(self, ax, changepoints, color='black', style='--'):
        for cp in changepoints:
            ax.axvline(x=cp, color=color, linestyle=style, linewidth=1)

    def _draw_stimulus(self, ax, stim_time=2000):
        ax.axvline(x=stim_time, color='black', linestyle=':', linewidth=1)

    def _save_figure(self, fig, path_base):
        fig.savefig(f"{path_base}.png", dpi=300)
        fig.savefig(f"{path_base}.svg", format='svg')
        plt.close(fig)

    # Plotting: Segmented Cell-Wise --------------------------------------------------
    def _plot_segmented_cellwise(self):
        for taste_idx in self.data.coords["taste"].values:
            for seg_idx in self.data.coords["segment"].values:
                neurons = self.data.coords["neuron"].values
                trials = self.data.coords["trial"].values

                nrows = math.ceil(len(neurons) / 5)
                fig, axes = plt.subplots(nrows, 5, figsize=(15, 3 * nrows), sharex=True, sharey=True)
                axes = axes.flatten()

                for i, neuron_idx in enumerate(neurons):
                    ax = axes[i]
                    rows = []
                    for trial_idx in trials:
                        vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx, neuron=neuron_idx).values
                        if vec is not None and np.any(~np.isnan(vec)):
                            rows.append(vec)
                        else:
                            rows.append(np.array([]))

                    if not self.is_warped and self.alignment:
                        rows = self._pad_align(rows, self.alignment)

                    if not self.is_warped and self.sort_by_length:
                        sorted_idx = self._sort_by_length(rows)
                        rows = [rows[j] for j in sorted_idx]

                    heat_data = np.full((len(rows), max(len(r) for r in rows)), np.nan)
                    for j, row in enumerate(rows):
                        heat_data[j, :len(row)] = row

                    sns.heatmap(heat_data, ax=ax, cmap="viridis", cbar=True, cbar_kws={'label': 'Firing Rate (Hz)'})
                    ax.set_title(f"Neuron {neuron_idx}")

                    if self.changepoints_dict and isinstance(self.changepoints_dict, dict):
                        cps_dict = self.changepoints_dict.get(self.dataset_name, {}).get(taste_idx, {})
                        if isinstance(cps_dict, dict) and trials[0] in cps_dict:
                            cps = cps_dict.get(int(trials[0]), [])
                            self._draw_changepoints(ax, cps, color='black', style='--')
                        self._draw_stimulus(ax)

                fig.suptitle(f"Cell-wise Heatmap\nDataset: {self.dataset_name}, Taste {taste_idx}, Epoch {seg_idx}", fontsize=14)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                red_line = mlines.Line2D([], [], color='black', linestyle=':', label='Stimulus')
                blue_line = mlines.Line2D([], [], color='black', linestyle='--', label='Changepoint')
                fig.legend(handles=[red_line, blue_line], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)

                taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
                os.makedirs(taste_dir, exist_ok=True)
                path_base = os.path.join(taste_dir, f"FR_Cellwise_Taste{taste_idx}_Epoch{seg_idx}_{self.dataset_name}")
                self._save_figure(fig, path_base)


    # Plotting: Segmented Trial-Wise --------------------------------------------------
    def _plot_segmented_cellwise(self):
        for taste_idx in self.data.coords["taste"].values:
            for seg_idx in self.data.coords["segment"].values:
                neurons = self.data.coords["neuron"].values
                trials = self.data.coords["trial"].values

                nrows = math.ceil(len(neurons) / 5)
                fig, axes = plt.subplots(nrows, 5, figsize=(15, 3 * nrows), sharex=True, sharey=True)
                axes = axes.flatten()

                for i, neuron_idx in enumerate(neurons):
                    ax = axes[i]
                    rows = []
                    for trial_idx in trials:
                        vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx, neuron=neuron_idx).values
                        if vec is not None and np.any(~np.isnan(vec)):
                            rows.append(vec)
                        else:
                            rows.append(np.array([]))

                    if not self.is_warped and self.alignment:
                        rows = self._pad_align(rows, self.alignment)

                    if not self.is_warped and self.sort_by_length:
                        sorted_idx = self._sort_by_length(rows)
                        rows = [rows[j] for j in sorted_idx]

                    heat_data = np.full((len(rows), max(len(r) for r in rows)), np.nan)
                    for j, row in enumerate(rows):
                        heat_data[j, :len(row)] = row

                    sns.heatmap(heat_data, ax=ax, cmap="viridis", cbar=True, cbar_kws={'label': 'Firing Rate (Hz)'})
                    ax.set_title(f"Neuron {neuron_idx}")

                    if self.changepoints_dict and isinstance(self.changepoints_dict, dict):
                        cps_dict = self.changepoints_dict.get(self.dataset_name, {}).get(taste_idx, {})
                        if isinstance(cps_dict, dict) and trials[0] in cps_dict:
                            cps = cps_dict.get(int(trials[0]), [])
                            self._draw_changepoints(ax, cps, color='black', style='--')
                        self._draw_stimulus(ax)

                fig.suptitle(f"Cell-wise Heatmap\nDataset: {self.dataset_name}, Taste {taste_idx}, Epoch {seg_idx}", fontsize=14)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                red_line = mlines.Line2D([], [], color='black', linestyle=':', label='Stimulus')
                blue_line = mlines.Line2D([], [], color='black', linestyle='--', label='Changepoint')
                fig.legend(handles=[red_line, blue_line], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)

                taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
                os.makedirs(taste_dir, exist_ok=True)
                path_base = os.path.join(taste_dir, f"FR_Cellwise_Taste{taste_idx}_Epoch{seg_idx}_{self.dataset_name}")
                self._save_figure(fig, path_base)

    # Plotting: Whole-Data Cell-Wise --------------------------------------------------
    def _plot_wholedata_cellwise(self):
        for taste_idx in self.data.coords["taste"].values:
            neurons = self.data.coords["neuron"].values
            trials = self.data.coords["trial"].values

            fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)

            for ax_idx, neuron_idx in enumerate(neurons[:4]):
                ax = axes[ax_idx]
                full_trials = []

                for trial_idx in trials:
                    segments = []
                    for seg_idx in self.data.coords["segment"].values:
                        vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx, neuron=neuron_idx).values
                        if vec is not None and np.any(~np.isnan(vec)):
                            segments.append(vec)
                    if segments:
                        full_vec = np.concatenate(segments)
                        full_trials.append(full_vec)

                heat_data = np.full((len(full_trials), max(len(t) for t in full_trials)), np.nan)
                for j, trial in enumerate(full_trials):
                    heat_data[j, :len(trial)] = trial

                sns.heatmap(heat_data, ax=ax, cmap="viridis", cbar=True, cbar_kws={'label': 'Firing Rate (Hz)'})
                ax.set_title(f"Neuron {neuron_idx}")
                self._draw_stimulus(ax)

                if self.changepoints_dict and isinstance(self.changepoints_dict, dict):
                    cps_dict = self.changepoints_dict.get(self.dataset_name, {}).get(taste_idx, {})
                    if isinstance(cps_dict, dict) and len(cps_dict) > 0:
                        sample_trial = next(iter(cps_dict))
                        sample_cps = cps_dict.get(sample_trial, [])
                        self._draw_changepoints(ax, sample_cps)

            fig.suptitle(f"Whole-Data Cell-Wise Firing Heatmap\nTaste {taste_idx}, Dataset {self.dataset_name}", fontsize=14)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            red_line = mlines.Line2D([], [], color='black', linestyle=':', label='Stimulus')
            blue_line = mlines.Line2D([], [], color='black', linestyle='--', label='Changepoint')
            fig.legend(handles=[red_line, blue_line], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)

            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)
            fname = f"wholedata_cellwise_taste_{taste_idx}_{self.dataset_name}"
            self._save_figure(fig, os.path.join(taste_dir, fname))

    # Plotting: Whole-Data Trial-Wise --------------------------------------------------

    def _plot_wholedata_trialwise(self):
        for taste_idx in self.data.coords["taste"].values:
            trials = self.data.coords["trial"].values
            neurons = self.data.coords["neuron"].values
            segs = self.data.coords["segment"].values
    
            for chunk_start in range(0, len(trials), 4):
                fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)
                fig_axes = axes if isinstance(axes, np.ndarray) else [axes]
    
                trial_subset = trials[chunk_start:chunk_start + 4]
    
                for ax_idx, trial_idx in enumerate(trial_subset):
                    ax = fig_axes[ax_idx]
                    all_vecs = []
    
                    for neuron_idx in neurons:
                        segments = []
                        for seg_idx in segs:
                            vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=seg_idx, neuron=neuron_idx).values
                            if vec is not None and np.any(~np.isnan(vec)):
                                segments.append(vec)
                        if segments:
                            full_vec = np.concatenate(segments)
                            all_vecs.append(full_vec)
    
                    if not all_vecs:
                        continue
    
                    max_len = max(len(t) for t in all_vecs)
                    time_step = 25
                    time_start = 1500
                    time_vector = np.arange(max_len) * time_step + time_start
    
                    heat_data = np.full((len(all_vecs), max_len), np.nan)
                    for j, row in enumerate(all_vecs):
                        heat_data[j, :len(row)] = row
    
                    extent = [time_vector[0], time_vector[-1], 0, heat_data.shape[0]]
                    im = ax.imshow(heat_data, aspect='auto', interpolation='nearest', cmap='plasma',
                                   extent=extent, origin='upper')
    
                    # Individual colorbar per subplot
                    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
                    cbar.set_label('Firing Rate (Hz)')
    
                    xticks = np.arange(time_start, time_vector[-1] + 1, 500)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(t) for t in xticks])
                    ax.tick_params(axis='x', labelbottom=True)  # Force labels on all subplots
                        
                    yticks = np.arange(0, heat_data.shape[0], 5)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([str(y) for y in yticks])
    
                    ax.set_title(f"Trial {trial_idx+1}")
                    ax.set_ylabel("Neuron")
                    ax.set_xlabel("Time (ms)")
    
                # Draw changepoints on all axes
                for ax_idx, trial_idx in enumerate(trial_subset):
                    ax = fig_axes[ax_idx]
                    ax.axvline(x=2000, color='black', linestyle=':', linewidth=3)  # stimulus
    
                    try:
                        cps = self.changepoints_dict[self.dataset_name][taste_idx][trial_idx]
                        for cp in cps:
                            if time_start <= cp <= time_vector[-1]:
                                ax.axvline(x=cp, color='black', linestyle='--', linewidth=3)
                    except (KeyError, IndexError, TypeError):
                        pass
    
                fig.suptitle(f"Whole-Data Trial-Wise Firing Heatmap\nTaste {taste_idx}, Dataset {self.dataset_name}", fontsize=14)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
                stim_line = mlines.Line2D([], [], color='black', linestyle=':', label='Stimulus')
                cp_line = mlines.Line2D([], [], color='black', linestyle='--', label='Changepoint')
                fig.legend(handles=[stim_line, cp_line], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
                taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
                os.makedirs(taste_dir, exist_ok=True)
                chunk_label = f"chunk_{chunk_start // 4 + 1}"
                fname_base = f"wholedata_trialwise_taste_{taste_idx}_{self.dataset_name}_{chunk_label}"
                self._save_figure(fig, os.path.join(taste_dir, fname_base))




# import os
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xarray as xr


# class FiringRatePlotter:
#     def __init__(
#         self,
#         fr_data_array: xr.DataArray,
#         dataset_dir: str,
#         dataset_name: str,
#         base_name: str,
#         is_warped: str,
#         mode: str,
#         alignment: str = None,
#         sort_by_length: bool = False
#     ):
#         self.data = fr_data_array
#         self.dataset_dir = dataset_dir
#         self.dataset_name = dataset_name
#         self.base_name = base_name
#         self.is_warped = is_warped
#         self.mode = mode
#         self.alignment = alignment
#         self.sort_by_length = sort_by_length

#         os.makedirs(self.dataset_dir, exist_ok=True)

#     @staticmethod
#     def cli_options():
#         print("\n--- Firing Rate Plotting Configuration ---")
#         mode = input("Select plotting mode ('cell-wise' or 'trial-wise'): ").strip().lower()
#         while mode not in ['cell-wise', 'trial-wise']:
#             mode = input("Invalid input. Choose 'cell-wise' or 'trial-wise': ").strip().lower()

#         is_whole_data = input("Use whole-data mode (no padding, full time)? (y/n): ").strip().lower() == 'y'
#         is_warped = 'whole-data' if is_whole_data else input("Plot warped data? (y/n): ").strip().lower() == 'y'

#         alignment = None
#         sort_by_length = False

#         if not is_whole_data and not is_warped:
#             alignment = input("Select alignment for unwarped data ('start', 'end', or 'none'): ").strip().lower()
#             while alignment not in ['start', 'end', 'none']:
#                 alignment = input("Invalid input. Choose 'start', 'end', or 'none': ").strip().lower()
#             if alignment == 'none':
#                 alignment = None

#             sort_by_length = input("Sort trials (heatmap only) by length? (y/n): ").strip().lower() == 'y'

#         print("--- Configuration complete. ---\n")
#         return mode, is_warped, alignment, sort_by_length

#     def plot_all(self):
#         if self.mode == "cell-wise":
#             self._plot_cellwise_heatmap()
#             self._plot_cellwise_lineplot()
#         elif self.mode == "trial-wise":
#             self._plot_trialwise_heatmap()
#             self._plot_trialwise_lineplot()

#     def _pad_align(self, arr_list, align_to):
#         max_len = max(len(row) for row in arr_list)
#         aligned = []
#         for row in arr_list:
#             row_len = len(row)
#             pad_size = max_len - row_len
#             if align_to == 'start':
#                 padded = np.concatenate([row, np.full(pad_size, np.nan)])
#             elif align_to == 'end':
#                 padded = np.concatenate([np.full(pad_size, np.nan), row])
#             else:
#                 padded = row
#             aligned.append(padded)
#         return np.array(aligned)

#     def _sort_by_non_nan_length(self, arr_list):
#         valid_counts = [np.sum(~np.isnan(arr)) for arr in arr_list]
#         return np.argsort(valid_counts)[::-1]

#     def _plot_cellwise_heatmap(self):
#         for taste_idx in self.data.coords["taste"].values:
#             for epoch_idx in self.data.coords["segment"].values:
#                 taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
#                 os.makedirs(taste_dir, exist_ok=True)

#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 neurons = self.data.coords["neuron"].values
#                 trials = self.data.coords["trial"].values

#                 heat_data = []
#                 for neuron_idx in neurons:
#                     rows = []
#                     for trial_idx in trials:
#                         fr_vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=epoch_idx, neuron=neuron_idx).values
#                         if fr_vec is not None and len(fr_vec) > 0 and not np.all(np.isnan(fr_vec)):
#                             rows.append(fr_vec)
#                     if not rows:
#                         continue
#                     if not self.is_warped and self.alignment:
#                         rows = self._pad_align(rows, self.alignment)
#                     mean = np.nanmean(rows, axis=0)
#                     heat_data.append(mean)

#                 if not heat_data:
#                     continue

#                 sns.heatmap(np.array(heat_data), ax=ax, cmap="viridis", cbar=True)
#                 ax.set_title(f"Firing Rate Heatmap\nTaste {taste_idx}, Epoch {epoch_idx}")
#                 ax.set_ylabel("Neuron")
#                 ax.set_xlabel("Time")

#                 fname_base = f"FR_Heatmap_Taste{taste_idx}_Epoch{epoch_idx}_{self.dataset_name}"
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.png"), dpi=300)
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.svg"), format='svg')
#                 plt.close(fig)

#     def _plot_cellwise_lineplot(self):
#         for taste_idx in self.data.coords["taste"].values:
#             for epoch_idx in self.data.coords["segment"].values:
#                 taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
#                 os.makedirs(taste_dir, exist_ok=True)

#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 neurons = self.data.coords["neuron"].values
#                 trials = self.data.coords["trial"].values

#                 for neuron_idx in neurons:
#                     rows = []
#                     for trial_idx in trials:
#                         fr_vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=epoch_idx, neuron=neuron_idx).values
#                         if fr_vec is not None and len(fr_vec) > 0 and not np.all(np.isnan(fr_vec)):
#                             rows.append(fr_vec)

#                     if not rows:
#                         continue
#                     if not self.is_warped and self.alignment:
#                         rows = self._pad_align(rows, self.alignment)

#                     mean = np.nanmean(rows, axis=0)
#                     sem = np.nanstd(rows, axis=0) / np.sqrt(len(rows))
#                     x = np.arange(len(mean))
#                     if np.any(np.isnan(mean)) or np.any(np.isnan(sem)):
#                         continue
#                     ax.plot(x, mean)
#                     ax.fill_between(x, mean - sem, mean + sem, alpha=0.3)

#                 ax.set_title(f"Firing Rate Line Plot\nTaste {taste_idx}, Epoch {epoch_idx}")
#                 ax.set_ylabel("Firing Rate (Hz)")
#                 ax.set_xlabel("Time")

#                 fname_base = f"FR_LinePlot_Taste{taste_idx}_Epoch{epoch_idx}_{self.dataset_name}"
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.png"), dpi=300)
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.svg"), format='svg')
#                 plt.close(fig)

#     def _plot_trialwise_heatmap(self):
#         for taste_idx in self.data.coords["taste"].values:
#             for epoch_idx in self.data.coords["segment"].values:
#                 taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
#                 os.makedirs(taste_dir, exist_ok=True)

#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 trials = self.data.coords["trial"].values
#                 neurons = self.data.coords["neuron"].values

#                 heat_data = []
#                 for trial_idx in trials:
#                     rows = []
#                     for neuron_idx in neurons:
#                         fr_vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=epoch_idx, neuron=neuron_idx).values
#                         if fr_vec is not None and len(fr_vec) > 0 and not np.all(np.isnan(fr_vec)):
#                             rows.append(fr_vec)

#                     if not rows:
#                         continue
#                     if not self.is_warped and self.alignment:
#                         rows = self._pad_align(rows, self.alignment)

#                     mean = np.nanmean(rows, axis=0)
#                     heat_data.append(mean)

#                 if not heat_data:
#                     continue

#                 if self.sort_by_length:
#                     sorted_idx = self._sort_by_non_nan_length(heat_data)
#                     heat_data = [heat_data[i] for i in sorted_idx]

#                 sns.heatmap(np.array(heat_data), ax=ax, cmap="viridis", cbar=True)
#                 ax.set_title(f"Trialwise Firing Rate Heatmap\nTaste {taste_idx}, Epoch {epoch_idx}")
#                 ax.set_ylabel("Trial")
#                 ax.set_xlabel("Time")

#                 fname_base = f"FR_TrialHeatmap_Taste{taste_idx}_Epoch{epoch_idx}_{self.dataset_name}"
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.png"), dpi=300)
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.svg"), format='svg')
#                 plt.close(fig)

#     def _plot_trialwise_lineplot(self):
#         for taste_idx in self.data.coords["taste"].values:
#             for epoch_idx in self.data.coords["segment"].values:
#                 taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
#                 os.makedirs(taste_dir, exist_ok=True)

#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 trials = self.data.coords["trial"].values
#                 neurons = self.data.coords["neuron"].values

#                 for trial_idx in trials:
#                     rows = []
#                     for neuron_idx in neurons:
#                         fr_vec = self.data.sel(taste=taste_idx, trial=trial_idx, segment=epoch_idx, neuron=neuron_idx).values
#                         if fr_vec is not None and len(fr_vec) > 0 and not np.all(np.isnan(fr_vec)):
#                             rows.append(fr_vec)

#                     if not rows:
#                         continue
#                     if not self.is_warped and self.alignment:
#                         rows = self._pad_align(rows, self.alignment)

#                     mean = np.nanmean(rows, axis=0)
#                     sem = np.nanstd(rows, axis=0) / np.sqrt(len(rows))
#                     x = np.arange(len(mean))
#                     if np.any(np.isnan(mean)) or np.any(np.isnan(sem)):
#                         continue
#                     ax.plot(x, mean)
#                     ax.fill_between(x, mean - sem, mean + sem, alpha=0.3)

#                 ax.set_title(f"Trialwise Firing Rate Line Plot\nTaste {taste_idx}, Epoch {epoch_idx}")
#                 ax.set_ylabel("Firing Rate (Hz)")
#                 ax.set_xlabel("Time")

#                 fname_base = f"FR_TrialLinePlot_Taste{taste_idx}_Epoch{epoch_idx}_{self.dataset_name}"
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.png"), dpi=300)
#                 fig.savefig(os.path.join(taste_dir, f"{fname_base}.svg"), format='svg')
#                 plt.close(fig)


