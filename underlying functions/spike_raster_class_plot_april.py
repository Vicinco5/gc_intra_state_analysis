#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:26:14 2025

April-- spike raster plotting class. Far more organized methods and way of doing things. Still some works in progress with this one.


@author: vincentcalia-bogan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import xarray as xr
import matplotlib.lines as mlines

# a fix for alignment? not quite -- alignment funcs need some work for sure


class SpikeRasterPlotter:
    """
    Plot spike-raster heatmaps and scatter rasters from an xarray DataArray of binary spike trains.

    Parameters
    ----------
    spike_data_array : xr.DataArray
        4-D DataArray indexed by (‘taste’, ‘trial’, ‘segment’, ‘neuron’) containing
        binary spike trains (1 = spike, 0 or NaN = no spike) per time bin.
    dataset_dir : str
        Directory where all figures will be saved. Subdirectories “taste_<idx>” will be created.
    dataset_name : str
        Short identifier for this dataset (used in figure titles and filenames).
    base_name : str
        Base filename prefix (included in titles; reserved for future use).
    is_warped : Union[bool, str]
        – True      : data are pre-warped/padded to equal length
        – False     : variable-length data (can be aligned/padded)
        – 'whole-data' : ignore segmentation and concatenate all segments into a unified time axis
    mode : str
        – 'cell-wise'  : one subplot per neuron (rows = trials)
        – 'trial-wise' : one subplot per trial (rows = neurons)
        – 'debug'      : interactive data‐inspection mode (no plots)
    alignment : Optional[str], default=None
        When `is_warped is False`, how to pad unwarped spike arrays:
        – 'start' : pad at end so all rasters start at t=0
        – 'end'   : pad at beginning so all rasters end together
        – None    : leave unpadded
    show_markers : bool, default=False
        For segmented plots, overlay epoch start/end markers from `changepoints_dict`.
    sort_by_length : bool, default=False
        When `is_warped is False`, sort raster rows by descending non-NaN duration.
    changepoints_dict : dict, default=None
        Nested mapping `{ dataset_name → { taste_idx → { trial_idx → [cp0, cp1, …] } } }`.
        Used to draw vertical lines or markers at each changepoint.

    Plot types generated
    --------------------
    1. Whole-Data Cell-Wise Raster (`_plot_wholedata_cellwise`):
       • One figure per taste; up to 4 neurons plotted vertically.
       • Spike times as tick marks (|) at their time indices per trial.
       • Red dashed line at stimulus time (2000 ms), blue dashed at changepoints.

    2. Whole-Data Trial-Wise Raster (`_plot_wholedata_trialwise`):
       • One figure per taste, in chunks of 4 trials; stacked plots.
       • Spike times as ticks per neuron index; x-axis in ms (starting at default t=1500 ms).
       • Stimulus and changepoint lines overlaid.

    3. Segmented Cell-Wise Raster (`_plot_cellwise`):
       • One figure per (taste, segment); grid of subplots (neurons across columns).
       • Each subplot: raster of spikes (time bin vs. trial index).
       • Optional alignment, row sorting, stimulus line, changepoint/start-end markers.

    4. Segmented Trial-Wise Raster (`_plot_trialwise`):
       • One figure per (taste, segment); grid of subplots (trials across columns).
       • Each subplot: raster of spikes (time bin vs. neuron index).
       • Optional alignment, column sorting, stimulus line, changepoint markers.

    Debug mode
    ----------
    – 'debug': launches an interactive prompt (`debug_extract_from_calc`) for exploring
      underlying spike arrays without generating plots.
    """

    def __init__(
        self,
        spike_data_array: xr.DataArray,
        dataset_dir: str,
        dataset_name: str,
        base_name: str,
        is_warped: str,  # can be True, False, or 'whole-data'
        mode: str,  # 'cell-wise', 'trial-wise', or 'debug'
        alignment: str = None,
        show_markers: bool = False,
        sort_by_length: bool = False,
        changepoints_dict: dict = None,
    ):
        self.data = spike_data_array
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.base_name = base_name
        self.is_warped = is_warped  # 'whole-data' is a valid value here
        self.mode = mode
        self.alignment = alignment
        self.show_markers = show_markers
        self.sort_by_length = sort_by_length
        self.changepoints_dict = changepoints_dict

        os.makedirs(self.dataset_dir, exist_ok=True)

    @staticmethod
    def cli_options():
        print("\n--- Spike Raster Plotting Configuration ---")
        mode = (
            input("Select plotting mode ('cell-wise', 'trial-wise', or 'debug'): ")
            .strip()
            .lower()
        )
        while mode not in ["cell-wise", "trial-wise", "debug"]:
            mode = (
                input("Invalid input. Choose 'cell-wise', 'trial-wise', or 'debug': ")
                .strip()
                .lower()
            )

        if mode == "debug":
            print("Debug mode selected. Skipping plotting config...")
            return mode, None, None, False, False

        is_whole_data = (
            input("Use whole-data mode (no padding, full time)? (y/n): ")
            .strip()
            .lower()
            == "y"
        )

        if is_whole_data:
            print("--- Configuration complete. ---\n")
            return mode, "whole-data", None, False, False

        is_warped_input = input("Plot warped data? (y/n): ").strip().lower()
        is_warped = is_warped_input == "y"

        alignment = None
        show_markers = False
        sort_by_length = False

        if not is_warped:
            alignment = (
                input(
                    "Select alignment for unwarped data ('start', 'end', or 'none'): "
                )
                .strip()
                .lower()
            )
            while alignment not in ["start", "end", "none"]:
                alignment = (
                    input("Invalid input. Choose 'start', 'end', or 'none': ")
                    .strip()
                    .lower()
                )
            if alignment == "none":
                alignment = None

            show_markers_input = (
                input("Show start/end markers? (y/n): ").strip().lower()
            )
            show_markers = show_markers_input == "y"

            sort_input = (
                input("Sort unwarped data by duration (longest first)? (y/n): ")
                .strip()
                .lower()
            )
            sort_by_length = sort_input == "y"

        print("--- Configuration complete. ---\n")
        return mode, is_warped, alignment, show_markers, sort_by_length

    def plot_all(self):
        if self.mode == "debug":
            self.debug_extract_from_calc()
            return

        if self.is_warped == "whole-data":
            if self.mode == "cell-wise":
                self._plot_wholedata_cellwise()
            elif self.mode == "trial-wise":
                self._plot_wholedata_trialwise()
        elif self.mode == "cell-wise":
            self._plot_cellwise()
        elif self.mode == "trial-wise":
            self._plot_trialwise()

    def _plot_wholedata(self):
        if self.data is None:
            print("No data to plot.")
            return
        if "segment" in self.data.dims:
            if "unified_time" not in self.data.dims:
                try:
                    self.data = self.data.stack(unified_time=("segment", "time"))
                except ValueError:
                    self.data = self.data.rename_dims({"time": "original_time"})
                    self.data = self.data.stack(
                        unified_time=("segment", "original_time")
                    )

    def _pad_align(self, arr_list, align_to):
        max_len = max(len(row) for row in arr_list)
        aligned = []

        for row in arr_list:
            row_len = len(row)
            pad_size = max_len - row_len
            if align_to == "start":
                padded = np.concatenate([row, np.full(pad_size, np.nan)])
            elif align_to == "end":
                padded = np.concatenate([np.full(pad_size, np.nan), row])
            else:
                padded = row  # no alignment
            aligned.append(padded)

        return np.array(aligned)

    def _sort_by_non_nan_length(self, arr_list):
        valid_counts = [np.sum(~np.isnan(arr)) for arr in arr_list]
        sorted_indices = np.argsort(valid_counts)[::-1]
        return sorted_indices

    def _extract_spike_times(self, spike_array):
        """
        Clean spike array of NaNs and extract spike time indices.
        """
        if np.isnan(spike_array).all():
            return []
        clean = np.nan_to_num(spike_array, nan=0.0)
        return np.where(clean == 1)[0]

    def _plot_wholedata_cellwise(self):
        print(f"Running _plot_wholedata_cellwise for dataset: {self.dataset_name}")

        if self.data is None:
            print("No data to plot.")
            return

        if (
            "taste" not in self.data.coords
            or "trial" not in self.data.coords
            or "neuron" not in self.data.coords
        ):
            print("Required coordinates missing in data.")
            return

        for taste_idx in self.data.coords["taste"].values:
            trials = self.data.coords["trial"].values
            neurons = self.data.coords["neuron"].values

            print(f"  Taste {taste_idx}: {len(trials)} trials, {len(neurons)} neurons")

            # Make sure the taste folder exists
            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)

            fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)

            for ax_idx, neuron_idx in enumerate(
                neurons[:4]
            ):  # Only plot 4 neurons per figure for now
                ax = axes[ax_idx]
                print(f"    Plotting Neuron {neuron_idx} on axis {ax_idx}")

                for trial_idx in trials:
                    try:
                        spikes = self.data.sel(
                            taste=taste_idx, neuron=neuron_idx, trial=trial_idx
                        ).values
                    except Exception as e:
                        print(
                            f"    Skipping trial {trial_idx} due to selection error: {e}"
                        )
                        continue

                    spike_times = self._extract_spike_times(spikes)
                    if len(spike_times) > 0:
                        ax.scatter(
                            spike_times,
                            [trial_idx] * len(spike_times),
                            color="k",
                            marker="|",
                            s=20,
                        )

                    if self.changepoints_dict:
                        cps = self.changepoints_dict[self.dataset_name][taste_idx][
                            trial_idx
                        ]
                        for cp in cps:
                            ax.axvline(x=cp, color="blue", linestyle="--", linewidth=1)

                    ax.axvline(x=2000, color="red", linestyle="--", linewidth=1)

                ax.set_title(f"Neuron {neuron_idx + 1}")

            fig.suptitle(
                f"Whole-Data Cell-Wise Raster\nTaste {taste_idx}, Dataset {self.dataset_name}",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

            # Legend (once per fig)
            red_line = mlines.Line2D(
                [], [], color="red", linestyle="--", label="Stimulus"
            )
            blue_line = mlines.Line2D(
                [], [], color="blue", linestyle="--", label="Changepoint"
            )
            fig.legend(
                handles=[red_line, blue_line],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
            )

            fname = f"wholedata_cellwise_taste_{taste_idx}_{self.dataset_name}.png"
            full_path = os.path.join(taste_dir, fname)
            print(f"  Saving figure: {full_path}")
            plt.savefig(full_path)
            plt.close(fig)

    def _plot_wholedata_trialwise(self):
        print(f"Running _plot_wholedata_trialwise for dataset: {self.dataset_name}")

        if self.data is None:
            print("No data to plot.")
            return

        if (
            "taste" not in self.data.coords
            or "trial" not in self.data.coords
            or "neuron" not in self.data.coords
        ):
            print("Required coordinates missing in data.")
            return

        for taste_idx in self.data.coords["taste"].values:
            trials = self.data.coords["trial"].values
            neurons = self.data.coords["neuron"].values

            print(f"  Taste {taste_idx}: {len(trials)} trials, {len(neurons)} neurons")

            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)

            # Define starting time
            t_start = 1500

            for chunk_start in range(0, len(trials), 4):
                fig, axes = plt.subplots(
                    4, 1, figsize=(10, 12), sharex=True, sharey=True
                )
                fig_axes = axes if isinstance(axes, np.ndarray) else [axes]

                for ax_idx, trial_idx in enumerate(
                    trials[chunk_start : chunk_start + 4]
                ):
                    ax = fig_axes[ax_idx]
                    max_time = 0

                    for neuron_idx in neurons:
                        segs = []
                        for segment_idx in self.data.coords["segment"].values:
                            segment = self.data.sel(
                                taste=taste_idx,
                                trial=trial_idx,
                                neuron=neuron_idx,
                                segment=segment_idx,
                            ).values
                            segment_clean = segment[~np.isnan(segment)]
                            segs.append(segment_clean)
                        full_vector = np.concatenate(segs)
                        spike_indices = np.where(full_vector == 1)[0]
                        spike_times = spike_indices + t_start

                        if spike_times.size > 0:
                            ax.scatter(
                                spike_times,
                                [neuron_idx] * len(spike_times),
                                color="k",
                                marker="|",
                                s=20,
                            )
                            max_time = max(max_time, spike_times[-1])

                    if self.changepoints_dict:
                        cps = self.changepoints_dict[self.dataset_name][taste_idx][
                            trial_idx
                        ]
                        for cp in cps:
                            if t_start <= cp <= max_time:
                                ax.axvline(
                                    x=cp, color="blue", linestyle="--", linewidth=1
                                )

                    if t_start <= 2000 <= max_time:
                        ax.axvline(x=2000, color="red", linestyle="--", linewidth=1)

                    ax.set_xlim(t_start, max_time + 100)
                    ax.set_xticks(np.linspace(t_start, max_time, num=9, dtype=int))
                    ax.set_xticklabels(
                        [str(int(t)) for t in np.linspace(t_start, max_time, num=9)]
                    )
                    ax.set_title(f"Trial {trial_idx + 1}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Neuron")
                    ax.tick_params(labelbottom=True)

                fig.suptitle(
                    f"Whole-Data Trial-Wise Raster\nTaste {taste_idx}, Dataset {self.dataset_name}",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])

                red_line = mlines.Line2D(
                    [], [], color="red", linestyle="--", label="Stimulus"
                )
                blue_line = mlines.Line2D(
                    [], [], color="blue", linestyle="--", label="Changepoint"
                )
                fig.legend(
                    handles=[red_line, blue_line],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=2,
                )

                chunk_label = f"chunk_{chunk_start // 4 + 1}"
                fname_base = f"wholedata_trialwise_taste_{taste_idx}_{self.dataset_name}_{chunk_label}"
                plot_path = os.path.join(taste_dir, fname_base)

                fig.savefig(f"{plot_path}.png", dpi=300)
                fig.savefig(f"{plot_path}.svg", format="svg")
                plt.close(fig)

    def _plot_cellwise(self):
        for taste_idx in self.data.coords["taste"].values:
            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)
            for epoch_idx in self.data.coords["segment"].values:
                neuron_list = self.data.coords["neuron"].values
                trial_list = self.data.coords["trial"].values

                num_neurons = len(neuron_list)
                num_cols = 5
                num_rows = math.ceil(num_neurons / num_cols)

                fig, axes = plt.subplots(
                    num_rows,
                    num_cols,
                    figsize=(15, 3 * num_rows),
                    sharex=True,
                    sharey=True,
                )
                axes = axes.flatten()

                for i, neuron_idx in enumerate(neuron_list):
                    ax = axes[i]
                    spike_rows = []
                    for trial_idx in trial_list:
                        spikes = self.data.sel(
                            taste=taste_idx,
                            trial=trial_idx,
                            segment=epoch_idx,
                            neuron=neuron_idx,
                        ).values
                        spike_rows.append(spikes)

                    if not self.is_warped and self.alignment:
                        spike_rows = self._pad_align(spike_rows, self.alignment)

                    if not self.is_warped and self.sort_by_length:
                        sorted_idx = self._sort_by_non_nan_length(spike_rows)
                        trial_list = trial_list[sorted_idx]
                        spike_rows = [spike_rows[i] for i in sorted_idx]

                    for j, (trial_idx, spikes) in enumerate(
                        zip(trial_list, spike_rows)
                    ):
                        spike_times = np.where(spikes == 1)[0]
                        ax.scatter(
                            spike_times,
                            [j] * len(spike_times),
                            color="k",
                            marker="|",
                            s=20,
                        )

                        if self.show_markers and self.changepoints_dict:
                            cp = self.changepoints_dict[self.dataset_name][taste_idx][
                                trial_idx
                            ]
                            ax.scatter(
                                [cp[epoch_idx]], [j], color="blue", marker="|", s=20
                            )
                            ax.scatter(
                                [cp[epoch_idx + 1]], [j], color="red", marker="|", s=20
                            )

                    ax.set_title(f"Neuron {neuron_idx+1}")

                fig.suptitle(
                    f"{self.mode.title()} Raster\nDataset {self.dataset_name}, Taste {taste_idx}, Epoch {epoch_idx}, {self.base_name}",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])

                if self.show_markers:
                    blue_marker = mlines.Line2D(
                        [],
                        [],
                        color="blue",
                        marker="|",
                        linewidth=0,
                        label="Epoch Start",
                    )
                    red_marker = mlines.Line2D(
                        [], [], color="red", marker="|", linewidth=0, label="Epoch End"
                    )
                    fig.legend(
                        handles=[blue_marker, red_marker],
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0),
                    )

                fig_filename = (
                    f"Taste_{taste_idx}_Epoch_{epoch_idx}_{self.dataset_name}.png"
                )
                fig.savefig(os.path.join(taste_dir, fig_filename))
                plt.close(fig)

    def _plot_trialwise(self):
        for taste_idx in self.data.coords["taste"].values:
            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)
            for epoch_idx in self.data.coords["segment"].values:
                trial_list = self.data.coords["trial"].values
                neuron_list = self.data.coords["neuron"].values

                num_trials = len(trial_list)
                num_cols = 5
                num_rows = math.ceil(num_trials / num_cols)

                fig, axes = plt.subplots(
                    num_rows,
                    num_cols,
                    figsize=(15, 3 * num_rows),
                    sharex=True,
                    sharey=True,
                )
                axes = axes.flatten()

                for i, trial_idx in enumerate(trial_list):
                    ax = axes[i]
                    spike_rows = []
                    for neuron_idx in neuron_list:
                        spikes = self.data.sel(
                            taste=taste_idx,
                            trial=trial_idx,
                            segment=epoch_idx,
                            neuron=neuron_idx,
                        ).values
                        spike_rows.append(spikes)

                    if not self.is_warped and self.alignment:
                        spike_rows = self._pad_align(spike_rows, self.alignment)

                    if not self.is_warped and self.sort_by_length:
                        sorted_idx = self._sort_by_non_nan_length(spike_rows)
                        neuron_list = neuron_list[sorted_idx]
                        spike_rows = [spike_rows[i] for i in sorted_idx]

                    for j, (neuron_idx, spikes) in enumerate(
                        zip(neuron_list, spike_rows)
                    ):
                        spike_times = np.where(spikes == 1)[0]
                        ax.scatter(
                            spike_times,
                            [j] * len(spike_times),
                            color="k",
                            marker="|",
                            s=20,
                        )

                        if self.show_markers and self.changepoints_dict:
                            cp = self.changepoints_dict[self.dataset_name][taste_idx][
                                trial_idx
                            ]
                            ax.scatter(
                                [cp[epoch_idx]], [j], color="blue", marker="|", s=20
                            )
                            ax.scatter(
                                [cp[epoch_idx + 1]], [j], color="red", marker="|", s=20
                            )

                    ax.set_title(f"Trial {trial_idx}")

                fig.suptitle(
                    f"{self.mode.title()} Raster\nDataset {self.dataset_name}, Taste {taste_idx}, Epoch {epoch_idx}, {self.base_name}",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])

                if self.show_markers:
                    blue_marker = mlines.Line2D(
                        [],
                        [],
                        color="blue",
                        marker="|",
                        linewidth=0,
                        label="Epoch Start",
                    )
                    red_marker = mlines.Line2D(
                        [], [], color="red", marker="|", linewidth=0, label="Epoch End"
                    )
                    fig.legend(
                        handles=[blue_marker, red_marker],
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0),
                    )

                fig_filename = f"Trialwise_Taste_{taste_idx}_Epoch_{epoch_idx}_{self.dataset_name}.png"
                fig.savefig(os.path.join(taste_dir, fig_filename))
                plt.close(fig)

    # Utility: Extract spike array for inspection from class
    @staticmethod
    def debug_extract_from_calc(all_calc_objs):
        """
        Run debug mode with dataset selection and looped spike array inspection.
        """
        print("\n--- Debug Mode: Spike Train Extractor ---")

        if not all_calc_objs:
            print("No datasets loaded. Cannot debug.")
            return

        print("Available datasets:")
        dataset_names = list(all_calc_objs.keys())
        for i, name in enumerate(dataset_names):
            print(f"  [{i}] {name}")

        selected_idx = int(input("Select dataset index: "))
        selected_name = dataset_names[selected_idx]
        calc = all_calc_objs[selected_name]

        print(f"\nSelected dataset: {selected_name}")
        print("Type 'exit' or 'q' at any time to quit.\n")

        while True:
            taste = input("Taste index (int): ")
            if taste.lower() in ["q", "exit"]:
                break
            trial = input("Trial index (int): ")
            if trial.lower() in ["q", "exit"]:
                break
            neuron = input("Neuron index (int): ")
            if neuron.lower() in ["q", "exit"]:
                break
            segment = input("Segment index (or blank if none): ").strip()
            if segment.lower() in ["q", "exit"]:
                break

            try:
                taste = int(taste)
                trial = int(trial)
                neuron = int(neuron)
                segment = int(segment) if segment else None

                result = SpikeRasterPlotter.extract_sample_spike_array_from_calc(
                    calc, taste=taste, trial=trial, neuron=neuron, segment=segment
                )

                print("\nExtracted spike array:")
                print(result)
            except Exception as e:
                print(f"Error extracting spike array: {e}")

    @staticmethod
    def extract_sample_spike_array_from_calc(
        calc_obj, taste=0, trial=0, neuron=0, segment=None
    ):
        """
        Extracts a sample spike array from a CalcFRStates instance.
        This checks both warped and unwarped spike arrays.
        """
        if calc_obj.spike_arrays_unwarped is not None:
            return SpikeRasterPlotter.extract_sample_spike_array(
                calc_obj.spike_arrays_unwarped, taste, trial, neuron, segment
            )
        elif calc_obj.spike_arrays_warped is not None:
            return SpikeRasterPlotter.extract_sample_spike_array(
                calc_obj.spike_arrays_warped, taste, trial, neuron, segment
            )
        else:
            raise ValueError("No spike arrays available in the CalcFRStates object.")

    @staticmethod
    def extract_sample_spike_array(
        data_array: xr.DataArray, taste=0, trial=0, neuron=0, segment=None
    ):
        """
        Utility function to help debug the structure of the spike train data.
        Returns the 1D numpy array representing the spike train for the given selection.
        If 'segment' is None, expects no segment dimension in the data.
        """
        if segment is not None and "segment" in data_array.dims:
            return data_array.sel(
                taste=taste, trial=trial, neuron=neuron, segment=segment
            ).values
        else:
            return data_array.sel(taste=taste, trial=trial, neuron=neuron).values


# individual functions that do this stuff-- commented out; still have some functionality that I need to adapt to being a part of the class:
# the class was built off of each of these

# base_output_dir ='/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/SPIKE_TRAIN_VIS/cell-wise-warped'
# def process_and_plot_datasets(
#     npz_path,
#     pkl_path,
#     base_output_dir,
#     window_length=200,
#     step_size=50
# ):
#     """
#     Loops over all datasets in the .npz, extracts spike arrays,
#     gets valid changepoints, calls CalcFRStates to compute FR/spike arrays,
#     and produces one figure per (taste, epoch) with up to 30 subplots.
#     """

#     # Example of how to get a "base_name" from .npz_path
#     filename = os.path.basename(npz_path)
#     base_name = filename.split("_repacked.npz")[0]

#     # Iterate through each dataset in the NPZ
#     for data in extract_from_npz(npz_path):
#         if isinstance(data, tuple) and len(data) == 4:
#             spike_array, dataset_num, index, key = data

#             # Convert dataset_num to str and strip out "_repacked.npz" if present
#             dataset_num_str = str(dataset_num)
#             dataset_num_clean = dataset_num_str.split("_repacked.npz")[0]
#             core_dataset_name = dataset_num_clean

#             print(f"Processing dataset number: {dataset_num_clean}")

#             # Extract changepoints
#             extracted_pkl = extract_valid_changepoints(
#                 pkl_path,
#                 spike_array,
#                 dataset_num_clean,
#                 index,
#                 key
#             )

#             if extracted_pkl is not None:
#                 try:
#                     # Suppose the 4th column has changepoints
#                     changepoints = extracted_pkl[:, 3]

#                     # Use your new class to compute FR/spike arrays
#                     calc = CalcFRStates(
#                         spike_array=spike_array,
#                         changepoints=changepoints,
#                         window_length=window_length,
#                         step_size=step_size,
#                         compute_unwarped_spike_arrays=True,
#                         compute_unwarped_firing_rates=False,
#                         compute_warped_spike_arrays=True,
#                         compute_warped_firing_rates=False,
#                         fixed_warp_duration=1000  # <-- Warp all segments to 1000 ms; ignore if we don't want this
#                     )

#                     (state_firing_rates_all_trials,
#                      state_spike_arrays_all_trials,
#                      state_firing_rates_warped,
#                      state_spike_arrays_warped) = calc.run()

#                     # Now create the folder for this dataset
#                     dataset_dir = os.path.join(base_output_dir, f"dataset_{dataset_num_clean}")
#                     os.makedirs(dataset_dir, exist_ok=True)

#                     # Plot the unwarped data in 30-subplot raster figures
#                     # (assuming state_spike_arrays_all_trials is an xarray DataArray)
#                     if isinstance(state_spike_arrays_all_trials, str):
#                         # If you returned dummy strings above, you'll need real data for real plotting
#                         print("Warning: `state_spike_arrays_all_trials` is dummy data in this example.")
#                     else:
#                         # plot_spike_rasters_30(
#                         #     state_spike_arrays_all_trials,
#                         #     dataset_dir,
#                         #     dataset_num_clean,
#                         #     base_name
#                         # )
#             #come up with a better way to do than then a comment
#                         plot_spike_rasters_cells_on_xaxis(
#                             state_spike_arrays_warped,
#                             dataset_dir,
#                             dataset_num_clean,
#                             base_name
#                         )
#                         # plot_spike_rasters_transition_aligned(
#                         #     state_spike_arrays_warped,
#                         #     standardized_changepoints_dict,
#                         #     core_dataset_name,
#                         #     dataset_dir,
#                         #     base_name
#                         # )
#                         # plot_spike_rasters_transition_aligned_sorted(
#                         #     state_spike_arrays_all_trials,
#                         #     standardized_changepoints_dict,
#                         #     core_dataset_name,
#                         #     dataset_dir,
#                         #     base_name
#                         # )

#                 except Exception as e:
#                     print(f"Error processing dataset {dataset_num_clean}: {e}")
#             else:
#                 print(f"No valid changepoints for dataset {dataset_num_clean}; skipping...")

#     print("All datasets processed.")


# def plot_spike_rasters_30(
#     state_spike_arrays_all_trials,
#     dataset_dir,
#     dataset_num_clean,
#     base_name
# ):
#     """
#     Example function that plots all 30 trials in a 6x5 grid per (taste, epoch),
#     storing the figures in `dataset_dir`.
#     """
#     # For each taste...
#     for taste_idx in state_spike_arrays_all_trials.coords["taste"].values:
#         # Make subdirectory for taste
#         taste_dir = os.path.join(dataset_dir, f"taste_{taste_idx}")
#         os.makedirs(taste_dir, exist_ok=True)

#         # For each epoch...
#         for epoch_idx in state_spike_arrays_all_trials.coords["segment"].values:
#             trial_list = state_spike_arrays_all_trials.coords["trial"].values

#             # Create a figure with 6 rows × 5 columns = 30 subplots
#             fig, axes = plt.subplots(nrows=6, ncols=5,
#                                      figsize=(14, 16),
#                                      sharex=True, sharey=True)
#             axes = axes.flatten()

#             # Plot each trial up to 30
#             for i, trial_idx in enumerate(trial_list[:30]):
#                 ax = axes[i]
#                 spike_data = state_spike_arrays_all_trials.sel(
#                     taste=taste_idx,
#                     trial=trial_idx,
#                     segment=epoch_idx
#                 ).values

#                 # For each neuron
#                 for neuron_idx in range(spike_data.shape[0]):
#                     spike_times = np.where(spike_data[neuron_idx] == 1)[0]
#                     ax.scatter(
#                         spike_times,
#                         [neuron_idx]*len(spike_times),
#                         color='k',
#                         marker='|',
#                         s=20
#                     )
#                 ax.set_title(f"Trial {trial_idx}")

#             # Hide unused subplots if fewer than 30 trials
#             for j in range(len(trial_list), 30):
#                 axes[j].set_visible(False)

#             # Suptitle
#             fig.suptitle(f"Dataset {dataset_num_clean}, Taste {taste_idx}, Epoch {epoch_idx}, {base_name}",
#                          fontsize=14)

#             # Save with a filename referencing the cleaned dataset_num
#             fig_filename = f"Taste_{taste_idx}_Epoch_{epoch_idx}_{dataset_num_clean}.png"
#             fig_path = os.path.join(taste_dir, fig_filename)

#             plt.tight_layout(rect=[0, 0, 1, 0.94])
#             plt.savefig(fig_path)
#             plt.close(fig)

#     print(f"Finished plotting for dataset {dataset_num_clean}.")
# import math
# def plot_spike_rasters_cells_on_xaxis(
#     spike_data_array,
#     dataset_dir,
#     dataset_num_clean,
#     base_name
# ):
#     """
#     Creates a single figure per (taste, epoch), with one subplot per neuron,
#     arranged in a grid of 5 columns × enough rows to accommodate all neurons.

#     Each subplot is a single neuron, with trials on the y-axis and time on the x-axis.
#     Spikes are in black.

#     Parameters
#     ----------
#     spike_data_array : xarray.DataArray
#         Spike data with dimensions: [taste, trial, segment, neuron, time].
#         This can be any appropriately-shaped dataset, warped or unwarped.
#     dataset_dir : str
#         Path to the dataset output folder, e.g. 'dataset_001'.
#     dataset_num_clean : str
#         Cleaned dataset number, e.g. '001'.
#     base_name : str
#         Base name from the .npz, e.g. everything before "_repacked.npz".
#     """

#     # Loop over tastes
#     for taste_idx in spike_data_array.coords["taste"].values:
#         # Make subdirectory for this taste
#         taste_dir = os.path.join(dataset_dir, f"taste_{taste_idx}")
#         os.makedirs(taste_dir, exist_ok=True)

#         # Loop over epochs
#         for epoch_idx in spike_data_array.coords["segment"].values:
#             # Gather neurons and trials
#             neuron_list = spike_data_array.coords["neuron"].values
#             trial_list = spike_data_array.coords["trial"].values

#             num_neurons = len(neuron_list)
#             num_cols = 5
#             num_rows = math.ceil(num_neurons / num_cols)

#             # Create figure with enough rows × 5 columns
#             fig, axes = plt.subplots(
#                 nrows=num_rows,
#                 ncols=num_cols,
#                 figsize=(15, 3 * num_rows),
#                 sharex=True,
#                 sharey=True
#             )

#             if num_rows == 1 and num_cols == 1:
#                 axes = np.array([axes])
#             axes = axes.flatten()

#             # Plot each neuron in its own subplot
#             for i, neuron_idx in enumerate(neuron_list):
#                 ax = axes[i]

#                 for trial_idx in trial_list:
#                     spike_data = spike_data_array.sel(
#                         taste=taste_idx,
#                         trial=trial_idx,
#                         segment=epoch_idx,
#                         neuron=neuron_idx
#                     ).values

#                     spike_times = np.where(spike_data == 1)[0]
#                     ax.scatter(
#                         spike_times,
#                         [trial_idx] * len(spike_times),
#                         color='k',
#                         marker='|',
#                         s=20
#                     )

#                 ax.set_title(f"Neuron {neuron_idx}")
#                 ax.set_ylabel("Trial")

#             # Optional: hide unused subplots
#             # for j in range(num_neurons, len(axes)):
#             #     axes[j].set_visible(False)

#             fig.suptitle(
#                 f"Warped Dataset {dataset_num_clean}, Taste {taste_idx}, Epoch {epoch_idx}, {base_name}",
#                 fontsize=14
#             )

#             fig_filename = f"Taste_{taste_idx}_Epoch_{epoch_idx}_{dataset_num_clean}.png"
#             fig_path = os.path.join(taste_dir, fig_filename)

#             plt.tight_layout(rect=[0, 0, 1, 0.93])
#             plt.savefig(fig_path)
#             plt.close(fig)

#     print(f"Finished plotting (cells on x-axis) for dataset {dataset_num_clean}.")


# import matplotlib.lines as mlines

# def plot_spike_rasters_transition_aligned(
#     state_spike_arrays_all_trials,
#     standardized_changepoints_dict,
#     core_dataset_name,
#     dataset_dir,
#     base_name
# ):
#     """
#     Plots "transition-aligned" rasters, looping over both taste and epoch (segment).
#     Each figure = one (taste, epoch). Subplots for all neurons in a 5-col grid.

#     The x-axis is in "absolute" time (ms), determined by offsetting each local
#     spike index by the start boundary of that epoch. Blue spikes mark the epoch's start
#     for each trial; red spikes mark the epoch's end for each trial; black are normal spikes.

#     Assumes:
#     - state_spike_arrays_all_trials has dims [taste, trial, segment, neuron, time].
#       * For each epoch, "time" runs from 0..(end-start) (local indices).
#     - standardized_changepoints_dict[core_dataset_name][taste_idx][trial_idx]
#       has the boundary times for that trial (e.g. 5 boundary times if there are 4 segments).
#     """

#     if core_dataset_name not in standardized_changepoints_dict:
#         print(f"Changepoints not found for {core_dataset_name}. Skipping...")
#         return

#     # For convenience:
#     changepoints_for_dataset = standardized_changepoints_dict[core_dataset_name]

#     taste_values = state_spike_arrays_all_trials.coords["taste"].values
#     trial_values = state_spike_arrays_all_trials.coords["trial"].values
#     epoch_values = state_spike_arrays_all_trials.coords["segment"].values  # e.g. [0,1,2,3]
#     neuron_values = state_spike_arrays_all_trials.coords["neuron"].values

#     # Loop over tastes
#     for taste_idx in taste_values:
#         # Loop over epochs (segments)
#         for epoch_idx in epoch_values:
#             # 1) Figure out the global min and max for this epoch across all trials
#             #    Typically we do: start boundary = cp[epoch_idx], end boundary = cp[epoch_idx+1].
#             #    We'll gather them for each trial to set x_min and x_max.
#             all_starts = []
#             all_ends = []

#             for trial_idx in trial_values:
#                 # For trial t, the boundaries array might be something like [1000, 1500, 2000, 2500, 3000]
#                 # if there are 4 segments => 5 boundary times.
#                 trial_cps = changepoints_for_dataset[taste_idx][trial_idx]
#                 if epoch_idx >= len(trial_cps) - 1:
#                     # If you only have exactly 4 boundary times for 4 segments,
#                     # then epoch_idx 3 = cp[3] to cp[4], but if cp[4] doesn't exist,
#                     # you need to handle or skip. We'll do a safety check:
#                     continue

#                 start_t = trial_cps[epoch_idx]
#                 end_t   = trial_cps[epoch_idx + 1]
#                 all_starts.append(start_t)
#                 all_ends.append(end_t)

#             if len(all_starts) == 0 or len(all_ends) == 0:
#                 # Means we couldn't find valid boundaries for this epoch => skip
#                 continue

#             x_min = min(all_starts)
#             x_max = max(all_ends)

#             # 2) Make a figure for this (taste, epoch)
#             num_neurons = len(neuron_values)
#             num_cols = 5
#             num_rows = math.ceil(num_neurons / num_cols)

#             fig, axes = plt.subplots(
#                 nrows=num_rows,
#                 ncols=num_cols,
#                 figsize=(15, 3 * num_rows),
#                 sharex=True,
#                 sharey=True
#             )

#             if isinstance(axes, np.ndarray):
#                 axes = axes.flatten()
#             else:
#                 axes = [axes]

#             # 3) Plot each neuron in its own subplot
#             for i, neuron_idx in enumerate(neuron_values):
#                 ax = axes[i]
#                 ax.set_xlim([x_min, x_max])

#                 # Plot each trial
#                 for trial_idx in trial_values:
#                     trial_cps = changepoints_for_dataset[taste_idx][trial_idx]
#                     if epoch_idx >= len(trial_cps) - 1:
#                         continue
#                     epoch_start = trial_cps[epoch_idx]      # Blue spike
#                     epoch_end   = trial_cps[epoch_idx + 1]  # Red spike

#                     # a) Local spike indices from the data array:
#                     # shape: (time,) => from 0..(end-start)
#                     local_spike_data = state_spike_arrays_all_trials.sel(
#                         taste=taste_idx,
#                         trial=trial_idx,
#                         segment=epoch_idx,
#                         neuron=neuron_idx
#                     ).values  # 0 or 1

#                     local_spike_times = np.where(local_spike_data == 1)[0]  # e.g. [0, 1, 5, ...]

#                     # b) Convert local times to absolute times by adding epoch_start
#                     #    so if local time=0 => absolute time=epoch_start
#                     abs_spike_times = local_spike_times + epoch_start

#                     # c) Plot black spikes-- all markers the same size
#                     ax.scatter(
#                         abs_spike_times,
#                         [trial_idx]*len(abs_spike_times),
#                         color='k',
#                         marker='|',
#                         s=20
#                     )

#                     # d) Plot a single blue spike at epoch_start
#                     ax.scatter(
#                         [epoch_start], [trial_idx],
#                         color='blue',
#                         marker='|',
#                         s=20
#                     )
#                     # e) Plot a single red spike at epoch_end
#                     ax.scatter(
#                         [epoch_end], [trial_idx],
#                         color='red',
#                         marker='|',
#                         s=20
#                     )

#                 ax.set_title(f"Neuron {neuron_idx}")
#                 ax.set_ylabel("Trial")

#             # If more subplots than neurons, they'll remain blank (you said don't hide them).

#             # 4) Legend at bottom
#             blue_marker = mlines.Line2D([], [], color='blue', marker='|', linewidth=0, label='Epoch Start')
#             red_marker  = mlines.Line2D([], [], color='red',  marker='|', linewidth=0, label='Epoch End')
#             fig.legend(
#                 handles=[blue_marker, red_marker],
#                 loc='lower center',
#                 bbox_to_anchor=(0.5, 0),
#                 fancybox=False,
#                 shadow=False,
#                 ncol=2
#             )

#             # 5) Suptitle
#             fig.suptitle(
#                 f"Transition-Aligned Raster\n"
#                 f"Dataset {core_dataset_name}, Taste {taste_idx}, Epoch {epoch_idx}, {base_name}",
#                 fontsize=14
#             )

#             # 6) Save figure
#             output_subdir = os.path.join(dataset_dir, f"taste_{taste_idx}")
#             os.makedirs(output_subdir, exist_ok=True)

#             fig_filename = (f"transition_aligned_taste_{taste_idx}_epoch_{epoch_idx}_"
#                             f"{core_dataset_name}.png")
#             fig_path = os.path.join(output_subdir, fig_filename)

#             plt.tight_layout(rect=[0, 0.05, 1, 0.90])  # space for legend
#             plt.savefig(fig_path)
#             plt.close(fig)

#     print(f"Finished transition-aligned plotting (with epochs) for dataset {core_dataset_name}.")

# def plot_spike_rasters_transition_aligned_sorted(
#     state_spike_arrays_all_trials,
#     standardized_changepoints_dict,
#     core_dataset_name,
#     dataset_dir,
#     base_name
# ):
#     """
#     Plots "transition-aligned" rasters, looping over taste and epoch (segment),
#     but sorts trials by their epoch-start time so the earliest-starting trial
#     is at the bottom and the latest-starting trial is at the top.

#     Each figure = one (taste, epoch). Subplots for all neurons in a 5-col grid.
#     The x-axis is absolute time (ms). We place a single blue spike at epoch-start
#     for each trial, a single red spike at epoch-end, and normal spikes in black.

#     Assumes:
#     - state_spike_arrays_all_trials has dims [taste, trial, segment, neuron, time].
#       For each epoch, "time" runs 0..(end-start).
#     - standardized_changepoints_dict[core_dataset_name][taste_idx][trial_idx]
#       yields the boundary times for that trial. If there are 4 segments, typically
#       we have 5 boundary times. The epoch_idx-th segment is [cp[epoch_idx], cp[epoch_idx+1]).
#     """

#     if core_dataset_name not in standardized_changepoints_dict:
#         print(f"Changepoints not found for {core_dataset_name}. Skipping...")
#         return

#     changepoints_for_dataset = standardized_changepoints_dict[core_dataset_name]

#     taste_values = state_spike_arrays_all_trials.coords["taste"].values
#     trial_values = state_spike_arrays_all_trials.coords["trial"].values
#     epoch_values = state_spike_arrays_all_trials.coords["segment"].values
#     neuron_values = state_spike_arrays_all_trials.coords["neuron"].values

#     # Loop over tastes
#     for taste_idx in taste_values:
#         # Loop over epochs
#         for epoch_idx in epoch_values:
#             # 1) For each trial, figure out the epoch's start time
#             #    Then we can sort trials by that time
#             trial_start_times = []
#             for t in trial_values:
#                 trial_cps = changepoints_for_dataset[taste_idx][t]
#                 # Safety check: ensure we have epoch_idx+1 within bounds
#                 if epoch_idx >= len(trial_cps) - 1:
#                     continue
#                 start_t = trial_cps[epoch_idx]
#                 trial_start_times.append((t, start_t))

#             if not trial_start_times:
#                 # No valid data for this epoch => skip
#                 continue

#             # Sort by start time ascending
#             trial_start_times.sort(key=lambda x: x[1])

#             # Now we have a sorted list like [(trial_idx, start_time), ...] in ascending order
#             # We'll store an index -> y_value mapping. The earliest trial => y=0, next => y=1, etc.
#             # So the earliest-starting trial is at the bottom, the latest is at the top.
#             trial_to_y = {}
#             for rank, (t_idx, start_t) in enumerate(trial_start_times):
#                 trial_to_y[t_idx] = rank

#             # We'll also find the global min & max for the x-axis
#             all_starts = []
#             all_ends = []
#             for (t_idx, start_t) in trial_start_times:
#                 trial_cps = changepoints_for_dataset[taste_idx][t_idx]
#                 # epoch start = trial_cps[epoch_idx]
#                 # epoch end   = trial_cps[epoch_idx+1]
#                 end_t = trial_cps[epoch_idx + 1]
#                 all_starts.append(start_t)
#                 all_ends.append(end_t)

#             x_min = min(all_starts)
#             x_max = max(all_ends)

#             # 2) Set up figure for this (taste, epoch)
#             num_neurons = len(neuron_values)
#             num_cols = 5
#             num_rows = math.ceil(num_neurons / num_cols)

#             fig, axes = plt.subplots(
#                 nrows=num_rows,
#                 ncols=num_cols,
#                 figsize=(15, 3 * num_rows),
#                 sharex=True,
#                 sharey=True
#             )

#             if isinstance(axes, np.ndarray):
#                 axes = axes.flatten()
#             else:
#                 axes = [axes]

#             # 3) Plot each neuron in its own subplot
#             for i, neuron_idx in enumerate(neuron_values):
#                 ax = axes[i]
#                 ax.set_xlim([x_min, x_max])

#                 # For each trial in ascending order of start time
#                 for (t_idx, start_time) in trial_start_times:
#                     trial_cps = changepoints_for_dataset[taste_idx][t_idx]
#                     epoch_start = trial_cps[epoch_idx]
#                     epoch_end   = trial_cps[epoch_idx + 1]

#                     # local spike data
#                     local_spike_data = state_spike_arrays_all_trials.sel(
#                         taste=taste_idx,
#                         trial=t_idx,
#                         segment=epoch_idx,
#                         neuron=neuron_idx
#                     ).values
#                     local_spike_times = np.where(local_spike_data == 1)[0]
#                     abs_spike_times = local_spike_times + epoch_start

#                     # get the sorted Y-value for this trial
#                     y_val = trial_to_y[t_idx]

#                     # plot black spikes
#                     ax.scatter(
#                         abs_spike_times,
#                         [y_val]*len(abs_spike_times),
#                         color='k',
#                         marker='|',
#                         s=20
#                     )

#                     # single blue spike at epoch_start
#                     ax.scatter(
#                         [epoch_start], [y_val],
#                         color='blue',
#                         marker='|',
#                         s=20
#                     )
#                     # single red spike at epoch_end
#                     ax.scatter(
#                         [epoch_end], [y_val],
#                         color='red',
#                         marker='|',
#                         s=20
#                     )

#                 ax.set_title(f"Neuron {neuron_idx}")
#                 ax.set_ylabel("Trial (sorted by start time)")

#             # If there's leftover axes beyond #neurons, we do not hide them, as requested.

#             # 4) Legend at bottom
#             blue_marker = mlines.Line2D([], [], color='blue', marker='|', linewidth=0, label='Epoch Start')
#             red_marker  = mlines.Line2D([], [], color='red',  marker='|', linewidth=0, label='Epoch End')
#             fig.legend(
#                 handles=[blue_marker, red_marker],
#                 loc='lower center',
#                 bbox_to_anchor=(0.5, 0),
#                 fancybox=False,
#                 shadow=False,
#                 ncol=2
#             )

#             # 5) Suptitle
#             fig.suptitle(
#                 f"Transition-Aligned Raster (Sorted Trials)\n"
#                 f"Dataset {core_dataset_name}, Taste {taste_idx}, Epoch {epoch_idx}, {base_name}",
#                 fontsize=14
#             )

#             # 6) Save figure
#             output_subdir = os.path.join(dataset_dir, f"taste_{taste_idx}")
#             os.makedirs(output_subdir, exist_ok=True)

#             fig_filename = (f"transition_aligned_sorted_taste_{taste_idx}_epoch_{epoch_idx}_"
#                             f"{core_dataset_name}.png")
#             fig_path = os.path.join(output_subdir, fig_filename)

#             plt.tight_layout(rect=[0, 0.05, 1, 0.90])  # space for legend
#             plt.savefig(fig_path)
#             plt.close(fig)

#     print(f"Finished transition-aligned plotting (sorted by start time) for dataset {core_dataset_name}.")
