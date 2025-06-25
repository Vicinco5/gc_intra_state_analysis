#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:13:52 2025

@author: vincentcalia-bogan
"""


## fully working class -- there are some things I would like to add including a variance plot and an SEM plot
import os
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm


class RNNWarpingIntraAroundCP:
    """
    Pipeline for warping RNN latent trajectories around changepoints,
    then plotting (warped, unwarped, or windowed) segments around each CP.
    """

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        return [
            c for c in df.columns if c.startswith("PC_") or c.startswith("latent_dim_")
        ]

    def __init__(
        self,
        standardized_changepoints_dict: dict,
        output_dir: str,
        mode: str = "unwarped",
        warp_length: int = 1000,
        window_size: int = None,
        start_time: int = 1500,
        end_time: int = 4500,
        plot_average: bool = False,
    ):
        if mode == "windowed" and window_size is None:
            raise ValueError("`window_size` must be set when mode='windowed'")
        self.changepoints = standardized_changepoints_dict
        self.output_dir = output_dir
        self.mode = mode
        self.warp_length = warp_length
        self.window_size = window_size
        self.start_time = start_time
        self.end_time = end_time
        self.plot_average = plot_average
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_changepoints(self, core_name: str):
        cp_list = self.changepoints.get(core_name)
        if cp_list is None:
            print(f"Changepoints not found for {core_name}. Skipping.")
            return None
        return [
            np.clip(np.array(arr), self.start_time, self.end_time) for arr in cp_list
        ]

    def align_trials(self, df: pl.DataFrame):
        aligned = {}
        for taste, trial in df.select(["taste", "trial"]).unique().rows():
            grp = df.filter((pl.col("taste") == taste) & (pl.col("trial") == trial))
            aligned[(taste, trial)] = grp.with_columns(
                (pl.col("time") - self.start_time).alias("time")
            )
        return aligned

    def _plot_non_windowed(
        self, dataset_name: str, df: pl.DataFrame, dims: list, cp_list: list
    ):
        tastes = df["taste"].unique().to_list()
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()
            trials_cps = cp_list[taste_idx]
            n_cps = trials_cps.shape[1]
            num_epochs = min(n_cps + 1, 4)
            fig, axes = plt.subplots(
                len(dims),
                num_epochs,
                figsize=(5 * num_epochs, 4 * len(dims)),
                sharex=(self.mode != "unwarped"),
            )
            axes = np.atleast_2d(axes)
            for row, dim in enumerate(dims):
                for eid in range(num_epochs):
                    ax = axes[row, eid]
                    epoch_vals = []
                    for trial in trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        cps = trials_cps[trial]
                        if eid == 0:
                            start, end = self.start_time, cps[0]
                        elif eid < n_cps:
                            start, end = cps[eid - 1], cps[eid]
                        else:
                            start, end = cps[-1], self.end_time
                        seg = trial_df.filter(
                            (pl.col("time") >= start) & (pl.col("time") <= end)
                        )
                        t = seg["time"].to_numpy() - start
                        v = seg[dim].to_numpy()
                        if t.size > 0:
                            if self.mode == "unwarped":
                                ax.plot(t, v, alpha=0.7, linewidth=0.5)
                                epoch_vals.append((t.astype(int), v))
                            else:
                                src = np.linspace(0, 1, t.size)
                                tgt = np.linspace(0, 1, self.warp_length)
                                f = interp1d(
                                    src, v, kind="linear", fill_value="extrapolate"
                                )
                                # play with this a lot-- as I may want to figure out which interpolation method is best.
                                # there are a lot fo types of interpolation methods that could work here; the idea is to find the right one.
                                # nearest looks quite blocky. Linear could do , but is clearly generating outliers in the extreme.
                                wv = f(tgt)
                                ta = np.linspace(0, self.warp_length, self.warp_length)
                                ax.plot(ta, wv, alpha=0.7, linewidth=0.5)
                                epoch_vals.append((ta, wv))
                    # if self.plot_average and epoch_vals:
                    #     grid=epoch_vals[0][0]
                    #     arr=np.vstack([np.interp(grid,tt,vv) for tt,vv in epoch_vals])
                    #     ax.plot(grid,np.nanmean(arr,axis=0),linewidth=2, alpha = 0.8, color='blue') # issue with unwarped; disspaearing
                    # fix below:
                    if self.plot_average and epoch_vals:
                        # Create a unified time grid from all trial times
                        grid = np.unique(np.concatenate([t for t, _ in epoch_vals]))
                        # Interpolate each trial onto the common grid
                        mat = np.vstack(
                            [
                                np.interp(grid, t, v, left=np.nan, right=np.nan)
                                for t, v in epoch_vals
                            ]
                        )
                        # Plot the mean across trials
                        ax.plot(
                            grid,
                            np.nanmean(mat, axis=0),
                            linewidth=2,
                            alpha=1,
                            color="gray",
                        )
                    ax.set_title(f"Epoch {eid}\n{dim}")
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("PC magnitude")
            fig.suptitle(
                f"Mode: {self.mode}, Taste: {taste}, Dataset: {dataset_name} (4 epochs)"
            )
            ann = (
                "Epoch 0: start→1st CP; Epoch 1: 1st→2nd; "
                f"Epoch 2: 2nd→3rd; Epoch 3: 3rd→end. \nStart and End times are T = {self.start_time} and T = {self.end_time} respecitvely. Average in gray"
            )
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.06)
            fig.text(0.5, 0.02, ann, ha="center", va="top", fontsize=16)
            od = os.path.join(self.output_dir, dataset_name, f"taste_{taste}")
            os.makedirs(od, exist_ok=True)
            fig.savefig(
                os.path.join(od, f"{self.mode}_taste_{taste}_{dataset_name}.png")
            )
            plt.close(fig)

    def _plot_windowed(
        self, dataset_name: str, df: pl.DataFrame, dims: list, cp_list: list
    ):
        # Plot mean and individual PC trajectories around stimulus (CP0) and 3 changepoints
        tastes = df["taste"].unique().to_list()
        for taste_idx, taste in enumerate(tastes):
            taste_df = df.filter(pl.col("taste") == taste)
            trials = taste_df["trial"].unique().to_list()
            trials_cps = cp_list[taste_idx]
            # Always plot 4 columns: CP0 (stimulus at 2000 ms) + up to 3 changepoints
            cp_cols = 1 + min(trials_cps.shape[1], 3)
            pc_columns = dims[:8]
            pc_count = len(pc_columns)

            fig, axs = plt.subplots(
                pc_count, cp_cols, figsize=(5 * cp_cols, 3 * pc_count), squeeze=False
            )
            fig.suptitle(
                f"RNN PCs Around Changepoints (window-aligned): Taste {taste}, Dataset {dataset_name}",
                fontsize=16,
            )

            for col in range(cp_cols):
                for row, pc in enumerate(pc_columns):
                    ax = axs[row, col]
                    trial_vals = []
                    for trial in trials:
                        trial_df = taste_df.filter(pl.col("trial") == trial)
                        if col == 0:
                            cp_time = 2000
                        else:
                            cp_time = trials_cps[trial][col - 1]
                        mask = (trial_df["time"] >= cp_time - self.window_size) & (
                            trial_df["time"] <= cp_time + self.window_size
                        )
                        seg = trial_df.filter(mask)
                        t = seg["time"].to_numpy() - cp_time
                        v = seg[pc].to_numpy()
                        if t.size > 0:
                            ax.plot(
                                t, v, linewidth=1
                            )  # change this to a rainbow-esque thing later on
                            trial_vals.append((t, v))
                    if self.plot_average and trial_vals:
                        grid = trial_vals[0][0]
                        arr = np.vstack([np.interp(grid, t, v) for t, v in trial_vals])
                        ax.plot(
                            grid,
                            np.nanmean(arr, axis=0),
                            color="gray",
                            alpha=1,
                            linewidth=2,
                        )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    label = "Stimulus" if col == 0 else f"CP {col}"
                    ax.set_title(f"{pc} — {label}")
                    ax.set_xlabel("Relative Time (ms)")
                    ax.set_ylabel("PC Value")

            # Annotation
            note = (
                "Stimulus delivery (2000 ms); CP1-3 = first three changepoints of the dataset.\n "
                "Time is relative to each CP, which is set to T=0. Data aligned to changepoint.\n average in light blue."
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.text(0.5, 0.02, note, ha="center", va="top", fontsize=16)

            out_dir = os.path.join(self.output_dir, dataset_name, f"taste_{taste}")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"windowed_taste_{taste}_{dataset_name}.png"
            fig.savefig(os.path.join(out_dir, fname))
            plt.close(fig)

    def _analyze_dataset(self, dataset_name: str, df: pl.DataFrame):
        core = dataset_name.split("_repacked_raw_latent_vectors")[0]
        cp_list = self.extract_changepoints(core)
        if cp_list is None:
            return
        dims = self.get_data_columns(df)
        if self.mode == "windowed":
            self._plot_windowed(dataset_name, df, dims, cp_list)
        else:
            self._plot_non_windowed(dataset_name, df, dims, cp_list)

    def run(self, dataset_dict: dict):
        for name, df in tqdm(dataset_dict.items(), desc="Processing datasets"):
            if not isinstance(df, pl.DataFrame):
                raise ValueError(f"Data under {name} is not a Polars DataFrame")
            self._analyze_dataset(name, df)


## I want to update this to warp to a standard 1000 ms. This will become part of a class which will do warping, epoch alignmnet, std.dev and SEM, as well as variance across time. (with an average as well ofc)
# def plot_RNN_epochs_with_interpolation(
#     epoch_dataframes_dict, standardized_changepoints_dict, output_dir, start_time=None, end_time=None
# ):
#     """
#     Plots RNN latent dimensions for each epoch using interpolation to align trial durations.

#     Parameters:
#     - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
#     - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
#     - output_dir: Directory to save the output plots.
#     - start_time: Manual start time for plotting (ms).
#     - end_time: Manual end time for plotting (ms).
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     for df_name, df in epoch_dataframes_dict.items():
#         # Extract the core name from df_name to match with standardized_changepoints_dict
#         core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]

#         if core_dataset_name not in standardized_changepoints_dict:
#             print(f"Changepoints not found for {core_dataset_name}. Skipping...")
#             continue

#         # Get the changepoint data for the dataset
#         changepoints = standardized_changepoints_dict[core_dataset_name]

#         # Create a subdirectory for the current DataFrame
#         df_output_dir = os.path.join(output_dir, df_name)
#         os.makedirs(df_output_dir, exist_ok=True)

#         # Extract unique tastes
#         unique_tastes = df['taste'].unique().to_list()

#         for taste_idx, taste in enumerate(unique_tastes):
#             # Filter for the current taste
#             taste_df = df.filter(pl.col('taste') == taste)

#             latent_dims = [col for col in df.columns if col.startswith('latent_dim_')]

#             for epoch_idx in range(len(changepoints[0]) + 1):
#                 # Define epoch boundaries
#                 if epoch_idx == 0:
#                     epoch_start = 2000
#                 else:
#                     epoch_start = changepoints[epoch_idx - 1] if epoch_idx > 0 else 2000

#                 if epoch_idx == len(changepoints[0]):
#                     epoch_end = end_time
#                 else:
#                     epoch_end = changepoints[epoch_idx] if epoch_idx < len(changepoints) else end_time

#                 # Find the longest duration for this epoch across all trials
#                 epoch_durations = []
#                 for trial_idx in range(len(changepoints)):
#                     if epoch_idx == 0:
#                         duration = changepoints[0] - 2000
#                     elif epoch_idx == len(changepoints[0]):
#                         duration = end_time - changepoints[-1]
#                     else:
#                         duration = changepoints[epoch_idx] - changepoints[epoch_idx - 1]
#                     epoch_durations.append(duration)

#                 max_epoch_duration = max(np.array(epoch_durations).flatten())

#                 # Interpolate all trials for this epoch to the longest duration
#                 interpolated_trials = {dim: [] for dim in latent_dims}

#                 for trial_idx in range(changepoints.shape[0]):
#                     trial_start = epoch_start
#                     trial_end = epoch_end

#                     trial_mask = (taste_df['time'] >= trial_start) & (taste_df['time'] <= trial_end)
#                     trial_df = taste_df.filter(trial_mask)

#                     for dim in latent_dims:
#                         trial_latent = trial_df[dim].to_numpy()
#                         if len(trial_latent) > 0:
#                             f_interp = interp1d(
#                                 np.linspace(0, 1, len(trial_latent)),
#                                 trial_latent,
#                                 kind='nearest',
#                                 fill_value="extrapolate"
#                             )
#                             interpolated_trials[dim].append(f_interp(np.linspace(0, 1, max_epoch_duration)))

#                 # Plot each latent dimension individually
#                 for dim_idx, dim in enumerate(latent_dims):
#                     fig, ax = plt.subplots(figsize=(10, 6))

#                     # Plot all trials in light gray
#                     for trial_latent in interpolated_trials[dim]:
#                         ax.plot(
#                             np.linspace(epoch_start, epoch_end, max_epoch_duration),
#                             trial_latent,
#                             color='gray',
#                             alpha=0.5,
#                             linewidth=0.5
#                         )

#                     # Compute and plot the average latent dimension
#                     avg_latent = np.mean(interpolated_trials[dim], axis=0)
#                     ax.plot(
#                         np.linspace(epoch_start, epoch_end, max_epoch_duration),
#                         avg_latent,
#                         color='blue',
#                         linewidth=2,
#                         label=f'Avg Latent Dim {dim_idx + 1}'
#                     )

#                     # Set plot title and labels
#                     ax.set_title(f"Taste: {taste}, Epoch: {epoch_idx + 1}, Latent: {dim}")
#                     ax.set_xlabel("Time (ms)")
#                     ax.set_ylabel("Latent Value")
#                     ax.legend()

#                     # Save the figure
#                     output_file = os.path.join(
#                         df_output_dir, f"taste_{taste}_epoch_{epoch_idx + 1}_latent_{dim_idx + 1}.png"
#                     )
#                     plt.tight_layout()
#                     plt.savefig(output_file)
#                     print(f"Saved plot: {output_file}")
#                     plt.close(fig)


# # plotting latant graphs-- RNN
# # also want to plot these data but instead of the lines, plot the variance
# def plot_thin_thick_rnn_latants(
#     epoch_dataframes_dict, standardized_changepoints_dict, output_dir, start_time=None, end_time=None
# ):
#     """
#     Plots RNN latent dimensions and averages across all trials for each taste.

#     Parameters:
#     - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
#     - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
#     - output_dir: Directory to save the output plots.
#     - start_time: Manual start time for plotting (ms).
#     - end_time: Manual end time for plotting (ms).
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     for df_name, df in epoch_dataframes_dict.items():
#         # Extract the core name from df_name to match with standardized_changepoints_dict
#         core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]

#         if core_dataset_name not in standardized_changepoints_dict:
#             print(f"Changepoints not found for {core_dataset_name}. Skipping...")
#             continue

#         # Get the changepoint data for the dataset
#         changepoints = standardized_changepoints_dict[core_dataset_name]

#         # Create a subdirectory for the current DataFrame
#         df_output_dir = os.path.join(output_dir, df_name)
#         os.makedirs(df_output_dir, exist_ok=True)

#         # Extract unique tastes
#         unique_tastes = df['taste'].unique().to_list()

#         for taste_idx, taste in enumerate(unique_tastes):
#             # Filter for the current taste
#             taste_df = df.filter(pl.col('taste') == taste)

#             # Prepare two figures for each taste (4 latent dimensions per figure)
#             for fig_num, latent_dims in enumerate([range(4), range(4, 8)]):
#                 fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#                 axs = axs.flatten()
#                 fig.suptitle(f"Averaged RNN latents for taste: {modified_tastes[taste_idx]} - {df_name}", fontsize=16)

#                 for ax, latent_idx in zip(axs, latent_dims):
#                     time_values = taste_df['time'].to_numpy()
#                     latent_values = taste_df[f'latent_dim_{latent_idx}'].to_numpy()

#                     # Apply time filtering
#                     if start_time is not None:
#                         time_mask = (time_values >= start_time) & (time_values <= end_time)
#                         time_values = time_values[time_mask]
#                     else:
#                         time_mask = np.ones_like(time_values, dtype=bool)

#                     # Split latent values by trial
#                     latent_trials = []
#                     for trial in taste_df['trial'].unique():
#                         trial_mask = (taste_df['trial'] == trial).to_numpy() & time_mask
#                         latent_trial = latent_values[trial_mask]
#                         if len(latent_trial) > 0:
#                             latent_trials.append(latent_trial)

#                     for trial_latent in latent_trials:
#                         ax.plot(time_values[:len(trial_latent)], trial_latent, color='gray', alpha=0.5, linewidth=0.5)

#                     # Compute and plot the average latent dimension over all trials
#                     avg_latent = np.mean(np.stack(latent_trials), axis=0)
#                     ax.plot(time_values[:len(avg_latent)], avg_latent, color='blue', linewidth=2, label=f'Avg Latent Dim {latent_idx+1}')

#                     # Add stimulus delivery line
#                     if start_time <= 2000 <= end_time:
#                         ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

#                     # Set plot title and labels
#                     ax.set_title(f"Latent Dimension {latent_idx+1}")
#                     ax.set_xlabel("Time (ms)")
#                     ax.set_ylabel("Latent Value")

#                 # Add a shared legend outside the subplots
#                 fig.legend(loc="lower center", ncol=5, fontsize='small', frameon=False)

#                 # Adjust layout and save the figure
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#                 output_file = os.path.join(df_output_dir, f"averaged_rnn_plots_taste_{taste}_fig_{fig_num + 1}.png")
#                 plt.savefig(output_file)
#                 print(f"Saved averaged RNN plot figure: {output_file}")

#                 plt.close(fig)
# # this but variance:
# thin_thick_output = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_Thin_Thick_Var'
# def plot_thin_thick_rnn_latants(
#     epoch_dataframes_dict, standardized_changepoints_dict, output_dir, start_time=None, end_time=None
# ):
#     """
#     Plots the variance of RNN latent dimensions across all trials for each taste.

#     Parameters:
#     - epoch_dataframes_dict: Dictionary of DataFrames for each dataset.
#     - standardized_changepoints_dict: Dictionary of changepoints for each dataset.
#     - output_dir: Directory to save the output plots.
#     - start_time: Manual start time for plotting (ms).
#     - end_time: Manual end time for plotting (ms).
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     for df_name, df in epoch_dataframes_dict.items():
#         # Extract the core name from df_name to match with standardized_changepoints_dict
#         core_dataset_name = df_name.split('_repacked_raw_latent_vectors')[0]

#         if core_dataset_name not in standardized_changepoints_dict:
#             print(f"Changepoints not found for {core_dataset_name}. Skipping...")
#             continue

#         # Get the changepoint data for the dataset
#         changepoints = standardized_changepoints_dict[core_dataset_name]

#         # Create a subdirectory for the current DataFrame
#         df_output_dir = os.path.join(output_dir, df_name)
#         os.makedirs(df_output_dir, exist_ok=True)

#         # Extract unique tastes
#         unique_tastes = df['taste'].unique().to_list()

#         for taste_idx, taste in enumerate(unique_tastes):
#             # Filter for the current taste
#             taste_df = df.filter(pl.col('taste') == taste)

#             # Prepare two figures for each taste (4 latent dimensions per figure)
#             for fig_num, latent_dims in enumerate([range(4), range(4, 8)]):
#                 fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#                 axs = axs.flatten()
#                 fig.suptitle(f"Variance of RNN latents for taste: {modified_tastes[taste_idx]} - {df_name}", fontsize=16)

#                 for ax, latent_idx in zip(axs, latent_dims):
#                     time_values = taste_df['time'].to_numpy()
#                     latent_values = taste_df[f'latent_dim_{latent_idx}'].to_numpy()

#                     # Apply time filtering
#                     if start_time is not None:
#                         time_mask = (time_values >= start_time) & (time_values <= end_time)
#                         time_values = time_values[time_mask]
#                     else:
#                         time_mask = np.ones_like(time_values, dtype=bool)

#                     # Split latent values by trial
#                     latent_trials = []
#                     for trial in taste_df['trial'].unique():
#                         trial_mask = (taste_df['trial'] == trial).to_numpy() & time_mask
#                         latent_trial = latent_values[trial_mask]
#                         if len(latent_trial) > 0:
#                             latent_trials.append(latent_trial)

#                     # Compute and plot the variance of latent dimensions over all trials
#                     if latent_trials:
#                         stacked_trials = np.stack(latent_trials)
#                         variance_latent = np.var(stacked_trials, axis=0)
#                         ax.plot(time_values[:len(variance_latent)], variance_latent, color='red', linewidth=2, label=f'Variance Latent Dim {latent_idx+1}')

#                     # Add stimulus delivery line
#                     if start_time <= 2000 <= end_time:
#                         ax.axvline(2000, color='black', linestyle=':', linewidth=2, label='Stimulus Delivery')

#                     # Set plot title and labels
#                     ax.set_title(f"Variance of Latent Dimension {latent_idx+1}")
#                     ax.set_xlabel("Time (ms)")
#                     ax.set_ylabel("Variance")

#                 # Add a shared legend outside the subplots
#                 fig.legend(loc="lower center", ncol=5, fontsize='small', frameon=False)

#                 # Adjust layout and save the figure
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#                 output_file = os.path.join(df_output_dir, f"variance_rnn_plots_taste_{taste}_fig_{fig_num + 1}.png")
#                 plt.savefig(output_file)
#                 print(f"Saved variance RNN plot figure: {output_file}")

#                 plt.close(fig)
