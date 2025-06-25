#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:36:40 2025

Backup of RNNPRCOESSOR -- workign code

@author: vincentcalia-bogan
"""


import numpy as np
import polars as pl
from pathlib import Path
from sklearn.decomposition import PCA

from extract_npz import extract_from_npz
from unpkl_generator import extract_valid_changepoints
from read_parquets import read_parquet_files_into_dict, all_nrns_to_df
from find_extract_info import find_copy_h5info, process_info_files, modify_tastes


class RNNLatentProcessor:
    def __init__(
        self,
        parquet_dir,
        npz_path,
        info_path,
        pkl_path,
        taste_replacements,
        bin_size_ms=25,
        start_time_ms=1500,
        max_time_ms=4500,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.npz_path = npz_path
        self.info_path = info_path
        self.pkl_path = pkl_path
        self.taste_replacements = taste_replacements
        self.bin_size_ms = bin_size_ms
        self.start_time_ms = start_time_ms
        self.max_time_ms = max_time_ms

        self.taste_latent = {}
        self.processed_data = {}
        self.changepoints_dict = {}
        self.epoch_dataframes_dict = {}

    def read_parquet_files(self):
        self.taste_latent = read_parquet_files_into_dict(self.parquet_dir)

    def structure_latent_arrays(self):
        # refactored slightly for speed
        for dataset_name, df in self.taste_latent.items():
            unique_tastes = df["taste"].unique()
            unique_trials = df["trial"].unique()
            latent_columns = [
                col for col in df.columns if col.startswith("latent_dim_")
            ]
            latent_dim = len(latent_columns)

            # Find number of time steps
            num_time_steps = len(
                df.filter(
                    (pl.col("trial") == unique_trials[0])
                    & (pl.col("taste") == unique_tastes[0])
                )
            )

            # Dataset array: (taste, trial, latent_dim, time)
            dataset_array = np.empty(
                (len(unique_tastes), len(unique_trials), latent_dim, num_time_steps)
            )

            # --- Pre-group by (taste, trial) once for speed
            trial_lookup = {}
            for taste in unique_tastes:
                for trial in unique_trials:
                    key = (taste, trial)
                    trial_data = df.filter(
                        (pl.col("taste") == taste) & (pl.col("trial") == trial)
                    )
                    if trial_data.height > 0:
                        trial_lookup[key] = (
                            trial_data.select(latent_columns).to_numpy().T
                        )  # (latent_dim, time)

            # --- Now fast loop
            for taste_idx, taste in enumerate(unique_tastes):
                for trial_idx, trial in enumerate(unique_trials):
                    key = (taste, trial)
                    if key in trial_lookup:
                        dataset_array[taste_idx, trial_idx, :, :] = trial_lookup[key]
                    else:
                        print(f"Warning: Missing data for taste {taste} trial {trial}")

            self.processed_data[dataset_name] = dataset_array

    # original, non-refactored version:
    # def structure_latent_arrays(self):
    # for dataset_name, df in self.taste_latent.items():
    #     unique_tastes = df['taste'].unique()
    #     unique_trials = df['trial'].unique()
    #     latent_columns = [col for col in df.columns if col.startswith("latent_dim_")]
    #     latent_dim = len(latent_columns)

    #     # Find number of time steps
    #     num_time_steps = len(
    #         df.filter((pl.col("trial") == unique_trials[0]) & (pl.col("taste") == unique_tastes[0]))
    #     )

    #     # Dataset array: (taste, trial, latent_dim, time)
    #     dataset_array = np.empty((len(unique_tastes), len(unique_trials), latent_dim, num_time_steps))

    #     for taste_idx, taste in enumerate(unique_tastes):
    #         for trial_idx, trial in enumerate(unique_trials):
    #             trial_data = df.filter((pl.col("taste") == taste) & (pl.col("trial") == trial))
    #             latent_values = trial_data.select(latent_columns).to_numpy().T  # (latent_dim, time)
    #             dataset_array[taste_idx, trial_idx, :, :] = latent_values

    #     self.processed_data[dataset_name] = dataset_array

    def extract_changepoints(self):
        for data in extract_from_npz(self.npz_path):
            if isinstance(data, tuple):
                spike_array, dataset_num, index, key = data
                dataset_name = f"dataset_{dataset_num}"
                dataset_tastes = process_info_files(self.info_path, dataset_num)
                modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
                extracted_pkl = extract_valid_changepoints(
                    self.pkl_path, spike_array, dataset_num, index, key
                )

                if extracted_pkl is not None:
                    try:
                        changepoints = extracted_pkl[
                            :, 3
                        ]  # Keep full array, not just one column
                        self.changepoints_dict[dataset_name] = changepoints
                    except IndexError:
                        print(
                            f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
                        )
                else:
                    print(f"No valid changepoints found for dataset {dataset_name}")

    def split_epochs_by_changepoints(self):
        standardized_changepoints_dict = {
            key.replace("dataset_", "").split("_repacked.npz")[0]: value
            for key, value in self.changepoints_dict.items()
        }

        for dataset_name, data_array in self.processed_data.items():
            core_dataset_name = dataset_name.split("_repacked_raw_latent_vectors")[0]

            changepoints = standardized_changepoints_dict.get(core_dataset_name, None)
            if changepoints is None:
                print(f"No changepoints for {core_dataset_name}, skipping.")
                continue

            num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape
            all_epochs = []
            time_in_ms = [
                (t * self.bin_size_ms) + self.start_time_ms
                for t in range(num_time_steps)
            ]

            if len(changepoints) != num_tastes:
                print(f"Mismatch in taste dimension for dataset {core_dataset_name}")
                continue

            for taste_idx in range(num_tastes):
                if len(changepoints[taste_idx]) < num_trials:
                    print(
                        f"Mismatch in trial dimension for dataset {core_dataset_name}, taste {taste_idx}"
                    )
                    continue

                for trial_idx in range(num_trials):
                    trial_changepoints = changepoints[taste_idx][trial_idx]
                    start_idx = 0

                    for cp_idx, changepoint_ms in enumerate(trial_changepoints):
                        end_idx = next(
                            (
                                i
                                for i, t in enumerate(time_in_ms)
                                if t >= changepoint_ms
                            ),
                            num_time_steps,
                        )

                        epoch_data = data_array[
                            taste_idx, start_idx:end_idx, :, trial_idx
                        ]  # (time, latent)

                        epoch_df = pl.DataFrame(
                            epoch_data,
                            schema=[f"latent_dim_{i}" for i in range(latent_dim)],
                        )
                        epoch_df = epoch_df.with_columns(
                            [
                                pl.Series("taste", [taste_idx] * len(epoch_df)),
                                pl.Series("trial", [trial_idx] * len(epoch_df)),
                                pl.Series("changepoint", [cp_idx] * len(epoch_df)),
                                pl.Series("time", time_in_ms[start_idx:end_idx]),
                            ]
                        )
                        all_epochs.append(epoch_df)
                        start_idx = end_idx

                    if start_idx < num_time_steps:
                        epoch_data = data_array[
                            taste_idx, start_idx:num_time_steps, :, trial_idx
                        ]
                        epoch_df = pl.DataFrame(
                            epoch_data,
                            schema=[f"latent_dim_{i}" for i in range(latent_dim)],
                        )
                        epoch_df = epoch_df.with_columns(
                            [
                                pl.Series("taste", [taste_idx] * len(epoch_df)),
                                pl.Series("trial", [trial_idx] * len(epoch_df)),
                                pl.Series(
                                    "changepoint",
                                    [len(trial_changepoints)] * len(epoch_df),
                                ),
                                pl.Series("time", time_in_ms[start_idx:num_time_steps]),
                            ]
                        )
                        all_epochs.append(epoch_df)

            if all_epochs:
                self.epoch_dataframes_dict[dataset_name] = pl.concat(all_epochs)
            else:
                print(f"No valid epochs for dataset {dataset_name}, skipping.")

    # now also a func that runs PCA on the RNN latents in one place so I can process from there
    def run_robust_pca_analysis(
        self,
        return_dicts=True,
        variance_threshold=95.0,
        compute_first_derivative=False,
        compute_second_derivative=False,
        derivative_source="threshold",
    ):
        """
        Perform PCA on concatenated data grouped by taste and changepoint, then split results back accordingly.
        Optionally computes first and second derivatives.

        Parameters
        ----------
        return_dicts : bool, optional
            Whether to return the PCA dictionaries.
        variance_threshold : float, optional
            The cumulative explained variance percentage to retain.
        compute_first_derivative : bool, optional
            Whether to compute and store first derivatives.
        compute_second_derivative : bool, optional
            Whether to compute and store second derivatives.
        derivative_source : str, optional
            "threshold" (default) to compute derivatives on robust_pca_<threshold>;
            "full" to compute derivatives on robust_pca_full.

        Returns
        -------
        (dict, dict) or None
            (robust_pca_threshold, robust_pca_full) if return_dicts=True, else None
        """
        pca_lat_dict_thresh = {}
        pca_lat_dict_full = {}

        for dataset_name, df in self.epoch_dataframes_dict.items():
            unique_tastes = df["taste"].unique().to_list()
            unique_changepoints = df["changepoint"].unique().to_list()

            processed_dataframes_thresh = []
            processed_dataframes_full = []

            for taste in unique_tastes:
                for changepoint in unique_changepoints:
                    filtered_df = df.filter(
                        (pl.col("taste") == taste)
                        & (pl.col("changepoint") == changepoint)
                    )

                    if filtered_df.is_empty():
                        continue

                    latent_columns = [
                        col for col in filtered_df.columns if "latent_dim_" in col
                    ]
                    if not latent_columns:
                        continue

                    unique_trials = filtered_df["trial"].unique().to_list()
                    trial_data_list = [
                        filtered_df.filter(pl.col("trial") == trial)
                        .select(latent_columns)
                        .to_numpy()
                        for trial in unique_trials
                    ]

                    concatenated_data = np.vstack(trial_data_list)

                    if concatenated_data.shape[0] < 2:
                        continue

                    pca = PCA()
                    transformed_data = pca.fit_transform(concatenated_data)
                    explained_variance = pca.explained_variance_ratio_ * 100

                    cumulative_variance = np.cumsum(explained_variance)
                    num_pcs_thresh = (
                        np.argmax(cumulative_variance > variance_threshold) + 1
                    )

                    split_indices = np.cumsum(
                        [arr.shape[0] for arr in trial_data_list]
                    )[:-1]
                    split_pca_data = np.split(transformed_data, split_indices)

                    pc_names_full = [
                        f"PC_{i+1}" for i in range(transformed_data.shape[1])
                    ]
                    explained_var_cols_full = [
                        f"explained_variance_{pc}" for pc in pc_names_full
                    ]

                    pc_names_thresh = [f"PC_{i+1}" for i in range(num_pcs_thresh)]
                    explained_var_cols_thresh = [
                        f"explained_variance_{pc}" for pc in pc_names_thresh
                    ]

                    for trial, trial_pca_data in zip(unique_trials, split_pca_data):
                        trial_metadata = filtered_df.filter(
                            pl.col("trial") == trial
                        ).select(["trial", "time"])

                        full_pca_df = pl.DataFrame(
                            np.hstack(
                                [
                                    trial_pca_data,
                                    np.tile(
                                        explained_variance, (len(trial_pca_data), 1)
                                    ),
                                ]
                            ),
                            schema=pc_names_full + explained_var_cols_full,
                        ).with_columns(
                            [
                                pl.lit(taste).alias("taste"),
                                pl.lit(changepoint).alias("changepoint"),
                            ]
                        )
                        full_pca_df = pl.concat(
                            [trial_metadata, full_pca_df], how="horizontal"
                        )
                        processed_dataframes_full.append(full_pca_df)

                        pca_df_thresh = pl.DataFrame(
                            np.hstack(
                                [
                                    trial_pca_data[:, :num_pcs_thresh],
                                    np.tile(
                                        explained_variance[:num_pcs_thresh],
                                        (len(trial_pca_data), 1),
                                    ),
                                ]
                            ),
                            schema=pc_names_thresh + explained_var_cols_thresh,
                        ).with_columns(
                            [
                                pl.lit(taste).alias("taste"),
                                pl.lit(changepoint).alias("changepoint"),
                            ]
                        )
                        pca_df_thresh = pl.concat(
                            [trial_metadata, pca_df_thresh], how="horizontal"
                        )
                        processed_dataframes_thresh.append(pca_df_thresh)

            def align_dataframes(dataframes):
                all_columns = set()
                for df in dataframes:
                    all_columns.update(df.columns)
                all_columns = sorted(all_columns)

                aligned_dfs = []
                for df in dataframes:
                    missing_columns = set(all_columns) - set(df.columns)
                    for col in missing_columns:
                        df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
                    df = df.select(all_columns)
                    aligned_dfs.append(df)
                return aligned_dfs

            if processed_dataframes_thresh:
                aligned_thresh = align_dataframes(processed_dataframes_thresh)
                pca_lat_dict_thresh[dataset_name] = pl.concat(aligned_thresh)
            if processed_dataframes_full:
                aligned_full = align_dataframes(processed_dataframes_full)
                pca_lat_dict_full[dataset_name] = pl.concat(aligned_full)

        setattr(self, f"robust_pca_{int(variance_threshold)}", pca_lat_dict_thresh)
        self.robust_pca_full = pca_lat_dict_full

        # Derivative source selector
        if derivative_source == "threshold":
            source_dict = pca_lat_dict_thresh
            source_name = str(int(variance_threshold))
        elif derivative_source == "full":
            source_dict = pca_lat_dict_full
            source_name = "full"
        else:
            raise ValueError(
                "Invalid derivative_source. Must be 'threshold' or 'full'."
            )

        # Compute derivatives if requested
        if compute_first_derivative:
            first_derivative_dict = {}
            for key, df in source_dict.items():
                first_derivative_df = self.compute_first_derivative(df)
                first_derivative_dict[key] = first_derivative_df
            setattr(self, f"first_derivatives_{source_name}", first_derivative_dict)

        if compute_second_derivative:
            second_derivative_dict = {}
            for key, df in source_dict.items():
                second_derivative_df = self.compute_second_derivative(df)
                second_derivative_dict[key] = second_derivative_df
            setattr(self, f"second_derivatives_{source_name}", second_derivative_dict)

        if return_dicts:
            return pca_lat_dict_thresh, pca_lat_dict_full
        else:
            return None

    def compute_first_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the first derivative of latent dimension columns in a Polars DataFrame.
        Assumes uniform time spacing (ignores actual 'time' values).
        """
        meta_cols = ["taste", "trial", "changepoint", "time"]
        latent_cols = [col for col in df.columns if col not in meta_cols]

        data = df.select(latent_cols).to_numpy()
        first_deriv = np.gradient(data, axis=0)

        result_df = df.with_columns(
            [
                pl.Series(name=col, values=first_deriv[:, j])
                for j, col in enumerate(latent_cols)
            ]
        )
        return result_df

    def compute_second_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the second derivative of latent dimension columns in a Polars DataFrame.
        Assumes uniform time spacing (ignores actual 'time' values).
        """
        meta_cols = ["taste", "trial", "changepoint", "time"]
        latent_cols = [col for col in df.columns if col not in meta_cols]

        data = df.select(latent_cols).to_numpy()
        first_deriv = np.gradient(data, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)

        result_df = df.with_columns(
            [
                pl.Series(name=col, values=second_deriv[:, j])
                for j, col in enumerate(latent_cols)
            ]
        )
        return result_df

    def full_pipeline(
        self,
        variance_threshold=95.0,
        compute_first_derivative=False,
        compute_second_derivative=False,
        derivative_source="threshold",
        return_derivatives=True,
    ):
        """
        Full one-liner pipeline:
        - Read parquet files
        - Structure latent arrays
        - Extract changepoints
        - Split into epochs
        - Run PCA
        - Optionally compute derivatives
        - Returns everything needed downstream.

        Parameters
        ----------
        variance_threshold : float
            Threshold for PCA explained variance (default 95).
        compute_first_derivative : bool
            Whether to compute first derivatives.
        compute_second_derivative : bool
            Whether to compute second derivatives.
        derivative_source : str
            "threshold" or "full" PCA to compute derivatives from.
        return_derivatives : bool
            Whether to return derivative outputs separately.

        Returns
        -------
        tuple
            epoch_dataframes_dict, robust_pca_threshold, robust_pca_full
            (plus first_derivatives, second_derivatives if return_derivatives=True)
        """

        # --- Step 1: Full pre-processing
        self.read_parquet_files()
        self.structure_latent_arrays()
        self.extract_changepoints()
        self.split_epochs_by_changepoints()

        # --- Step 2: Run PCA
        pca_thresh, pca_full = self.run_robust_pca_analysis(
            return_dicts=True, variance_threshold=variance_threshold
        )

        # --- Step 3: Compute derivatives if needed
        first_derivatives = None
        second_derivatives = None

        if compute_first_derivative or compute_second_derivative:
            if derivative_source == "threshold":
                source_dict = pca_thresh
            elif derivative_source == "full":
                source_dict = pca_full
            else:
                raise ValueError(
                    "Invalid derivative_source. Must be 'threshold' or 'full'."
                )

        if compute_first_derivative:
            first_derivatives = {}
            for key, df in source_dict.items():
                first_derivatives[key] = self.compute_first_derivative(df)

        if compute_second_derivative:
            second_derivatives = {}
            for key, df in source_dict.items():
                second_derivatives[key] = self.compute_second_derivative(df)

        # --- Step 4: Return everything
        if return_derivatives:
            return (
                self.epoch_dataframes_dict,
                pca_thresh,
                pca_full,
                first_derivatives,
                second_derivatives,
            )
        else:
            return self.epoch_dataframes_dict, pca_thresh, pca_full

    def save_analysis_outputs(self, tld):
        """
        Saves all analysis outputs (epoch_dataframes_dict, robust_pca_95, robust_pca_full,
        first_derivs, second_derivs) into organized subdirectories as Parquet files.

        Parameters
        ----------
        tld : str or Path
            Top-level directory to save outputs into.
        """
        tld = Path(tld)
        tld.mkdir(parents=True, exist_ok=True)  # Ensure top-level directory exists

        # Dictionary of analysis results to save
        analysis_dicts = {
            "epoch_dataframes_dict": self.epoch_dataframes_dict,
            "robust_pca_95": getattr(self, "robust_pca_95", {}),
            "robust_pca_full": getattr(self, "robust_pca_full", {}),
            "first_derivs": getattr(self, "first_derivs", {}),
            "second_derivs": getattr(self, "second_derivs", {}),
        }

        for analysis_name, analysis_data in analysis_dicts.items():
            # Create subdirectory for this analysis type
            analysis_dir = tld / analysis_name
            analysis_dir.mkdir(parents=True, exist_ok=True)

            if not analysis_data:
                print(f"[WARNING] No data found for {analysis_name}, skipping.")
                continue

            for dataset_name, df in analysis_data.items():
                if isinstance(df, pl.DataFrame):
                    save_path = analysis_dir / f"{dataset_name}_{analysis_name}.parquet"
                    df.write_parquet(save_path)
                    print(f"Saved {dataset_name} [{analysis_name}] to {save_path}")
                else:
                    print(
                        f"[WARNING] Skipping {dataset_name} for {analysis_name} — not a Polars DataFrame."
                    )
