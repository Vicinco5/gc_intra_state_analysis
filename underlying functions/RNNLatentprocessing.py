#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:48:23 2025

New: class and funcs that I can sue to import the RNN latent data from my exernal drive

*** WIP THAT DOESN'T CURRENTLY WORK! ***

Given that this is not mission critical for the thesis, leaving this on ice for the moment.

@author: vincentcalia-bogan



HOW TO USE THE OUTPUT OF THIS CLASS:
# --------------------------------------------------------------------------------
# 1. Import necessary modules (assuming your class file is imported)
# --------------------------------------------------------------------------------


from RNNLatentprocessing import RNNLatentProcessor
# Also make sure read_parquet_files_into_dict, extract_from_npz, process_info_files, modify_tastes, extract_valid_changepoints are available/imported.

# Addiitonal files you will need:
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path  # if not already imported

# --------------------------------------------------------------------------------
# 2. Define file paths
# --------------------------------------------------------------------------------

parquet_dir = '/path/to/your/parquet_folder'   # Folder containing your parquet latent files
npz_path = '/path/to/your/npz_file.npz'         # npz file path
info_path = '/path/to/your/info_file.pkl'       # info file path (for taste mapping)
pkl_path = '/path/to/your/pkl_file.pkl'         # pkl file path (for changepoints)
taste_replacements = {'original_taste1': 'new_name1', 'original_taste2': 'new_name2'}  # Customize replacements

# --------------------------------------------------------------------------------
# 3. Initialize the processor
# --------------------------------------------------------------------------------

processor = RNNLatentProcessor(
    parquet_dir=parquet_dir,
    npz_path=npz_path,
    info_path=info_path,
    pkl_path=pkl_path,
    taste_replacements=taste_replacements,
    bin_size_ms=25,
    start_time_ms=1500,
    max_time_ms=4500
)

# --------------------------------------------------------------------------------
# 4. Run processing steps
# --------------------------------------------------------------------------------

processor.read_parquet_files()          # Load parquet files into memory
processor.structure_latent_arrays()     # Build latent arrays (taste × trial × latent × time)
processor.extract_changepoints()        # Extract changepoints from NPZ/PKL files
processor.split_epochs_by_changepoints()# Split latent arrays into epochs based on changepoints

# Alternatively, there's a helper function in here that allows this all to be called more succintly
# it can be done as follows:

epoch_dataframes_dict, robust_pca_95, robust_pca_full = processor.full_pipeline(
    variance_threshold=95.0,
    compute_first_derivative=False,
    compute_second_derivative=False,
    derivative_source="threshold",
    return_derivatives=False
)

# Optional: if you want derivatives too
# epoch_dataframes_dict, robust_pca_95, robust_pca_full, first_derivs, second_derivs = processor.full_pipeline(
#     variance_threshold=95.0,
#     compute_first_derivative=True,
#     compute_second_derivative=True,
#     derivative_source="threshold",
#     return_derivatives=True
# )


# Now your data is ready!

# --------------------------------------------------------------------------------
# 5. Access structured outputs
# --------------------------------------------------------------------------------

# processed_data contains raw latent arrays
latent_arrays = processor.processed_data  # Dict: {dataset_name: np.ndarray of (taste, trial, latent, time)}

# epoch_dataframes_dict contains Polars DataFrames split by taste/trial/epoch ;this is the main datatype used for RNN output.
epoch_dataframes_dict = processor.epoch_dataframes_dict  # Dict: {dataset_name: Polars DataFrame}

# changepoints_dict contains raw changepoints
changepoints_dict = processor.changepoints_dict  # Dict: {dataset_name: np.ndarray}

# if running PCA on the RNN latents
pca_thresholded = processor.robust_pca_95                # PCA thresholded (95%)
pca_full = processor.robust_pca_full                     # Full PCA (all components)

# If computed: First and Second diff of the PCA latents:
# first_derivatives = processor.first_derivatives_95 or processor.first_derivatives_full
# second_derivatives = processor.second_derivatives_95 or processor.second_derivatives_full


# --------------------------------------------------------------------------------
# 6. Helper: Function to extract PC or latent_dim columns
# --------------------------------------------------------------------------------

def get_data_columns(df: pl.DataFrame):

   # Return PC_x or latent_dim_x columns from a Polars DataFrame.
   # Useful for looping across dimensions.

    return [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]

# --------------------------------------------------------------------------------
# 7. Example: Looping through datasets and columns
# --------------------------------------------------------------------------------

data_object = robust_pca_95  # or epoch_dataframes_dict, or first_derivs, or second_derivs

for dataset_name, df in data_object.items():
    print(f"Dataset: {dataset_name}")
    data_cols = get_data_columns(df)

    for col in data_cols:
        values = df[col].to_numpy()
        times = df['time'].to_numpy()

        plt.plot(times, values, label=col, alpha=0.7)

    plt.title(f"{dataset_name} - PCs or Latents")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# --------------------------------------------------------------------------------
# 8. Example: Peek and subset specific epochs
# --------------------------------------------------------------------------------

# Choose dataset
dataset_name = list(epoch_dataframes_dict.keys())[0]
epochs_df = epoch_dataframes_dict[dataset_name]

# Show top rows
print(epochs_df.head())

# Select all rows for taste 2, changepoint 1
taste_idx = 2
changepoint_idx = 1

subset_df = epochs_df.filter(
    (pl.col('taste') == taste_idx) & (pl.col('changepoint') == changepoint_idx)
)
print(subset_df)

# --------------------------------------------------------------------------------
# 9. Example: Visualize a single trial
# --------------------------------------------------------------------------------

# Pick a trial within that subset
trial_idx = 5

trial_df = epochs_df.filter(
    (pl.col('taste') == taste_idx) & (pl.col('trial') == trial_idx)
)

# Plot latent_dim_0 or PC_1
plt.plot(trial_df['time'], trial_df['latent_dim_0'])  # If using PCA, change to 'PC_1'
plt.title(f"Taste {taste_idx} — Trial {trial_idx}")
plt.xlabel("Time (ms)")
plt.ylabel("Latent Dimension Value")
plt.grid()
plt.show()

# --------------------------------------------------------------------------------
# 10. Notes
# --------------------------------------------------------------------------------
# - Latent dimensions are aligned per trial and taste.
# - Changepoint segmentation is respected during split.
# - DataFrames can be grouped easily by taste, trial, changepoint.
# - PCA reduces dimensionality and decorrelates features.
# - Derivatives (if computed) provide velocity or acceleration features over time.
# - get_data_columns() automatically detects PCs or latents without hardcoding.

"""

import numpy as np
import polars as pl
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import pickle

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
        save_dir=None,
        warp_length=None,           # warp duration in ms
        variance_threshold=95.0,    # PCA variance threshold
    ):
        self.parquet_dir = Path(parquet_dir)
        self.npz_path = npz_path
        self.info_path = info_path
        self.pkl_path = pkl_path
        self.taste_replacements = taste_replacements
        self.bin_size_ms = bin_size_ms
        self.start_time_ms = start_time_ms
        self.max_time_ms = max_time_ms
        self.warp_length = warp_length
        self.variance_threshold = variance_threshold

        self.save_dir = Path(save_dir) if save_dir else None
        self.taste_latent = {}
        self.processed_data = {}
        self.changepoints_dict = {}

        # unwarped outputs
        self.epoch_dataframes_dict_unwarped = {}
        self.robust_pca_thresh_unwarped = {}
        self.robust_pca_full_unwarped = {}
        self.first_derivs_thresh_unwarped = {}
        self.second_derivs_thresh_unwarped = {}

        # warped outputs
        self.epoch_dataframes_dict_warped = {}
        self.robust_pca_thresh_warped = {}
        self.robust_pca_full_warped = {}
        self.first_derivs_thresh_warped = {}
        self.second_derivs_thresh_warped = {}

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
                        ]  
                        self.changepoints_dict[dataset_name] = changepoints
                    except IndexError:
                        print(
                            f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
                        )
                else:
                    print(f"No valid changepoints found for dataset {dataset_name}")
    # method that saves changepoints so the entire processor doesn't need to be run again and again 
    def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
        """
        Extract only the changepoints for each dataset (populates self.changepoints_dict)
        without running PCA or derivative steps.
        Optionally saves them to disk as 'changepoints_dict.pkl' under self.save_dir.
        """
        # (re)initialize
        self.changepoints_dict = {}
        # delegate to original: 
        self.extract_changepoints()
        if save_outputs:
            save_path = Path(self.save_dir) / "changepoints"
            save_path.mkdir(parents=True, exist_ok=True)
            # optionally persist for future runs
            for dataset_name, changepoints in self.changepoints_dict.items():
                clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
                out_path = save_path / f"{clean_name}_changepoints.pkl"
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        pickle.dump(self.changepoints_dict, f)

        return self.changepoints_dict

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
                self.epoch_dataframes_dict_unwarped[
                    f"{core_dataset_name}_latent_unwarped"
                ] = pl.concat(all_epochs)
            else:
                print(f"No valid epochs for dataset {dataset_name}, skipping.")
                
    # warping helper 
    # def _warp_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
    #     """
    #     Time-warp a Polars DataFrame by interpolating all non-meta columns
    #     across fixed bins determined by warp_length.
    #     """
    #     meta_cols = ['taste', 'trial', 'changepoint', 'time']
    #     data_cols = [c for c in df.columns if c not in meta_cols]
    #     arr = df.select(data_cols).to_numpy().T  # shape (features, time)
    #     orig_axes = np.linspace(0, 1, arr.shape[1])
    #     nbins = max((self.warp_length - self.bin_size_ms) // self.bin_size_ms + 1, 1)
    #     target_axes = np.linspace(0, 1, nbins)
    #     warped_arr = np.vstack([
    #         interp1d(orig_axes, arr[i], kind='nearest',
    #                  bounds_error=False, fill_value='extrapolate')(target_axes)
    #         for i in range(arr.shape[0])
    #     ])
    #     # rebuild DataFrame
    #     warped_df = pl.DataFrame({
    #         data_cols[i]: warped_arr[i].tolist() for i in range(warped_arr.shape[0])
    #     })
    #     start_time = df['time'][0]
    #     times = [start_time + b * self.bin_size_ms for b in range(nbins)]
    #     warped_df = warped_df.with_columns([
    #         pl.Series('taste', [df['taste'][0]] * nbins),
    #         pl.Series('trial', [df['trial'][0]] * nbins),
    #         pl.Series('changepoint', [df['changepoint'][0]] * nbins),
    #         pl.Series('time', times)
    #     ])
    #     return warped_df

            
    def warp_all_outputs(self):
        """
        Time-warp each unwarped DataFrame (epochs, PCA, derivatives)
        on a per-taste, per-trial, per-epoch basis using actual ms boundaries
        and rescaling to warp_length (ms) with binning by bin_size_ms.
        """
        if not self.warp_length:
            return

        # number of warped bins = warp_length / bin_size_ms
        n_bins = max(int(self.warp_length // self.bin_size_ms), 1) # double check +1 -- idk about it

        # standardized changepoints by core name
        standardized_cp = {
            key.replace("dataset_", "").split("_repacked.npz")[0]: value
            for key, value in self.changepoints_dict.items()
        }

        # helper to warp any DataFrame by taste/trial/changepoint segments
        def warp_df(df: pl.DataFrame, core: str) -> pl.DataFrame:
            dims = [c for c in df.columns if c.startswith("PC_") or c.startswith("latent_dim_")]
            warped_segments = []
            tastes = df["taste"].unique().to_list()
            for taste_idx, taste in enumerate(tastes):
                taste_df = df.filter(pl.col("taste") == taste)
                trials = taste_df["trial"].unique().to_list()
                cp_per_taste = standardized_cp[core][taste_idx]
                n_cps = cp_per_taste.shape[1]
                for trial_idx, trial in enumerate(trials):
                    trial_df = taste_df.filter(pl.col("trial") == trial)
                    cps = cp_per_taste[trial_idx]
                    for eid in range(n_cps + 1):
                        if eid == 0:
                            start = self.start_time_ms
                            end = cps[0]
                        elif eid < n_cps:
                            start = cps[eid - 1]
                            end = cps[eid]
                        else:
                            start = cps[-1]
                            end = self.max_time_ms
                        seg = trial_df.filter(
                            (pl.col("time") >= start) & (pl.col("time") < end)
                        )
                        if seg.is_empty():
                            continue
                        t_rel = seg["time"].to_numpy().astype(float) - start
                        # normalized source
                        src = np.linspace(0, 1, len(t_rel))
                        tgt = np.linspace(0, 1, n_bins)
                        warped_vals = {}
                        for dim in dims:
                            v = seg[dim].to_numpy()
                            f = interp1d(src, v, kind="linear", fill_value="extrapolate")
                            warped_vals[dim] = f(tgt).tolist()
                        warped_df = pl.DataFrame(warped_vals)
                        # rebuild time in ms increments-- two ways to do it:
                        times = (start + np.arange(n_bins) * self.bin_size_ms).tolist()
                        # or: (should be inclusive of the end bin?)
                        #times = np.linspace(start, end, n_bins).tolist() # eval later...

                        warped_df = warped_df.with_columns([
                            pl.Series("taste", [taste] * n_bins),
                            pl.Series("trial", [trial] * n_bins),
                            pl.Series("changepoint", [eid] * n_bins),
                            pl.Series("time", times),
                        ])
                        warped_segments.append(warped_df)
            return pl.concat(warped_segments) if warped_segments else pl.DataFrame()
# TODO: Add RNN to the naming scheme of these so as to not confuse with any PCA that's done on firing rate-- yikes, this needs to be fixed...
        # 1) Epochs
        new_epochs = {}
        for key, df in self.epoch_dataframes_dict_unwarped.items():
            # find base dataset name by matching prefix in standardized_cp
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            new_epochs[f"{base}_latent_warped"] = warp_df(df, base)
        self.epoch_dataframes_dict_warped = new_epochs

        # 2) PCA thresholded
        thresh = int(self.variance_threshold)
        pca_unw = getattr(self, f"robust_pca_{thresh}_unwarped", {})
        new_pca = {}
        for key, df in pca_unw.items():
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            new_pca[f"{base}_pca_{thresh}_warped"] = warp_df(df, base)
        setattr(self, f"robust_pca_{thresh}_warped", new_pca)

        # 3) PCA full
        new_full = {}
        for key, df in self.robust_pca_full_unwarped.items():
            base = next((b for b in standardized_cp if key.startswith(b)), None)
            if base is None:
                raise KeyError(f"Cannot find base dataset for key {key}")
            new_full[f"{base}_pca_full_warped"] = warp_df(df, base)
        self.robust_pca_full_warped = new_full

        # 4) Derivatives
        fd_attr = f"first_derivatives_{thresh}_unwarped"
        if hasattr(self, fd_attr):
            src_fd = getattr(self, fd_attr)
            new_fd = {}
            for key, df in src_fd.items():
                base = next((b for b in standardized_cp if key.startswith(b)), None)
                if base is None:
                    raise KeyError(f"Cannot find base dataset for key {key}")
                new_fd[f"{base}_first_derivatives_{thresh}_warped"] = warp_df(df, base)
            setattr(self, f"first_derivatives_{thresh}_warped", new_fd)

        sd_attr = f"second_derivatives_{thresh}_unwarped"
        if hasattr(self, sd_attr):
            src_sd = getattr(self, sd_attr)
            new_sd = {}
            for key, df in src_sd.items():
                base = next((b for b in standardized_cp if key.startswith(b)), None)
                if base is None:
                    raise KeyError(f"Cannot find base dataset for key {key}")
                new_sd[f"{base}_second_derivatives_{thresh}_warped"] = warp_df(df, base)
            setattr(self, f"second_derivatives_{thresh}_warped", new_sd)


    # now also a func that runs PCA on the RNN latents in one place so I can process from there
    def run_robust_pca_analysis(
        self,
        return_dicts=True,
        # variance_threshold=95.0, # now a global
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

        for dataset_name, df in self.epoch_dataframes_dict_unwarped.items():
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
                        np.argmax(cumulative_variance > int(self.variance_threshold)) + 1
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
                pca_lat_dict_thresh[
                    f"{dataset_name}_pca_{processed_dataframes_thresh}_unwarped"
                ] = pl.concat(aligned_thresh)
            if processed_dataframes_full:
                aligned_full = align_dataframes(processed_dataframes_full)
                pca_lat_dict_full[
                    f"{dataset_name}_pca_full_unwarped"
                ] = pl.concat(aligned_full)


        # setattr(self, f"robust_pca_{int(variance_threshold)}_unwarped", pca_lat_dict_thresh)
        # self.robust_pca_full_unwarped = pca_lat_dict_full
        setattr(self,
        f"robust_pca_{int(self.variance_threshold)}_unwarped",
        pca_lat_dict_thresh)
        setattr(self,
        "robust_pca_full_unwarped",
        pca_lat_dict_full)

        # Derivative source selector
        if derivative_source == "threshold":
            source_dict = pca_lat_dict_thresh
            source_name = str(int(self.variance_threshold))
        elif derivative_source == "full":
            source_dict = pca_lat_dict_full
            source_name = "full"
        else:
            raise ValueError(
                "Invalid derivative_source. Must be 'threshold' or 'full'."
            )
       
        if compute_first_derivative:
            first_derivative_dict = {
                f"{key}_first_derivatives_{source_name}_unwarped":
                    self.compute_first_derivative(df)
                for key, df in source_dict.items()
            }
            setattr(
                self,
                f"first_derivatives_{source_name}_unwarped",
                first_derivative_dict
            )

        # Compute second derivatives with descriptive keys
        if compute_second_derivative:
            second_derivative_dict = {
                f"{key}_second_derivatives_{source_name}_unwarped":
                    self.compute_second_derivative(df)
                for key, df in source_dict.items()
            }
            setattr(
                self,
                f"second_derivatives_{source_name}_unwarped",
                second_derivative_dict
            )

        return (pca_lat_dict_thresh, pca_lat_dict_full) if return_dicts else None

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

    def save_analysis_outputs(self, tld):
        """
        Saves all analysis outputs (unwarped and warped) into organized subdirectories as Parquet files.

        Parameters
        ----------
        tld : str or Path
            Top-level directory to save outputs into.
        """
        tld = Path(tld); tld.mkdir(parents=True, exist_ok=True)
        thr = int(self.variance_threshold)
    
        outputs = {
            "epoch_dataframes": (
                self.epoch_dataframes_dict_unwarped,
                self.epoch_dataframes_dict_warped
            ),
            f"robust_pca_{thr}": (
                getattr(self, f"robust_pca_{thr}_unwarped", {}),
                getattr(self, f"robust_pca_{thr}_warped", {})
            ),
            "robust_pca_full": (
                self.robust_pca_full_unwarped,
                self.robust_pca_full_warped
            ),
            f"first_derivatives_{thr}": (
                getattr(self, f"first_derivatives_{thr}_unwarped", {}),
                getattr(self, f"first_derivatives_{thr}_warped", {})
            ),
            f"second_derivatives_{thr}": (
                getattr(self, f"second_derivatives_{thr}_unwarped", {}),
                getattr(self, f"second_derivatives_{thr}_warped", {})
            ),
        }
        for name, (unw, w) in outputs.items():
            for suffix, d in (("_unwarped", unw), ("_warped", w)):
                subdir = tld / f"{name}{suffix}"
                subdir.mkdir(parents=True, exist_ok=True)
                if not d:
                    print(f"[WARNING] No {suffix[1:]} data for {name}, skipping.")
                    continue
                for ds_key, df in d.items():
                    core = ds_key.split(suffix)[0]
                    out = subdir / f"{core}_{name}{suffix}.parquet"
                    df.write_parquet(out)
                    print(f"Saved {core} [{name}{suffix}] to {out}")

    def full_pipeline(
        self,
        compute_first_derivative=False,
        compute_second_derivative=False,
        derivative_source="threshold",
        return_derivatives=True,
        save_outputs=False
    ):
        """
        Full pipeline: preprocess, PCA, warp, and save.
        Returns separate dicts for each analysis type (unwarped and, optionally, warped).
        """
        # Preprocessing
        self.read_parquet_files()
        self.structure_latent_arrays()
        self.extract_changepoints()
        self.split_epochs_by_changepoints()

        # PCA and derivatives
        self.run_robust_pca_analysis(
            compute_first_derivative=compute_first_derivative,
            compute_second_derivative=compute_second_derivative,
            derivative_source=derivative_source
        )
        # Warping
        self.warp_all_outputs()

        # Optional saving
        if save_outputs and self.save_dir:
            self.save_analysis_outputs(self.save_dir)

        # Build and return separate dicts
        thresh = int(self.variance_threshold)

        # Epochs
        epochs_unw = self.epoch_dataframes_dict_unwarped
        epochs_w   = self.epoch_dataframes_dict_warped if return_derivatives else {}

        # PCA thresholded
        pca_thresh_unw = getattr(self, f"robust_pca_{thresh}_unwarped", {})
        pca_thresh_w   = getattr(self, f"robust_pca_{thresh}_warped", {}) if return_derivatives else {}

        # PCA full
        pca_full_unw = self.robust_pca_full_unwarped
        pca_full_w   = self.robust_pca_full_warped if return_derivatives else {}

        # Derivatives
        fd_unw = getattr(self, f"first_derivatives_{thresh}_unwarped", {}) if compute_first_derivative else {}
        sd_unw = getattr(self, f"second_derivatives_{thresh}_unwarped", {}) if compute_second_derivative else {}
        fd_w   = getattr(self, f"first_derivatives_{thresh}_warped", {}) if (compute_first_derivative and return_derivatives) else {}
        sd_w   = getattr(self, f"second_derivatives_{thresh}_warped", {}) if (compute_second_derivative and return_derivatives) else {}

        return (
            epochs_unw,
            pca_thresh_unw,
            pca_full_unw,
            fd_unw,
            sd_unw,
            epochs_w,
            pca_thresh_w,
            pca_full_w,
            fd_w,
            sd_w,
        )


# # debugging land: 

# processor = RNNLatentProcessor(
#     parquet_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/raw_lat_parquet",
#     npz_path=npz_path,
#     info_path=info_path,
#     pkl_path=pkl_path,
#     taste_replacements=taste_replacements,
#     save_dir="/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/RNN_PROCESSING_PARQUETS",  # <-- NEW
#     bin_size_ms=25,
#     start_time_ms=1500,
#     max_time_ms=4500,
#     warp_length = 1000,
#     variance_threshold = 95.0,
# )

# for ds_key, df in processor.epoch_dataframes_dict_unwarped.items():
#     # count unique (taste,trial,changepoint) groups
#     n_segs = df.select(["taste","trial","changepoint"]).unique().height
#     warped = processor.epoch_dataframes_dict_warped.get(ds_key.replace("_unwarped","_warped"))
#     warped_rows = warped.height if warped is not None else 0
#     print(f"{ds_key}:")
#     print(f"  → unwarped segments: {n_segs}")
#     print(f"  → warp_length: {processor.warp_length}")
#     print(f"  → expected warped rows = {n_segs} * {processor.warp_length} = {n_segs * processor.warp_length}")
#     print(f"  → actual warped rows: {warped_rows}")
    
# import numpy as np
# for ds_key, df in processor.epoch_dataframes_dict_unwarped.items():
#     seg_sizes = [
#         len(grp)
#         for _, grp in df.groupby(["taste","trial","changepoint"])
#     ]
#     print(f"{ds_key} segment‐lengths summary:")
#     print("   min, median, max =", np.min(seg_sizes), np.median(seg_sizes), np.max(seg_sizes))
#     print("   unique lengths:", np.unique(seg_sizes)[:10], "…")

# for ds_key, df in processor.epoch_dataframes_dict_unwarped.items():
#     cps = df.select("changepoint").unique().sort("changepoint")["changepoint"].to_list()
#     print(f"{ds_key} has changepoint levels: {cps}")

# sd = processor.changepoints_dict  # raw from NPZ
# for ds_key, arr in sd.items():
#     print(ds_key, "raw changepoints array shape:", np.array(arr).shape)
#     print("  first taste trial changepoints:", arr[0][0])
#     break  # just the first dataset to sanity‐check


# original runtime that didn't involve warping that also works: 

# class RNNLatentProcessor:
#     def __init__(
#         self,
#         parquet_dir,
#         npz_path,
#         info_path,
#         pkl_path,
#         taste_replacements,
#         bin_size_ms=25,
#         start_time_ms=1500,
#         max_time_ms=4500,
#         save_dir=None,
#     ):
#         self.parquet_dir = Path(parquet_dir)
#         self.npz_path = npz_path
#         self.info_path = info_path
#         self.pkl_path = pkl_path
#         self.taste_replacements = taste_replacements
#         self.bin_size_ms = bin_size_ms
#         self.start_time_ms = start_time_ms
#         self.max_time_ms = max_time_ms

#         self.save_dir = Path(save_dir) if save_dir else None  # Save path
#         self.taste_latent = {}
#         self.processed_data = {}
#         self.changepoints_dict = {}
#         self.epoch_dataframes_dict = {}

#     def read_parquet_files(self):
#         self.taste_latent = read_parquet_files_into_dict(self.parquet_dir)

#     def structure_latent_arrays(self):
#         # refactored slightly for speed
#         for dataset_name, df in self.taste_latent.items():
#             unique_tastes = df["taste"].unique()
#             unique_trials = df["trial"].unique()
#             latent_columns = [
#                 col for col in df.columns if col.startswith("latent_dim_")
#             ]
#             latent_dim = len(latent_columns)

#             # Find number of time steps
#             num_time_steps = len(
#                 df.filter(
#                     (pl.col("trial") == unique_trials[0])
#                     & (pl.col("taste") == unique_tastes[0])
#                 )
#             )

#             # Dataset array: (taste, trial, latent_dim, time)
#             dataset_array = np.empty(
#                 (len(unique_tastes), len(unique_trials), latent_dim, num_time_steps)
#             )

#             # --- Pre-group by (taste, trial) once for speed
#             trial_lookup = {}
#             for taste in unique_tastes:
#                 for trial in unique_trials:
#                     key = (taste, trial)
#                     trial_data = df.filter(
#                         (pl.col("taste") == taste) & (pl.col("trial") == trial)
#                     )
#                     if trial_data.height > 0:
#                         trial_lookup[key] = (
#                             trial_data.select(latent_columns).to_numpy().T
#                         )  # (latent_dim, time)

#             # --- Now fast loop
#             for taste_idx, taste in enumerate(unique_tastes):
#                 for trial_idx, trial in enumerate(unique_trials):
#                     key = (taste, trial)
#                     if key in trial_lookup:
#                         dataset_array[taste_idx, trial_idx, :, :] = trial_lookup[key]
#                     else:
#                         print(f"Warning: Missing data for taste {taste} trial {trial}")

#             self.processed_data[dataset_name] = dataset_array

#     def extract_changepoints(self):
#         for data in extract_from_npz(self.npz_path):
#             if isinstance(data, tuple):
#                 spike_array, dataset_num, index, key = data
#                 dataset_name = f"dataset_{dataset_num}"
#                 dataset_tastes = process_info_files(self.info_path, dataset_num)
#                 modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
#                 extracted_pkl = extract_valid_changepoints(
#                     self.pkl_path, spike_array, dataset_num, index, key
#                 )

#                 if extracted_pkl is not None:
#                     try:
#                         changepoints = extracted_pkl[
#                             :, 3
#                         ]  
#                         self.changepoints_dict[dataset_name] = changepoints
#                     except IndexError:
#                         print(
#                             f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
#                         )
#                 else:
#                     print(f"No valid changepoints found for dataset {dataset_name}")
#     # method that saves changepoints so the entire processor doesn't need to be run again and again 
#     def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
#         """
#         Extract only the changepoints for each dataset (populates self.changepoints_dict)
#         without running PCA or derivative steps.
#         Optionally saves them to disk as 'changepoints_dict.pkl' under self.save_dir.
#         """
#         # (re)initialize
#         self.changepoints_dict = {}
#         # delegate to original: 
#         self.extract_changepoints()
#         if save_outputs:
#             save_path = Path(self.save_dir) / "changepoints"
#             save_path.mkdir(parents=True, exist_ok=True)
#             # optionally persist for future runs
#             for dataset_name, changepoints in self.changepoints_dict.items():
#                 clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
#                 out_path = save_path / f"{clean_name}_changepoints.pkl"
#                 if not out_path.exists():
#                     with open(out_path, "wb") as f:
#                         pickle.dump(self.changepoints_dict, f)

#         return self.changepoints_dict

#     def split_epochs_by_changepoints(self):
#         standardized_changepoints_dict = {
#             key.replace("dataset_", "").split("_repacked.npz")[0]: value
#             for key, value in self.changepoints_dict.items()
#         }

#         for dataset_name, data_array in self.processed_data.items():
#             core_dataset_name = dataset_name.split("_repacked_raw_latent_vectors")[0]

#             changepoints = standardized_changepoints_dict.get(core_dataset_name, None)
#             if changepoints is None:
#                 print(f"No changepoints for {core_dataset_name}, skipping.")
#                 continue

#             num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape
#             all_epochs = []
#             time_in_ms = [
#                 (t * self.bin_size_ms) + self.start_time_ms
#                 for t in range(num_time_steps)
#             ]

#             if len(changepoints) != num_tastes:
#                 print(f"Mismatch in taste dimension for dataset {core_dataset_name}")
#                 continue

#             for taste_idx in range(num_tastes):
#                 if len(changepoints[taste_idx]) < num_trials:
#                     print(
#                         f"Mismatch in trial dimension for dataset {core_dataset_name}, taste {taste_idx}"
#                     )
#                     continue

#                 for trial_idx in range(num_trials):
#                     trial_changepoints = changepoints[taste_idx][trial_idx]
#                     start_idx = 0

#                     for cp_idx, changepoint_ms in enumerate(trial_changepoints):
#                         end_idx = next(
#                             (
#                                 i
#                                 for i, t in enumerate(time_in_ms)
#                                 if t >= changepoint_ms
#                             ),
#                             num_time_steps,
#                         )

#                         epoch_data = data_array[
#                             taste_idx, start_idx:end_idx, :, trial_idx
#                         ]  # (time, latent)

#                         epoch_df = pl.DataFrame(
#                             epoch_data,
#                             schema=[f"latent_dim_{i}" for i in range(latent_dim)],
#                         )
#                         epoch_df = epoch_df.with_columns(
#                             [
#                                 pl.Series("taste", [taste_idx] * len(epoch_df)),
#                                 pl.Series("trial", [trial_idx] * len(epoch_df)),
#                                 pl.Series("changepoint", [cp_idx] * len(epoch_df)),
#                                 pl.Series("time", time_in_ms[start_idx:end_idx]),
#                             ]
#                         )
#                         all_epochs.append(epoch_df)
#                         start_idx = end_idx

#                     if start_idx < num_time_steps:
#                         epoch_data = data_array[
#                             taste_idx, start_idx:num_time_steps, :, trial_idx
#                         ]
#                         epoch_df = pl.DataFrame(
#                             epoch_data,
#                             schema=[f"latent_dim_{i}" for i in range(latent_dim)],
#                         )
#                         epoch_df = epoch_df.with_columns(
#                             [
#                                 pl.Series("taste", [taste_idx] * len(epoch_df)),
#                                 pl.Series("trial", [trial_idx] * len(epoch_df)),
#                                 pl.Series(
#                                     "changepoint",
#                                     [len(trial_changepoints)] * len(epoch_df),
#                                 ),
#                                 pl.Series("time", time_in_ms[start_idx:num_time_steps]),
#                             ]
#                         )
#                         all_epochs.append(epoch_df)

#             if all_epochs:
#                 self.epoch_dataframes_dict[dataset_name] = pl.concat(all_epochs)
#             else:
#                 print(f"No valid epochs for dataset {dataset_name}, skipping.")

#     # now also a func that runs PCA on the RNN latents in one place so I can process from there
#     def run_robust_pca_analysis(
#         self,
#         return_dicts=True,
#         variance_threshold=95.0,
#         compute_first_derivative=False,
#         compute_second_derivative=False,
#         derivative_source="threshold",
#     ):
#         """
#         Perform PCA on concatenated data grouped by taste and changepoint, then split results back accordingly.
#         Optionally computes first and second derivatives.

#         Parameters
#         ----------
#         return_dicts : bool, optional
#             Whether to return the PCA dictionaries.
#         variance_threshold : float, optional
#             The cumulative explained variance percentage to retain.
#         compute_first_derivative : bool, optional
#             Whether to compute and store first derivatives.
#         compute_second_derivative : bool, optional
#             Whether to compute and store second derivatives.
#         derivative_source : str, optional
#             "threshold" (default) to compute derivatives on robust_pca_<threshold>;
#             "full" to compute derivatives on robust_pca_full.

#         Returns
#         -------
#         (dict, dict) or None
#             (robust_pca_threshold, robust_pca_full) if return_dicts=True, else None
#         """
#         pca_lat_dict_thresh = {}
#         pca_lat_dict_full = {}

#         for dataset_name, df in self.epoch_dataframes_dict.items():
#             unique_tastes = df["taste"].unique().to_list()
#             unique_changepoints = df["changepoint"].unique().to_list()

#             processed_dataframes_thresh = []
#             processed_dataframes_full = []

#             for taste in unique_tastes:
#                 for changepoint in unique_changepoints:
#                     filtered_df = df.filter(
#                         (pl.col("taste") == taste)
#                         & (pl.col("changepoint") == changepoint)
#                     )

#                     if filtered_df.is_empty():
#                         continue

#                     latent_columns = [
#                         col for col in filtered_df.columns if "latent_dim_" in col
#                     ]
#                     if not latent_columns:
#                         continue

#                     unique_trials = filtered_df["trial"].unique().to_list()
#                     trial_data_list = [
#                         filtered_df.filter(pl.col("trial") == trial)
#                         .select(latent_columns)
#                         .to_numpy()
#                         for trial in unique_trials
#                     ]

#                     concatenated_data = np.vstack(trial_data_list)

#                     if concatenated_data.shape[0] < 2:
#                         continue

#                     pca = PCA()
#                     transformed_data = pca.fit_transform(concatenated_data)
#                     explained_variance = pca.explained_variance_ratio_ * 100

#                     cumulative_variance = np.cumsum(explained_variance)
#                     num_pcs_thresh = (
#                         np.argmax(cumulative_variance > variance_threshold) + 1
#                     )

#                     split_indices = np.cumsum(
#                         [arr.shape[0] for arr in trial_data_list]
#                     )[:-1]
#                     split_pca_data = np.split(transformed_data, split_indices)

#                     pc_names_full = [
#                         f"PC_{i+1}" for i in range(transformed_data.shape[1])
#                     ]
#                     explained_var_cols_full = [
#                         f"explained_variance_{pc}" for pc in pc_names_full
#                     ]

#                     pc_names_thresh = [f"PC_{i+1}" for i in range(num_pcs_thresh)]
#                     explained_var_cols_thresh = [
#                         f"explained_variance_{pc}" for pc in pc_names_thresh
#                     ]

#                     for trial, trial_pca_data in zip(unique_trials, split_pca_data):
#                         trial_metadata = filtered_df.filter(
#                             pl.col("trial") == trial
#                         ).select(["trial", "time"])

#                         full_pca_df = pl.DataFrame(
#                             np.hstack(
#                                 [
#                                     trial_pca_data,
#                                     np.tile(
#                                         explained_variance, (len(trial_pca_data), 1)
#                                     ),
#                                 ]
#                             ),
#                             schema=pc_names_full + explained_var_cols_full,
#                         ).with_columns(
#                             [
#                                 pl.lit(taste).alias("taste"),
#                                 pl.lit(changepoint).alias("changepoint"),
#                             ]
#                         )
#                         full_pca_df = pl.concat(
#                             [trial_metadata, full_pca_df], how="horizontal"
#                         )
#                         processed_dataframes_full.append(full_pca_df)

#                         pca_df_thresh = pl.DataFrame(
#                             np.hstack(
#                                 [
#                                     trial_pca_data[:, :num_pcs_thresh],
#                                     np.tile(
#                                         explained_variance[:num_pcs_thresh],
#                                         (len(trial_pca_data), 1),
#                                     ),
#                                 ]
#                             ),
#                             schema=pc_names_thresh + explained_var_cols_thresh,
#                         ).with_columns(
#                             [
#                                 pl.lit(taste).alias("taste"),
#                                 pl.lit(changepoint).alias("changepoint"),
#                             ]
#                         )
#                         pca_df_thresh = pl.concat(
#                             [trial_metadata, pca_df_thresh], how="horizontal"
#                         )
#                         processed_dataframes_thresh.append(pca_df_thresh)

#             def align_dataframes(dataframes):
#                 all_columns = set()
#                 for df in dataframes:
#                     all_columns.update(df.columns)
#                 all_columns = sorted(all_columns)

#                 aligned_dfs = []
#                 for df in dataframes:
#                     missing_columns = set(all_columns) - set(df.columns)
#                     for col in missing_columns:
#                         df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
#                     df = df.select(all_columns)
#                     aligned_dfs.append(df)
#                 return aligned_dfs

#             if processed_dataframes_thresh:
#                 aligned_thresh = align_dataframes(processed_dataframes_thresh)
#                 pca_lat_dict_thresh[dataset_name] = pl.concat(aligned_thresh)
#             if processed_dataframes_full:
#                 aligned_full = align_dataframes(processed_dataframes_full)
#                 pca_lat_dict_full[dataset_name] = pl.concat(aligned_full)

#         setattr(self, f"robust_pca_{int(variance_threshold)}", pca_lat_dict_thresh)
#         self.robust_pca_full = pca_lat_dict_full

#         # Derivative source selector
#         if derivative_source == "threshold":
#             source_dict = pca_lat_dict_thresh
#             source_name = str(int(variance_threshold))
#         elif derivative_source == "full":
#             source_dict = pca_lat_dict_full
#             source_name = "full"
#         else:
#             raise ValueError(
#                 "Invalid derivative_source. Must be 'threshold' or 'full'."
#             )

#         # Compute derivatives if requested
#         if compute_first_derivative:
#             first_derivative_dict = {}
#             for key, df in source_dict.items():
#                 first_derivative_df = self.compute_first_derivative(df)
#                 first_derivative_dict[key] = first_derivative_df
#             setattr(self, f"first_derivatives_{source_name}", first_derivative_dict)

#         if compute_second_derivative:
#             second_derivative_dict = {}
#             for key, df in source_dict.items():
#                 second_derivative_df = self.compute_second_derivative(df)
#                 second_derivative_dict[key] = second_derivative_df
#             setattr(self, f"second_derivatives_{source_name}", second_derivative_dict)

#         if return_dicts:
#             return pca_lat_dict_thresh, pca_lat_dict_full
#         else:
#             return None

#     def compute_first_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
#         """
#         Compute the first derivative of latent dimension columns in a Polars DataFrame.
#         Assumes uniform time spacing (ignores actual 'time' values).
#         """
#         meta_cols = ["taste", "trial", "changepoint", "time"]
#         latent_cols = [col for col in df.columns if col not in meta_cols]

#         data = df.select(latent_cols).to_numpy()
#         first_deriv = np.gradient(data, axis=0)

#         result_df = df.with_columns(
#             [
#                 pl.Series(name=col, values=first_deriv[:, j])
#                 for j, col in enumerate(latent_cols)
#             ]
#         )
#         return result_df

#     def compute_second_derivative(self, df: pl.DataFrame) -> pl.DataFrame:
#         """
#         Compute the second derivative of latent dimension columns in a Polars DataFrame.
#         Assumes uniform time spacing (ignores actual 'time' values).
#         """
#         meta_cols = ["taste", "trial", "changepoint", "time"]
#         latent_cols = [col for col in df.columns if col not in meta_cols]

#         data = df.select(latent_cols).to_numpy()
#         first_deriv = np.gradient(data, axis=0)
#         second_deriv = np.gradient(first_deriv, axis=0)

#         result_df = df.with_columns(
#             [
#                 pl.Series(name=col, values=second_deriv[:, j])
#                 for j, col in enumerate(latent_cols)
#             ]
#         )
#         return result_df

#     def save_analysis_outputs(self, tld):
#         """
#         Saves all analysis outputs (epoch_dataframes_dict, robust_pca_95, robust_pca_full,
#         first_derivs, second_derivs) into organized subdirectories as Parquet files.

#         Parameters
#         ----------
#         tld : str or Path
#             Top-level directory to save outputs into.
#         """
#         tld = Path(tld)
#         tld.mkdir(parents=True, exist_ok=True)  # Ensure top-level directory exists

#         # Dictionary of analysis results to save
#         analysis_dicts = {
#             "epoch_dataframes_dict": self.epoch_dataframes_dict,
#             "robust_pca_95": getattr(self, "robust_pca_95", {}),
#             "robust_pca_full": getattr(self, "robust_pca_full", {}),
#             "first_derivs": getattr(self, "first_derivs", {}),
#             "second_derivs": getattr(self, "second_derivs", {}),
#         }

#         for analysis_name, analysis_data in analysis_dicts.items():
#             # Create subdirectory for this analysis type
#             analysis_dir = tld / analysis_name
#             analysis_dir.mkdir(parents=True, exist_ok=True)

#             if not analysis_data:
#                 print(f"[WARNING] No data found for {analysis_name}, skipping.")
#                 continue

#             for dataset_name, df in analysis_data.items():
#                 core_dataset_name = dataset_name.split("_repacked_raw_latent_vectors")[0]
#                 if isinstance(df, pl.DataFrame):
#                     save_path = analysis_dir / f"{core_dataset_name}_{analysis_name}.parquet"
#                     df.write_parquet(save_path)
#                     print(f"Saved {core_dataset_name} [{analysis_name}] to {save_path}")
#                 else:
#                     print(
#                         f"[WARNING] Skipping {core_dataset_name} for {analysis_name} — not a Polars DataFrame."
#                     )

#     def full_pipeline(
#         self,
#         variance_threshold=95.0,
#         compute_first_derivative=False,
#         compute_second_derivative=False,
#         derivative_source="threshold",
#         return_derivatives=True,
#         save_outputs=False,
#     ):
#         """
#         Full one-liner pipeline:
#         - Read parquet files
#         - Structure latent arrays
#         - Extract changepoints
#         - Split into epochs
#         - Run PCA
#         - Optionally compute derivatives
#         - Optionally save outputs
#         - Returns everything needed downstream.

#         Parameters
#         ----------
#         variance_threshold : float
#             Threshold for PCA explained variance (default 95).
#         compute_first_derivative : bool
#             Whether to compute first derivatives.
#         compute_second_derivative : bool
#             Whether to compute second derivatives.
#         derivative_source : str
#             "threshold" or "full" PCA to compute derivatives from.
#         return_derivatives : bool
#             Whether to return derivative outputs separately.
#         save_outputs : bool
#             Whether to save outputs automatically to save_dir.

#         Returns
#         -------
#         tuple
#             epoch_dataframes_dict, robust_pca_threshold, robust_pca_full
#             (plus first_derivatives, second_derivatives if return_derivatives=True)
#         """

#         # --- Step 1: Full pre-processing
#         self.read_parquet_files()
#         self.structure_latent_arrays()
#         self.extract_changepoints()
#         self.split_epochs_by_changepoints()

#         # --- Step 2: Run PCA
#         self.robust_pca_95, self.robust_pca_full = self.run_robust_pca_analysis(
#             return_dicts=True, variance_threshold=variance_threshold
#         )

#         # --- Step 3: Compute derivatives if needed
#         self.first_derivs = {}
#         self.second_derivs = {}

#         if compute_first_derivative or compute_second_derivative:
#             if derivative_source == "threshold":
#                 source_dict = self.robust_pca_95
#             elif derivative_source == "full":
#                 source_dict = self.robust_pca_full
#             else:
#                 raise ValueError(
#                     "Invalid derivative_source. Must be 'threshold' or 'full'."
#                 )

#         if compute_first_derivative:
#             for key, df in source_dict.items():
#                 self.first_derivs[key] = self.compute_first_derivative(df)

#         if compute_second_derivative:
#             for key, df in source_dict.items():
#                 self.second_derivs[key] = self.compute_second_derivative(df)

#         # --- Step 4: Save outputs if requested
#         if save_outputs and self.save_dir is not None:
#             self.save_analysis_outputs(tld=self.save_dir)

#         # --- Step 5: Return outputs
#         if return_derivatives:
#             return (
#                 self.epoch_dataframes_dict,
#                 self.robust_pca_95,
#                 self.robust_pca_full,
#                 self.first_derivs,
#                 self.second_derivs,
#             )
#         else:
#             return self.epoch_dataframes_dict, self.robust_pca_95, self.robust_pca_full

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



# Keeping this here: 
    
    # def extract_changepoints(self):
    #     for data in extract_from_npz(self.npz_path):
    #         if isinstance(data, tuple):
    #             spike_array, dataset_num, index, key = data
    #             dataset_name = f"dataset_{dataset_num}"
    #             dataset_tastes = process_info_files(self.info_path, dataset_num)
    #             modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
    #             extracted_pkl = extract_valid_changepoints(
    #                 self.pkl_path, spike_array, dataset_num, index, key
    #             )

    #             if extracted_pkl is not None:
    #                 try:
    #                     changepoints = extracted_pkl[
    #                         :, 3
    #                     ]  # Keep full array, not just one column
    #                     self.changepoints_dict[dataset_name] = changepoints
    #                 except IndexError:
    #                     print(
    #                         f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
    #                     )
    #             else:
    #                 print(f"No valid changepoints found for dataset {dataset_name}")
    # # method that saves changepoints so the entire processor doesn't need to be run again and again 
    # def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
    #     """
    #     Extract only the changepoints for each dataset (populates self.changepoints_dict)
    #     without running PCA or derivative steps.
    #     Optionally saves them to disk as 'changepoints_dict.pkl' under self.save_dir.
    #     """
    #     # (re)initialize
    #     self.changepoints_dict = {}
    #     # delegate to original: 
    #     self.extract_changepoints()
    
    #     # loop through the same NPZ‐based generator you use in extract_changepoints()
    #     for data in extract_from_npz(self.npz_path):
    #         if not isinstance(data, tuple):
    #             continue
    #         spike_array, dataset_num, index, key = data
    #         dataset_name = f"dataset_{dataset_num}"
    
    #         # replicate the minimal preprocessing to get at your PKL
    #         dataset_tastes = process_info_files(self.info_path, dataset_num)
    #         modified_tastes = modify_tastes(dataset_tastes, self.taste_replacements)
    #         extracted_pkl = extract_valid_changepoints(
    #             self.pkl_path, spike_array, dataset_num, index, key
    #         )
    
    #         if extracted_pkl is None:
    #             print(f"No valid changepoints found for dataset {dataset_name}")
    #             continue
    
    #         try:
    #             # column 3 holds your full changepoint array
    #             changepoints = extracted_pkl[:, 3]
    #             self.changepoints_dict[dataset_name] = changepoints
    #         except IndexError:
    #             print(
    #                 f"Index error with dataset {dataset_name}: "
    #                 f"PKL shape was {extracted_pkl.shape}"
    #             )
    #     save_path = Path(self.save_dir) / "changepoints"
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     # optionally persist for future runs
    #     for dataset_name, changepoints in self.changepoints_dict.items():
    #         clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
    #         out_path = save_path / f"{clean_name}_changepoints.pkl"
    #         if not out_path.exists():
    #             with open(out_path, "wb") as f:
    #                 pickle.dump(self.changepoints_dict, f)

    #     return self.changepoints_dict



# old PCA method:
# def run_robust_pca_analysis(self, return_dicts=True, variance_threshold=95.0):
#     """
#     Perform PCA on concatenated data grouped by taste and changepoint, then split results back accordingly.
#     Stores two dictionaries internally:
#     1. self.robust_pca_<variance_threshold> - PCs up to given explained variance.
#     2. self.robust_pca_full - All PCs without thresholding.

#     Parameters
#     ----------
#     return_dicts : bool, optional
#         If True, return the PCA dictionaries. If False, only store them internally.
#     variance_threshold : float, optional
#         The cumulative explained variance percentage to retain (default 95%).

#     Returns
#     -------
#     (dict, dict) or None
#         (robust_pca_threshold, robust_pca_full) if return_dicts=True, else None
#     """
#     pca_lat_dict_thresh = {}
#     pca_lat_dict_full = {}

#     for dataset_name, df in self.epoch_dataframes_dict.items():
#         unique_tastes = df['taste'].unique().to_list()
#         unique_changepoints = df['changepoint'].unique().to_list()

#         processed_dataframes_thresh = []
#         processed_dataframes_full = []

#         for taste in unique_tastes:
#             for changepoint in unique_changepoints:
#                 filtered_df = df.filter((pl.col('taste') == taste) & (pl.col('changepoint') == changepoint))

#                 if filtered_df.is_empty():
#                     continue

#                 latent_columns = [col for col in filtered_df.columns if 'latent_dim_' in col]
#                 if not latent_columns:
#                     continue

#                 unique_trials = filtered_df['trial'].unique().to_list()
#                 trial_data_list = [
#                     filtered_df.filter(pl.col('trial') == trial).select(latent_columns).to_numpy()
#                     for trial in unique_trials
#                 ]

#                 concatenated_data = np.vstack(trial_data_list)

#                 if concatenated_data.shape[0] < 2:
#                     continue

#                 pca = PCA()
#                 transformed_data = pca.fit_transform(concatenated_data)
#                 explained_variance = pca.explained_variance_ratio_ * 100

#                 cumulative_variance = np.cumsum(explained_variance)
#                 num_pcs_thresh = np.argmax(cumulative_variance > variance_threshold) + 1

#                 split_indices = np.cumsum([arr.shape[0] for arr in trial_data_list])[:-1]
#                 split_pca_data = np.split(transformed_data, split_indices)

#                 pc_names_full = [f"PC_{i+1}" for i in range(transformed_data.shape[1])]
#                 explained_var_cols_full = [f"explained_variance_{pc}" for pc in pc_names_full]

#                 pc_names_thresh = [f"PC_{i+1}" for i in range(num_pcs_thresh)]
#                 explained_var_cols_thresh = [f"explained_variance_{pc}" for pc in pc_names_thresh]

#                 for trial, trial_pca_data in zip(unique_trials, split_pca_data):
#                     trial_metadata = filtered_df.filter(pl.col('trial') == trial).select(["trial", "time"])

#                     full_pca_df = pl.DataFrame(
#                         np.hstack([trial_pca_data, np.tile(explained_variance, (len(trial_pca_data), 1))]),
#                         schema=pc_names_full + explained_var_cols_full
#                     ).with_columns([
#                         pl.lit(taste).alias("taste"),
#                         pl.lit(changepoint).alias("changepoint")
#                     ])
#                     full_pca_df = pl.concat([trial_metadata, full_pca_df], how="horizontal")
#                     processed_dataframes_full.append(full_pca_df)

#                     pca_df_thresh = pl.DataFrame(
#                         np.hstack([
#                             trial_pca_data[:, :num_pcs_thresh],
#                             np.tile(explained_variance[:num_pcs_thresh], (len(trial_pca_data), 1))
#                         ]),
#                         schema=pc_names_thresh + explained_var_cols_thresh
#                     ).with_columns([
#                         pl.lit(taste).alias("taste"),
#                         pl.lit(changepoint).alias("changepoint")
#                     ])
#                     pca_df_thresh = pl.concat([trial_metadata, pca_df_thresh], how="horizontal")
#                     processed_dataframes_thresh.append(pca_df_thresh)

#         def align_dataframes(dataframes):
#             all_columns = set()
#             for df in dataframes:
#                 all_columns.update(df.columns)
#             all_columns = sorted(all_columns)

#             aligned_dfs = []
#             for df in dataframes:
#                 missing_columns = set(all_columns) - set(df.columns)
#                 for col in missing_columns:
#                     df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
#                 df = df.select(all_columns)
#                 aligned_dfs.append(df)
#             return aligned_dfs

#         if processed_dataframes_thresh:
#             aligned_thresh = align_dataframes(processed_dataframes_thresh)
#             pca_lat_dict_thresh[dataset_name] = pl.concat(aligned_thresh)
#         if processed_dataframes_full:
#             aligned_full = align_dataframes(processed_dataframes_full)
#             pca_lat_dict_full[dataset_name] = pl.concat(aligned_full)

#     # Save internally with dynamic name
#     setattr(self, f'robust_pca_{int(variance_threshold)}', pca_lat_dict_thresh)
#     self.robust_pca_full = pca_lat_dict_full

#     if return_dicts:
#         return pca_lat_dict_thresh, pca_lat_dict_full
#     else:
#         return None


# Archive of the old stuff: working loops that are not a class


# taste_latent = read_parquet_files_into_dict('/Volumes/T7 Shield/octRNN/parquert')
# taste_sep_dir = '/Users/vincentcalia-bogan/Desktop/1BRANDEIS MAJOR STUFF/Katz lab/Senior thesis work/Figure datasets/RNN_taste_sep_dir'

# # generating arrays I can work with:
# # Assume `dataframes_dict` is the dictionary containing dataframes loaded from the Parquet files
# processed_data = {}

# for dataset_name, df in taste_latent.items():
#     # Get unique tastes and trials
#     unique_tastes = df['taste'].unique()
#     unique_trials = df['trial'].unique()

#     # Identify latent dimension columns explicitly
#     latent_columns = [col for col in df.columns if col.startswith("latent_dim_")]
#     latent_dim = len(latent_columns)

#     # Determine the number of time steps by counting entries for the first taste and trial
#     num_time_steps = len(df.filter(pl.col("trial") == unique_trials[0])
#                             .filter(pl.col("taste") == unique_tastes[0]))

#     # Initialize array to store the result in shape (taste, trial, latent_dim, time)
#     dataset_array = np.empty((len(unique_tastes), len(unique_trials), latent_dim, num_time_steps))

#     for taste_idx, taste in enumerate(unique_tastes):
#         for trial_idx, trial in enumerate(unique_trials):
#             # Filter data for this taste and trial
#             trial_data = df.filter((pl.col("taste") == taste) & (pl.col("trial") == trial))

#             # Extract latent dimensions and reshape to (latent_dim, time)
#             latent_values = trial_data.select(latent_columns).to_numpy().T  # Shape: (latent_dim, time)

#             # Insert into the appropriate location in the dataset array
#             dataset_array[taste_idx, trial_idx, :, :] = latent_values

#     # Store the result for this dataset
#     processed_data[dataset_name] = dataset_array

# # storing changepoints
# changepoints_dict = {}

# for data in extract_from_npz(npz_path):
#     if isinstance(data, tuple):
#         spike_array, dataset_num, index, key = data
#         print(f'Dataset number: {dataset_num}')

#         # Derive dataset name or adjust this to match your naming convention
#         dataset_name = f"dataset_{dataset_num}"

#         # Process taste data
#         dataset_tastes = process_info_files(info_path, dataset_num)
#         modified_tastes = modify_tastes(dataset_tastes, taste_replacements)

#         # Extract and process changepoints for the dataset
#         extracted_pkl = extract_valid_changepoints(pkl_path, spike_array, dataset_num, index, key)

#         # Check if extracted_pkl is not None and has the expected shape
#         if extracted_pkl is not None:
#             try:
#                 changepoints = extracted_pkl[:, 3]  # Select the relevant column for changepoints
#                 changepoints_dict[dataset_name] = changepoints
#             except IndexError:
#                 print(f"Index error with dataset {dataset_name}: extracted_pkl shape is {extracted_pkl.shape}")
#         else:
#             print(f"No valid changepoints found for dataset {dataset_name}")
# #processing in CP's now


# # Constants for RNN timing and binning
# BIN_SIZE_MS = 25  # Each bin represents 25 ms
# START_TIME_MS = 1500  # Start time in ms
# MAX_TIME_MS = 4500 # how long I let the RNN run... do post stim later ig

# epoch_dataframes_dict = {}

# # Standardize changepoints_dict keys to core names without prefixes/suffixes
# standardized_changepoints_dict = {
#     key.replace('dataset_', '').split('_repacked.npz')[0]: value
#     for key, value in changepoints_dict.items()
# }

# for dataset_name, data_array in processed_data.items():
#     # Extract the core name from dataset_name
#     core_dataset_name = dataset_name.split('_repacked_raw_latent_vectors')[0]

#     # Check if core_dataset_name exists in the standardized changepoints dictionary
#     if core_dataset_name in standardized_changepoints_dict:
#         changepoints = standardized_changepoints_dict[core_dataset_name]

#         # Get the number of tastes, time steps, latent dimensions, and trials
#         num_tastes, num_time_steps, latent_dim, num_trials = data_array.shape
#         all_epochs = []  # Store epochs as DataFrames for concatenation later

#         # Convert RNN time steps to ms
#         time_in_ms = [(t * BIN_SIZE_MS) + START_TIME_MS for t in range(num_time_steps)]

#         # Check if changepoints dimensions align with data_array dimensions
#         if len(changepoints) < num_tastes:
#             print(f"Mismatch in taste dimension for dataset {core_dataset_name}")
#             continue

#         # Iterate over each taste and trial
#         for taste_idx in range(num_tastes):
#             if len(changepoints[taste_idx]) < num_trials:
#                 print(f"Mismatch in trial dimension for dataset {core_dataset_name}, taste index {taste_idx}")
#                 continue

#             for trial_idx in range(num_trials):
#                 # Retrieve the changepoints for this taste and trial
#                 trial_changepoints = changepoints[taste_idx][trial_idx]

#                 # Split data into sub-arrays (epochs) based on ms-based changepoints
#                 start_idx = 0
#                 for cp_idx, changepoint_ms in enumerate(trial_changepoints):
#                     # Find the closest RNN time index that matches or exceeds the changepoint in ms
#                     end_idx = next((i for i, t in enumerate(time_in_ms) if t >= changepoint_ms), num_time_steps)

#                     # Slice the data array from start_idx to end_idx
#                     epoch_data = data_array[taste_idx, start_idx:end_idx, :, trial_idx]

#                     # Create a DataFrame for this epoch
#                     epoch_df = pl.DataFrame(
#                         epoch_data,  # No need to transpose as time is now the first dimension
#                         schema=[f'latent_dim_{i}' for i in range(latent_dim)]
#                     )

#                     # Add metadata columns for taste, trial, changepoint index, and time in ms
#                     epoch_df = epoch_df.with_columns([
#                         pl.Series("taste", [taste_idx] * len(epoch_df)),
#                         pl.Series("trial", [trial_idx] * len(epoch_df)),
#                         pl.Series("changepoint", [cp_idx] * len(epoch_df)),
#                         pl.Series("time", time_in_ms[start_idx:end_idx])
#                     ])

#                     # Append the epoch DataFrame to the list
#                     all_epochs.append(epoch_df)

#                     # Update start_idx to the current changepoint for the next epoch
#                     start_idx = end_idx

#                 # Handle any remaining time after the last changepoint as the final epoch
#                 if start_idx < num_time_steps:
#                     epoch_data = data_array[taste_idx, start_idx:num_time_steps, :, trial_idx]

#                     epoch_df = pl.DataFrame(
#                         epoch_data,
#                         schema=[f'latent_dim_{i}' for i in range(latent_dim)]
#                     )
#                     epoch_df = epoch_df.with_columns([
#                         pl.Series("taste", [taste_idx] * len(epoch_df)),
#                         pl.Series("trial", [trial_idx] * len(epoch_df)),
#                         pl.Series("changepoint", [len(trial_changepoints)] * len(epoch_df)),
#                         pl.Series("time", time_in_ms[start_idx:num_time_steps])
#                     ])
#                     all_epochs.append(epoch_df)

#         # Concatenate all epochs for the current dataset
#         dataset_epoch_df = pl.concat(all_epochs)

#         # Store the concatenated DataFrame in the dictionary
#         epoch_dataframes_dict[dataset_name] = dataset_epoch_df


#####  Running PCA on this RNN data now-- building this directly into the class as this is what we need to be working with:


# Original PCA on RNN function that I've built into this class:
# def run_robust_pca_analysis_to_dataframe(epoch_dataframes_dict):
#     """
#     Perform PCA on concatenated data grouped by taste and changepoint, then split results back accordingly.
#     Returns two dictionaries:
#     1. pca_lat_dict_95 - Dictionary storing PCs up to 95% explained variance.
#     2. pca_lat_dict_full - Dictionary storing all PCs without thresholding.

#     Parameters:
#     - epoch_dataframes_dict (dict): Dictionary where keys are dataset names and values are polars DataFrames.

#     Returns:
#     - pca_lat_dict_95 (dict)
#     - pca_lat_dict_full (dict)
#     """
#     pca_lat_dict_95 = {}
#     pca_lat_dict_full = {}

#     for dataset_name, df in epoch_dataframes_dict.items():
#         unique_tastes = df['taste'].unique().to_list()
#         unique_changepoints = df['changepoint'].unique().to_list()

#         processed_dataframes_95 = []
#         processed_dataframes_full = []

#         for taste in unique_tastes:
#             for changepoint in unique_changepoints:
#                 filtered_df = df.filter((pl.col('taste') == taste) & (pl.col('changepoint') == changepoint))

#                 if filtered_df.is_empty():
#                     continue

#                 latent_columns = [col for col in filtered_df.columns if 'latent_dim_' in col]
#                 if not latent_columns:
#                     continue

#                 unique_trials = filtered_df['trial'].unique().to_list()
#                 trial_data_list = [
#                     filtered_df.filter(pl.col('trial') == trial).select(latent_columns).to_numpy()
#                     for trial in unique_trials
#                 ]

#                 concatenated_data = np.vstack(trial_data_list)

#                 if concatenated_data.shape[0] < 2:
#                     continue

#                 pca = PCA()
#                 transformed_data = pca.fit_transform(concatenated_data)
#                 explained_variance = pca.explained_variance_ratio_ * 100

#                 cumulative_variance = np.cumsum(explained_variance)
#                 num_pcs_95 = np.argmax(cumulative_variance > 95) + 1

#                 split_indices = np.cumsum([arr.shape[0] for arr in trial_data_list])[:-1]
#                 split_pca_data = np.split(transformed_data, split_indices)

#                 pc_names_full = [f"PC_{i+1}" for i in range(transformed_data.shape[1])]
#                 explained_var_cols_full = [f"explained_variance_{pc}" for pc in pc_names_full]

#                 pc_names_95 = [f"PC_{i+1}" for i in range(num_pcs_95)]
#                 explained_var_cols_95 = [f"explained_variance_{pc}" for pc in pc_names_95]

#                 for trial, trial_pca_data in zip(unique_trials, split_pca_data):
#                     trial_metadata = filtered_df.filter(pl.col('trial') == trial).select(["trial", "time"])

#                     full_pca_df = pl.DataFrame(
#                         np.hstack([trial_pca_data, np.tile(explained_variance, (len(trial_pca_data), 1))]),
#                         schema=pc_names_full + explained_var_cols_full
#                     ).with_columns([
#                         pl.lit(taste).alias("taste"),
#                         pl.lit(changepoint).alias("changepoint")
#                     ])
#                     full_pca_df = pl.concat([trial_metadata, full_pca_df], how="horizontal")
#                     processed_dataframes_full.append(full_pca_df)

#                     pca_df_95 = pl.DataFrame(
#                         np.hstack([
#                             trial_pca_data[:, :num_pcs_95],
#                             np.tile(explained_variance[:num_pcs_95], (len(trial_pca_data), 1))
#                         ]),
#                         schema=pc_names_95 + explained_var_cols_95
#                     ).with_columns([
#                         pl.lit(taste).alias("taste"),
#                         pl.lit(changepoint).alias("changepoint")
#                     ])
#                     pca_df_95 = pl.concat([trial_metadata, pca_df_95], how="horizontal")
#                     processed_dataframes_95.append(pca_df_95)

#         def align_dataframes(dataframes):
#             all_columns = set()
#             for df in dataframes:
#                 all_columns.update(df.columns)
#             all_columns = sorted(all_columns)

#             aligned_dfs = []
#             for df in dataframes:
#                 missing_columns = set(all_columns) - set(df.columns)
#                 for col in missing_columns:
#                     df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
#                 df = df.select(all_columns)
#                 aligned_dfs.append(df)
#             return aligned_dfs

#         if processed_dataframes_95:
#             aligned_95 = align_dataframes(processed_dataframes_95)
#             pca_lat_dict_95[dataset_name] = pl.concat(aligned_95)
#         if processed_dataframes_full:
#             aligned_full = align_dataframes(processed_dataframes_full)
#             pca_lat_dict_full[dataset_name] = pl.concat(aligned_full)

#     return pca_lat_dict_95, pca_lat_dict_full
# robust_rnn_pca_95, robust_rnn_pca_full = run_robust_pca_analysis_to_dataframe(epoch_dataframes_dict)


# diff funcs (deprecated in favor of this class)
## following up on the 3/3 notes:

# def compute_first_diff(data_dict):
#     """
#     Takes a dictionary of Polars DataFrames and computes the first difference
#     (using np.diff) for each column that starts with 'latent_dim'. The metadata
#     columns (e.g. 'taste', 'trial', 'changepoint', 'time') are kept unchanged.

#     Parameters:
#         data_dict (dict): Dictionary where keys are names and values are Polars DataFrames.

#     Returns:
#         dict: New dictionary with the same keys, where each DataFrame has the first differences
#               computed for the latent columns and metadata columns preserved.
#     """
#     diff_dict = {}

#     for key, df in data_dict.items():
#         # Identify the latent columns (assumed to be the ones starting with "latent_dim")
#         latent_cols = [col for col in df.columns if col.startswith("latent_dim")]
#         # The rest are assumed to be metadata columns
#         metadata_cols = [col for col in df.columns if col not in latent_cols]

#         diff_data = {}

#         # Compute np.diff for each latent column and prepend a NaN
#         for col in latent_cols:
#             col_arr = df[col].to_numpy()
#             # np.diff returns an array one element shorter than col_arr.
#             diff_arr = np.diff(col_arr)
#             # Prepend np.nan so that the output has the same length as the original column.
#             diff_arr = np.insert(diff_arr, 0, np.nan)
#             diff_data[col] = diff_arr

#         # Add metadata columns unchanged.
#         for col in metadata_cols:
#             diff_data[col] = df[col].to_numpy()

#         # Create a new Polars DataFrame and ensure the column order remains the same.
#         new_df = pl.DataFrame(diff_data)[df.columns]
#         diff_dict[key] = new_df

#     return diff_dict
# diff_rob_pca_95 = compute_first_diff(robust_rnn_pca_95) # check this output via plotting more
# diff_rnn = compute_first_diff(epoch_dataframes_dict) # this is broken lol fix it later

# ## Slightly more robust way to compute these differences:
# # First derivative:

# def compute_first_derivative(df: pl.DataFrame) -> pl.DataFrame:
#     """
#     Compute the first derivative of latent dimension columns in a Polars DataFrame.
#     Assumes uniform time spacing (ignores actual 'time' values).
#     Preserves metadata columns and pads the first and last rows with NaN.
#     """
#     # Define metadata columns to preserve
#     meta_cols = ["taste", "trial", "changepoint", "time"]
#     # Identify latent dimension columns by excluding metadata
#     latent_cols = [col for col in df.columns if col not in meta_cols]

#     # Convert latent columns to a NumPy array for gradient calculation
#     data = df.select(latent_cols).to_numpy()

#     # Compute first derivative along axis=0 (time axis) with uniform spacing
#     first_deriv = np.gradient(data, axis=0)

#     # Pad the edges (first and last rows) with NaN to maintain length consistency-- causing artifacting?
#     #first_deriv[0, :] = np.nan
#     #first_deriv[-1, :] = np.nan

#     # Create a new DataFrame with the same structure: metadata + first derivative values
#     result_df = df.with_columns([
#         pl.Series(name=col, values=first_deriv[:, j])
#         for j, col in enumerate(latent_cols)
#     ])
#     return result_df
# # applying this: (change names as needed)
# first_der_dict = {}
# for key, df in robust_rnn_pca_95.items(): # be sure to have this defined but have a better way to do so ultimately
#     first_deriv_df = compute_first_derivative(df)
#     first_der_dict[key] = first_deriv_df

# # Second derivative:
# def compute_second_derivative(df: pl.DataFrame) -> pl.DataFrame:
#     """
#     Compute the second derivative of latent dimension columns in a Polars DataFrame.
#     Assumes uniform time spacing and preserves metadata columns.
#     Pads the first and last rows with NaN.
#     """
#     meta_cols = ["taste", "trial", "changepoint", "time"]
#     latent_cols = [col for col in df.columns if col not in meta_cols]

#     # Convert latent data to NumPy array for gradient calculations
#     data = df.select(latent_cols).to_numpy()

#     # Compute first derivative (includes edges via one-sided differences)
#     first_deriv = np.gradient(data, axis=0)
#     # Compute second derivative from the first derivative
#     second_deriv = np.gradient(first_deriv, axis=0)

#     # Pad edges with NaN to maintain original length -- causing artifacting?
#    # second_deriv[0, :] = np.nan
#    # second_deriv[-1, :] = np.nan

#     # Create new DataFrame with metadata + second derivative values
#     result_df = df.with_columns([
#         pl.Series(name=col, values=second_deriv[:, j])
#         for j, col in enumerate(latent_cols)
#     ])
#     return result_df

# # application
# second_der_dict = {}
# for key, df in robust_rnn_pca_95.items(): # be sure to have this defined but have a better way to do so ultimately
#     second_deriv_df = compute_second_derivative(df)
#     second_der_dict[key] = second_deriv_df


## a fully deprecated, statistically poor-power function:  (but I want to preserve it)

# # UPDATED SCRIPT: runs pca on the inferred dims and then saves them from there-- probably highly inefficent but we go from here
# def run_pca_analysis_to_dataframe(epoch_dataframes_dict, tot_var):
#     """
#     Perform PCA on epoch data and return a dictionary of DataFrames containing PCs up to 95% explained variance.

#     Parameters:
#     - epoch_dataframes_dict (dict): Dictionary where keys are dataset names and values are polars DataFrames.

#     Returns:
#     - result_dict (dict): Dictionary with the same keys as input, containing processed polars DataFrames.
#     """
#     pca_lat_dict = {}

#     for dataset_name, df in epoch_dataframes_dict.items():
#         # Extract unique tastes and changepoints from the dataset
#         unique_tastes = df['taste'].unique().to_list()
#         unique_changepoints = df['changepoint'].unique().to_list()

#         # Initialize a list to store processed DataFrames for each combination
#         processed_dataframes = []

#         for taste in unique_tastes:
#             for changepoint in unique_changepoints:
#                 # Filter the DataFrame for the specific taste and changepoint
#                 filtered_df = df.filter((pl.col('taste') == taste) & (pl.col('changepoint') == changepoint))

#                 if filtered_df.is_empty():
#                     continue

#                 # Extract unique trials
#                 unique_trials = filtered_df['trial'].unique().to_list()

#                 for trial in unique_trials:
#                     # Filter data for the specific trial
#                     trial_df = filtered_df.filter(pl.col('trial') == trial)

#                     # Extract latent dimensions for PCA
#                     latent_columns = [col for col in trial_df.columns if 'latent_dim_' in col]
#                     latent_data = trial_df.select(latent_columns).to_numpy()

#                     if latent_data.shape[0] < 2:
#                         # Skip if there's insufficient data for PCA
#                         continue

#                     # Run PCA
#                     pca = PCA()
#                     pca.fit(latent_data)

#                     # Get explained variance percentage for each PC
#                     explained_variance = pca.explained_variance_ratio_ * 100

#                     # Find the number of PCs capturing >95% variance
#                     cumulative_variance = np.cumsum(explained_variance)
#                     num_pcs_95 = np.argmax(cumulative_variance > tot_var) + 1

#                     # Extract the PCs needed for 95% variance
#                     required_pcs = pca.transform(latent_data)[:, :num_pcs_95]
#                     pc_names = [f"PC_{i+1}" for i in range(num_pcs_95)]

#                     # Combine PC values and explained variance into a single DataFrame
#                     combined_data = np.hstack([
#                         required_pcs,
#                         np.tile(explained_variance[:num_pcs_95], (len(required_pcs), 1))
#                     ])

#                     combined_columns = pc_names + [f"explained_variance_{pc}" for pc in pc_names]

#                     pc_df = pl.DataFrame(
#                         combined_data,
#                         schema=combined_columns
#                     )

#                     # Add metadata columns for taste, trial, changepoint, and time
#                     pc_df = pc_df.with_columns([
#                         pl.Series("taste", [taste] * len(pc_df)),
#                         pl.Series("trial", [trial] * len(pc_df)),
#                         pl.Series("changepoint", [changepoint] * len(pc_df)),
#                         trial_df["time"]  # Retain the original time column
#                     ])

#                     # Append the processed DataFrame to the list
#                     processed_dataframes.append(pc_df)

#         # Concatenate all processed DataFrames for the current dataset
#         if processed_dataframes:
#             # Collect all unique column names
#             all_columns = set()
#             for df in processed_dataframes:
#                 all_columns.update(df.columns)
#             all_columns = sorted(all_columns)

#             # Align all DataFrames to have the same columns
#             aligned_dataframes = []
#             for df in processed_dataframes:
#                 missing_columns = set(all_columns) - set(df.columns)
#                 for col in missing_columns:
#                     df = df.with_columns(pl.Series(col, [np.nan] * len(df)))
#                 df = df.select(all_columns)
#                 aligned_dataframes.append(df)

#             # Concatenate the aligned DataFrames
#             pca_lat_dict[dataset_name] = pl.concat(aligned_dataframes)

#     return pca_lat_dict
# pca_rnn_95pct = run_pca_analysis_to_dataframe(epoch_dataframes_dict, tot_var = 95) # deprecated-- there's a more robust way of doing this
