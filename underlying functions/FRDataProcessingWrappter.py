#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:52:54 2025

@author: vincentcalia-bogan

Wrapper on the exisitng firing rate class processor that modernizes the codebase a little bit. 
This should really be merged into the firing rate processor itself--
But this is such critical infastructure that this is how it has to be right now. 
"""

import os
import pandas as pd
import polars as pl
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d


from extract_npz import extract_from_npz
from unpkl_generator import extract_valid_changepoints
from find_extract_info import modify_tastes, process_info_files
from calc_fr_class_war_unwar import CalcFRStates


class FRDataProcessor:
    def __init__(
        self,
        npz_path: str,
        info_path: str,
        pkl_path: str,
        taste_replacements: dict,
        output_dir: str,
        window_length: int = 250,
        step_size: int = 25,
        fixed_warp_duration: int = 1000,
    ):
        self.npz_path = npz_path
        self.info_path = info_path
        self.pkl_path = pkl_path
        self.taste_replacements = taste_replacements
        self.output_dir = output_dir

        self.window_length = window_length
        self.step_size = step_size
        self.fixed_warp_duration = fixed_warp_duration

        # will hold 2D changepoint arrays
        self.changepoints_dict: dict[str, np.ndarray] = {}

        # will hold raw DataArrays per dataset
        self.results_xr: dict[str, dict[str, xr.DataArray]] = {}

        # will hold Polars DataFrames per dataset
        self.fr_unwarped_df: dict[str, pl.DataFrame] = {}
        self.spikes_unwarped_df: dict[str, pl.DataFrame] = {}
        self.fr_warped_df: dict[str, pl.DataFrame] = {}
        self.spikes_warped_df: dict[str, pl.DataFrame] = {}

    def extract_changepoints(self):
        for data in extract_from_npz(self.npz_path):
            if not (isinstance(data, tuple) and len(data) == 4):
                continue
            spike_array, dataset_num, index, key = data
            ds_name = f"dataset_{dataset_num}"

            # Optional taste processing
            tastes = process_info_files(self.info_path, dataset_num)
            _ = modify_tastes(tastes, self.taste_replacements)

            pkl = extract_valid_changepoints(
                self.pkl_path, spike_array, dataset_num, index, key
            )
            if pkl is None:
                continue
            try:
                cps = pkl[:, 3]
                self.changepoints_dict[ds_name] = cps
            except Exception as e:
                print(f"Could not extract CPS for {ds_name}: {e}")

    def _ensure_dirs(self):
        for sub in ("fr_unwarped","fr_warped","spikes_unwarped","spikes_warped"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

# considerably faster version of what came before 

    def _dataarray_to_polars(self, da: xr.DataArray, ms_offset: int) -> pl.DataFrame:
        """
        Fast NumPy → Polars. 
        Assumes da.dims = [taste, trial, segment, neuron, time_bin|time].
        """
        arr = da.values
        T, R, S, N, L = arr.shape
        total = T * R * S * L

        data = {f"neuron_{i}": arr[..., i, :].reshape(total) for i in range(N)}

        # metadata
        tastes   = np.repeat(np.arange(T),      R*S*L)
        one_taste_trials = np.repeat(np.arange(R), S*L)
        trials   = np.tile(one_taste_trials,    T)
        one_trial_segs  = np.repeat(np.arange(S), L)
        segments = np.tile(one_trial_segs,      T*R)

        data["taste"]   = tastes
        data["trial"]   = trials
        data["segment"] = segments

        # time in ms
        if da.dims[-1] == "time_bin":
            bins = np.tile(np.arange(L), T*R*S)
            data["time"] = ms_offset + bins * self.step_size
        else:
            tvec = da.coords["time"].values  # length L
            data["time"] = np.tile(tvec, T*R*S)

        return pl.DataFrame(data)

    def process_all(self):
        self._ensure_dirs()

        for data in extract_from_npz(self.npz_path):
            if not (isinstance(data, tuple) and len(data) == 4):
                continue
            spike_array, dataset_num, index, key = data
            ds_name = f"dataset_{dataset_num}"
            clean_name = ds_name.replace("dataset_", "").split("_repacked.npz")[0]
            if ds_name not in self.changepoints_dict:
                continue

            cps = self.changepoints_dict[ds_name]
            calc = CalcFRStates(
                spike_array=spike_array,
                changepoints=cps,
                window_length=self.window_length,
                step_size=self.step_size,
                compute_unwarped_spike_arrays=True,
                compute_unwarped_firing_rates=True,
                compute_warped_spike_arrays=True,
                compute_warped_firing_rates=True,
                fixed_warp_duration=self.fixed_warp_duration,
            )
            fr_unw, spikes_unw, fr_w, spikes_w = calc.run()

            # store raw DataArrays
            self.results_xr[clean_name] = {
                "fr_unwarped":   fr_unw,
                "spikes_unwarped": spikes_unw,
                "fr_warped":     fr_w,
                "spikes_warped":   spikes_w,
            }

            # convert & save each to disk + to our df‐dicts
            mapping = [
                ("fr_unwarped",   fr_unw,     self.fr_unwarped_df,   "fr_unwarped"),
                ("spikes_unwarped", spikes_unw, self.spikes_unwarped_df, "spikes_unwarped"),
                ("fr_warped",     fr_w,       self.fr_warped_df,     "fr_warped"),
                ("spikes_warped",   spikes_w,   self.spikes_warped_df,   "spikes_warped"),
            ]
            for key, da, df_dict, subdir in mapping:
                df = self._dataarray_to_polars(da, ms_offset=1500)
                # store in memory
                df_dict[clean_name] = df
                # write to disk
                out = os.path.join(self.output_dir, subdir, f"{clean_name}_{key}.parquet")
                df.write_parquet(out)

    def run_full(self):
        """
        1) extract changepoints
        2) process & save everything
        3) return all 8 objects
        """
        self.extract_changepoints()
        self.process_all()

        # unpack raw DataArrays into four dicts
        fr_unwarped_xr   = {ds: v["fr_unwarped"]   for ds, v in self.results_xr.items()}
        spikes_unwarped_xr = {ds: v["spikes_unwarped"] for ds, v in self.results_xr.items()}
        fr_warped_xr     = {ds: v["fr_warped"]     for ds, v in self.results_xr.items()}
        spikes_warped_xr   = {ds: v["spikes_warped"]   for ds, v in self.results_xr.items()}

        # the DataFrame dicts are already self.fr_unwarped_df, etc.
        return (
            fr_unwarped_xr,
            spikes_unwarped_xr,
            fr_warped_xr,
            spikes_warped_xr,
            self.fr_unwarped_df,
            self.spikes_unwarped_df,
            self.fr_warped_df,
            self.spikes_warped_df,
        )


# e.g.
# fr_unw_xr["dataset_42"]       → xarray.DataArray
# fr_unw_df["dataset_42"]       → polars.DataFrame
