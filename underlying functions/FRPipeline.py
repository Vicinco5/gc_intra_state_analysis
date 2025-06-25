#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:52:39 2025

Attempting to modernize my firing rate calculation class so I can finally get to re-making some plots. Ugh this is taking far too long.

@author: vincentcalia-bogan
"""
import os
import pickle
from pathlib import Path
import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from extract_npz import extract_from_npz
from unpkl_generator import extract_valid_changepoints
from find_extract_info import modify_tastes, process_info_files

class FRPipeline:
    """
    Unified pipeline: extract changepoints, compute spike segments & firing rates,
    accumulate results directly into Polars DataFrames, and optionally save.
    """

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
        start_time: int = 1500,
        end_time: int | None = None,
        save_outputs: bool = False,
    ):
        self.npz_path = npz_path
        self.info_path = info_path
        self.pkl_path = pkl_path
        self.taste_replacements = taste_replacements
        self.output_dir = output_dir
        self.window_length = window_length
        self.step_size = step_size
        self.fixed_warp_duration = fixed_warp_duration
        self.start_time = start_time
        self.end_time = end_time
        self.save_outputs = save_outputs

        self.changepoints_dict: dict[str, np.ndarray] = {}
        self.fr_unwarped_df: dict[str, pl.DataFrame] = {}
        self.spikes_unwarped_df: dict[str, pl.DataFrame] = {}
        self.fr_warped_df: dict[str, pl.DataFrame] = {}
        self.spikes_warped_df: dict[str, pl.DataFrame] = {}

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
                        changepoints = extracted_pkl[:, 3]  # Keep full array
                        self.changepoints_dict[dataset_name] = changepoints
                    except IndexError:
                        print(
                            f"Index error with dataset {dataset_name}: shape {extracted_pkl.shape}"
                        )
                else:
                    print(f"No valid changepoints found for dataset {dataset_name}")

    def extract_changepoints_dict(self, save_outputs: bool = False) -> dict:
        """
        Extract only the changepoints for each dataset (populates self.changepoints_dict).
        Optionally saves them under <output_dir>/changepoints.
        """
        # (re)initialize
        self.changepoints_dict = {}
        # delegate to extract_changepoints
        self.extract_changepoints()

        if save_outputs:
            save_path = Path(self.output_dir) / "changepoints"
            save_path.mkdir(parents=True, exist_ok=True)
            for dataset_name, changepoints in self.changepoints_dict.items():
                clean_name = dataset_name.replace("dataset_", "").split("_repacked.npz")[0]
                out_path = save_path / f"{clean_name}_changepoints.pkl"
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        pickle.dump(self.changepoints_dict, f)

        return self.changepoints_dict

    def _ensure_dirs(self):
        for sub in ("rr_firing_unwarped", "spikes_unwarped", "rr_firing_warped", "spikes_warped"):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def _calc_fr_rr(self, seg: np.ndarray) -> np.ndarray:
        wl, ss = self.window_length, self.step_size
        nrn, tlen = seg.shape
        nbins = max((tlen - wl) // ss + 1, 1)
        fr = np.zeros((nrn, nbins))
        for b in range(nbins):
            s = b * ss
            e = min(s + wl, tlen)
            fr[:, b] = seg[:, s:e].sum(axis=1) / (wl / 1000.0)
        return fr

    def process_all(self):
        """
        Core processing: for each dataset, split epochs by changepoints,
        directly build Polars DataFrames, and save if requested.
        """
        self._ensure_dirs()
        for data in extract_from_npz(self.npz_path):
            if not (isinstance(data, tuple) and len(data) == 4):
                continue
            spike_array, ds_num, idx, key = data
            ds_name = f"dataset_{ds_num}"
            clean = ds_name.replace("dataset_", "").split("_repacked.npz")[0]
            if ds_name not in self.changepoints_dict:
                continue
            cps_all = self.changepoints_dict[ds_name]

            spu_list, fru_list, spw_list, frw_list = [], [], [], []
            T, R, N, _ = spike_array.shape

            for t in range(T):
                if t >= len(cps_all):
                    continue  # taste-trial mismatch
                taste_cps = cps_all[t]
                if not hasattr(taste_cps, '__len__') or len(taste_cps) != R:
                    continue

                for tr in range(R):
                    trial_cps = np.array(taste_cps[tr], dtype=float)
                    if np.isnan(trial_cps).any():
                        continue
                    start = int(self.start_time)

                    for si, cp in enumerate(trial_cps):
                        raw_end = int(cp)
                        if self.end_time is not None: 
                            end = min(raw_end, self.end_time)
                        else:
                            end = raw_end
                        if end <= start:
                            continue
                        segment = spike_array[t, tr][:, start:end]

                        # unwarped spikes
                        if segment.size:
                            times = list(range(start, start + segment.shape[1]))
                            spu_list.append(
                                pl.DataFrame({
                                    **{f"neuron_{i}": segment[i].tolist() for i in range(N)},
                                    "taste": [t] * segment.shape[1],
                                    "trial": [tr] * segment.shape[1],
                                    "changepoint": [si] * segment.shape[1],
                                    "time": times,
                                })
                            )

                            # unwarped firing rates
                            fr_seg = self._calc_fr_rr(segment)
                            fr_times = [start + b * self.step_size for b in range(fr_seg.shape[1])]
                            fru_list.append(
                                pl.DataFrame({
                                    **{f"neuron_{i}": fr_seg[i].tolist() for i in range(N)},
                                    "taste": [t] * fr_seg.shape[1],
                                    "trial": [tr] * fr_seg.shape[1],
                                    "changepoint": [si] * fr_seg.shape[1],
                                    "time": fr_times,
                                })
                            )
                            # warped firing rates
                            if fr_seg.shape[1] > 1:
                                # ms-length of the epoch you want to warp to
                                wlen_ms = self.fixed_warp_duration or int(cp - start)
                                # how many FR-bins that becomes, given your wl/ss
                                nbins_w = max((wlen_ms - self.window_length) // self.step_size + 1, 1) # double check the +1 here
                                # build normalized axes for interpolation
                                orig_axes = np.linspace(0, 1, fr_seg.shape[1])
                                target_axes = np.linspace(0, 1, nbins_w)
                                # interpolate each neuron’s firing‐rate curve
                                frw_seg = np.vstack([
                                    interp1d(orig_axes, fr_seg[n], kind='nearest',
                                             bounds_error=False, fill_value='extrapolate')(target_axes)
                                    for n in range(N)
                                ])
                                # recompute the “time” column for these warped‐FR bins
                                frw_times = [start + b * self.step_size for b in range(nbins_w)]
                                frw_list.append(
                                    pl.DataFrame({
                                        **{f"neuron_{i}": frw_seg[i].tolist() for i in range(N)},
                                        "taste":       [t] * nbins_w,
                                        "trial":       [tr] * nbins_w,
                                        "changepoint": [si] * nbins_w,
                                        "time":        frw_times,
                                    })
                                )
                            # warped segment & rates
                            if segment.shape[1] > 1:
                                norm = np.linspace(0, 1, segment.shape[1])
                                wlen = self.fixed_warp_duration or int(cp - start)
                                wnorm = np.linspace(0, 1, wlen)
                                war_seg = np.vstack([
                                    interp1d(norm, segment[n], kind='nearest', bounds_error=False, fill_value='extrapolate')(wnorm)
                                    for n in range(N)
                                ])
                                wtimes = list(range(start, start + war_seg.shape[1]))
                                spw_list.append(
                                    pl.DataFrame({
                                        **{f"neuron_{i}": war_seg[i].tolist() for i in range(N)},
                                        "taste": [t] * war_seg.shape[1],
                                        "trial": [tr] * war_seg.shape[1],
                                        "changepoint": [si] * war_seg.shape[1],
                                        "time": wtimes,
                                    })
                                ) 
                                # potential error: I am warping the spike trains and then doing rolling window on it. 
                                # I may want to do the warp directly on the rolling window firing rate for the given epoch instead, as 
                                # the current method is not acceptible when it comes to outliers. 
                                # frw_seg = self._calc_fr_rr(war_seg)
                                # frw_times = [start + b * self.step_size for b in range(frw_seg.shape[1])]
                                # frw_list.append(
                                #     pl.DataFrame({
                                #         **{f"neuron_{i}": frw_seg[i].tolist() for i in range(N)},
                                #         "taste": [t] * frw_seg.shape[1],
                                #         "trial": [tr] * frw_seg.shape[1],
                                #         "changepoint": [si] * frw_seg.shape[1],
                                #         "time": frw_times,
                                #     })
                                # )

                        start = end

            # concatenate and save with the analysis type
            analysis_type = ("rr_firing_unwarped", "spikes_unwarped", "rr_firing_warped", "spikes_warped")
            if spu_list:
                self.spikes_unwarped_df[f"{clean}_{analysis_type[1]}"] = pl.concat(spu_list)
            if fru_list:
                self.fr_unwarped_df[f"{clean}_{analysis_type[0]}"] = pl.concat(fru_list)
            if spw_list:
                self.spikes_warped_df[f"{clean}_{analysis_type[3]}"] = pl.concat(spw_list)
            if frw_list:
                self.fr_warped_df[f"{clean}_{analysis_type[2]}"] = pl.concat(frw_list)

            if self.save_outputs:
                for name, df in [
                    ('spikes_unwarped', self.spikes_unwarped_df.get(f"{clean}_{analysis_type[1]}")),
                    ('rr_firing_unwarped',     self.fr_unwarped_df.get(f"{clean}_{analysis_type[0]}")), 
                    # bug: seems to be stepping at odd intervals for unwarped firing; which shouldn't be happening. not sure why, will investigate later.
                    ('spikes_warped',   self.spikes_warped_df.get(f"{clean}_{analysis_type[3]}")),
                    ('rr_firing_warped',       self.fr_warped_df.get(f"{clean}_{analysis_type[2]}")),
                ]:
                    if df is not None:
                        path = os.path.join(self.output_dir, name, f"{clean}_{name}.parquet")
                        df.write_parquet(path)

    def full_pipeline(self) -> tuple[dict, dict, dict, dict]:
        """
        Run extract_changepoints and process_all in one call.
        Returns four dicts: fr_unwarped_df, spikes_unwarped_df, fr_warped_df, spikes_warped_df
        """
        self.extract_changepoints_dict(self.save_outputs)
        self.process_all()
        return (
            self.fr_unwarped_df,
            self.spikes_unwarped_df,
            self.fr_warped_df,
            self.spikes_warped_df,
        )



    # below is an updated, much more optimized version of the original method that I wrote over 2 years ago
    # but has to be validated before being pushed. 
    #
    
    # def calc_fr_rr(self, state_spike_array, window_length, step_size):
    #     """
    #     Optimized version of the original nested rolling window method, using cumulative sum and vectorization. 
    #     Computes the firing rate in Hz for each neuron using a sliding window.
    #     """
    #     nrn_num, time_num = state_spike_array.shape
    #     num_bins = max((time_num - window_length) // step_size + 1, 1)
    
    #     # build cumulative sum along time (pad with zero at t=0)
    #     cs = np.zeros((nrn_num, time_num + 1), dtype=state_spike_array.dtype)
    #     cs[:, 1:] = np.cumsum(state_spike_array, axis=1)
    
    #     # start/end indices for each window
    #     starts = np.arange(num_bins) * step_size
    #     ends   = np.minimum(starts + window_length, time_num)
    
    #     # counts per window = cs[:, end] - cs[:, start]
    #     counts = cs[:, ends] - cs[:, starts]
    
    #     # convert to Hz (same divisor as original)
    #     return counts / (window_length / 1000.0)

