#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:37:47 2025

@author: vincentcalia-bogan

firing_rate_plotter
"""
"""
Plot firing‑rate heatmaps **and** non‑stationarity‑vs‑reliability scatter plots.

New (June 2025)
----------------
• ``compute_effect_icc`` – per‑neuron *effect size* (Cohen‑d vs population) & ICC.
• ``plot_effect_vs_icc`` – scatter of Effect (y) vs Inter‑trial ICC (x).
• ``effect_similarity_all`` – convenience wrapper across all tastes.

*Effect size definition*
------------------------
For the chosen *warped* segment(s):
    * **Group A** = all firing‑rate samples of the neuron of interest across trials & time‑bins.
    * **Group B** = all firing‑rate samples of **all other neurons** (same taste, same segment(s)).
We compute **Cohen’s d** between A and B (signed), giving a measure of how strongly that
neuron’s activity deviates from the population mean within the processing state.

The x‑axis reliability metric remains an ICC(1) across trials for that neuron.

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


# new one:
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import matplotlib.lines as mlines

# 6/10-- additional functionality for non-stationarity vs inter-trial similarity: 
# Try to import scikit‑learn for AUC; fall back to a minimal implementation if absent
try:
    from sklearn.metrics import roc_auc_score  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    def roc_auc_score(y_true, y_score):  # minimalist, ties handled by ranking-- not as good as Scikit 
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) != 2:
            return np.nan
        # Rank the scores (average ranks for ties)
        order = y_score.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(y_score)) + 1  # 1‑based ranks
        pos = y_true == 1
        n_pos = pos.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return auc
    # is all of the above really strictly needed?

class FiringRatePlotter:
    def __init__(
        self,
        fr_data_array: xr.DataArray,
        dataset_dir: str,
        dataset_name: str,
        base_name: str,
        is_warped: str,                # True, False, or 'whole-data'
        mode: str,                     # 'cell-wise', 'trial-wise', 'debug'
        alignment: str | None = None,
        sort_by_length: bool = False,
        changepoints_dict: dict | None = None,
        *,
        start_time: int = 1500,
        end_time: int = 4500,
    ) -> None:
        self.data = fr_data_array
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.base_name = base_name
        self.is_warped = is_warped
        self.mode = mode
        self.alignment = alignment
        self.sort_by_length = sort_by_length
        self.changepoints_dict = changepoints_dict or {} # raw nested dict
        self.start_time = start_time
        self.end_time = end_time
        os.makedirs(self.dataset_dir, exist_ok=True)

    #-----------------------------------------------------
    # Changepoint extraction & safe lookup
    #-----------------------------------------------------
    def extract_changepoints(self, key: str):
        cp_root = self.changepoints_dict.get(key)
        if cp_root is None:
            return None
        def clip_obj(obj):
            if isinstance(obj, dict):
                return {k: clip_obj(v) for k,v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clip_obj(v) for v in obj]
            arr = np.asarray(obj, float)
            return np.clip(arr, self.start_time, self.end_time)
        return clip_obj(cp_root)

    def _get_cps(self, taste_idx: int, trial_idx: int | None = None) -> list:
        cps = self.extract_changepoints(self.dataset_name)
        if cps is None:
            return []
        # cps can be dict[taste]->dict[trial]->list or dict[taste]->list or list of arrays
        if isinstance(cps, dict):
            taste_obj = cps.get(taste_idx)
            if taste_obj is None:
                return []
            if isinstance(taste_obj, dict):
                if trial_idx is None:
                    trial_idx = next(iter(taste_obj), None)
                return list(taste_obj.get(trial_idx, []))
            return list(taste_obj)
        if isinstance(cps, list):
            if trial_idx is None:
                return list(cps[0]) if cps else []
            if 0 <= trial_idx < len(cps):
                return list(cps[trial_idx])
        return []

    @staticmethod
    def cli_options():
        print("\n--- Firing Rate Plotting Configuration ---")
        mode = (
            input("Select plotting mode ('cell-wise' or 'trial-wise'): ")
            .strip()
            .lower()
        )
        while mode not in ["cell-wise", "trial-wise"]:
            mode = (
                input("Invalid input. Choose 'cell-wise' or 'trial-wise': ")
                .strip()
                .lower()
            )

        is_whole_data = (
            input("Use whole-data mode (no padding, full time)? (y/n): ")
            .strip()
            .lower()
            == "y"
        )
        is_warped = (
            "whole-data"
            if is_whole_data
            else input("Plot warped data? (y/n): ").strip().lower() == "y"
        )

        alignment = None
        sort_by_length = False

        if not is_whole_data and not is_warped:
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

            sort_by_length = (
                input("Sort vectors by length (only for unwarped)? (y/n): ")
                .strip()
                .lower()
                == "y"
            )

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
            
    def _icc1(mat: np.ndarray) -> float:
        """Intraclass correlation ICC(1) for *trials* (columns) repeated over *time* (rows).
        NaNs are ignored using pairwise deletion.
        """
        # mat shape: (n_timebins, n_trials)
        if mat.shape[1] < 2:
            return np.nan
        # Replace per‑column NaN means to keep unbiased sums
        col_means = np.nanmean(mat, axis=0)
        grand_mean = np.nanmean(col_means)
        # Between‑trial sum‑of‑squares
        ss_between = np.nansum((col_means - grand_mean) ** 2) * mat.shape[0]
        df_between = mat.shape[1] - 1
        # Within‑trial SS
        ss_within = np.nansum((mat - col_means) ** 2)
        df_within = np.nansum(~np.isnan(mat)) - mat.shape[1]
        if df_between == 0 or df_within == 0:
            return np.nan
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        denom = ms_between + (mat.shape[1] - 1) * ms_within
        return (ms_between - ms_within) / denom if denom != 0 else np.nan

    def compute_effect_icc(
        self,
        taste_idx: int,
        seg_idx: int | tuple[int, ...] = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``icc, effect, neuron_ids`` for a given taste and segment(s).

        *Effect* = Cohen‑d between neuron and population (excluding the neuron).
        ICC is computed over the same segment(s).
        """
        if self.is_warped is not True:
            raise ValueError("Effect/ICC analysis requires warped data (is_warped=True).")
        if isinstance(seg_idx, int):
            seg_idx = (seg_idx,)

        neurons = self.data.coords["neuron"].values
        trials = self.data.coords["trial"].values
        effects: list[float] = []
        iccs: list[float] = []

        # Pre‑gather population data for speed
        pop_cache: dict[int, np.ndarray] = {}
        for n in neurons:
            all_vals = []
            for s in seg_idx:
                seg_data = self.data.sel(taste=taste_idx, segment=s, neuron=n).values
                all_vals.append(seg_data.reshape(-1))
            pop_cache[n] = np.concatenate(all_vals).astype(float)
            pop_cache[n] = pop_cache[n][~np.isnan(pop_cache[n])]

        for neuron_idx in neurons:
            neuron_vals = pop_cache[neuron_idx]
            other_vals = np.concatenate([pop_cache[m] for m in neurons if m != neuron_idx])
            # Cohen's d
            n1, n2 = len(neuron_vals), len(other_vals)
            if n1 < 2 or n2 < 2:
                effects.append(np.nan)
            else:
                mean1, mean2 = neuron_vals.mean(), other_vals.mean()
                var1, var2 = neuron_vals.var(ddof=1), other_vals.var(ddof=1)
                pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                effects.append((mean1 - mean2) / pooled_sd if pooled_sd else np.nan)

            # ICC across trials (concatenate segment(s))
            trial_vecs = []
            for trial_idx in trials:
                seg_vecs = []
                for s in seg_idx:
                    vec = self.data.sel(
                        taste=taste_idx,
                        trial=trial_idx,
                        segment=s,
                        neuron=neuron_idx,
                    ).values.astype(float)
                    vec = vec[~np.isnan(vec)]
                    if vec.size:
                        seg_vecs.append(vec)
                if seg_vecs:
                    trial_vecs.append(np.concatenate(seg_vecs))
            if len(trial_vecs) >= 2:
                max_len = max(map(len, trial_vecs))
                mat = np.full((max_len, len(trial_vecs)), np.nan)
                for c, v in enumerate(trial_vecs):
                    mat[: len(v), c] = v
                iccs.append(self._icc1(mat))
            else:
                iccs.append(np.nan)

        return np.asarray(iccs), np.asarray(effects), neurons

    # Utilities ------------------------------------------------------------
    def _pad_align(self, rows, align_to):
        maxlen = max(len(r) for r in rows)
        padded = []
        for r in rows:
            pad = maxlen - len(r)
            if align_to=='start': padded.append(np.concatenate([r, np.full(pad, np.nan)]))
            elif align_to=='end': padded.append(np.concatenate([np.full(pad, np.nan), r]))
            else: padded.append(r)
        return np.array(padded)

    def _sort_by_length(self, vectors):
        lengths = [np.sum(~np.isnan(v)) for v in vectors]
        return np.argsort(lengths)[::-1]

    def _draw_changepoints(self, ax, cps, color="black", style="--"):
        for cp in cps:
            ax.axvline(x=cp, color=color, linestyle=style, linewidth=1)

    def _draw_stimulus(self, ax, stim_time=2000):
        ax.axvline(x=stim_time, color="black", linestyle=":", linewidth=1)

    def _save_figure(self, fig, path_base):
        fig.savefig(f"{path_base}.png", dpi=300)
        fig.savefig(f"{path_base}.svg", format="svg")
        plt.close(fig)

    #-----------------------------------------------------
    # Plotting: Segmented Cell-wise
    #-----------------------------------------------------
    def _plot_segmented_cellwise(self):
        for taste in self.data.coords['taste'].values:
            for seg in self.data.coords['segment'].values:
                neurons = self.data.coords['neuron'].values
                trials  = self.data.coords['trial'].values
                nrows   = math.ceil(len(neurons)/5)
                fig,axes=plt.subplots(nrows,5,figsize=(15,3*nrows),sharex=True,sharey=True)
                axes=axes.flatten()
                for i,n in enumerate(neurons):
                    ax=axes[i]
                    rows=[]
                    for t in trials:
                        v=self.data.sel(taste=taste,trial=t,segment=seg,neuron=n).values
                        rows.append(v if v.size else np.array([]))
                    if not self.is_warped and self.alignment:
                        rows=self._pad_align(rows,self.alignment)
                    if not self.is_warped and self.sort_by_length:
                        rows=[rows[j] for j in self._sort_by_length(rows)]
                    heat=np.full((len(rows),max(len(r) for r in rows)),np.nan)
                    for j,r in enumerate(rows): heat[j,:len(r)]=r
                    sns.heatmap(heat,ax=ax,cmap='viridis',cbar_kws={'label':'Firing Rate (Hz)'})
                    ax.set_title(f'Neuron {n}')
                    self._draw_stimulus(ax)
                    cps=self._get_cps(taste,int(trials[0]))
                    self._draw_changepoints(ax,cps)
                fig.suptitle(f'Cell-wise Heatmap – {self.dataset_name}, Taste {taste}, Seg {seg}')
                fig.tight_layout(rect=[0,0.05,1,0.95])
                legend_lines=[mlines.Line2D([],[],color='black',linestyle=':',label='Stimulus'),mlines.Line2D([],[],color='black',linestyle='--',label='Changepoint')]
                fig.legend(handles=legend_lines,loc='lower center',bbox_to_anchor=(0.5,0.02),ncol=2)
                out=os.path.join(self.dataset_dir,f'taste_{taste}')
                os.makedirs(out,exist_ok=True)
                self._save_figure(fig,os.path.join(out,f'FR_Cell_T{taste}_S{seg}_{self.dataset_name}'))

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
                        vec = self.data.sel(
                            taste=taste_idx,
                            trial=trial_idx,
                            segment=seg_idx,
                            neuron=neuron_idx,
                        ).values
                        if vec is not None and np.any(~np.isnan(vec)):
                            segments.append(vec)
                    if segments:
                        full_vec = np.concatenate(segments)
                        full_trials.append(full_vec)

                heat_data = np.full(
                    (len(full_trials), max(len(t) for t in full_trials)), np.nan
                )
                for j, trial in enumerate(full_trials):
                    heat_data[j, : len(trial)] = trial

                sns.heatmap(
                    heat_data,
                    ax=ax,
                    cmap="viridis",
                    cbar=True,
                    cbar_kws={"label": "Firing Rate (Hz)"},
                )
                ax.set_title(f"Neuron {neuron_idx}")
                self._draw_stimulus(ax)

                if self.changepoints_dict and isinstance(self.changepoints_dict, dict):
                    cps_dict = self.changepoints_dict.get(self.dataset_name, {}).get(
                        taste_idx, {}
                    )
                    if isinstance(cps_dict, dict) and len(cps_dict) > 0:
                        sample_trial = next(iter(cps_dict))
                        sample_cps = cps_dict.get(sample_trial, [])
                        self._draw_changepoints(ax, sample_cps)

            fig.suptitle(
                f"Whole-Data Cell-Wise Firing Heatmap\nTaste {taste_idx}, Dataset {self.dataset_name}",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            red_line = mlines.Line2D(
                [], [], color="black", linestyle=":", label="Stimulus"
            )
            blue_line = mlines.Line2D(
                [], [], color="black", linestyle="--", label="Changepoint"
            )
            fig.legend(
                handles=[red_line, blue_line],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
            )

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
                fig, axes = plt.subplots(
                    4, 1, figsize=(10, 12), sharex=True, sharey=True
                )
                fig_axes = axes if isinstance(axes, np.ndarray) else [axes]

                trial_subset = trials[chunk_start : chunk_start + 4]

                for ax_idx, trial_idx in enumerate(trial_subset):
                    ax = fig_axes[ax_idx]
                    all_vecs = []

                    for neuron_idx in neurons:
                        segments = []
                        for seg_idx in segs:
                            vec = self.data.sel(
                                taste=taste_idx,
                                trial=trial_idx,
                                segment=seg_idx,
                                neuron=neuron_idx,
                            ).values
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
                        heat_data[j, : len(row)] = row

                    extent = [time_vector[0], time_vector[-1], 0, heat_data.shape[0]]
                    im = ax.imshow(
                        heat_data,
                        aspect="auto",
                        interpolation="nearest",
                        cmap="plasma",
                        extent=extent,
                        origin="upper",
                    )

                    # Individual colorbar per subplot
                    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
                    cbar.set_label("Firing Rate (Hz)")

                    xticks = np.arange(time_start, time_vector[-1] + 1, 500)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(t) for t in xticks])
                    ax.tick_params(
                        axis="x", labelbottom=True
                    )  # Force labels on all subplots

                    yticks = np.arange(0, heat_data.shape[0], 5)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([str(y) for y in yticks])

                    ax.set_title(f"Trial {trial_idx+1}")
                    ax.set_ylabel("Neuron")
                    ax.set_xlabel("Time (ms)")

                # Draw changepoints on all axes
                for ax_idx, trial_idx in enumerate(trial_subset):
                    ax = fig_axes[ax_idx]
                    ax.axvline(
                        x=2000, color="black", linestyle=":", linewidth=3
                    )  # stimulus

                    try:
                        cps = self.changepoints_dict[self.dataset_name][taste_idx][
                            trial_idx
                        ]
                        for cp in cps:
                            if time_start <= cp <= time_vector[-1]:
                                ax.axvline(
                                    x=cp, color="black", linestyle="--", linewidth=3
                                )
                    except (KeyError, IndexError, TypeError):
                        pass

                fig.suptitle(
                    f"Whole-Data Trial-Wise Firing Heatmap\nTaste {taste_idx}, Dataset {self.dataset_name}",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])

                stim_line = mlines.Line2D(
                    [], [], color="black", linestyle=":", label="Stimulus"
                )
                cp_line = mlines.Line2D(
                    [], [], color="black", linestyle="--", label="Changepoint"
                )
                fig.legend(
                    handles=[stim_line, cp_line],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=2,
                )

                taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
                os.makedirs(taste_dir, exist_ok=True)
                chunk_label = f"chunk_{chunk_start // 4 + 1}"
                fname_base = f"wholedata_trialwise_taste_{taste_idx}_{self.dataset_name}_{chunk_label}"
                self._save_figure(fig, os.path.join(taste_dir, fname_base))
                
    def _plot_segmented_trialwise(self):
        for taste in self.data.coords['taste'].values:
            for seg in self.data.coords['segment'].values:
                neurons=self.data.coords['neuron'].values
                trials =self.data.coords['trial'].values
                nrows  =math.ceil(len(trials)/5)
                fig,axes=plt.subplots(nrows,5,figsize=(15,3*nrows),sharex=True,sharey=True)
                axes=axes.flatten()
                for i,t in enumerate(trials):
                    ax=axes[i]
                    rows=[]
                    for n in neurons:
                        v=self.data.sel(taste=taste,trial=t,segment=seg,neuron=n).values
                        rows.append(v if v.size else np.array([]))
                    if not self.is_warped and self.alignment:
                        rows=self._pad_align(rows,self.alignment)
                    if not self.is_warped and self.sort_by_length:
                        rows=[rows[j] for j in self._sort_by_length(rows)]
                    heat=np.full((len(rows),max(len(r) for r in rows)),np.nan)
                    for j,r in enumerate(rows): heat[j,:len(r)]=r
                    sns.heatmap(heat,ax=ax,cmap='viridis',cbar_kws={'label':'Firing Rate (Hz)'})
                    ax.set_title(f'Trial {t}')
                    self._draw_stimulus(ax)
                    cps=self._get_cps(taste,t)
                    self._draw_changepoints(ax,cps)
                fig.suptitle(f'Trial-wise Heatmap – {self.dataset_name}, Taste {taste}, Seg {seg}')
                fig.tight_layout(rect=[0,0.05,1,0.95])
                legend_lines=[mlines.Line2D([],[],color='black',linestyle=':',label='Stimulus'),mlines.Line2D([],[],color='black',linestyle='--',label='Changepoint')]
                fig.legend(handles=legend_lines,loc='lower center',bbox_to_anchor=(0.5,0.02),ncol=2)
                out=os.path.join(self.dataset_dir,f'taste_{taste}')
                os.makedirs(out,exist_ok=True)
                self._save_figure(fig,os.path.join(out,f'FR_Trial_T{taste}_S{seg}_{self.dataset_name}'))
    # ---------------------------------------------------------------------
    # New – Scatter plotting
    # ---------------------------------------------------------------------
    def plot_effect_vs_icc(
        self,
        seg_idx: int | tuple[int, ...] = 1,
        figsize: tuple[int, int] = (6, 6),
    ) -> None:
        if self.is_warped is not True:
            raise ValueError("Scatter plot requires warped data (is_warped=True).")
        if isinstance(seg_idx, int):
            seg_desc = f"Seg {seg_idx}"
        else:
            seg_desc = "Segs " + ",".join(map(str, seg_idx))
        for taste_idx in self.data.coords["taste"].values:
            iccs, effects, neurons = self.compute_effect_icc(taste_idx=taste_idx, seg_idx=seg_idx)
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(iccs, effects, alpha=0.7, edgecolor="k")
            ax.set_xlabel("Inter‑trial similarity (ICC)")
            ax.set_ylabel("Effect size (Cohen‑d vs population)")
            ax.set_title(f"Effect vs ICC\n{seg_desc} – Taste {taste_idx} – {self.dataset_name}")
            ax.axhline(0, color="gray", linestyle=":")
            ax.grid(True, linestyle=":", linewidth=0.5)
            taste_dir = os.path.join(self.dataset_dir, f"taste_{taste_idx}")
            os.makedirs(taste_dir, exist_ok=True)
            fname = f"effect_vs_icc_{seg_desc.replace(' ', '_')}_t{taste_idx}_{self.dataset_name}"
            self._save_figure(fig, os.path.join(taste_dir, fname))

    # ---------------------------------------------------------------------
    # Convenience wrapper (does not interfere with existing plot_all logic)
    # ---------------------------------------------------------------------
    def effect_similarity_all(self, seg_idx: int | tuple[int, ...] = 1) -> None:
        self.plot_effect_vs_icc(seg_idx=seg_idx)



## note-taking for new module of non stationarity (effect size) vs inter trial similarity (correlation across trials) for all neurons, exlusively warped 
# Effect size measure: either spearman P (which is pairwise) or actually, Area under ROC (with no formal distribution-- unsure if this is the right one)
# also see what abu and I just discussed 
# to control for warping artifacts-- which can do weird stuff-- I want to plot warp length vs simiularity as well 
# Inter-trial similariry: Mean pairwise spearman P? Cos similariry (a dot product)? Perhaps fano factor (no-- as this is a measure of relative variance)?
# Intra-class correlation coefficent? (ICC?)-- Let's stick to cos similarirty 
# ICC may assume equal variances, which is less good. But a place to start -- sticking to cos-similarity
# there are a lot of plots that can be made with this stuff; startign with this one. 
# starting with Area under ROC vs ICC plot-- as AUC should give a reliable measure of taste decode, while ICC is trial reliablilitY...?
# 




# lo and behold, this class has a lot of issues. fuck. will deal later. 

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
