#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:24:11 2025

@author: vincentcalia-bogan
"""

# new func: 
# this should be working and I am not sure why it's not-- gotta talk to abu what's going on 
# also, this kinda sucks ass, ngl -- it's because I totally overlooked one of the key assumptions when it comes to signficance testing
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, ks_2samp, kruskal

class SignificanceTester:
    """
Perform statistical significance testing (Welch’s t-test, KS test, Kruskal–Wallis, permutation test)
on latent/PC dimensions in Polars DataFrames, and save the results to text files.

Parameters
----------
data_dict : dict[str, pl.DataFrame]
    Mapping from dataset name to a Polars DataFrame containing at least the columns:
    ‘taste’, ‘changepoint’, ‘trial’, ‘time’, and any ‘PC_…’ or ‘latent_dim_…’ columns.
rng : numpy.random.Generator
    Random number generator for reproducible shuffling in permutation tests.
save_dir : str or pathlib.Path
    Directory in which to write the result text files.
n_permutations : int, default=10000
    Number of shuffles to perform in the permutation test.

Result files generated
----------------------
1. `<dataset_name>_significance_results.txt`
   Full log of every test (t-test, KS, Kruskal–Wallis, permutation) for each taste/changepoint/latent.
2. `<dataset_name>_significant_only.txt`
   Subset of the above containing only entries with at least one p < 0.05.

Console output
--------------
As each latent dimension is processed, prints a colored summary:
  – “Significant” (green) or “Not Significant” (red) along with p-values.

Return value
------------
dict[str, list[dict]]
    Mapping from dataset name to a list of result entries:
    {
      'taste': int,
      'changepoint': int,
      'results_per_latent': {
         column_name: {
           't-test':    {'p_val': float, 'stat': float, 'significant': bool},
           'ks-test':   {...},
           'kruskal':   {...},
           'permutation': {'p_val': float, 'real_stat': float, 'significant': bool}
         }, …
      }
    }
"""
    def __init__(self, data_dict: dict, rng: np.random.Generator, save_dir: str, n_permutations: int = 10000):
        """
        Initialize the SignificanceTester class.

        Parameters
        ----------
        data_dict : dict
            Dictionary of dataset_name -> Polars DataFrames.
        rng : np.random.Generator
            Numpy random generator for reproducibility.
        save_dir : str
            Directory where text files will be saved.
        n_permutations : int
            Number of permutations for permutation testing.
        """
        self.data_dict = data_dict
        self.rng = rng
        self.save_dir = Path(save_dir)
        self.n_permutations = n_permutations

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_data_columns(df: pl.DataFrame):
        """
        Return PC_x or latent_dim_x columns from a Polars DataFrame.
        Useful for looping across latent dimensions.
        """
        return [col for col in df.columns if col.startswith('PC_') or col.startswith('latent_dim_')]
    @staticmethod
    def permutation_test(actual_data, shuffled_data, rng, n_permutations):
        """
        Permutation test using controlled random generator (rng.permutation).
        """
        combined = np.concatenate([actual_data, shuffled_data])
        real_statistic = np.mean(actual_data) - np.mean(shuffled_data)
    
        n = len(actual_data)
        perm_stats = np.empty(n_permutations)
    
        for i in range(n_permutations):
            permuted = rng.permutation(combined)  # Use rng.permutation instead of np.random.shuffle
            perm_group1 = permuted[:n]
            perm_group2 = permuted[n:]
            perm_stats[i] = np.mean(perm_group1) - np.mean(perm_group2)
    
        p_val = np.mean(np.abs(perm_stats) >= np.abs(real_statistic))
    
        return p_val, real_statistic

    @staticmethod
    def colored_significance(is_significant):
        if is_significant:
            return "\033[92mSignificant\033[0m"  # Green
        else:
            return "\033[91mNot Significant\033[0m"  # Red

    @staticmethod
    def summarize_tests_console(p_vals_and_significance):
        print("\nSummary of significance across tests:")
        for test_name, results in p_vals_and_significance.items():
            p_val = results["p_val"]
            is_sig = results["significant"]
            color = "\033[92m" if is_sig else "\033[91m"
            reset = "\033[0m"
            print(f"  {test_name}: {color}{'Significant' if is_sig else 'Not Significant'}{reset} (p = {p_val:.4f})")
        print("-" * 50)

    def run(self):
        """
        Main function to run significance tests on all datasets.
        Returns
        -------
        dict
            Dataset_name -> List of result dicts.
        """
        results_summary = {}

        for dataset_name, df in self.data_dict.items():
            print(f"\nProcessing dataset: {dataset_name}")
            output_file = self.save_dir / f"{dataset_name}_significance_results.txt"
            sig_output_file = self.save_dir / f"{dataset_name}_significant_only.txt"

            all_lines = []
            sig_lines = []

            latent_cols = get_data_columns(df)
            unique_tastes = sorted(df["taste"].unique().to_list())
            unique_cps = sorted(df["changepoint"].unique().to_list())

            dataset_results = []

            for taste in unique_tastes:
                for cp in unique_cps:
                    sub_df = df.filter(
                        (pl.col("taste") == taste) & (pl.col("changepoint") == cp)
                    )

                    if sub_df.height == 0:
                        continue

                    result_entry = {
                        "taste": taste,
                        "changepoint": cp,
                        "results_per_latent": {}
                    }

                    for col in latent_cols:
                        trial_arrays = []

                        unique_trials = sorted(sub_df["trial"].unique().to_list())
                        for trial_id in unique_trials:
                            trial_df = sub_df.filter(pl.col("trial") == trial_id).sort("time")
                            trial_array = trial_df[col].to_numpy()
                            trial_arrays.append(trial_array)

                        if len(trial_arrays) < 2:
                            continue

                        actual_data = np.concatenate(trial_arrays)
                        shuffled_data = actual_data.copy()
                        self.rng.shuffle(shuffled_data)

                        # Statistical tests
                        t_stat, p_val_t = ttest_ind(actual_data, shuffled_data, alternative='two-sided', equal_var=False)
                        ks_stat, p_val_ks = ks_2samp(actual_data, shuffled_data, alternative='two-sided')
                        h_stat, p_val_kruskal = kruskal(actual_data, shuffled_data)
                        p_val_perm, real_statistic = self.permutation_test(actual_data, shuffled_data, self.rng, self.n_permutations)
                        sig_ttest = p_val_t < 0.05
                        sig_ks = p_val_ks < 0.05
                        sig_kruskal = p_val_kruskal < 0.05
                        sig_perm = p_val_perm < 0.05

                        result_entry["results_per_latent"][col] = {
                            "t-test": {"p_val": p_val_t, "stat": t_stat, "significant": sig_ttest},
                            "ks-test": {"p_val": p_val_ks, "stat": ks_stat, "significant": sig_ks},
                            "kruskal": {"p_val": p_val_kruskal, "stat": h_stat, "significant": sig_kruskal},
                            "permutation": {"p_val": p_val_perm, "real_stat": real_statistic, "significant": sig_perm},
                        }

                        block = f"Taste {taste}, Changepoint {cp}, Latent {col}:\n"
                        block += f"  Welch's t-test: p = {p_val_t:.4f}, t = {t_stat:.4f} --> {'Significant' if sig_ttest else 'Not Significant'}\n"
                        block += f"  Kolmogorov-Smirnov test: p = {p_val_ks:.4f}, D = {ks_stat:.4f} --> {'Significant' if sig_ks else 'Not Significant'}\n"
                        block += f"  Kruskal-Wallis test: p = {p_val_kruskal:.4f}, H = {h_stat:.4f} --> {'Significant' if sig_kruskal else 'Not Significant'}\n"
                        block += f"  Permutation test: p = {p_val_perm:.4f}, diff = {real_statistic:.4f} --> {'Significant' if sig_perm else 'Not Significant'}\n\n"
                        
                        all_lines.append(block)

                        if any([sig_ttest, sig_ks, sig_kruskal, sig_perm]):
                            sig_lines.append(block)

                        p_vals_and_significance = {
                            "Welch's t-test": {"p_val": p_val_t, "significant": sig_ttest},
                            "Kolmogorov-Smirnov test": {"p_val": p_val_ks, "significant": sig_ks},
                            "Kruskal-Wallis test": {"p_val": p_val_kruskal, "significant": sig_kruskal},
                            "Permutation test": {"p_val": p_val_perm, "significant": sig_perm},
                        }
                        self.summarize_tests_console(p_vals_and_significance)

                    dataset_results.append(result_entry)

            with open(output_file, "w") as f:
                f.writelines(all_lines)

            with open(sig_output_file, "w") as f:
                if sig_lines:
                    f.write(f"Significant Results for {dataset_name}\n\n")
                    f.writelines(sig_lines)
                else:
                    f.write(f"No significant results found for {dataset_name}.\n")

            results_summary[dataset_name] = dataset_results

        return results_summary
    
# extra stuff: 

# # the below sucks 
# def analyze_significant_epochs(df: pl.DataFrame, dataset_name: str, rng: np.random.Generator) -> pl.DataFrame:
#     """
#     For a given Polars DataFrame that contains latent dimension columns along with 
#     metadata columns ("taste", "trial", "changepoint", "time"), this function:
    
#       1. For each (taste, changepoint) combination, groups the data by trial.
#       2. For each latent column, stacks the time-series data of each trial (sorted by time) 
#          to form the 'actual' distribution.
#       3. Creates a surrogate ('shuffled') distribution by shuffling the order of the trial arrays 
#          before stacking them.
#       4. Runs a two-sample t-test between the actual and shuffled distributions.
#       5. Prints a message if p < 0.05.
#       6. Records for each latent column a tuple (p-value, t-statistic).
      
#     The output DataFrame has one row per unique (taste, changepoint) combination, with 
#     the same column structure as the original (metadata fields "trial" and "time" set to None).
#     """
#     meta_cols = ["taste", "trial", "changepoint", "time"]
#     latent_cols = [col for col in df.columns if col not in meta_cols]
    
#     # Get unique taste and changepoint values.
#     unique_tastes = sorted(df["taste"].unique().to_list())
#     unique_cps = sorted(df["changepoint"].unique().to_list())
    
#     rows = []
    
#     for taste in unique_tastes:
#         for cp in unique_cps:
#             # Filter the data for this (taste, changepoint) combination.
#             sub_df = df.filter((pl.col("taste") == taste) & (pl.col("changepoint") == cp))
#             if sub_df.height == 0:
#                 continue  # Skip if no data exists for this combination.
            
#             # Get unique trial identifiers.
#             trial_ids = sorted(sub_df["trial"].unique().to_list())
#             result_row = {"taste": taste, "changepoint": cp}
            
#             for col in latent_cols:
#                 # For each trial, extract and sort data by time, then store the 1D array.
#                 trial_arrays = []
#                 for trial in trial_ids:
#                     trial_df = sub_df.filter(pl.col("trial") == trial).sort("time")
#                     trial_data = trial_df[col].to_numpy()
#                     trial_arrays.append(trial_data)
                
#                 # Stack trials in natural order to form the actual data.
#                 actual_data = np.concatenate(trial_arrays)
                
#                 # For the shuffled distribution, shuffle the list of trial arrays before stacking.
#                 shuffled_trials = trial_arrays.copy()
#                 rng.shuffle(shuffled_trials)
#                 shuffled_data = np.concatenate(shuffled_trials)
                
#                 # Run the two-sample t-test between actual and shuffled distributions.
#                 t_stat, p_val = ttest_ind(actual_data, shuffled_data, alternative='two-sided', equal_var=False)
                
#                 if p_val < 0.05:
#                     print(f"For dataset {dataset_name}, taste {taste} epoch {cp}, latent '{col}' is significant (p = {p_val:.3f}).")
                
#                 result_row[col] = (p_val, t_stat)
            
#             # For metadata fields that no longer apply after aggregation.
#             result_row["trial"] = None
#             result_row["time"] = None
#             rows.append(result_row)
    
#     # Construct the DataFrame with latent columns first, then metadata.
#     new_columns_order = latent_cols + ["taste", "trial", "changepoint", "time"]
#     result_df = pl.DataFrame(rows)[new_columns_order]
#     return result_df

# def quantify_significant_epochs(data_dict: dict, rng: np.random.Generator) -> dict:
#     """
#     Processes each DataFrame in the input dictionary (e.g., first_deriv_dict) and returns a dictionary
#     (sig_epochs_marker) where each key is the same as in the input, and the corresponding value is a DataFrame
#     with t-test results (p-value and t-statistic) for each (taste, changepoint) combination.
    
#     Parameters:
#         data_dict (dict): Dictionary of Polars DataFrames.
#         rng (np.random.Generator): A NumPy random generator for shuffling.
    
#     Returns:
#         dict: Dictionary where keys are dataset names and values are marker DataFrames.
#     """
#     sig_epochs_marker = {}
    
#     for dataset_name, df in data_dict.items():
#         marker_df = analyze_significant_epochs(df, dataset_name, rng)
#         sig_epochs_marker[dataset_name] = marker_df
        
#     return sig_epochs_marker

# # Example usage:
# # Assume first_deriv_dict is your input dictionary of Polars DataFrames.
# rng = np.random.default_rng()
# sig_epochs_marker = quantify_significant_epochs(first_der_dict, rng)

# # Now, sig_epochs_marker is a dict of DataFrames with t-test results.

