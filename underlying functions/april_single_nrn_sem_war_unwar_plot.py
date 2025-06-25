#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:10:00 2025

Code update 4/15/25 single neuron analysis; SEM warping and similar scripts

@author: vincentcalia-bogan
"""

## further updates to the above-- getting messy, will clean later.


def plot_neuron_mean_with_sem_grid_single_dataset(
    dataset_df,
    firing_rate_data,
    output_dir,
    taste_labels,
    epoch_labels,
    step_size,
    dataset_name,
    use_warped,
    unaligned_mode=None,
    standardized_changepoints_dict=None,
):
    """
    Plots neuron mean firing rates + SEM for one dataset.
    Supports start-aligned, end-aligned, and natural-time aligned modes.
    """
    neurons_per_figure = 16
    num_tastes = len(taste_labels)
    num_changepoints = firing_rate_data.sizes["segment"]
    unique_neurons = dataset_df["neuron"].unique(maintain_order=True).to_list()

    for taste_idx in range(num_tastes):
        for cp_idx in range(num_changepoints):
            fig_num = 1
            for neuron_start in range(0, len(unique_neurons), neurons_per_figure):
                fig, axes = plt.subplots(
                    4, 4, figsize=(20, 20), sharex=False, sharey=False
                )
                fig.suptitle(
                    f'Mean {"Warped" if use_warped else "Unwarped"} Firing Rates with SEM\n'
                    f"Epoch: {epoch_labels[cp_idx]}, Taste: {taste_labels[taste_idx]}, Dataset: {dataset_name}",
                    fontsize=18,
                    y=0.995,
                )

                # Extract trial-level changepoints for natural alignment
                if unaligned_mode == "none":
                    taste_cps = standardized_changepoints_dict[dataset_name][taste_idx]
                    segment_starts = []
                    segment_ends = []

                    for row in taste_cps:
                        start = 2000 if cp_idx == 0 else row[cp_idx - 1]
                        end = row[cp_idx]
                        segment_starts.append(start)
                        segment_ends.append(end)

                    global_start = min(segment_starts)
                    global_end = max(segment_ends)
                    global_bins = int(np.ceil((global_end - global_start) / step_size))

                for i, neuron_idx in enumerate(
                    unique_neurons[neuron_start : neuron_start + neurons_per_figure]
                ):
                    row, col = divmod(i, 4)
                    ax = axes[row, col]
                    ax.set_title(f"Neuron {neuron_idx + 1}")
                    ax.set_xlabel("Time (ms)")
                    if col == 0:
                        ax.set_ylabel("Firing Rate (Hz)")

                    try:
                        trials_data = firing_rate_data.sel(
                            taste=taste_idx, segment=cp_idx, neuron=neuron_idx
                        ).values  # (num_trials, time_bins)

                        valid_trials = [
                            row[~np.isnan(row)]
                            for row in trials_data
                            if not np.all(np.isnan(row))
                        ]
                        if not valid_trials:
                            continue

                        if unaligned_mode == "none":
                            all_data = np.full((len(valid_trials), global_bins), np.nan)

                            for j, row_data in enumerate(valid_trials):
                                # Timing info
                                trial_cp = standardized_changepoints_dict[dataset_name][
                                    taste_idx
                                ][j]
                                trial_start = (
                                    2000 if cp_idx == 0 else trial_cp[cp_idx - 1]
                                )
                                offset_ms = trial_start - global_start
                                offset_bins = int(np.round(offset_ms / step_size))

                                # Place data at proper offset
                                if offset_bins + len(row_data) <= global_bins:
                                    all_data[
                                        j, offset_bins : offset_bins + len(row_data)
                                    ] = row_data
                                else:
                                    # If it would overrun the array (rare), truncate
                                    available_bins = global_bins - offset_bins
                                    all_data[j, offset_bins:] = row_data[
                                        :available_bins
                                    ]

                            time_axis = np.arange(global_bins) * step_size

                        else:
                            max_len = max(len(row) for row in valid_trials)
                            all_data = np.full((len(valid_trials), max_len), np.nan)

                            for j, row_data in enumerate(valid_trials):
                                row_len = len(row_data)
                                if unaligned_mode == "start" or use_warped:
                                    all_data[j, :row_len] = row_data
                                elif unaligned_mode == "end":
                                    all_data[j, max_len - row_len :] = (
                                        row_data  # bug fix I think
                                    )
                                else:  # fallback
                                    all_data[j, :row_len] = row_data

                            time_axis = np.arange(max_len) * step_size

                        mean_data = np.nanmean(all_data, axis=0)
                        sem_data = np.nanstd(all_data, axis=0) / np.sqrt(
                            np.sum(~np.isnan(all_data), axis=0)
                        )

                        ax.plot(time_axis, mean_data, color="black", linewidth=2)
                        ax.fill_between(
                            time_axis,
                            mean_data - sem_data,
                            mean_data + sem_data,
                            color="gray",
                            alpha=0.5,
                        )

                    except KeyError:
                        continue

                plt.tight_layout()
                plot_path = os.path.join(
                    output_dir,
                    f'mean_with_sem_{"warped" if use_warped else "unwarped"}_epoch_{cp_idx}_taste_{taste_idx}_fig_{fig_num}.png',
                )
                plt.savefig(plot_path)
                plt.close(fig)
                fig_num += 1


# relevant helper funcs:
# Helper function
def find_dataset_in_sig_dict(sig_nrns_dict, dataset_num_clean):
    """
    Searches across all polars.DataFrames in sig_nrns_dict for rows containing
    dataset_num_clean as a substring. Returns the filtered DataFrame or None.
    """
    for df_name, df in sig_nrns_dict.items():
        # Convert to string and search with "contains"
        matching_rows = df.filter(
            pl.col("dataset").cast(pl.Utf8).str.contains(dataset_num_clean)
        )
        if not matching_rows.is_empty():
            return matching_rows
    return None


def prompt_plot_flags():
    plot_warped = input("Plot warped data? (y/n): ").strip().lower() == "y"
    plot_unwarped = input("Plot unwarped data? (y/n): ").strip().lower() == "y"
    unaligned_mode = None
    if plot_unwarped:
        print("Choose unwarped alignment mode:")
        print("  - 'start': align all trials to start of segment")
        print("  - 'end': align all trials to end of segment")
        print("  - 'none': do not align")
        unaligned_mode = (
            input("Enter alignment mode (start/end/none): ").strip().lower()
        )
    return plot_warped, plot_unwarped, unaligned_mode


def process_and_plot_datasets(
    npz_path,
    pkl_path,
    sig_nrns_dict,
    modified_tastes,
    epoch_labels,
    base_output_dir,
    standardized_changepoints_dict,
    window_length=250,
    step_size=25,
):
    plot_warped, plot_unwarped, unaligned_mode = prompt_plot_flags()

    for data in extract_from_npz(npz_path):
        if isinstance(data, tuple) and len(data) == 4:
            spike_array, dataset_num, index, key = data
            dataset_num_clean = str(dataset_num).split("_repacked.npz")[0]
            print(f"\nProcessing dataset: {dataset_num_clean}")

            dataset_df = find_dataset_in_sig_dict(sig_nrns_dict, dataset_num_clean)
            if dataset_df is None:
                print(f"No matching sig_df for {dataset_num_clean}, skipping.")
                continue

            extracted_pkl = extract_valid_changepoints(
                pkl_path, spike_array, dataset_num_clean, index, key
            )

            if extracted_pkl is not None:
                try:
                    changepoints = extracted_pkl[:, 3]
                    dataset_output_dir = os.path.join(
                        base_output_dir, f"dataset_{dataset_num_clean}"
                    )
                    os.makedirs(dataset_output_dir, exist_ok=True)

                    # Compute firing rates using CalcFRStates
                    calc = CalcFRStates(
                        spike_array=spike_array,
                        changepoints=changepoints,
                        window_length=window_length,
                        step_size=step_size,
                        compute_unwarped_firing_rates=plot_unwarped,
                        compute_warped_firing_rates=plot_warped,
                        compute_unwarped_spike_arrays=False,
                        compute_warped_spike_arrays=False,
                        fixed_warp_duration=1000 if plot_warped else None,
                    )
                    fr_unwarped, _, fr_warped, _ = calc.run()

                    if plot_unwarped:
                        plot_neuron_mean_with_sem_grid_single_dataset(
                            dataset_df=dataset_df,
                            firing_rate_data=fr_unwarped,
                            output_dir=dataset_output_dir,
                            taste_labels=modified_tastes,
                            epoch_labels=epoch_labels,
                            step_size=step_size,
                            dataset_name=dataset_num_clean,
                            use_warped=False,
                            unaligned_mode=unaligned_mode,
                            standardized_changepoints_dict=(
                                standardized_changepoints_dict
                                if unaligned_mode == "none"
                                else None
                            ),
                        )

                    if plot_warped:
                        plot_neuron_mean_with_sem_grid_single_dataset(
                            dataset_df=dataset_df,
                            firing_rate_data=fr_warped,
                            output_dir=dataset_output_dir,
                            taste_labels=modified_tastes,
                            epoch_labels=epoch_labels,
                            step_size=step_size,
                            dataset_name=dataset_num_clean,
                            use_warped=True,
                            unaligned_mode=None,
                        )

                except Exception as e:
                    print(f"❌ Error processing {dataset_num_clean}: {e}")
            else:
                print(f"⚠️ No valid changepoints for {dataset_num_clean}; skipping...")

    print("\n✅ All datasets processed.")
