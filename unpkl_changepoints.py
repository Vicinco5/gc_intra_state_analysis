
import pandas as pd 
import os
import numpy as np 


def extract_valid_changepoints(pkl_path, dataset_num):
    def truncate_name(name):
        numbers = []
        parts = name.split("_")
        truncated_parts = []
        for part in parts:
            if any(char.isdigit() for char in part):
                numbers.extend(char for char in part if char.isdigit())
                truncated_parts.append(part)
        return "_".join(truncated_parts)

    try:
        truncated_dataset_num = truncate_name(dataset_num)
        print(f"Processing dataset number: {truncated_dataset_num}")
        raw_pkl = []
        pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
        for pkl_file in pkl_files:
            pkl_file_path = os.path.join(pkl_path, pkl_file)
            try:
                df = pd.read_pickle(pkl_file_path)
                for idx, row in df.iterrows():
                    row_basename = os.path.basename(row.iloc[0])
                    truncated_basename = truncate_name(row_basename)
                    if truncated_basename == truncated_dataset_num:
                        extracted_row = row.values
                        if isinstance(extracted_row[3], np.ndarray):
                            raw_pkl.append(extracted_row)
                        else:
                            print(
                                f"Skipping dataset {truncated_dataset_num} due to invalid data structure at index {idx}: {extracted_row[3]}"
                            )
                            raw_pkl = []
                            break
            except Exception as e:
                print(f"Error processing file {pkl_file}: {e}")
                raw_pkl = []
                break

        if not raw_pkl:
            print(
                f"Skipping dataset {truncated_dataset_num} due to invalid data structures."
            )
            return None

        print(
            f"Finished processing dataset {truncated_dataset_num}. Starting to process raw_pkl."
        )
        raw_pkl = np.array(raw_pkl, dtype=object)

        # Create extracted_pkl by ensuring correct length and adding the fourth number
        extracted_pkl = []
        for i in range(len(raw_pkl)):
            new_row = list(raw_pkl[i])
            new_sub_array = []
            for j in range(len(new_row[3])):
                row = new_row[3][j]
                if isinstance(row, (list, np.ndarray)) and len(row) == 3:
                    row = np.append(row, row[2] + 2000)
                elif isinstance(row, (list, np.ndarray)) and len(row) < 3:
                    padding = [np.nan] * (3 - len(row))
                    row = np.append(row, padding)
                    row = np.append(row, row[2] + 2000)
                else:
                    print(
                        f"Unexpected data structure for row {j} in new_row[{i}]: {row}"
                    )
                    continue
                new_sub_array.append(row)
            new_row[3] = np.array(new_sub_array)
            extracted_pkl.append(new_row)

        extracted_pkl = np.array(extracted_pkl, dtype=object)
        print(f"Finished processing valid dataset {truncated_dataset_num}.")
        return extracted_pkl if len(extracted_pkl) > 0 else None
    except FileNotFoundError:
        print(f"Error: Directory {pkl_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# helpers to grabbing relevant changepoints: 
def load_changepoints(pkl_dir, dataset_name):
    """Per-taste changepoint arrays for a dataset, or None if not found."""
    extracted = extract_valid_changepoints(pkl_dir, dataset_name)
    if extracted is None:
        return None
    return extracted[:, 3]          # (n_tastes,) object; each (n_trials, n_cp)

def get_trial_changepoints(cps_per_taste, taste_ind, trial_idx, n_real=3):
    """The real HMM changepoints (ms) for one taste/trial, NaNs dropped."""
    row = np.asarray(cps_per_taste[taste_ind][trial_idx], dtype=float)
    real = row[:n_real]
    return [float(x) for x in real if np.isfinite(x)]