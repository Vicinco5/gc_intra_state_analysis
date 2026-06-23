# script explicitly for loading paul's simulated data
# NOTE: see the notebook 'convert paul data' for how I'm handling his .dat files 
# NOTE: thanks to fixed point finder I had to do some stuff to my numpy version and that's making this a bit more of a pain in the ass than I had bargened for. 
# (bargain? maybe that's the right spelling. idk anyways)

# per paul's notes / email: 
"""
Here is a folder of 20 trials of 30 cells with spikes coming from these
sorts of transitions to oscillations at different times on different trials.

You can see what it looks like when you do the same analyses. I added a
lot of heterogeneity in the connections and excitability of cells etc.
but the basis is the set of 4 populations that look something like the
figure attached. In the figure, the yellow is an all-inhibitory
population and I only sent spikes from the excitatory populations
(produced as a Poisson process from each of 30 units rate vectors).

The zipped folder has one file per trial with spikes sorted in two
columns, first column are times, second column neuron IDs.
"""

# actual formal docstring: 
"""
load_paul.py — loader/adapter for Paul's generated .dat spike data.
 
Goal: produce a `spike_array` with the SAME structure run_rnn.py already
expects from ephys_data so that I can run this stuff right through the network; 
so Paul's data is drop-in for the existing taste loop and `preprocess_taste` 
with NO changes to the shared pipeline. Should be reasonably easy. 
 
Contract (read off run_rnn.py):
    spike_array : (n_tastes, n_trials, n_neurons, time_ms)
                  int counts at 1 ms resolution
                  -> taste loop iterates axis 0
                  -> taste_spikes[..., time_lims[0]:time_lims[1]] slices ms
                  -> preprocess_taste bins (sum) over bin_size ms
 
Paul's raw data:
    one .dat file per trial, whitespace-delimited, two columns:
        col 0 = spike time (SECONDS, scientific notation)
        col 1 = neuron id  (float, e.g. 2.6e+01; 1-indexed per filename units40)
    There is no taste structure -> n_tastes = 1.
 
Notes:
    * Parsing is pure-polars (no np.loadtxt thanks to numpy versioning) and the dense array is filled by
      plain integer indexing into a preallocated np.zeros. Neither path touches
      the polars<->numpy capsule that was panicking under numpy 2.x, so this
      loader is safe even before that env issue is resolved (whenever I get around to that, perhaps soon). 
      This issue is genuinely such a pain in my ass-- so I'll fix it later. 
    * time_lims / stim_time_val are NOT decided here — they are experiment
      metadata and belong in config, exactly like the blech path. This loader
      only builds the full-resolution array up to `duration_ms`. 
      NOTE: as far as I can tell, there simply is no stim_time_val at all. set it to 0 in the func call. 
    * But also this data is only simulated for 2 seconds. 
"""

import re
from pathlib import Path
import os 
import numpy as np
import polars as pl
 
# never-occurring separator -> read_csv puts each whole line in one column,
# then tokenize on runs of non-whitespace (that is, literally for the names of the files). 
# handles leading/repeated spaces and scientific notation without a single-char delimiter. 
# this is prob more fancy than it needed to be but hey it works! 
_UNIT_SEP = "\x1f"
_TRIAL_RE = re.compile(r"trial_(\d+)$")
 
 
def read_paul_long_df(data_dir, neuron_base=1):
    """Parse every *.dat trial file into one long polars df.
    Returns columns: spk_time (f64, seconds), neuron (i64), trial (i64).
    `neuron` is left in its raw (1-indexed) id space; `neuron_base` is only
    recorded for the caller, not subtracted here.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"no .dat files under {data_dir}")
 
    frames = []
    for f in files:
        m = _TRIAL_RE.search(f.stem)
        if m is None:
            raise ValueError(f"could not parse trial number from {f.name}")
        trial_num = int(m.group(1))
 
        dat = (
            pl.read_csv(f, has_header=False, separator=_UNIT_SEP,
                        new_columns=["raw"])
            .with_columns(pl.col("raw").str.extract_all(r"\S+").alias("tok"))
            .select(
                pl.col("tok").list.get(0).cast(pl.Float64).alias("spk_time"),
                pl.col("tok").list.get(1).cast(pl.Float64).round()
                  .cast(pl.Int64).alias("neuron"),
            )
            .with_columns(pl.lit(trial_num).alias("trial"))
        )
        frames.append(dat)
 
    return pl.concat(frames, how="vertical").sort("trial", "spk_time")
 
 
def build_spike_array(df, n_neurons, n_time_ms, neuron_base=1):
    """Densify the long df into (1, n_trials, n_neurons, n_time_ms) int counts.
 
    Trials are positioned by ascending trial number (0..n_trials-1). Spikes at
    or beyond `n_time_ms`, or with an id outside [neuron_base, neuron_base+n_neurons),
    are dropped (with a count returned for sanity-checking).
    """
    trial_ids = sorted(df["trial"].unique().to_list())
    trial_pos = {t: i for i, t in enumerate(trial_ids)}
    n_trials = len(trial_ids)
 
    arr = np.zeros((1, n_trials, n_neurons, n_time_ms), dtype=np.int16)
 
    dropped_time = 0
    dropped_neuron = 0
    for trial, neuron, t in zip(df["trial"].to_list(),
                                df["neuron"].to_list(),
                                df["spk_time"].to_list()):
        j = neuron - neuron_base                 # id -> 0-indexed neuron row
        if not (0 <= j < n_neurons):
            dropped_neuron += 1
            continue
        k = int(round(t * 1000.0))               # seconds -> ms index
        if not (0 <= k < n_time_ms):
            dropped_time += 1
            continue
        arr[0, trial_pos[trial], j, k] += 1
 
    stats = dict(
        n_trials=n_trials,
        trial_ids=trial_ids,
        dropped_out_of_window=dropped_time,
        dropped_bad_neuron_id=dropped_neuron,
        total_spikes=int(arr.sum()), # literally: sum of all the spikes we get lol 
    )
    return arr, stats
 
 
def load_paul(data_dir, n_neurons=30, duration_ms=2000, neuron_base=1,
              h5_out=None, verbose=True):
    """End-to-end: .dat dir -> spike_array matching the run_rnn.py expectation. 
    This is the function that 1:1 loads Paul's data, mimicing the hdf5 file input 
 
    Args:
        data_dir:    folder of *.dat trial files
        n_neurons:   full unit set incl. silent units (filenames say units40)
        duration_ms: length of the 1 ms time axis to build
        neuron_base: 1 if ids are 1-indexed (MATLAB), 0 if 0-indexed
        h5_out:      optional path; if given, writes the raw array + metadata
                     so there's a real file for save_to_hdf5 to append to
        verbose:     print a short sanity summary
 
    Returns:
        spike_array : (1, n_trials, n_neurons, duration_ms) int16
        hdf5_path   : str or None (path written, mirrors data.hdf5_path)
        stats       : dict (trial ids, dropped-spike counts, totals)
    """
    df = read_paul_long_df(data_dir, neuron_base=neuron_base)
    spike_array, stats = build_spike_array(df, n_neurons, duration_ms,
                                           neuron_base=neuron_base)
 
    if verbose:
        print(f"[load_paul] {data_dir}")
        print(f"            spike_array shape: {spike_array.shape}  "
              f"(tastes, trials, neurons, ms)")
        print(f"            trials: {stats['n_trials']}  "
              f"ids {stats['trial_ids'][:5]}{'...' if stats['n_trials'] > 5 else ''}")
        print(f"            total spikes kept: {stats['total_spikes']}")
        if stats["dropped_out_of_window"]:
            print(f"            WARNING dropped {stats['dropped_out_of_window']} "
                  f"spikes >= {duration_ms} ms (raise duration_ms?)")
        if stats["dropped_bad_neuron_id"]:
            print(f"            WARNING dropped {stats['dropped_bad_neuron_id']} "
                  f"spikes with id outside [{neuron_base}, {neuron_base + n_neurons})")
 
    hdf5_path = None
    if h5_out is not None:
        import h5py
        hdf5_path = str(h5_out)
        # make the dir as it def does not exist 
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True) 
        with h5py.File(hdf5_path, "w") as fh:
            fh.create_dataset("spike_array", data=spike_array,
                              compression="gzip")
            fh.attrs["source"] = str(data_dir)
            fh.attrs["n_neurons"] = n_neurons
            fh.attrs["duration_ms"] = duration_ms
            fh.attrs["neuron_base"] = neuron_base
            fh.attrs["resolution_ms"] = 1
        if verbose:
            print(f"            wrote {hdf5_path}")
 
    return spike_array, hdf5_path, stats
 
 
if __name__ == "__main__":
    # smoke test
    DATA_DIR = ("/home/vincent/Senior thesis work/blechRNN-master/"
                "data/paul_data/spikes/spikes/")
    arr, _, stats = load_paul(DATA_DIR, n_neurons=30, duration_ms=2000,
                              neuron_base=1)
    # quick rate sanity check: mean spikes/neuron/trial over the window
    mean_count = arr.sum() / (arr.shape[1] * arr.shape[2])
    print(f"            mean spikes / neuron / trial: {mean_count:.1f}")
 
