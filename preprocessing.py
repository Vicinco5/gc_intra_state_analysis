"""
pre-processor for the data prior to being fed into the actual rnn. 
performs: 
spike binning, scaling, PCA, context concatenation,
and train/test splitting.

"""


import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def bin_spikes(taste_spikes, bin_size):
    """
    Bin spike trains.

    Args:
        taste_spikes: (trials, neurons, time_ms)
        bin_size: int, bin width in ms

    Returns:
        binned_spikes: (trials, neurons, n_bins)
    """
    # taste_spikes shape: (trials, neurons, time)
    n_trials, n_neurons, n_time = taste_spikes.shape

    # Truncate to largest multiple of bin_size so reshape works for any bin size-- otherwise amount of time is an issue 
    # cutting a little time off the end is not the end of the world. 
    n_bins = n_time // bin_size
    usable = n_bins * bin_size
    if usable != n_time:
        taste_spikes = taste_spikes[..., :usable]

    return np.reshape(
        taste_spikes,
        (n_trials, n_neurons, n_bins, bin_size)
    ).sum(axis=-1)


def scale_inputs(inputs):
    """
    Z-score inputs across time and trials (long-form).

    Args:
        inputs: (time, trials, neurons)

    Returns:
        inputs_scaled: same shape, z-scored
        scaler: fitted StandardScaler (for inverse transform later)
    """
    orig_shape = inputs.shape
    inputs_long = inputs.reshape(-1, orig_shape[-1])
    scaler = StandardScaler()
    inputs_long = scaler.fit_transform(inputs_long)
    inputs_scaled = inputs_long.reshape(orig_shape)
    return inputs_scaled, scaler


def apply_pca(inputs, scaler, n_components=0.95):
    """
    Apply PCA to scaled inputs.

    Args:
        inputs: (time, trials, neurons) — already z-scored
        scaler: fitted StandardScaler (kept for reference)
        n_components: float (variance threshold) or int

    Returns:
        inputs_pca: (time, trials, n_components)
        pca_obj: fitted PCA object
    """
    orig_shape = inputs.shape
    inputs_long = inputs.reshape(-1, orig_shape[-1])
    pca_obj = PCA(n_components=n_components)
    inputs_pca = pca_obj.fit_transform(inputs_long)
    n_comp = inputs_pca.shape[-1]
    inputs_pca = inputs_pca.reshape(orig_shape[0], orig_shape[1], n_comp)
    return inputs_pca, pca_obj


def add_context(inputs, stim_time_bin, trial_num):
    """
    Concatenate stimulus indicator and trial context to inputs.

    Args:
        inputs: (time, trials, features)
        stim_time_bin: int, bin index for stimulus onset
        trial_num: (n_trials,) array of trial indices

    Returns:
        inputs_plus_context: (time, trials, features + 2)
        context_dim: int (always 2)
    """
    n_time, n_trials, _ = inputs.shape

    stim_input = np.zeros((n_time, n_trials))
    stim_input[:, stim_time_bin] = 1

    trial_context = np.broadcast_to(
        trial_num / trial_num.max(), (n_time, n_trials)
    )

    inputs_plus_context = np.concatenate([
        inputs,
        stim_input[:, :, None],
        trial_context[:, :, None],
    ], axis=-1)

    return inputs_plus_context, 2  # context_dim


def prepare_tensors(inputs_plus_context, inputs):
    """
    Create input/label tensors with the 1-step time shift.

    Args:
        inputs_plus_context: (time, trials, input_size)
        inputs: (time, trials, output_features) — without context

    Returns:
        inputs_tensor: (time-1, trials, input_size)
        labels_tensor: (time-1, trials, output_size)
    """
    inputs_tensor = torch.tensor(
        inputs_plus_context[:-1], dtype=torch.float32
    )
    labels_tensor = torch.tensor(
        inputs[1:], dtype=torch.float32
    )
    return inputs_tensor, labels_tensor


def train_test_split_trials(inputs_tensor, labels_tensor, split_ratio=0.75):
    """
    Random train/test split along the trial (batch) dimension.

    Args:
        inputs_tensor: (time, n_trials, input_size)
        labels_tensor: (time, n_trials, output_size)
        split_ratio: float

    Returns:
        train_inputs, train_labels, test_inputs, test_labels,
        train_inds, test_inds
    """
    n_trials = inputs_tensor.shape[1]
    train_inds = np.random.choice(
        np.arange(n_trials),
        int(split_ratio * n_trials),
        replace=False
    )
    test_inds = np.setdiff1d(np.arange(n_trials), train_inds)

    return (
        inputs_tensor[:, train_inds],
        labels_tensor[:, train_inds],
        inputs_tensor[:, test_inds],
        labels_tensor[:, test_inds],
        train_inds,
        test_inds,
    )


def preprocess_taste(taste_spikes, bin_size, stim_time_val, use_pca=False):
    """
    Full preprocessing pipeline for one taste.

    Args:
        taste_spikes: (trials, neurons, time_ms) — already time-sliced
        bin_size: int
        stim_time_val: int, stim time in ms (relative to time_lims)
        use_pca: bool

    Returns:
        dict with keys:
            binned_spikes, inputs, inputs_plus_context,
            inputs_tensor, labels_tensor,
            input_size, output_size, num_neurons,
            scaler, pca_obj (or None),
            trial_num, stim_time_bin
    """
    binned_spikes = bin_spikes(taste_spikes, bin_size)
    inputs = np.moveaxis(binned_spikes.copy(), -1, 0)  # (time, trial, neuron)

    trial_num = np.arange(inputs.shape[1])
    num_neurons = inputs.shape[2]

    # Scale
    inputs, scaler = scale_inputs(inputs)

    # Optional PCA
    pca_obj = None
    if use_pca:
        print('  Performing PCA on inputs')
        inputs, pca_obj = apply_pca(inputs, scaler, n_components=0.95)
    if use_pca and pca_obj is not None:
        evr = pca_obj.explained_variance_ratio_
        cum = np.cumsum(evr)
        print(f"    [PCA] {len(evr)} components | "
            f"total var explained: {cum[-1]*100:.1f}%")
        for i, (e, c) in enumerate(zip(evr, cum)):
            print(f"      PC{i:2d}: {e*100:5.2f}%  (cumulative {c*100:5.1f}%)")

    # Context
    stim_time_bin = stim_time_val // bin_size
    inputs_plus_context, context_dim = add_context(
        inputs, stim_time_bin, trial_num
    )

    # Sizes
    input_size = inputs_plus_context.shape[-1]
    output_size = input_size - context_dim

    # Tensors
    inputs_tensor, labels_tensor = prepare_tensors(inputs_plus_context, inputs)

    # now also adding in the raw count labels specifically for the Poisson LL -- no z-scored or PCA'd
    # binned_spikes is (trials, neurons, time), need (time, trials, neurons)
    raw_inputs = np.moveaxis(binned_spikes.copy(), -1, 0)  # (time, trial, neuron)
    raw_labels_tensor = torch.tensor(
        raw_inputs[1:], dtype=torch.float32
    )  # same time shift as labels_tensor

    return dict(
        binned_spikes=binned_spikes,
        inputs=inputs,
        inputs_plus_context=inputs_plus_context,
        inputs_tensor=inputs_tensor,
        labels_tensor=labels_tensor,
        raw_labels_tensor=raw_labels_tensor,
        input_size=input_size,
        output_size=output_size,
        num_neurons=num_neurons,
        scaler=scaler,
        pca_obj=pca_obj,
        trial_num=trial_num,
        stim_time_bin=stim_time_bin,
    )