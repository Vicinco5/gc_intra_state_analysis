"""
fpf_analysis.py

Fixed-point analysis of single-layer autoencoderRNN latent dynamics using
FixedPointFinder (Golub & Sussillo 2018, JOSS 3(31):1003). All credit to these guys for figuring this out. 

WHY SINGLE-LAYER ONLY:
    FixedPointFinder's PyTorch implementation hardcodes a single-layer hidden
    state — in FixedPointFinderTorch._run_joint_optimization the hidden state
    is built as initial_states.unsqueeze(0), i.e. shape (1, batch, n_states),
    which assumes num_layers == 1. More fundamentally, a multi-layer RNN's true
    dynamical state is the stack of ALL layers' hidden states; the last layer
    alone is not a closed map, so "fixed points of the last layer" is not even
    well-defined. For a single layer, the latent state you visualize IS the full
    recurrent state, so fixed points live in the same space as your trajectories.
    => Run this only on models trained with rnn_layers == 1.
    Basically, a multi-layer rnn is very difficult to interpert fixed points-- 
    to the point that you kinda cannot. 

WHAT IT DOES:
    - Static analysis: fixed points under the mean encoded input.
    - Windowed analysis: fixed points conditioned on the input in a one-sided
      window around each HMM changepoint (before / after, never straddling the
      changepoint) plus mid-epoch and stimulus probes.
    - For each fixed point: Jacobian eigenvalues, stability classification, and
      for COMPLEX eigenvalue pairs, the implied oscillation frequency. An
      unstable fixed point with a complex pair (|lambda| > 1) is the dynamical
      signature of a surrounding LIMIT CYCLE; its eigenvalue angle gives the
      cycle frequency. this is CRITICAL INFO lol 

EIGENVALUE -> FREQUENCY (discrete time):
    A complex eigenvalue lambda = r * exp(i*theta) rotates by theta radians per
    timestep (per bin). Implied oscillation frequency:
        f_Hz = |theta| * 1000 / (2*pi * bin_size_ms)
    e.g. a 6.5 Hz cycle at 25 ms bins -> theta ~ 1.0 rad.

GENERAL USAGE (from run.py, single-layer model only):
    from fpf_analysis import run_fpf_analysis
    run_fpf_analysis(
        net=net, prep=prep, latent_outs=latent_outs,
        dataset_name=dataset_name, taste_ind=taste_ind,
        output_dir=fp_dir, device=device,
        changepoints_ms=[2300, 3100, 3800],   # HMM changepoints for this trial (ms)
        # NOTE: changepoints are at different times along different trials. 
        time_lims=params['time_lims'], bin_size=params['bin_size'],
        stim_ms=2000, trial_idx=0,
    )
"""

import os
import sys
import json
import numpy as np
import torch
from torch.autograd.functional import jacobian
#### THE FIXED POINT FINDER STUFF
import sys
sys.path.append('/home/vincent/Senior thesis work/blechRNN-master/fixed-point-finder')
sys.path.append('/home/vincent/Senior thesis work/blechRNN-master/recurrent-whisperer')
try:
    from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
except ImportError as e:
    raise ImportError(
        "Could not import FixedPointFinderTorch. Add the fixed-point-finder and "
        "recurrent-whisperer repos to PYTHONPATH (or uncomment the sys.path lines "
        "at the top of fpf_analysis.py).\nOriginal error: %s" % e
    )


# ================================================================
# Guards and small helpers
# ================================================================

def _check_single_layer(net):
    """Raise if the model is not single-layer. FPF requires num_layers == 1."""
    n_layers = net.rnn.num_layers
    if n_layers != 1:
        raise ValueError(
            f"fpf_analysis requires a single-layer RNN (rnn_layers == 1), "
            f"but this model has num_layers == {n_layers}. FixedPointFinder's "
            f"PyTorch state handling assumes one layer, and multi-layer fixed "
            f"points live in the stacked all-layer state, not the last layer. "
            f"Retrain with rnn_layers=1 for fixed-point analysis."
        )
    if net.rnn.bidirectional:
        raise ValueError("fpf_analysis does not support bidirectional RNNs.")


def ms_to_bin(t_ms, time_lims, bin_size):
    """Convert an absolute time in ms to a bin index within the analysis window."""
    return int(round((t_ms - time_lims[0]) / bin_size))


def encode_inputs(net, inputs_tensor, device):
    """
    Run the encoder to get the per-timestep input the RNN actually sees.

    Returns:
        encoded: (time, trials, hidden_size) numpy array
    """
    net.eval()
    with torch.no_grad():
        encoded = net.encoder(inputs_tensor.to(device))
    return encoded.detach().cpu().numpy()


def build_input_condition(encoded, t_indices, trial_idx=None):
    """
    Build the constant input condition for one FPF run.

    Args:
        encoded: (time, trials, hidden) numpy array
        t_indices: iterable of bin indices to average the input over.
            For before/after-changepoint probes these must be one-sided
            (never straddling the changepoint) — enforced by the probe builder.
        trial_idx: int for a single trial, or None to average across trials.

    Returns:
        (1, n_inputs) float32 numpy array (the shape FPF expects for a shared
        input across all initializations).
    """
    t_indices = np.asarray(t_indices, dtype=int)
    sub = encoded[t_indices]  # (len, trials, hidden)
    if trial_idx is None:
        cond = sub.mean(axis=(0, 1))          # average over window and trials
    else:
        cond = sub[:, trial_idx, :].mean(axis=0)  # average over window, one trial
    return cond[None, :].astype(np.float32)


def sample_initial_states(latent_outs, n_inits=1024, t_window=None,
                          trial_idx=None, noise=0.0, seed=None):
    """
    Sample initial hidden states to seed FPF's optimization.

    For a single-layer RNN, latent_outs IS the full recurrent state, so we can
    seed directly from the observed trajectories.

    Args:
        latent_outs: (time, trials, hidden) numpy array
        n_inits: number of initializations
        t_window: optional (t0, t1) bin range to restrict sampling near a probe.
            If None, sample from the whole trajectory.
        trial_idx: optional int to restrict to one trial; None uses all trials.
        noise: std of Gaussian noise added to sampled states.
        seed: optional RNG seed.

    Returns:
        (n_inits, hidden) float32 numpy array
    """
    rng = np.random.default_rng(seed)
    arr = latent_outs
    if t_window is not None:
        t0, t1 = t_window
        t0 = max(0, t0)
        t1 = min(arr.shape[0], t1)
        arr = arr[t0:t1]
    if trial_idx is not None:
        arr = arr[:, trial_idx:trial_idx + 1, :]
    states = arr.reshape(-1, arr.shape[-1])
    idx = rng.choice(len(states), size=n_inits, replace=(n_inits > len(states)))
    sampled = states[idx].copy().astype(np.float32)
    if noise > 0:
        sampled += rng.standard_normal(sampled.shape).astype(np.float32) * noise
    return sampled


# ================================================================
# Jacobian + eigenvalue / frequency analysis
# ================================================================

def _rnn_single_step(rnn, h, x):
    """
    One step of a SINGLE-LAYER RNN: h_next = F(h, x).

    Args:
        rnn: net.rnn (nn.RNN, num_layers == 1)
        h: (n, hidden) torch tensor
        x: (n, input) torch tensor  (constant input)

    Returns:
        (n, hidden) torch tensor
    """
    if rnn.batch_first:
        x_in = x.unsqueeze(1)   # (n, 1, input)
    else:
        x_in = x.unsqueeze(0)   # (1, n, input)
    h_in = h.unsqueeze(0)       # (1, n, hidden)  num_layers == 1
    _, h_next = rnn(x_in, h_in)
    return h_next[-1]           # (n, hidden), last (only) layer


def _jacobian_at(rnn, fp, x_const, device):
    """
    Jacobian dF/dh at a single fixed point.

    Args:
        fp: (hidden,) numpy array
        x_const: (input,) numpy array

    Returns:
        (hidden, hidden) numpy array
    """
    fp_t = torch.tensor(fp, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_const, dtype=torch.float32, device=device).unsqueeze(0)

    def f(h_vec):
        h = h_vec.unsqueeze(0)                    # (1, hidden)
        return _rnn_single_step(rnn, h, x_t).squeeze(0)  # (hidden,)

    # cuDNN cannot backprop through an RNN in eval mode; disable it here.
    with torch.backends.cudnn.flags(enabled=False):
        J = jacobian(f, fp_t)
    return J.detach().cpu().numpy()


def _eig_frequency(eigvals, bin_size_ms):
    """
    Convert complex eigenvalues to implied oscillation frequencies (Hz).

    Returns a list of (freq_hz, magnitude) for each complex eigenvalue with
    positive imaginary part (conjugate pairs counted once).
    """
    out = []
    for lam in eigvals:
        if abs(lam.imag) > 1e-6 and lam.imag > 0:   # one of each conjugate pair
            theta = abs(np.angle(lam))              # rad per bin
            if theta > 1e-9:
                f_hz = theta * 1000.0 / (2.0 * np.pi * bin_size_ms)
                out.append((float(f_hz), float(abs(lam))))
    return out


def summarize_fixed_points(fps, rnn, input_cond, bin_size_ms, device,
                           q_tol=1e-8):
    """
    Build a per-fixed-point summary from an FPF FixedPoints object.

    Args:
        fps: FixedPoints object returned by FixedPointFinder.find_fixed_points
        rnn: net.rnn (single layer)
        input_cond: (1, input) numpy array used for this run
        bin_size_ms: bin size in ms (for eigenvalue -> frequency)
        device: torch device
        q_tol: q above which a point is flagged as not-converged (slow point)

    Returns:
        list of dicts, one per fixed point.
    """
    xstar = np.asarray(fps.xstar)         # (n, hidden)
    qstar = np.asarray(fps.qstar).ravel() if getattr(fps, 'qstar', None) is not None \
        else np.full(len(xstar), np.nan)
    x_const = input_cond[0]

    summary = []
    for i in range(len(xstar)):
        J = _jacobian_at(rnn, xstar[i], x_const, device)
        eigs = np.linalg.eigvals(J)
        abs_eigs = np.abs(eigs)
        max_abs = float(np.max(abs_eigs))

        if max_abs < 1.0:
            stability = 'stable'
        elif np.all(abs_eigs > 1.0):
            stability = 'unstable'
        else:
            stability = 'saddle'

        freqs = _eig_frequency(eigs, bin_size_ms)
        is_spiral = len(freqs) > 0
        # Limit-cycle signature: complex pair on an unstable fixed point.
        is_unstable_spiral = is_spiral and max_abs > 1.0

        summary.append(dict(
            index=i,
            xstar=xstar[i],
            q=float(qstar[i]) if i < len(qstar) else float('nan'),
            converged=bool((qstar[i] < q_tol) if i < len(qstar) else False),
            eigenvalues=eigs,
            max_abs_eig=max_abs,
            stability=stability,
            is_spiral=is_spiral,
            is_unstable_spiral=is_unstable_spiral,
            frequencies_hz=freqs,   # list of (freq_hz, |lambda|)
        ))
    return summary


# ================================================================
# Probe-point construction (changepoint windowing)
# ================================================================

def build_probes(changepoint_bins, stim_bin, n_time, bin_size,
                 offset_lo_ms=50, offset_hi_ms=100, include_mid=True,
                 include_stim=True):
    """
    Build the list of probe points for the windowed analysis. 
    We give a time point of a changepoint-- but we want to know what is going on 
    before and after. 

    So: 
    For each changepoint c, emits a 'before' and an 'after' probe, each using a
    ONE-SIDED window that never crosses c (so the pre- and post-transition
    dynamics are not blended). Optionally adds mid-epoch probes (between
    consecutive changepoints, and the baseline / final epochs) and stimulus
    before/after probes.

    Args:
        changepoint_bins: list of changepoint bin indices (already converted)
        stim_bin: stimulus bin index
        n_time: number of time bins in the trajectory
        bin_size: bin size in ms
        offset_lo_ms, offset_hi_ms: window edges relative to the event, in ms
            (default 50-100 ms -> 2-4 bins at 25 ms)
        include_mid: add mid-epoch probes
        include_stim: add stimulus before/after probes

    Returns:
        list of dicts: {label, t_center, t_idx (np.ndarray of bins to average)}
    """
    off_lo = max(1, int(round(offset_lo_ms / bin_size)))
    off_hi = max(off_lo + 1, int(round(offset_hi_ms / bin_size)))

    def clip(b):
        return int(np.clip(b, 0, n_time - 1))

    def window(lo, hi):
        lo, hi = clip(lo), clip(hi)
        if hi < lo:
            lo, hi = hi, lo
        return np.arange(lo, hi + 1)

    probes = []

    events = []
    # stim is always 2000 ms
    if include_stim and stim_bin is not None:
        events.append(('stim', stim_bin))
    for k, c in enumerate(changepoint_bins):
        events.append((f'cp{k}', c))

    for label, c in events:
        # one-sided windows, strictly excluding c itself
        before_idx = window(c - off_hi, c - off_lo)
        after_idx = window(c + off_lo, c + off_hi)
        probes.append(dict(label=f'{label}_before', t_center=c - (off_lo + off_hi) // 2,
                           t_idx=before_idx))
        probes.append(dict(label=f'{label}_after', t_center=c + (off_lo + off_hi) // 2,
                           t_idx=after_idx))

    if include_mid and len(changepoint_bins) > 0:
        # midpoints of epochs: baseline -> cp0 -> ... -> cpN -> end
        boundaries = [0] + list(changepoint_bins) + [n_time - 1]
        for j in range(len(boundaries) - 1):
            a, b = boundaries[j], boundaries[j + 1]
            mid = (a + b) // 2
            # small symmetric window safely inside the epoch
            half = max(1, (b - a) // 6)
            lo, hi = clip(mid - half), clip(mid + half)
            probes.append(dict(label=f'epoch{j}_mid', t_center=mid,
                               t_idx=np.arange(lo, hi + 1)))

    return probes


# ================================================================
# Single-run wrapper around FixedPointFinder
# ================================================================

def find_fps(rnn, initial_states, input_cond, fpf_hps=None):
    """
    Run FixedPointFinder for one constant input condition.

    Args:
        rnn: net.rnn (single layer)
        initial_states: (n, hidden) float32 numpy
        input_cond: (1, input) float32 numpy
        fpf_hps: dict of FixedPointFinder hyperparameters (optional)

    Returns:
        FixedPoints object
    """
    default_hps = dict(
        tol_q=1e-12,
        tol_dq=1e-20,
        max_iters=5000,
        tol_unique=1e-3,
        # Jacobians ON so FPF's native plot_fps can draw eigenmodes. I still
        # recompute my Jacobians for the frequency analysis; (ideally) the
        # consistency check confirms the two agree.
        do_compute_jacobians=True,
        verbose=False,
        super_verbose=False,
    )
    if fpf_hps:
        default_hps.update(fpf_hps)

    fpf = FixedPointFinder(rnn, **default_hps)
    # cuDNN cannot backprop through an RNN in eval mode; disable for the search.
    # note: when deduplicating (unique_fps, all_fps), we need to return a unique set: 
    with torch.backends.cudnn.flags(enabled=False):
        unique_fps, all_fps = fpf.find_fixed_points(initial_states, input_cond)
    return unique_fps


def verify_jacobian_consistency(net, prep, latent_outs, device=None,
                                n_inits=256, tol=1e-4, fpf_hps=None, seed=42):
    """
    One-off check that OUR Jacobian (used for the frequency analysis) matches
    FixedPointFinder's internal Jacobian at the same fixed points.

    Both compute dF/dh of the same single RNN step via autograd, so they should
    agree to numerical precision. This runs a small static-input search with
    FPF's Jacobian computation enabled, then compares element-wise Jacobians and
    eigenvalues.

    Returns:
        dict with max_abs_jac_diff, max_abs_eig_diff, and pass/fail.
    """
    _check_single_layer(net)
    if device is None:
        device = next(net.parameters()).device

    rnn = net.rnn
    encoded = encode_inputs(net, prep['inputs_tensor'], device)
    static_input = encoded.reshape(-1, encoded.shape[-1]).mean(axis=0)[None, :].astype(np.float32)
    inits = sample_initial_states(latent_outs, n_inits=n_inits, seed=seed)

    # Run FPF WITH Jacobian computation enabled so fps.J_xstar is populated.
    hps = dict(tol_q=1e-12, tol_dq=1e-20, max_iters=5000, tol_unique=1e-3,
               do_compute_jacobians=True, verbose=False, super_verbose=False)
    if fpf_hps:
        hps.update(fpf_hps)
        hps['do_compute_jacobians'] = True  # force on for this check

    fpf = FixedPointFinder(rnn, **hps)
    with torch.backends.cudnn.flags(enabled=False):
         unique_fps, all_fps = fpf.find_fixed_points(inits, static_input)
    fps = unique_fps

    fpf_J = np.asarray(fps.J_xstar)   # (n, n_states, n_states)
    xstar = np.asarray(fps.xstar)

    max_jac_diff = 0.0
    max_eig_diff = 0.0
    for i in range(len(xstar)):
        our_J = _jacobian_at(rnn, xstar[i], static_input[0], device)
        max_jac_diff = max(max_jac_diff, float(np.max(np.abs(our_J - fpf_J[i]))))
        our_eig = np.sort_complex(np.linalg.eigvals(our_J))
        fpf_eig = np.sort_complex(np.linalg.eigvals(fpf_J[i]))
        max_eig_diff = max(max_eig_diff, float(np.max(np.abs(our_eig - fpf_eig))))

    passed = (max_jac_diff < tol) and (max_eig_diff < tol)
    print(f"  Jacobian consistency check ({len(xstar)} fixed points):")
    print(f"    max |J_ours - J_fpf|      = {max_jac_diff:.2e}")
    print(f"    max |eig_ours - eig_fpf|  = {max_eig_diff:.2e}")
    print(f"    {'PASS' if passed else 'FAIL'} (tol={tol:.0e})")
    return dict(max_abs_jac_diff=max_jac_diff,
                max_abs_eig_diff=max_eig_diff,
                passed=passed, n_fixed_points=len(xstar))


def _plot_fpf_native(fps, latent_outs, save_path, trial_idx=None,
                     max_trials=8):
    """
    Generate FixedPointFinder's native fixed-point plot and save it.

    Uses plot_utils.plot_fps, which fits PCA to the trajectories, plots stable
    fixed points (black), unstable fixed points (red) and their unstable modes
    (red line segments), with example trajectories (blue).

    NOTE: requires that the FixedPoints object was produced with Jacobian
    computation ENABLED (plot_fixed_point reads fp.J_xstar / eigval_J_xstar).
    """
    import matplotlib
    matplotlib.use('Agg')   # headless: save without a display
    import matplotlib.pyplot as plt
    try:
        from plot_utils import plot_fps
    except ImportError as e:
        print(f"    [plot] could not import plot_utils.plot_fps: {e}")
        return

    # latent_outs: (time, trials, hidden) -> state_traj: (n_batch, n_time, n_states)
    state_traj = np.transpose(latent_outs, (1, 0, 2))
    if trial_idx is not None:
        batch_idx = [trial_idx]
    else:
        batch_idx = list(range(min(max_trials, state_traj.shape[0])))

    try:
        fig = plot_fps(fps, state_traj=state_traj, plot_batch_idx=batch_idx)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"    [plot] saved {save_path}")
    except Exception as e:
        print(f"    [plot] plot_fps failed: {e}")


# ================================================================
# Top-level entry point
# ================================================================

def run_fpf_analysis(net, prep, latent_outs, dataset_name, taste_ind,
                     output_dir, device=None, changepoints_ms=None,
                     time_lims=(1500, 7000), bin_size=25, stim_ms=2000,
                     trial_idx=0, n_inits=1024, offset_lo_ms=50,
                     offset_hi_ms=100, fpf_hps=None, seed=42):
    """
    Run static + windowed fixed-point analysis on a SINGLE-LAYER model.

    Saves a .npz of fixed points / eigenvalues / frequencies per probe and a
    JSON summary flagging oscillatory (limit-cycle) fixed points. Plot wiring is
    intentionally minimal here — richer plots are layered on separately.

    Args:
        net: trained autoencoderRNN (rnn_layers == 1)
        prep: preprocessing dict (uses prep['inputs_tensor'])
        latent_outs: (time, trials, hidden) numpy array
        dataset_name, taste_ind: identifiers
        output_dir: output directory
        device: torch device
        changepoints_ms: list of HMM changepoint times in ms (for this trial),
            or None to run only the static + (optional) stimulus analysis
        time_lims, bin_size, stim_ms: timing info for ms->bin conversion
        trial_idx: which trial to condition on (None = average across trials)
        n_inits: initializations per FPF run
        offset_lo_ms, offset_hi_ms: changepoint window edges (ms)
        fpf_hps: optional FixedPointFinder hyperparameter overrides
        seed: RNG seed for initial-state sampling

    Returns:
        dict: {'static': summary, 'probes': {label: summary, ...}}
    """
    _check_single_layer(net)

    if device is None:
        device = next(net.parameters()).device
    trial_tag = f'_trial{trial_idx}' if trial_idx is not None else ''
    fp_dir = os.path.join(output_dir, 'fixed_points')
    os.makedirs(fp_dir, exist_ok=True)

    print(f"\n  FPF analysis (single-layer) — {dataset_name}, taste {taste_ind}")

    encoded = encode_inputs(net, prep['inputs_tensor'], device)   # (time, trials, hidden)
    n_time = encoded.shape[0]
    rnn = net.rnn

    results = {'static': None, 'probes': {}}

    # ---------------- Static: mean input over all time and trials ------------
    print("    Static (mean input over all time/trials):")
    static_input = encoded.reshape(-1, encoded.shape[-1]).mean(axis=0)[None, :].astype(np.float32)
    static_inits = sample_initial_states(latent_outs, n_inits=n_inits, seed=seed)
    static_fps = find_fps(rnn, static_inits, static_input, fpf_hps=fpf_hps)
    static_summary = summarize_fixed_points(static_fps, rnn, static_input,
                                            bin_size, device)
    results['static'] = static_summary
    _print_summary('static', static_summary)
    _plot_fpf_native(
        static_fps, latent_outs,
        save_path=os.path.join(fp_dir,
                               f'fpf_static_taste_{trial_tag}_{taste_ind}_{dataset_name}.png'),
        trial_idx=None,
    )

    # ---------------- Windowed probes around changepoints / stim -------------
    if changepoints_ms is not None or stim_ms is not None:
        cp_bins = [ms_to_bin(t, time_lims, bin_size) for t in (changepoints_ms or [])]
        stim_bin = ms_to_bin(stim_ms, time_lims, bin_size) if stim_ms is not None else None

        probes = build_probes(
            cp_bins, stim_bin, n_time, bin_size,
            offset_lo_ms=offset_lo_ms, offset_hi_ms=offset_hi_ms,
            include_mid=(len(cp_bins) > 0), include_stim=(stim_bin is not None),
        )

        for probe in probes:
            label = probe['label']
            t_idx = probe['t_idx']
            input_cond = build_input_condition(encoded, t_idx, trial_idx=trial_idx)
            # seed near this probe's window for relevance
            t_window = (int(t_idx.min()), int(t_idx.max()) + 1)
            inits = sample_initial_states(latent_outs, n_inits=n_inits,
                                          t_window=t_window, trial_idx=trial_idx,
                                          seed=seed)
            fps = find_fps(rnn, inits, input_cond, fpf_hps=fpf_hps)
            summ = summarize_fixed_points(fps, rnn, input_cond, bin_size, device)
            summ_meta = dict(t_center=int(probe['t_center']),
                             t_idx=t_idx.tolist(), summary=summ)
            results['probes'][label] = summ_meta
            _print_summary(label, summ, t_center=probe['t_center'])
            _plot_fpf_native(
                fps, latent_outs,
                save_path=os.path.join(
                    fp_dir,
                    f'fpf_{label}_{trial_tag}_taste_{taste_ind}_{dataset_name}.png'),
                trial_idx=trial_idx,
            )

    # ---------------- Save -----------------------------------------------------
    _save_results(results, fp_dir, dataset_name, taste_ind, bin_size,
                  trial_idx, time_lims)

    return results


# ================================================================
# Printing + saving
# ================================================================

def _print_summary(label, summary, t_center=None):
    """Print a compact per-probe summary, flagging limit-cycle signatures."""
    n = len(summary)
    n_stable = sum(1 for s in summary if s['stability'] == 'stable')
    n_saddle = sum(1 for s in summary if s['stability'] == 'saddle')
    n_unstable = sum(1 for s in summary if s['stability'] == 'unstable')
    n_spiral = sum(1 for s in summary if s['is_spiral'])
    where = f" @bin {t_center}" if t_center is not None else ""
    print(f"      [{label}{where}] {n} FPs: "
          f"{n_stable} stable, {n_saddle} saddle, {n_unstable} unstable; "
          f"{n_spiral} oscillatory")
    # Highlight limit-cycle candidates (unstable spirals) and their frequencies
    for s in summary:
        if s['is_unstable_spiral']:
            freqs = ", ".join(f"{f:.2f} Hz (|λ|={m:.3f})"
                              for f, m in s['frequencies_hz'])
            print(f"        -> LIMIT-CYCLE candidate (unstable spiral): {freqs}")


def _save_results(results, fp_dir, dataset_name, taste_ind, bin_size,
                  trial_idx, time_lims):
    """Save full results to .npz and a JSON summary."""
    # ---- npz (full arrays) ----
    npz = {}

    def _dump(prefix, summary):
        for s in summary:
            i = s['index']
            npz[f'{prefix}_fp{i}_xstar'] = s['xstar']
            npz[f'{prefix}_fp{i}_eigs'] = s['eigenvalues']

    if results['static']:
        _dump('static', results['static'])
    for label, meta in results['probes'].items():
        _dump(label, meta['summary'])

    npz_path = os.path.join(fp_dir,
                            f'fpf_taste_{taste_ind}_{dataset_name}.npz')
    np.savez(npz_path, **npz)

    # ---- JSON summary (scalars + frequencies, for quick inspection) ----
    def _ser(summary):
        out = []
        for s in summary:
            out.append(dict(
                index=s['index'],
                q=s['q'],
                converged=s['converged'],
                max_abs_eig=s['max_abs_eig'],
                stability=s['stability'],
                is_spiral=s['is_spiral'],
                is_unstable_spiral=s['is_unstable_spiral'],
                frequencies_hz=[{'freq_hz': f, 'magnitude': m}
                                for f, m in s['frequencies_hz']],
            ))
        return out

    json_obj = dict(
        dataset=dataset_name,
        taste=taste_ind,
        bin_size=bin_size,
        trial_idx=trial_idx,
        time_lims=list(time_lims),
        static=_ser(results['static']) if results['static'] else [],
        probes={label: dict(t_center=meta['t_center'],
                            fixed_points=_ser(meta['summary']))
                for label, meta in results['probes'].items()},
    )
    json_path = os.path.join(fp_dir,
                             f'fpf_summary_taste_{taste_ind}_{dataset_name}.json')
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=2)

    print(f"    Saved: {npz_path}")
    print(f"    Saved: {json_path}")