
import numpy as np
import torch 
import torch.nn as nn 
from torch.autograd.functional import jacobian
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 

#### THE FIXED POINT FINDER STUFF
# I have to update this whoooooooooooolleleeeeee thing for the fixed point finder data strucures 
# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

# func that adapts these old plots (and the data structure they expect) to the new one: 
def fpf_results_to_rolling(results):
    """
    Adapt run_fpf_analysis() output into the list-of-dicts 'rolling_results'
    format consumed by plot_rolling_fixed_points / plot_eigenvalue_complex_plane
    / plot_distance_to_trajectory.

    results: dict returned by fpf_analysis.run_fpf_analysis
    Returns: list of dicts, one per probe, sorted by time, with keys
        time_idx, fixed_points (n,hidden), eigenvalues (list), stability (list).
    """
    rolling = []
    for label, meta in results.get('probes', {}).items():
        summary = meta['summary']
        if len(summary) == 0:
            continue
        rolling.append(dict(
            time_idx=meta['t_center'],
            label=label,
            fixed_points=np.array([s['xstar'] for s in summary]),
            eigenvalues=[s['eigenvalues'] for s in summary],
            stability=[s['stability'] for s in summary],
            q_values=np.array([s['q'] for s in summary]),
            frequencies=[s['frequencies_hz'] for s in summary], # frequencies object from fpf is also useful
        ))
    rolling.sort(key=lambda r: r['time_idx'])   # order by time for the over-time plots
    return rolling

 
def plot_rolling_fixed_points(rolling_results, latent_trajectories=None,
                              pca_dims=(0, 1, 2), title='Rolling Fixed Points',
                              output_dir=None, filename=None):
    """
    Plot how fixed points evolve over time.
 
    Figure 1: 3D view with fixed points colored by time.
    Figure 2: Fixed point positions over time (per latent dim).
    Figure 3: Stability eigenvalues over time.
    """
    if not rolling_results or all(len(r['fixed_points']) == 0 for r in rolling_results):
        print("  No fixed points found in rolling analysis.")
        return
 
    d0, d1, d2 = pca_dims
    time_indices = [r['time_idx'] for r in rolling_results]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(time_indices), max(time_indices))
 
    # --- Figure 1: 3D with time-colored fixed points ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
 
    if latent_trajectories is not None:
        n_trials = latent_trajectories.shape[1]
        for trial in range(min(n_trials, 5)):
            traj = latent_trajectories[:, trial, :]
            ax.plot(traj[:, d0], traj[:, d1], traj[:, d2],
                    alpha=0.2, linewidth=0.5, color='gray')
 
    for r in rolling_results:
        fps = r['fixed_points']
        t_idx = r['time_idx']
        stability = r['stability']
        if len(fps) == 0:
            continue
        color = cmap(norm(t_idx))
        for i, (fp, stab) in enumerate(zip(fps, stability)):
            marker = 'o' if stab == 'stable' else ('^' if stab == 'saddle' else 'x')
            ax.scatter(fp[d0], fp[d1], fp[d2], c=[color], marker=marker,
                       s=80, edgecolors='k', linewidths=0.5, zorder=5)
 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time index', shrink=0.6)
    ax.set_xlabel(f'Latent dim {d0}')
    ax.set_ylabel(f'Latent dim {d1}')
    ax.set_zlabel(f'Latent dim {d2}')
    ax.set_title(title)
 
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
 
    # --- Figure 2: Max eigenvalue magnitude over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rolling_results:
        t_idx = r['time_idx']
        for i, (eigs, stab) in enumerate(zip(r['eigenvalues'], r['stability'])):
            max_eig = np.max(np.abs(eigs))
            color = {'stable': 'green', 'saddle': 'orange', 'unstable': 'red'}[stab]
            ax.scatter(t_idx, max_eig, c=color, s=50, edgecolors='k', linewidths=0.5)
 
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Stability boundary')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Max |eigenvalue|')
    ax.set_title('Fixed Point Stability Over Time')
    ax.legend()
 
    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 filename.replace('.png', '_eigenvalues.png')
                                 if filename else 'eigenvalues_over_time.png'),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
 
    # --- Figure 3: Number of fixed points over time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    n_stable = [sum(1 for s in r['stability'] if s == 'stable') for r in rolling_results]
    n_saddle = [sum(1 for s in r['stability'] if s == 'saddle') for r in rolling_results]
    n_unstable = [sum(1 for s in r['stability'] if s == 'unstable') for r in rolling_results]
 
    ax.bar(time_indices, n_stable, label='Stable', color='green', alpha=0.7)
    ax.bar(time_indices, n_saddle, bottom=n_stable, label='Saddle', color='orange', alpha=0.7)
    ax.bar(time_indices, n_unstable,
           bottom=[a + b for a, b in zip(n_stable, n_saddle)],
           label='Unstable', color='red', alpha=0.7)
    ax.set_xlabel('Time index')
    ax.set_ylabel('Number of fixed points')
    ax.set_title('Fixed Point Count Over Time')
    ax.legend()
 
    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 filename.replace('.png', '_counts.png')
                                 if filename else 'fp_counts_over_time.png'),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
 
 
def plot_eigenvalue_complex_plane(
        rolling_results,
        hmm_changepoints=None,
        title='Eigenvalue Trajectories',
        output_dir=None,
        filename='eigenvalues_complex_plane.png',
        track=True,
        ):
    """
    Plot Jacobian eigenvalues in the complex plane across timepoints.
 
    The unit circle is the stability boundary: eigenvalues inside are
    contracting (stable), outside are expanding (unstable). Watching an
    eigenvalue cross the circle is a direct view of a bifurcation.
 
    Args:
        rolling_results: list of dicts from rolling_fixed_points
        hmm_changepoints: optional list of time indices where HMM detects
            transitions. If a rolling timepoint matches a changepoint, its
            eigenvalues are highlighted.
        title: str
        output_dir: str
        filename: str
        track: bool — if True, connect eigenvalues across consecutive
            timepoints with faint lines (nearest-neighbor matching)
 
    Notes:
        Eigenvalues come in complex-conjugate pairs for real matrices, so
        the plot is symmetric about the real axis. A pair off the real axis
        means rotational/oscillatory dynamics around that fixed point.
    """
    valid = [r for r in rolling_results if len(r['fixed_points']) > 0]
    if not valid:
        print("  No fixed points to plot eigenvalues for.")
        return
 
    time_indices = [r['time_idx'] for r in valid]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(time_indices), max(time_indices))
 
    fig, ax = plt.subplots(figsize=(9, 9))
 
    # Unit circle (stability boundary)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5,
            linewidth=1.5, label='Unit circle (|λ|=1)')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
 
    # For tracking: collect all eigenvalues per timepoint (across all FPs)
    # We plot every eigenvalue from every fixed point at every timepoint.
    prev_eigs = None
    for r in valid:
        t_idx = r['time_idx']
        color = cmap(norm(t_idx))
        is_changepoint = (hmm_changepoints is not None and
                          t_idx in hmm_changepoints)
 
        # Gather all eigenvalues at this timepoint
        all_eigs_this_t = []
        for eigs in r['eigenvalues']:
            all_eigs_this_t.extend(eigs)
        all_eigs_this_t = np.array(all_eigs_this_t)
 
        if len(all_eigs_this_t) == 0:
            continue
 
        # Plot
        marker_size = 90 if is_changepoint else 40
        edge = 'red' if is_changepoint else 'k'
        edge_w = 2.0 if is_changepoint else 0.5
        ax.scatter(all_eigs_this_t.real, all_eigs_this_t.imag,
                   c=[color], s=marker_size, edgecolors=edge,
                   linewidths=edge_w, alpha=0.8, zorder=4)
 
        # Track: connect to nearest eigenvalue at previous timepoint
        if track and prev_eigs is not None:
            for e in all_eigs_this_t:
                # Find nearest previous eigenvalue
                dists = np.abs(prev_eigs - e)
                nearest = prev_eigs[np.argmin(dists)]
                ax.plot([nearest.real, e.real], [nearest.imag, e.imag],
                        color=color, alpha=0.3, linewidth=0.8, zorder=2)
 
        prev_eigs = all_eigs_this_t
 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time index', shrink=0.7)
 
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
 
    # Annotate the changepoint highlighting in the legend area
    if hmm_changepoints is not None:
        ax.scatter([], [], c='gray', s=90, edgecolors='red', linewidths=2.0,
                   label='At HMM changepoint')
        ax.legend(loc='upper left', fontsize=9)
 
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved eigenvalue complex plane: {filename}")
 
 
# ----------------------------------------------------------------
# 2. Distance from trajectory to nearest fixed point over time
# ----------------------------------------------------------------
 
def plot_distance_to_trajectory(
        rolling_results,
        latent_trajectories,
        trial_idx=0,
        hmm_changepoints=None,
        title='Distance to Nearest Fixed Point',
        output_dir=None,
        filename='distance_to_fp.png',
        only_stable=False,
        ):
    """
    Plot the distance from the actual latent state to the nearest fixed point
    at each rolling timepoint.
 
    Low distance = the trajectory is sitting in/near an attractor (a "state").
    High distance = the trajectory is in transit between fixed points.
 
    Overlaying HMM changepoints tests the hypothesis that HMM states
    correspond to attractor-dwelling and transitions to between-attractor
    flight.
 
    Args:
        rolling_results: list of dicts from rolling_fixed_points
        latent_trajectories: (time, trials, hidden_size) numpy array
        trial_idx: int — which trial's trajectory to measure against
            (should match the trial_idx used in rolling_fixed_points)
        hmm_changepoints: optional list of time indices to overlay as vertical lines
        title: str
        output_dir: str
        filename: str
        only_stable: bool — if True, measure distance only to stable fixed
            points (attractors), ignoring saddles/unstable points
 
    Notes:
        The fixed points at timepoint t are those found conditioned on the
        input at timepoint t, so this measures how close the trajectory is
        to the fixed points of its *instantaneous* dynamics.
    """
    valid = [r for r in rolling_results if len(r['fixed_points']) > 0]
    if not valid:
        print("  No fixed points for distance plot.")
        return
 
    time_indices = []
    distances = []
    nearest_stability = []
 
    for r in valid:
        t_idx = r['time_idx']
        fps = r['fixed_points']
        stability = r['stability']
 
        # Optionally restrict to stable fixed points
        if only_stable:
            stable_mask = [i for i, s in enumerate(stability) if s == 'stable']
            if not stable_mask:
                continue
            fps_use = fps[stable_mask]
            stab_use = [stability[i] for i in stable_mask]
        else:
            fps_use = fps
            stab_use = stability
 
        # Actual latent state at this timepoint
        state = latent_trajectories[t_idx, trial_idx, :]  # (hidden_size,)
 
        # Distance to each fixed point, take the minimum
        dists = np.linalg.norm(fps_use - state[None, :], axis=1)
        min_idx = np.argmin(dists)
 
        time_indices.append(t_idx)
        distances.append(dists[min_idx])
        nearest_stability.append(stab_use[min_idx])
 
    if not time_indices:
        print("  No valid timepoints for distance plot.")
        return
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    # Color the line segments / points by the stability of the nearest FP
    stab_colors = {'stable': 'green', 'saddle': 'orange', 'unstable': 'red'}
    point_colors = [stab_colors.get(s, 'gray') for s in nearest_stability]
 
    # Connecting line
    ax.plot(time_indices, distances, '-', color='steelblue',
            linewidth=1.5, alpha=0.6, zorder=2)
    # Points colored by nearest-FP stability
    ax.scatter(time_indices, distances, c=point_colors, s=60,
               edgecolors='k', linewidths=0.5, zorder=3)
 
    # HMM changepoints as vertical lines
    if hmm_changepoints is not None:
        for i, cp in enumerate(hmm_changepoints):
            ax.axvline(cp, color='purple', linestyle='--', alpha=0.6,
                       linewidth=1.5,
                       label='HMM changepoint' if i == 0 else None)
 
    ax.set_xlabel('Time index')
    ax.set_ylabel('Distance to nearest fixed point')
    ax.set_title(title + (' (stable FPs only)' if only_stable else ''))
 
    # Build a legend with stability colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=8, label='Nearest = stable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label='Nearest = saddle'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Nearest = unstable'),
    ]
    if hmm_changepoints is not None:
        legend_elements.append(
            Line2D([0], [0], color='purple', linestyle='--',
                   label='HMM changepoint')
        )
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
 
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved distance-to-trajectory plot: {filename}")


def plot_frequencies_over_time(rolling_results, hmm_changepoints=None,
                               expected_band=None,
                               title='Oscillation Frequencies Over Time (circle size scale with |λ| magnitude)',
                               output_dir=None, filename='fps_frequencies.png'):
    """
    Implied oscillation frequencies (from complex Jacobian eigenvalues) of
    fixed points across probe timepoints.

    Each oscillatory fixed point contributes a point at (t_center, freq_hz).
        filled red  = unstable spiral (|λ|>1): LIMIT-CYCLE candidate
        open blue   = stable/damped spiral (|λ|<1): damped oscillation
    Marker size scales with |λ| (distance from the unit circle = how strongly
    expanding/contracting the rotation is).

    expected_band: optional (lo, hi) Hz to shade, e.g. (6, 7) for the rhythem we've seen in the past. 

    """
    t_lc, f_lc, m_lc = [], [], []     # limit-cycle candidates (|λ|>1)
    t_dp, f_dp, m_dp = [], [], []     # damped/stable spirals (|λ|<1)

    for r in rolling_results:
        t = r['time_idx']
        freqs_per_fp = r.get('frequencies', None)
        if not freqs_per_fp:
            continue
        for fp_freqs in freqs_per_fp:
            for (f_hz, mag) in fp_freqs:
                if mag > 1.0:
                    t_lc.append(t); f_lc.append(f_hz); m_lc.append(mag)
                else:
                    t_dp.append(t); f_dp.append(f_hz); m_dp.append(mag)

    if not (t_lc or t_dp):
        print("  No oscillatory fixed points found — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    if expected_band is not None:
        ax.axhspan(expected_band[0], expected_band[1], color='gold', alpha=0.2,
                   label=f'{expected_band[0]}–{expected_band[1]} Hz (expected)')

    def _sizes(mags):
        # exaggerate distance from unit circle for visibility
        return [40 + 400 * abs(m - 1.0) for m in mags]

    if t_dp:
        ax.scatter(t_dp, f_dp, s=_sizes(m_dp), facecolors='none',
                   edgecolors='steelblue', linewidths=1.5,
                   label='Damped spiral (|λ|<1)', zorder=3)
    if t_lc:
        ax.scatter(t_lc, f_lc, s=_sizes(m_lc), c='red', edgecolors='k',
                   linewidths=0.5, alpha=0.8,
                   label='Limit-cycle candidate (|λ|>1)', zorder=4)

    if hmm_changepoints is not None:
        for i, cp in enumerate(hmm_changepoints):
            ax.axvline(cp, color='purple', linestyle='--', alpha=0.6,
                       linewidth=1.5,
                       label='HMM changepoint' if i == 0 else None)

    ax.set_xlabel('Time (bin index)')
    ax.set_ylabel('Implied frequency (Hz)')
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(fontsize=9, loc='best')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved frequencies-over-time plot: {filename}")