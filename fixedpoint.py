"""
Fixed point analysis for autoencoderRNN latent dynamics.
 
Finds fixed points of the RNN recurrence, computes Jacobians, classifies
stability, and supports rolling (input-conditioned) analysis across timesteps.

The idea here is that I want to see how the network is evolving over time and if that 
can somehow furhter support the notion that the RNN is in fact picking up on latent structure 
within the firing of the cells. 

Ideally, we find some sort of latent structure that has/with bifrucations in it (?) 
 
Based on the method from:
    Sussillo & Barak (2013) "Opening the Black Box: Low-Dimensional Dynamics
    in High-Dimensional Recurrent Neural Networks." Neural Computation.
 
Usage:
    from fixed_points import find_fixed_points, rolling_fixed_points, plot_fixed_points
 
    # Static analysis (autonomous or constant input):
    fps = find_fixed_points(net, latent_trajectories, input_condition=None)
 
    # Input-conditioned at a specific timestep:
    fps = find_fixed_points(net, latent_trajectories, input_condition=encoded_input_t)
 
    # Rolling analysis across all timesteps:
    rolling = rolling_fixed_points(net, latent_trajectories, encoded_inputs,
                                    time_indices=[2250, 3000, 3400, 3800])
                                    # time indicies should be when the hmm detects changepoints 
"""

import numpy as np
import torch 
import torch.nn as nn 
from torch.autograd.functional import jacobian
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 

# helpers! 
def extract_rnn_cell(net): 
    """
    Extract the RNN cell (single-step recurrence) from the autoencoderRNN.
    """
    rnn = net.rnn # nn.RNN module from torch 
    hidden_size = rnn.hidden_size
    num_layers = rnn.num_layers
    # could put an if statement if for some reason someone (me) tries to feed
    # something (say, bidirectional) RNN through-- but with this analysis, that's 
    # never actually going to happen as that defeats the whole point. 
    # still: 
    if rnn.bidirectional: 
        raise ValueError("Fixed point analysis is 1. not supported for bidirectional RNN and 2. irrelevant for this study ya numpty")
    return rnn, hidden_size, num_layers

def encode_input(net, raw_input, device=None): 
    """
    Run the encoder portion of the autoencoderRNN to get encoded input.
 
    Args:
        net: trained autoencoderRNN
        raw_input: (time, trials, input_size) tensor — the model inputs
        device: torch device
 
    Returns:
        encoded: (time, trials, hidden_size) tensor — encoder output
    """
    if device is None:
        # do whatever the default is. hope it's cuda! 
        device = next(net.parameters()).device
    net.eval()
    with torch.no_grad(): 
        encoded = net.encoder(raw_input.to(device))
    # pass back to cpu cuz yah
    return encoded.cpu()


# ----------------------------------------------------------------
# Core: find fixed points
# ----------------------------------------------------------------
 
def _rnn_step(rnn, h, x, num_layers):
    """
    Single RNN step: h_next = F(h, x).
 
    Args:
        rnn: torch.nn.RNN or GRU module
        h: (n_points, hidden_size) — current hidden states
        x: (n_points, encoded_dim) — encoded input (constant across optimization)
        num_layers: int
 
    Returns:
        h_next: (n_points, hidden_size)
    """
    n_points = h.shape[0]
    hidden_size = h.shape[1]
 
    # RNN expects input: (seq_len=1, batch=n_points, input_size)
    # and hidden: (num_layers, batch=n_points, hidden_size)
    x_in = x.unsqueeze(0)  # (1, n_points, encoded_dim)
    h_in = h.unsqueeze(0).expand(num_layers, -1, -1).contiguous()  # (num_layers, n_points, hidden_size)
 
    _, h_out = rnn(x_in, h_in)
 
    # h_out: (num_layers, n_points, hidden_size) — take last layer
    return h_out[-1]  # (n_points, hidden_size)


def find_fixed_points(
        net,
        latent_trajectories,
        input_condition=None,
        n_inits=200,
        lr=0.01,
        max_steps=5000,
        tol=1e-6,
        patience=500,
        noise_scale=0.1,
        cluster_eps=0.05,
        device=None,
        verbose=True,
        ):
    """
    Find fixed points of the RNN dynamics.

    NOTE: this is based on Sussilo & Barack 2013. On the shoulders of giants do I hope to 
    make my small difference. 
 
    Minimizes q(h) = ||F(h, x) - h||xp(2) from multiple initial conditions
    sampled from actual latent trajectories (with optional noise).
 
    Args:
        net: trained autoencoderRNN
        latent_trajectories: (time, trials, hidden_size) numpy array —
            latent states from run_prediction. Used to sample initial conditions.
        input_condition: (encoded_dim,) tensor or None.
            If provided, find fixed points of F(h, x) with x fixed to this value.
            If None, find fixed points of the autonomous system (x=0).
        n_inits: int, number of optimization initializations
        lr: float, learning rate for Adam
        max_steps: int, max optimization steps
        tol: float, q threshold — solutions below this are considered fixed points
        patience: int, stop if q hasn't improved in this many steps
        noise_scale: float, std of noise added to initial conditions
        cluster_eps: float, DBSCAN eps for clustering nearby fixed points
        device: torch device
        verbose: bool
 
    Returns:
        dict with:
            fixed_points: (n_fps, hidden_size) numpy array
            q_values: (n_fps,) — how close each is to a true fixed point
            eigenvalues: list of (hidden_size,) complex arrays — Jacobian eigenvalues
            stability: list of str — 'stable', 'unstable', or 'saddle'
            jacobians: list of (hidden_size, hidden_size) numpy arrays
            n_found_raw: int — number before clustering
    """
    if device is None:
        device = next(net.parameters()).device
 
    rnn, hidden_size, num_layers = extract_rnn_cell(net)
    rnn.eval()
 
    # --- Sample initial conditions from trajectories ---
    traj_flat = latent_trajectories.reshape(-1, hidden_size)  # (time*trials, hidden_size)
    n_available = traj_flat.shape[0]
    indices = np.random.choice(n_available, size=n_inits, replace=(n_inits > n_available))
    inits = traj_flat[indices].copy()
    inits += np.random.randn(*inits.shape) * noise_scale
 
    # --- Prepare input condition ---
    if input_condition is not None:
        x_const = input_condition.float().to(device)
        if x_const.dim() == 1:
            x_const = x_const.unsqueeze(0).expand(n_inits, -1)  # (n_inits, encoded_dim)
    else:
        # double check this here -- as decoder input is the same size as hidden size but not autoencoder output 
        x_const = torch.zeros(n_inits, hidden_size, device=device)
 
    # --- Optimize ---
    h = torch.tensor(inits, dtype=torch.float32, device=device, requires_grad=True)
    # using the Adam optimizer which if nothing else is fast and probably totally sufficient for this
    # stan and Metroplis-Hastings has totally ruined me 
    optimizer = torch.optim.Adam([h], lr=lr)
 
    best_q = torch.full((n_inits,), float('inf'), device=device)
    best_h = h.detach().clone()
    steps_since_improvement = torch.zeros(n_inits, dtype=torch.int, device=device)
    active = torch.ones(n_inits, dtype=torch.bool, device=device)
    # to fix a bug to do with cudnn: 
    # cuDNN RNN backward requires training mode; disable cuDNN to allow
    # gradient computation through the RNN in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        for step in range(max_steps):
            optimizer.zero_grad()
    
            h_next = _rnn_step(rnn, h, x_const, num_layers)
            q_per_point = torch.sum((h_next - h) ** 2, dim=-1)  # (n_inits,)
            q_total = q_per_point[active].sum()
    
            q_total.backward()
            optimizer.step()
    
            # Track best per initialization
            with torch.no_grad():
                improved = q_per_point < best_q
                best_q[improved] = q_per_point[improved]
                best_h[improved] = h[improved].clone()
                steps_since_improvement[improved] = 0
                steps_since_improvement[~improved] += 1
    
                # Deactivate converged or stuck points
                active = (best_q > tol) & (steps_since_improvement < patience)
    
                if not active.any():
                    if verbose:
                        print(f"    All points converged/stopped at step {step}")
                    break
    
            if verbose and step % 1000 == 0:
                n_active = active.sum().item()
                min_q = best_q.min().item()
                n_found = (best_q < tol).sum().item()
                print(f"    Step {step}: {n_active} active, min q={min_q:.2e}, "
                    f"{n_found} below tol")
 
    # --- Filter: keep only points with q < tol ---
    best_h_np = best_h.detach().cpu().numpy()
    best_q_np = best_q.detach().cpu().numpy()
    mask = best_q_np < tol
    n_found_raw = mask.sum()
 
    if n_found_raw == 0:
        if verbose:
            # Also report slow points
            slow_mask = best_q_np < (tol * 100)
            print(f"    No fixed points found (q < {tol}). "
                  f"{slow_mask.sum()} slow points (q < {tol*100:.1e}).")
            print(f"    Min q: {best_q_np.min():.2e}")
        return {
            'fixed_points': np.array([]).reshape(0, hidden_size),
            'q_values': np.array([]),
            'eigenvalues': [],
            'stability': [],
            'jacobians': [],
            'n_found_raw': 0,
            'slow_points': best_h_np[best_q_np < (tol * 100)],
            'slow_q_values': best_q_np[best_q_np < (tol * 100)],
        }
 
    found_h = best_h_np[mask]
    found_q = best_q_np[mask]
 
    if verbose:
        print(f"    Found {n_found_raw} raw fixed points")
 
    # --- Cluster nearby fixed points ---
    if len(found_h) > 1:
        clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(found_h)
        labels = clustering.labels_
        unique_labels = set(labels)
        clustered_h = []
        clustered_q = []
        for label in unique_labels:
            members = found_h[labels == label]
            q_members = found_q[labels == label]
            # Keep the one with lowest q
            best_idx = np.argmin(q_members)
            clustered_h.append(members[best_idx])
            clustered_q.append(q_members[best_idx])
        found_h = np.array(clustered_h)
        found_q = np.array(clustered_q)
 
    if verbose:
        print(f"    After clustering: {len(found_h)} unique fixed points")
 
    # --- Compute Jacobians and classify stability ---
    eigenvalues_list = []
    stability_list = []
    jacobians_list = []
 
    for i, fp in enumerate(found_h):
        jac = _compute_jacobian(rnn, fp, x_const[0].cpu() if x_const is not None else None,
                                hidden_size, num_layers, device)
        jacobians_list.append(jac)
 
        eigs = np.linalg.eigvals(jac)
        eigenvalues_list.append(eigs)
 
        # Classify: for discrete-time systems, stability is |eigenvalue| < 1
        max_abs_eig = np.max(np.abs(eigs))
        if max_abs_eig < 1.0:
            stability_list.append('stable')
        elif np.all(np.abs(eigs) > 1.0):
            stability_list.append('unstable')
        else:
            stability_list.append('saddle')
 
        if verbose:
            print(f"    FP {i}: q={found_q[i]:.2e}, "
                  f"max|eig|={max_abs_eig:.4f}, {stability_list[-1]}")
 
    return {
        'fixed_points': found_h,
        'q_values': found_q,
        'eigenvalues': eigenvalues_list,
        'stability': stability_list,
        'jacobians': jacobians_list,
        'n_found_raw': int(n_found_raw),
    }

## to review code: 
def _compute_jacobian(rnn, fp, x_const, hidden_size, num_layers, device):
    """
    Compute the Jacobian dF/dh at a fixed point.
    Review notes: seems reasonable 
    Returns:
        jac: (hidden_size, hidden_size) numpy array
    """
    fp_t = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0)  # (1, hidden_size)
 
    if x_const is not None:
        x_t = x_const.float().to(device).unsqueeze(0)  # (1, encoded_dim)
    else:
        x_t = torch.zeros(1, hidden_size, device=device)
 
    def f(h_flat):
        h = h_flat.unsqueeze(0)  # (1, hidden_size)
        h_next = _rnn_step(rnn, h, x_t, num_layers)
        return h_next.squeeze(0)  # (hidden_size,)
    # guard here: 
    with torch.backends.cudnn.flags(enabled=False):
        jac = jacobian(f, fp_t.squeeze(0))  # (hidden_size, hidden_size)
    return jac.detach().cpu().numpy()
 
 
# ----------------------------------------------------------------
# Rolling fixed point analysis
# ----------------------------------------------------------------
 
def rolling_fixed_points(
        net,
        latent_trajectories,
        encoded_inputs,
        time_indices=None,
        trial_idx=0,
        **find_kwargs,
        ):
    """
    # NOTE: Ultimately, I think I might like to pass the RNN's changepoints into this somehow. 
    # as that's probably among the more efficient ways of doing this. 
    Run fixed point analysis at multiple timesteps with input-conditioned dynamics.
 
    At each selected timestep, the encoded input is frozen and fixed points
    are found for the dynamics conditioned on that input.
 
    Args:
        net: trained autoencoderRNN
        latent_trajectories: (time, trials, hidden_size) numpy array
        encoded_inputs: (time, trials, encoded_dim) tensor — output of encoder
        time_indices: list of int — which timesteps to analyze. If None,
            uses 10 evenly spaced points.
        trial_idx: int — which trial to use for the input condition
        **find_kwargs: passed to find_fixed_points
 
    Returns:
        results: list of dicts (one per timestep), each from find_fixed_points,
                 with added 'time_idx' key
    """
    n_time = latent_trajectories.shape[0]
 
    if time_indices is None:
        time_indices = np.linspace(0, n_time - 1, 10, dtype=int).tolist()
 
    results = []
    for t_idx in time_indices:
        print(f"\n  Time index {t_idx}/{n_time}:")
        x_t = encoded_inputs[t_idx, trial_idx]  # (encoded_dim,)
 
        fps = find_fixed_points(
            net, latent_trajectories,
            input_condition=x_t,
            **find_kwargs,
        )
        fps['time_idx'] = t_idx
        results.append(fps)
 
    return results
 
 
# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------
 
def plot_fixed_points(fps_result, latent_trajectories=None, pca_dims=(0, 1, 2),
                      title='Fixed Points', output_dir=None, filename=None):
    """
    Plot fixed points in latent space (3D PCA projection).
 
    Args:
        fps_result: dict from find_fixed_points
        latent_trajectories: (time, trials, hidden_size) — optional, overlay trajectories
        pca_dims: tuple of 3 ints — which latent dims to plot (or PCA components)
        title: str
        output_dir: str — if provided, save figure
        filename: str — filename for saved figure
    """
    fps = fps_result['fixed_points']
    stability = fps_result['stability']
 
    if len(fps) == 0:
        print("  No fixed points to plot.")
        return
 
    d0, d1, d2 = pca_dims
 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
 
    # Plot trajectories if provided
    if latent_trajectories is not None:
        n_time, n_trials, _ = latent_trajectories.shape
        for trial in range(min(n_trials, 10)):  # plot up to 10 trials
            traj = latent_trajectories[:, trial, :]
            ax.plot(traj[:, d0], traj[:, d1], traj[:, d2],
                    alpha=0.3, linewidth=0.5, color='steelblue')
 
    # Plot fixed points colored by stability
    # NOTE THIS KEY AS IT'S VERY IMPORTANT 
    # (also there is a legend)
    colors = {'stable': 'green', 'unstable': 'red', 'saddle': 'orange'}
    markers = {'stable': 'o', 'unstable': 'x', 'saddle': '^'}
    sizes = {'stable': 100, 'unstable': 80, 'saddle': 80}
 
    for stab_type in ['stable', 'saddle', 'unstable']:
        mask = [i for i, s in enumerate(stability) if s == stab_type]
        if mask:
            pts = fps[mask]
            ax.scatter(pts[:, d0], pts[:, d1], pts[:, d2],
                       c=colors[stab_type], marker=markers[stab_type],
                       s=sizes[stab_type], label=stab_type,
                       edgecolors='k', linewidths=0.5, zorder=5)
 
    ax.set_xlabel(f'Latent dim {d0}')
    ax.set_ylabel(f'Latent dim {d1}')
    ax.set_zlabel(f'Latent dim {d2}')
    ax.set_title(title)
    ax.legend()
 
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=200)
    plt.close(fig)
 
 
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
 
 
# ----------------------------------------------------------------
# Integration helper: call from run_rnn.py after training-- literaly just pop this in that function 
# ----------------------------------------------------------------
 
def run_fixed_point_analysis(net, prep, latent_outs, dataset_name, taste_ind,
                             output_dir, device=None, time_indices=None):
    """
    Run both static and rolling fixed point analysis on a trained model.
 
    Call this from run_rnn.py after training and prediction:
 
        from fixed_points import run_fixed_point_analysis
 
        run_fixed_point_analysis(
            net=net,
            prep=prep,
            latent_outs=latent_outs,
            dataset_name=dataset_name,
            taste_ind=taste_ind,
            output_dir=fp_dir,
            device=device,
            time_indices=[some collection similar to that of the changepoints...],
        )
 
    Args:
        net: trained autoencoderRNN
        prep: dict from preprocess_taste (contains inputs_tensor)
        latent_outs: (time, trials, hidden_size) numpy array from run_prediction
        dataset_name: str
        taste_ind: int
        output_dir: str
        device: torch device
        time_indices: list of int — timesteps for rolling analysis
    """
    if device is None:
        device = next(net.parameters()).device
 
    fp_dir = os.path.join(output_dir, 'fixed_points')
    os.makedirs(fp_dir, exist_ok=True)
 
    print(f"\n  Fixed point analysis — {dataset_name}, taste {taste_ind}")
 
    # --- Get encoded inputs ---
    encoded = encode_input(net, prep['inputs_tensor'], device)
    # encoded: (time, trials, encoded_dim)
 
    # --- Static analysis (mean input across all time and trials) ---
    print("\n  Static analysis (mean input):")
    mean_input = encoded.mean(dim=(0, 1))  # (encoded_dim,)
    static_fps = find_fixed_points(
        net, latent_outs, input_condition=mean_input,
        n_inits=200, tol=1e-6, verbose=True, device=device,
    )
    plot_fixed_points(
        static_fps, latent_outs,
        title=f'Fixed Points (mean input) — {dataset_name} taste {taste_ind}',
        output_dir=fp_dir,
        filename=f'fps_static_taste_{taste_ind}_{dataset_name}.png',
    )
 
    # --- Rolling analysis ---
    print("\n  Rolling analysis:")
    rolling = rolling_fixed_points(
        net, latent_outs, encoded,
        time_indices=time_indices,
        trial_idx=0,
        n_inits=100,  # fewer inits per timestep for speed
        tol=1e-5,     # slightly relaxed tolerance
        verbose=True,
        device=device,
    )
    plot_rolling_fixed_points(
        rolling, latent_outs,
        title=f'Rolling Fixed Points — {dataset_name} taste {taste_ind}',
        output_dir=fp_dir,
        filename=f'fps_rolling_taste_{taste_ind}_{dataset_name}.png',
    )
 
    # --- Save results ---
    save_path = os.path.join(fp_dir,
                             f'fps_taste_{taste_ind}_{dataset_name}.npz')
    save_dict = {
        'static_fps': static_fps['fixed_points'],
        'static_q': static_fps['q_values'],
        'static_stability': np.array(static_fps['stability']),
    }
    for i, r in enumerate(rolling):
        save_dict[f'rolling_{i}_fps'] = r['fixed_points']
        save_dict[f'rolling_{i}_q'] = r['q_values']
        save_dict[f'rolling_{i}_time'] = r['time_idx']
        save_dict[f'rolling_{i}_stability'] = np.array(r['stability']) if r['stability'] else np.array([])
    np.savez(save_path, **save_dict)
    print(f"  Saved fixed point results to {save_path}")
 
    return static_fps, rolling