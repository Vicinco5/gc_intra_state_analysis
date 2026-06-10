"""
Post-hoc per-neuron reconstruction evaluation.

This is NOT neuron leave-one-out in the sense of removing a neuron from
the model's input during training. (as an aside, this is something that I very much wish to do, but alas.)

for this mode: 
The model has already been trained on all neurons. Instead, this evaluates reconstruction quality
on a per-neuron basis by slicing the model's existing predictions.

The idea:
    1. Run a single forward pass through the trained model.
    2. For each neuron j, compute how well the model's predicted firing
       rate for neuron j matches the actual activity of neuron j.
    3. Report Gaussian LL (on z-scored data) and Poisson LL (on raw
       spike counts) separately for each neuron.

In doing this, I'm trying to ask the question:
"Which neurons does the model reconstruct well, and which does it struggle with?" without any re-training. 
It's kinda cheap, but ideally an effective way to estimate how well the model is fitting to the neurons in general. 
Neurons (especially multiple) with poor reconstruction may indicate:
    - The latent bottleneck is too small to capture that neuron's dynamics
    - The neuron has idiosyncratic activity not shared with the population
    - The neuron has very low or very high firing rates that are hard to
      predict under the current loss function

Because no retraining is involved, this runs in seconds and can be
called after any training run regardless of validation mode.

Usage (in run_rnn.py, after training and prediction):

    from neuron_eval import evaluate_neurons

    evaluate_neurons(
        net=net,
        inputs_tensor=prep['inputs_tensor'],
        labels_tensor=prep['labels_tensor'],
        raw_labels_tensor=prep['raw_labels_tensor'],
        scaler=prep['scaler'],
        pca_obj=prep['pca_obj'],
        binned_spikes=prep['binned_spikes'],
        dataset_name=dataset_name,
        taste_ind=taste_ind,
        output_dir=model_eval_dir,
        device=device,
    )
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train import gaussian_log_likelihood, poisson_log_likelihood


def evaluate_neurons(
        net,
        inputs_tensor,
        labels_tensor,
        raw_labels_tensor,
        scaler,
        pca_obj=None,
        binned_spikes=None,
        dataset_name='',
        taste_ind=0,
        output_dir='.',
        device=None,
        ):
    """
    Post-hoc per-neuron reconstruction evaluation.

    Runs one forward pass, then slices predictions and labels by neuron
    to compute per-neuron Gaussian and Poisson log-likelihoods.

    Args:
        net: trained model
        inputs_tensor: (time, trials, input_size) — model inputs
        labels_tensor: (time, trials, output_size) — z-scored labels
        raw_labels_tensor: (time, trials, n_neurons) — raw binned spike counts
        scaler: fitted StandardScaler
        pca_obj: fitted PCA object or None
        binned_spikes: (trials, neurons, time) — for computing mean firing rates
        dataset_name: str
        taste_ind: int
        output_dir: str, where to save plots
        device: torch device

    Returns:
        results: dict with per-neuron metrics
    """
    if device is None:
        device = torch.device('cpu')

    neuron_dir = os.path.join(output_dir, 'neuron_diagnostics')
    os.makedirs(neuron_dir, exist_ok=True)

    # --- Single forward pass ---
    net.eval()
    with torch.no_grad():
        pred, _ = net(inputs_tensor.to(device))
    pred = pred.cpu()
    labels = labels_tensor.cpu()

    # --- Per-neuron Gaussian LL (z-scored space) ---
    # pred and labels are (time, trials, output_size)
    n_neurons_output = pred.shape[-1]
    per_neuron_gaussian_ll = []
    for j in range(n_neurons_output):
        g_ll = gaussian_log_likelihood(pred[:, :, j:j+1], labels[:, :, j:j+1])
        per_neuron_gaussian_ll.append(g_ll)
    per_neuron_gaussian_ll = np.array(per_neuron_gaussian_ll)

    # --- Per-neuron Poisson LL (raw count space) ---
    per_neuron_poisson_ll = None
    has_poisson = (raw_labels_tensor is not None and scaler is not None)

    if has_poisson:
        # Inverse-transform predictions to count space
        pred_np = pred.numpy()
        orig_shape = pred_np.shape
        pred_long = pred_np.reshape(-1, orig_shape[-1])

        if pca_obj is not None:
            try:
                pred_long = pca_obj.inverse_transform(pred_long)
            except Exception as e:
                print(f"  [WARNING] Neuron eval: PCA inverse failed: {e}")
                has_poisson = False

        if has_poisson:
            try:
                pred_long = scaler.inverse_transform(pred_long)
                pred_long = np.clip(pred_long, a_min=1e-8, a_max=None)
            except Exception as e:
                print(f"  [WARNING] Neuron eval: scaler inverse failed: {e}")
                has_poisson = False

        if has_poisson:
            # pred_long is now (time*trials, n_neurons)
            raw_np = raw_labels_tensor.numpy()
            raw_long = raw_np.reshape(-1, raw_np.shape[-1])
            n_neurons_raw = raw_long.shape[-1]

            if pred_long.shape == raw_long.shape:
                per_neuron_poisson_ll = []
                for j in range(n_neurons_raw):
                    p_pred = torch.tensor(pred_long[:, j:j+1], dtype=torch.float32)
                    p_raw = torch.tensor(raw_long[:, j:j+1], dtype=torch.float32)
                    p_ll = poisson_log_likelihood(p_pred, p_raw)
                    per_neuron_poisson_ll.append(p_ll)
                per_neuron_poisson_ll = np.array(per_neuron_poisson_ll)
            else:
                print(f"  [WARNING] Neuron eval: shape mismatch after inverse: "
                      f"{pred_long.shape} vs {raw_long.shape}")

    # --- Mean firing rates per neuron (for scatter plot) ---
    mean_rates = None
    if binned_spikes is not None:
        # binned_spikes is (trials, neurons, time)
        mean_rates = binned_spikes.mean(axis=(0, 2))  # (neurons,)

    # --- Build results dict ---
    results = dict(
        per_neuron_gaussian_ll=per_neuron_gaussian_ll,
        per_neuron_poisson_ll=per_neuron_poisson_ll,
        mean_rates=mean_rates,
        n_neurons_output=n_neurons_output,
        dataset_name=dataset_name,
        taste_ind=taste_ind,
    )

    # --- Plot ---
    _plot_neuron_diagnostics(results, dataset_name, taste_ind, neuron_dir)

    # --- Save results ---
    save_path = os.path.join(neuron_dir,
                             f'neuron_eval_taste_{taste_ind}_{dataset_name}.npz')
    save_dict = dict(per_neuron_gaussian_ll=per_neuron_gaussian_ll)
    if per_neuron_poisson_ll is not None:
        save_dict['per_neuron_poisson_ll'] = per_neuron_poisson_ll
    if mean_rates is not None:
        save_dict['mean_rates'] = mean_rates
    np.savez(save_path, **save_dict)

    net.train()
    return results

def _plot_neuron_diagnostics(results, dataset_name, taste_ind, neuron_dir):
    g_ll       = results['per_neuron_gaussian_ll']
    p_ll       = results['per_neuron_poisson_ll']
    mean_rates = results['mean_rates']
    n_neurons  = len(g_ll)
    x          = np.arange(n_neurons)
    has_poisson = p_ll is not None

    g_mean = np.mean(g_ll)
    g_std  = np.std(g_ll)
    g_sem  = g_std / np.sqrt(n_neurons)
    worst_idx   = np.argmin(g_ll)
    best_idx    = np.argmax(g_ll)
    worst_neurons = np.where(g_ll < (g_mean - 2 * g_std))[0]

    if has_poisson:
        p_mean    = np.nanmean(p_ll)
        p_std     = np.nanstd(p_ll)
        p_sem     = p_std / np.sqrt(np.sum(~np.isnan(p_ll)))
        n_poisson = len(p_ll)
        p_worst   = np.nanargmin(p_ll)
        p_best    = np.nanargmax(p_ll)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # ================================================================
    # ROW 0: Gaussian
    # ================================================================

    # --- [0,0] Per-neuron Gaussian LL bar ---
    ax = axes[0, 0]
    colors = ['salmon' if v < (g_mean - 2 * g_std) else 'steelblue' for v in g_ll]
    ax.bar(x, g_ll, color=colors)
    ax.axhline(g_mean, color='k', linestyle='-', linewidth=1.5, label=f'Mean: {g_mean:.1f}')
    ax.axhline(g_mean + g_std, color='k', linestyle='--', alpha=0.5)
    ax.axhline(g_mean - g_std, color='k', linestyle='--', alpha=0.5, label=f'+/-1 SD: {g_std:.1f}')
    ax.axhspan(g_mean - g_sem, g_mean + g_sem, color='gold', alpha=0.3, label=f'SEM: {g_sem:.1f}')
    for wn in worst_neurons:
        ax.annotate(f'{wn}', xy=(wn, g_ll[wn]), fontsize=7, color='red', ha='center', va='top')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Gaussian Log-Likelihood')
    ax.set_title('Per-Neuron Gaussian LL (z-scored)')
    ax.legend(fontsize=8)

    # --- [0,1] Gaussian LL vs firing rate ---
    ax = axes[0, 1]
    if mean_rates is not None and len(mean_rates) == n_neurons:
        ax.scatter(mean_rates, g_ll, c='steelblue', edgecolors='k', s=40, alpha=0.7)
        ax.set_xlabel('Mean Firing Rate (counts/bin)')
        ax.set_ylabel('Gaussian LL')
        ax.set_title('Gaussian LL vs Firing Rate')
        if n_neurons > 2:
            corr = np.corrcoef(mean_rates, g_ll)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')
    elif mean_rates is not None:
        ax.text(0.5, 0.5,
                f'Output dim ({n_neurons}) != neuron count ({len(mean_rates)})\n'
                f'(PCA in use — cannot map components to neurons)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.set_title('Gaussian LL vs Firing Rate (N/A with PCA)')
    else:
        ax.text(0.5, 0.5, 'No firing rate data available',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Gaussian LL vs Firing Rate')

    # --- [0,2] Gaussian LL distribution ---
    ax = axes[0, 2]
    ax.hist(g_ll, bins=min(20, n_neurons), color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(g_mean,          color='red',    linestyle='--', label=f'Mean: {g_mean:.1f}')
    ax.axvline(np.median(g_ll), color='orange', linestyle='--', label=f'Median: {np.median(g_ll):.1f}')
    ax.set_xlabel('Gaussian Log-Likelihood')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Gaussian LL')
    ax.legend(fontsize=8)

    # ================================================================
    # ROW 1: Poisson (or greyed out)
    # ================================================================

    if has_poisson:
        # --- [1,0] Per-neuron Poisson LL bar ---
        ax = axes[1, 0]
        p_colors = ['salmon' if (not np.isnan(v) and v < (p_mean - 2 * p_std))
                    else 'seagreen' for v in p_ll]
        ax.bar(np.arange(n_poisson), p_ll, color=p_colors)
        ax.axhline(p_mean, color='k', linestyle='-', linewidth=1.5, label=f'Mean: {p_mean:.1f}')
        ax.axhline(p_mean + p_std, color='k', linestyle='--', alpha=0.5)
        ax.axhline(p_mean - p_std, color='k', linestyle='--', alpha=0.5, label=f'+/-1 SD: {p_std:.1f}')
        ax.axhspan(p_mean - p_sem, p_mean + p_sem, color='gold', alpha=0.3, label=f'SEM: {p_sem:.1f}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Poisson Log-Likelihood')
        ax.set_title('Per-Neuron Poisson LL (raw counts)')
        ax.legend(fontsize=8)

        # --- [1,1] Poisson LL vs firing rate ---
        ax = axes[1, 1]
        if mean_rates is not None and len(mean_rates) == n_poisson:
            ax.scatter(mean_rates, p_ll, c='seagreen', edgecolors='k', s=40, alpha=0.7)
            ax.set_xlabel('Mean Firing Rate (counts/bin)')
            ax.set_ylabel('Poisson LL')
            ax.set_title('Poisson LL vs Firing Rate')
            valid = ~np.isnan(p_ll)
            if np.sum(valid) > 2:
                corr = np.corrcoef(mean_rates[valid], p_ll[valid])[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')
        else:
            ax.text(0.5, 0.5, 'Firing rate data unavailable\nor dimension mismatch',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title('Poisson LL vs Firing Rate')

        # --- [1,2] Poisson LL distribution ---
        ax = axes[1, 2]
        valid_p = p_ll[~np.isnan(p_ll)]
        ax.hist(valid_p, bins=min(20, n_poisson), color='seagreen', edgecolor='k', alpha=0.7)
        ax.axvline(p_mean,          color='red',    linestyle='--', label=f'Mean: {p_mean:.1f}')
        ax.axvline(np.median(valid_p), color='orange', linestyle='--', label=f'Median: {np.median(valid_p):.1f}')
        ax.set_xlabel('Poisson Log-Likelihood')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Poisson LL')
        ax.legend(fontsize=8)

    else:
        for col in range(3):
            axes[1, col].axis('off')
            axes[1, col].text(0.5, 0.5, 'Poisson data not available',
                              transform=axes[1, col].transAxes,
                              ha='center', va='center', fontsize=11,
                              color='gray', style='italic')

    # ================================================================
    # ROW 2: Gaussian vs Poisson scatter | summary stats | spare
    # ================================================================

    # --- [2,0] Gaussian vs Poisson LL scatter ---
    ax = axes[2, 0]
    if has_poisson and n_poisson == n_neurons:
        valid = ~np.isnan(p_ll)
        ax.scatter(g_ll[valid], p_ll[valid], c='mediumpurple', edgecolors='k', s=50, alpha=0.7)
        # annotate worst Poisson neuron
        ax.annotate(f'nrn {p_worst}',
                    xy=(g_ll[p_worst], p_ll[p_worst]),
                    fontsize=8, color='red', xytext=(5, -10), textcoords='offset points')
        ax.set_xlabel('Gaussian LL (z-scored)')
        ax.set_ylabel('Poisson LL (raw counts)')
        ax.set_title('Gaussian vs Poisson LL per Neuron')
        if np.sum(valid) > 2:
            corr = np.corrcoef(g_ll[valid], p_ll[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')
    elif has_poisson:
        ax.text(0.5, 0.5,
                f'Gaussian dim ({n_neurons}) != Poisson dim ({n_poisson})\n'
                f'(PCA in use — cannot directly compare)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.set_title('Gaussian vs Poisson LL (dim mismatch)')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Poisson data not available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')

    # --- [2,1] Combined summary stats ---
    ax = axes[2, 1]
    ax.axis('off')
    stats_lines = [
        f"Neuron Reconstruction -- Taste {taste_ind}",
        f"{'=' * 40}",
        f"Neurons evaluated:  {n_neurons}",
        f"{'=' * 40}",
        f"GAUSSIAN (z-scored space)",
        f"  Mean LL:          {g_mean:.2f}",
        f"  SD LL:            {g_std:.2f}",
        f"  SEM LL:           {g_sem:.2f}",
        f"  Median LL:        {np.median(g_ll):.2f}",
        f"  Best  (nrn {best_idx:>3d}):  {g_ll[best_idx]:.2f}",
        f"  Worst (nrn {worst_idx:>3d}):  {g_ll[worst_idx]:.2f}",
    ]
    if len(worst_neurons) > 0:
        stats_lines.append(f"  Outliers (<2 SD):  {list(worst_neurons)}")
    if mean_rates is not None and len(mean_rates) == n_neurons:
        ranked  = np.argsort(g_ll)
        bottom5 = ranked[:min(5, n_neurons)]
        top5    = ranked[-min(5, n_neurons):][::-1]
        stats_lines += [
            f"  Bottom 5: {list(bottom5)}",
            f"    rates:  {[f'{mean_rates[i]:.2f}' for i in bottom5]}",
            f"  Top 5:    {list(top5)}",
            f"    rates:  {[f'{mean_rates[i]:.2f}' for i in top5]}",
        ]
    stats_lines.append(f"{'=' * 40}")
    if has_poisson:
        any_positive = np.any(p_ll > 0)
        stats_lines += [
            f"POISSON (raw count space)",
            f"  Neurons:          {n_poisson}",
            f"  Mean LL:          {p_mean:.2f}",
            f"  SD LL:            {p_std:.2f}",
            f"  SEM LL:           {p_sem:.2f}",
            f"  Best  (nrn {p_best:>3d}):  {p_ll[p_best]:.2f}",
            f"  Worst (nrn {p_worst:>3d}):  {p_ll[p_worst]:.2f}",
            f"{'=' * 40}",
        ]
        if any_positive:
            pos_neurons = np.where(p_ll > 0)[0]
            stats_lines += [
                f"  WARNING: {len(pos_neurons)} neurons positive LL",
                f"  Neurons: {list(pos_neurons[:10])}",
                f"  (possible inverse transform bug)",
            ]
        else:
            stats_lines.append("  All Poisson LL negative (good).")
    ax.text(0.05, 0.95, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, family='monospace', va='top')

    # --- [2,2] Poisson LL vs firing rate ---
    ax = axes[2, 2]
    if has_poisson and mean_rates is not None and len(mean_rates) == n_poisson:
        valid = ~np.isnan(p_ll)
        ax.scatter(mean_rates[valid], p_ll[valid], c='seagreen', edgecolors='k', s=40, alpha=0.7)
        ax.set_xlabel('Mean Firing Rate (counts/bin)')
        ax.set_ylabel('Poisson LL')
        ax.set_title('Poisson LL vs Firing Rate')
        if np.sum(valid) > 2:
            corr = np.corrcoef(mean_rates[valid], p_ll[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')
    elif has_poisson and mean_rates is not None:
        ax.text(0.5, 0.5,
                f'Poisson dim ({n_poisson}) != rate dim ({len(mean_rates)})\n'
                f'(PCA in use — cannot map)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.set_title('Poisson LL vs Firing Rate (N/A with PCA)')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Poisson data not available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
    # ================================================================
    # Save
    # ================================================================
    fig.suptitle(f'Per-Neuron Reconstruction -- {dataset_name}, Taste {taste_ind}',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(neuron_dir,
                     f'neuron_diagnostics_taste_{taste_ind}_{dataset_name}.png'),
        bbox_inches='tight', dpi=200
    )
    plt.close(fig)
    print(f"  [INFO] Neuron diagnostics saved for taste {taste_ind}")