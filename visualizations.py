"""
Visualization functions for RNN training and evaluation.

All functions take data + paths and handle their own figure creation,
saving, and closing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


# ----------------------------------------------------------------
# Helper: import visualize module (vz) - note that this is Abu's personal (?) libary, sooooo 
# ----------------------------------------------------------------
_vz = None

def _get_vz():
    global _vz
    if _vz is None:
        import visualize as vz
        _vz = vz
    return _vz


# ----------------------------------------------------------------
# Input sanity check
# ----------------------------------------------------------------
def plot_inputs(inputs_plus_context, dataset_name, taste_ind, plots_dir):
    """Heatmap of RNN inputs for sanity checking."""
    vz = _get_vz()
    vz.firing_overview(
        inputs_plus_context.T, figsize=(10, 10),
        cmap='viridis', zscore_bool=False
    )
    plt.suptitle(f"RNN Input_{dataset_name}")
    plt.savefig(os.path.join(plots_dir, f'inputs_taste_{taste_ind}_{dataset_name}.png'))
    plt.close()


# ----------------------------------------------------------------
# Loss curves
# ----------------------------------------------------------------
def plot_loss_curves(loss, cross_val_loss, dataset_name, taste_ind, plots_dir):
    """Train and test loss over training steps."""
    # Simple version
    fig, ax = plt.subplots()
    ax.plot(loss, label='Train')
    if cross_val_loss:
        keys = [int(k) if isinstance(k, str) else k for k in cross_val_loss.keys()]
        ax.plot(keys, list(cross_val_loss.values()), label='Test')
    ax.legend()
    ax.set_title(f"Loss Curve_{dataset_name}")
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    fig.savefig(os.path.join(plots_dir, f'loss_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)

    # Detailed version with external legend
    fig, ax = plt.subplots()
    ax.plot(loss, label='Train Loss')
    if cross_val_loss:
        keys = [int(k) if isinstance(k, str) else k for k in cross_val_loss.keys()]
        ax.plot(keys, list(cross_val_loss.values()), label='Test Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title(f'Losses_{dataset_name}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    fig.savefig(
        os.path.join(plots_dir, f'run_loss_taste_{taste_ind}_{dataset_name}.png'),
        bbox_inches='tight'
    )
    plt.close(fig)


# ----------------------------------------------------------------
# Firing rate overview plots
# ----------------------------------------------------------------
def plot_firing_overview(pred_firing, binned_spikes, dataset_name, taste_ind, plots_dir):
    """Predicted and true firing rate heatmaps."""
    vz = _get_vz()

    vz.firing_overview(pred_firing.swapaxes(0, 1))
    fig = plt.gcf()
    plt.suptitle(f'RNN Predicted Firing Rates_{dataset_name}')
    fig.savefig(os.path.join(plots_dir, f'firing_pred_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)

    vz.firing_overview(binned_spikes.swapaxes(0, 1))
    fig = plt.gcf()
    plt.suptitle(f'Binned Firing Rates_{dataset_name}')
    fig.savefig(os.path.join(plots_dir, f'firing_binned_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)


# ----------------------------------------------------------------
# Mean firing rates (pred vs true, raw and z-scored)
# ----------------------------------------------------------------
def plot_mean_firing(pred_firing, binned_spikes, dataset_name, taste_ind, plots_dir):
    """Side-by-side mean firing rate heatmaps."""
    pred_mean = pred_firing.mean(axis=0)
    binned_mean = binned_spikes.mean(axis=0)

    # Raw
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pred_mean, aspect='auto', interpolation='none')
    ax[1].imshow(binned_mean, aspect='auto', interpolation='none')
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    fig.savefig(os.path.join(plots_dir, f'mean_firing_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)

    # Z-scored
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(zscore(pred_mean, axis=-1), aspect='auto', interpolation='none')
    ax[1].imshow(zscore(binned_mean, axis=-1), aspect='auto', interpolation='none')
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    fig.savefig(os.path.join(plots_dir, f'mean_firing_zscored_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)


# ----------------------------------------------------------------
# Latent factors
# ----------------------------------------------------------------
def plot_latent_factors(latent_outs, dataset_name, taste_ind, plots_dir):
    """Heatmap of each latent dimension across trials."""
    n_latent = latent_outs.shape[-1]
    fig, ax = plt.subplots(n_latent, 1, figsize=(5, 10), sharex=True, sharey=True)
    if n_latent == 1:
        ax = [ax]
    for i in range(n_latent):
        ax[i].imshow(latent_outs[..., i].T, aspect='auto')
    plt.suptitle(f'Latent Factors_{dataset_name}')
    fig.savefig(os.path.join(plots_dir, f'latent_factors_taste_{taste_ind}_{dataset_name}.png'))
    plt.close(fig)


def plot_trial_latents(latent_outs, dataset_name, taste_ind, plots_dir):
    """Per-trial latent factor traces (raw and z-scored)."""
    trial_latent_dir = os.path.join(plots_dir, 'trial_latent')
    os.makedirs(trial_latent_dir, exist_ok=True)

    for i in range(latent_outs.shape[1]):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(latent_outs[1:, i], alpha=0.5)
        ax[0].set_title(f'Latent factors for trial {i} {dataset_name}')
        ax[1].plot(zscore(latent_outs[1:, i], axis=0), alpha=0.5)
        fig.savefig(os.path.join(trial_latent_dir, f'taste_{taste_ind}_trial_{i}_latent.png'))
        plt.close(fig)


# ----------------------------------------------------------------
# Individual neuron plots (raster + conv + RNN predicted)
# ----------------------------------------------------------------
def plot_individual_neurons(
        taste_spikes, binned_spikes, pred_firing, conv_rate, conv_x,
        bin_size, stim_time_val, dataset_name, taste_ind, plots_dir
    ):
    """Per-neuron raster, convolved rate, and RNN prediction."""
    vz = _get_vz()
    ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
    os.makedirs(ind_plot_dir, exist_ok=True)

    binned_x = np.arange(0, binned_spikes.shape[-1] * bin_size, bin_size)

    for i in range(binned_spikes.shape[1]):
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=False)
        ax[0] = vz.raster(ax[0], taste_spikes[:, i], marker='|')
        ax[1].plot(conv_x, conv_rate[:, i].T, c='k', alpha=0.1)
        ax[2].plot(binned_x[1:], pred_firing[:, i].T, c='k', alpha=0.1)
        for this_ax in ax:
            this_ax.axvline(stim_time_val, c='r', linestyle='--')
        ax[1].set_title(f'Convolved Firing Rate : Kernel Size 250')
        ax[2].set_title('RNN Predicted Firing Rate')
        fig.savefig(os.path.join(
            ind_plot_dir, f'neuron_{i}_taste_{taste_ind}_raster_conv_pred.png'
        ))
        plt.close(fig)


def plot_mean_neurons_across_tastes(
        spike_array, pred_firing_list, binned_spikes_list,
        conv_rate_list, conv_x_list, bin_size, stim_time_val,
        time_lims, dataset_name, plots_dir
    ):
    """
    Cross-taste mean neuron plots: raster, convolved, and predicted.
    Called ONCE after all tastes are processed.
    """
    vz = _get_vz()
    ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
    os.makedirs(ind_plot_dir, exist_ok=True)

    cmap = plt.get_cmap('tab10')
    n_neurons = binned_spikes_list[0].shape[1]
    binned_x = np.arange(0, binned_spikes_list[0].shape[-1] * bin_size, bin_size)

    # --- Per-neuron taste-mean summary ---
    pred_firing_taste_mean = np.stack(
        [pf.mean(axis=0) for pf in pred_firing_list]
    )
    binned_spikes_taste_mean = np.stack(
        [bs.mean(axis=0) for bs in binned_spikes_list]
    )

    fig, ax = vz.gen_square_subplots(n_neurons, figsize=(10, 10), sharex=True)
    for nrn_ind in range(n_neurons):
        for t_ind in range(len(pred_firing_list)):
            ax.flatten()[nrn_ind].plot(
                pred_firing_taste_mean[t_ind, nrn_ind], alpha=1, c=cmap(t_ind)
            )
            ax.flatten()[nrn_ind].plot(
                binned_spikes_taste_mean[t_ind, nrn_ind], alpha=0.3, c=cmap(t_ind)
            )
        ax.flatten()[nrn_ind].set_ylabel(str(nrn_ind))
    fig.savefig(os.path.join(plots_dir, 'mean_neuron_firing.png'))
    plt.close(fig)

    # --- Per-neuron cross-taste raster + conv + pred ---
    for i in range(n_neurons):
        this_spikes_list = [x[:, i] for x in spike_array]
        trial_counts = [len(x) for x in this_spikes_list]
        cum_trial_counts = np.cumsum([0, *trial_counts])
        this_cat_spikes = np.concatenate(this_spikes_list)[..., time_lims[0]:time_lims[1]]

        this_conv_rate = np.stack([x[:, i] for x in conv_rate_list])
        this_pred_firing = np.stack([x[:, i] for x in pred_firing_list])
        mean_conv = this_conv_rate.mean(axis=1)
        mean_pred = this_pred_firing.mean(axis=1)
        sd_conv = this_conv_rate.std(axis=1)
        sd_pred = this_pred_firing.std(axis=1)

        conv_x = conv_x_list[0]  # same for all tastes

        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=False)
        ax[0] = vz.raster(ax[0], this_cat_spikes, marker='|', color='k')
        for j in range(len(cum_trial_counts) - 1):
            ax[0].axhspan(
                cum_trial_counts[j], cum_trial_counts[j + 1],
                color=cmap(j), alpha=0.1, zorder=0
            )

        for j in range(mean_conv.shape[0]):
            ax[1].plot(conv_x, mean_conv[j].T, c=cmap(j), linewidth=2)
            ax[1].fill_between(
                conv_x, mean_conv[j] - sd_conv[j], mean_conv[j] + sd_conv[j],
                color=cmap(j), alpha=0.1
            )
            ax[2].plot(binned_x[1:], mean_pred[j].T, c=cmap(j), linewidth=2)
            ax[2].fill_between(
                binned_x[1:], mean_pred[j] - sd_pred[j], mean_pred[j] + sd_pred[j],
                color=cmap(j), alpha=0.1
            )

        for this_ax in ax:
            this_ax.axvline(stim_time_val, c='r', linestyle='--')
        ax[1].set_title('Convolved Firing Rate : Kernel Size 250')
        ax[2].set_title('RNN Predicted Firing Rate')
        fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_mean_raster_conv_pred.png'))
        plt.close(fig)


def plot_pred_vs_true_neurons(pred_firing_list, binned_spikes_list, plots_dir):
    """Per-neuron heatmap: all trials concatenated across tastes."""
    ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
    os.makedirs(ind_plot_dir, exist_ok=True)

    n_neurons = pred_firing_list[0].shape[1]

    for i in range(n_neurons):
        cat_pred = np.concatenate([x[:, i] for x in pred_firing_list])
        cat_true = np.concatenate([x[:, i] for x in binned_spikes_list])

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        img_kwargs = dict(aspect='auto', interpolation='none', cmap='viridis')
        im0 = ax[0].imshow(cat_pred, **img_kwargs)
        im1 = ax[1].imshow(cat_true[:, 1:], **img_kwargs)
        ax[0].set_title('Pred')
        ax[1].set_title('True')
        fig.colorbar(im0, ax=ax[0], orientation='horizontal', label='Firing Rate (Hz)')
        fig.colorbar(im1, ax=ax[1], orientation='horizontal', label='Firing Rate (Hz)')
        fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_firing.png'))
        plt.close(fig)


# ----------------------------------------------------------------
# AIC/BIC summary (same as before, now a function)
# ----------------------------------------------------------------
def plot_aic_bic_summary(info_criteria_all, dataset_name, plots_dir):
    """Bar plot of AIC/BIC/LL/params across tastes."""
    taste_inds = sorted(info_criteria_all.keys())
    if not taste_inds:
        print("[WARNING] No info_criteria to plot.")
        return

    aic_vals = [info_criteria_all[t]['aic'] for t in taste_inds]
    bic_vals = [info_criteria_all[t]['bic'] for t in taste_inds]
    n_params = [info_criteria_all[t]['n_params'] for t in taste_inds]
    ll_vals = [info_criteria_all[t]['log_likelihood'] for t in taste_inds]

    x = np.arange(len(taste_inds))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x - w / 2, aic_vals, w, label='AIC', color='steelblue')
    axes[0].bar(x + w / 2, bic_vals, w, label='BIC', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Taste {t}' for t in taste_inds])
    axes[0].set_ylabel('Information Criterion')
    axes[0].set_title('AIC / BIC by Taste')
    axes[0].legend()

    axes[1].bar(x, ll_vals, color='seagreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Taste {t}' for t in taste_inds])
    axes[1].set_ylabel('Log-Likelihood')
    axes[1].set_title('Poisson Log-Likelihood by Taste')

    axes[2].bar(x, n_params, color='slategray')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'Taste {t}' for t in taste_inds])
    axes[2].set_ylabel('Trainable Parameters')
    axes[2].set_title('Model Complexity')

    fig.suptitle(f'Model Summary — {dataset_name}', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, f'aic_bic_summary_{dataset_name}.png'),
        bbox_inches='tight', dpi=250
    )
    plt.close(fig)



# ----------------------------------------------------------------
# LOO diagnostics (dual: Gaussian + Poisson)
# ----------------------------------------------------------------
def plot_loo_diagnostics(info_criteria, dataset_name, taste_ind, plots_dir):
    """
    Comprehensive LOO diagnostic plots for a single taste.

    One big mega figure:
      Figure 1 (Gaussian, 2x3): consistent with MSE training objective
        1. Per-trial Gaussian LL (bar + mean/SEM)
        2. Per-trial final training loss (bar + mean/SEM)
        3. All fold loss curves overlaid (overfitting diagnostic)
        4. Gaussian LL vs train loss scatter (outlier detection)
        5. Convergence steps histogram
        6. Summary stats
        Popisson: on raw spike counts — only if available
        7. Per-trial Poisson LL (bar + mean/SEM)
        8. Poisson LL vs Gaussian LL scatter (consistency check)
        9. Poisson summary stats
    """
    per_gauss_ll = np.array(info_criteria['per_trial_ll'])
    per_loss = np.array(info_criteria['per_trial_train_loss'])
    fold_histories = info_criteria.get('per_fold_loss_history', [])
    fold_steps = info_criteria.get('per_fold_n_steps', [])
    per_poisson_ll = info_criteria.get('per_trial_poisson_ll', [])
    has_poisson = len(per_poisson_ll) > 0 and not all(np.isnan(v) for v in per_poisson_ll)

    n_trials = len(per_gauss_ll)
    x = np.arange(n_trials)

    # ==================================================================
    # Figure 1: Gaussian diagnostics
    # ==================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # --- Panel 1: Per-trial Gaussian LL ---
    ax = axes[0, 0]
    g_mean = np.mean(per_gauss_ll)
    g_std = np.std(per_gauss_ll)
    g_sem = g_std / np.sqrt(n_trials)
    colors = ['salmon' if v < (g_mean - 2 * g_std) else 'steelblue'
              for v in per_gauss_ll]
    ax.bar(x, per_gauss_ll, color=colors)
    ax.axhline(g_mean, color='k', linestyle='-', linewidth=1.5,
               label=f'Mean: {g_mean:.1f}')
    ax.axhline(g_mean + g_std, color='k', linestyle='--', alpha=0.5,
               label=f'+/-1 SD: {g_std:.1f}')
    ax.axhline(g_mean - g_std, color='k', linestyle='--', alpha=0.5)
    ax.axhspan(g_mean - g_sem, g_mean + g_sem,
               color='gold', alpha=0.3, label=f'SEM: {g_sem:.1f}')
    ax.set_xlabel('Held-out Trial')
    ax.set_ylabel('Gaussian Log-Likelihood')
    ax.set_title('Per-Trial Gaussian LL (z-scored space)')
    ax.legend(fontsize=8)

    # --- Panel 2: Per-trial final training loss ---
    ax = axes[0, 1]
    ax.bar(x, per_loss, color='mediumpurple')
    loss_mean = np.mean(per_loss)
    loss_sem = np.std(per_loss) / np.sqrt(n_trials)
    ax.axhline(loss_mean, color='k', linestyle='-', linewidth=1.5,
               label=f'Mean: {loss_mean:.4f}')
    ax.axhspan(loss_mean - loss_sem, loss_mean + loss_sem,
               color='gold', alpha=0.3, label=f'SEM: {loss_sem:.4f}')
    ax.set_xlabel('Fold (held-out trial)')
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Per-Fold Final Train Loss')
    ax.legend(fontsize=8)

    # --- Panel 3: All fold loss curves overlaid ---
    ax = axes[0, 2]
    if fold_histories:
        for j, hist in enumerate(fold_histories):
            ax.plot(hist, alpha=0.25, color='steelblue', linewidth=0.8)
        max_len = max(len(h) for h in fold_histories)
        padded = np.full((n_trials, max_len), np.nan)
        for j, h in enumerate(fold_histories):
            padded[j, :len(h)] = h
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)
        steps_arr = np.arange(max_len)
        ax.plot(steps_arr, mean_curve, color='darkblue', linewidth=2, label='Mean')
        ax.fill_between(steps_arr, mean_curve - std_curve, mean_curve + std_curve,
                        color='steelblue', alpha=0.2, label='+/-1 SD')
        ax.legend(fontsize=8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Fold Loss Curves (overfitting check)')

    # --- Panel 4: Gaussian LL vs train loss scatter ---
    ax = axes[1, 0]
    ax.scatter(per_loss, per_gauss_ll, c='steelblue', edgecolors='k', s=50, alpha=0.7)
    worst_idx = np.argmin(per_gauss_ll)
    ax.annotate(f'Trial {worst_idx}',
                xy=(per_loss[worst_idx], per_gauss_ll[worst_idx]),
                fontsize=8, color='red',
                xytext=(5, -10), textcoords='offset points')
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Gaussian Held-Out LL')
    ax.set_title('Train Loss vs Gaussian LL')
    if len(per_loss) > 2:
        corr = np.corrcoef(per_loss, per_gauss_ll)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=10, va='top')

    # --- Panel 5: Convergence steps histogram ---
    ax = axes[1, 1]
    if fold_steps:
        ax.hist(fold_steps, bins=min(15, n_trials), color='teal',
                edgecolor='k', alpha=0.7)
        ax.axvline(np.mean(fold_steps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(fold_steps):.0f}')
        ax.axvline(np.median(fold_steps), color='orange', linestyle='--',
                   label=f'Median: {np.median(fold_steps):.0f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Steps to Convergence')
    ax.set_ylabel('Count')
    ax.set_title('Fold Convergence Distribution')

    # --- Panel 6: Combined summary stats (Gaussian + Poisson) ---
    ax = axes[1, 2]
    ax.axis('off')
    stats_lines = [
        f"LOO Summary -- Taste {taste_ind}",
        f"{'=' * 40}",
        f"Folds:              {n_trials}",
        f"Parameters:         {info_criteria.get('n_params', '?')}",
        f"Hidden size:        {info_criteria.get('hidden_size', '?')}",
        f"{'=' * 40}",
        f"GAUSSIAN (z-scored space)",
        f"  Total LL:         {info_criteria['log_likelihood']:.2f}",
        f"  Mean LL/trial:    {g_mean:.2f}",
        f"  SD LL:            {g_std:.2f}",
        f"  SEM LL:           {g_sem:.2f}",
        f"  Min LL (trial {worst_idx}): {per_gauss_ll[worst_idx]:.2f}",
        f"  Max LL (trial {np.argmax(per_gauss_ll)}): {np.max(per_gauss_ll):.2f}",
        f"  AIC:              {info_criteria['aic']:.2f}",
        f"  AICr:              {info_criteria['aicr']:.2f}",
        f"  BIC:              {info_criteria['bic']:.2f}",
        f"{'=' * 40}",
    ]
    if fold_steps:
        stats_lines += [
            f"Mean steps:         {np.mean(fold_steps):.0f}",
            f"Median steps:       {np.median(fold_steps):.0f}",
            f"Total time:         {info_criteria.get('total_time_s', 0):.1f}s",
            f"Time/fold:          {info_criteria.get('total_time_s', 0) / n_trials:.1f}s",
            f"{'=' * 40}",
        ]
    if has_poisson:
        p_ll_summary = info_criteria.get('poisson_log_likelihood', float('nan'))
        stats_lines += [
            f"POISSON (raw count space)",
            f"  Total LL:         {p_ll_summary:.2f}",
            f"  Mean LL/trial:    {np.nanmean(per_poisson_ll):.2f}",
            f"  SD LL:            {np.nanstd(per_poisson_ll):.2f}",
            f"  SEM LL:           {np.nanstd(per_poisson_ll) / np.sqrt(np.sum(~np.isnan(per_poisson_ll))):.2f}",
            f"  Min LL (trial {np.nanargmin(per_poisson_ll)}): {np.nanmin(per_poisson_ll):.2f}",
            f"  Max LL (trial {np.nanargmax(per_poisson_ll)}): {np.nanmax(per_poisson_ll):.2f}",
            f"  AIC:              {info_criteria.get('poisson_aic', float('nan')):.2f}",
            f"  AICr:              {info_criteria.get('poisson_aicr', float('nan')):.2f}",
            f"  BIC:              {info_criteria.get('poisson_bic', float('nan')):.2f}",
            f"{'=' * 40}",
            f"Note: Poisson LL should be negative.",
            f"Positive values = inverse transform bug.",
        ]
    ax.text(0.05, 0.95, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, family='monospace', va='top')


    # ==================================================================
    # Panels 6-9: Poisson diagnostics (only if available)
    # ==================================================================
    if has_poisson:
        per_p_ll    = np.array(per_poisson_ll)
        valid_mask  = ~np.isnan(per_p_ll)
        per_p_valid = per_p_ll[valid_mask]
        p_mean = np.nanmean(per_p_ll)
        p_std  = np.nanstd(per_p_ll)
        p_sem  = p_std / np.sqrt(len(per_p_valid))

        # --- [2,0] Per-trial Poisson LL ---
        ax = axes[2, 0]
        colors = ['salmon' if (not np.isnan(v) and v < (p_mean - 2 * p_std)) else 'seagreen'
                  for v in per_p_ll]
        ax.bar(x, np.where(np.isnan(per_p_ll), 0, per_p_ll), color=colors)
        ax.axhline(p_mean, color='k', linestyle='-',  linewidth=1.5, label=f'Mean: {p_mean:.1f}')
        ax.axhline(p_mean + p_std, color='k', linestyle='--', alpha=0.5, label=f'+/-1 SD: {p_std:.1f}')
        ax.axhline(p_mean - p_std, color='k', linestyle='--', alpha=0.5)
        ax.axhspan(p_mean - p_sem, p_mean + p_sem, color='gold', alpha=0.3, label=f'SEM: {p_sem:.1f}')
        ax.set_xlabel('Held-out Trial')
        ax.set_ylabel('Poisson Log-Likelihood')
        ax.set_title('Per-Trial Poisson LL (raw count space)')
        ax.legend(fontsize=8)

        # --- [2,1] Poisson LL vs Gaussian LL scatter ---
        ax = axes[2, 1]
        ax.scatter(per_gauss_ll[valid_mask], per_p_valid,
                   c='seagreen', edgecolors='k', s=50, alpha=0.7)
        ax.set_xlabel('Gaussian LL (z-scored)')
        ax.set_ylabel('Poisson LL (raw counts)')
        ax.set_title('Gaussian vs Poisson LL')
        worst_p_idx = np.nanargmin(per_p_ll)
        ax.annotate(f'Trial {worst_p_idx}',
                    xy=(per_gauss_ll[worst_p_idx], per_p_ll[worst_p_idx]),
                    fontsize=8, color='red', xytext=(5, -10), textcoords='offset points')
        if len(per_p_valid) > 2:
            corr = np.corrcoef(per_gauss_ll[valid_mask], per_p_valid)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')

        # --- [2,2] Poisson LL vs train loss scatter (NEW) ---
        ax = axes[2, 2]
        ax.scatter(per_loss[valid_mask], per_p_valid,
                   c='seagreen', edgecolors='k', s=50, alpha=0.7)
        ax.annotate(f'Trial {worst_p_idx}',
                    xy=(per_loss[worst_p_idx], per_p_ll[worst_p_idx]),
                    fontsize=8, color='red', xytext=(5, -10), textcoords='offset points')
        ax.set_xlabel('Final Training Loss')
        ax.set_ylabel('Poisson Held-Out LL')
        ax.set_title('Train Loss vs Poisson LL')
        if len(per_p_valid) > 2:
            corr = np.corrcoef(per_loss[valid_mask], per_p_valid)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')

    else:
        for col in range(3):
            axes[2, col].axis('off')
            axes[2, col].text(0.5, 0.5, 'Poisson data not available',
                              transform=axes[2, col].transAxes,
                              ha='center', va='center', fontsize=11,
                              color='gray', style='italic')

    fig.suptitle(f'LOO Gaussian Diagnostics -- {dataset_name}, Taste {taste_ind}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir,
                     f'loo_diagnostics_taste_{taste_ind}_{dataset_name}.png'),
        bbox_inches='tight', dpi=250
    )
    plt.close(fig)

    print(f"  [INFO] LOO diagnostics saved for taste {taste_ind}")