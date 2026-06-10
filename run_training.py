"""
Model training and logic. 
"""
import os
import json
import time
import numpy as np
import torch
from model import autoencoderRNN
from train import (
    train_model, compute_aic_bic,
    poisson_log_likelihood, count_parameters, MSELoss, gaussian_log_likelihood, compute_aicr_penalty
)


# ----------------------------------------------------------------
# Mode 1: Standard train/test split
# ----------------------------------------------------------------



# NOTE: for the details on AICr, see where it's implemented. 

def train_or_load(
        input_size,
        hidden_size,
        output_size,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        device,
        criterion,
        train_steps,
        patience,
        lr=0.001,
        rnn_layers=2,
        dropout=0.2,
        retrain=True,
        model_save_path=None,
        artifacts_dir=None,
        taste_ind=None,
        ):
    """
    Train a new model or load an existing one.
    Computes AIC/BIC in both cases.

    Returns:
        net: trained/loaded model
        loss: list of per-step training losses
        cross_val_loss: dict of step -> test loss
        info_criteria: dict with AIC/BIC stats
    """
    net = autoencoderRNN(
        input_size, hidden_size, output_size,
        rnn_layers=rnn_layers, dropout=dropout
    )
    net.to(device)

    if retrain or not os.path.exists(model_save_path):
        net, loss, cross_val_loss, info_criteria = train_model(
            net, train_inputs, train_labels, output_size,
            train_steps=train_steps, lr=lr,
            criterion=criterion,
            test_inputs=test_inputs, test_labels=test_labels,
            patience=patience,
        )
        # Save model and loss histories
        if model_save_path:
            torch.save(net, model_save_path)
        if artifacts_dir and taste_ind is not None:
            with open(os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(loss, f)
            with open(os.path.join(artifacts_dir, f'cross_val_loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(cross_val_loss, f)
    else:
        net = torch.load(model_save_path)
        info_criteria = compute_aic_bic(
            net, test_inputs.to(device), test_labels.to(device)
        )
        info_criteria['eval_set'] = 'test'
        # Load saved loss histories
        loss_path = os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json')
        cv_path = os.path.join(artifacts_dir, f'cross_val_loss_taste_{taste_ind}.json')
        loss = json.load(open(loss_path)) if os.path.exists(loss_path) else []
        cross_val_loss = json.load(open(cv_path)) if os.path.exists(cv_path) else {}

    # Save info criteria
    if artifacts_dir and taste_ind is not None:
        with open(os.path.join(artifacts_dir, f'info_criteria_taste_{taste_ind}.json'), 'w') as f:
            json.dump(info_criteria, f, indent=2)

    return net, loss, cross_val_loss, info_criteria



# ----------------------------------------------------------------
# Mode 2: LOO evaluation + final retrain on all data
# ----------------------------------------------------------------
#### NOTE: this LOO shchema is currently leaving one TRIAL out, not one neuron out. 
# Leave one neuron out BEFORE the model gets trained is a module coming in the future; one thing at a time. 
# I'm doing something similar for model evaluation but where we withhold from some statistics and see how that impacts things...

# ultimately, I will want to make a LOO shcema for neurons prior to training. But that comes AFTER (maybe? talk to abu)

def loo_then_train(
        inputs_tensor,
        labels_tensor,
        input_size,
        hidden_size,
        output_size,
        device,
        criterion,
        train_steps,
        patience,
        lr=0.001,
        rnn_layers=2,
        dropout=0.2,
        retrain=True,
        model_save_path=None,
        artifacts_dir=None,
        taste_ind=None,
        verbose=True,
        loo_train_steps=None,
        loo_patience=None,
        scaler=None,
        pca_obj=None,
        raw_labels_tensor=None,
        ):
    """
    Phase 1: LOO cross-validation to get robust AIC/BIC.
             Computes both Gaussian LL (on z-scored data, consistent with MSE)
             and Poisson LL (on raw counts, scientifically meaningful).
             Uses loo_train_steps/loo_patience if provided (faster folds).
    Phase 2: Retrain a single model on ALL trials for downstream use.
             Uses full train_steps/patience.

    Returns:
        net: final model trained on all data
        loss: training loss history (from final retrain)
        cross_val_loss: {} (no held-out set for final model)
        info_criteria: dict with LOO-based AIC/BIC (Gaussian + Poisson) + per-trial LLs
    """
    n_trials = inputs_tensor.shape[1]

    # LOO fold settings — default to full settings if not specified
    fold_train_steps = loo_train_steps if loo_train_steps is not None else train_steps
    fold_patience = loo_patience if loo_patience is not None else patience

    # --- Check for cached LOO results ---
    loo_cache_path = None
    if artifacts_dir and taste_ind is not None:
        loo_cache_path = os.path.join(
            artifacts_dir, f'loo_results_taste_{taste_ind}.json'
        )

    run_loo = retrain or (loo_cache_path is None) or not os.path.exists(loo_cache_path)

    if run_loo:
        # ============================================================
        # Phase 1: LOO cross-validation
        # ============================================================
        print(f"  Running LOO ({n_trials} folds)...")

        # Parameter count (same for all folds)
        dummy_net = autoencoderRNN(
            input_size, hidden_size, output_size,
            rnn_layers=rnn_layers, dropout=dropout
        )
        n_params = count_parameters(dummy_net)
        del dummy_net

        per_trial_gaussian_ll = []
        per_trial_poisson_ll = []
        per_trial_train_loss = []
        per_fold_loss_history = []
        per_fold_n_steps = []
        has_raw_labels = raw_labels_tensor is not None and scaler is not None
        # debugs: 
        #print(f"  [DEBUG] raw_labels_tensor is None: {raw_labels_tensor is None}")
        #print(f"  [DEBUG] scaler is None: {scaler is None}")
        #print(f"  [DEBUG] has_raw_labels: {has_raw_labels}")
        total_start = time.time()

        for j in range(n_trials):
            fold_start = time.time()

            # Hold out trial j
            train_mask = [i for i in range(n_trials) if i != j]
            fold_train_inputs = inputs_tensor[:, train_mask].to(device)
            fold_train_labels = labels_tensor[:, train_mask].to(device)
            fold_test_inputs = inputs_tensor[:, j:j+1].to(device)
            fold_test_labels = labels_tensor[:, j:j+1].to(device)

            # Fresh model
            fold_net = autoencoderRNN(
                input_size, hidden_size, output_size,
                rnn_layers=rnn_layers, dropout=dropout
            )
            fold_net.to(device)

            # Train (quiet mode — LOO folds don't print per-step output)
            fold_net, fold_loss, _, _ = train_model(
                fold_net, fold_train_inputs, fold_train_labels, output_size,
                train_steps=fold_train_steps, lr=lr,
                criterion=criterion,
                test_inputs=fold_test_inputs, test_labels=fold_test_labels,
                patience=fold_patience,
                quiet=True,
            )

            # Evaluate held-out trial — Gaussian LL (on z-scored data)
            fold_net.eval()
            with torch.no_grad():
                pred, _ = fold_net(fold_test_inputs)

            g_ll = gaussian_log_likelihood(pred, fold_test_labels)
            per_trial_gaussian_ll.append(g_ll)

            # Evaluate held-out trial — Poisson LL (on raw count data)
            # temporarily adding some debugs. 
            if has_raw_labels:
                fold_raw_labels = raw_labels_tensor[:, j:j+1]
                pred_np = pred.cpu().numpy()
                pred_long = pred_np.reshape(-1, pred_np.shape[-1])
                
                if j == 0:  # debug first fold only
                    #print(f"    [DEBUG] pred_long shape after reshape: {pred_long.shape}")
                    #print(f"    [DEBUG] pca_obj: {pca_obj}")
                    if pca_obj is not None:
                        print(f"    [DEBUG] pca n_components: {pca_obj.n_components_}")
                
                if pca_obj is not None:
                    try:
                        pred_long = pca_obj.inverse_transform(pred_long)
                        #if j == 0:
                            #print(f"    [DEBUG] pred_long after inverse PCA: {pred_long.shape}")
                    except Exception as e:
                        print(f"    [DEBUG] PCA inverse FAILED: {e}")
                        pred_long = None
                
                if pred_long is not None:
                    try:
                        pred_long = scaler.inverse_transform(pred_long)
                        pred_long = np.clip(pred_long, a_min=1e-8, a_max=None)
                        pred_counts = torch.tensor(pred_long, dtype=torch.float32)
                        raw_flat = fold_raw_labels.reshape(-1, fold_raw_labels.shape[-1])
                        #if j == 0:
                            #print(f"    [DEBUG] pred_counts shape: {pred_counts.shape}")
                            #print(f"    [DEBUG] raw_flat shape: {raw_flat.shape}")
                        if pred_counts.shape == raw_flat.shape:
                            p_ll = poisson_log_likelihood(pred_counts, raw_flat)
                        else:
                            print(f"    [DEBUG] SHAPE MISMATCH: {pred_counts.shape} vs {raw_flat.shape}")
                            p_ll = float('nan')
                    except Exception as e:
                        print(f"    [DEBUG] scaler inverse FAILED: {e}")
                        p_ll = float('nan')
                else:
                    p_ll = float('nan')
                per_trial_poisson_ll.append(p_ll)

            per_trial_train_loss.append(fold_loss[-1])
            per_fold_loss_history.append(fold_loss)
            per_fold_n_steps.append(len(fold_loss))

            if verbose:
                elapsed = time.time() - fold_start
                p_str = f" | Poisson LL: {per_trial_poisson_ll[-1]:.2f}" if has_raw_labels else ""
                print(f"    Fold {j+1}/{n_trials} | "
                      f"Gauss LL: {g_ll:.2f}{p_str} | "
                      f"Final loss: {fold_loss[-1]:.4f} | "
                      f"{elapsed:.1f}s")

            del fold_net

        # NEW: getting the regression of loss func vs ll -- both gaussian and poisson 
        loss_gaussian_corr = float('nan')
        loss_poisson_corr = float('nan')
    
        if len(per_trial_train_loss) > 2 and len(per_trial_gaussian_ll) > 2:
            try:
                loss_gaussian_corr = float(
                    np.corrcoef(per_trial_train_loss, per_trial_gaussian_ll)[0, 1]
                )
            except Exception:
                pass
    
        has_valid_poisson = per_trial_poisson_ll and len(per_trial_poisson_ll) > 2
        if has_valid_poisson:
            valid = [i for i, p in enumerate(per_trial_poisson_ll) if not np.isnan(p)]
            if len(valid) > 2:
                losses = [per_trial_train_loss[i] for i in valid]
                lls = [per_trial_poisson_ll[i] for i in valid]
                try:
                    loss_poisson_corr = float(np.corrcoef(losses, lls)[0, 1])
                except Exception:
                    pass

        # Aggregate LOO results — Gaussian (primary, consistent with MSE training)
        gaussian_total_ll = sum(per_trial_gaussian_ll)
        n_obs_zscore = labels_tensor.numel()
        gaussian_aic = 2 * n_params - 2 * gaussian_total_ll
        gaussian_aicr = compute_aicr_penalty(n_obs_zscore, n_params) - 2 * gaussian_total_ll
        gaussian_bic = n_params * np.log(n_obs_zscore) - 2 * gaussian_total_ll

        # Aggregate LOO results — Poisson (on raw counts)
        if has_raw_labels and per_trial_poisson_ll:
            valid_poisson = [v for v in per_trial_poisson_ll if not np.isnan(v)]
            if valid_poisson:
                poisson_total_ll = sum(valid_poisson)
                n_obs_raw = raw_labels_tensor.numel()
                poisson_aic = 2 * n_params - 2 * poisson_total_ll
                poisson_aicr = compute_aicr_penalty(n_obs_zscore, n_params) - 2 * poisson_total_ll
                poisson_bic = n_params * np.log(n_obs_raw) - 2 * poisson_total_ll
            else:
                poisson_total_ll = float('nan')
                poisson_aic = float('nan')
                poisson_aicr = float('nan')
                poisson_bic = float('nan')
        else:
            per_trial_poisson_ll = []
            poisson_total_ll = float('nan')
            poisson_aic = float('nan')
            poisson_aicr = float('nan')
            poisson_bic = float('nan')

        total_elapsed = time.time() - total_start

        info_criteria = dict(
            # Gaussian (primary — consistent with MSE)
            aic=gaussian_aic,
            aicr=gaussian_aicr,
            bic=gaussian_bic,
            log_likelihood=gaussian_total_ll,
            per_trial_ll=per_trial_gaussian_ll,
            loss_gaussian_corr=loss_gaussian_corr,
            # Poisson (on raw counts)
            poisson_aic=poisson_aic,
            poisson_aicr=poisson_aicr,
            poisson_bic=poisson_bic,
            poisson_log_likelihood=poisson_total_ll,
            per_trial_poisson_ll=per_trial_poisson_ll,
            loss_poisson_corr=loss_poisson_corr,
            # Shared
            per_trial_train_loss=per_trial_train_loss,
            per_fold_loss_history=per_fold_loss_history,
            per_fold_n_steps=per_fold_n_steps,
            n_params=n_params,
            n_observations=n_obs_zscore,
            n_trials=n_trials,
            hidden_size=hidden_size,
            eval_set='loo',
            total_time_s=total_elapsed,
        )

        print(f"\n  --- LOO Summary (taste {taste_ind}, hidden={hidden_size}) ---")
        print(f"    Folds:         {n_trials}")
        print(f"    Fold settings: train_steps={fold_train_steps}, patience={fold_patience}")
        print(f"    Params:        {n_params}")
        print(f"    --- Gaussian (z-scored space) ---")
        print(f"    Total LL:      {gaussian_total_ll:.2f}")
        print(f"    Mean LL/trial: {np.mean(per_trial_gaussian_ll):.2f} "
              f"+/- {np.std(per_trial_gaussian_ll):.2f}")
        print(f"    gAIC:           {gaussian_aic:.2f}")
        print(f"    gAICr:           {gaussian_aicr:.2f}")
        print(f"    gBIC:           {gaussian_bic:.2f}")
        if not np.isnan(poisson_total_ll):
            print(f"    --- Poisson (raw count space) ---")
            print(f"    Total LL:      {poisson_total_ll:.2f}")
            print(f"    Mean LL/trial: {np.nanmean(per_trial_poisson_ll):.2f} "
                  f"+/- {np.nanstd(per_trial_poisson_ll):.2f}")
            print(f"    pAIC:           {poisson_aic:.2f}")
            print(f"    pAICr:           {poisson_aicr:.2f}")
            print(f"    BIC:           {poisson_bic:.2f}")
        else: 
            print("WARN: Poisson LL (raw count space) is nan for some reason.")
        print(f"    Time:          {total_elapsed:.1f}s")

        # Cache LOO results
        if loo_cache_path:
            save_dict = {k: v for k, v in info_criteria.items()
                         if not isinstance(v, list)}
            save_dict['per_trial_ll'] = per_trial_gaussian_ll
            save_dict['per_trial_poisson_ll'] = per_trial_poisson_ll
            save_dict['per_trial_train_loss'] = per_trial_train_loss
            with open(loo_cache_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

    else:
        print(f"  Loading cached LOO results from {loo_cache_path}")
        with open(loo_cache_path, 'r') as f:
            info_criteria = json.load(f)

    # ============================================================
    # Phase 2: Retrain on ALL trials for downstream outputs
    # ============================================================
    should_retrain_final = retrain or (
        model_save_path is not None and not os.path.exists(model_save_path)
    )

    if should_retrain_final:
        print(f"  Retraining final model on all {n_trials} trials...")
        net = autoencoderRNN(
            input_size, hidden_size, output_size,
            rnn_layers=rnn_layers, dropout=dropout
        )
        net.to(device)

        net, loss, _, _ = train_model(
            net,
            inputs_tensor.to(device),
            labels_tensor.to(device),
            output_size,
            train_steps=train_steps, lr=lr,
            criterion=criterion,
            patience=patience,
        )

        if model_save_path:
            torch.save(net, model_save_path)
        if artifacts_dir and taste_ind is not None:
            with open(os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(loss, f)
    else:
        print(f"  Loading final model from {model_save_path}")
        net = torch.load(model_save_path)
        loss_path = os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json')
        loss = json.load(open(loss_path)) if os.path.exists(loss_path) else []

    cross_val_loss = {}
    return net, loss, cross_val_loss, info_criteria

# ----------------------------------------------------------------
# Shared: forward pass for predictions
# ----------------------------------------------------------------

def run_prediction(net, inputs_tensor, device):
    """
    Forward pass through trained model.

    Args:
        net: trained model
        inputs_tensor: (time, trials, input_size)
        device: torch device

    Returns:
        outs: numpy array (time, trials, output_size)
        latent_outs: numpy array (time, trials, hidden_size)
    """
    net.eval()
    with torch.no_grad():
        outs, latent_outs = net(inputs_tensor.to(device))
    outs = outs.cpu().numpy()
    latent_outs = latent_outs.cpu().numpy()
    return outs, latent_outs


# ----------------------------------------------------------------
# K-fold cross-validation (faster alternative to LOO for sweeps)

# NOTE: Edit the number of folds to modify how much is held out. 
# A note, Optuna is designed to be reasonably resistant to noise, 
# but not immune. The higher the K, theoretically, the higher the 
# noise of the info we feed it. There is a fundamental trade-off here. 
# ----------------------------------------------------------------

def kfold_evaluate(
        inputs_tensor,
        labels_tensor,
        input_size,
        hidden_size,
        output_size,
        device,
        criterion,
        train_steps,
        patience,
        lr=0.001,
        rnn_layers=2,
        dropout=0.2,
        scaler=None,
        pca_obj=None,
        raw_labels_tensor=None,
        n_folds=5,
        seed=None,
        verbose=False,
        taste_ind=None,
        ):
    """
    K-fold cross-validation for fast hyperparameter evaluation.

    Unlike loo_then_train, this does NOT retrain a final model — it only
    returns evaluation metrics. Designed to be called from Optuna sweeps
    where you need a reliable ranking signal without the cost of 30-fold
    LOO + full retrain. Attempting to hit a balance between holding stuff out 
    and computational cost. 

    Trials are randomly assigned to K folds. Each fold is held out once
    while the model trains on the remaining K-1 folds. Gaussian and
    Poisson log-likelihoods are computed on each held-out fold and
    aggregated into AIC/BIC.

    Args:
        inputs_tensor: (time, trials, input_size)
        labels_tensor: (time, trials, output_size)
        input_size, hidden_size, output_size: int
        device: torch device
        criterion: loss function
        train_steps, patience: training settings for each fold
        lr, rnn_layers, dropout: model/optimizer settings
        scaler: fitted StandardScaler (for Poisson LL)
        pca_obj: fitted PCA or None (for Poisson LL)
        raw_labels_tensor: (time, trials, n_neurons) raw counts (for Poisson LL)
        n_folds: int, number of folds (default 5)
        seed: int or None, random seed for fold assignment
        verbose: bool
        taste_ind: int, for logging

    Returns:
        info_criteria: dict with K-fold AIC/BIC (Gaussian + Poisson),
                       per-fold LLs, fold loss histories, timing
    """
    n_trials = inputs_tensor.shape[1]

    # Assign trials to folds randomly
    rng = np.random.RandomState(seed)
    fold_ids = np.zeros(n_trials, dtype=int)
    perm = rng.permutation(n_trials)
    for i, idx in enumerate(perm):
        fold_ids[idx] = i % n_folds

    # Parameter count
    dummy_net = autoencoderRNN(
        input_size, hidden_size, output_size,
        rnn_layers=rnn_layers, dropout=dropout
    )
    n_params = count_parameters(dummy_net)
    del dummy_net

    has_raw_labels = raw_labels_tensor is not None and scaler is not None

    per_fold_gaussian_ll = []
    per_fold_poisson_ll = []
    per_fold_train_loss = []
    per_fold_loss_history = []
    per_fold_n_steps = []
    total_start = time.time()

    for k in range(n_folds):
        fold_start = time.time()

        test_mask = np.where(fold_ids == k)[0]
        train_mask = np.where(fold_ids != k)[0]

        fold_train_inputs = inputs_tensor[:, train_mask].to(device)
        fold_train_labels = labels_tensor[:, train_mask].to(device)
        fold_test_inputs = inputs_tensor[:, test_mask].to(device)
        fold_test_labels = labels_tensor[:, test_mask].to(device)

        # Fresh model
        fold_net = autoencoderRNN(
            input_size, hidden_size, output_size,
            rnn_layers=rnn_layers, dropout=dropout
        )
        fold_net.to(device)

        # Train
        fold_net, fold_loss, _, _ = train_model(
            fold_net, fold_train_inputs, fold_train_labels, output_size,
            train_steps=train_steps, lr=lr,
            criterion=criterion,
            test_inputs=fold_test_inputs, test_labels=fold_test_labels,
            patience=patience,
            quiet=True,
        )

        # Evaluate — Gaussian LL
        fold_net.eval()
        with torch.no_grad():
            pred, _ = fold_net(fold_test_inputs)

        g_ll = gaussian_log_likelihood(pred, fold_test_labels)
        per_fold_gaussian_ll.append(g_ll)

        # Evaluate — Poisson LL
        if has_raw_labels:
            fold_raw_labels = raw_labels_tensor[:, test_mask]
            pred_np = pred.cpu().numpy()
            pred_long = pred_np.reshape(-1, pred_np.shape[-1])

            p_ll = float('nan')
            if pca_obj is not None:
                try:
                    pred_long = pca_obj.inverse_transform(pred_long)
                except Exception:
                    pred_long = None

            if pred_long is not None:
                try:
                    pred_long = scaler.inverse_transform(pred_long)
                    pred_long = np.clip(pred_long, a_min=1e-8, a_max=None)
                    pred_counts = torch.tensor(pred_long, dtype=torch.float32)
                    raw_flat = fold_raw_labels.reshape(-1, fold_raw_labels.shape[-1])
                    if pred_counts.shape == raw_flat.shape:
                        p_ll = poisson_log_likelihood(pred_counts, raw_flat)
                except Exception:
                    pass

            per_fold_poisson_ll.append(p_ll)

        per_fold_train_loss.append(fold_loss[-1])
        per_fold_loss_history.append(fold_loss)
        per_fold_n_steps.append(len(fold_loss))

        if verbose:
            elapsed = time.time() - fold_start
            p_str = f" | Poisson LL: {per_fold_poisson_ll[-1]:.2f}" if has_raw_labels else ""
            n_test = len(test_mask)
            print(f"    Fold {k+1}/{n_folds} ({n_test} held-out trials) | "
                  f"Gauss LL: {g_ll:.2f}{p_str} | "
                  f"Final loss: {fold_loss[-1]:.4f} | "
                  f"{elapsed:.1f}s")

        del fold_net
    # Loss vs LL conbsistency (higher r = more reliable fits across DS, better generalizations) -- both poisson and gauss bc whyt not 
    loss_gaussian_corr = float('nan')
    loss_poisson_corr = float('nan')
    if len(per_fold_train_loss) > 2 and len(per_fold_gaussian_ll) > 2:
        try:
            loss_gaussian_corr = float(
                np.corrcoef(per_fold_train_loss, per_fold_gaussian_ll)[0, 1]
            )
        except Exception:
            pass
 
    if has_raw_labels and len(per_fold_train_loss) > 2:
        valid = [i for i, p in enumerate(per_fold_poisson_ll) if not np.isnan(p)]
        if len(valid) > 2:
            losses = [per_fold_train_loss[i] for i in valid]
            lls = [per_fold_poisson_ll[i] for i in valid]
            try:
                loss_poisson_corr = float(np.corrcoef(losses, lls)[0, 1])
            except Exception:
                pass
 
    # --- Aggregate: Gaussian ---
    gaussian_total_ll = sum(per_fold_gaussian_ll)
    n_obs_zscore = labels_tensor.numel()
    gaussian_aic = 2 * n_params - 2 * gaussian_total_ll
    gaussian_aicr = compute_aicr_penalty(n_obs_zscore, n_params) - 2 * gaussian_total_ll 
    gaussian_bic = n_params * np.log(n_obs_zscore) - 2 * gaussian_total_ll

    # --- Aggregate: Poisson ---
    if has_raw_labels and per_fold_poisson_ll:
        valid_poisson = [v for v in per_fold_poisson_ll if not np.isnan(v)]
        if valid_poisson:
            poisson_total_ll = sum(valid_poisson)
            n_obs_raw = raw_labels_tensor.numel()
            poisson_aic = 2 * n_params - 2 * poisson_total_ll
            poisson_aicr = compute_aicr_penalty(n_obs_zscore, n_params) - 2 * poisson_total_ll
            poisson_bic = n_params * np.log(n_obs_raw) - 2 * poisson_total_ll
        else:
            poisson_total_ll = float('nan')
            poisson_aic = float('nan')
            poisson_aicr = float('nan')
            poisson_bic = float('nan')
    else:
        per_fold_poisson_ll = []
        poisson_total_ll = float('nan')
        poisson_aic = float('nan')
        poisson_aicr = float('nan')
        poisson_bic = float('nan')

    total_elapsed = time.time() - total_start

    info_criteria = dict(
        # Gaussian
        aic=gaussian_aic,
        aicr = gaussian_aicr,
        bic=gaussian_bic,
        log_likelihood=gaussian_total_ll,
        per_fold_gaussian_ll=per_fold_gaussian_ll,
        loss_gaussian_corr=loss_gaussian_corr,
        # Poisson
        poisson_aic=poisson_aic,
        poisson_aicr = poisson_aicr,
        poisson_bic=poisson_bic,
        poisson_log_likelihood=poisson_total_ll,
        per_fold_poisson_ll=per_fold_poisson_ll,
        loss_poisson_corr=loss_poisson_corr,
        # Shared
        per_fold_train_loss=per_fold_train_loss,
        per_fold_loss_history=per_fold_loss_history,
        per_fold_n_steps=per_fold_n_steps,
        n_params=n_params,
        n_observations=n_obs_zscore,
        n_folds=n_folds,
        n_trials=n_trials,
        hidden_size=hidden_size,
        eval_set='kfold',
        total_time_s=total_elapsed,
    )

    if verbose:
        print(f"\n  --- K-Fold Summary (taste {taste_ind}, K={n_folds}, "
              f"hidden={hidden_size}) ---")
        print(f"    Params:        {n_params}")
        print(f"    --- Gaussian ---")
        print(f"    Total LL:      {gaussian_total_ll:.2f}")
        print(f"    gAIC:           {gaussian_aic:.2f}")
        print(f"    gAICr:           {gaussian_aicr:.2f}")
        if not np.isnan(poisson_total_ll):
            print(f"    --- Poisson ---")
            print(f"    Total LL:      {poisson_total_ll:.2f}")
            print(f"    AIC:           {poisson_aic:.2f}")
            print(f"    AICr:           {poisson_aicr:.2f}")
        print(f"    Time:          {total_elapsed:.1f}s")

    return info_criteria