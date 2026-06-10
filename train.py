"""
Class to handle model training and prediction
""" 

############################################################
# Imports
############################################################

import time
import numpy as np
import pylab as plt
from tqdm import tqdm, trange
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score
import sys
import os
file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(file_path)
sys.path.append(src_dir)
from model import autoencoderRNN

############################################################
# Define Model 
############################################################
# Define networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F
import math
from scipy.stats import poisson, zscore

class smooth_MSELoss(nn.Module):
    """
    MSE loss with temporal smoothness constraint
    """

    def __init__(self, alpha=0.05):
        super(smooth_MSELoss, self).__init__()
        self.loss1 = nn.MSELoss()
        self.alpha = alpha

    def mean_diffrence(self, x):
        """
        Calculate the mean difference between adjacent elements

        Args:
            x: (seq_len, batch, output_size)
        """
        return torch.mean(torch.abs(x[1:] - x[:-1])) * self.alpha

    def forward(self, input, target):
        """
        Args:
            input: (seq_len,batch, output_size)
            target: (seq_len,batch, output_size)
        """
        loss = self.loss1(input, target) + self.mean_diffrence(input)
        return loss

def MSELoss():
    return nn.MSELoss()

# a few utils for better model comparisons 
def poisson_log_likelihood(predicted_rates, actual_counts): 
    """
    Poisson log-likelihood: sum( y*log(r) - r - log(y!) )
    
    Args:
        predicted_rates: tensor, model output (must be > 0)
        actual_counts: tensor, observed spike counts
    
    Returns:
        float, total log-likelihood
    """
    r = torch.clamp(predicted_rates, min=1e-8)
    y = actual_counts
    ll = torch.sum(y * torch.log(r) - r - torch.lgamma(y + 1))
    return ll.item()

def gaussian_log_likelihood(predicted, actual):
    """
    Gaussian log-likelihood derived from MSE.

    LL = -n/2 * ln(2*pi*sigma^2) - 1/(2*sigma^2) * sum((y - r)^2)

    where sigma^2 = MSE (maximum likelihood estimate of variance).
    Appropriate when the model is trained with MSE loss on z-scored data.

    Args:
        predicted: tensor, model output
        actual: tensor, target values

    Returns:
        float, total log-likelihood
    """
    n = actual.numel()
    residuals = actual - predicted
    mse = torch.mean(residuals ** 2)
    # MLE variance estimate
    sigma2 = mse.item()
    if sigma2 < 1e-12:
        sigma2 = 1e-12
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return ll


def count_parameters(net):
    """Count trainable parameters in a network."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# now to compute AIC and BIC (Alkine & Baysiean information cirterion-- relevant to prediction error). 
# when comparing across models, lower is better. 
def compute_aic_bic(net, inputs, labels):
    """
    Compute AIC and BIC using Gaussian log-likelihood (consistent with MSE training).
    
    Args:
        net: trained model
        inputs: (seq_len, batch, input_size)
        labels: (seq_len, batch, output_size) — z-scored data
    
    Returns:
        dict with gaussian aic/bic/ll, n_params, n_observations
    """
    net.eval()
    with torch.no_grad():
        output, _ = net(inputs)
    
    ll = gaussian_log_likelihood(output, labels)
    k = count_parameters(net)
    n = labels.numel()
    
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    
    net.train()
    
    return dict(
        aic=aic,
        bic=bic,
        log_likelihood=ll,
        n_params=k,
        n_observations=n,
    )


def compute_poisson_aic_bic(net, inputs, raw_labels, scaler, pca_obj=None):
    """
    Compute AIC and BIC using Poisson log-likelihood on raw count space.

    Inverse-transforms model predictions back to count space before
    computing the Poisson LL against raw (non-z-scored, non-PCA'd) labels.

    Args:
        net: trained model
        inputs: (seq_len, batch, input_size) — model inputs (z-scored + context)
        raw_labels: (seq_len, batch, n_neurons) — raw binned spike counts
        scaler: fitted StandardScaler
        pca_obj: fitted PCA object or None

    Returns:
        dict with poisson aic/bic/ll, n_params, n_observations
    """
    net.eval()
    with torch.no_grad():
        output, _ = net(inputs)
    
    # Inverse transform predictions back to count space
    pred_np = output.cpu().numpy()
    orig_shape = pred_np.shape
    pred_long = pred_np.reshape(-1, orig_shape[-1])

    if pca_obj is not None:
        try:
            pred_long = pca_obj.inverse_transform(pred_long)
        except Exception as e:
            print(f"  [WARNING] Poisson LL: PCA inverse failed: {e}")
            net.train()
            return None

    try:
        pred_long = scaler.inverse_transform(pred_long)
    except Exception as e:
        print(f"  [WARNING] Poisson LL: scaler inverse failed: {e}")
        net.train()
        return None

    # Clamp to non-negative (firing rates can't be negative)
    pred_long = np.clip(pred_long, a_min=1e-8, a_max=None)
    pred_tensor = torch.tensor(pred_long, dtype=torch.float32)

    # raw_labels should already be (seq_len, batch, n_neurons) 
    raw_flat = raw_labels.reshape(-1, raw_labels.shape[-1])

    if pred_tensor.shape != raw_flat.shape:
        print(f"  [WARNING] Poisson LL shape mismatch: "
              f"pred {pred_tensor.shape} vs raw {raw_flat.shape}")
        net.train()
        return None

    ll = poisson_log_likelihood(pred_tensor, raw_flat)
    k = count_parameters(net)
    n = raw_labels.numel()

    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    net.train()

    return dict(
        poisson_aic=aic,
        poisson_bic=bic,
        poisson_log_likelihood=ll,
        n_params=k,
        n_observations=n,
    )


### NOTE: NEW
# so AIC still punishes extra model complexity (punishment of 2k), which may be an issue. 
# Aicr deals with this, and corrects for out of sample (out-of-x) stuff. 
# see: https://www.sciencedirect.com/science/article/pii/S0167715221000262

def compute_aicr_penalty(n_obs, n_params):
    """AICr penalty term (DelSole & Tippett 2021)."""
    N, M = n_obs, n_params
    if N <= M + 2 or N <= M + 1:
        return float('nan')
    return (N * (N + 1) / (N - M - 2)) * (1 + (M - 1) / (N - M - 1))

# Standard:  aic  = 2 * n_params - 2 * total_ll
# Corrected: aicr = compute_aicr_penalty(n_obs, n_params) - 2 * total_ll


def train_model(
        net, 
        inputs, 
        labels, 
        output_size,
        train_steps = 1000, 
        lr=0.01, 
        delta_loss = 0.01,
        device = None,
        criterion = MSELoss(), 
        test_inputs = None,
        test_labels = None,
        patience=10,  # New parameter for early stopping
        quiet = False
        ):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
        inputs: shape (seq_len, batch, input_size)
        labels: shape (seq_len * batch, output_size)

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    cross_val_bool = np.logical_and(
            test_inputs is not None, 
            test_labels is not None
            )

    loss_history = []
    cross_val_loss = {}
    best_loss = float('inf')
    patience_counter = 0
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        # labels = labels.reshape(-1, output_size)
        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # # Reshape to (SeqLen x Batch, OutputSize)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Only compute cross_val_loss every 100 steps
        # because it's expensive
        if cross_val_bool and (i % 100 == 99): 
            test_out, _ = net(test_inputs)
            test_out_flat = test_out.reshape(-1, output_size)
            test_labels_flat = test_labels.reshape(-1, output_size)
            # consider the potential that the below is a bug-- is test labels getting reshaped every single loop here? hmmm
            # test_labels = test_labels.reshape(-1, output_size)
            test_loss = criterion(test_out_flat, test_labels_flat)
            cross_val_loss[i] = test_loss.item()
            # Early stopping check
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at step {i+1}')
                break

            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
        else:
            cross_str = ''

        # Compute the running loss every 100 steps
        current_loss = loss.item()
        loss_history.append(current_loss)
        running_loss += current_loss 
        if i % 100 == 99:
            running_loss /= 100
            if not quiet:
                print('Step {}, Loss {:0.4f}, {}, Time {:0.1f}s'.format(
                    i+1, running_loss, cross_str, time.time() - start_time))
            running_loss = 0
    # now calculating AIC/BIC at the end of every training: 
    if cross_val_bool: 
        eval_inputs = test_inputs
        eval_labels = test_labels
        eval_set = 'test'
    else:
        eval_inputs = inputs
        eval_labels = labels
        eval_set = 'train'

    info_criteria = compute_aic_bic(net, eval_inputs, eval_labels)
    info_criteria['eval_set'] = eval_set
    # print some info at the end of training just bc? 
    print(f"\n--- Information Criteria ({eval_set} set) ---")
    print(f"  Params:         {info_criteria['n_params']}")
    print(f"  Observations:   {info_criteria['n_observations']}")
    print(f"  Gauss. Log-Likelihood: {info_criteria['log_likelihood']:.2f}")
    print(f"  AIC:            {info_criteria['aic']:.2f}")
    print(f"  BIC:            {info_criteria['bic']:.2f}")
    return net, loss_history, cross_val_loss, info_criteria


