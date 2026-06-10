# updated as of 7/11/2025 
# merging some of abu's blech rnn updates with teh stuff that I had previously built up.

"""
Run autoencoder RNN to infer firing rates from spike trains.
Supports:
- Per-taste training
- JSON config
- Modular loss function
- Early stopping
- Full artifact saving (plots, latent outputs, firing rates)
"""

import os
import sys
import json
import glob
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import tables

import tempfile
import shutil

# ------------------------
# Load Config + Setup Paths
# ------------------------
config_path = '/home/vincent/Senior thesis work/blechRNN-master/config/blechrnn_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

sys.path.append(config['paths']['underlying_functions'])
sys.path.append(config['paths']['ephys_data'])
sys.path.append(config['paths']['src'])

from ephys_data import ephys_data
from model import autoencoderRNN
from train import train_model, MSELoss, smooth_MSELoss
import visualize as vz
# helper funcs (taken from ephys_data so I can modify them)

# ------------------------
# Parameters
# ------------------------
train_steps = config['parameters']['train_steps']
hidden_size = config['parameters']['hidden_size']
bin_size = config['parameters']['bin_size']
train_test_split = config['parameters']['train_test_split']
use_pca = config['parameters']['use_pca']
retrain = config['parameters']['retrain']
time_lims = config['parameters']['time_lims']
loss_name = config['parameters'].get('loss_name', 'mse')
patience = config['parameters'].get('patience', 12000)

h5_dir = config['paths']['h5_dir']
output_base_dir = config['paths']['output_base_dir']
stim_time_val = 2000 - time_lims[0]

pred_fr_dir = os.path.join(output_base_dir, 'pred_fr')
pred_lat_dir = os.path.join(output_base_dir, 'pred_latent')
os.makedirs(pred_fr_dir, exist_ok=True)
os.makedirs(pred_lat_dir, exist_ok=True)

# Loss Function Selector
loss_dict = {
    'mse': MSELoss(),
    'smooth': smooth_MSELoss(alpha=0.05),
}
criterion = loss_dict.get(loss_name, MSELoss())

# something that works with non-nested input h5's, but which takes forever and you have to babysit: 
#for file in os.listdir(h5_dir):
#    if not file.endswith(".h5"):
#        continue
#
#    dataset_name = os.path.splitext(file)[0]
#    print(f"Processing: {dataset_name}")
#    # Save outputs
#    output_path = os.path.join(output_base_dir, dataset_name)
#    plots_dir = os.path.join(output_path, 'plots')
#    artifacts_dir = os.path.join(output_path, 'artifacts')
#    os.makedirs(plots_dir, exist_ok=True)
#    os.makedirs(artifacts_dir, exist_ok=True)
#
#    data = ephys_data(h5_dir)  
#    data.get_spikes()
#    spike_array = np.stack(data.spikes)
#
#    pred_firing_list = []
#    latent_out_list = []
#    binned_spikes_list = []
#    pred_x_list = []
#    conv_rate_list = []
#    conv_x_list = []

# ------------------------
# Loop Over H5 Datasets
# ------------------------
# if I want to do all at once: in order to subert ephys_data and avoid changing any part of it, 
# I have elected to move all files to individual locations so that I can process all at once. 
# what an epic pain in my ass 
# if you want to do one at a time, point to one file OR uncomment the above and use the cli options. 

for subdir in sorted(os.listdir(h5_dir)):
    full_subdir_path = os.path.join(h5_dir, subdir)
    if not os.path.isdir(full_subdir_path):
        continue

    h5_files = [f for f in os.listdir(full_subdir_path) if f.endswith(".h5")]
    if len(h5_files) != 1:
        print(f"Skipping {subdir} — expected exactly one .h5 file, found {len(h5_files)}")
        continue

    h5_file = h5_files[0]
    dataset_name = os.path.splitext(h5_file)[0]
    print(f"Processing: {dataset_name}")

    output_path = os.path.join(output_base_dir, dataset_name)
    plots_dir = os.path.join(output_path, 'plots')
    artifacts_dir = os.path.join(output_path, 'artifacts')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # This now works without prompt!
    data = ephys_data(full_subdir_path)
    data.get_spikes()
    spike_array = np.stack(data.spikes)

    pred_firing_list = []
    latent_out_list = []
    binned_spikes_list = []
    pred_x_list = []
    conv_rate_list = []
    conv_x_list = []

    for taste_ind, taste_spikes in enumerate(spike_array):
        print(f"  Taste {taste_ind}")
        model_name = f'taste_{taste_ind}_hidden_{hidden_size}_loss_{loss_name}'
        model_save_path = os.path.join(artifacts_dir, f'{model_name}.pt')

        taste_spikes = taste_spikes[..., time_lims[0]:time_lims[1]] # eq. to concatenate?

        #trial_num = np.arange(taste_spikes.shape[1]) # double check this here...

        binned_spikes = np.reshape(taste_spikes, (*taste_spikes.shape[:2], -1, bin_size)).sum(-1)
        binned_spikes_list.append(binned_spikes)

        inputs = np.moveaxis(binned_spikes.copy(), -1, 0)  # (time, trial, neuron)
        # double check the indexing below
        trial_num = np.arange(inputs.shape[1]) # original indexing, which I'm not convinced of?
        # see definition of trial_num above now 

        # Save original neuron count -- trying to keep up with using both pca and not
        num_neurons = inputs.shape[2]
        inputs_long = inputs.reshape(-1, inputs.shape[-1])
        scaler = StandardScaler()
        inputs_long = scaler.fit_transform(inputs_long)

        if use_pca:
            print('Performing PCA on inputs')
            pca_obj = PCA(n_components=0.95)
            inputs_pca = pca_obj.fit_transform(inputs_long)
            n_components = inputs_pca.shape[-1]

            # z-score the PCA outputs too -- sacale accordingly

            # if the inputs_long are already scaled, I do not actualy need to scale again (I think...)
            # TODO ask abu about this
            # inputs_pca = StandardScaler().fit_transform(inputs_pca)

            inputs_trial_pca = inputs_pca.reshape(inputs.shape[0], -1, n_components)
            inputs = inputs_trial_pca.copy()
            # input_size = n_components  + 2  # PCA + stim + trial (double check why needed?)
        else:
            inputs = inputs_long.reshape(inputs.shape)
            # input_size = num_neurons   + 2  # Neurons + stim + trial (double check why needed)

        stim_input = np.zeros((inputs.shape[0], inputs.shape[1]))
       # stim_input[stim_time_val // bin_size, :] = 1 # what I had-- changed to match abu; double check idx'ing
        stim_input[:, stim_time_val // bin_size] = 1 
        # adding trial context (rationale? I actually forget why I did this; could be holdover)
        trial_context = np.broadcast_to(trial_num / trial_num.max(), inputs.shape[:2])

        inputs_plus_context = np.concatenate([
            inputs,
            stim_input[:, :, None],
            trial_context[:, :, None]
        ], axis=-1)

        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # output_size = input_size - 2 # original construct
        # Rectifying a shape mismatch 
        context_dim = 2  # stim + trial -- a note: only used here...

        input_feature_dim = inputs.shape[-1]  # either PCA components or raw neurons

        # input_size = input_feature_dim + context_dim
        # try
        input_size = inputs_plus_context.shape[-1] # original construct
        output_size = inputs_plus_context.shape[-1] - 2 # do not predict the stim time or trial numb

        # debug statements
        print("Input shape to model:", inputs_plus_context.shape)
        print("Model input_size:", input_size)

        # output_size = num_neurons  # hmm I want neuron space but not sure why this is here...

        # Plot input sanity check
        vz.firing_overview(inputs_plus_context.T, figsize=(10,10), cmap='viridis', zscore_bool=False)
        plt.suptitle(f"RNN Input_{dataset_name}")
        plt.savefig(os.path.join(plots_dir, f'inputs_taste_{taste_ind}_{dataset_name}.png'))
        plt.close()

        inputs_plus_context = inputs_plus_context[:-1]
        labels = inputs[1:]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        inputs_tensor = torch.tensor(inputs_plus_context, dtype=torch.float32)

        # Split into train and test -- update from abu new code
        train_test_split = 0.75
        train_inds = np.random.choice(
                np.arange(inputs.shape[1]), 
                int(train_test_split * inputs.shape[1]), 
                replace = False)
        test_inds = np.setdiff1d(np.arange(inputs.shape[1]), train_inds)
        
        train_inputs = inputs_tensor[:, train_inds].to(device)
        train_labels = labels_tensor[:, train_inds].to(device)
        test_inputs = inputs_tensor[:, test_inds].to(device)
        test_labels = labels_tensor[:, test_inds].to(device)

        # Train model
        net = autoencoderRNN(input_size, hidden_size, output_size, rnn_layers=2, dropout=0.2)
        net.to(device)
        # patience allows for early stopping if things look like they're going in a good direction
        if retrain or not os.path.exists(model_save_path):
            net, loss, cross_val_loss = train_model(
                net, train_inputs, train_labels, output_size,
                train_steps=train_steps, lr=0.001,
                criterion=criterion,
                test_inputs=test_inputs, test_labels=test_labels,
                patience=patience
            )
            torch.save(net, model_save_path)
            with open(os.path.join(artifacts_dir, f'loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(loss, f)
            with open(os.path.join(artifacts_dir, f'cross_val_loss_taste_{taste_ind}.json'), 'w') as f:
                json.dump(cross_val_loss, f)
        else:
            net = torch.load(model_save_path)
        
        # attempting to reconstruct firing as best I can -- that said, if PCA inputs, then can't go back 
        # to whole nrn space...

        ##### ATTEMPTING TO RECONSTRUCT BACK INTO NEURON SPACE? INVERT THE Z-SCORE? ######

        ## THIS PREPARATION HERE is what allows for the inverse scaler ## 

        # step 0: The below: 
        # Forward pass through the trained model
        #   - inputs_tensor: shape (time, trial, input_size)
        #   - outs: predicted output, shape (time, trial, output_size)
        #   - latent_outs: latent state from RNN (e.g., hidden state or bottleneck), shape (time, trial, hidden_size)
        outs, latent_outs = net(inputs_tensor.to(device))
        # Detach from computation graph and move results to CPU for further processing-- idrk why is needed 
        # but without the detatch, throws an error. Stack overflow told me to use detatch 
        outs = outs.cpu().detach().numpy()
        latent_outs = latent_outs.cpu().detach().numpy()
        # rearranging axes to be easier to work with 
        pred_firing = np.moveaxis(outs, 0, -1) # trial, output_size, time
        # store output
        latent_out_list.append(latent_outs)

        # Step 1: [pred_time, trial, output_size] -> [trial, output_size, time]
        pred_firing = np.moveaxis(pred_firing, 0, -1).T  
        # Step 2: Reshape to long form: (trial × time, neuron or (more accurately) output_size)
        pred_firing_long = pred_firing.reshape(-1, pred_firing.shape[-1])

        # much simpler implementation of step 3? 
        """
        # If pca was performed, first reverse PCA, then reverse pca standard scaling
        # highly simplistic version of what's below that I used for debugging 

        if use_pca: 
            # # Reverse NMF scaling
            # pred_firing_long = nmf_scaler.inverse_transform(pred_firing_long)
            # pred_firing_long = pca_scaler.inverse_transform(pred_firing_long)

            # Reverse NMF transform
            # pred_firing_long = nmf_obj.inverse_transform(pred_firing_long)
            pred_firing_long = pca_obj.inverse_transform(pred_firing_long)

        # Reverse standard scaling
        pred_firing_long = scaler.inverse_transform(pred_firing_long)

        # review the implementation of the below-- not convinced I ned to be comparing against 
        # neurons at all...
        # reverting to original implementation (as I've confused myself with below)


        """
        # rationale: 
        # I want to scale back into the original neuron space (num_neurons)
        # However, because I am doing pca on my inputs, will nessecarily reduce dims somehow (?)
        # and actually, what I'm doing is decoding back out into PCA'd firing rate space
        # or so I think. At any rate, with a 95% var pca, I will very likely never reach the total number 
        # of components as I had neurons. 
        # this is a big thing to check with abu to make sure I'm unpacking the correct thing. 

        # Step 3: Reverse PCA and/or scaling if possible
        if use_pca:
            try:
                # Case 1: Model decoded into PCA space — safe to reverse PCA
                if pred_firing_long.shape[1] == pca_obj.n_components_:
                    pred_firing_long = pca_obj.inverse_transform(pred_firing_long)
                    print(f"[INFO] Successfully reversed PCA: shape {pred_firing_long.shape}")
                    
                    # Try to reverse z-scoring (StandardScaler) if shapes match
                    if pred_firing_long.shape[1] == scaler.mean_.shape[0]:
                        pred_firing_long = scaler.inverse_transform(pred_firing_long)
                        print(f"[INFO] Successfully reversed z-scoring after PCA")
                    else:
                        print(f"[INFO] Skipping scaler inverse: shape {pred_firing_long.shape[1]} ≠ scaler mean dim {scaler.mean_.shape[0]}")

                # Case 2: Model decoded directly into neuron space — skip inverse transforms
                elif output_size == num_neurons:
                    print(f"[INFO] Skipping PCA/scaler inverse: output dim {output_size} = neuron dim {num_neurons}")

                # Case 3: dimensions mismatched — alert and skip
                else:
                    print(f"[WARNING] Could not apply inverse scaling: output dim {output_size} "
                        f"≠ PCA components {pca_obj.n_components_} and ≠ neuron dim {num_neurons}")
            except Exception as e:
                print(f"[WARNING] PCA/scaler inverse failed: {e}")
        
        # Step 4: Reshape back to [trial, neuron, time]
        # Reverse standard scaling
        # pred_firing_long = scaler.inverse_transform(pred_firing_long) # can this work? 
        pred_firing = pred_firing_long.reshape((*pred_firing.shape[:2], -1))

        # Step 5: Final moveaxis to [trial, neuron, time]
        pred_firing = np.moveaxis(pred_firing, 1, 2)
        pred_firing_list.append(pred_firing)

       # Old use of pca that was producing shape errors as I tried to extract back to firing rate space.  
       # if use_pca:
       #     pred_flat = pred_firing.reshape(-1, pred_firing.shape[-1])
       #     pred_flat = pca_obj.inverse_transform(pred_flat)  # crashes if output_size ≠ PCA components bc pca var thresh
       #     pred_flat = scaler.inverse_transform(pred_flat)
       #     pred_firing = pred_flat.reshape((*pred_firing.shape[:2], -1))

        # Plot train/test loss
        fig, ax = plt.subplots()
        ax.plot(loss, label='Train')
        ax.plot(cross_val_loss.keys(), cross_val_loss.values(), label='Test')
        ax.legend()
        ax.set_title(f"Loss Curve_{dataset_name}")
        fig.savefig(os.path.join(plots_dir, f'loss_taste_{taste_ind}_{dataset_name}.png'))
        plt.close()
            # --------------------------
            #   Where the plots live
            # --------------------------

    # Loss plot
        fig, ax = plt.subplots()
        ax.plot(loss, label = 'Train Loss') 
        ax.plot(cross_val_loss.keys(), cross_val_loss.values(), label = 'Test Loss')
        ax.legend(
                bbox_to_anchor=(1.05, 1), 
                loc='upper left', borderaxespad=0.)
        ax.set_title(f'Losses_{dataset_name}') 
        fig.savefig(os.path.join(plots_dir,f'run_loss_taste_{taste_ind}_{dataset_name}.png'),
                    bbox_inches = 'tight')
        plt.close(fig)

        # Firing rate plots
        vz.firing_overview(pred_firing.swapaxes(0,1))
        fig = plt.gcf()
        plt.suptitle(f'RNN Predicted Firing Rates_{dataset_name}')
        fig.savefig(os.path.join(plots_dir, f'firing_pred_taste_{taste_ind}_{dataset_name}.png'))
        plt.close(fig)
        vz.firing_overview(binned_spikes.swapaxes(0,1))
        fig = plt.gcf()
        plt.suptitle(f'Binned Firing Rates_{dataset_name}')
        fig.savefig(os.path.join(plots_dir, f'firing_binned_taste_{taste_ind}_{dataset_name}.png'))
        plt.close(fig)

        # Latent factors
        fig, ax = plt.subplots(latent_outs.shape[-1], 1, figsize = (5,10),
                            sharex = True, sharey = True)
        for i in range(latent_outs.shape[-1]):
            ax[i].imshow(latent_outs[...,i].T, aspect = 'auto')
        plt.suptitle(f'Latent Factors_{dataset_name}')
        fig.savefig(os.path.join(plots_dir, f'latent_factors_taste_{taste_ind}_{dataset_name}.png'))
        plt.close(fig)

        # Mean firing rates
        pred_firing_mean = pred_firing.mean(axis = 0)
        binned_spikes_mean = binned_spikes.mean(axis = 0)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_firing_mean, aspect = 'auto', interpolation = 'none')
        ax[1].imshow(binned_spikes_mean, aspect = 'auto', interpolation = 'none')
        ax[0].set_title('Pred')
        ax[1].set_title('True')
        fig.savefig(os.path.join(plots_dir, f'mean_firing_taste_{taste_ind}_{dataset_name}.png'))
        plt.close(fig)
        # plt.show()

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(zscore(pred_firing_mean,axis=-1), aspect = 'auto', interpolation = 'none')
        ax[1].imshow(zscore(binned_spikes_mean,axis=-1), aspect = 'auto', interpolation = 'none')
        ax[0].set_title('Pred')
        ax[1].set_title('True')
        fig.savefig(os.path.join(plots_dir, f'mean_firing_zscored_taste_{taste_ind}_{dataset_name}.png'))
        plt.close(fig)

        # For every neuron, plot 1) spike raster, 2) convolved firing rate , 
        # 3) RNN predicted firing rate
        ind_plot_dir = os.path.join(plots_dir, 'individual_neurons')
        if not os.path.exists(ind_plot_dir):
            os.makedirs(ind_plot_dir)

        binned_x = np.arange(0, binned_spikes.shape[-1]*bin_size, bin_size)
        pred_x_list.append(binned_x)

        conv_kern = np.ones(250) / 250
        conv_rate = np.apply_along_axis(
                lambda m: np.convolve(m, conv_kern, mode = 'valid'),
                axis = -1, arr = taste_spikes)*bin_size
        conv_x = np.convolve(
                np.arange(taste_spikes.shape[-1]), conv_kern, mode = 'valid')
        conv_rate_list.append(conv_rate)
        conv_x_list.append(conv_x)

        for i in range(binned_spikes.shape[1]):
            fig, ax = plt.subplots(3,1, figsize = (10,10),
                                sharex = True, sharey = False)
            ax[0] = vz.raster(ax[0], taste_spikes[:, i], marker = '|')
            ax[1].plot(conv_x, conv_rate[:,i].T, c = 'k', alpha = 0.1)
            # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
            ax[2].plot(binned_x[1:], pred_firing[:,i].T, 
                    c = 'k', alpha = 0.1)
            # ax[2].sharey(ax[1])
            for this_ax in ax:
                # this_ax.set_xlim([1500, 4000])
                this_ax.axvline(stim_time_val, c = 'r', linestyle = '--')
            ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
            ax[2].set_title('RNN Predicted Firing Rate')
            fig.savefig(
                    os.path.join(ind_plot_dir, 
                                f'neuron_{i}_taste_{taste_ind}_raster_conv_pred.png')
                    )
            plt.close(fig)

        # Plot single-trial latent factors
        trial_latent_dir = os.path.join(plots_dir, 'trial_latent')
        if not os.path.exists(trial_latent_dir):
            os.makedirs(trial_latent_dir)

        for i in range(latent_outs.shape[1]):
            fig, ax = plt.subplots(2,1)
            ax[0].plot(latent_outs[1:,i], alpha = 0.5)
            ax[0].set_title(f'Latent factors for trial {i} {dataset_name}')
            ax[1].plot(zscore(latent_outs[1:,i], axis = 0), alpha = 0.5)
            fig.savefig(os.path.join(trial_latent_dir, f'taste_{taste_ind}_trial_{i}_latent.png'))
            plt.close(fig)
            # Mean neuron firing
            pred_firing_taste_mean = np.stack(
                    [pred_firing_list[i].mean(axis = 0) for i in range(len(pred_firing_list))])
            binned_spikes_taste_mean = np.stack(
                    [bin_spikes.mean(axis = 0) for bin_spikes in binned_spikes_list])

            cmap = plt.get_cmap('tab10')
            fig, ax = vz.gen_square_subplots(len(pred_firing_mean),
                                            figsize = (10,10),
                                            sharex = True,)
            for nrn_ind in range(pred_firing_taste_mean.shape[1]):
                for taste_ind, (pred, bin) in enumerate(
                        zip(pred_firing_taste_mean, binned_spikes_taste_mean)
                        ):
                    ax.flatten()[nrn_ind].plot(pred[nrn_ind], alpha = 1, c = cmap(taste_ind))
                    ax.flatten()[nrn_ind].plot(bin[nrn_ind], alpha = 0.3, c = cmap(taste_ind))
                ax.flatten()[nrn_ind].set_ylabel(str(nrn_ind))
            fig.savefig(os.path.join(plots_dir, f'mean_neuron_firing.png'))
            plt.close(fig)

        # Make another plot with taste_mean firing rates
        cmap = plt.get_cmap('tab10')
        # Iterate over neurons
        for i in range(binned_spikes.shape[1]):
            fig, ax = plt.subplots(3,1, figsize = (10,10),
                                sharex = True, sharey = False)
            # Get spikes from all tastes for this neuron
            this_spikes_list = [x[:,i] for x in spike_array]
            trial_counts = [len(x) for x in this_spikes_list]
            cum_trial_counts = np.cumsum([0, *trial_counts])
            this_cat_spikes = np.concatenate(this_spikes_list)[..., time_lims[0]:time_lims[1]]

            ax[0] = vz.raster(ax[0], this_cat_spikes, marker = '|', color = 'k')
            # Plot colors behind raster traces
            for j in range(len(cum_trial_counts)-1):
                ax[0].axhspan(cum_trial_counts[j], cum_trial_counts[j+1],
                            color = cmap(j), alpha = 0.1, zorder = 0)

            this_conv_rate = np.stack([x[:,i] for x in conv_rate_list])
            this_pred_firing = np.stack([x[:,i] for x in pred_firing_list])

            mean_conv_rate = this_conv_rate.mean(axis = 1)
            mean_pred_firing = this_pred_firing.mean(axis = 1)
            sd_conv_rate = this_conv_rate.std(axis = 1)
            sd_pred_firing = this_pred_firing.std(axis = 1)
            for j in range(mean_conv_rate.shape[0]):
                ax[1].plot(conv_x, mean_conv_rate[j].T, c = cmap(j),
                        linewidth = 2)
                ax[1].fill_between(
                        conv_x, 
                        mean_conv_rate[j] - sd_conv_rate[j],
                        mean_conv_rate[j] + sd_conv_rate[j],
                        color = cmap(j), alpha = 0.1)
                # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
                ax[2].plot(binned_x[1:], mean_pred_firing[j].T,
                        c = cmap(j), linewidth = 2)
                ax[2].fill_between(
                        binned_x[1:], 
                        mean_pred_firing[j] - sd_pred_firing[j],
                        mean_pred_firing[j] + sd_pred_firing[j],
                        color = cmap(j), alpha = 0.1)
                # ax[2].sharey(ax[1])
            for this_ax in ax:
                # this_ax.set_xlim([1500, 4000])
                this_ax.axvline(stim_time_val, c = 'r', linestyle = '--')
            ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
            ax[2].set_title('RNN Predicted Firing Rate')
            fig.savefig(
                    os.path.join(
                        ind_plot_dir, 
                        f'neuron_{i}_mean_raster_conv_pred.png'))
            plt.close(fig)


        # Plot predicted activity vs true activity for every neuron
        for i in range(pred_firing.shape[1]):
            cat_pred_firing = np.concatenate([x[:,i] for x in pred_firing_list])
            cat_binned_spikes = np.concatenate([x[:,i] for x in binned_spikes_list])

            fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
            min_val = min(cat_pred_firing.min(), cat_binned_spikes.min())
            max_val = max(cat_pred_firing.max(), cat_binned_spikes.max())
            img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis',
                        }
            #'vmin':min_val, 'vmax':max_val}
            im0 = ax[0].imshow(cat_pred_firing, **img_kwargs) 
            im1 = ax[1].imshow(cat_binned_spikes[:,1:], **img_kwargs) 
            ax[0].set_title('Pred')
            ax[1].set_title('True')
            # Colorbars under each subplot
            cbar0 = fig.colorbar(im0, ax = ax[0], orientation = 'horizontal')
            cbar1 = fig.colorbar(im1, ax = ax[1], orientation = 'horizontal')
            cbar0.set_label('Firing Rate (Hz)')
            cbar1.set_label('Firing Rate (Hz)')
            fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_firing.png'))
            plt.close(fig)

    # stuff that saves: 
    # ------------------------
    # Save to HDF5
    # ------------------------
    with tables.open_file(data.hdf5_path, 'r+') as hf5:
        if '/rnn_output' not in hf5:
            hf5.create_group('/', 'rnn_output', 'RNN Output')
        rnn_output = hf5.get_node('/rnn_output')

        for taste_ind, (pred, latent) in enumerate(zip(pred_firing_list, latent_out_list)):
            group_name = f'taste_{taste_ind}'
            if hasattr(rnn_output, group_name):
                hf5.remove_node(rnn_output, group_name, recursive=True)
            taste_group = hf5.create_group(rnn_output, group_name, f"Taste {taste_ind}")
            hf5.create_array(taste_group, 'pred_firing', pred)
            hf5.create_array(taste_group, 'latent_out', latent)
            hf5.create_array(taste_group, 'pred_x', np.arange(pred.shape[-1]) * bin_size)

    # ------------------------
    # Save latents to parquet
    # ------------------------
    all_latents = []
    for taste_ind, latent in enumerate(latent_out_list):
        num_trials, num_time, num_latent = latent.shape
        df = pl.DataFrame(latent.reshape(-1, num_latent), schema=[f'latent_dim_{i}' for i in range(num_latent)])
        df = df.with_columns([
            pl.Series("taste", [taste_ind] * len(df)),
            pl.Series("trial", np.repeat(np.arange(num_trials), num_time)),
            pl.Series("time", np.tile(np.arange(num_time), num_trials)),
        ])
        all_latents.append(df)

    combined_df = pl.concat(all_latents)
    combined_df.write_parquet(os.path.join(artifacts_dir, f'{dataset_name}_raw_latent_vectors.parquet'))
    print(f"✅ Saved latent outputs for {dataset_name}")
    combined_df.write_parquet(os.path.join(pred_lat_dir, f'{dataset_name}_raw_latent_vectors.parquet'))
    print(f"📂 Also saved to {pred_lat_dir}")
    # ------------------------
    # Save predicted firing rates to parquet
    # ------------------------
    all_firing = []
    neuron_counts = []

    for taste_ind, pred_firing in enumerate(pred_firing_list):
        # pred_firing: (trials, neurons, time)
        num_trials, num_neurons, num_time = pred_firing.shape
        neuron_counts.append(num_neurons)

        # Rearrange to (trials, time, neurons) → then flatten to (trial*time, neurons)
        reshaped_pred = pred_firing.transpose(0, 2, 1).reshape(-1, num_neurons)

        # Create DataFrame with appropriate schema
        df = pl.DataFrame(
            reshaped_pred,
            schema=[f'neuron_{i}' for i in range(num_neurons)]
        )

        # Add metadata columns
        df = df.with_columns([
            pl.Series("taste", [taste_ind] * len(df)),
            pl.Series("trial", np.repeat(np.arange(num_trials), num_time)),
            pl.Series("time", np.tile(np.arange(num_time), num_trials)),
        ])

        all_firing.append(df)

    # Verify neuron shape consistency
    if len(set(neuron_counts)) != 1:
        raise ValueError(f"[ERROR] Neuron counts differ across tastes: {neuron_counts}")

    # Combine and save
    combined_firing_df = pl.concat(all_firing)
    combined_firing_df.write_parquet(os.path.join(artifacts_dir, f"{dataset_name}_raw_predicted_firing.parquet"))
    print(f" DONE: Saved predicted firing rates for {dataset_name}")
    # Also save to shared firing output directory
    combined_firing_df.write_parquet(os.path.join(pred_fr_dir, f"{dataset_name}_raw_predicted_firing.parquet"))
    print(f"Also saved to {pred_fr_dir}")