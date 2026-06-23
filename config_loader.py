"""
config loader for this whole runtime thing

"""



import os
import sys
import json


def load_config(config_path):
    """
    Load config JSON and return paths dict, params dict, and criterion.

    Returns:
        config: raw config dict
        paths: dict with all resolved paths
        params: dict with all parameters
        criterion: loss function instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- Append to sys.path ---
    for key in ['underlying_functions', 'ephys_data', 'src']:
        p = config['paths'].get(key)
        if p and p not in sys.path:
            sys.path.append(p)

    # --- Parameters ---
    # Start with everything in the JSON so new keys flow through automatically
    params = dict(config['parameters'])

    # Apply defaults for optional keys that may be absent
    params.setdefault('loss_name', 'mse')
    params.setdefault('patience', 12000)
    params.setdefault('validation_mode', 'split')   # 'split' or 'loo'; split is default
    params.setdefault('loo_train_steps', 3000)
    params.setdefault('loo_patience', 75)
    params.setdefault('rnn_layers', 2)
    params.setdefault('dropout', 0.2)
    params.setdefault('lr', 0.001)

    # --- Paths ---
    paths = dict(config['paths'])

    output_base_dir = paths['output_base_dir']
    paths['stim_time_val'] = 2000 - params['time_lims'][0]
    paths['pred_fr_dir'] = os.path.join(output_base_dir, 'pred_fr')
    paths['pred_lat_dir'] = os.path.join(output_base_dir, 'pred_latent')
    os.makedirs(paths['pred_fr_dir'], exist_ok=True)
    os.makedirs(paths['pred_lat_dir'], exist_ok=True)

    #h5_dir = config['paths']['h5_dir']
    #output_base_dir = config['paths']['output_base_dir']
    #stim_time_val = 2000 - params['time_lims'][0]

    #pred_fr_dir = os.path.join(output_base_dir, 'pred_fr')
    #pred_lat_dir = os.path.join(output_base_dir, 'pred_latent')
    #os.makedirs(pred_fr_dir, exist_ok=True)
    #os.makedirs(pred_lat_dir, exist_ok=True)

    # paths = dict(
    #     h5_dir=h5_dir,
    #     output_base_dir=output_base_dir,
    #     pred_fr_dir=pred_fr_dir,
    #     pred_lat_dir=pred_lat_dir,
    #     stim_time_val=stim_time_val,
    # )

    # --- Loss function ---
    from train import MSELoss, smooth_MSELoss
    loss_dict = {
        'mse': MSELoss(),
        'smooth': smooth_MSELoss(alpha=0.05),
    }
    criterion = loss_dict.get(params['loss_name'], MSELoss())

    return config, paths, params, criterion
