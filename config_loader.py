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
    params = dict(
        train_steps=config['parameters']['train_steps'],
        hidden_size=config['parameters']['hidden_size'],
        bin_size=config['parameters']['bin_size'],
        train_test_split=config['parameters']['train_test_split'],
        use_pca=config['parameters']['use_pca'],
        retrain=config['parameters']['retrain'],
        time_lims=config['parameters']['time_lims'],
        loss_name=config['parameters'].get('loss_name', 'mse'),
        patience=config['parameters'].get('patience', 12000),
        validation_mode=config['parameters'].get('validation_mode', 'split'),  # 'split' or 'loo'. Split is defaulkt
        loo_train_steps=config['parameters'].get('loo_train_steps', 3000),
        loo_patience=config['parameters'].get('loo_patience', 75),
    )

    # --- Paths ---
    h5_dir = config['paths']['h5_dir']
    output_base_dir = config['paths']['output_base_dir']
    stim_time_val = 2000 - params['time_lims'][0]

    pred_fr_dir = os.path.join(output_base_dir, 'pred_fr')
    pred_lat_dir = os.path.join(output_base_dir, 'pred_latent')
    os.makedirs(pred_fr_dir, exist_ok=True)
    os.makedirs(pred_lat_dir, exist_ok=True)

    paths = dict(
        h5_dir=h5_dir,
        output_base_dir=output_base_dir,
        pred_fr_dir=pred_fr_dir,
        pred_lat_dir=pred_lat_dir,
        stim_time_val=stim_time_val,
    )

    # --- Loss function ---
    from train import MSELoss, smooth_MSELoss
    loss_dict = {
        'mse': MSELoss(),
        'smooth': smooth_MSELoss(alpha=0.05),
    }
    criterion = loss_dict.get(params['loss_name'], MSELoss())

    return config, paths, params, criterion
