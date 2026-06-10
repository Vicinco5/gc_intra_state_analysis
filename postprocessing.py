"""
post-processing: taking the reconstructed predicted firing rates back out into neuron space. 
Inverse PCA, inverse scaling. 

"""
import numpy as np

def reconstruct_firing(outs, scaler, pca_obj=None, num_neurons=None, use_pca=False):
    """
    Reconstruct firing rates from model output.

    Pipeline:
        1. Rearrange axes: (time, trial, output) -> (trial, output, time)
        2. Reshape to long form
        3. Inverse PCA if applicable
        4. Inverse scaling if applicable
        5. Reshape back to (trial, neuron, time)

    Args:
        outs: (time, trials, output_size) — raw model output (numpy)
        scaler: fitted StandardScaler
        pca_obj: fitted PCA object or None
        num_neurons: int, original neuron count
        use_pca: bool

    Returns:
        pred_firing: (trials, neurons_or_components, time)
    """
    # (time, trial, output) -> (trial, output, time)
    pred_firing = np.moveaxis(outs, 0, -1)
    pred_firing = np.moveaxis(pred_firing, 0, -1).T

    # Long form: (trial * time, features)
    pred_firing_long = pred_firing.reshape(-1, pred_firing.shape[-1])

    # Inverse transforms
    if use_pca and pca_obj is not None:
        pred_firing_long = _inverse_pca_and_scale(
            pred_firing_long, scaler, pca_obj, num_neurons
        )

    # Reshape back: (trials, features, time)
    pred_firing = pred_firing_long.reshape((*pred_firing.shape[:2], -1))
    pred_firing = np.moveaxis(pred_firing, 1, 2)

    return pred_firing


def _inverse_pca_and_scale(pred_long, scaler, pca_obj, num_neurons):
    """
    Attempt inverse PCA and inverse scaling. Handles shape mismatches
    gracefully with informative messages.

    Args:
        pred_long: (n_samples, n_features)
        scaler: fitted StandardScaler
        pca_obj: fitted PCA
        num_neurons: int

    Returns:
        pred_long: (n_samples, n_features) — possibly expanded
    """
    try:
        if pred_long.shape[1] == pca_obj.n_components_:
            pred_long = pca_obj.inverse_transform(pred_long)
            print(f"  [INFO] Reversed PCA: shape {pred_long.shape}")

            if pred_long.shape[1] == scaler.mean_.shape[0]:
                pred_long = scaler.inverse_transform(pred_long)
                print(f"  [INFO] Reversed z-scoring after PCA")
            else:
                print(f"  [INFO] Skipping scaler inverse: "
                      f"shape {pred_long.shape[1]} != scaler dim {scaler.mean_.shape[0]}")

        elif pred_long.shape[1] == num_neurons:
            print(f"  [INFO] Output dim = neuron dim ({num_neurons}), "
                  f"skipping PCA/scaler inverse")

        else:
            print(f"  [WARNING] Shape mismatch: output dim {pred_long.shape[1]} "
                  f"!= PCA components {pca_obj.n_components_} "
                  f"and != neuron dim {num_neurons}")

    except Exception as e:
        print(f"  [WARNING] PCA/scaler inverse failed: {e}")

    return pred_long