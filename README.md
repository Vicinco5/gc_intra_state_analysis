# blechRNN — Refactored

Autoencoder-RNN for predicting neural firing rates from binned spike trains. Encoder compresses population activity to a low-dimensional latent space, an RNN learns temporal dynamics in that space, and a decoder projects back to firing rate predictions.

## NOTE: 
Still working on making this whole thing tick. At the moment, it's mostly working but there are still some fit issues that have to be ironed out. 
Due to the RNN being a stochastic measure (after all), there is unfortunately a real limit on what I can reasonably achieve. 

To abu: Working on pushing some of the validation stuff I've got here to DVC... this is going on github for now though. Stuff to do! 

## Pipeline overview

```
Raw spikes → Bin → Z-score → (optional PCA) → Train autoencoder-RNN → Predict firing rates
                                                                      → Extract latent trajectories
```

Two main entry points:

- **`run.py`** — Run the full pipeline with fixed parameters (from config JSON or Optuna override).
- **`optuna_sweep.py`** — Hyperparameter optimization via Optuna, then optionally run the full pipeline with the best params.

## Files

### Core pipeline

| File | Purpose |
|---|---|
| `run.py` | Main script. Loops over datasets and tastes, calls everything else. Supports `USE_OPTUNA_PARAMS` flag to override config with optimized params. |
| `config_loader.py` | Parses `blechrnn_config.json`, sets up paths, parameters, and loss function. |
| `preprocessing.py` | Spike binning, z-scoring, PCA, stimulus/trial context concatenation, tensor creation. Also creates `raw_labels_tensor` for Poisson LL computation. |
| `run_training.py` | Training modes: `train_or_load` (standard split), `loo_then_train` (LOO cross-validation + final retrain), `kfold_evaluate` (K-fold for fast hyperparameter sweeps). |
| `training.py` | Core `train_model` loop, loss functions (`MSELoss`, `smooth_MSELoss`), log-likelihood functions (`gaussian_log_likelihood`, `poisson_log_likelihood`), AIC/BIC utilities, AICr computation. |
| `model.py` | Network architectures: `autoencoderRNN` (primary), `CTRNN`, `CTRNN_plus_output`. |
| `postprocessing.py` | Inverse PCA and inverse z-scoring to reconstruct predicted firing rates. |
| `visualizations.py` | All plotting: loss curves, firing rate heatmaps, latent factors, per-neuron rasters, AIC/BIC summaries, LOO diagnostics. |
| `save_outputs.py` | Saves predicted firing rates and latent vectors to HDF5 and Parquet. |
| `neuron_eval.py` | Post-hoc per-neuron reconstruction evaluation. No retraining — slices existing predictions by neuron. |
| `fixedpoint.py` | Fixed point analysis of the RNN latent dynamics. Finds fixed points, computes Jacobians, classifies stability, supports rolling (input-conditioned) analysis across timesteps. Added plots: Additional time-resolved fixed point visualizations: eigenvalue complex plane and distance-to-trajectory. |

### Hyperparameter optimization

| File | Purpose |
|---|---|
| `optuna_sweep.py` | Main optimization script. All config flags at the top. Supports single-taste, all-tastes, and multi-dataset modes. |
| `optuna_multiobjective.py` | Multi-objective functions (AICr/obs + loss-LL correlation), Pareto front plotting and saving. |
| `optuna_core.py` | Single-objective functions (legacy), `run_optimized_pipeline` for running the full pipeline with best params. |
 
## Config parameters  

Set in `blechrnn_config.json` under `"parameters"`:

```json
{
    "train_steps": 12000,
    "hidden_size": 8,
    "bin_size": 25,
    "train_test_split": 0.75,
    "use_pca": true,
    "retrain": true,
    "time_lims": [1500, 7000],
    "patience": 12000,
    "loss_name": "mse",
    "validation_mode": "loo",
    "loo_train_steps": 3000,
    "loo_patience": 15
}
```

These are defaults. When `USE_OPTUNA_PARAMS = True` in `run.py`, the Optuna params JSON overrides `hidden_size`, `rnn_layers`, `dropout`, `lr`, and `loss_name`.

## Model evaluation metrics

### Dual log-likelihood

The model trains with MSE loss on z-scored (optionally PCA-reduced) data. Evaluation computes two log-likelihoods:

**Gaussian LL (z-scored space):** Directly consistent with the MSE training objective. Treats the MSE as a variance estimate and computes the log-likelihood of a normal distribution. Always well-defined, always negative.

**Poisson LL (raw count space):** Predictions are inverse-transformed back to raw spike counts, then evaluated against actual counts under a Poisson model. Scientifically meaningful — tests whether predicted rates explain observed spikes. Requires `raw_labels_tensor`, `scaler`, and optionally `pca_obj` for the inverse transform.

### AIC and AICr

```
AIC  = 2k - 2LL                    (standard)
AICr = penalty(N, M) - 2LL         (Random-X corrected, DelSole & Tippett 2021)
```

Where `k` = number of trainable parameters, `N` = number of observations, `M` = number of parameters.

**Why AICr:** Standard AIC and even AICc assume the same predictor values in training and validation (Same-X). In cross-validation, held-out trials have different neural activity (Random-X). AICr corrects for this, penalizing complexity more appropriately. It is less likely to select overfitted models.

**Normalized AICr:** For cross-dataset comparisons, AICr is divided by number of observations (`n_obs = time × trials × neurons`). This prevents large datasets from dominating the optimization.

### Loss-LL correlation

The correlation between per-fold training loss and per-fold held-out log-likelihood. Computed for both Gaussian and Poisson LL.

High positive r means the model generalizes consistently — when it trains well, it also predicts well on held-out data. Low or negative r means training performance doesn't translate to generalization (underfitting or overfitting). This metric is naturally bounded between -1 and 1 and doesn't suffer from cross-dataset scaling issues.

## Validation modes

### Train/test split (`validation_mode: "split"`)

Randomly holds out 25% of trials. Trains on 75%, evaluates on the held-out set. Fast (one training run per taste) but noisy with small trial counts and the saved model never sees the held-out data.

### Leave-one-out (`validation_mode: "loo"`)

**Phase 1:** For each of N trials, hold it out, train on N-1, evaluate on the held-out trial. Sum per-trial log-likelihoods for total LOO LL, compute AIC/BIC/AICr. Deterministic, robust.

**Phase 2:** Train a single model on ALL N trials. This model is saved and used for all downstream outputs.

LOO folds use `loo_train_steps` and `loo_patience` for speed. The final model uses `train_steps` and `patience`.

### K-fold (`kfold_evaluate`)

Trials randomly assigned to K folds (default K=5). Each fold held out once, model trained on remaining K-1 folds. Returns AIC/AICr and loss-LL correlations. Designed for Optuna sweeps where you need reliable ranking without the cost of 30-fold LOO. ~6x faster than LOO.

## Hyperparameter optimization (Optuna)

### Overview

`optuna_sweep.py` uses Bayesian optimization (TPE sampler) to search the hyperparameter space. It runs in three modes depending on config flags:

- **Single taste** (`MULTI_DATASET=False`, `ALL_TASTES=False`): Evaluates one taste from one dataset per trial.
- **All tastes** (`MULTI_DATASET=False`, `ALL_TASTES=True`): Evaluates all tastes from one dataset per trial.
- **Multi-dataset** (`MULTI_DATASET=True`): Evaluates all tastes from multiple datasets per trial. Best for finding generalizable params.

### Multi-objective optimization

Each Optuna trial returns two objectives (both minimized):

1. **Poisson AICr / observation** — predictive accuracy penalized for complexity.
2. **Negative loss-LL correlation** — generalization consistency (negated because Optuna minimizes).

Optuna finds the **Pareto front**: the set of configs where you can't improve one metric without worsening the other. A trial on the Pareto front is not dominated by any other trial on both metrics.

### Selecting from the Pareto front

The `PARETO_AICR_WEIGHT` flag (0.0 to 1.0) controls how the best trial is selected:

- `0.0` = pure correlation (most consistent generalization)
- `1.0` = pure AICr (best predictive accuracy per complexity)
- `0.5` = balanced (default)

Both metrics are normalized to [0, 1] across the Pareto front before combining.

### Search space

Defined in `define_search_space()` in `optuna_sweep.py`:

```python
{
    'hidden_size': 8 to 128 (log scale),
    'rnn_layers':  1 to 4,
    'dropout':     0.0 to 0.5 (step 0.05),
    'lr':          1e-4 to 1e-2 (log scale),
    'loss_name':   ['mse'],
}
```

**Important:** Keep `lr` upper bound at `1e-2`. Higher values can produce good K-fold metrics (models stop early before damage) but cause weight explosion during full training runs.

### Outputs

All outputs go to `<output_base_dir>/optuna_multidataset/` (multi-dataset) or `<output_base_dir>/<dataset>/optuna_results/` (per-dataset):

- `optuna_study_*.db` — SQLite database, resumable with `RESUME=True`.
- `optuna_results_*.txt` — Human-readable report with best-by-AICr, best-by-correlation, full trial log.
- `optuna_results_*.json` — Machine-readable results.
- `optuna_params_<datetime>.json` — Timestamped best params export. This is what `run.py` reads when `USE_OPTUNA_PARAMS=True`.
- `pareto_front_*.png` — Pareto front scatter plot (AICr/obs vs correlation, Pareto-optimal trials starred).
- `pareto_params_*.png` — How each parameter varies along the Pareto front.

### Running the optimized pipeline

When `RUN_OPTIMIZED=True`, after the study completes, the full `run.py` pipeline runs with the selected Pareto trial's params. For multi-dataset mode, it runs on all datasets in `DATASET_SUBDIRS`. Outputs go to `<output_base_dir>/optuna_optimization/<dataset>/`.

### Using optimized params in run.py

```python
USE_OPTUNA_PARAMS = True
OPTUNA_PARAMS_PATH = '/path/to/optuna_params_20260313_045103.json'
```

This overrides `hidden_size`, `rnn_layers`, `dropout`, `lr`, and `loss_name` from the config JSON with the Optuna-selected values.

## Neuron-level evaluation

`neuron_eval.py` runs a single forward pass through the trained model and evaluates reconstruction quality per neuron. No retraining. Answers "which neurons does the model reconstruct well?"

Generates diagnostic plots:
- Per-neuron Gaussian LL bar chart with outlier flagging
- LL vs mean firing rate scatter (do high-firing neurons reconstruct better?)
- LL distribution histogram
- Per-neuron Poisson LL (if raw labels available)
- Gaussian vs Poisson LL consistency scatter

When PCA is in use, Gaussian LL evaluates PCA components while Poisson LL evaluates actual neurons (after inverse transform).

## LOO diagnostic plots

Generated per taste when running in LOO mode:

**Gaussian diagnostics (2x3 panels):**
1. Per-trial held-out Gaussian LL — outliers flagged red (>2 SD below mean)
2. Per-fold final training loss
3. Fold loss curves overlaid — overfitting diagnostic
4. Train loss vs held-out LL scatter — generalization check (positive r = healthy)
5. Convergence steps histogram — are folds hitting the step ceiling?
6. Summary statistics box

**Poisson diagnostics (1x3 panels, if available):**
1. Per-trial Poisson LL bars
2. Gaussian vs Poisson LL scatter — consistency between metrics
3. Poisson summary statistics

## Fixed point analysis

`fixedpoint.py` analyzes the dynamical structure the RNN has learned in its latent space. A fixed point is a hidden state `h*` where the dynamics are stationary: `F(h*, x) = h*`. Finding and classifying these reveals the computational scaffold underlying the latent trajectories — attractors the network settles into, saddles that channel transitions between them.

Based on the method from Sussillo & Barak (2013), "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks."

> NOTE: I strongly reccomend reading and then re-reading Sussillo & Barak (2013). 

### How it works

For a given input condition `x`, the analysis minimizes `q(h) = ||F(h, x) - h||²` (the squared speed of the dynamics) from many initial conditions sampled along actual latent trajectories. Points where `q` drops below tolerance are fixed points. Nearby solutions are clustered (DBSCAN) to remove duplicates. The Jacobian `dF/dh` is computed at each fixed point and its eigenvalues classify stability.

### Reading the eigenvalues

For this discrete-time system, the stability boundary is `|eigenvalue| = 1`:
- All `|eig| < 1` → **stable** (attractor): trajectories converge here.
- All `|eig| > 1` → **unstable** (repeller): trajectories are pushed away.
- Mixed → **saddle**: attracting along some directions, repelling along others. Saddles mediate transitions between attractors.

The magnitude sets the timescale: a mode with `|eig| = λ` decays by a factor of λ each timestep. Healthy attractors sit around 0.8–0.95 (decay over a biologically plausible ~100–200 ms). Complex-conjugate eigenvalue pairs indicate rotational/oscillatory dynamics around the fixed point.

### Static vs rolling analysis

**Static (mean input):** Fixed points of the dynamics under a constant input averaged across all timepoints. A coarse reference for the baseline topology. Caveat: the mean input is an average over a non-stationary trial and may never occur at any real timepoint, so it should not be over-interpreted as "the" structure of the dynamics.

**Rolling (input-conditioned):** Fixed points found at multiple timepoints, each conditioned on the actual encoded input at that moment. These are the fixed points genuinely shaping the trajectory at each point in the trial. This is the primary analysis — it shows how the dynamical landscape reorganizes as the trial unfolds.

### Connection to HMM states

The motivating hypothesis: discrete HMM states in the firing correspond to dynamical regimes (attractor-dwelling), and HMM state transitions correspond to **bifurcations** — qualitative reorganizations of the fixed point landscape (an attractor losing stability, a new fixed point appearing, eigenvalues crossing the unit circle). Passing HMM changepoint times as the rolling `time_indices` lets you check whether transitions the HMM detects coincide with dynamical reorganization.

Note: HMM changepoints must be converted to bin indices to match rolling `time_indices`: `bin_idx = (changepoint_ms - time_lims[0]) / bin_size`.

### Visualizations

From `fixedpoint.py`:
- 3D scatter of fixed points overlaid on latent trajectories (stable=green, saddle=orange, unstable=red)
- Rolling 3D view with fixed points colored by time
- Max eigenvalue magnitude over time (watch for crossings of the 1.0 line — bifurcations)
- Fixed point count over time (creation/annihilation of attractors)

From the plot part of `fixedpoint.py` (clearer time-resolved views for the HMM question):
- **Eigenvalue complex plane:** all eigenvalues plotted against the unit circle, tracked across timepoints, with HMM changepoints highlighted. The most direct way to *see* a bifurcation.
- **Distance to trajectory:** distance from the actual latent state to the nearest fixed point over time, colored by nearest-FP stability, with HMM boundaries overlaid. Tests whether HMM states correspond to attractor-dwelling (low distance) and transitions to between-attractor flight (high distance).

### Integration

`run_fixed_point_analysis()` in `fixedpoint.py` runs both static and rolling analysis and generates all plots. Call it from `run.py` after training and prediction. cuDNN must be disabled during the optimization (handled internally via `torch.backends.cudnn.flags(enabled=False)`) because cuDNN does not support RNN backward in eval mode.


## Typical workflow

1. **Set config parameters** in `blechrnn_config.json`.
2. **Run Optuna sweep** (`python optuna_sweep.py`) to find optimal hyperparameters across datasets.
3. **Inspect results**: Pareto front plot, results text file, diagnostic plots.
4. **Run full pipeline** either via `RUN_OPTIMIZED=True` in the sweep or manually with `run.py` + `USE_OPTUNA_PARAMS=True`.
5. **Evaluate**: Check LOO diagnostics, neuron evaluations, predicted firing rate plots, latent trajectories.

## Known issues and pitfalls

- **Learning rate too high:** LR > 0.01 can produce good K-fold metrics (early stopping hides the damage) but causes tanh saturation during full training. Keep the search space upper bound at 1e-2.
- **LOO folds not converging:** If the convergence histogram shows all folds hitting the step ceiling, increase `LOO_TRAIN_STEPS` or decrease `LOO_PATIENCE`.
- **Poisson LL is NaN:** Check that `raw_labels_tensor`, `scaler`, and `pca_obj` are being passed to the training function. Positive Poisson LL values indicate an inverse transform bug.
- **AICr favoring tiny models:** With undertrained folds, small models that plateau quickly look artificially good. Ensure folds converge before evaluation. Trying to avoid seemingly pathological (???) fits? Too small a model really fails flat. 
- **Cross-dataset AIC scaling:** Raw AIC values are not comparable across datasets with different neuron counts. Use AICr/observation for fair comparison.
- **cuDNN RNN backward error during fixed point analysis:** cuDNN can't backprop through an RNN in eval mode. The fixed point code wraps the optimization in `torch.backends.cudnn.flags(enabled=False)` to use the native implementation. Don't remove this.
- **Mean-input fixed points are a fiction:** The mean input averages over a non-stationary trial and may not correspond to any real moment. Use rolling analysis for single-trial interpretation; treat static/mean-input results as a coarse reference only.