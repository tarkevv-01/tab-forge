# Tuning Module

## Why You Need This

Generative models are sensitive to hyperparameters: too small a learning rate — the model does not converge; too large — unstable training. Manual search is inefficient. Tab-Forge provides two tools:

- `AutoTuningStudy` — full pipeline with k-fold CV and preset search spaces
- `TuningStudy` — thin wrapper around Optuna for flexible manual tuning

---

## AutoTuningStudy — Recommended Path

`AutoTuningStudy` is the main tuning tool in Tab-Forge. It handles everything: splitting data into folds, training the model, evaluating quality, aggregating metrics, and passing to Optuna.

### Mode "extended" — automatic search space

The simplest way: specify the model class and let the library define the hyperparameter ranges automatically:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = CTGANSynthesizer,
    search_space_mode = "extended",
    cv                = 3,
)

study.optimize(dataset, n_trials=25)

print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
```

!!! tip "Model name from LLM ranking"
    If you have a ranking from LLMRunner, you can pass the model name string directly:

    ```python
    study = AutoTuningStudy(
        model_class       = result.final_ranking[0],  # "GAN-MFS", "CTGAN", etc.
        search_space_mode = "extended",
        cv                = 3,
    )
    ```

### Mode "manual" — custom search space

If you want to control the ranges yourself, define a `get_params` function:

```python
from tab_forge.models import GANMFSSynthesizer
from tab_forge.tuning import AutoTuningStudy

def my_params(trial):
    return {
        "epochs":       trial.suggest_int("epochs", 200, 500),
        "generator_lr": trial.suggest_float("generator_lr", 1e-4, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "mfs_lambda":   trial.suggest_float("mfs_lambda", 0.01, 0.5, log=True),
    }

study = AutoTuningStudy(
    model_class       = GANMFSSynthesizer,
    get_params        = my_params,
    search_space_mode = "manual",
    cv                = 3,
    direction         = "maximize",
)

study.optimize(dataset, n_trials=30)
```

---

## How Tuning Works Internally

At each `trial`, Optuna proposes a set of hyperparameters. `AutoTuningStudy` runs the following cycle:

```
dataset
  │
  ├── split_folds(cv=3) → [fold_0, fold_1, fold_2]
  │
  └── for each fold:
       train_folds = merge(remaining folds)
       val_fold    = current fold
       
       model = ModelClass(**trial_params)
       model.fit(train_folds)
       
       synth = model.structed_generate(len(val_fold))
       score = benchmark.evaluate(synth, val_fold)
       
  → average score across folds → objective for Optuna
```

Optuna minimizes or maximizes this objective depending on the `direction` parameter.

---

## AutoTuningStudy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_class` | class or string | — | Model class or its name (from LLM ranking) |
| `get_params` | callable | `None` | Function `(trial) -> dict` for `"manual"` mode |
| `cv` | int | `3` | Number of folds |
| `sampler` | Optuna sampler | `TPESampler` | Search algorithm |
| `search_space_mode` | `"manual"` / `"extended"` | `"manual"` | Source of search space |
| `benchmark` | `Benchmark` | automatic | Benchmark instance for evaluation |
| `direction` | `"maximize"` / `"minimize"` | depends on model | Optimization direction |

!!! warning "Default direction"
    In `"extended"` mode, the optimization direction is set by the model config. For R² it is `"maximize"`, for RMSE — `"minimize"`. In `"manual"` mode, make sure `direction` matches your metric.

---

## Search Spaces in "extended" Mode

!!! note "batch_size"
    In all models, `batch_size` is chosen dynamically based on dataset size: `[max(384, n//20), max(1024, n//10)]`, result rounded to 10.

### CTGAN

| Hyperparameter | Range |
|---------------|-------|
| `epochs` | 100–1000 (step 100) |
| `batch_size` | dynamic (see above) |
| `generator_dim` | 1–4 layers × 50–150 neurons |
| `discriminator_dim` | 1–4 layers × 50–150 neurons |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `generator_decay` | [1e-4, 1e-3] |
| `discriminator_decay` | [1e-4, 1e-3] |
| `discriminator_steps` | 1–3 |

### WGAN-GP

| Hyperparameter | Range |
|---------------|-------|
| `epochs` | 100–1000 (step 100) |
| `batch_size` | dynamic (see above) |
| `generator_dim` | [1,2,3,4] layers × 50–150 neurons |
| `discriminator_dim` | [1,2,3,4] layers × 50–150 neurons |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `embedding_dim` | [128] |
| `gp_weight` | 1.0–10.0 |
| `critic_iterations` | [1, 2, 3] |

### GAN-MFS

| Hyperparameter | Range |
|---------------|-------|
| `epochs` | 100–1000 (step 100) |
| `batch_size` | dynamic (see above) |
| `generator_dim` | [1,2,3,4] layers × 50–150 neurons |
| `discriminator_dim` | [1,2,3,4] layers × 50–150 neurons |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `embedding_dim` | [128] |
| `mfs_lambda` | 0.1–1.5 |
| `sample_frac` | 0.3–0.5 |
| `gp_weight` | 1.0–10.0 |
| `critic_iterations` | [1, 2, 3] |

### CTABGAN+

| Hyperparameter | Range |
|---------------|-------|
| `epochs` | 100–1000 (step 100) |
| `batch_size` | dynamic (see above) |
| `class_dim` | 1–4 layers × [64, 128, 256] neurons |
| `lr` | [1e-4, 2e-4, 1e-3] |
| `random_dim` | [64, 128, 256, 512] |
| `critic_iterations` | 1–3 |
| `l2scale` | [1e-4, 1e-3] |

### TVAE

| Hyperparameter | Range |
|---------------|-------|
| `epochs` | 100–1000 (step 100) |
| `batch_size` | dynamic (see above) |
| `compress_dims` | [2,3,4] layers × [256, 512] neurons |
| `decompress_dims` | [2,4] layers × [256, 512] neurons |
| `embedding_dim` | [16, 32, 64] |
| `l2scale` | 1e-5 – 6.3e-5 (log) |
| `loss_factor` | [2, 3] |

### TabDDPM

| Hyperparameter | Range |
|---------------|-------|
| `batch_size` | dynamic: 64 – min(4096, n//4) (step 64) |
| `lr` | 1e-5 – 2e-3 (log) |
| `num_timesteps` | [100, 250, 500, 1000] |
| `scheduler` | ['linear', 'cosine'] |
| `n_layers_hidden` | 2–6 |
| `n_units_hidden` | [256, 512, 1024] |
| `dropout` | 0.0–0.2 (step 0.05) |

---

## TuningStudy — Low-Level Interface

If you need full control over Optuna:

```python
from tab_forge.tuning import TuningStudy
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark
from tab_forge.dataset import split_folds, merge_datasets

bench = Benchmark([("r2", {"model": "xgboost"})])
study = TuningStudy(direction="maximize")

def objective(trial):
    params = {
        "epochs":    trial.suggest_int("epochs", 100, 500),
        "batch_size": trial.suggest_categorical("batch_size", [100, 500]),
    }
    folds = split_folds(dataset, n_splits=3)
    scores = []
    for i in range(3):
        val = folds[i]
        train = merge_datasets([f for j, f in enumerate(folds) if j != i])
        model = CTGANSynthesizer(**params)
        model.fit(train)
        synth = model.structed_generate(len(val))
        result = bench.evaluate(synth, val)
        scores.append(list(result.metrics.values())[0])
    return sum(scores) / len(scores)

study.optimize(objective, n_trials=25)
print(study.best_params)
```

---

## Visualizing Results

`TuningStudy` and `AutoTuningStudy` return an Optuna `study` object with complete trial data. You can use Optuna's built-in visualization:

```python
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study.best_trial.study)

# Hyperparameter importance
fig2 = vis.plot_param_importances(study.best_trial.study)
```

Detailed examples are in `examples/test_tuning.ipynb`.
