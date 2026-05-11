# API Reference — Tuning

## Import

```python
from tab_forge.tuning import AutoTuningStudy, TuningStudy
from tab_forge.tuning import TPESampler, RandomSampler  # Optuna samplers
```

---

## `AutoTuningStudy`

```
AutoTuningStudy(
    model_class: Union[type, str],
    get_params: Optional[Callable] = None,
    cv: int = 3,
    sampler = None,
    search_space_mode: str = "manual",
    benchmark: Optional[Benchmark] = None,
    direction: Optional[str] = None,
    **study_kwargs,
)
```

Full Bayesian hyperparameter optimization pipeline with k-fold cross-validation. Integrates `Dataset`, models, `Benchmark`, and Optuna into a unified workflow.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_class` | class or `str` | — | Synthesizer class or name string (`"CTGAN"`, `"GAN-MFS"`, etc.) |
| `get_params` | `Callable(trial) -> dict` | `None` | Search space function for `"manual"` mode |
| `cv` | `int` | `3` | Number of folds |
| `sampler` | Optuna sampler | `TPESampler` | Hyperparameter search algorithm |
| `search_space_mode` | `str` | `"manual"` | `"manual"` or `"extended"` |
| `benchmark` | `Benchmark` | auto | Benchmark object for quality evaluation |
| `direction` | `str` | from config | `"maximize"` or `"minimize"` |
| `**study_kwargs` | — | — | Additional arguments for `optuna.create_study` |

!!! note "Model name strings"
    Valid strings for `model_class`:
    `"CTGAN"`, `"WGAN-GP"`, `"GAN-MFS"`, `"CTABGAN+"`, `"TVAE"`, `"DDPM"`

---

### `AutoTuningStudy.optimize`

```
optimize(
    dataset: Dataset,
    n_trials: int,
    verbose: bool = False,
    **kwargs,
) -> optuna.Study
```

Launches Bayesian optimization.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset for tuning |
| `n_trials` | Number of Optuna trials |
| `verbose` | Enable Optuna output |

**Returns:** `optuna.Study` object with complete trial history.

!!! example "Extended mode"

    ```python
    from tab_forge.models import CTGANSynthesizer
    from tab_forge.tuning import AutoTuningStudy

    study = AutoTuningStudy(
        model_class       = CTGANSynthesizer,
        search_space_mode = "extended",
        cv                = 3,
    )
    study.optimize(dataset, n_trials=25)
    print(study.best_params)
    ```

!!! example "Manual mode"

    ```python
    from tab_forge.models import GANMFSSynthesizer
    from tab_forge.tuning import AutoTuningStudy

    def my_params(trial):
        return {
            "epochs":       trial.suggest_int("epochs", 200, 500),
            "generator_lr": trial.suggest_float("generator_lr", 1e-4, 1e-3, log=True),
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

### `AutoTuningStudy` Properties

| Property | Type | Description |
|----------|------|-------------|
| `best_params` | `dict` | Best found hyperparameters |
| `best_value` | `float` | Best value of the target metric |
| `best_trial` | `optuna.Trial` | Best Optuna trial object |

---

## `TuningStudy`

```
TuningStudy(
    direction: str = "maximize",
    **study_kwargs,
)
```

Thin wrapper around `optuna.Study`. Use when full control over the objective function is needed.

**Parameters:**

- `direction` — `"maximize"` or `"minimize"`
- `**study_kwargs` — arguments for `optuna.create_study` (e.g., `sampler=TPESampler(seed=42)`)

---

### `TuningStudy.optimize`

```
optimize(
    objective: Callable,
    n_trials: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
)
```

Runs optimization with an arbitrary objective function.

!!! example ""

    ```python
    from tab_forge.tuning import TuningStudy

    study = TuningStudy(direction="maximize")

    def objective(trial):
        params = {"epochs": trial.suggest_int("epochs", 100, 500)}
        # ... training and evaluation ...
        return score

    study.optimize(objective, n_trials=20)
    print(study.best_params)
    ```

---

### `TuningStudy` Properties

| Property | Type | Description |
|----------|------|-------------|
| `best_params` | `dict` | Best found parameters |
| `best_value` | `float` | Best objective value |
| `best_trial` | `optuna.Trial` | Best trial object |

---

## Samplers

Tab-Forge re-exports optimal samplers from Optuna:

```python
from tab_forge.tuning import TPESampler, RandomSampler
```

| Sampler | When to use |
|---------|------------|
| `TPESampler` | Default. Bayesian optimization via Tree-structured Parzen Estimator. Effective with 20+ trials. |
| `RandomSampler` | Random search. Suitable for a small number of trials or as a baseline. |

```python
from tab_forge.tuning import AutoTuningStudy, TPESampler

study = AutoTuningStudy(
    model_class = "CTGAN",
    search_space_mode = "extended",
    sampler = TPESampler(seed=42),  # fix seed for reproducibility
)
```
