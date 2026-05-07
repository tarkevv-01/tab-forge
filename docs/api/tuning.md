# API Reference — Tuning

## Импорт

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

Полноценный пайплайн байесовской оптимизации гиперпараметров с k-fold кросс-валидацией. Интегрирует `Dataset`, модели, `Benchmark` и Optuna в единый workflow.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `model_class` | класс или `str` | — | Класс синтезатора или строка-имя (`"CTGAN"`, `"GAN-MFS"`, и т.д.) |
| `get_params` | `Callable(trial) -> dict` | `None` | Функция пространства поиска для режима `"manual"` |
| `cv` | `int` | `3` | Количество фолдов |
| `sampler` | Optuna sampler | `TPESampler` | Алгоритм поиска гиперпараметров |
| `search_space_mode` | `str` | `"manual"` | `"manual"` или `"extended"` |
| `benchmark` | `Benchmark` | авто | Объект Benchmark для оценки качества |
| `direction` | `str` | из конфига | `"maximize"` или `"minimize"` |
| `**study_kwargs` | — | — | Дополнительные аргументы для `optuna.create_study` |

!!! note "Строки-имена моделей"
    Допустимые строки для `model_class`:
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

Запускает байесовскую оптимизацию.

**Параметры:**

| Параметр | Описание |
|----------|----------|
| `dataset` | Датасет для тюнинга |
| `n_trials` | Количество триалов Optuna |
| `verbose` | Включить вывод Optuna |

**Возвращает:** объект `optuna.Study` с полной историей триалов.

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

### Свойства `AutoTuningStudy`

| Свойство | Тип | Описание |
|----------|-----|----------|
| `best_params` | `dict` | Лучшие найденные гиперпараметры |
| `best_value` | `float` | Лучшее значение целевой метрики |
| `best_trial` | `optuna.Trial` | Объект лучшего триала Optuna |

---

## `TuningStudy`

```
TuningStudy(
    direction: str = "maximize",
    **study_kwargs,
)
```

Тонкая обёртка над `optuna.Study`. Используйте когда нужен полный контроль над objective-функцией.

**Параметры:**

- `direction` — `"maximize"` или `"minimize"`
- `**study_kwargs` — аргументы для `optuna.create_study` (например, `sampler=TPESampler(seed=42)`)

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

Запускает оптимизацию с произвольной objective-функцией.

!!! example ""

    ```python
    from tab_forge.tuning import TuningStudy

    study = TuningStudy(direction="maximize")

    def objective(trial):
        params = {"epochs": trial.suggest_int("epochs", 100, 500)}
        # ... обучение и оценка ...
        return score

    study.optimize(objective, n_trials=20)
    print(study.best_params)
    ```

---

### Свойства `TuningStudy`

| Свойство | Тип | Описание |
|----------|-----|----------|
| `best_params` | `dict` | Лучшие найденные параметры |
| `best_value` | `float` | Лучшее значение objective |
| `best_trial` | `optuna.Trial` | Объект лучшего триала |

---

## Samplers

Tab-Forge реэкспортирует оптимальные samplers из Optuna:

```python
from tab_forge.tuning import TPESampler, RandomSampler
```

| Sampler | Когда использовать |
|---------|-------------------|
| `TPESampler` | По умолчанию. Байесовская оптимизация через Tree-structured Parzen Estimator. Эффективен при 20+ триалах. |
| `RandomSampler` | Случайный поиск. Подходит для небольшого числа триалов или как baseline. |

```python
from tab_forge.tuning import AutoTuningStudy, TPESampler

study = AutoTuningStudy(
    model_class = "CTGAN",
    search_space_mode = "extended",
    sampler = TPESampler(seed=42),  # фиксируем seed для воспроизводимости
)
```
