# Модуль Tuning

## Зачем это нужно

Генеративные модели чувствительны к гиперпараметрам: слишком маленький learning rate — модель не сходится, слишком большой — нестабильное обучение. Перебирать вручную неэффективно. Tab-Forge предоставляет два инструмента:

- `AutoTuningStudy` — полноценный пайплайн с k-fold CV и предустановленными пространствами поиска
- `TuningStudy` — тонкая обёртка над Optuna для гибкого ручного тюнинга

---

## AutoTuningStudy — рекомендуемый путь

`AutoTuningStudy` — главный инструмент тюнинга в Tab-Forge. Он берёт на себя всё: разбиение данных на фолды, обучение модели, оценку качества, агрегацию метрик и передачу Optuna.

### Режим "extended" — автоматическое пространство поиска

Самый простой способ: указать класс модели и дать библиотеке самой определить диапазоны гиперпараметров:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = CTGANSynthesizer,
    search_space_mode = "extended",
    cv                = 3,
)

study.optimize(dataset, n_trials=25)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение:", study.best_value)
```

!!! tip "Имя модели из рейтинга LLM"
    Если у вас есть рейтинг от LLMRunner, можно передать строку с именем модели напрямую:

    ```python
    study = AutoTuningStudy(
        model_class       = result.final_ranking[0],  # "GAN-MFS", "CTGAN" и т.д.
        search_space_mode = "extended",
        cv                = 3,
    )
    ```

### Режим "manual" — ручное пространство поиска

Если вы хотите контролировать диапазоны сами, определите функцию `get_params`:

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

## Как работает тюнинг внутри

На каждом `trial` Optuna предлагает набор гиперпараметров. `AutoTuningStudy` запускает следующий цикл:

```
dataset
  │
  ├── split_folds(cv=3) → [fold_0, fold_1, fold_2]
  │
  └── for каждого fold:
       train_folds = merge(остальные фолды)
       val_fold    = текущий фолд
       
       model = ModelClass(**trial_params)
       model.fit(train_folds)
       
       synth = model.structed_generate(len(val_fold))
       score = benchmark.evaluate(synth, val_fold)
       
  → среднее score по фолдам → цель для Optuna
```

Optuna минимизирует или максимизирует эту цель в зависимости от параметра `direction`.

---

## Параметры AutoTuningStudy

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `model_class` | класс или строка | — | Класс модели или её имя (из рейтинга LLM) |
| `get_params` | callable | `None` | Функция `(trial) -> dict` для режима `"manual"` |
| `cv` | int | `3` | Количество fold-ов |
| `sampler` | Optuna sampler | `TPESampler` | Алгоритм поиска |
| `search_space_mode` | `"manual"` / `"extended"` | `"manual"` | Откуда брать пространство поиска |
| `benchmark` | `Benchmark` | автоматически | Инстанс Benchmark для оценки |
| `direction` | `"maximize"` / `"minimize"` | зависит от модели | Направление оптимизации |

!!! warning "direction по умолчанию"
    В режиме `"extended"` направление оптимизации задаётся конфигом модели. Для R² это `"maximize"`, для RMSE — `"minimize"`. В режиме `"manual"` убедитесь, что `direction` соответствует вашей метрике.

---

## Пространства поиска в режиме "extended"

!!! note "batch_size"
    Во всех моделях `batch_size` подбирается динамически исходя из размера датасета: `[max(384, n//20), max(1024, n//10)]`, результат округляется до 10.

### CTGAN

| Гиперпараметр | Диапазон |
|--------------|----------|
| `epochs` | 100–1000 (шаг 100) |
| `batch_size` | динамически (см. выше) |
| `generator_dim` | 1–4 слоя × 50–150 нейронов |
| `discriminator_dim` | 1–4 слоя × 50–150 нейронов |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `generator_decay` | [1e-4, 1e-3] |
| `discriminator_decay` | [1e-4, 1e-3] |
| `discriminator_steps` | 1–3 |

### WGAN-GP

| Гиперпараметр | Диапазон |
|--------------|----------|
| `epochs` | 100–1000 (шаг 100) |
| `batch_size` | динамически (см. выше) |
| `generator_dim` | [1,2,3,4] слоя × 50–150 нейронов |
| `discriminator_dim` | [1,2,3,4] слоя × 50–150 нейронов |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `embedding_dim` | [128] |
| `gp_weight` | 1.0–10.0 |
| `critic_iterations` | [1, 2, 3] |

### GAN-MFS

| Гиперпараметр | Диапазон |
|--------------|----------|
| `epochs` | 100–1000 (шаг 100) |
| `batch_size` | динамически (см. выше) |
| `generator_dim` | [1,2,3,4] слоя × 50–150 нейронов |
| `discriminator_dim` | [1,2,3,4] слоя × 50–150 нейронов |
| `generator_lr` | [1e-4, 2e-4, 1e-3] |
| `discriminator_lr` | [1e-4, 2e-4, 1e-3] |
| `embedding_dim` | [128] |
| `mfs_lambda` | 0.1–1.5 |
| `sample_frac` | 0.3–0.5 |
| `gp_weight` | 1.0–10.0 |
| `critic_iterations` | [1, 2, 3] |

### CTABGAN+

| Гиперпараметр | Диапазон |
|--------------|----------|
| `epochs` | 100–1000 (шаг 100) |
| `batch_size` | динамически (см. выше) |
| `class_dim` | 1–4 слоя × [64, 128, 256] нейронов |
| `lr` | [1e-4, 2e-4, 1e-3] |
| `random_dim` | [64, 128, 256, 512] |
| `critic_iterations` | 1–3 |
| `l2scale` | [1e-4, 1e-3] |

### TVAE

| Гиперпараметр | Диапазон |
|--------------|----------|
| `epochs` | 100–1000 (шаг 100) |
| `batch_size` | динамически (см. выше) |
| `compress_dims` | [2,3,4] слоя × [256, 512] нейронов |
| `decompress_dims` | [2,4] слоя × [256, 512] нейронов |
| `embedding_dim` | [16, 32, 64] |
| `l2scale` | 1e-5 – 6.3e-5 (log) |
| `loss_factor` | [2, 3] |

### TabDDPM

| Гиперпараметр | Диапазон |
|--------------|----------|
| `batch_size` | динамически: 64 – min(4096, n//4) (шаг 64) |
| `lr` | 1e-5 – 2e-3 (log) |
| `num_timesteps` | [100, 250, 500, 1000] |
| `scheduler` | ['linear', 'cosine'] |
| `n_layers_hidden` | 2–6 |
| `n_units_hidden` | [256, 512, 1024] |
| `dropout` | 0.0–0.2 (шаг 0.05) |

---

## TuningStudy — низкоуровневый интерфейс

Если нужен полный контроль над Optuna:

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

## Визуализация результатов

`TuningStudy` и `AutoTuningStudy` возвращают `study` объект Optuna с полными данными трайлов. Можно использовать встроенную визуализацию Optuna:

```python
import optuna.visualization as vis

# История оптимизации
fig = vis.plot_optimization_history(study.best_trial.study)

# Важность гиперпараметров
fig2 = vis.plot_param_importances(study.best_trial.study)
```

Подробные примеры — в `examples/test_tuning.ipynb`.
