# API Reference — Benchmark

## Импорт

```python
from tab_forge.benchmark import Benchmark, BenchmarkResult
```

---

## `Benchmark`

```
Benchmark(
    metrics_spec: Union[List[Tuple], Dict[str, Tuple]],
)
```

Оценивает качество синтетических данных относительно реальных по набору метрик.

**Параметры:**

- `metrics_spec` — спецификация метрик. Поддерживает два формата:

=== "Список"

    ```python
    bench = Benchmark([
        ("r2",       {"model": "xgboost"}),
        ("rmse",     {"model": "xgboost"}),
        ("js_mean",  {}),
        ("frob_corr",{}),
        ("frob_mi",  {}),
    ])
    ```

=== "Словарь (с именами)"

    ```python
    bench = Benchmark({
        "r2_xgb":    ("r2",   {"model": "xgboost"}),
        "r2_linear": ("r2",   {"model": "linearregression"}),
        "js":        ("js_mean", {}),
    })
    ```

---

### Доступные метрики

| Строка | Класс метрики | Параметры |
|--------|--------------|-----------|
| `"r2"` | `r2_metric` | `model`: `"xgboost"` / `"linearregression"` |
| `"rmse"` | `rmse_metric` | `model`: `"xgboost"` / `"linearregression"` |
| `"js_mean"` | `jensen_shannon_metric` | — |
| `"frob_corr"` | `frob_corr_metric` | — |
| `"frob_mi"` | `frob_mi_metric` | — |

---

### `Benchmark.evaluate`

```
evaluate(
    synthetic_data: Dataset,
    real_data: Dataset,
) -> BenchmarkResult
```

Вычисляет все заданные метрики.

**Параметры:**

- `synthetic_data` — синтетические данные в виде `Dataset`
- `real_data` — реальные данные в виде `Dataset` (тестовая выборка)

**Возвращает:** `BenchmarkResult`

!!! example ""

    ```python
    synth = model.structed_generate(n_samples=len(test))
    result = bench.evaluate(synth, test)
    print(result.metrics)
    # {'r2_xgb': 0.743, 'js': 0.041}
    ```

!!! note ""
    Метод `fit` — алиас для `evaluate`. Оба работают одинаково.

---

## `BenchmarkResult`

Датакласс с результатами оценки.

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `metrics` | `dict` | Словарь `{имя_метрики: значение}` |

!!! example ""

    ```python
    print(result)
    # BenchmarkResult(r2_xgb=0.743, rmse=1.823, js=0.041)

    # Доступ к конкретной метрике
    r2_value = result.metrics["r2_xgb"]
    ```

---

## Метрики подробно

### `r2_metric(synthetic: Dataset, real: Dataset, model="xgboost") -> float`

Обучает ML-модель на синтетических данных, оценивает на реальных. Возвращает R².

**Схема:** `fit(synth_X, synth_y)` → `predict(real_X)` → `R²(real_y, predictions)`

### `rmse_metric(synthetic: Dataset, real: Dataset, model="xgboost") -> float`

Аналогично R², но возвращает RMSE (среднеквадратичная ошибка).

### `jensen_shannon_metric(synthetic: Dataset, real: Dataset) -> float`

Среднее значение Jensen–Shannon Divergence по всем числовым признакам. JSD вычисляется через гистограммы.

### `frob_corr_metric(synthetic: Dataset, real: Dataset) -> float`

Норма Фробениуса разности корреляционных матриц:
\(\| \text{Corr}(X_\text{real}) - \text{Corr}(X_\text{synth}) \|_F\)

### `frob_mi_metric(synthetic: Dataset, real: Dataset) -> float`

Норма Фробениуса разности матриц взаимной информации:
\(\| \text{MI}(X_\text{real}) - \text{MI}(X_\text{synth}) \|_F\)

---

## Кастомные метрики

`Benchmark` принимает произвольную callable вместо строки метрики. Сигнатура функции:

```python
def my_metric(synthetic: Dataset, real: Dataset, **kwargs) -> float:
    ...
```

Функция получает два объекта `Dataset` и должна вернуть одно число. Параметры из словаря `kwargs` можно передавать через спецификацию метрик.

### Пример: Wasserstein-дистанция и анонимность

Допустим, вас интересуют две дополнительные характеристики синтетических данных, которых нет во встроенных метриках:

1. **Wasserstein-дистанция** — чувствительнее к форме распределений, чем JS-дивергенция
2. **Мера анонимности** — средняя дистанция до ближайшего соседа в реальных данных (чем больше — тем «безопаснее» синтетика с точки зрения приватности)

```python
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree

def wasserstein_mean(synthetic, real):
    """Средняя Wasserstein-дистанция по всем числовым признакам."""
    num_cols = real.numerical_features
    distances = [
        wasserstein_distance(
            real.data[col].dropna(),
            synthetic.data[col].dropna(),
        )
        for col in num_cols
    ]
    return float(np.mean(distances))


def nearest_neighbour_distance(synthetic, real, quantile=0.05):
    """
    Оценка приватности: 5-й перцентиль расстояния от каждой синтетической
    точки до ближайшей реальной (нормировано на стандартное отклонение).
    Чем больше значение — тем дальше синтетика от реальных записей.
    """
    num_cols = real.numerical_features
    real_arr  = real.data[num_cols].dropna().values
    synth_arr = synthetic.data[num_cols].dropna().values

    std = real_arr.std(axis=0)
    std[std == 0] = 1.0
    real_norm  = real_arr  / std
    synth_norm = synth_arr / std

    tree = cKDTree(real_norm)
    dists, _ = tree.query(synth_norm, k=1)
    return float(np.quantile(dists, quantile))


benchmark = Benchmark({
    # Стандартные метрики качества предсказания
    "r2_xgb":        ("r2",        {"model": "xgboost"}),
    "r2_linear":     ("r2",        {"model": "linear"}),
    "rmse_xgb":      ("rmse",      {"model": "xgboost"}),
    # Статистическое сходство распределений
    "js_mean":       ("js_mean",   {}),
    "frob_corr":     ("frob_corr", {}),
    "frob_mi":       ("frob_mi",   {}),
    # Кастомные метрики
    "wasserstein":   (wasserstein_mean,          {}),
    "privacy_p05":   (nearest_neighbour_distance, {"quantile": 0.05}),
})

result = benchmark.evaluate(synth_dataset, test_dataset)
print(result.metrics)
# {
#   'r2_xgb': 0.821,  'r2_linear': 0.764,
#   'rmse_xgb': 1.14, 'js_mean': 0.031,
#   'frob_corr': 0.18, 'frob_mi': 0.22,
#   'wasserstein': 0.043, 'privacy_p05': 1.87
# }
```

!!! tip "Kwargs в кастомных метриках"
    Параметры, заданные в спецификации (`{"quantile": 0.05}`), передаются в функцию как keyword-аргументы. Это позволяет переиспользовать одну функцию с разными настройками:

    ```python
    benchmark = Benchmark({
        "privacy_p05": (nearest_neighbour_distance, {"quantile": 0.05}),
        "privacy_p25": (nearest_neighbour_distance, {"quantile": 0.25}),
    })
    ```
