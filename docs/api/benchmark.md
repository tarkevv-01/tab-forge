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
