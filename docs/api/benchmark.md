# API Reference — Benchmark

## Import

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

Evaluates the quality of synthetic data relative to real data using a set of metrics.

**Parameters:**

- `metrics_spec` — metric specification. Supports two formats:

=== "List"

    ```python
    bench = Benchmark([
        ("r2",       {"model": "xgboost"}),
        ("rmse",     {"model": "xgboost"}),
        ("js_mean",  {}),
        ("frob_corr",{}),
        ("frob_mi",  {}),
    ])
    ```

=== "Dictionary (with names)"

    ```python
    bench = Benchmark({
        "r2_xgb":    ("r2",   {"model": "xgboost"}),
        "r2_linear": ("r2",   {"model": "linearregression"}),
        "js":        ("js_mean", {}),
    })
    ```

---

### Available Metrics

| String | Metric class | Parameters |
|--------|-------------|-----------|
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

Computes all specified metrics.

**Parameters:**

- `synthetic_data` — synthetic data as a `Dataset`
- `real_data` — real data as a `Dataset` (test set)

**Returns:** `BenchmarkResult`

!!! example ""

    ```python
    synth = model.structed_generate(n_samples=len(test))
    result = bench.evaluate(synth, test)
    print(result.metrics)
    # {'r2_xgb': 0.743, 'js': 0.041}
    ```

!!! note ""
    The `fit` method is an alias for `evaluate`. Both work the same way.

---

## `BenchmarkResult`

Dataclass with evaluation results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `metrics` | `dict` | Dictionary `{metric_name: value}` |

!!! example ""

    ```python
    print(result)
    # BenchmarkResult(r2_xgb=0.743, rmse=1.823, js=0.041)

    # Access a specific metric
    r2_value = result.metrics["r2_xgb"]
    ```

---

## Metrics in Detail

### `r2_metric(synthetic: Dataset, real: Dataset, model="xgboost") -> float`

Trains an ML model on synthetic data, evaluates on real data. Returns R².

**Scheme:** `fit(synth_X, synth_y)` → `predict(real_X)` → `R²(real_y, predictions)`

### `rmse_metric(synthetic: Dataset, real: Dataset, model="xgboost") -> float`

Same as R², but returns RMSE (root mean square error).

### `jensen_shannon_metric(synthetic: Dataset, real: Dataset) -> float`

Mean Jensen–Shannon Divergence across all numerical features. JSD is computed via histograms.

### `frob_corr_metric(synthetic: Dataset, real: Dataset) -> float`

Frobenius norm of the difference between correlation matrices:
\(\| \text{Corr}(X_\text{real}) - \text{Corr}(X_\text{synth}) \|_F\)

### `frob_mi_metric(synthetic: Dataset, real: Dataset) -> float`

Frobenius norm of the difference between mutual information matrices:
\(\| \text{MI}(X_\text{real}) - \text{MI}(X_\text{synth}) \|_F\)

---

## Custom Metrics

`Benchmark` accepts an arbitrary callable instead of a metric string. Function signature:

```python
def my_metric(synthetic: Dataset, real: Dataset, **kwargs) -> float:
    ...
```

The function receives two `Dataset` objects and must return a single number. Parameters from the `kwargs` dictionary can be passed via the metric specification.

### Example: Wasserstein Distance and Anonymity

Suppose you are interested in two additional synthetic data characteristics not available in the built-in metrics:

1. **Wasserstein distance** — more sensitive to distribution shape than JS divergence
2. **Anonymity measure** — average distance to the nearest neighbor in real data (the larger — the "safer" the synthetic data in terms of privacy)

```python
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree

def wasserstein_mean(synthetic, real):
    """Mean Wasserstein distance across all numerical features."""
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
    Privacy estimate: 5th percentile of distance from each synthetic
    point to the nearest real one (normalized by standard deviation).
    Larger value means synthetic data is farther from real records.
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
    # Standard prediction quality metrics
    "r2_xgb":        ("r2",        {"model": "xgboost"}),
    "r2_linear":     ("r2",        {"model": "linear"}),
    "rmse_xgb":      ("rmse",      {"model": "xgboost"}),
    # Statistical distribution similarity
    "js_mean":       ("js_mean",   {}),
    "frob_corr":     ("frob_corr", {}),
    "frob_mi":       ("frob_mi",   {}),
    # Custom metrics
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

!!! tip "Kwargs in custom metrics"
    Parameters defined in the specification (`{"quantile": 0.05}`) are passed to the function as keyword arguments. This allows reusing one function with different settings:

    ```python
    benchmark = Benchmark({
        "privacy_p05": (nearest_neighbour_distance, {"quantile": 0.05}),
        "privacy_p25": (nearest_neighbour_distance, {"quantile": 0.25}),
    })
    ```
