# Benchmark Module

## Why You Need This

After generating synthetic data, you need to answer the question: how good is it? Tab-Forge provides a standardized tool for evaluation — `Benchmark`. It accepts synthetic and real data as `Dataset` objects and returns numbers for selected metrics.

---

## Basic Usage

```python
from tab_forge.benchmark import Benchmark

bench = Benchmark([
    ("r2",      {"model": "xgboost"}),
    ("rmse",    {"model": "xgboost"}),
    ("js_mean", {}),
])

result = bench.evaluate(synth_dataset, real_dataset)
print(result)
```

---

## Metric Specification

Benchmark accepts metrics in two formats:

### List (without names)

```python
bench = Benchmark([
    ("r2",        {"model": "xgboost"}),
    ("rmse",      {"model": "linearregression"}),
    ("js_mean",   {}),
    ("frob_corr", {}),
    ("frob_mi",   {}),
])
```

### Dictionary (with names)

Convenient when you need to run the same metric with different parameters:

```python
bench = Benchmark({
    "r2_xgb":    ("r2",   {"model": "xgboost"}),
    "r2_linear": ("r2",   {"model": "linearregression"}),
    "js_diverg": ("js_mean", {}),
})
```

---

## Available Metrics

| String | Description | Parameters |
|--------|-------------|-----------|
| `"r2"` | Coefficient of determination | `model`: `"xgboost"` / `"linearregression"` |
| `"rmse"` | Root mean square error | `model`: `"xgboost"` / `"linearregression"` |
| `"js_mean"` | Mean Jensen–Shannon across all features | — |
| `"frob_corr"` | Frobenius norm of the difference between correlation matrices | — |
| `"frob_mi"` | Frobenius norm of the difference between mutual information matrices | — |

!!! note "ML models for r2 and rmse"
    The `model` parameter specifies which ML algorithm is used for evaluation. `"xgboost"` is recommended — it is nonlinear and better reflects the real utility of synthetic data.

---

## BenchmarkResult

`bench.evaluate()` returns a `BenchmarkResult`:

```python
result = bench.evaluate(synth, test)

# Get all metrics
print(result.metrics)
# {'r2_xgb': 0.743, 'rmse_xgb': 1.823, 'js_diverg': 0.041}

# Or call repr
print(result)
# BenchmarkResult(r2_xgb=0.743, rmse_xgb=1.823, js_diverg=0.041)
```

---

## Interpreting Results

### R²

- `R² > 0.8` — excellent synthetic data: ML models train on it almost as well as on real data
- `R² 0.5–0.8` — acceptable for augmentation
- `R² < 0.3` — synthetic data is practically useless for ML

### RMSE

Depends on the scale of the target variable. Use RMSE of a real model (trained on real data) as a benchmark:

```python
# Real model as baseline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Train on real train, evaluate on real test
real_model = LinearRegression().fit(train.get_X(), train.get_target())
baseline_rmse = np.sqrt(mean_squared_error(test.get_target(), real_model.predict(test.get_X())))

print(f"Baseline RMSE (real data): {baseline_rmse:.3f}")
print(f"Synth RMSE: {result.metrics['rmse']:.3f}")
```

### Jensen–Shannon

- `JS < 0.05` — distributions are very close
- `JS 0.05–0.2` — small divergence, acceptable
- `JS > 0.3` — significant divergence of marginal distributions

### Frobenius Correlation and MI

Absolute values depend on dimensionality (number of features). It makes sense to compare different models against each other or track changes during tuning.

---

## Use in AutoTuningStudy

`AutoTuningStudy` creates `Benchmark` automatically from the model config by default. But you can provide your own:

```python
from tab_forge.benchmark import Benchmark
from tab_forge.tuning import AutoTuningStudy

custom_bench = Benchmark([
    ("r2",   {"model": "xgboost"}),
    ("rmse", {"model": "xgboost"}),
])

study = AutoTuningStudy(
    model_class       = "CTGAN",
    search_space_mode = "extended",
    cv                = 3,
    benchmark         = custom_bench,
    direction         = "maximize",
)
```

!!! warning "Single metric for tuning"
    Tuning requires **one** scalar objective. If your `Benchmark` returns multiple metrics, `AutoTuningStudy` averages them. For separate optimization, run multiple individual studies.

---

## Complete Evaluation Example

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark

# Load data
dataset = Dataset(data="abalone.csv", target="Rings", task_type="regression",
                  numerical_features=["Length", "Diameter", "Height",
                                      "Whole weight", "Shucked weight"],
                  categorical_features=["Sex"])

train, test = dataset.train_test_split(test_size=0.2, random_state=42)

# Train the model
model = CTGANSynthesizer(epochs=300)
model.fit(train)

# Generate and evaluate
synth = model.structed_generate(n_samples=len(test))

bench = Benchmark({
    "r2":       ("r2",       {"model": "xgboost"}),
    "rmse":     ("rmse",     {"model": "xgboost"}),
    "js":       ("js_mean",  {}),
    "frob_c":   ("frob_corr",{}),
    "frob_mi":  ("frob_mi",  {}),
})

result = bench.evaluate(synth, test)
print(result)
```
