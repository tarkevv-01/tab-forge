# Dataset Module

## Why You Need This

Different generative models expect data in different formats: some want a plain `pd.DataFrame`, others need information about feature types, and others need numerical and categorical columns separately. Manually propagating this information each time is tedious and error-prone.

`Dataset` is a unified container that stores data together with meta-information: the target variable, task type, and lists of numerical and categorical features. All Tab-Forge modules work with `Dataset` — this makes the pipeline type-safe and predictable.

---

## The Main Class — `Dataset`

### Create from CSV

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="abalone.csv",           # path to CSV or pd.DataFrame
    target="Rings",               # target variable
    task_type="regression",       # "regression" or "classification"
    numerical_features=[
        "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight",
    ],
    categorical_features=["Sex"],
)
```

### Create from DataFrame

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
dataset = Dataset(
    data=df,
    target="target_col",
    task_type="classification",
    categorical_features=["category_a", "category_b"],
    # numerical_features can be omitted — the rest will be determined automatically
)
```

---

## DatasetInfo — Meta-Information

After creating a `Dataset` object, the `info` attribute is available:

```python
print(dataset.info)
```

```
DatasetInfo(
  n_samples=4177,
  n_features=8,
  n_numerical=7,
  n_categorical=1,
  n_registered=8,
  task_type='regression',
  target_name='Rings',
  ...
)
```

`DatasetInfo` stores:

| Field | Meaning |
|-------|---------|
| `n_samples` | Number of rows |
| `n_features` | Number of features (excluding target) |
| `n_numerical` | Number of numerical features |
| `n_categorical` | Number of categorical features |
| `n_registered` | Features ready for model processing |
| `task_type` | `"regression"` or `"classification"` |
| `target_name` | Name of the target variable |

---

## Splitting Data

### train_test_split

```python
train, test = dataset.train_test_split(
    test_size    = 0.2,
    random_state = 42,
    shuffle      = True,
    stratify     = False,  # True only for classification
)

print(f"Train: {len(train)}, Test: {len(test)}")
```

!!! note "Stratification"
    The `stratify=True` parameter is available only for `task_type="classification"` and uses `StratifiedKFold` under the hood.

### split_folds — for cross-validation

```python
from tab_forge.dataset import split_folds

folds = split_folds(dataset, n_splits=5, shuffle=True, random_state=42)
# folds — list of 5 Dataset objects

for i, fold in enumerate(folds):
    print(f"Fold {i}: {len(fold)} rows")
```

!!! note ""
    `split_folds` is used internally by `AutoTuningStudy` automatically — you do not need to call it manually during tuning.

---

## Merging Datasets

```python
from tab_forge.dataset import merge_datasets

# Merge several compatible datasets
combined = merge_datasets([train, test])
```

!!! warning "Dataset compatibility"
    `merge_datasets` checks that all objects have the same: target variable, task type, and feature set. An exception is raised on mismatch.

---

## Accessing Data

```python
# Get all data including the target variable
df = dataset.get_data()

# Get only features (without target)
X = dataset.get_X()

# Get only the target variable
y = dataset.get_target()

# Brief summary
summary = dataset.summary()
# {"shape": (4177, 9), "task_type": "regression", "missing": {...}}
```

---

## Registering Features

Sometimes a dataset contains additional features that need to be added to an already created object:

```python
dataset.register_features(
    numerical   = ["new_num_feature"],
    categorical = ["new_cat_feature"],
)
```

---

## Common Usage Patterns

!!! example "Pattern 1: load + split"

    ```python
    dataset = Dataset(data="data.csv", target="y", task_type="regression")
    train, test = dataset.train_test_split(test_size=0.2)
    ```

!!! example "Pattern 2: k-fold for tuning"

    ```python
    from tab_forge.dataset import split_folds, merge_datasets

    folds = split_folds(train, n_splits=5)
    # For fold 0: validate on folds[0], train on merge(folds[1:])
    val_fold = folds[0]
    train_folds = merge_datasets(folds[1:])
    ```

!!! example "Pattern 3: from synthetic data back to Dataset"

    When a model returns a `Dataset` via `structed_generate`, it can be directly passed to `Benchmark`:

    ```python
    synth = model.structed_generate(n_samples=500)
    result = bench.evaluate(synth, test)
    ```
