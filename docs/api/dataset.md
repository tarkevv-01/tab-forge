# API Reference — Dataset

## Import

```python
from tab_forge.dataset import Dataset, merge_datasets, split_folds
```

---

## `Dataset`

```
Dataset(
    data: Union[str, pd.DataFrame],
    target: str,
    task_type: str,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
)
```

A unified container for tabular data. Stores a DataFrame together with meta-information about features, the target variable, and task type. It is the central object of the pipeline — all models accept `Dataset` as input and return it from `structed_generate`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `str` or `pd.DataFrame` | Path to a CSV file or a ready DataFrame |
| `target` | `str` | Name of the target variable (column in the data) |
| `task_type` | `str` | Task type: `"regression"` or `"classification"` |
| `numerical_features` | `list[str]`, optional | List of numerical features |
| `categorical_features` | `list[str]`, optional | List of categorical features |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `info` | `DatasetInfo` | Dataset meta-information |

!!! example "Creating a Dataset"

    ```python
    dataset = Dataset(
        data="abalone.csv",
        target="Rings",
        task_type="regression",
        numerical_features=["Length", "Diameter", "Height"],
        categorical_features=["Sex"],
    )
    ```

---

### `Dataset.train_test_split`

```
train_test_split(
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False,
) -> Tuple[Dataset, Dataset]
```

Splits the dataset into train and test sets.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_size` | `float` | `0.2` | Fraction of the test set (0–1) |
| `random_state` | `int` | `None` | Seed for reproducibility |
| `shuffle` | `bool` | `True` | Whether to shuffle data before splitting |
| `stratify` | `bool` | `False` | Stratified split (classification only) |

**Returns:** tuple `(train_dataset, test_dataset)` — two `Dataset` objects.

!!! example ""

    ```python
    train, test = dataset.train_test_split(test_size=0.2, random_state=42)
    ```

---

### `Dataset.get_data`

```
get_data() -> pd.DataFrame
```

Returns the full DataFrame, including the target variable.

---

### `Dataset.get_X`

```
get_X(registered_only: bool = False) -> pd.DataFrame
```

Returns a DataFrame with features only (without the target variable).

**Parameters:**

- `registered_only` — if `True`, returns only "registered" features (passed at creation or via `register_features`)

---

### `Dataset.get_target`

```
get_target() -> pd.Series
```

Returns the target variable as a `pd.Series`.

---

### `Dataset.register_features`

```
register_features(
    numerical: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
)
```

Adds additional features to an already created dataset.

---

### `Dataset.summary`

```
summary() -> dict
```

Returns a brief summary: data shape, task type, number of missing values per column.

---

## `DatasetInfo`

Dataclass with dataset meta-information. Accessible via `dataset.info`.

| Field | Type | Description |
|-------|------|-------------|
| `n_samples` | `int` | Number of rows |
| `n_features` | `int` | Number of features |
| `n_numerical` | `int` | Number of numerical features |
| `n_categorical` | `int` | Number of categorical features |
| `n_registered` | `int` | Number of registered features |
| `task_type` | `str` | Task type |
| `target_name` | `str` | Name of the target variable |

---

## `merge_datasets`

```
merge_datasets(
    datasets: List[Dataset],
    reset_index: bool = True,
) -> Dataset
```

Merges a list of compatible `Dataset` objects into one.

**Parameters:**

- `datasets` — list of `Dataset` objects
- `reset_index` — whether to reset the index after merging

**Exceptions:** raises `ValueError` if datasets are incompatible (different targets, task types, or feature sets).

!!! example ""

    ```python
    combined = merge_datasets([fold_1, fold_2, fold_3])
    ```

---

## `split_folds`

```
split_folds(
    dataset: Dataset,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratified: bool = False,
) -> List[Dataset]
```

Splits the dataset into `n_splits` folds for cross-validation.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_splits` | `5` | Number of folds |
| `shuffle` | `True` | Shuffle before splitting |
| `random_state` | `None` | Seed |
| `stratified` | `False` | Stratified split (for classification) |

**Returns:** list of `n_splits` `Dataset` objects of approximately equal size.

!!! example ""

    ```python
    folds = split_folds(dataset, n_splits=3, random_state=42)
    val = folds[0]
    train = merge_datasets(folds[1:])
    ```
