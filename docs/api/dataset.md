# API Reference — Dataset

## Импорт

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

Унифицированный контейнер для табличных данных. Хранит DataFrame вместе с метаинформацией о признаках, целевой переменной и типе задачи. Является центральным объектом пайплайна — все модели принимают `Dataset` на вход и возвращают его из `structed_generate`.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | `str` или `pd.DataFrame` | Путь к CSV-файлу или готовый DataFrame |
| `target` | `str` | Имя целевой переменной (колонка в данных) |
| `task_type` | `str` | Тип задачи: `"regression"` или `"classification"` |
| `numerical_features` | `list[str]`, опционально | Список числовых признаков |
| `categorical_features` | `list[str]`, опционально | Список категориальных признаков |

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `info` | `DatasetInfo` | Мета-информация о датасете |

!!! example "Создание Dataset"

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

Разбивает датасет на обучающую и тестовую выборки.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `test_size` | `float` | `0.2` | Доля тестовой выборки (0–1) |
| `random_state` | `int` | `None` | Seed для воспроизводимости |
| `shuffle` | `bool` | `True` | Перемешивать ли данные перед разбиением |
| `stratify` | `bool` | `False` | Стратифицированное разбиение (только для classification) |

**Возвращает:** кортеж `(train_dataset, test_dataset)` — два объекта `Dataset`.

!!! example ""

    ```python
    train, test = dataset.train_test_split(test_size=0.2, random_state=42)
    ```

---

### `Dataset.get_data`

```
get_data() -> pd.DataFrame
```

Возвращает полный DataFrame, включая целевую переменную.

---

### `Dataset.get_X`

```
get_X(registered_only: bool = False) -> pd.DataFrame
```

Возвращает DataFrame только с признаками (без целевой переменной).

**Параметры:**

- `registered_only` — если `True`, возвращает только «зарегистрированные» признаки (переданные при создании или через `register_features`)

---

### `Dataset.get_target`

```
get_target() -> pd.Series
```

Возвращает целевую переменную как `pd.Series`.

---

### `Dataset.register_features`

```
register_features(
    numerical: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
)
```

Добавляет дополнительные признаки к уже созданному датасету.

---

### `Dataset.summary`

```
summary() -> dict
```

Возвращает краткую сводку: форму данных, тип задачи, количество пропусков по колонкам.

---

## `DatasetInfo`

Датакласс с мета-информацией о датасете. Доступен через `dataset.info`.

| Поле | Тип | Описание |
|------|-----|----------|
| `n_samples` | `int` | Количество строк |
| `n_features` | `int` | Количество признаков |
| `n_numerical` | `int` | Число числовых признаков |
| `n_categorical` | `int` | Число категориальных признаков |
| `n_registered` | `int` | Зарегистрированных признаков |
| `task_type` | `str` | Тип задачи |
| `target_name` | `str` | Имя целевой переменной |

---

## `merge_datasets`

```
merge_datasets(
    datasets: List[Dataset],
    reset_index: bool = True,
) -> Dataset
```

Объединяет список совместимых `Dataset`-объектов в один.

**Параметры:**

- `datasets` — список объектов `Dataset`
- `reset_index` — сбрасывать ли индекс после объединения

**Исключения:** поднимает `ValueError` если датасеты несовместимы (разные таргеты, типы задач или наборы признаков).

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

Разбивает датасет на `n_splits` фолдов для кросс-валидации.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_splits` | `5` | Количество фолдов |
| `shuffle` | `True` | Перемешивание перед разбиением |
| `random_state` | `None` | Seed |
| `stratified` | `False` | Стратифицированное разбиение (для classification) |

**Возвращает:** список из `n_splits` объектов `Dataset` примерно равного размера.

!!! example ""

    ```python
    folds = split_folds(dataset, n_splits=3, random_state=42)
    val = folds[0]
    train = merge_datasets(folds[1:])
    ```
