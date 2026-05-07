# Модуль Dataset

## Зачем это нужно

Разные генеративные модели ожидают данные в разных форматах: одни хотят чистый `pd.DataFrame`, другим нужна информация о типах признаков, третьим — отдельно числовые и категориальные колонки. Каждый раз вручную пробрасывать эту информацию утомительно и чревато ошибками.

`Dataset` — это единый контейнер, который хранит данные вместе с метаинформацией: целевую переменную, тип задачи, списки числовых и категориальных признаков. Все модули Tab-Forge работают именно с `Dataset` — это делает пайплайн типобезопасным и предсказуемым.

---

## Основной класс — `Dataset`

### Создание из CSV

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="abalone.csv",           # путь к CSV или pd.DataFrame
    target="Rings",               # целевая переменная
    task_type="regression",       # "regression" или "classification"
    numerical_features=[
        "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight",
    ],
    categorical_features=["Sex"],
)
```

### Создание из DataFrame

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
dataset = Dataset(
    data=df,
    target="target_col",
    task_type="classification",
    categorical_features=["category_a", "category_b"],
    # numerical_features можно опустить — остальное будет определено автоматически
)
```

---

## DatasetInfo — мета-информация

После создания объекта `Dataset` доступен атрибут `info`:

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

`DatasetInfo` хранит:

| Поле | Смысл |
|------|-------|
| `n_samples` | Количество строк |
| `n_features` | Количество признаков (без целевой) |
| `n_numerical` | Число числовых признаков |
| `n_categorical` | Число категориальных признаков |
| `n_registered` | Признаки, готовые к обработке моделями |
| `task_type` | `"regression"` или `"classification"` |
| `target_name` | Имя целевой переменной |

---

## Разбиение данных

### train_test_split

```python
train, test = dataset.train_test_split(
    test_size    = 0.2,
    random_state = 42,
    shuffle      = True,
    stratify     = False,  # True только для classification
)

print(f"Train: {len(train)}, Test: {len(test)}")
```

!!! note "Стратификация"
    Параметр `stratify=True` доступен только для `task_type="classification"` и использует `StratifiedKFold` под капотом.

### split_folds — для кросс-валидации

```python
from tab_forge.dataset import split_folds

folds = split_folds(dataset, n_splits=5, shuffle=True, random_state=42)
# folds — список из 5 Dataset-объектов

for i, fold in enumerate(folds):
    print(f"Fold {i}: {len(fold)} строк")
```

!!! note ""
    `split_folds` используется внутри `AutoTuningStudy` автоматически — вам не нужно вызывать его вручную при тюнинге.

---

## Объединение датасетов

```python
from tab_forge.dataset import merge_datasets

# Объединение нескольких совместимых датасетов
combined = merge_datasets([train, test])
```

!!! warning "Совместимость датасетов"
    `merge_datasets` проверяет, что у всех объектов одинаковые: целевая переменная, тип задачи, набор признаков. При несовпадении выбрасывается исключение.

---

## Получение данных

```python
# Получить все данные включая целевую переменную
df = dataset.get_data()

# Получить только признаки (без target)
X = dataset.get_X()

# Получить только целевую переменную
y = dataset.get_target()

# Краткая сводка
summary = dataset.summary()
# {"shape": (4177, 9), "task_type": "regression", "missing": {...}}
```

---

## Регистрация признаков

Иногда датасет содержит дополнительные признаки, которые нужно добавить к уже созданному объекту:

```python
dataset.register_features(
    numerical   = ["new_num_feature"],
    categorical = ["new_cat_feature"],
)
```

---

## Типичные паттерны использования

!!! example "Паттерн 1: загрузка + сплит"

    ```python
    dataset = Dataset(data="data.csv", target="y", task_type="regression")
    train, test = dataset.train_test_split(test_size=0.2)
    ```

!!! example "Паттерн 2: k-fold для тюнинга"

    ```python
    from tab_forge.dataset import split_folds, merge_datasets

    folds = split_folds(train, n_splits=5)
    # Для fold 0: валидация на folds[0], обучение на merge(folds[1:])
    val_fold = folds[0]
    train_folds = merge_datasets(folds[1:])
    ```

!!! example "Паттерн 3: из синтетики обратно в Dataset"

    Когда модель возвращает `Dataset` через `structed_generate`, его можно сразу передать в `Benchmark`:

    ```python
    synth = model.structed_generate(n_samples=500)
    result = bench.evaluate(synth, test)
    ```
