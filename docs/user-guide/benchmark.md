# Модуль Benchmark

## Зачем это нужно

После генерации синтетических данных нужно ответить на вопрос: насколько они хороши? Tab-Forge предоставляет стандартизированный инструмент для оценки — `Benchmark`. Он принимает синтетику и реальные данные в виде `Dataset`-объектов и возвращает числа по выбранным метрикам.

---

## Базовое использование

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

## Спецификация метрик

Benchmark принимает метрики в двух форматах:

### Список (без имён)

```python
bench = Benchmark([
    ("r2",        {"model": "xgboost"}),
    ("rmse",      {"model": "linearregression"}),
    ("js_mean",   {}),
    ("frob_corr", {}),
    ("frob_mi",   {}),
])
```

### Словарь (с именами)

Удобно когда нужно запустить одну и ту же метрику с разными параметрами:

```python
bench = Benchmark({
    "r2_xgb":    ("r2",   {"model": "xgboost"}),
    "r2_linear": ("r2",   {"model": "linearregression"}),
    "js_diverg": ("js_mean", {}),
})
```

---

## Доступные метрики

| Строка | Описание | Параметры |
|--------|----------|-----------|
| `"r2"` | Коэффициент детерминации | `model`: `"xgboost"` / `"linearregression"` |
| `"rmse"` | Среднеквадратичная ошибка | `model`: `"xgboost"` / `"linearregression"` |
| `"js_mean"` | Среднее Jensen–Shannon по всем признакам | — |
| `"frob_corr"` | Норма Фробениуса разности корреляционных матриц | — |
| `"frob_mi"` | Норма Фробениуса разности матриц взаимной информации | — |

!!! note "ML-модели для r2 и rmse"
    Параметр `model` задаёт, какой ML-алгоритм используется для оценки. Рекомендуется `"xgboost"` — он нелинейный и лучше отражает реальную полезность синтетики.

---

## BenchmarkResult

`bench.evaluate()` возвращает `BenchmarkResult`:

```python
result = bench.evaluate(synth, test)

# Получить все метрики
print(result.metrics)
# {'r2_xgb': 0.743, 'rmse_xgb': 1.823, 'js_diverg': 0.041}

# Или вызвать repr
print(result)
# BenchmarkResult(r2_xgb=0.743, rmse_xgb=1.823, js_diverg=0.041)
```

---

## Интерпретация результатов

### R²

- `R² > 0.8` — отличная синтетика: ML-модели обучаются на ней почти так же хорошо, как на реальных данных
- `R² 0.5–0.8` — приемлемо для аугментации
- `R² < 0.3` — синтетика практически бесполезна для ML

### RMSE

Зависит от масштаба целевой переменной. Ориентируйтесь на RMSE реальной модели (обученной на реальных данных) как на бенчмарк:

```python
# Реальная модель как baseline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Обучаем на реальных train, оцениваем на реальных test
real_model = LinearRegression().fit(train.get_X(), train.get_target())
baseline_rmse = np.sqrt(mean_squared_error(test.get_target(), real_model.predict(test.get_X())))

print(f"Baseline RMSE (real data): {baseline_rmse:.3f}")
print(f"Synth RMSE: {result.metrics['rmse']:.3f}")
```

### Jensen–Shannon

- `JS < 0.05` — распределения очень близки
- `JS 0.05–0.2` — небольшое расхождение, приемлемо
- `JS > 0.3` — значительное расхождение маргинальных распределений

### Frobenius Correlation и MI

Абсолютные значения зависят от размерности (количества признаков). Имеет смысл сравнивать разные модели между собой или трекать изменение при тюнинге.

---

## Использование в AutoTuningStudy

`AutoTuningStudy` по умолчанию создаёт `Benchmark` автоматически из конфига модели. Но вы можете передать свой:

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

!!! warning "Одна метрика для тюнинга"
    Для тюнинга нужна **одна** скалярная цель. Если ваш `Benchmark` возвращает несколько метрик, `AutoTuningStudy` усредняет их. Для раздельной оптимизации запустите несколько отдельных study.

---

## Полный пример оценки

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark

# Загрузка данных
dataset = Dataset(data="abalone.csv", target="Rings", task_type="regression",
                  numerical_features=["Length", "Diameter", "Height",
                                      "Whole weight", "Shucked weight"],
                  categorical_features=["Sex"])

train, test = dataset.train_test_split(test_size=0.2, random_state=42)

# Обучение модели
model = CTGANSynthesizer(epochs=300)
model.fit(train)

# Генерация и оценка
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
