# Tab-Forge — Быстрый старт

## Установка

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

---

## Шаг 1. Загрузка данных

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="abalone.csv",           # путь к CSV или pd.DataFrame
    target="Rings",               # целевая переменная
    task_type="regression",       # "regression" или "classification"
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

print(dataset.info)
```

При разбиении на выборки:

```python
from tab_forge.dataset import train_test_split

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
```

---

## Шаг 2. Формирование промта (PromptGenerator)

`PromptGenerator` вычисляет мета-характеристики датасета и собирает из них структурированный промт для LLM.

```python
from prompt_generator import PromptGenerator

gen = PromptGenerator()

prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",   # целевая метрика оптимизации
    shot_mode     = "few",         # "zero" или "few"
    mfe_features  = "short",       # "short", "full" или список строк
)

print(prompt)
```

**`target_metric`** — по какой метрике LLM ранжирует модели:

| Значение | Метрика |
|----------|---------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon divergence |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

**`shot_mode`:**
- `"zero"` — промт содержит только мета-фичи вашего датасета и описания моделей
- `"few"` — дополнительно включаются результаты предварительных экспериментов на 5 референсных датасетах; LLM видит, как каждая модель проявила себя на похожих задачах

---

## Шаг 3. LLM-ранжирование моделей (LLMRunner)

```python
from llm_runner import LLMRunner

runner = LLMRunner(
    base_url = "https://api.openai.com/v1",   # или URL локального сервера
    api_key  = "sk-...",
    model    = "gpt-4o",
)

result = runner.run(
    prompt      = prompt,
    n_runs      = 5,       # количество независимых запросов (self-consistency)
    temperature = 0.7,
)

print(result)
# Выводит итоговый рейтинг и средние ранги каждой модели

print(result.final_ranking)
# ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']
```

`n_runs` — число независимых вызовов LLM. Ранги из каждого ответа усредняются, что делает итоговый список устойчивым к случайному поведению модели.

---

## Шаг 4. Тюнинг гиперпараметров (AutoTuningStudy)

### Режим `"extended"` — рекомендуемый

Пространство поиска для архитектуры задаётся автоматически. Достаточно передать класс модели:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = CTGANSynthesizer,
    search_space_mode = "extended",   # автоматическое пространство поиска
    cv                = 3,            # количество фолдов
)

study.optimize(dataset, n_trials=25)

print(study.best_params)
print(study.best_value)
```

### Режим `"manual"` — гиперпараметры задаются вручную

```python
from tab_forge.models import GANMFSSynthesizer
from tab_forge.tuning import AutoTuningStudy

def my_params(trial):
    return {
        "epochs":       trial.suggest_int("epochs", 200, 500),
        "generator_lr": trial.suggest_float("generator_lr", 1e-4, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }

study = AutoTuningStudy(
    model_class       = GANMFSSynthesizer,
    get_params        = my_params,
    search_space_mode = "manual",
    cv                = 3,
    direction         = "maximize",   # зависит от выбранной метрики
)

study.optimize(dataset, n_trials=30)
```

---

## Полный пайплайн

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer, GANMFSSynthesizer
from tab_forge.tuning import AutoTuningStudy
from prompt_generator import PromptGenerator
from llm_runner import LLMRunner

# Загрузка данных
dataset = Dataset(
    data="abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

# LLM выбирает приоритет моделей
gen = PromptGenerator()
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="few")

runner = LLMRunner(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o")
result = runner.run(prompt, n_runs=5, temperature=0.7)

print("Рейтинг моделей:", result.final_ranking)

# Тюнинг модели с наивысшим приоритетом
MODEL_REGISTRY = {
    "CTGAN":   CTGANSynthesizer,
    "GAN-MFS": GANMFSSynthesizer,
    # ... остальные модели
}

top_model_name = result.final_ranking[0]
model_class = MODEL_REGISTRY.get(top_model_name)

if model_class:
    study = AutoTuningStudy(
        model_class       = model_class,
        search_space_mode = "extended",
        cv                = 3,
    )
    study.optimize(dataset, n_trials=25)
    print(f"Лучшие параметры для {top_model_name}:", study.best_params)
```

---

## Оценка качества (Benchmark)

Если нужно просто оценить синтетические данные:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark

model = CTGANSynthesizer(epochs=300)
model.fit(train)

synth = model.structed_generate(len(test))   # возвращает Dataset

bench = Benchmark([
    ("r2",   {"model": "xgboost"}),
    ("rmse", {"model": "linearregression"}),
    ("js_mean", {}),
])

result = bench.evaluate(synth, test)
print(result)
```

Именованный вариант спецификации метрик:

```python
bench = Benchmark({
    "r2_xgb":     ("r2",   {"model": "xgboost"}),
    "r2_linear":  ("r2",   {"model": "linearregression"}),
    "js_diverg":  ("js_mean", {}),
})
```

---

## Использование модели без тюнинга

```python
from tab_forge.models import CTGANSynthesizer

model = CTGANSynthesizer(epochs=300)
model.fit(dataset)

# Генерация как pd.DataFrame
synth_df = model.generate(n_samples=1000)

# Генерация как Dataset (для Benchmark и AutoTuningStudy)
synth_dataset = model.structed_generate(n_samples=1000)
```

---

## Примеры

Подробные Jupyter-ноутбуки находятся в папке `examples/`:

| Ноутбук | Что показывает |
|---------|----------------|
| `test_dataset.ipynb` | Загрузка данных, разбиение, утилиты |
| `test_model.ipynb` | Обучение и генерация для каждой архитектуры |
| `test_benchmark.ipynb` | Оценка качества синтетических данных |
| `test_tuning.ipynb` | `TuningStudy` и `AutoTuningStudy`, визуализация Optuna |
| `test_prompt_generator.ipynb` | Построение промтов, zero/few-shot, разные наборы мета-фич |
| `test_llm_runner.ipynb` | Полный цикл: промт → LLM → рейтинг |
