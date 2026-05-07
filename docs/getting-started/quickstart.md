# Быстрый старт

За эту страницу мы пройдём полный цикл: загрузка данных → LLM-выбор модели → тюнинг → генерация синтетики → оценка качества.

Для примера используем датасет **abalone** (задача регрессии: предсказание возраста моллюска по физическим характеристикам).

---

## Шаг 1 — Загрузка данных

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="examples/abalone.csv",          # путь к CSV или pd.DataFrame
    target="Rings",                        # целевая переменная
    task_type="regression",               # "regression" или "classification"
    numerical_features=[
        "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight",
    ],
    categorical_features=["Sex"],
)

print(dataset.info)
# DatasetInfo(n_samples=4177, n_features=8, n_numerical=7, n_categorical=1, ...)
```

Разобьём на обучающую и тестовую выборки:

```python
train, test = dataset.train_test_split(test_size=0.2, random_state=42)
print(f"Train: {len(train)} строк, Test: {len(test)} строк")
```

---

## Шаг 2 — LLM выбирает лучшую модель

`PromptGenerator` вычисляет мета-характеристики датасета (размер, соотношение числовых/категориальных признаков, пропуски, корреляции и пр.) и формирует из них структурированный промпт для LLM.

```python
from tab_forge.prompt_generator import PromptGenerator

gen = PromptGenerator()
prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",   # по какой метрике ранжировать модели
    shot_mode     = "few",         # few-shot: LLM видит опыт на референсных датасетах
    mfe_features  = "short",       # кураторский набор из 12 ключевых мета-фич
)
```

Теперь отправляем промпт в LLM. `LLMRunner` делает `n_runs` независимых запросов и усредняет ранги — это называется **self-consistency**:

```python
from tab_forge.llm_runner import LLMRunner

runner = LLMRunner(
    base_url = "https://api.openai.com/v1",
    api_key  = "sk-...",
    model    = "gpt-4o",
)

result = runner.run(prompt, n_runs=5, temperature=0.7)

print("Итоговый рейтинг:", result.final_ranking)
# → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

print("Средние ранги:", result.average_ranks)
# → {'GAN-MFS': 1.4, 'CTGAN': 2.2, 'WGAN-GP': 2.8, ...}
```

!!! note "Локальная LLM"
    LLMRunner совместим с любым OpenAI-совместимым API, включая локальные серверы на базе llama.cpp, Ollama и LM Studio. Просто укажите `base_url` нужного сервера.

---

## Шаг 3 — Тюнинг гиперпараметров

Берём лучшую по версии LLM модель и запускаем байесовский тюнинг через Optuna. Режим `"extended"` автоматически задаёт пространство поиска для данной архитектуры:

```python
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = result.final_ranking[0],  # строка-имя из рейтинга
    search_space_mode = "extended",               # автоматическое пространство поиска
    cv                = 3,                        # количество fold-ов для CV
)

study.optimize(dataset, n_trials=25)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение:", study.best_value)
```

!!! tip "Тюнинг по рейтингу"
    Если у вас есть время — запустите тюнинг по очереди для нескольких первых моделей из `result.final_ranking` и сравните итоговые метрики.

---

## Шаг 4 — Генерация и оценка качества

```python
from tab_forge.models import GANMFSSynthesizer
from tab_forge.benchmark import Benchmark

# Создаём модель с лучшими параметрами и обучаем на полном train
model = GANMFSSynthesizer(**study.best_params)
model.fit(train)

# Генерируем синтетику размером с тест
synth = model.structed_generate(n_samples=len(test))

# Оцениваем качество
bench = Benchmark([
    ("r2",      {"model": "xgboost"}),
    ("rmse",    {"model": "xgboost"}),
    ("js_mean", {}),
])

result_bench = bench.evaluate(synth, test)
print(result_bench)
```

---

## Полный пайплайн одним блоком

```python
from tab_forge.dataset import Dataset
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner
from tab_forge.tuning import AutoTuningStudy
from tab_forge.benchmark import Benchmark

# ── 1. Данные ──────────────────────────────────────────────
dataset = Dataset(
    data="examples/abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight",
                        "Viscera weight", "Shell weight"],
    categorical_features=["Sex"],
)
train, test = dataset.train_test_split(test_size=0.2, random_state=42)

# ── 2. LLM-выбор модели ────────────────────────────────────
prompt = PromptGenerator().build_prompt(
    dataset=dataset, target_metric="r2_metric", shot_mode="few"
)
llm_result = LLMRunner(
    base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o"
).run(prompt, n_runs=5)
print("Рейтинг:", llm_result.final_ranking)

# ── 3. Тюнинг ──────────────────────────────────────────────
study = AutoTuningStudy(
    model_class=llm_result.final_ranking[0],
    search_space_mode="extended",
    cv=3,
)
study.optimize(train, n_trials=25)
print("Лучшие параметры:", study.best_params)

# ── 4. Генерация и оценка ──────────────────────────────────
from tab_forge.models import (
    GANMFSSynthesizer, CTGANSynthesizer, WGANGPSynthesizer,
    CTABGANPlusSynthesizer, TVAESynthesizer, DDPMSynthesizer,
)
MODEL_REGISTRY = {
    "GAN-MFS": GANMFSSynthesizer, "CTGAN": CTGANSynthesizer,
    "WGAN-GP": WGANGPSynthesizer, "CTABGAN+": CTABGANPlusSynthesizer,
    "TVAE":    TVAESynthesizer,    "DDPM":    DDPMSynthesizer,
}

model_class = MODEL_REGISTRY[llm_result.final_ranking[0]]
model = model_class(**study.best_params)
model.fit(train)
synth = model.structed_generate(n_samples=len(test))

bench = Benchmark([("r2", {"model": "xgboost"}), ("js_mean", {})])
print(bench.evaluate(synth, test))
```

---

## Следующие шаги

- Узнайте подробнее о [метриках качества](../concepts/metrics.md)
- Изучите все [6 моделей](../user-guide/models.md) и когда их использовать
- Посмотрите [результаты экспериментов](../experiments/tuning-results.md)
