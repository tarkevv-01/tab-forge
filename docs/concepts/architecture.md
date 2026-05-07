# Архитектура Tab-Forge

## Схема модулей

![Схема архитектуры Tab-Forge](../img/arch.png)

---

## Принцип модульности

Каждый модуль Tab-Forge — самостоятельный и не требует остальных. Вы можете:

- Использовать только **Dataset + Models + Benchmark** без LLM
- Запустить **AutoTuningStudy** вручную без LLM-рейтинга
- Использовать **PromptGenerator** отдельно для изучения мета-фич датасета
- Подключить свою модель к **Benchmark** вне Tab-Forge

---

## Поток данных в полном пайплайне
![Схема универсального pipeline Tab-forge](../img/pipeline.png)

---

## Модуль `dataset`

**Ключевые экспорты:** `Dataset`, `merge_datasets`, `split_folds`

`Dataset` — центральный объект библиотеки. Все модели принимают его на вход и возвращают его из `structed_generate`. Это позволяет беспрепятственно передавать данные между модулями.

```python
from tab_forge.dataset import Dataset, merge_datasets, split_folds
```

---

## Модуль `models`

**Ключевые экспорты:** 6 классов-синтезаторов

Все модели наследуют `BaseGenerativeModel` и реализуют единый интерфейс:

| Метод | Описание |
|-------|----------|
| `fit(dataset)` | Обучение на `Dataset` |
| `generate(n_samples)` | Генерация как `pd.DataFrame` |
| `structed_generate(n_samples)` | Генерация как `Dataset` (для Benchmark/AutoTuning) |
| `get_losses()` | История потерь в процессе обучения |

---

## Модуль `tuning`

**Ключевые экспорты:** `AutoTuningStudy`, `TuningStudy`

`TuningStudy` — тонкая обёртка над `optuna.Study`. `AutoTuningStudy` — полноценный пайплайн CV-тюнинга с предустановленными пространствами поиска для каждой архитектуры.

---

## Модуль `benchmark`

**Ключевые экспорты:** `Benchmark`, `BenchmarkResult`

Все метрики реализованы по схеме train-on-synthetic / test-on-real или через сравнение статистик. Принимает `Dataset` объекты.

---

## Модуль `prompt_generator`

**Ключевые экспорты:** `PromptGenerator`

Использует `pymfe` для вычисления мета-фич и загружает предрасчитанные результаты экспериментов из `experiment_results/` для few-shot режима.

---

## Модуль `llm_runner`

**Ключевые экспорты:** `LLMRunner`, `RunnerResult`

Работает с любым OpenAI-совместимым API. Реализует self-consistency: N независимых вызовов → усреднение рангов через `_aggregate`.
