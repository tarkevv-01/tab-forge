# Модуль LLM Selection

## Зачем это нужно

Выбор модели для конкретного датасета — это по сути задача мета-обучения: «какая архитектура исторически работала лучше всего на похожих данных?». Вместо того чтобы запускать все 6 моделей и смотреть, Tab-Forge формулирует этот вопрос как промпт для LLM.

Ключевое открытие: LLM, снабжённый мета-характеристиками датасета и результатами экспериментов на референсных наборах данных, способен предсказывать лучшую модель с точностью **top1-acc до 0.84** (few-shot режим).

---

## PromptGenerator

`PromptGenerator` делает две вещи:
1. Извлекает мета-характеристики вашего датасета
2. Собирает из них структурированный промпт

```python
from tab_forge.prompt_generator import PromptGenerator

gen = PromptGenerator()
prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",
    shot_mode     = "few",
    mfe_features  = "short",
)
print(prompt[:500])  # посмотреть начало промпта
```

### Параметры `build_prompt`

| Параметр | Значения | По умолчанию | Описание |
|----------|---------|-------------|----------|
| `dataset` | `Dataset` | — | Ваш датасет |
| `target_metric` | см. ниже | `"r2_metric"` | Целевая метрика для ранжирования |
| `shot_mode` | `"zero"`, `"few"` | `"zero"` | Zero-shot или few-shot |
| `mfe_features` | `"short"`, `"full"`, список | `"short"` | Набор мета-фич |
| `mfe_groups` | список групп pymfe | `None` | Группы для pymfe |
| `models_to_include` | список моделей | все 6 | Ограничить набор моделей |

### Значения target_metric

| Строка | Метрика |
|--------|---------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

---

## Мета-фичи датасета

### Что вычисляется

`PromptGenerator` автоматически вычисляет следующие мета-характеристики:

| Мета-фича | Смысл |
|-----------|-------|
| `nr_inst` | Количество строк |
| `nr_attr` | Количество признаков |
| `nr_num` | Числовые признаки |
| `nr_cat` | Категориальные признаки |
| `missing_pct` | Доля пропущенных значений, % |
| `task_type` | Тип задачи (regression/classification) |
| `abs_corr_mean` | Средняя абсолютная корреляция |
| `abs_corr_max` | Максимальная абсолютная корреляция |
| `skewness_mean` | Средняя асимметрия признаков |
| `kurtosis_mean` | Среднее значение эксцесса |
| `kurtosis_std` | СКО эксцесса |
| `std_mean` | Среднее стандартное отклонение |

В режиме `mfe_features="full"` дополнительно вычисляются все доступные мета-фичи из выбранных групп `pymfe`.

### Важность мета-фич

По результатам экспериментов на мета-классификаторе, наибольшее влияние на предсказание лучшей модели оказывают:

1. **`nr_num`** — количество числовых признаков
2. **`missing_pct`** — доля пропущенных значений
3. **`task_type`** — тип задачи

Это означает: числовые датасеты без пропусков и датасеты с большой долей пропусков ведут себя по-разному, и LLM это учитывает.

---

## Zero-shot vs Few-shot

### Zero-shot

Промпт содержит только:
- Мета-характеристики вашего датасета
- Описания 6 поддерживаемых моделей
- Целевую метрику

LLM отвечает на основе общих знаний об архитектурах и их свойствах.

```python
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="zero")
```

### Few-shot

Дополнительно к zero-shot промпту добавляются результаты предварительных экспериментов на 5 референсных датасетах:

- **abalone** (регрессия, ~4177 строк)
- **cl-housing** (регрессия, недвижимость)
- **air-quality** (регрессия, качество воздуха)
- **wind** (регрессия, скорость ветра)
- **gats** (регрессия)

LLM видит: на похожем датасете модель X показала такие-то результаты по такой-то метрике — это конкретный опыт, на который можно опираться.

```python
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="few")
```

!!! tip "Когда что использовать"
    - **Few-shot** — всегда предпочтительнее: даёт top1-acc до 0.84 vs 0.6 для zero-shot (по экспериментам)
    - **Zero-shot** — если хотите получить быстрый ответ без overhead'а чтения экспериментальных данных

---

## LLMRunner

`LLMRunner` отправляет промпт в LLM и агрегирует результаты нескольких запусков.

### Создание и запуск

```python
from tab_forge.llm_runner import LLMRunner

runner = LLMRunner(
    base_url        = "https://api.openai.com/v1",
    api_key         = "sk-...",
    model           = "gpt-4o",
    retry_on_parse_error = True,   # повторить если LLM не дал корректный ответ
    request_delay   = 0.5,         # пауза между запросами в секундах
)

result = runner.run(
    prompt      = prompt,
    n_runs      = 5,        # количество независимых запросов
    temperature = 0.7,
    top_p       = 1.0,
    max_tokens  = 2048,
)
```

### Параметры `run`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_runs` | `1` | Количество независимых запросов (self-consistency) |
| `temperature` | `0.7` | Температура генерации |
| `top_p` | `1.0` | Top-p sampling |
| `max_tokens` | `2048` | Максимальная длина ответа |
| `seed` | `None` | Фиксация seed (если поддерживается моделью) |

---

## Self-consistency: как это работает

Self-consistency — ключевая техника, делающая LLM-выбор надёжным.

```
Запрос 1 → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']
Запрос 2 → ['CTGAN', 'GAN-MFS', 'WGAN-GP', 'TVAE', 'CTABGAN+', 'DDPM']
Запрос 3 → ['GAN-MFS', 'WGAN-GP', 'CTGAN', 'CTABGAN+', 'DDPM', 'TVAE']
Запрос 4 → ['GAN-MFS', 'CTGAN', 'CTABGAN+', 'WGAN-GP', 'TVAE', 'DDPM']
Запрос 5 → ['CTGAN', 'GAN-MFS', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

Средние ранги:
  GAN-MFS:  (1+2+1+1+2)/5 = 1.4
  CTGAN:    (2+1+3+2+1)/5 = 1.8
  WGAN-GP:  (3+3+2+4+3)/5 = 3.0
  ...

Итоговый рейтинг: ['GAN-MFS', 'CTGAN', 'WGAN-GP', ...]
```

LLM ведёт себя стохастически при `temperature > 0`. 5 независимых запросов с усреднением рангов дают статистически устойчивый результат.

---

## RunnerResult

```python
result = runner.run(prompt, n_runs=5)

print(result.final_ranking)
# ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

print(result.average_ranks)
# {'GAN-MFS': 1.4, 'CTGAN': 1.8, ...}

print(result.n_runs)    # 5
print(result.n_parsed)  # сколько из 5 ответов LLM успешно распарсилось
```

!!! warning "Ошибки парсинга"
    Если `n_parsed < n_runs`, значит LLM не дала валидный ответ в нужном формате для части запросов. Включите `retry_on_parse_error=True` чтобы повторять запросы при ошибке парсинга.

---

## Подключение локальной LLM

Tab-Forge совместим с любым OpenAI-совместимым API:

=== "Ollama"

    ```python
    runner = LLMRunner(
        base_url = "http://localhost:11434/v1",
        api_key  = "ollama",          # любая строка
        model    = "llama3.1:8b",
    )
    ```

=== "LM Studio"

    ```python
    runner = LLMRunner(
        base_url = "http://localhost:1234/v1",
        api_key  = "lm-studio",
        model    = "local-model",
    )
    ```

=== "OpenAI"

    ```python
    runner = LLMRunner(
        base_url = "https://api.openai.com/v1",
        api_key  = "sk-...",
        model    = "gpt-4o",
    )
    ```

!!! note "Качество моделей"
    Качество рейтинга зависит от способности LLM следовать инструкциям и рассуждать о табличных данных. GPT-4o и аналогичные модели показывают лучшие результаты. Небольшие локальные модели могут давать менее стабильные рейтинги — увеличьте `n_runs` для компенсации.
