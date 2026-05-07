# API Reference — LLM Selection

## Импорт

```python
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner, RunnerResult, SingleRun
```

---

## `PromptGenerator`

```
PromptGenerator(
    experiment_dir: Optional[str] = None,
    datasets_dir: Optional[str] = None,
)
```

Строит структурированный промпт для LLM на основе мета-характеристик датасета.

При инициализации без аргументов использует встроенные директории с экспериментальными данными (`prompt_generator/experiment_results/`) и референсными датасетами (`prompt_generator/datasets/`).

**Параметры:**

- `experiment_dir` — путь к директории с результатами экспериментов (необязательно)
- `datasets_dir` — путь к директории с референсными датасетами (необязательно)

!!! warning ""
    При инициализации проверяется наличие папок `experiment_results/` и `datasets/`. Если они не найдены — поднимается исключение.

---

### `PromptGenerator.build_prompt`

```
build_prompt(
    dataset: Dataset,
    target_metric: str = "r2_metric",
    shot_mode: str = "zero",
    mfe_features: Union[str, List[str]] = "short",
    mfe_groups: Optional[List[str]] = None,
    models_to_include: Optional[List[str]] = None,
) -> str
```

Формирует текстовый промпт для LLM.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `dataset` | `Dataset` | — | Датасет для анализа |
| `target_metric` | `str` | `"r2_metric"` | Целевая метрика для ранжирования |
| `shot_mode` | `str` | `"zero"` | `"zero"` или `"few"` |
| `mfe_features` | `str` или `list` | `"short"` | Набор мета-фич: `"short"`, `"full"` или список строк |
| `mfe_groups` | `list[str]` | `None` | Группы pymfe (например, `["statistical", "info-theory"]`) |
| `models_to_include` | `list[str]` | все 6 | Ограничить набор моделей в промпте |

**Возвращает:** строку-промпт готовую для отправки в LLM.

**Допустимые значения `target_metric`:**

| Значение | Метрика |
|----------|---------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon Divergence |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

!!! example ""

    ```python
    gen = PromptGenerator()
    prompt = gen.build_prompt(
        dataset       = dataset,
        target_metric = "r2_metric",
        shot_mode     = "few",
        mfe_features  = "short",
    )
    print(f"Длина промпта: {len(prompt)} символов")
    ```

---

## `LLMRunner`

```
LLMRunner(
    base_url: str,
    api_key: str,
    model: str,
    retry_on_parse_error: bool = False,
    request_delay: float = 0.0,
)
```

Отправляет промпт в OpenAI-совместимый API и агрегирует результаты нескольких запросов (self-consistency).

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `base_url` | `str` | — | URL API endpoint (`"https://api.openai.com/v1"`) |
| `api_key` | `str` | — | API ключ |
| `model` | `str` | — | Имя модели (`"gpt-4o"`, `"llama3.1:8b"`, и т.д.) |
| `retry_on_parse_error` | `bool` | `False` | Повторять запрос если LLM дал непарсируемый ответ |
| `request_delay` | `float` | `0.0` | Задержка между запросами (секунды) |

---

### `LLMRunner.run`

```
run(
    prompt: str,
    n_runs: int = 1,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    seed: Optional[int] = None,
) -> RunnerResult
```

Выполняет `n_runs` независимых запросов к LLM и агрегирует рейтинги.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `prompt` | — | Текст промпта |
| `n_runs` | `1` | Количество независимых запросов (self-consistency) |
| `temperature` | `0.7` | Температура генерации (0 = детерминировано) |
| `top_p` | `1.0` | Top-p (nucleus) sampling |
| `max_tokens` | `2048` | Максимальное число токенов в ответе |
| `seed` | `None` | Фиксация seed для воспроизводимости (если LLM поддерживает) |

**Возвращает:** `RunnerResult`

!!! example ""

    ```python
    runner = LLMRunner(
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o",
    )
    result = runner.run(prompt, n_runs=5, temperature=0.7)
    print(result.final_ranking)
    ```

---

## `RunnerResult`

Датакласс с результатами всех запусков.

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `runs` | `List[SingleRun]` | Детали каждого запуска |
| `average_ranks` | `dict` | Средний ранг каждой модели |
| `final_ranking` | `List[str]` | Список моделей, упорядоченный по среднему рангу |
| `n_runs` | `int` | Запрошенное число запусков |
| `n_parsed` | `int` | Успешно распарсенных ответов |
| `model` | `str` | Имя использованной LLM |
| `prompt_chars` | `int` | Длина промпта в символах |

!!! example ""

    ```python
    print(result.final_ranking)
    # ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

    print(result.average_ranks)
    # {'GAN-MFS': 1.4, 'CTGAN': 1.8, 'WGAN-GP': 3.0, ...}

    print(f"Успешно распарсено: {result.n_parsed}/{result.n_runs}")
    ```

---

## `SingleRun`

Датакласс с деталями одного запуска.

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `run_index` | `int` | Индекс запуска |
| `raw_response` | `str` | Сырой текст ответа LLM |
| `ranking` | `List[str]` | Распарсенный рейтинг моделей |
| `parsed_ok` | `bool` | Успешно ли распарсился ответ |

---

## Вспомогательные функции

### `extract_meta_features`

```python
from tab_forge.prompt_generator import extract_meta_features

features = extract_meta_features(
    df           = dataset.get_data(),
    target_col   = "Rings",
    mfe_groups   = None,
    mfe_features = "short",
)
print(features)
# {'nr_inst': 4177, 'nr_attr': 8, 'nr_num': 7, 'missing_pct': 0.0, ...}
```

Вычисляет мета-характеристики датасета. Использует как ручные вычисления (базовые статистики), так и `pymfe` для расширенных фич.
