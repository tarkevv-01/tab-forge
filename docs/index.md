# Tab-Forge

**Модульная Python-библиотека для автоматизированной настройки генеративных моделей на табличных данных**

---

Tab-Forge решает реальную задачу: выбор и настройка генеративной модели для синтеза табличных данных — дело трудоёмкое. Нужно понять, какую из шести совершенно разных архитектур (GAN, VAE, Diffusion) попробовать первой, какие метрики качества важны для вашей задачи, и сколько времени тратить на тюнинг.

Tab-Forge автоматизирует весь этот процесс: LLM анализирует мета-характеристики вашего датасета и предсказывает, какие модели стоит попробовать в первую очередь — а дальше Optuna-тюнинг делает остальное.

---

## Для кого эта библиотека

- **Data scientists**, которым нужны синтетические данные для аугментации, балансировки классов или тестирования
- **ML-инженеры**, работающие с приватными или ограниченными датасетами
- **Исследователи**, изучающие генеративные модели для табличных данных
- Все, кто устал вручную подбирать гиперпараметры GAN

---

## Ключевые преимущества

| Возможность | Детали |
|-------------|--------|
| **6 архитектур из коробки** | CTGAN, WGAN-GP, GAN-MFS, CTABGAN+, TVAE, TabDDPM — единый интерфейс |
| **LLM-ассистированный выбор** | PromptGenerator + LLMRunner ранжируют модели по мета-фичам вашего датасета |
| **Self-consistency** | N независимых запросов к LLM, агрегация через голосование — устойчивый результат |
| **Байесовский тюнинг** | AutoTuningStudy через Optuna с k-fold CV; поддержка extended и manual режимов |
| **5 метрик качества** | R², RMSE, Jensen–Shannon, Frobenius Correlation, Frobenius MI |
| **Модульность** | Каждый компонент можно использовать отдельно |

---

## Быстрый пример

```python
from tab_forge.dataset import Dataset
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner
from tab_forge.tuning import AutoTuningStudy

# 1. Загружаем данные
dataset = Dataset(
    data="abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

# 2. LLM выбирает лучшую модель для этого датасета
prompt = PromptGenerator().build_prompt(
    dataset=dataset,
    target_metric="r2_metric",
    shot_mode="few",
)
result = LLMRunner(base_url="https://api.openai.com/v1",
                   api_key="sk-...", model="gpt-4o").run(prompt, n_runs=5)

print("Рейтинг:", result.final_ranking)
# → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

# 3. Тюним лучшую модель
study = AutoTuningStudy(
    model_class=result.final_ranking[0],  # строка из рейтинга LLM
    search_space_mode="extended",
    cv=3,
)
study.optimize(dataset, n_trials=25)
print("Лучшие параметры:", study.best_params)
```

---

## Архитектура пайплайна

```
Ваш датасет
     │
     ▼
PromptGenerator  ──── мета-фичи + опыт референсных датасетов
     │
     ▼
LLMRunner  ──── self-consistency (N запросов → усреднение рангов)
     │
     ▼
AutoTuningStudy  ──── Optuna + k-fold CV
     │
     ├──► Models  (CTGAN / WGAN-GP / GAN-MFS / CTABGAN+ / TVAE / DDPM)
     │
     └──► Benchmark  (R² / RMSE / JS / FrobCorr / FrobMI)
```

---

## Разделы документации

<div class="grid cards" markdown>

- **[Установка](getting-started/installation.md)**
  Как поставить библиотеку и зависимости

- **[Быстрый старт](getting-started/quickstart.md)**
  Полный пайплайн за 5 минут

- **[Метрики качества](concepts/metrics.md)**
  Что измеряет каждая метрика и когда её использовать

- **[Архитектура](concepts/architecture.md)**
  Как устроены модули и как они взаимодействуют

- **[Руководство пользователя](user-guide/dataset.md)**
  Подробное описание каждого модуля

- **[Эксперименты](experiments/tuning-results.md)**
  Результаты исследований и выводы

</div>
