# Tab-Forge

**Модульная Python-библиотека для автоматизированного тюнинга GAN на табличных данных**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Описание

Tab-Forge — это комплексное решение для работы с генеративно-состязательными сетями (GAN) на табличных данных. Библиотека предоставляет модульную архитектуру, объединяющую загрузку данных, обучение моделей, оценку качества и автоматический тюнинг гиперпараметров в единый пайплайн.

## 🎯 Ключевые возможности

### 1. **Dataset** — Работа с данными
- Унифицированная загрузка и препроцессинг табличных данных
- Поддержка числовых и категориальных признаков
- Автоматическое разбиение на обучающую, валидационную и тестовую выборки

### 2. **Models** — GAN архитектуры
- Единый интерфейс для различных архитектур GAN
- Поддержка моделей: **CTGAN**, **WGAN-GP**, **GAN-MFS**, **CTABGAN+**
- Простая интеграция новых архитектур

### 3. **Benchmark** — Оценка качества
- Вычисление метрик качества синтетических данных
- Поддержка множества метахарактеристик качества
- Сравнение результатов между конфигурациями и датасетами

### 4. **Tuning** — Оптимизация
- Автоматический и полуавтоматический тюнинг гиперпараметров
- Интеграция с **Optuna** для эффективного поиска
- Выбор оптимальной конфигурации на основе результатов экспериментов

## 🏗️ Архитектура

```
┌─────────────┐
│   Tuning    │ ← Оркестрация
└──────┬──────┘
       │
   ┌───┴────────────────┐
   ↓                    ↓
┌─────────┐      ┌──────────────┐
│ Models  │      │  Benchmark   │
│ GAN     │      │  Оценка      │
└────┬────┘      └──────┬───────┘
     │                  │
     │  обучение и      │  сравнение
     │  синтез          │
     ↓                  ↓
  ┌──────────────────────┐
  │      Dataset         │
  │  Данные и метаданные │
  └──────────────────────┘
```

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

### Импорт модулей

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer
from tab_forge.tuning import AutoTuningStudy
```

### Инициализация данных

```python
dataset = Dataset(
    data="abalone.csv",
    target="Rings",
    task_type="regression",
    categorical_features=["Sex"],
    numerical_features=["Length", "Diameter", "Height",
                       "Whole weight", "Shucked weight"]
)
```

### Запуск Optuna Tuning

```python
study_extended = AutoTuningStudy(
    model_class=CTGANSynthesizer,
    get_params=None,  # настраиваемые рамки параметров
    benchmark=None,   # настраиваемый параметр оценки
    search_space_mode='extended'
)
```

### Финальный результат

```python
study_extended.optimize(dataset, n_trials=25)

model = study_extended.best_model
synth_data = model.generate(100)
```

## 📊 Применение

Tab-Forge решает следующие задачи:
- **Дополнение данных** (data augmentation) для улучшения ML-моделей
- **Приватность данных** через генерацию синтетических датасетов
- **Балансировка классов** в несбалансированных выборках
- **Исследование данных** и тестирование гипотез

## 🔬 Поддерживаемые модели

| Модель | Описание |
|--------|----------|
| **CTGAN** | Conditional Tabular GAN с mode-specific normalization |
| **WGAN-GP** | Wasserstein GAN с gradient penalty |
| **GAN-MFS** | GAN с multi-frequency sampling |
| **CTABGAN+** | Улучшенная версия CTABGAN с дополнительными механизмами |

## 📈 Метрики качества

- **R²** — доля объясненной дисперсии реальных данных
- **RMSE** — среднеквадратичная ошибка реконструкции
- **Jensen–Shannon divergence** — симметричная мера расхождения распределений
- **Frobenius correlation** — норма разности корреляционных матриц
- **Frobenius mutual information** — сохранение нелинейных зависимостей


## 👥 Авторы

**ИТМО Университет**

GitHub: [tarkevv-01](https://github.com/tarkevv-01)


---

**⭐ Если проект оказался полезным, поставьте звезду на GitHub!**
