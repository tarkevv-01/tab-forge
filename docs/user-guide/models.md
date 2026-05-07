# Модуль Models

## Зачем это нужно

Шесть поддерживаемых архитектур реализованы в очень разных библиотеках: SDV, Synthcity, PyTorch с кастомными реализациями. Каждая предоставляет свой API. Tab-Forge накрывает всех единым интерфейсом `BaseGenerativeModel` с тремя ключевыми методами: `fit`, `generate`, `structed_generate`.

---

## Единый интерфейс

```python
# Все модели работают одинаково
model.fit(dataset)                          # обучение
synth_df  = model.generate(n_samples=500)  # → pd.DataFrame
synth_ds  = model.structed_generate(500)   # → Dataset (для Benchmark / AutoTuning)
losses    = model.get_losses()             # история функции потерь
```

---

## Все 6 моделей

### CTGAN — Conditional Tabular GAN

```python
from tab_forge.models import CTGANSynthesizer

model = CTGANSynthesizer(
    epochs          = 300,
    batch_size      = 500,
    embedding_dim   = 128,
    generator_lr    = 2e-4,
    discriminator_lr= 2e-4,
)
model.fit(dataset)
synth = model.structed_generate(1000)
```

**Когда использовать:** CTGAN — надёжный универсальный выбор. Реализован через SDV и хорошо работает на смешанных данных (числовые + категориальные). Особенно эффективен когда в данных есть сложные условные распределения по категориальным признакам.

!!! tip "Оптимальные метрики для CTGAN"
    По экспериментам CTGAN показал стабильные результаты по **R²** (5 улучшений) и **RMSE** (6 улучшений). Хороший общий выбор для задач регрессии.

---

### WGAN-GP — Wasserstein GAN + Gradient Penalty

```python
from tab_forge.models import WGANGPSynthesizer

model = WGANGPSynthesizer(
    epochs          = 300,
    batch_size      = 256,
    generator_lr    = 1e-4,
    discriminator_lr= 1e-4,
    n_critic        = 5,       # шагов дискриминатора на шаг генератора
    lambda_gp       = 10,      # коэффициент gradient penalty
)
model.fit(dataset)
```

**Когда использовать:** WGAN-GP теоретически более стабилен в обучении, чем обычный GAN — gradient penalty предотвращает mode collapse. Собственная реализация на PyTorch внутри Tab-Forge.

!!! tip "Оптимальные метрики для WGAN-GP"
    Рекордные результаты по **JS-дивергенции** (12 улучшений из 25!) — это логично, ведь Wasserstein-расстояние напрямую связано с расхождением распределений. Также хорошие результаты по **RMSE** (6 улучшений).

---

### GAN-MFS — GAN с Meta-Feature Similarity

```python
from tab_forge.models import GANMFSSynthesizer

model = GANMFSSynthesizer(
    epochs       = 300,
    batch_size   = 256,
    generator_lr = 1e-4,
    mfs_lambda   = 0.1,    # вес MFS-регуляризатора в функции потерь
    subset_mfs   = 10,     # размер подвыборки для MFS
    sample_frac  = 0.1,    # доля данных для вычисления MFS
)
model.fit(dataset)
```

**Когда использовать:** GAN-MFS расширяет WGAN-GP дополнительным слагаемым в функции потерь — расстоянием Wasserstein на распределениях мета-фич (статистик данных). Это делает генерируемые данные статистически похожими на реальные.

!!! tip "GAN-MFS — выбор номер один для R²"
    В экспериментах GAN-MFS показал **наилучшие результаты по R²** среди всех моделей (8 улучшений из 25). Если ваша задача — обучить ML-модель на синтетике, начните с GAN-MFS.

---

### CTABGAN+ — CTABGAN с вспомогательными головами

```python
from tab_forge.models import CTABGANPlusSynthesizer

model = CTABGANPlusSynthesizer(
    epochs            = 150,
    batch_size        = 500,
    lr                = 2e-4,
    random_dim        = 100,
    critic_iterations = 1,
    class_dim         = (256, 256, 256, 256),
    l2scale           = 1e-5,
)
model.fit(dataset)
```

**Когда использовать:** CTABGAN+ добавляет к обычному CTGAN вспомогательные классификационные/регрессионные головы в дискриминаторе. Это принуждает генератор учитывать целевую переменную и сохранять её распределение.

!!! tip "Оптимальные метрики для CTABGAN+"
    Показал выдающиеся результаты по **RMSE** (11 улучшений — лучший результат среди всех моделей!). Хороший выбор когда важна точность предсказания.

!!! note "Тип задачи"
    CTABGAN+ автоматически определяет тип вспомогательной задачи из `dataset.info.task_type`. Укажите правильный тип задачи при создании `Dataset`.

---

### TVAE — Tabular Variational Autoencoder

```python
from tab_forge.models import TVAESynthesizer

model = TVAESynthesizer(
    epochs      = 300,
    batch_size  = 500,
    embedding_dim = 128,
    compress_dims = (128, 128),
    decompress_dims = (128, 128),
    l2scale     = 1e-5,
)
model.fit(dataset)
```

**Когда использовать:** TVAE — VAE-based архитектура от SDV. В отличие от GAN, VAE обучается стабильнее (нет соревнования генератора и дискриминатора) и хорошо работает на датасетах с пропущенными значениями.

!!! tip "Когда выбирать TVAE"
    Если GAN-обучение нестабильно или данных мало — попробуйте TVAE. Модель хорошо справляется с задачами где важна **RMSE** (8 улучшений).

---

### TabDDPM — Диффузионная модель

```python
from tab_forge.models import DDPMSynthesizer

model = DDPMSynthesizer(
    epochs         = 100,
    batch_size     = 256,
    lr             = 1e-3,
    num_timesteps  = 1000,
    model_type     = "mlp",
    scheduler      = "cosine",
)
model.fit(dataset)
```

**Когда использовать:** TabDDPM — современная диффузионная архитектура (Synthcity). Хорошо улавливает сложные многомодальные распределения и нелинейные зависимости. Обычно требует больше времени на обучение.

!!! tip "Оптимальные метрики для TabDDPM"
    Хорошие результаты по **R²** (7 улучшений) и **MI** (6 улучшений). Рекомендуется для датасетов с нелинейными зависимостями.

!!! warning "Время обучения"
    Диффузионные модели обучаются медленнее GAN. При ограниченных ресурсах уменьшите `num_timesteps` или `epochs`.

---

## Сравнение моделей

| Модель | Архитектура | Лучшая метрика | Скорость | Стабильность |
|--------|------------|----------------|----------|--------------|
| CTGAN | Conditional GAN (SDV) | R², RMSE | Быстрая | Средняя |
| WGAN-GP | Wasserstein GAN | JS divergence | Средняя | Высокая |
| GAN-MFS | WGAN-GP + MFS loss | **R²** ⭐ | Средняя | Высокая |
| CTABGAN+ | CTGAN + aux heads | **RMSE** ⭐ | Низкая | Низкая |
| TVAE | Variational AE (SDV) | RMSE | Быстрая | Очень высокая |
| TabDDPM | Diffusion (Synthcity) | R², MI | Медленная | Высокая |

---

## Гиперпараметры по умолчанию

!!! note "Откуда берутся дефолты?"
    Дефолтные значения гиперпараметров взяты из оригинальных статей и репозиториев каждой архитектуры. При использовании `AutoTuningStudy` с режимом `"extended"` они служат отправной точкой для поиска.

Подробные диапазоны всех гиперпараметров для тюнинга описаны в разделе [Tuning](./tuning.md) и в [API Reference](../api/models.md).
