# API Reference — Models

## Импорт

```python
from tab_forge.models import (
    CTGANSynthesizer,
    WGANGPSynthesizer,
    GANMFSSynthesizer,
    CTABGANPlusSynthesizer,
    TVAESynthesizer,
    DDPMSynthesizer,
)
```

---

## `BaseGenerativeModel` (ABC)

Абстрактный базовый класс для всех синтезаторов. Определяет единый интерфейс.

### Общие методы всех моделей

#### `fit`

```
fit(dataset: Dataset, **kwargs) -> None
```

Обучает модель на переданном датасете.

#### `generate`

```
generate(n_samples: int) -> pd.DataFrame
```

Генерирует `n_samples` строк синтетических данных. Возвращает `pd.DataFrame`.

!!! warning ""
    Модель должна быть обучена (`fit`) перед вызовом `generate`. Иначе поднимается исключение.

#### `structed_generate`

```
structed_generate(n_samples: int) -> Dataset
```

Аналог `generate`, но возвращает объект `Dataset` с сохранёнными метаданными (типы признаков, целевая переменная). Используйте этот метод когда результат нужно передать в `Benchmark` или `AutoTuningStudy`.

#### `get_losses`

```
get_losses() -> dict
```

Возвращает словарь с историей функции потерь за время обучения.

#### `set_hyperparameters` / `get_hyperparameters`

```
set_hyperparameters(**kwargs) -> None
get_hyperparameters() -> dict
```

Установка и получение текущих гиперпараметров модели.

---

## `CTGANSynthesizer`

```
CTGANSynthesizer(
    epochs: int = 300,
    batch_size: int = 500,
    embedding_dim: int = 128,
    generator_dim: tuple = (256, 256),
    discriminator_dim: tuple = (256, 256),
    generator_lr: float = 2e-4,
    generator_decay: float = 1e-6,
    discriminator_lr: float = 2e-4,
    discriminator_decay: float = 1e-6,
    discriminator_steps: int = 1,
    log_frequency: bool = True,
    pac: int = 10,
    verbose: bool = False,
)
```

Условный табличный GAN на основе реализации SDV. Поддерживает смешанные данные с числовыми и категориальными признаками.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `epochs` | `300` | Число эпох обучения |
| `batch_size` | `500` | Размер мини-батча |
| `embedding_dim` | `128` | Размерность embedding для категориальных признаков |
| `generator_dim` | `(256, 256)` | Размерности слоёв генератора |
| `discriminator_dim` | `(256, 256)` | Размерности слоёв дискриминатора |
| `generator_lr` | `2e-4` | Learning rate генератора |
| `discriminator_lr` | `2e-4` | Learning rate дискриминатора |
| `pac` | `10` | Размер pac для PacGAN (объединение сэмплов в дискриминаторе) |

!!! example ""

    ```python
    model = CTGANSynthesizer(epochs=300, batch_size=500)
    model.fit(dataset)
    synth = model.structed_generate(1000)
    ```

---

## `WGANGPSynthesizer`

```
WGANGPSynthesizer(
    epochs: int = 300,
    batch_size: int = 256,
    generator_lr: float = 1e-4,
    discriminator_lr: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    generator_dim: tuple = (256, 256),
    discriminator_dim: tuple = (256, 256),
)
```

Wasserstein GAN с Gradient Penalty. Собственная PyTorch-реализация. Теоретически более стабилен чем стандартный GAN.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_critic` | `5` | Шагов дискриминатора на один шаг генератора |
| `lambda_gp` | `10.0` | Коэффициент gradient penalty |

!!! example ""

    ```python
    model = WGANGPSynthesizer(epochs=300, n_critic=5)
    model.fit(dataset)
    ```

---

## `GANMFSSynthesizer`

```
GANMFSSynthesizer(
    epochs: int = 300,
    batch_size: int = 256,
    generator_lr: float = 1e-4,
    discriminator_lr: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    mfs_lambda: float = 0.1,
    subset_mfs: int = 10,
    sample_number: int = 100,
    sample_frac: float = 0.1,
)
```

Расширение WGAN-GP с дополнительным регуляризатором Meta-Feature Similarity (MFS). Регуляризатор добавляет слагаемое в функцию потерь: расстояние Wasserstein между распределениями мета-фич реальных и генерируемых данных.

**Дополнительные параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `mfs_lambda` | `0.1` | Вес MFS-регуляризатора |
| `subset_mfs` | `10` | Размер подвыборки для вычисления мета-фич |
| `sample_frac` | `0.1` | Доля данных для вычисления MFS на каждой итерации |

!!! tip ""
    Наилучшие результаты по R² среди всех моделей в экспериментах. Рекомендуется как первый выбор для задач аугментации данных.

!!! example ""

    ```python
    model = GANMFSSynthesizer(epochs=300, mfs_lambda=0.1)
    model.fit(dataset)
    ```

---

## `CTABGANPlusSynthesizer`

```
CTABGANPlusSynthesizer(
    epochs: int = 150,
    batch_size: int = 500,
    lr: float = 2e-4,
    random_dim: int = 100,
    critic_iterations: int = 1,
    class_dim: tuple = (256, 256, 256, 256),
    l2scale: float = 1e-5,
)
```

Обёртка над CTAB-GAN+ — расширением CTABGAN с вспомогательными регрессионными/классификационными головами в дискриминаторе. Тип головы определяется из `dataset.info.task_type`.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `random_dim` | `100` | Размерность латентного пространства |
| `critic_iterations` | `1` | Шагов дискриминатора на генераторный шаг |
| `class_dim` | `(256, 256, 256, 256)` | Архитектура вспомогательных голов |
| `l2scale` | `1e-5` | L2-регуляризация |

!!! tip ""
    Лучшие результаты по RMSE среди всех моделей (11/25). Отличный выбор когда важна точность предсказания.

!!! example ""

    ```python
    model = CTABGANPlusSynthesizer(epochs=150, batch_size=500)
    model.fit(dataset)
    ```

---

## `TVAESynthesizer`

```
TVAESynthesizer(
    epochs: int = 300,
    batch_size: int = 500,
    embedding_dim: int = 128,
    compress_dims: tuple = (128, 128),
    decompress_dims: tuple = (128, 128),
    l2scale: float = 1e-5,
)
```

Табличный Variational Autoencoder от SDV. Использует адаптированный ELBO как функцию потерь. Обучается стабильнее GAN-архитектур.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `compress_dims` | `(128, 128)` | Размерности слоёв энкодера |
| `decompress_dims` | `(128, 128)` | Размерности слоёв декодера |
| `l2scale` | `1e-5` | L2-регуляризация |

!!! example ""

    ```python
    model = TVAESynthesizer(epochs=300, embedding_dim=128)
    model.fit(dataset)
    ```

---

## `DDPMSynthesizer`

```
DDPMSynthesizer(
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_timesteps: int = 1000,
    model_type: str = "mlp",
    scheduler: str = "cosine",
    **model_params,
)
```

Диффузионная модель Tab-DDPM, реализованная через плагин `ddpm` библиотеки Synthcity.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `num_timesteps` | `1000` | Число шагов диффузии |
| `model_type` | `"mlp"` | Тип нейросети шумоподавителя |
| `scheduler` | `"cosine"` | Расписание шумового процесса (`"cosine"` / `"linear"`) |
| `**model_params` | — | Дополнительные параметры архитектуры для плагина Synthcity |

!!! warning "Время обучения"
    TabDDPM обучается значительно медленнее GAN-архитектур. Начните с `epochs=50` и `num_timesteps=500` для быстрой проверки.

!!! example ""

    ```python
    model = DDPMSynthesizer(epochs=100, num_timesteps=1000)
    model.fit(dataset)
    ```
