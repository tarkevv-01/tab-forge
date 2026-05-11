# API Reference — Models

## Import

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

Abstract base class for all synthesizers. Defines a unified interface.

### Common Methods for All Models

#### `fit`

```
fit(dataset: Dataset, **kwargs) -> None
```

Trains the model on the provided dataset.

#### `generate`

```
generate(n_samples: int) -> pd.DataFrame
```

Generates `n_samples` rows of synthetic data. Returns `pd.DataFrame`.

!!! warning ""
    The model must be trained (`fit`) before calling `generate`. Otherwise an exception is raised.

#### `structed_generate`

```
structed_generate(n_samples: int) -> Dataset
```

Analogous to `generate`, but returns a `Dataset` object with preserved metadata (feature types, target variable). Use this method when the result needs to be passed to `Benchmark` or `AutoTuningStudy`.

#### `get_losses`

```
get_losses() -> dict
```

Returns a dictionary with the training loss history.

#### `set_hyperparameters` / `get_hyperparameters`

```
set_hyperparameters(**kwargs) -> None
get_hyperparameters() -> dict
```

Set and retrieve the current model hyperparameters.

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

Conditional tabular GAN based on the SDV implementation. Supports mixed data with numerical and categorical features.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `300` | Number of training epochs |
| `batch_size` | `500` | Mini-batch size |
| `embedding_dim` | `128` | Embedding dimensionality for categorical features |
| `generator_dim` | `(256, 256)` | Generator layer sizes |
| `discriminator_dim` | `(256, 256)` | Discriminator layer sizes |
| `generator_lr` | `2e-4` | Generator learning rate |
| `discriminator_lr` | `2e-4` | Discriminator learning rate |
| `pac` | `10` | PacGAN pac size (sample grouping in discriminator) |

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

Wasserstein GAN with Gradient Penalty. Custom PyTorch implementation. Theoretically more stable than standard GAN.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_critic` | `5` | Discriminator steps per generator step |
| `lambda_gp` | `10.0` | Gradient penalty coefficient |

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

Extension of WGAN-GP with an additional Meta-Feature Similarity (MFS) regularizer. The regularizer adds a term to the loss function: Wasserstein distance between meta-feature distributions of real and generated data.

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mfs_lambda` | `0.1` | MFS regularizer weight |
| `subset_mfs` | `10` | Subsample size for meta-feature computation |
| `sample_frac` | `0.1` | Fraction of data used for MFS computation each iteration |

!!! tip ""
    Best R² results among all models in experiments. Recommended as the first choice for data augmentation tasks.

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

Wrapper around CTAB-GAN+ — an extension of CTABGAN with auxiliary regression/classification heads in the discriminator. The head type is determined from `dataset.info.task_type`.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_dim` | `100` | Latent space dimensionality |
| `critic_iterations` | `1` | Discriminator steps per generator step |
| `class_dim` | `(256, 256, 256, 256)` | Auxiliary head architecture |
| `l2scale` | `1e-5` | L2 regularization |

!!! tip ""
    Best RMSE results among all models (11/25). Excellent choice when prediction accuracy matters.

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

Tabular Variational Autoencoder from SDV. Uses adapted ELBO as the loss function. Trains more stably than GAN architectures.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compress_dims` | `(128, 128)` | Encoder layer sizes |
| `decompress_dims` | `(128, 128)` | Decoder layer sizes |
| `l2scale` | `1e-5` | L2 regularization |

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

Tab-DDPM diffusion model, implemented via the `ddpm` plugin of the Synthcity library.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_timesteps` | `1000` | Number of diffusion steps |
| `model_type` | `"mlp"` | Type of denoiser neural network |
| `scheduler` | `"cosine"` | Noise schedule (`"cosine"` / `"linear"`) |
| `**model_params` | — | Additional architecture parameters for the Synthcity plugin |

!!! warning "Training time"
    TabDDPM trains significantly slower than GAN architectures. Start with `epochs=50` and `num_timesteps=500` for a quick check.

!!! example ""

    ```python
    model = DDPMSynthesizer(epochs=100, num_timesteps=1000)
    model.fit(dataset)
    ```
