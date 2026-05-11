# Models Module

## Why You Need This

The six supported architectures are implemented in very different libraries: SDV, Synthcity, PyTorch with custom implementations. Each provides its own API. Tab-Forge wraps all of them with a unified `BaseGenerativeModel` interface with three key methods: `fit`, `generate`, `structed_generate`.

---

## Unified Interface

```python
# All models work the same way
model.fit(dataset)                          # train
synth_df  = model.generate(n_samples=500)  # → pd.DataFrame
synth_ds  = model.structed_generate(500)   # → Dataset (for Benchmark / AutoTuning)
losses    = model.get_losses()             # training loss history
```

---

## All 6 Models

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

**When to use:** CTGAN is a reliable general-purpose choice. Implemented via SDV and works well on mixed data (numerical + categorical). Especially effective when data has complex conditional distributions by categorical features.

!!! tip "Optimal metrics for CTGAN"
    Experiments show CTGAN produced stable results for **R²** (5 improvements) and **RMSE** (6 improvements). A good general choice for regression tasks.

---

### WGAN-GP — Wasserstein GAN + Gradient Penalty

```python
from tab_forge.models import WGANGPSynthesizer

model = WGANGPSynthesizer(
    epochs          = 300,
    batch_size      = 256,
    generator_lr    = 1e-4,
    discriminator_lr= 1e-4,
    n_critic        = 5,       # discriminator steps per generator step
    lambda_gp       = 10,      # gradient penalty coefficient
)
model.fit(dataset)
```

**When to use:** WGAN-GP is theoretically more stable in training than standard GAN — gradient penalty prevents mode collapse. Custom PyTorch implementation inside Tab-Forge.

!!! tip "Optimal metrics for WGAN-GP"
    Record results for **JS divergence** (12 improvements out of 25!) — this makes sense, as Wasserstein distance is directly related to distribution divergence. Also good results for **RMSE** (6 improvements).

---

### GAN-MFS — GAN with Meta-Feature Similarity

```python
from tab_forge.models import GANMFSSynthesizer

model = GANMFSSynthesizer(
    epochs       = 300,
    batch_size   = 256,
    generator_lr = 1e-4,
    mfs_lambda   = 0.1,    # MFS regularizer weight in loss function
    subset_mfs   = 10,     # subsample size for MFS
    sample_frac  = 0.1,    # fraction of data for MFS computation
)
model.fit(dataset)
```

**When to use:** GAN-MFS extends WGAN-GP with an additional term in the loss function — Wasserstein distance on meta-feature (data statistics) distributions. This makes generated data statistically similar to real data.

!!! tip "GAN-MFS — the number one choice for R²"
    In experiments, GAN-MFS showed **the best R² results** among all models (8 improvements out of 25). If your goal is to train an ML model on synthetic data, start with GAN-MFS.

---

### CTABGAN+ — CTABGAN with Auxiliary Heads

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

**When to use:** CTABGAN+ adds auxiliary classification/regression heads in the discriminator to standard CTGAN. This forces the generator to account for the target variable and preserve its distribution.

!!! tip "Optimal metrics for CTABGAN+"
    Outstanding results for **RMSE** (11 improvements — the best result among all models!). A great choice when prediction accuracy matters.

!!! note "Task type"
    CTABGAN+ automatically determines the auxiliary task type from `dataset.info.task_type`. Make sure to specify the correct task type when creating the `Dataset`.

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

**When to use:** TVAE is a VAE-based architecture from SDV. Unlike GAN, VAE trains more stably (no generator-discriminator competition) and works well on datasets with missing values.

!!! tip "When to choose TVAE"
    If GAN training is unstable or data is limited — try TVAE. The model handles tasks where **RMSE** is important well (8 improvements).

---

### TabDDPM — Diffusion Model

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

**When to use:** TabDDPM is a modern diffusion architecture (Synthcity). It captures complex multimodal distributions and nonlinear dependencies well. Usually requires more training time.

!!! tip "Optimal metrics for TabDDPM"
    Good results for **R²** (7 improvements) and **MI** (6 improvements). Recommended for datasets with nonlinear dependencies.

!!! warning "Training time"
    Diffusion models train slower than GAN. With limited resources, reduce `num_timesteps` or `epochs`.

---

## Model Comparison

| Model | Architecture | Best metric | Speed | Stability |
|-------|------------|-------------|-------|----------|
| CTGAN | Conditional GAN (SDV) | R², RMSE | Fast | Medium |
| WGAN-GP | Wasserstein GAN | JS divergence | Medium | High |
| GAN-MFS | WGAN-GP + MFS loss | **R²** ⭐ | Medium | High |
| CTABGAN+ | CTGAN + aux heads | **RMSE** ⭐ | Slow | Low |
| TVAE | Variational AE (SDV) | RMSE | Fast | Very high |
| TabDDPM | Diffusion (Synthcity) | R², MI | Slow | High |

---

## Default Hyperparameters

!!! note "Where do defaults come from?"
    Default hyperparameter values are taken from the original papers and repositories of each architecture. When using `AutoTuningStudy` with `"extended"` mode, they serve as the starting point for the search.

Detailed hyperparameter ranges for tuning are described in the [Tuning](./tuning.md) section and in the [API Reference](../api/models.md).
