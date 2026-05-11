# Tab-Forge — Overview

Tab-Forge is a modular library for automated generation of synthetic tabular data. Its core idea: **before expensive hyperparameter search**, an LLM analyzes the meta-characteristics of your dataset and task, predicts which generative models will perform best, and proposes a prioritized tuning order.

---

## Library Structure

```
tab_forge/
├── dataset/           # Data loading and management
├── models/            # Generative architectures (CTGAN, WGAN-GP, GAN-MFS, CTABGAN+, TVAE, DDPM)
├── benchmark/         # Synthetic data quality metrics
├── prompt_generator/  # LLM prompt assembly + dataset meta-features
├── llm_runner/        # LLM interaction, self-consistency aggregation
└── tuning/            # Hyperparameter tuning (Optuna)
```

---

## Pipeline

```
┌──────────────────────────────────────────┐
│            PromptGenerator               │  ← dataset meta-features +
│                                          │    experience on reference datasets
└───────────────────┬──────────────────────┘
                    │ assembled prompt
                    ↓
┌──────────────────────────────────────────┐
│               LLMRunner                  │  ← model ranking
│       (self-consistency, N runs)         │    via self-consistency
└───────────────────┬──────────────────────┘
                    │ prioritized model list
                    ↓
┌──────────────────────────────────────────┐
│           AutoTuningStudy                │  ← k-fold CV + Optuna
│                                          │    for each model in the ranking
└──────┬───────────────────────┬───────────┘
       │                       │
       ↓                       ↓
┌──────────────┐   ┌───────────────────────┐
│    Models    │   │        Dataset        │
└──────────────┘   └──────────┬────────────┘
                               ↓
                  ┌────────────────────────┐
                  │        Benchmark       │  ← synthetic data quality evaluation
                  └────────────────────────┘
```

Each step can be used independently.

---

## Components

### Dataset

A wrapper around `pd.DataFrame` (or a path to a CSV file). Stores meta-information: the target variable, task type (`regression` / `classification`), and lists of numerical and categorical features.

Provides utilities:
- `train_test_split()` — split into train/test sets
- `split_folds()` — k-fold split (for cross-validation in tuning)
- `merge_datasets()` — merge multiple `Dataset` objects

---

### PromptGenerator

Builds a text prompt for the LLM, including:

- **Dataset meta-characteristics** — statistics computed automatically (some via the `pymfe` library):

  | Meta-feature | Meaning |
  |-------------|---------|
  | `nr_inst` | Number of rows |
  | `nr_attr` | Number of features |
  | `nr_num` / `nr_cat` | Numerical / categorical features |
  | `missing_pct` | Fraction of missing values, % |
  | `task_type` | Task type |
  | `abs_corr_mean` / `abs_corr_max` | Mean / maximum absolute correlation |
  | `skewness_mean` | Mean skewness |
  | `kurtosis_mean` / `kurtosis_std` | Mean / std of kurtosis |
  | `std_mean` | Mean standard deviation |

- **Descriptions of supported models** and the target metric.
- **(few-shot)** Results of preliminary experiments on 5 reference datasets (`abalone`, `cl-housing`, `air`, `wind`, `gats`) — the LLM sees how each model performed on similar tasks.

`mfe_features` parameter:
- `"short"` (default) — a curated set of 12 key meta-features
- `"full"` — all features from the selected `mfe_groups`
- `list[str]` — an arbitrary user-defined set

---

### LLMRunner

Sends the prompt to any OpenAI-compatible API (OpenAI, local server, etc.).

**Self-consistency:** the request is repeated `n_runs` times independently; model ranks from each response are averaged. This reduces the influence of random LLM behavior and makes the final ranking more stable.

Returns a `RunnerResult` object with:
- `average_ranks` — average rank of each model (lower is better)
- `final_ranking` — list of models ordered by average rank

---

### Models

A unified base interface `BaseGenerativeModel` with `fit` / `generate` / `structed_generate` methods. Six implementations are available out of the box:

| Class | Architecture |
|-------|-------------|
| `CTGANSynthesizer` | Conditional Tabular GAN (SDV) |
| `WGANGPSynthesizer` | Wasserstein GAN + Gradient Penalty |
| `GANMFSSynthesizer` | WGAN-GP with MFS regularizer in the loss function |
| `CTABGANPlusSynthesizer` | CTAB-GAN+ with auxiliary regression/classification heads |
| `TVAESynthesizer` | Variational Autoencoder with tabular ELBO adaptation |
| `DDPMSynthesizer` | Diffusion model (Tab-DDPM) |

All six models participate in LLM ranking; tuning is launched for models in the order proposed by the LLM.

---

### Benchmark

Evaluates the quality of synthetic data relative to real data. Accepts a metric specification as a list or dictionary of tuples.

#### Quality Metrics

| Metric | Meaning | Direction |
|--------|---------|-----------|
| **R²** | Fraction of real data variance explained by synthetic data (via ML model) | higher → better |
| **RMSE** | Root mean square prediction error | lower → better |
| **Jensen–Shannon** | Divergence of marginal feature distributions | lower → better |
| **Frobenius Correlation** | Norm of the difference between correlation matrices | lower → better |
| **Frobenius MI** | Preservation of nonlinear dependencies (mutual information matrix) | lower → better |

---

### AutoTuningStudy

Performs automated hyperparameter search via Optuna.

**Workflow:**
1. Data is split into `cv` folds.
2. For each hyperparameter set (`trial`): the model is trained on combined train folds, generates synthetic data the size of the val fold, `Benchmark` computes the metric.
3. Average metric across folds — optimization objective.

**Search space modes:**
- `"manual"` — only hyperparameters defined by the user via `get_params(trial)`
- `"extended"` — search space is extended with preset ranges for the given architecture (layer sizes, learning rate, epochs, batch size, etc.)
