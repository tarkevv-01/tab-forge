# Tab-Forge

**A modular Python library for automated tuning of generative models on tabular data**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Description

Tab-Forge is a comprehensive solution for working with generative models on tabular data. The library provides a modular architecture that combines data loading, model training, quality evaluation, automated hyperparameter tuning, and LLM-assisted model selection into a unified pipeline.

## Key Features

### 1. **Dataset** — Data Management
- Unified loading and preprocessing of tabular data
- Support for numerical and categorical features
- Automatic splitting into train, validation, and test sets

### 2. **Models** — Generative Architectures
- Unified interface for various architectures
- Supported models: **CTGAN**, **WGAN-GP**, **GAN-MFS**, **CTABGAN+**, **TVAE**, **DDPM**
- Easy integration of new architectures

### 3. **Benchmark** — Quality Evaluation
- Computation of synthetic data quality metrics
- Support for multiple quality meta-characteristics
- Comparison of results across configurations and datasets

### 4. **PromptGenerator** — LLM Context Assembly
- Extraction of meta-characteristics of user dataset (via pymfe)
- Assembly of a structured prompt: model descriptions, target metric, dataset meta-features
- Support for zero-shot and few-shot modes — in few-shot mode the prompt includes experiment results on reference datasets

### 5. **LLMRunner** — LLM-Assisted Model Selection
- Sends the assembled prompt to an LLM (any OpenAI-compatible API)
- Self-consistency: N independent runs with rank aggregation
- Output is an ordered list of models by average rank; this order determines tuning priority

### 6. **Tuning** — Optimization
- Automatic and semi-automatic hyperparameter tuning
- Integration with **Optuna** for efficient search
- Selection of the optimal configuration based on experiment results

## Architecture

```
┌──────────────────────────────────────────┐
│            PromptGenerator               │ ← dataset meta-features +
│                                          │   experience on reference datasets
└───────────────────┬──────────────────────┘
                    │ assembled prompt
                    ↓
┌──────────────────────────────────────────┐
│               LLMRunner                  │ ← model ranking
│       (self-consistency, N runs)         │   via self-consistency
└───────────────────┬──────────────────────┘
                    │ prioritized model list
                    ↓
┌──────────────────────────────────────────┐
│                 Tuning                   │ ← single control point:
│                                          │   iterates models, datasets,
│                                          │   launches experiments
└──────┬───────────────────────┬───────────┘
       │                       │
       ↓                       ↓
┌──────────────┐   ┌───────────────────────┐
│    Models    │   │        Dataset        │
└──────────────┘   └───────────────────────┘
                              │
                              ↓
               ┌──────────────────────────┐
               │        Benchmark         │ ← read-only results:
               │                          │   metrics, evaluation, logging
               └──────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

### Full Pipeline

```python
from tab_forge.dataset import Dataset
from tab_forge.tuning import AutoTuningStudy
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner

# 1. Load data
dataset = Dataset(
    data="your_data.csv",           # path to CSV or pd.DataFrame
    target="target_column",         # target variable
    task_type="regression",         # "regression" or "classification"
    numerical_features=["feat1", "feat2", "feat3"],
    categorical_features=["cat_feat"],
)

# 2. LLM ranks models based on dataset meta-characteristics
prompt = PromptGenerator().build_prompt(
    dataset=dataset,
    target_metric="r2_metric",  # target task
    shot_mode="few",            # "zero" or "few"
)

runner = LLMRunner(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o")
result = runner.run(prompt, n_runs=5, temperature=0.7)
print("Model ranking:", result.final_ranking)
# ['GAN-MFS', 'CTGAN', 'WGAN-GP', ...]

# 3. Tune the highest-priority model
# AutoTuningStudy accepts a string — the model name from the LLM ranking
study = AutoTuningStudy(model_class=result.final_ranking[0], search_space_mode="extended", cv=3)
study.optimize(dataset, n_trials=25)
print("Best parameters:", study.best_params)

# 4. Generate synthetic data
synth_df = study.best_model.generate(n_samples=1000)
```

> More details: [docs/quickstart.md](docs/quickstart.md) and examples in the `examples/` folder.

## Applications

Tab-Forge addresses the following tasks:
- **Data augmentation** to improve ML models
- **Data privacy** through generation of synthetic datasets
- **Class balancing** for imbalanced datasets
- **Data exploration** and hypothesis testing

## Supported Models

| Model | Description |
|-------|-------------|
| **CTGAN** | Conditional Tabular GAN with mode-specific normalization |
| **WGAN-GP** | Wasserstein GAN with gradient penalty |
| **GAN-MFS** | GAN with meta-feature distribution matching in the loss function |
| **CTABGAN+** | Extended CTABGAN with auxiliary classification and regression heads |
| **TVAE** | Variational Autoencoder with tabular ELBO adaptation |
| **DDPM** | Diffusion model (Tab-DDPM) based on iterative denoising |

## Quality Metrics

- **R²** — proportion of explained variance of real data
- **RMSE** — root mean square error of reconstruction
- **Jensen–Shannon divergence** — symmetric measure of distribution divergence
- **Frobenius correlation** — norm of the difference between correlation matrices
- **Frobenius mutual information** — preservation of nonlinear dependencies

## Authors

**ITMO University**

GitHub: [tarkevv-01](https://github.com/tarkevv-01)

---

**⭐ If the project was useful, give it a star on GitHub!**
