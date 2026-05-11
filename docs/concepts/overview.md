# Overview and Motivation

## The Problem

Synthetic tabular data is needed everywhere — for training set augmentation, privacy protection, class balancing, and ML pipeline testing. But selecting the right generative model is a non-trivial task.

Today there are dozens of architectures: CTGAN, WGAN, VAE, Diffusion models. Each works differently well on different types of data. The classic approach — trying all options — takes days of computation and requires deep understanding of each architecture.

**Key questions every practitioner faces:**

1. Which model should I try first for my dataset?
2. Which quality metric should I use as the optimization objective?
3. How long should I tune and in what hyperparameter range?

---

## Tab-Forge's Solution

Tab-Forge offers a three-level answer to these questions:

### Level 1: LLM as Meta-Classifier

Before launching expensive tuning, the library extracts **meta-characteristics** of your dataset (size, feature types, missing values, correlations, skewness, etc.) and sends them to an LLM along with descriptions of all supported models.

The LLM analyzes the context — in few-shot mode it also sees results of preliminary experiments on similar datasets — and outputs a **prioritized model ranking**. This is not a random choice: experiments show top1-acc up to **0.84** for some metrics.

### Level 2: Bayesian Tuning with k-fold CV

`AutoTuningStudy` runs Optuna optimization with cross-validation: at each trial the model is trained on train folds, generates synthetic data the size of the val fold, and `Benchmark` evaluates quality. This provides a reliable estimate of the hyperparameter generalization ability.

### Level 3: Standardized Evaluation

Five quality metrics (`r2`, `rmse`, `js_mean`, `frob_corr`, `frob_mi`) cover different aspects: from distributions of individual features to the structure of dependencies between them. You can optimize by one metric and observe how the others change.

---

## Where to Apply Tab-Forge

!!! tip "Typical use cases"
    - **Data augmentation**: few training examples — synthetic data helps ML models generalize
    - **Class balancing**: generate synthetic examples for underrepresented classes
    - **Privacy**: synthetic data as a replacement for real data in development and testing
    - **Research**: comparing architectures on the same dataset with the same pipeline

---

## What Tab-Forge Does Not Do

- Does not work with text, images, or time series — only **tabular data**
- Does not guarantee differential privacy (but CTABGAN+ has a corresponding mechanism)
- Does not replace deep understanding of data — the LLM ranking is a starting point, not a final answer
