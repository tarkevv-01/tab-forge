# Synthetic Data Quality Metrics

Tab-Forge evaluates synthetic data by five metrics, each looking at the problem from a different angle. Understanding their meaning helps you correctly choose the optimization objective for your task.

---

## Summary Table

| Metric | Code string | Direction | What it measures |
|--------|------------|-----------|-----------------|
| R² | `"r2"` | ↑ higher = better | ML utility: how well a model trained on synthetic data performs on real data |
| RMSE | `"rmse"` | ↓ lower = better | Prediction error of an ML model trained on synthetic data |
| Jensen–Shannon | `"js_mean"` | ↓ lower = better | Divergence of marginal feature distributions |
| Frobenius Correlation | `"frob_corr"` | ↓ lower = better | Difference between the synthetic and real correlation matrices |
| Frobenius MI | `"frob_mi"` | ↓ lower = better | Difference between mutual information matrices: nonlinear dependencies |

---

## R² — ML Utility

### What it measures

R² (coefficient of determination) evaluates the **practical utility** of synthetic data: how well an ML model trained **only** on synthetic data performs predictions on **real** test data.

Evaluation scheme:

```
Synthetic data → train ML model → predict on real data → R²
```

### Formula

\[
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
\]

Value ranges from \(-\infty\) to \(1\). When \(R^2 = 1\), synthetic data perfectly reproduces the relationships in real data. When \(R^2 = 0\), the model is no better than the mean.

### When to use as optimization objective

R² is the **most informative** metric if you want synthetic data to replace real data in ML tasks: augmentation, knowledge transfer, training on synthetic data. According to experiments, GAN-MFS showed the best R² results (8 improvements out of 25 trials) — making it a good first choice for tasks where preserving variance is important.

!!! tip "When to choose R²"
    If the goal is to train an ML model on synthetic data and then apply it to real data, optimize by R².

---

## RMSE — Prediction Error

### What it measures

Same train-on-synthetic/test-on-real scheme, but the metric is the root mean square prediction error:

\[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

Unlike R², RMSE is not normalized by the variance of the target variable — its absolute value depends on the scale of the data.

### When to use as optimization objective

!!! tip "When to choose RMSE"
    If prediction accuracy in original units matters (e.g., error in dollars, grams, days), rather than the explained fraction of variance, RMSE is more interpretable than R². Experiments show CTGAN and WGAN-GP produced stable RMSE results (6 improvements each).

---

## Jensen–Shannon Divergence — Distribution Similarity

### What it measures

For each feature, the divergence between its distribution in synthetic and real data is computed. The final metric is the **average** across all features.

JSD is a symmetric version of KL divergence:

\[
\text{JSD}(P \| Q) = \frac{1}{2} D_\text{KL}(P \| M) + \frac{1}{2} D_\text{KL}(Q \| M), \quad M = \frac{P + Q}{2}
\]

Value ranges from 0 (identical distributions) to \(\ln 2 \approx 0.693\) (orthogonal distributions). Tab-Forge uses a normalized version: **0 — perfect, closer to 1 — worse**.

### When to use as optimization objective

!!! tip "When to choose JS"
    If accurate reproduction of the distribution of each feature individually is critical — e.g., for auditing or regulatory compliance. Note: WGAN-GP showed the best JS divergence results (12 improvements!), which corresponds to its theoretical basis — minimizing Wasserstein distance.

!!! warning "Limitation"
    JS looks at features independently. A model may reproduce each marginal distribution well but break dependencies between features. Complement JS with Frobenius metrics.

---

## Frobenius Correlation (LF) — Linear Dependency Structure

### What it measures

Compares the **correlation matrices** of real and synthetic data using the Frobenius norm:

\[
\text{FrobCorr} = \| \text{Corr}(X_\text{real}) - \text{Corr}(X_\text{synth}) \|_F
= \sqrt{\sum_{i,j} \left( r_{ij}^\text{real} - r_{ij}^\text{synth} \right)^2}
\]

A value of 0 means the synthetic data perfectly reproduces all pairwise linear correlations between features.

### When to use as optimization objective

!!! tip "When to choose FrobCorr"
    If the data has strong linear dependencies between features (multicollinearity) and you want to preserve them in the synthetic data — choose this metric. Especially important for financial and medical data.

---

## Frobenius MI — Nonlinear Dependency Structure

### What it measures

Analogous to FrobCorr, but uses the **mutual information matrix** instead of correlation:

\[
\text{FrobMI} = \| \text{MI}(X_\text{real}) - \text{MI}(X_\text{synth}) \|_F
\]

Mutual information \(I(X_i; X_j)\) captures **nonlinear** dependencies that Pearson correlation misses. A value of 0 — perfect reproduction.

### When to use as optimization objective

!!! tip "When to choose FrobMI"
    When the data contains complex nonlinear dependencies (e.g., biological, physical processes). Experiments show TabDDPM produced the best MI results (6 improvements) — diffusion models capture complex dependencies well.

---

## How to Choose a Metric for Optimization

A practical guide:

```
My goal                                    →  Recommended metric
─────────────────────────────────────────────────────────────────────
Train an ML model on synthetic data        →  r2 or rmse
Reproduce the distribution of each
  feature (for auditing/compliance)        →  js_mean
Preserve linear correlations               →  frob_corr
Preserve nonlinear dependencies            →  frob_mi
General quality check                      →  all five together
```

!!! note "Optimizing one metric and its effect on others"
    Experiments revealed an interesting fact: optimizing GAN-MFS by R² simultaneously improved most other metrics. This indicates that R² is the most "general" quality metric for regression tasks. Details in the [Tuning Results](../experiments/tuning-results.md) section.

---

## Using Multiple Metrics Simultaneously

Benchmark supports computing multiple metrics at the same time:

```python
from tab_forge.benchmark import Benchmark

bench = Benchmark([
    ("r2",       {"model": "xgboost"}),
    ("rmse",     {"model": "xgboost"}),
    ("js_mean",  {}),
    ("frob_corr", {}),
    ("frob_mi",  {}),
])

result = bench.evaluate(synth_dataset, real_dataset)
print(result.metrics)
```

For tuning by multiple metrics at once, use a named dictionary and `direction="maximize"` / `"minimize"` in `AutoTuningStudy` — or normalize the metrics yourself via a custom `get_params`.
