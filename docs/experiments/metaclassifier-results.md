# Meta-Classifier Results (LLM Selection)

## What We Measured

The key question: **how well does the LLM predict the best model for a dataset?**

We tested 4 prompt construction strategies:

- **Zero-shot + full prompt** — meta-features + full model descriptions, no examples
- **Zero-shot + short prompt** — meta-features + brief descriptions, no examples
- **Few-shot + full prompt** — meta-features + full descriptions + results on reference datasets
- **Few-shot + short prompt** — meta-features + brief descriptions + results on reference datasets

And two types of prediction quality metrics:

- **ρ Spearman** — how well the LLM orders the models (rank correlation with the real ranking)
- **top1-acc** — how often the LLM correctly names the **best** model on the first attempt

---

## What is Zero-shot and Few-shot

### Zero-shot

The LLM sees only:
- Dataset meta-characteristics (size, features, statistics)
- Model descriptions
- Target metric

The model responds based on general knowledge — **without examples** of how models behaved on real data.

### Few-shot

In addition to the zero-shot content, **results of preliminary experiments** on 5 reference datasets are added. The LLM sees specific numbers: for example, "on the abalone dataset when optimizing by R², the GAN-MFS model showed the best result". This is contextual experience that helps the LLM reason better.

### Full vs Short — dataset meta-feature set

This refers to the number of meta-characteristics of the user dataset that are passed in the prompt.

- **Full** — the complete set of **43 meta-features** (all available pymfe features): distribution statistics, information content characteristics, dimensionality, etc.
- **Short** — a reduced set of meta-features formed by feature importance analysis (threshold > 0.05). Includes the key features the LLM actually relies on for ranking — the full set provides no significant quality gain.

---

## Results Table

### ρ Spearman (average across datasets)

*How well the LLM orders models — from 0 (random) to 1 (perfect). A value of 0.5+ is considered good.*

| Metric | zero-shot full | zero-shot short | few-shot full | few-shot short |
|--------|---------------|-----------------|---------------|----------------|
| **R²** | 0.5815 | 0.5040 | **0.7234** | 0.6937 |
| **LF** | **0.5794** | 0.5406 | 0.4080 | 0.4171 |
| **JS** | 0.1611 | 0.2846 | 0.3029 | 0.2891 |
| **MI** | 0.4948 | 0.5085 | **0.7463** | 0.6251 |
| **RMSE** | 0.2891 | 0.3143 | 0.3280 | 0.2800 |

### top1-acc (average across datasets)

*Fraction of cases in which the LLM correctly named the best model in first place. 0.84 = in 84% of cases the LLM is correct.*

| Metric | zero-shot full | zero-shot short | few-shot full | few-shot short |
|--------|---------------|-----------------|---------------|----------------|
| **R²** | 0.48 | 0.36 | 0.60 | 0.60 |
| **LF** | 0.28 | 0.36 | 0.60 | 0.64 |
| **JS** | 0.16 | 0.40 | **0.84** | 0.76 |
| **MI** | 0.44 | 0.44 | **0.84** | 0.76 |
| **RMSE** | 0.32 | 0.32 | 0.44 | 0.52 |

!!! note "Result highlighting"
    In the original presentation, green cells indicate high values (good), red indicate low. In both tables, **few-shot + full prompt** dominates for most metrics.

---

## Interpretation of Results

### Few-shot is significantly better than zero-shot

For most metrics, switching to few-shot improves prediction quality. The most striking example:

- **JS zero-shot full**: top1-acc = **0.16** (LLM is almost choosing randomly)
- **JS few-shot full**: top1-acc = **0.84** (in 84% of cases the LLM correctly names the best model!)

A 5x gap! This is explainable: from general knowledge it is hard to understand which GAN architecture will better reproduce JS divergence — but specific experimental data gives the LLM the necessary context.

### Best results — for JS and MI

Few-shot + full prompt for **JS** and **MI** gives top1-acc = **0.84**. This means that in 84% of cases the LLM correctly names the best model on the first attempt — without running a single tuning trial.

### R² and LF — confident ranking

For **R²** ρ Spearman = **0.72** (few-shot full) — the LLM doesn't just guess the best model, but also orders the rest reasonably well. This means: the order in which you tune models by LLM ranking corresponds to the real order of their quality.

### RMSE — a difficult metric for LLM

For **RMSE** all strategies show relatively modest results (top1-acc 0.32–0.52, ρ 0.28–0.33). This may be because RMSE strongly depends on data scale and target variable specifics — information that is hard to express through standard meta-features.

---

## What is ρ Spearman in This Context

**ρ Spearman** is the rank correlation coefficient. In our context:

1. We have a **real ranking** of 6 models by metric X on dataset Y (obtained after tuning all models)
2. The LLM outputs a **predicted ranking**
3. ρ Spearman measures how closely these two rankings agree

\[
\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}
\]

where \(d_i\) is the rank difference for the i-th model.

- **ρ = 1.0** — LLM predicted the order perfectly
- **ρ = 0.0** — ranking is random
- **ρ = -1.0** — LLM ranked everything in reverse order

Values of 0.7+ (achieved for R² and MI in few-shot mode) are very good results for a meta-learning task.

---

## Practical Conclusions

### 1. Use few-shot mode

Always use `shot_mode="few"` — the difference from zero-shot is significant for most metrics.

### 2. Full vs Short meta-feature set

Despite the fact that **full** (43 meta-features) is on average slightly better or equal to short, the difference is negligible. Use `mfe_features="short"` — the reduced set covers the features the LLM actually relies on, and gives comparable quality with a more compact prompt.

### 3. If optimizing by RMSE — double-check the LLM ranking

The LLM is less reliable at predicting the best model for RMSE. Consider tuning two or three top models from the ranking, not just the first.

### 4. Recommended configuration

```python
prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",  # or "mi_matrix_metric"
    shot_mode     = "few",        # ✓ few-shot
    mfe_features  = "short",      # ✓ 12 key features
)
result = runner.run(prompt, n_runs=5, temperature=0.7)
# top1-acc ~0.60–0.84 depending on the metric
```
