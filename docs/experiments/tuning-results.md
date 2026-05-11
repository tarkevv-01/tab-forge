# Tuning Experiment Results

## Experiment Objective

We wanted to answer two questions:

1. If you optimize a model by one metric — do the others improve as well?
2. Which tuning metric provides the greatest "generalizing" effect for each model?

This matters: running tuning across 5 metrics simultaneously is expensive. If one metric "pulls" the others along — it is sufficient to optimize by that one alone.

---

## Experimental Setup

- **5 datasets:** abalone, cl-housing, air-quality, wind, gats
- **6 models:** CTGAN, WGAN-GP, GAN-MFS, CTABGAN+, TVAE, TabDDPM
- **25 trials** of tuning for each metric and each model
- **k-fold cross-validation** in each trial
- After tuning — evaluation of **all 5 metrics** on the holdout set

The number in each table cell: **how many times out of 25 trials** did optimizing by the given metric improve all *other* metrics (not the one being optimized).

---

## Results Table

*Numbers represent how many trials (out of 25) in which optimization by the given metric improved the other synthetic data quality metrics.*

| Model | R² | LF (FrobCorr) | JS | MI (FrobMI) | RMSE |
|-------|----|---------------|----|-------------|------|
| **CTGAN** | 5 | 6 | 4 | 4 | 6 |
| **WGAN-GP** | 4 | 1 | 12 | 2 | 6 |
| **GAN-MFS** | **8** | 3 | 7 | 3 | 4 |
| **CTABGAN+** | 4 | 1 | 4 | 5 | **11** |
| **TVAE** | 2 | 3 | 7 | 5 | 8 |
| **TabDDPM** | 7 | 4 | **3** | **6** | 5 |

!!! note "How to read the table"
    Row GAN-MFS, column R²: value **8** means that in 8 out of 25 trials, when tuning optimized GAN-MFS by R², all 4 other metrics also improved (LF, JS, MI, RMSE). Green color in the original presentation indicated high values (good), red indicated low.

---

## Interpretation by Model

### CTGAN

Results are uniform across all metrics (4–6). No clear leader. If you use CTGAN and do not know which metric to tune by — choose **LF (FrobCorr)** or **RMSE** — both give 6 generalizing improvements.

### WGAN-GP

Clear leader — **JS divergence** (12 improvements — the maximum result in the entire table!). This is theoretically justified: WGAN-GP minimizes Wasserstein distance between distributions, which is directly related to JS. Optimizing WGAN-GP by JS — you are "playing to the architecture's strength".

!!! tip "Recommendation for WGAN-GP"
    Tune WGAN-GP by the `js_mean` metric. This will very likely also improve the other metrics.

### GAN-MFS

Clear leader — **R²** (8 improvements). GAN-MFS with its meta-feature regularizer particularly well preserves the statistical structure of data, which directly affects the ML utility of synthetic data.

!!! tip "Recommendation for GAN-MFS"
    Tune GAN-MFS by the `r2` metric. GAN-MFS showed **the best overall R² result among all models** — this is your best choice when ML utility of synthetic data is important.

### CTABGAN+

Record result — **RMSE** (11 improvements — second result after WGAN-GP/JS). CTABGAN+ with auxiliary regression heads is specifically aimed at target variable prediction accuracy. Tuning by RMSE "hits the target".

!!! tip "Recommendation for CTABGAN+"
    Tune CTABGAN+ by the `rmse` metric. The auxiliary regression heads in the discriminator make this model a natural RMSE optimizer.

### TVAE

Good results for **JS** (7) and **RMSE** (8). TVAE is a VAE architecture, and the ELBO loss function implicitly minimizes distribution divergence.

!!! tip "Recommendation for TVAE"
    Tune by `rmse` or `js_mean`. Both provide stable generalizing improvement.

### TabDDPM

Good results for **R²** (7) and **MI** (6), but **JS** — only 3 (one of the worst). Diffusion models model complex nonlinear dependencies well, but marginal distributions are not their strong suit.

!!! tip "Recommendation for TabDDPM"
    Tune by `r2` or `frob_mi` (MI). Do not expect excellent JS results from TabDDPM.

---

## General Conclusions

### 1. There is no universal "best metric" for all models

The architecture determines which metric optimally "unlocks" it. Tuning by the wrong metric is an inefficient use of resources.

### 2. Each model has its "native" metric

| Model | Recommended tuning metric |
|-------|--------------------------|
| CTGAN | LF or RMSE |
| WGAN-GP | JS ⭐ |
| GAN-MFS | R² ⭐ |
| CTABGAN+ | RMSE ⭐ |
| TVAE | RMSE or JS |
| TabDDPM | R² or MI |

### 3. LF (Frobenius Correlation) — weak signal for tuning

For most models, optimization by LF provides few generalizing improvements (1–6). This does not mean LF is not important — it just performs poorly as an **optimization objective**.

### 4. Use the LLM ranking to select the model, then tune by the right metric

Example of an optimal pipeline:

```
LLM recommends GAN-MFS → tune GAN-MFS by r2 → achieve maximum ML utility
```
