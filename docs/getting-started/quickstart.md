# Quick Start

On this page we will go through the full cycle: data loading → LLM model selection → tuning → synthetic data generation → quality evaluation.

For the example we use the **abalone** dataset (regression task: predicting the age of an abalone from its physical characteristics).

---

## Step 1 — Load Data

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="examples/abalone.csv",          # path to CSV or pd.DataFrame
    target="Rings",                        # target variable
    task_type="regression",               # "regression" or "classification"
    numerical_features=[
        "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight",
    ],
    categorical_features=["Sex"],
)

print(dataset.info)
# DatasetInfo(n_samples=4177, n_features=8, n_numerical=7, n_categorical=1, ...)
```

Split into train and test sets:

```python
train, test = dataset.train_test_split(test_size=0.2, random_state=42)
print(f"Train: {len(train)} rows, Test: {len(test)} rows")
```

---

## Step 2 — LLM Selects the Best Model

`PromptGenerator` computes dataset meta-characteristics (size, ratio of numerical/categorical features, missing values, correlations, etc.) and assembles them into a structured prompt for the LLM.

```python
from tab_forge.prompt_generator import PromptGenerator

gen = PromptGenerator()
prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",   # metric to rank models by
    shot_mode     = "few",         # few-shot: LLM sees experience on reference datasets
    mfe_features  = "short",       # curated set of 12 key meta-features
)
```

Now send the prompt to the LLM. `LLMRunner` makes `n_runs` independent requests and averages the ranks — this is called **self-consistency**:

```python
from tab_forge.llm_runner import LLMRunner

runner = LLMRunner(
    base_url = "https://api.openai.com/v1",
    api_key  = "sk-...",
    model    = "gpt-4o",
)

result = runner.run(prompt, n_runs=5, temperature=0.7)

print("Final ranking:", result.final_ranking)
# → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

print("Average ranks:", result.average_ranks)
# → {'GAN-MFS': 1.4, 'CTGAN': 2.2, 'WGAN-GP': 2.8, ...}
```

!!! note "Local LLM"
    LLMRunner is compatible with any OpenAI-compatible API, including local servers based on llama.cpp, Ollama, and LM Studio. Just specify the `base_url` of the desired server.

---

## Step 3 — Hyperparameter Tuning

Take the best model according to the LLM and run Bayesian tuning via Optuna. The `"extended"` mode automatically defines the search space for the given architecture:

```python
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = result.final_ranking[0],  # model name string from ranking
    search_space_mode = "extended",               # automatic search space
    cv                = 3,                        # number of CV folds
)

study.optimize(dataset, n_trials=25)

print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
```

!!! tip "Tuning by ranking"
    If you have time — run tuning sequentially for the first few models from `result.final_ranking` and compare the final metrics.

---

## Step 4 — Generation and Quality Evaluation

```python
from tab_forge.models import GANMFSSynthesizer
from tab_forge.benchmark import Benchmark

# Create model with best parameters and train on full train set
model = GANMFSSynthesizer(**study.best_params)
model.fit(train)

# Generate synthetic data the size of the test set
synth = model.structed_generate(n_samples=len(test))

# Evaluate quality
bench = Benchmark([
    ("r2",      {"model": "xgboost"}),
    ("rmse",    {"model": "xgboost"}),
    ("js_mean", {}),
])

result_bench = bench.evaluate(synth, test)
print(result_bench)
```

---

## Full Pipeline in One Block

```python
from tab_forge.dataset import Dataset
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner
from tab_forge.tuning import AutoTuningStudy
from tab_forge.benchmark import Benchmark

# ── 1. Data ────────────────────────────────────────────────
dataset = Dataset(
    data="examples/abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight",
                        "Viscera weight", "Shell weight"],
    categorical_features=["Sex"],
)
train, test = dataset.train_test_split(test_size=0.2, random_state=42)

# ── 2. LLM model selection ─────────────────────────────────
prompt = PromptGenerator().build_prompt(
    dataset=dataset, target_metric="r2_metric", shot_mode="few"
)
llm_result = LLMRunner(
    base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o"
).run(prompt, n_runs=5)
print("Ranking:", llm_result.final_ranking)

# ── 3. Tuning ───────────────────────────────────────────────
study = AutoTuningStudy(
    model_class=llm_result.final_ranking[0],
    search_space_mode="extended",
    cv=3,
)
study.optimize(train, n_trials=25)
print("Best parameters:", study.best_params)

# ── 4. Generation and evaluation ───────────────────────────
from tab_forge.models import (
    GANMFSSynthesizer, CTGANSynthesizer, WGANGPSynthesizer,
    CTABGANPlusSynthesizer, TVAESynthesizer, DDPMSynthesizer,
)
MODEL_REGISTRY = {
    "GAN-MFS": GANMFSSynthesizer, "CTGAN": CTGANSynthesizer,
    "WGAN-GP": WGANGPSynthesizer, "CTABGAN+": CTABGANPlusSynthesizer,
    "TVAE":    TVAESynthesizer,    "DDPM":    DDPMSynthesizer,
}

model_class = MODEL_REGISTRY[llm_result.final_ranking[0]]
model = model_class(**study.best_params)
model.fit(train)
synth = model.structed_generate(n_samples=len(test))

bench = Benchmark([("r2", {"model": "xgboost"}), ("js_mean", {})])
print(bench.evaluate(synth, test))
```

---

## Next Steps

- Learn more about [quality metrics](../concepts/metrics.md)
- Explore all [6 models](../user-guide/models.md) and when to use them
- See the [experiment results](../experiments/tuning-results.md)
