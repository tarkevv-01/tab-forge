# LLM Selection Module

## Why You Need This

Choosing a model for a specific dataset is essentially a meta-learning task: "which architecture has historically worked best on similar data?". Instead of running all 6 models and comparing results, Tab-Forge formulates this question as a prompt for an LLM.

Key finding: an LLM equipped with dataset meta-characteristics and experiment results on reference datasets is able to predict the best model with accuracy **top1-acc up to 0.84** (few-shot mode).

---

## PromptGenerator

`PromptGenerator` does two things:
1. Extracts meta-characteristics of your dataset
2. Assembles them into a structured prompt

```python
from tab_forge.prompt_generator import PromptGenerator

gen = PromptGenerator()
prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",
    shot_mode     = "few",
    mfe_features  = "short",
)
print(prompt[:500])  # view the beginning of the prompt
```

### `build_prompt` Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `dataset` | `Dataset` | — | Your dataset |
| `target_metric` | see below | `"r2_metric"` | Target metric for ranking |
| `shot_mode` | `"zero"`, `"few"` | `"zero"` | Zero-shot or few-shot |
| `mfe_features` | `"short"`, `"full"`, list | `"short"` | Meta-feature set |
| `mfe_groups` | list of pymfe groups | `None` | Groups for pymfe |
| `models_to_include` | list of models | all 6 | Limit the model set |

### target_metric Values

| String | Metric |
|--------|--------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

---

## Dataset Meta-Features

### What is Computed

`PromptGenerator` automatically computes the following meta-characteristics:

| Meta-feature | Meaning |
|-------------|---------|
| `nr_inst` | Number of rows |
| `nr_attr` | Number of features |
| `nr_num` | Numerical features |
| `nr_cat` | Categorical features |
| `missing_pct` | Fraction of missing values, % |
| `task_type` | Task type (regression/classification) |
| `abs_corr_mean` | Mean absolute correlation |
| `abs_corr_max` | Maximum absolute correlation |
| `skewness_mean` | Mean feature skewness |
| `kurtosis_mean` | Mean kurtosis value |
| `kurtosis_std` | Kurtosis standard deviation |
| `std_mean` | Mean standard deviation |

In `mfe_features="full"` mode, all available meta-features from the selected `pymfe` groups are additionally computed.

### Meta-Feature Importance

Based on meta-classifier experiments, the greatest influence on predicting the best model comes from:

1. **`nr_num`** — number of numerical features
2. **`missing_pct`** — fraction of missing values
3. **`task_type`** — task type

This means: numerical datasets without missing values and datasets with a large fraction of missing values behave differently, and the LLM accounts for this.

---

## Zero-shot vs Few-shot

### Zero-shot

The prompt contains only:
- Meta-characteristics of your dataset
- Descriptions of 6 supported models
- Target metric

The LLM responds based on general knowledge about architectures and their properties.

```python
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="zero")
```

### Few-shot

In addition to the zero-shot prompt, results of preliminary experiments on 5 reference datasets are added:

- **abalone** (regression, ~4177 rows)
- **cl-housing** (regression, real estate)
- **air-quality** (regression, air quality)
- **wind** (regression, wind speed)
- **gats** (regression)

The LLM sees: on a similar dataset, model X showed such-and-such results for such-and-such metric — this is concrete experience to rely on.

```python
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="few")
```

!!! tip "When to use which"
    - **Few-shot** — always preferred: gives top1-acc up to 0.84 vs 0.6 for zero-shot (according to experiments)
    - **Zero-shot** — if you want a quick answer without the overhead of reading experimental data

---

## LLMRunner

`LLMRunner` sends the prompt to the LLM and aggregates results from multiple runs.

### Creating and Running

```python
from tab_forge.llm_runner import LLMRunner

runner = LLMRunner(
    base_url        = "https://api.openai.com/v1",
    api_key         = "sk-...",
    model           = "gpt-4o",
    retry_on_parse_error = True,   # retry if LLM gave an invalid response
    request_delay   = 0.5,         # pause between requests in seconds
)

result = runner.run(
    prompt      = prompt,
    n_runs      = 5,        # number of independent requests
    temperature = 0.7,
    top_p       = 1.0,
    max_tokens  = 2048,
)
```

### `run` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | `1` | Number of independent requests (self-consistency) |
| `temperature` | `0.7` | Generation temperature |
| `top_p` | `1.0` | Top-p sampling |
| `max_tokens` | `2048` | Maximum response length |
| `seed` | `None` | Seed (if supported by the model) |

---

## Self-Consistency: How It Works

Self-consistency is the key technique that makes LLM-based selection reliable.

```
Request 1 → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']
Request 2 → ['CTGAN', 'GAN-MFS', 'WGAN-GP', 'TVAE', 'CTABGAN+', 'DDPM']
Request 3 → ['GAN-MFS', 'WGAN-GP', 'CTGAN', 'CTABGAN+', 'DDPM', 'TVAE']
Request 4 → ['GAN-MFS', 'CTGAN', 'CTABGAN+', 'WGAN-GP', 'TVAE', 'DDPM']
Request 5 → ['CTGAN', 'GAN-MFS', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

Average ranks:
  GAN-MFS:  (1+2+1+1+2)/5 = 1.4
  CTGAN:    (2+1+3+2+1)/5 = 1.8
  WGAN-GP:  (3+3+2+4+3)/5 = 3.0
  ...

Final ranking: ['GAN-MFS', 'CTGAN', 'WGAN-GP', ...]
```

The LLM behaves stochastically when `temperature > 0`. 5 independent requests with rank averaging give a statistically robust result.

---

## RunnerResult

```python
result = runner.run(prompt, n_runs=5)

print(result.final_ranking)
# ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

print(result.average_ranks)
# {'GAN-MFS': 1.4, 'CTGAN': 1.8, ...}

print(result.n_runs)    # 5
print(result.n_parsed)  # how many of the 5 LLM responses were successfully parsed
```

!!! warning "Parse errors"
    If `n_parsed < n_runs`, the LLM did not give a valid response in the required format for some requests. Enable `retry_on_parse_error=True` to retry requests on parse errors.

---

## Connecting a Local LLM

Tab-Forge is compatible with any OpenAI-compatible API:

=== "Ollama"

    ```python
    runner = LLMRunner(
        base_url = "http://localhost:11434/v1",
        api_key  = "ollama",          # any string
        model    = "llama3.1:8b",
    )
    ```

=== "LM Studio"

    ```python
    runner = LLMRunner(
        base_url = "http://localhost:1234/v1",
        api_key  = "lm-studio",
        model    = "local-model",
    )
    ```

=== "OpenAI"

    ```python
    runner = LLMRunner(
        base_url = "https://api.openai.com/v1",
        api_key  = "sk-...",
        model    = "gpt-4o",
    )
    ```

!!! note "Model quality"
    Ranking quality depends on the LLM's ability to follow instructions and reason about tabular data. GPT-4o and similar models produce the best results. Small local models may give less stable rankings — increase `n_runs` to compensate.
