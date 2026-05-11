# Tab-Forge Architecture

## Module Diagram

![Tab-Forge Architecture Diagram](../img/arch.png)

---

## Modularity Principle

Each Tab-Forge module is self-contained and does not require the others. You can:

- Use only **Dataset + Models + Benchmark** without LLM
- Run **AutoTuningStudy** manually without an LLM ranking
- Use **PromptGenerator** separately to explore dataset meta-features
- Connect your own model to **Benchmark** outside of Tab-Forge

---

## Data Flow in the Full Pipeline
![Universal Tab-forge pipeline diagram](../img/pipeline.png)

---

## `dataset` Module

**Key exports:** `Dataset`, `merge_datasets`, `split_folds`

`Dataset` is the central object of the library. All models accept it as input and return it from `structed_generate`. This allows seamless data transfer between modules.

```python
from tab_forge.dataset import Dataset, merge_datasets, split_folds
```

---

## `models` Module

**Key exports:** 6 synthesizer classes

All models inherit `BaseGenerativeModel` and implement a unified interface:

| Method | Description |
|--------|-------------|
| `fit(dataset)` | Train on `Dataset` |
| `generate(n_samples)` | Generate as `pd.DataFrame` |
| `structed_generate(n_samples)` | Generate as `Dataset` (for Benchmark/AutoTuning) |
| `get_losses()` | Training loss history |

---

## `tuning` Module

**Key exports:** `AutoTuningStudy`, `TuningStudy`

`TuningStudy` is a thin wrapper around `optuna.Study`. `AutoTuningStudy` is a full CV-tuning pipeline with preset search spaces for each architecture.

---

## `benchmark` Module

**Key exports:** `Benchmark`, `BenchmarkResult`

All metrics are implemented using the train-on-synthetic / test-on-real scheme or by comparing statistics. Accepts `Dataset` objects.

---

## `prompt_generator` Module

**Key exports:** `PromptGenerator`

Uses `pymfe` to compute meta-features and loads pre-computed experiment results from `experiment_results/` for few-shot mode.

---

## `llm_runner` Module

**Key exports:** `LLMRunner`, `RunnerResult`

Works with any OpenAI-compatible API. Implements self-consistency: N independent calls → rank averaging via `_aggregate`.
