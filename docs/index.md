# Tab-Forge

**A modular Python library for automated tuning of generative models on tabular data**

---

Tab-Forge solves a real problem: selecting and configuring a generative model for tabular data synthesis is a labor-intensive task. You need to figure out which of six completely different architectures (GAN, VAE, Diffusion) to try first, which quality metrics matter for your task, and how much time to spend on tuning.

Tab-Forge automates this entire process: an LLM analyzes the meta-characteristics of your dataset and predicts which models to try first — then Optuna tuning takes care of the rest.

---

## Who This Library Is For

- **Data scientists** who need synthetic data for augmentation, class balancing, or testing
- **ML engineers** working with private or limited datasets
- **Researchers** studying generative models for tabular data
- Anyone tired of manually tuning GAN hyperparameters

---

## Key Advantages

| Feature | Details |
|---------|---------|
| **6 architectures out of the box** | CTGAN, WGAN-GP, GAN-MFS, CTABGAN+, TVAE, TabDDPM — unified interface |
| **LLM-assisted selection** | PromptGenerator + LLMRunner rank models by meta-features of your dataset |
| **Self-consistency** | N independent LLM requests, rank aggregation via voting — robust result |
| **Bayesian tuning** | AutoTuningStudy via Optuna with k-fold CV; supports extended and manual modes |
| **5 quality metrics** | R², RMSE, Jensen–Shannon, Frobenius Correlation, Frobenius MI |
| **Modularity** | Each component can be used independently |

---

## Quick Example

```python
from tab_forge.dataset import Dataset
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner
from tab_forge.tuning import AutoTuningStudy

# 1. Load data
dataset = Dataset(
    data="abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

# 2. LLM selects the best model for this dataset
prompt = PromptGenerator().build_prompt(
    dataset=dataset,
    target_metric="r2_metric",
    shot_mode="few",
)
result = LLMRunner(base_url="https://api.openai.com/v1",
                   api_key="sk-...", model="gpt-4o").run(prompt, n_runs=5)

print("Ranking:", result.final_ranking)
# → ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

# 3. Tune the best model
study = AutoTuningStudy(
    model_class=result.final_ranking[0],  # string from LLM ranking
    search_space_mode="extended",
    cv=3,
)
study.optimize(dataset, n_trials=25)
print("Best parameters:", study.best_params)
```

---

## Pipeline Architecture

```
Your dataset
     │
     ▼
PromptGenerator  ──── meta-features + experience on reference datasets
     │
     ▼
LLMRunner  ──── self-consistency (N requests → rank averaging)
     │
     ▼
AutoTuningStudy  ──── Optuna + k-fold CV
     │
     ├──► Models  (CTGAN / WGAN-GP / GAN-MFS / CTABGAN+ / TVAE / DDPM)
     │
     └──► Benchmark  (R² / RMSE / JS / FrobCorr / FrobMI)
```

---

## Documentation Sections

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)**
  How to install the library and its dependencies

- **[Quick Start](getting-started/quickstart.md)**
  Full pipeline in 5 minutes

- **[Quality Metrics](concepts/metrics.md)**
  What each metric measures and when to use it

- **[Architecture](concepts/architecture.md)**
  How modules are organized and how they interact

- **[User Guide](user-guide/dataset.md)**
  Detailed description of each module

- **[Experiments](experiments/tuning-results.md)**
  Research results and conclusions

</div>
