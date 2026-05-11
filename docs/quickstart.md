# Tab-Forge — Quick Start

## Installation

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

---

## Step 1. Load Data

```python
from tab_forge.dataset import Dataset

dataset = Dataset(
    data="abalone.csv",           # path to CSV or pd.DataFrame
    target="Rings",               # target variable
    task_type="regression",       # "regression" or "classification"
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

print(dataset.info)
```

Splitting into sets:

```python
from tab_forge.dataset import train_test_split

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
```

---

## Step 2. Build the Prompt (PromptGenerator)

`PromptGenerator` computes dataset meta-characteristics and assembles them into a structured prompt for the LLM.

```python
from tab_forge.prompt_generator import PromptGenerator

gen = PromptGenerator()

prompt = gen.build_prompt(
    dataset       = dataset,
    target_metric = "r2_metric",   # target optimization metric
    shot_mode     = "few",         # "zero" or "few"
    mfe_features  = "short",       # "short", "full", or list of strings
)

print(prompt)
```

**`target_metric`** — which metric the LLM uses to rank models:

| Value | Metric |
|-------|--------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon divergence |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

**`shot_mode`:**
- `"zero"` — prompt contains only meta-features of your dataset and model descriptions
- `"few"` — additionally includes results of preliminary experiments on 5 reference datasets; the LLM sees how each model performed on similar tasks

---

## Step 3. LLM Model Ranking (LLMRunner)

```python
from tab_forge.llm_runner import LLMRunner

runner = LLMRunner(
    base_url = "https://api.openai.com/v1",   # or local server URL
    api_key  = "sk-...",
    model    = "gpt-4o",
)

result = runner.run(
    prompt      = prompt,
    n_runs      = 5,       # number of independent requests (self-consistency)
    temperature = 0.7,
)

print(result)
# Prints the final ranking and average ranks for each model

print(result.final_ranking)
# ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']
```

`n_runs` — number of independent LLM calls. Ranks from each response are averaged, making the final list robust to random model behavior.

---

## Step 4. Hyperparameter Tuning (AutoTuningStudy)

### Mode `"extended"` — recommended

The search space for the architecture is set automatically. Just pass the model class:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.tuning import AutoTuningStudy

study = AutoTuningStudy(
    model_class       = CTGANSynthesizer,
    search_space_mode = "extended",   # automatic search space
    cv                = 3,            # number of folds
)

study.optimize(dataset, n_trials=25)

print(study.best_params)
print(study.best_value)
```

### Mode `"manual"` — manually defined hyperparameters

```python
from tab_forge.models import GANMFSSynthesizer
from tab_forge.tuning import AutoTuningStudy

def my_params(trial):
    return {
        "epochs":       trial.suggest_int("epochs", 200, 500),
        "generator_lr": trial.suggest_float("generator_lr", 1e-4, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }

study = AutoTuningStudy(
    model_class       = GANMFSSynthesizer,
    get_params        = my_params,
    search_space_mode = "manual",
    cv                = 3,
    direction         = "maximize",   # depends on the chosen metric
)

study.optimize(dataset, n_trials=30)
```

---

## Full Pipeline

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer, GANMFSSynthesizer
from tab_forge.tuning import AutoTuningStudy
from prompt_generator import PromptGenerator
from llm_runner import LLMRunner

# Load data
dataset = Dataset(
    data="abalone.csv",
    target="Rings",
    task_type="regression",
    numerical_features=["Length", "Diameter", "Height",
                        "Whole weight", "Shucked weight"],
    categorical_features=["Sex"],
)

# LLM selects model priority
gen = PromptGenerator()
prompt = gen.build_prompt(dataset=dataset, target_metric="r2_metric", shot_mode="few")

runner = LLMRunner(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o")
result = runner.run(prompt, n_runs=5, temperature=0.7)

print("Model ranking:", result.final_ranking)

# Tune the highest-priority model
MODEL_REGISTRY = {
    "CTGAN":   CTGANSynthesizer,
    "GAN-MFS": GANMFSSynthesizer,
    # ... other models
}

top_model_name = result.final_ranking[0]
model_class = MODEL_REGISTRY.get(top_model_name)

if model_class:
    study = AutoTuningStudy(
        model_class       = model_class,
        search_space_mode = "extended",
        cv                = 3,
    )
    study.optimize(dataset, n_trials=25)
    print(f"Best parameters for {top_model_name}:", study.best_params)
```

---

## Quality Evaluation (Benchmark)

To simply evaluate synthetic data:

```python
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark

model = CTGANSynthesizer(epochs=300)
model.fit(train)

synth = model.structed_generate(len(test))   # returns Dataset

bench = Benchmark([
    ("r2",   {"model": "xgboost"}),
    ("rmse", {"model": "linearregression"}),
    ("js_mean", {}),
])

result = bench.evaluate(synth, test)
print(result)
```

Named metric specification:

```python
bench = Benchmark({
    "r2_xgb":     ("r2",   {"model": "xgboost"}),
    "r2_linear":  ("r2",   {"model": "linearregression"}),
    "js_diverg":  ("js_mean", {}),
})
```

---

## Using a Model Without Tuning

```python
from tab_forge.models import CTGANSynthesizer

model = CTGANSynthesizer(epochs=300)
model.fit(dataset)

# Generate as pd.DataFrame
synth_df = model.generate(n_samples=1000)

# Generate as Dataset (for Benchmark and AutoTuningStudy)
synth_dataset = model.structed_generate(n_samples=1000)
```

---

## Examples

Detailed Jupyter notebooks are located in the `examples/` folder:

| Notebook | What it demonstrates |
|----------|---------------------|
| `test_dataset.ipynb` | Data loading, splitting, utilities |
| `test_model.ipynb` | Training and generation for each architecture |
| `test_benchmark.ipynb` | Synthetic data quality evaluation |
| `test_tuning.ipynb` | `TuningStudy` and `AutoTuningStudy`, Optuna visualization |
| `test_prompt_generator.ipynb` | Prompt construction, zero/few-shot, different meta-feature sets |
| `test_llm_runner.ipynb` | Full cycle: prompt → LLM → ranking |
