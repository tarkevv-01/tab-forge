# API Reference — LLM Selection

## Import

```python
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner, RunnerResult, SingleRun
```

---

## `PromptGenerator`

```
PromptGenerator(
    experiment_dir: Optional[str] = None,
    datasets_dir: Optional[str] = None,
)
```

Builds a structured prompt for the LLM based on dataset meta-characteristics.

When initialized without arguments, uses built-in directories with experimental data (`prompt_generator/experiment_results/`) and reference datasets (`prompt_generator/datasets/`).

**Parameters:**

- `experiment_dir` — path to the directory with experiment results (optional)
- `datasets_dir` — path to the directory with reference datasets (optional)

!!! warning ""
    On initialization, the presence of `experiment_results/` and `datasets/` folders is verified. If they are not found — an exception is raised.

---

### `PromptGenerator.build_prompt`

```
build_prompt(
    dataset: Dataset,
    target_metric: str = "r2_metric",
    shot_mode: str = "zero",
    mfe_features: Union[str, List[str]] = "short",
    mfe_groups: Optional[List[str]] = None,
    models_to_include: Optional[List[str]] = None,
) -> str
```

Builds a text prompt for the LLM.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | — | Dataset to analyze |
| `target_metric` | `str` | `"r2_metric"` | Target metric for ranking |
| `shot_mode` | `str` | `"zero"` | `"zero"` or `"few"` |
| `mfe_features` | `str` or `list` | `"short"` | Meta-feature set: `"short"`, `"full"`, or list of strings |
| `mfe_groups` | `list[str]` | `None` | pymfe groups (e.g., `["statistical", "info-theory"]`) |
| `models_to_include` | `list[str]` | all 6 | Limit the set of models in the prompt |

**Returns:** a prompt string ready to send to the LLM.

**Valid `target_metric` values:**

| Value | Metric |
|-------|--------|
| `"r2_metric"` | R² |
| `"rmse_metric"` | RMSE |
| `"jensen_shannon_metric"` | Jensen–Shannon Divergence |
| `"lf_metric"` | Frobenius Correlation |
| `"mi_matrix_metric"` | Frobenius MI |

!!! example ""

    ```python
    gen = PromptGenerator()
    prompt = gen.build_prompt(
        dataset       = dataset,
        target_metric = "r2_metric",
        shot_mode     = "few",
        mfe_features  = "short",
    )
    print(f"Prompt length: {len(prompt)} characters")
    ```

---

## `LLMRunner`

```
LLMRunner(
    base_url: str,
    api_key: str,
    model: str,
    retry_on_parse_error: bool = False,
    request_delay: float = 0.0,
)
```

Sends the prompt to an OpenAI-compatible API and aggregates results from multiple requests (self-consistency).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | — | API endpoint URL (`"https://api.openai.com/v1"`) |
| `api_key` | `str` | — | API key |
| `model` | `str` | — | Model name (`"gpt-4o"`, `"llama3.1:8b"`, etc.) |
| `retry_on_parse_error` | `bool` | `False` | Retry the request if the LLM response is unparseable |
| `request_delay` | `float` | `0.0` | Delay between requests (seconds) |

---

### `LLMRunner.run`

```
run(
    prompt: str,
    n_runs: int = 1,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    seed: Optional[int] = None,
) -> RunnerResult
```

Performs `n_runs` independent LLM requests and aggregates rankings.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Prompt text |
| `n_runs` | `1` | Number of independent requests (self-consistency) |
| `temperature` | `0.7` | Generation temperature (0 = deterministic) |
| `top_p` | `1.0` | Top-p (nucleus) sampling |
| `max_tokens` | `2048` | Maximum number of tokens in the response |
| `seed` | `None` | Seed for reproducibility (if supported by the LLM) |

**Returns:** `RunnerResult`

!!! example ""

    ```python
    runner = LLMRunner(
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o",
    )
    result = runner.run(prompt, n_runs=5, temperature=0.7)
    print(result.final_ranking)
    ```

---

## `RunnerResult`

Dataclass with results of all runs.

| Attribute | Type | Description |
|-----------|------|-------------|
| `runs` | `List[SingleRun]` | Details of each run |
| `average_ranks` | `dict` | Average rank of each model |
| `final_ranking` | `List[str]` | List of models ordered by average rank |
| `n_runs` | `int` | Requested number of runs |
| `n_parsed` | `int` | Successfully parsed responses |
| `model` | `str` | Name of the LLM used |
| `prompt_chars` | `int` | Prompt length in characters |

!!! example ""

    ```python
    print(result.final_ranking)
    # ['GAN-MFS', 'CTGAN', 'WGAN-GP', 'CTABGAN+', 'TVAE', 'DDPM']

    print(result.average_ranks)
    # {'GAN-MFS': 1.4, 'CTGAN': 1.8, 'WGAN-GP': 3.0, ...}

    print(f"Successfully parsed: {result.n_parsed}/{result.n_runs}")
    ```

---

## `SingleRun`

Dataclass with details of a single run.

| Attribute | Type | Description |
|-----------|------|-------------|
| `run_index` | `int` | Run index |
| `raw_response` | `str` | Raw LLM response text |
| `ranking` | `List[str]` | Parsed model ranking |
| `parsed_ok` | `bool` | Whether the response was successfully parsed |

---

## Utility Functions

### `extract_meta_features`

```python
from tab_forge.prompt_generator import extract_meta_features

features = extract_meta_features(
    df           = dataset.get_data(),
    target_col   = "Rings",
    mfe_groups   = None,
    mfe_features = "short",
)
print(features)
# {'nr_inst': 4177, 'nr_attr': 8, 'nr_num': 7, 'missing_pct': 0.0, ...}
```

Computes dataset meta-characteristics. Uses both manual calculations (basic statistics) and `pymfe` for extended features.
