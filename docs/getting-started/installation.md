# Installation

## Requirements

- Python **3.8+**
- pip

!!! warning "Heavy dependencies"
    Tab-Forge uses `torch`, `tensorflow`, `sdv`, and `synthcity` — the total environment size can reach several gigabytes. We recommend creating a separate virtual environment.

---

## Install from Source

Tab-Forge is not yet published on PyPI, so it is installed by cloning the repository:

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

---

## Recommended: Virtual Environment

=== "venv"

    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux / macOS
    source .venv/bin/activate

    git clone https://github.com/tarkevv-01/tab-forge.git
    cd tab-forge
    pip install -r requirements.txt
    ```

=== "conda"

    ```bash
    conda create -n tab-forge python=3.10
    conda activate tab-forge

    git clone https://github.com/tarkevv-01/tab-forge.git
    cd tab-forge
    pip install -r requirements.txt
    ```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | WGAN-GP, GAN-MFS, CTABGAN+ |
| `tensorflow` | CTABGAN+ (some parts) |
| `sdv`, `ctgan` | CTGAN and TVAE |
| `synthcity` | TabDDPM |
| `optuna` | Bayesian optimization |
| `pymfe` | Dataset meta-feature extraction |
| `openai` | LLM API requests |
| `xgboost`, `scikit-learn` | Synthetic data quality evaluation |

---

## Verify Installation

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark
from tab_forge.tuning import AutoTuningStudy
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner

print("Tab-Forge installed successfully!")
```

!!! tip "GPU Support"
    To accelerate training of PyTorch models (WGAN-GP, GAN-MFS, CTABGAN+), make sure the CUDA-enabled version of `torch` is installed:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
