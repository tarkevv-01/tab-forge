# Установка

## Требования

- Python **3.8+**
- pip

!!! warning "Тяжёлые зависимости"
    Tab-Forge использует `torch`, `tensorflow`, `sdv` и `synthcity` — суммарный размер окружения может достигать нескольких гигабайт. Рекомендуем создать отдельное виртуальное окружение.

---

## Установка из исходного кода

Tab-Forge пока не опубликован на PyPI, поэтому устанавливается клонированием репозитория:

```bash
git clone https://github.com/tarkevv-01/tab-forge.git
cd tab-forge
pip install -r requirements.txt
```

---

## Рекомендуемый способ: виртуальное окружение

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

## Ключевые зависимости

| Пакет | Зачем |
|-------|-------|
| `torch` | WGAN-GP, GAN-MFS, CTABGAN+ |
| `tensorflow` | CTABGAN+ (некоторые части) |
| `sdv`, `ctgan` | CTGAN и TVAE |
| `synthcity` | TabDDPM |
| `optuna` | Байесовская оптимизация |
| `pymfe` | Извлечение мета-фич датасета |
| `openai` | Запросы к LLM API |
| `xgboost`, `scikit-learn` | Оценка качества синтетики |

---

## Проверка установки

```python
from tab_forge.dataset import Dataset
from tab_forge.models import CTGANSynthesizer
from tab_forge.benchmark import Benchmark
from tab_forge.tuning import AutoTuningStudy
from tab_forge.prompt_generator import PromptGenerator
from tab_forge.llm_runner import LLMRunner

print("Tab-Forge успешно установлен!")
```

!!! tip "Работа с GPU"
    Для ускорения обучения PyTorch-моделей (WGAN-GP, GAN-MFS, CTABGAN+) убедитесь, что установлена версия `torch` с поддержкой CUDA:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
