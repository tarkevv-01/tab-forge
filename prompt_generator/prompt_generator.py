"""
Synthetic Data Model Recommender — Prompt Generator
====================================================

Folder layout (everything must sit next to this file):

    prompt_generator.py          ← this file
    experiment_results/
        CTGAN/
            abalone.txt
            ...
        WGAN-GP/ ...
        GAN-MFS/ ...
        CTABGAN+/ ...
        TVAE/ ...
        DDPM/ ...
    datasets/
        abalone.csv
        ...

Usage
-----
    from prompt_generator import PromptGenerator

    gen = PromptGenerator()           # directories resolved automatically

    prompt = gen.build_prompt(
        dataset       = my_dataset,   # Dataset object (from dataset.py)
        target_metric = "r2_metric",

        # Shot mode: 'zero' or 'few'
        shot_mode     = "zero",       # default

        # Meta-feature selection:
        #   'short'    — curated KEEP_MFS list (default)
        #   'full'     — all features from mfe_groups
        #   list[str]  — custom list of pymfe feature names
        mfe_features  = "short",

        mfe_groups    = ["general", "statistical"],

        # Optional model filter
        models_to_include = None,
    )
    print(prompt)

few-shot reference datasets (fixed):
    ['abalone', 'cl-housing', 'air', 'wind', 'gats']
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Resolve sibling directories relative to this file
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

_DEFAULT_EXPERIMENT_DIR = _HERE / "experiment_results"
_DEFAULT_DATASETS_DIR   = _HERE / "datasets"

# ─────────────────────────────────────────────────────────────────────────────
# Meta-feature presets
# ─────────────────────────────────────────────────────────────────────────────

KEEP_MFS = [
    "nr_num", "missing_pct", "task_type", "abs_corr_max",
    "skewness_mean", "kurtosis_mean", "abs_corr_mean",
    "nr_inst", "target_n_unique",
    "nr_attr", "std_mean", "kurtosis_std",
]

FEW_SHOT_DATASETS = ["abalone", "cl-housing", "air", "wind", "gats"]

FEW_SHOT_TARGETS = {
    "abalone":    "Rings",
    "cl-housing": "target",
    "air":        "CO(GT)",
    "wind":       "MAL",
    "gats":       "gd_downlink_multicast",
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS = ["CTGAN", "WGAN-GP", "GAN-MFS", "CTABGAN+", "TVAE", "DDPM"]

METRIC_DIRECTION = {
    "r2_metric":             "higher is better",
    "lf_metric":             "lower is better",
    "jensen_shannon_metric": "lower is better",
    "mi_matrix_metric":      "lower is better",
    "rmse_metric":           "lower is better",
}

MODEL_DESCRIPTIONS = {
    "DDPM": (
        "Generative framework (Tab-DDPM) based on iterative denoising. "
        "Particularly effective on datasets with heavy-tailed feature distributions, "
        "high inter-feature correlations, and heterogeneous marginal shapes — "
        "conditions under which the diffusion process can capture joint structure "
        "that adversarial and VAE-based objectives tend to smooth over."
    ),
    "CTGAN": (
        "Conditional GAN for tabular data using mode-specific normalization and conditional sampling. "
        "A well-established baseline that handles categorical imbalance through resampling, "
        "though performance can degrade on datasets with strong feature dependencies."
    ),
    "WGAN-GP": (
        "Wasserstein GAN with gradient penalty enforcing Lipschitz continuity. "
        "Provides stable adversarial training with consistent convergence behavior "
        "across a range of tabular dataset types."
    ),
    "GAN-MFS": (
        "WGAN-GP variant that adds a meta-feature distribution matching objective to the generator loss. "
        "During training, meta-features (e.g. mean, variance, skewness, kurtosis) are computed over "
        "random subsamples (variates) of both real and generated data, forming empirical distributions "
        "over each statistic. The generator is then penalized via Wasserstein distance between these "
        "real and synthetic meta-feature distributions — not pointwise column statistics, but their "
        "spread across subsamples. Supports per-feature lambda weighting. Well-suited when preserving "
        "marginal statistical properties across the feature space is a priority."
    ),
    "CTABGAN+": (
        "Extended conditional GAN with auxiliary classification and regression heads "
        "for mixed-type columns. The auxiliary objectives add a regularizing effect "
        "on the generator, which can improve synthesis quality on datasets with "
        "predominantly numerical features and relatively uniform marginal shapes."
    ),
    "TVAE": (
        "Variational Autoencoder with a tabular-adapted ELBO. Encodes data into a continuous latent space "
        "enabling smooth synthesis; tends to produce stable results with reasonable fidelity "
        "across numerical and low-cardinality categorical features."
    ),
}

METRIC_SHORT_DESCRIPTIONS = {
    "r2_metric":             "R² score of XGBoost trained on synthetic data and tested on real data (higher is better).",
    "lf_metric":             "Frobenius norm between correlation matrices of synthetic and real data (lower is better).",
    "jensen_shannon_metric": (
        "Mean Jensen-Shannon divergence across marginal feature distributions "
        "of synthetic vs real data (lower is better). "
        "Purely per-column metric: reflects how faithfully each individual feature's "
        "shape, spread, and modes are reproduced — skewness, heavy tails, multimodality. "
        "Models that handle heterogeneous marginal shapes and non-Gaussian distributions "
        "well will have an advantage here."
    ),
    "mi_matrix_metric":      "Frobenius norm between mutual information matrices of synthetic vs real data (lower is better).",
    "rmse_metric": (
        "Train-on-synthetic, test-on-real RMSE using XGBoost (lower is better). "
        "Measures how accurately a model trained on synthetic data predicts real targets "
        "in absolute units. Sensitive to datasets where the target has high variance or "
        "heavy tails — models that better capture the joint feature-target distribution "
        "in extreme regions will score better."
    ),
}

EXPERIMENT_METHOD_INFO = """\
## EXPERIMENT METHODOLOGY

For each generative model on each dataset, independent hyperparameter tuning
was conducted using Optuna (TPE sampler, 25 trials per run).

Each full cycle (= 1 iteration):
  1. Data split: 70% train / 30% test, random_state=42.
  2. For each of the 5 quality metrics an independent Optuna run is performed
     on the train split with 3-fold cross-validation.
  3. Best hyperparameters are used to retrain on the full train set;
     all 5 metrics are then evaluated on the test set.
  4. One cycle → 5 result sets: "tuned by metric X → all 5 metric values".

The full cycle is repeated N=5 times to assess stability.

How to read the reference dataset results:
  For each (dataset, tuning metric) pair, values are collected across 5
  iterations, then reported as:
    best  — best single value across 5 runs
    mean ± std — mean and standard deviation across 5 runs
"""

META_FEATURE_EXPLANATIONS = """
Basic structural features:
  nr_inst        — number of rows (instances) in the dataset
  nr_attr        — number of feature columns (excluding target)
  nr_num         — number of numerical (continuous) feature columns
  nr_cat         — number of categorical feature columns
  missing_pct    — percentage of missing values across all cells (0 = no missing)
  task_type      — inferred task type: 'regression' if target has >20 unique values,
                   otherwise 'classification'
  target_n_unique — number of unique values in the target column

Distribution statistics (computed on numerical features):
  skewness_mean  — mean skewness across all numerical features;
                   values far from 0 indicate heavy-tailed or asymmetric distributions
  skewness_std   — standard deviation of per-feature skewness;
                   high values mean features differ strongly in shape
  kurtosis_mean  — mean excess kurtosis; high positive values indicate sharp peaks
                   and heavy tails (leptokurtic); negative values mean flat distributions
  kurtosis_std   — standard deviation of per-feature kurtosis
  std_mean       — mean standard deviation across numerical features (spread of values)
  std_std        — heterogeneity of spread across features
  mean_mean      — mean of feature means (overall magnitude of values)

Correlation features (computed on numerical features):
  abs_corr_mean  — mean absolute pairwise Pearson correlation between features;
                   high values indicate strong linear dependencies
  abs_corr_max   — maximum absolute pairwise correlation (strongest single pair)

pymfe group — general:
  Covers counts of rows, columns, classes, missing values, and ratio features.
  Key features: nr_inst, nr_attr, nr_num, nr_cat, nr_missing_values,
  nr_class (for classification), inst_to_attr (rows-to-columns ratio).

pymfe group — statistical:
  Covers distributional properties of numerical features.
  Key features: mean, sd, skewness, kurtosis, cor (pairwise correlations),
  cov (covariance), mad (mean absolute deviation), range, iq_range.
  Each feature is summarised with .mean and .sd suffixes across columns.

pymfe group — info-theory:
  Information-theoretic properties.
  Key features: attr_ent (feature entropy), class_ent (target entropy),
  eq_num_attr, joint_ent, mut_inf, ns_ratio.

pymfe group — model-based:
  Properties extracted from a fitted decision tree.
  Key features: tree_depth, tree_imbalance, tree_shape, nodes_per_attr,
  leaves_branch, leaves_corrob, leaves_homo.

pymfe group — landmarking:
  Performance of simple baseline algorithms used as meta-features.
  Key features: accuracy or R² of 1-NN, naive Bayes, linear discriminant,
  decision stump. Useful for estimating task difficulty.

pymfe group — complexity:
  Dataset complexity / class separability.
  Key features: f1 (Fisher's discriminant ratio), f2, f3 (feature overlap),
  n1 (fraction of boundary points), n2 (ratio of intra/inter-class distance),
  c1 (entropy of class proportions), t2 (average number of features per point).
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: resolve mfe_features argument
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mfe_features(mfe_features: Union[str, list, None]) -> Optional[list]:
    """
    Translate the user-facing mfe_features argument into what pymfe expects.

    'short'   -> KEEP_MFS  (curated subset)
    'full'    -> None      (pymfe returns everything for the given groups)
    list[str] -> that list (passed straight to pymfe)
    """
    if mfe_features == "short":
        return KEEP_MFS
    if mfe_features == "full":
        return None
    if isinstance(mfe_features, list):
        return mfe_features
    raise ValueError(
        f"mfe_features must be 'short', 'full', or a list of feature names; "
        f"got: {mfe_features!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset meta-features
# ─────────────────────────────────────────────────────────────────────────────

def extract_meta_features(
    df:           pd.DataFrame,
    target_col:   Optional[str],
    mfe_groups:   list,
    mfe_features: Optional[list],   # already resolved: list or None (= full)
) -> dict:
    """
    Compute hand-crafted + optional pymfe meta-features.

    Parameters
    ----------
    df           : DataFrame (registered features + target)
    target_col   : name of the target column
    mfe_groups   : pymfe groups to extract
    mfe_features : list of feature names to keep, or None for all
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=[np.number]).columns.tolist()
    feat_num = [c for c in numeric_cols if c != target_col]
    feat_cat = [c for c in cat_cols     if c != target_col]

    # ── Hand-crafted features (always computed in full first) ─────────────────
    meta = {
        "nr_inst":     int(len(df)),
        "nr_attr":     int(len(df.columns) - (1 if target_col else 0)),
        "nr_num":      int(len(feat_num)),
        "nr_cat":      int(len(feat_cat)),
        "missing_pct": round(float(df.isnull().mean().mean()) * 100, 2),
    }

    if feat_num:
        X = df[feat_num].fillna(df[feat_num].median())
        meta["skewness_mean"] = round(float(X.skew().mean()), 4)
        meta["skewness_std"]  = round(float(X.skew().std()), 4)
        meta["kurtosis_mean"] = round(float(X.kurtosis().mean()), 4)
        meta["kurtosis_std"]  = round(float(X.kurtosis().std()), 4)
        meta["std_mean"]      = round(float(X.std().mean()), 4)
        meta["std_std"]       = round(float(X.std().std()), 4)
        meta["mean_mean"]     = round(float(X.mean().mean()), 4)

        if len(feat_num) > 1:
            corr = X.corr().abs().values.copy()
            np.fill_diagonal(corr, np.nan)
            meta["abs_corr_mean"] = round(float(np.nanmean(corr)), 4)
            meta["abs_corr_max"]  = round(float(np.nanmax(corr)), 4)

    if target_col and target_col in df.columns:
        n_unique = int(df[target_col].nunique())
        meta["target_n_unique"] = n_unique
        meta["task_type"] = "regression" if n_unique > 20 else "classification"

    # ── pymfe (optional) ──────────────────────────────────────────────────────
    try:
        from pymfe.mfe import MFE

        X_mfe = df[feat_num].fillna(df[feat_num].median()).values \
                if feat_num else np.zeros((len(df), 1))
        y_mfe = df[target_col].values \
                if (target_col and target_col in df.columns) else np.zeros(len(df))

        kwargs = {"groups": mfe_groups, "suppress_warnings": True}
        if mfe_features:
            kwargs["features"] = mfe_features

        mfe = MFE(**kwargs)
        mfe.fit(X_mfe, y_mfe)
        names, values = mfe.extract(suppress_warnings=True)

        added = 0
        for n, v in zip(names, values):
            if isinstance(v, (int, float)) and np.isfinite(v):
                if n not in meta:
                    meta[n] = round(float(v), 6)
                    added += 1
        print(f"    pymfe added {added} features (groups: {mfe_groups})")

    except Exception as e:
        print(f"    pymfe: {e}")

    # ── Filter hand-crafted keys when a specific subset was requested ─────────
    # When mfe_features is a list (covers both 'short' and custom list cases),
    # drop any hand-crafted keys that are not in that list.
    if mfe_features is not None:
        keep_set = set(mfe_features)
        meta = {k: v for k, v in meta.items() if k in keep_set}

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parsing a single statistics txt file
# ─────────────────────────────────────────────────────────────────────────────

def parse_statistics(path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    result = {}

    match = re.search(
        r"=== Summary \(best \+ mean\) for dataset: \S+ ===\n(.+)",
        text, re.DOTALL
    )
    if not match:
        return result

    block = match.group(1)
    parts = re.split(r"Tuning metric: (\w+)\n", block)

    it = iter(parts[1:])
    for tuning_metric, section in zip(it, it):
        result[tuning_metric] = {}
        for metric_name, best, mean, std in re.findall(
            r"(\w+): ([\d.]+) \(([\d.]+) ± ([\d.]+)\)",
            section
        ):
            result[tuning_metric][metric_name] = {
                "best": float(best),
                "mean": float(mean),
                "std":  float(std),
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main class
# ─────────────────────────────────────────────────────────────────────────────

class PromptGenerator:
    """
    Generates a zero-shot or few-shot ranking prompt for synthetic data models.

    The ``experiment_results/`` and ``datasets/`` directories are expected to
    sit next to this file — no path configuration required.

    Parameters
    ----------
    (none — directories are resolved automatically from __file__)
    """

    def __init__(self):
        self.experiment_dir = _DEFAULT_EXPERIMENT_DIR
        self.datasets_dir   = _DEFAULT_DATASETS_DIR

        if not self.experiment_dir.exists():
            raise FileNotFoundError(
                f"experiment_results/ not found at: {self.experiment_dir}\n"
                "Place it next to prompt_generator.py."
            )
        if not self.datasets_dir.exists():
            raise FileNotFoundError(
                f"datasets/ not found at: {self.datasets_dir}\n"
                "Place it next to prompt_generator.py."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def build_prompt(
        self,
        dataset,                                            # Dataset object
        target_metric:       str              = "r2_metric",
        shot_mode:           str              = "zero",     # 'zero' | 'few'
        mfe_features:        Union[str, list] = "short",    # 'short'|'full'|list
        mfe_groups:          list             = None,
        models_to_include:   Optional[list]   = None,
    ) -> str:
        """
        Build the ranking prompt.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object. ``get_registered_data()`` supplies the dataframe
            and ``summary()['target']`` supplies the target column name.
        target_metric : str
            One of: r2_metric, lf_metric, jensen_shannon_metric,
                    mi_matrix_metric, rmse_metric.
        shot_mode : {'zero', 'few'}
            'zero' — prompt contains only the user dataset meta-features.
            'few'  — prompt also includes the fixed FEW_SHOT_DATASETS examples.
        mfe_features : 'short' | 'full' | list[str]
            'short'   — use the curated KEEP_MFS subset.
            'full'    — extract all features pymfe can compute for mfe_groups.
            list[str] — use exactly this list of pymfe feature names.
        mfe_groups : list[str], optional
            pymfe groups to extract. Default: ['general', 'statistical'].
        models_to_include : list[str], optional
            Subset of ALL_MODELS to rank. Default: all six models.
        """
        if mfe_groups is None:
            mfe_groups = ["general", "statistical"]

        if shot_mode not in ("zero", "few"):
            raise ValueError(f"shot_mode must be 'zero' or 'few'; got: {shot_mode!r}")

        if target_metric not in METRIC_SHORT_DESCRIPTIONS:
            raise ValueError(
                f"Unknown target_metric '{target_metric}'. "
                f"Choose from: {list(METRIC_SHORT_DESCRIPTIONS)}"
            )

        models       = models_to_include or ALL_MODELS
        resolved_mfe = _resolve_mfe_features(mfe_features)

        # ── Extract data from Dataset object ──────────────────────────────────
        df_new     = dataset.get_registered_data()
        target_col = dataset.summary()["target"]

        # ── Reference datasets (few-shot only) ────────────────────────────────
        available_csvs = {p.stem: p for p in self.datasets_dir.glob("*.csv")}

        if shot_mode == "few":
            hist_datasets = [d for d in FEW_SHOT_DATASETS if d in available_csvs]
            missing = [d for d in FEW_SHOT_DATASETS if d not in available_csvs]
            if missing:
                print(f"  [!] Few-shot datasets not found in datasets/: {missing}")
        else:
            hist_datasets = []

        # ── Load experiment statistics ────────────────────────────────────────
        print("\n[1/4] Loading statistics txt files...")
        model_stats: dict = {}
        for model in models:
            model_dir = self.experiment_dir / model
            if not model_dir.exists():
                print(f"  [!] Directory not found: {model_dir}")
                continue
            model_stats[model] = {}
            for ds_name in hist_datasets:
                candidates = (
                    list(model_dir.glob(f"{ds_name}*.txt")) +
                    list(model_dir.glob(f"*{ds_name}*.txt"))
                )
                if not candidates:
                    all_txts = sorted(model_dir.glob("*.txt"))
                    candidates = [
                        t for t in all_txts
                        if ds_name.lower().replace("_", "") in
                           t.stem.lower().replace("_", "")
                    ]
                if candidates:
                    parsed = parse_statistics(candidates[0])
                    if parsed:
                        model_stats[model][ds_name] = parsed
            loaded = len(model_stats[model])
            print(f"  [OK] {model}: {loaded}/{len(hist_datasets)} datasets")

        # ── Meta-features of reference datasets ───────────────────────────────
        print(f"\n[2/4] Extracting meta-features for {len(hist_datasets)} reference datasets...")
        hist_meta: dict = {}
        for ds_name in hist_datasets:
            csv_path   = available_csvs[ds_name]
            ref_target = FEW_SHOT_TARGETS.get(ds_name)
            print(f"  {ds_name}  (target: {ref_target})...")
            df_h = pd.read_csv(csv_path)
            hist_meta[ds_name] = extract_meta_features(
                df_h, ref_target, mfe_groups, resolved_mfe
            )

        # ── Meta-features of user dataset ─────────────────────────────────────
        print("\n[3/4] Extracting meta-features for user dataset...")
        new_meta = extract_meta_features(df_new, target_col, mfe_groups, resolved_mfe)

        # ── Assemble prompt ───────────────────────────────────────────────────
        mode_label = "few-shot" if shot_mode == "few" else "zero-shot"
        print(f"\n[4/4] Generating prompt (mode: {mode_label})...")

        return self._assemble(
            new_meta      = new_meta,
            models        = models,
            target_metric = target_metric,
            hist_datasets = hist_datasets,
            hist_meta     = hist_meta,
            model_stats   = model_stats,
            mfe_groups    = mfe_groups,
            mfe_features  = resolved_mfe,
            is_few_shot   = (shot_mode == "few"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _assemble(self, **kw) -> str:
        if kw["is_few_shot"]:
            return self._build_few_shot_prompt(**kw)
        return self._build_zero_shot_prompt(**kw)

    # ─────────────────────────────────────────────────────────────────────────
    # Zero-shot prompt
    # ─────────────────────────────────────────────────────────────────────────

    def _build_zero_shot_prompt(self, **kw) -> str:
        new_meta      = kw["new_meta"]
        models        = kw["models"]
        target_metric = kw["target_metric"]
        mfe_groups    = kw["mfe_groups"]
        mfe_features  = kw["mfe_features"]

        sep   = "═" * 72
        parts = []

        parts.append(f"{sep}\n{self._role_block(len(models), 0, False)}")
        parts.append(f"{sep}\n{self._model_block(models)}")
        parts.append(f"{sep}\n{self._metric_block(target_metric)}")
        parts.append(f"{sep}\n{self._meta_legend_block(mfe_groups, mfe_features)}")
        parts.append(
            f"{sep}\n## META-FEATURES OF THE USER DATASET\n\n"
            f"{self._format_meta(new_meta)}"
        )
        parts.append(f"{sep}\n{self._response_format_block(models)}")

        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Few-shot prompt
    # ─────────────────────────────────────────────────────────────────────────

    def _build_few_shot_prompt(self, **kw) -> str:
        new_meta      = kw["new_meta"]
        models        = kw["models"]
        target_metric = kw["target_metric"]
        hist_datasets = kw["hist_datasets"]
        hist_meta     = kw["hist_meta"]
        model_stats   = kw["model_stats"]
        mfe_groups    = kw["mfe_groups"]
        mfe_features  = kw["mfe_features"]

        sep   = "═" * 72
        parts = []

        parts.append(f"{sep}\n{self._role_block(len(models), len(hist_datasets), True)}")
        parts.append(f"{sep}\n{self._model_block(models)}")
        parts.append(f"{sep}\n{self._metric_block(target_metric)}")
        parts.append(f"{sep}\n{self._meta_legend_block(mfe_groups, mfe_features)}")
        parts.append(f"{sep}\n{EXPERIMENT_METHOD_INFO}")
        parts.append(
            f"{sep}\n## META-FEATURES OF THE USER DATASET\n\n"
            f"{self._format_meta(new_meta)}"
        )

        for ds_name in hist_datasets:
            meta  = hist_meta.get(ds_name, {})
            block = self._format_exp_block(
                ds_name, meta, models, model_stats, target_metric
            )
            parts.append(f"{sep}\n## REFERENCE DATASET: {ds_name}\n\n{block}")

        parts.append(f"{sep}\n{self._response_format_block(models)}")

        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _role_block(n_models: int, n_hist: int, is_few_shot: bool) -> str:
        few_shot_line = (
            f"\n  2. Meta-features of {n_hist} reference datasets from the experimental base\n"
            f"  3. Experiment results for each model on those reference datasets\n"
            f"  4. A win summary table across all reference datasets"
            if is_few_shot else ""
        )
        return (
            f"You are an expert in generative models for tabular data.\n\n"
            f"You are given:\n"
            f"  1. Meta-features of the user dataset{few_shot_line}\n\n"
            f"Your goal is to rank the {n_models} generative models listed below "
            f"from most to least likely to perform well on the given dataset "
            f"with respect to the provided target quality metric."
        )

    @staticmethod
    def _model_block(models: list) -> str:
        lines = "\n".join(
            f"  {m}: {MODEL_DESCRIPTIONS.get(m, m)}" for m in models
        )
        return f"## GENERATIVE MODELS\n\n{lines}"

    @staticmethod
    def _metric_block(target_metric: str) -> str:
        desc      = METRIC_SHORT_DESCRIPTIONS[target_metric]
        direction = METRIC_DIRECTION[target_metric]
        return (
            f"## TARGET QUALITY METRIC\n\n"
            f"  {target_metric} ({direction}): {desc}\n\n"
            f"Rank the models by their expected performance on this metric."
        )

    @staticmethod
    def _meta_legend_block(mfe_groups: list, mfe_features: Optional[list]) -> str:
        features_line = (
            f"Specific features: {', '.join(mfe_features)}"
            if mfe_features else
            "Features: all from selected groups"
        )
        return (
            f"## META-FEATURE LEGEND\n\n"
            f"pymfe groups used: {', '.join(mfe_groups)}\n"
            f"{features_line}\n\n"
            f"{META_FEATURE_EXPLANATIONS}"
        )

    @staticmethod
    def _format_meta(meta: dict) -> str:
        base_keys = [
            "nr_inst", "nr_attr", "nr_num", "nr_cat", "missing_pct",
            "skewness_mean", "skewness_std", "kurtosis_mean", "kurtosis_std",
            "std_mean", "std_std", "mean_mean", "abs_corr_mean", "abs_corr_max",
            "target_n_unique", "task_type",
        ]
        lines = ["Basic:"]
        for k in base_keys:
            if k in meta:
                lines.append(f"  {k}: {meta[k]}")

        extra = {k: v for k, v in meta.items() if k not in base_keys}
        if extra:
            lines.append(f"\npymfe ({len(extra)} features):")
            for k, v in sorted(extra.items()):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_exp_block(
        ds_name:       str,
        meta:          dict,
        models:        list,
        model_stats:   dict,
        target_metric: str,
    ) -> str:
        lines = ["Meta-features:"]
        lines.append(PromptGenerator._format_meta(meta))
        lines.append(f"\nExperiment results (target metric: {target_metric}):")

        for model in models:
            ds_data = model_stats.get(model, {}).get(ds_name)
            if not ds_data:
                lines.append(f"\n  {model}: no data")
                continue

            lines.append(f"\n  {model}:")
            metric_results = ds_data.get(target_metric, {})
            if target_metric in metric_results:
                vals = metric_results[target_metric]
                b, m, s = vals["best"], vals["mean"], vals["std"]
                lines.append(
                    f"    tuned by {target_metric:<28} "
                    f"best={b:.6f}  mean={m:.6f} ± {s:.6f}"
                )
            else:
                lines.append(f"    tuned by {target_metric}: no data")

        return "\n".join(lines)

    @staticmethod
    def _response_format_block(models: list) -> str:
        rank_lines = "\n".join(f"  {i+1}. [MODEL_NAME]" for i in range(len(models)))
        return f"""\
## REQUIRED RESPONSE FORMAT

Follow this structure exactly. Do NOT skip any section.

### Section 1 — Dataset characterisation
Describe the dataset based on its meta-features: size, feature types,
distribution shape (skewness, kurtosis), correlations, task type.
Highlight properties relevant to generative model performance.

### Section 2 — Model-by-model reasoning
For each model, reason about how well it fits this dataset, considering:
  — its architecture and known strengths/weaknesses
  — the dataset properties identified in Section 1
  — the target quality metric

Write a short paragraph (2–4 sentences) per model.
Do this for ALL {len(models)} models before moving on.

After completing all model paragraphs, add a subsection:

### Section 2b — Key meta-features driving the ranking
List the 3–5 meta-features that had the strongest influence on your
reasoning. For each one, explain in 1–2 sentences:
  — what value it took in this dataset
  — why that value pushed certain models up or down in the ranking

### Section 3 — Final ranking

After completing Sections 1, 2, and 2b, output the ranking block below.
The block must appear VERBATIM — no extra text inside it, no blank lines
between entries, no scores or comments after the model names.

RANKING_START
{rank_lines}
RANKING_END

Replace each [MODEL_NAME] with one of: {', '.join(models)}.
List all {len(models)} models, best first, no repetitions."""
