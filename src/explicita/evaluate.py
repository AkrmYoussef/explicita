"""Can we trust an explanation? Three simple checks.

1. Faithfulness: if we remove the features the explanation calls important,
   does the prediction really change more than when we remove random features?
2. Stability: if the input changes a tiny bit, does the explanation stay similar?
3. Agreement: do SHAP and LIME point to the same important features?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .explain import explain, ranking
from .model import ModelBundle


def _remove(bundle: ModelBundle, row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """'Remove' features by replacing them with the value of a typical applicant."""
    out = row.copy()
    for f in features:
        out[f] = bundle.baseline_row[f].iloc[0]
    return out


def faithfulness(
    bundle: ModelBundle,
    row: pd.DataFrame,
    method: str = "shap",
    k: int = 3,
    n_random: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Deletion test: compare removing the top-k features with removing k random ones."""
    p0 = float(bundle.predict_proba(row)[0])
    top = ranking(explain(bundle, row, method))[:k]
    top_change = abs(p0 - float(bundle.predict_proba(_remove(bundle, row, top))[0]))

    rng = np.random.default_rng(seed)
    random_changes = []
    for _ in range(n_random):
        pick = list(rng.choice(bundle.input_columns, size=k, replace=False))
        random_changes.append(abs(p0 - float(bundle.predict_proba(_remove(bundle, row, pick))[0])))
    random_change = float(np.mean(random_changes))

    return {
        "method": method,
        "k": k,
        "top_features": top,
        "top_k_change": round(top_change, 4),
        "random_k_change": round(random_change, 4),
        "passed": top_change >= random_change,
    }


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _perturb(bundle: ModelBundle, row: pd.DataFrame, scale: float, rng) -> pd.DataFrame:
    """Add small noise to the numeric features (scale x the feature's std)."""
    out = row.copy()
    for col in bundle.numeric:
        std = float(bundle.X_train[col].std()) or 1.0
        out[col] = float(out[col].iloc[0]) + rng.normal(0, scale * std)
    return out


def stability(
    bundle: ModelBundle,
    row: pd.DataFrame,
    method: str = "shap",
    k: int = 3,
    n_runs: int = 5,
    noise: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Average top-k overlap (Jaccard, 0 to 1) between the original and noisy explanations."""
    rng = np.random.default_rng(seed)
    top = ranking(explain(bundle, row, method))[:k]
    overlaps = [
        _jaccard(top, ranking(explain(bundle, _perturb(bundle, row, noise, rng), method))[:k])
        for _ in range(n_runs)
    ]
    return {"method": method, "k": k, "mean_top_k_overlap": round(float(np.mean(overlaps)), 4)}


def agreement(bundle: ModelBundle, row: pd.DataFrame, k: int = 3) -> dict[str, Any]:
    """How much SHAP and LIME agree on which features matter."""
    shap_w = explain(bundle, row, "shap")
    lime_w = explain(bundle, row, "lime")
    features = bundle.feature_names
    rho = spearmanr([abs(shap_w[f]) for f in features], [abs(lime_w[f]) for f in features])[0]
    return {
        "k": k,
        "top_k_overlap": round(_jaccard(ranking(shap_w)[:k], ranking(lime_w)[:k]), 4),
        "rank_correlation": round(float(np.nan_to_num(rho)), 4),
    }


def evaluate(bundle: ModelBundle, row: pd.DataFrame, k: int = 3) -> dict[str, Any]:
    """Run all three checks for one input."""
    return {
        "faithfulness": {m: faithfulness(bundle, row, m, k=k) for m in ("shap", "lime")},
        "stability": {m: stability(bundle, row, m, k=k) for m in ("shap", "lime")},
        "agreement": agreement(bundle, row, k=k),
    }
