"""Feature-importance explanations for one prediction: SHAP and LIME.

Both methods return the same shape: a dict {feature: weight}.
A positive weight pushes the model towards "approved".
A negative weight pushes it towards "rejected".
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from .model import ModelBundle

Weights = dict[str, float]


# ---- SHAP -------------------------------------------------------------------
def _shap_explainer(bundle: ModelBundle) -> shap.TreeExplainer:
    if "shap" not in bundle.cache:
        bundle.cache["shap"] = shap.TreeExplainer(bundle.classifier)
    return bundle.cache["shap"]


def _positive_class(values, expected):
    """SHAP returns different shapes for binary models in different versions."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    expected = np.atleast_1d(expected)
    base = float(expected[1] if expected.size > 1 else expected[0])
    return values, base


def shap_values(bundle: ModelBundle, row: pd.DataFrame) -> tuple[Weights, float]:
    """SHAP weights (in log-odds) and the base value for one row.

    Property: base_value + sum(weights) == the model's log-odds for this row.
    """
    values, base = _positive_class(*_shap_raw(bundle, row))
    return dict(zip(bundle.feature_names, map(float, values[0]))), base


def _shap_raw(bundle: ModelBundle, row: pd.DataFrame):
    explainer = _shap_explainer(bundle)
    values = explainer.shap_values(bundle.transform(row))
    return values, explainer.expected_value


def shap_weights(bundle: ModelBundle, row: pd.DataFrame, **_) -> Weights:
    return shap_values(bundle, row)[0]


# ---- LIME -------------------------------------------------------------------
def _lime_explainer(bundle: ModelBundle, seed: int) -> LimeTabularExplainer:
    if "lime" not in bundle.cache:
        n_cat = len(bundle.categorical)
        bundle.cache["lime"] = LimeTabularExplainer(
            training_data=bundle.X_train_transformed,
            feature_names=bundle.feature_names,
            categorical_features=list(range(n_cat)),
            categorical_names={
                i: [str(v) for v in cats] for i, cats in enumerate(bundle.encoder.categories_)
            },
            class_names=["rejected", "approved"],
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )
    return bundle.cache["lime"]


def lime_weights(
    bundle: ModelBundle, row: pd.DataFrame, num_samples: int = 2000, seed: int = 0, **_
) -> Weights:
    """LIME weights for one row, for all features."""
    explainer = _lime_explainer(bundle, seed)
    # Re-seed for every call so the same input gives the same explanation.
    rng = np.random.RandomState(seed)
    explainer.random_state = rng
    explainer.base.random_state = rng
    if explainer.discretizer is not None:
        explainer.discretizer.random_state = rng
    explanation = explainer.explain_instance(
        bundle.transform(row)[0],
        bundle.classifier.predict_proba,
        labels=(1,),
        num_features=len(bundle.feature_names),
        num_samples=num_samples,
    )
    weights = {name: 0.0 for name in bundle.feature_names}
    for idx, weight in explanation.as_map()[1]:
        weights[bundle.feature_names[idx]] = float(weight)
    return weights


# ---- shared helpers ---------------------------------------------------------
METHODS: dict[str, Callable[..., Weights]] = {"shap": shap_weights, "lime": lime_weights}


def explain(bundle: ModelBundle, row: pd.DataFrame, method: str = "shap", **kwargs) -> Weights:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of: {sorted(METHODS)}")
    return METHODS[method](bundle, row, **kwargs)


def ranking(weights: Weights) -> list[str]:
    """Features sorted from most to least important (by absolute weight)."""
    return sorted(weights, key=lambda f: abs(weights[f]), reverse=True)
