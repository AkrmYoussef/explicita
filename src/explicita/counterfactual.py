"""Counterfactual explanations with DiCE.

A counterfactual answers: "What is a small change that flips the decision?"
Example: "If the loan duration were 12 months instead of 36, you would be approved."
"""

from __future__ import annotations

from typing import Any

import dice_ml
import pandas as pd

from .data import IMMUTABLE_FEATURES, TARGET
from .model import ModelBundle

def _dice(bundle: ModelBundle) -> dice_ml.Dice:
    if "dice" not in bundle.cache:
        data = dice_ml.Data(
            dataframe=bundle.train_df,
            continuous_features=bundle.numeric,
            outcome_name=TARGET,
        )
        model = dice_ml.Model(model=bundle.pipeline, backend="sklearn")
        bundle.cache["dice"] = dice_ml.Dice(data, model, method="random")
    return bundle.cache["dice"]


def counterfactuals(
    bundle: ModelBundle,
    row: pd.DataFrame,
    total: int = 3,
    immutable: list[str] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return up to `total` changes that flip the model's decision.

    Features in `immutable` (default: age, personal status, ...) are never changed.
    """
    immutable = IMMUTABLE_FEATURES if immutable is None else immutable
    to_vary = [c for c in bundle.input_columns if c not in immutable]

    result = _dice(bundle).generate_counterfactuals(
        row,
        total_CFs=total,
        desired_class="opposite",
        features_to_vary=to_vary,
        random_seed=seed,
        verbose=False,
    )
    cf_df = result.cf_examples_list[0].final_cfs_df
    if cf_df is None or cf_df.empty:
        return []

    original = row.iloc[0]
    out = []
    for _, cf in cf_df.iterrows():
        cf_row = bundle.to_frame(cf.drop(labels=[TARGET], errors="ignore").to_dict())
        changes = [
            {"feature": col, "from": _plain(original[col]), "to": _plain(cf_row[col].iloc[0])}
            for col in bundle.input_columns
            if not _same(original[col], cf_row[col].iloc[0])
        ]
        out.append(
            {
                "changes": changes,
                "new_probability": round(float(bundle.predict_proba(cf_row)[0]), 4),
            }
        )
    return out


def _same(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _plain(value: Any) -> Any:
    """Convert numpy values to plain Python so they can be sent as JSON."""
    return value.item() if hasattr(value, "item") else value
