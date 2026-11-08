"""The model and everything the explainers need to know about it."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import TARGET


def build_pipeline(categorical: list[str], numeric: list[str], seed: int = 0) -> Pipeline:
    """Ordinal-encode categories and keep numbers as they are.

    Ordinal encoding keeps one column per original feature, so every
    explanation maps directly to a feature the user understands.
    """
    prep = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


@dataclass
class ModelBundle:
    """A trained pipeline plus the training data the explainers need."""

    pipeline: Pipeline
    categorical: list[str]
    numeric: list[str]
    train_df: pd.DataFrame  # raw training rows, including the target column
    metrics: dict[str, float] = field(default_factory=dict)
    # Explainers are slow to build, so we keep them here. Not saved to disk.
    cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ---- column helpers -------------------------------------------------
    @property
    def input_columns(self) -> list[str]:
        """Columns in the order the raw input uses."""
        return [c for c in self.train_df.columns if c != TARGET]

    @property
    def feature_names(self) -> list[str]:
        """Columns in the order the model sees them (after preprocessing)."""
        return self.categorical + self.numeric

    @property
    def classifier(self) -> LGBMClassifier:
        return self.pipeline.named_steps["clf"]

    @property
    def encoder(self) -> OrdinalEncoder:
        return self.pipeline.named_steps["prep"].named_transformers_["cat"]

    @cached_property
    def X_train(self) -> pd.DataFrame:
        return self.train_df[self.input_columns]

    @cached_property
    def X_train_transformed(self) -> np.ndarray:
        return self.transform(self.X_train)

    # ---- model calls ----------------------------------------------------
    def to_frame(self, record: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Turn one input record into a one-row DataFrame with the right dtypes."""
        df = record.copy() if isinstance(record, pd.DataFrame) else pd.DataFrame([record])
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        df = df[self.input_columns]
        for col in self.categorical:
            df[col] = df[col].astype(str)
        for col in self.numeric:
            df[col] = pd.to_numeric(df[col])
        return df

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.named_steps["prep"].transform(X), dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of approval (class 1) for each row."""
        return self.pipeline.predict_proba(X)[:, 1]

    # ---- data summaries -------------------------------------------------
    @cached_property
    def baseline_row(self) -> pd.DataFrame:
        """A "typical" applicant: median for numbers, most common value for categories."""
        values = {c: self.X_train[c].median() for c in self.numeric}
        values.update({c: self.X_train[c].mode().iloc[0] for c in self.categorical})
        return self.to_frame(values)

    def schema(self) -> dict[str, Any]:
        """Describe every input feature. The UI uses this to build its form."""
        features = []
        for col in self.input_columns:
            if col in self.categorical:
                features.append(
                    {
                        "name": col,
                        "type": "categorical",
                        "options": sorted(self.X_train[col].astype(str).unique().tolist()),
                        "default": str(self.baseline_row[col].iloc[0]),
                    }
                )
            else:
                s = self.X_train[col]
                features.append(
                    {
                        "name": col,
                        "type": "numeric",
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "default": float(self.baseline_row[col].iloc[0]),
                        "integer": bool(np.all(np.equal(np.mod(s, 1), 0))),
                    }
                )
        return {"target": TARGET, "features": features}

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "categorical": self.categorical,
                "numeric": self.numeric,
                "train_df": self.train_df,
                "metrics": self.metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        return cls(**joblib.load(path))
