"""Load and prepare the German Credit dataset.

Each row is one person who asked a bank for a loan.
The target column `approved` is 1 for a "good" credit risk and 0 for a "bad" one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "approved"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "german_credit.csv"

# Features a person cannot (or should not be asked to) change.
# Counterfactuals never suggest changes to these.
IMMUTABLE_FEATURES = ["age", "personal_status", "foreign_worker", "num_dependents"]


def load_german_credit(cache: bool = True) -> pd.DataFrame:
    """Return the German Credit data as a DataFrame.

    The first call downloads the data from OpenML and saves a copy in data/.
    Later calls read the local copy.
    """
    if DATA_FILE.exists():
        # keep_default_na=False keeps category values like "none" as text.
        return pd.read_csv(DATA_FILE, keep_default_na=False)

    from sklearn.datasets import fetch_openml

    raw = fetch_openml("credit-g", version=1, as_frame=True)
    df = raw.frame.copy()
    df[TARGET] = (df.pop("class").astype(str) == "good").astype(int)

    categorical, _ = split_features(df.drop(columns=[TARGET]))
    for col in categorical:
        df[col] = df[col].astype(str)

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_FILE, index=False)
    return df


def split_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into (categorical, numeric), keeping their original order."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return categorical, numeric
