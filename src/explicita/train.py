"""Train the loan model and save it.

Run:  python -m explicita.train
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import PROJECT_ROOT, TARGET, load_german_credit, split_features
from .model import ModelBundle, build_pipeline

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def train(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> ModelBundle:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    categorical, numeric = split_features(X)
    for col in categorical:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipeline = build_pipeline(categorical, numeric, seed=seed)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, proba >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    return ModelBundle(pipeline, categorical, numeric, train_df, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="Where to save the model")
    args = parser.parse_args()

    bundle = train(load_german_credit())
    bundle.save(args.out)
    print(f"Saved model to {args.out}")
    print(json.dumps(bundle.metrics, indent=2))


if __name__ == "__main__":
    main()
