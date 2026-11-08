"""Test fixtures.

Tests use a small synthetic dataset, so they run fast and need no internet.
The target depends mostly on `duration` and `checking_status`, so we know
which features a good explanation should find.
"""

import numpy as np
import pandas as pd
import pytest

from explicita.data import TARGET
from explicita.train import train


def make_synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "checking_status": rng.choice(["<0", "0<=X<200", "no checking"], n),
            "purpose": rng.choice(["car", "education", "furniture"], n),
            "personal_status": rng.choice(["single", "married"], n),
            "duration": rng.integers(6, 60, n),
            "credit_amount": rng.integers(500, 15000, n),
            "age": rng.integers(19, 70, n),
        }
    )
    score = (
        -0.12 * (df["duration"] - 30)
        + np.where(df["checking_status"] == "no checking", 2.5, 0.0)
        + np.where(df["checking_status"] == "<0", -2.5, 0.0)
        + rng.normal(0, 0.5, n)
    )
    df[TARGET] = (score > 0).astype(int)
    return df


@pytest.fixture(scope="session")
def bundle():
    return train(make_synthetic())


@pytest.fixture
def risky_row(bundle):
    """An applicant the model should reject."""
    return bundle.to_frame(
        {
            "checking_status": "<0",
            "purpose": "car",
            "personal_status": "single",
            "duration": 48,
            "credit_amount": 8000,
            "age": 30,
        }
    )
