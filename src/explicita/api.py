"""REST API.

Run:  uvicorn explicita.api:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .counterfactual import counterfactuals
from .evaluate import evaluate
from .explain import explain, ranking, shap_values
from .model import ModelBundle
from .train import DEFAULT_MODEL_PATH

app = FastAPI(
    title="Explicita",
    description="Get a loan decision, the reasons for it, and what would change it.",
    version="0.1.0",
)


@lru_cache
def get_bundle() -> ModelBundle:
    path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not os.path.exists(path):
        raise HTTPException(503, f"No model at {path}. Run `make train` first.")
    return ModelBundle.load(path)


class Applicant(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="One value for every input feature. See GET /schema.",
        examples=[{"duration": 24, "credit_amount": 3000, "checking_status": "<0"}],
    )


def _row(bundle: ModelBundle, applicant: Applicant):
    try:
        return bundle.to_frame(applicant.features)
    except (ValueError, TypeError) as err:
        raise HTTPException(422, str(err)) from err


def _decision(bundle: ModelBundle, row) -> dict[str, Any]:
    p = float(bundle.predict_proba(row)[0])
    return {"approved": p >= 0.5, "probability": round(p, 4)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema(bundle: ModelBundle = Depends(get_bundle)) -> dict[str, Any]:
    return {**bundle.schema(), "metrics": bundle.metrics}


@app.post("/predict")
def predict(applicant: Applicant, bundle: ModelBundle = Depends(get_bundle)) -> dict[str, Any]:
    return _decision(bundle, _row(bundle, applicant))


@app.post("/explain")
def explain_endpoint(
    applicant: Applicant,
    method: Literal["shap", "lime"] = "shap",
    bundle: ModelBundle = Depends(get_bundle),
) -> dict[str, Any]:
    row = _row(bundle, applicant)
    weights = explain(bundle, row, method)
    result: dict[str, Any] = {
        "method": method,
        "decision": _decision(bundle, row),
        "contributions": [
            {"feature": f, "value": _plain(row[f].iloc[0]), "weight": round(weights[f], 4)}
            for f in ranking(weights)
        ],
    }
    if method == "shap":
        _, base = shap_values(bundle, row)
        result["base_value"] = round(base, 4)
        result["units"] = "log-odds"
    return result


@app.post("/counterfactuals")
def counterfactuals_endpoint(
    applicant: Applicant, total: int = 3, bundle: ModelBundle = Depends(get_bundle)
) -> dict[str, Any]:
    row = _row(bundle, applicant)
    return {
        "decision": _decision(bundle, row),
        "counterfactuals": counterfactuals(bundle, row, total=total),
    }


@app.post("/evaluate")
def evaluate_endpoint(
    applicant: Applicant, k: int = 3, bundle: ModelBundle = Depends(get_bundle)
) -> dict[str, Any]:
    row = _row(bundle, applicant)
    return {"decision": _decision(bundle, row), **evaluate(bundle, row, k=k)}


def _plain(value: Any) -> Any:
    value = value.item() if hasattr(value, "item") else value
    return None if isinstance(value, float) and math.isnan(value) else value
