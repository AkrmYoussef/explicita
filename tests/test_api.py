import pytest
from fastapi.testclient import TestClient

from explicita.api import app, get_bundle


@pytest.fixture
def client(bundle):
    app.dependency_overrides[get_bundle] = lambda: bundle
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def body(risky_row):
    return {"features": risky_row.iloc[0].to_dict()}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_schema(client, bundle):
    data = client.get("/schema").json()
    assert len(data["features"]) == len(bundle.input_columns)


def test_predict(client, body):
    data = client.post("/predict", json=body).json()
    assert data["approved"] is False
    assert 0 <= data["probability"] < 0.5


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_explain(client, body, method):
    response = client.post(f"/explain?method={method}", json=body)
    assert response.status_code == 200
    contributions = response.json()["contributions"]
    assert contributions[0]["feature"] in {"duration", "checking_status"}


def test_counterfactuals(client, body):
    data = client.post("/counterfactuals?total=2", json=body).json()
    assert data["counterfactuals"]


def test_evaluate(client, body):
    data = client.post("/evaluate", json=body).json()
    assert "agreement" in data


def test_missing_feature_gives_422(client):
    assert client.post("/predict", json={"features": {"age": 30}}).status_code == 422
