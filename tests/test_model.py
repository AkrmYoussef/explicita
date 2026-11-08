import numpy as np

from explicita.model import ModelBundle


def test_model_learns(bundle):
    assert bundle.metrics["roc_auc"] > 0.85


def test_risky_applicant_is_rejected(bundle, risky_row):
    assert bundle.predict_proba(risky_row)[0] < 0.5


def test_save_and_load_give_same_predictions(bundle, risky_row, tmp_path):
    path = tmp_path / "model.joblib"
    bundle.save(path)
    loaded = ModelBundle.load(path)
    np.testing.assert_allclose(loaded.predict_proba(risky_row), bundle.predict_proba(risky_row))


def test_schema_lists_every_feature(bundle):
    names = [f["name"] for f in bundle.schema()["features"]]
    assert names == bundle.input_columns
