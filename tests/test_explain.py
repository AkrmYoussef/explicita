import numpy as np
import pytest

from explicita.explain import explain, ranking, shap_values


def test_shap_adds_up_to_the_model_output(bundle, risky_row):
    """base value + sum of SHAP weights == model log-odds."""
    weights, base = shap_values(bundle, risky_row)
    p = bundle.predict_proba(risky_row)[0]
    assert base + sum(weights.values()) == pytest.approx(np.log(p / (1 - p)), abs=1e-4)


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_explanation_finds_the_real_drivers(bundle, risky_row, method):
    top = ranking(explain(bundle, risky_row, method))[:2]
    assert set(top) == {"duration", "checking_status"}


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_drivers_push_towards_rejection(bundle, risky_row, method):
    weights = explain(bundle, risky_row, method)
    assert weights["duration"] < 0
    assert weights["checking_status"] < 0


def test_lime_is_repeatable(bundle, risky_row):
    assert explain(bundle, risky_row, "lime") == explain(bundle, risky_row, "lime")


def test_unknown_method(bundle, risky_row):
    with pytest.raises(ValueError):
        explain(bundle, risky_row, "magic")
