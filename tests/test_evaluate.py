import pytest

from explicita.evaluate import agreement, evaluate, faithfulness, stability


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_explanations_are_faithful(bundle, risky_row, method):
    result = faithfulness(bundle, risky_row, method, k=2)
    assert result["passed"]
    assert result["top_k_change"] > result["random_k_change"]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_stability_is_between_0_and_1(bundle, risky_row, method):
    score = stability(bundle, risky_row, method)["mean_top_k_overlap"]
    assert 0.0 <= score <= 1.0


def test_shap_and_lime_agree_on_clear_data(bundle, risky_row):
    assert agreement(bundle, risky_row, k=2)["top_k_overlap"] == 1.0


def test_evaluate_returns_all_checks(bundle, risky_row):
    assert set(evaluate(bundle, risky_row)) == {"faithfulness", "stability", "agreement"}
