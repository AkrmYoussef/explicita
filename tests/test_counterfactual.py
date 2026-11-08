from explicita.counterfactual import counterfactuals


def test_counterfactuals_flip_the_decision(bundle, risky_row):
    results = counterfactuals(bundle, risky_row, total=3)
    assert results, "expected at least one counterfactual"
    for cf in results:
        assert cf["new_probability"] >= 0.5
        assert cf["changes"]


def test_counterfactuals_keep_immutable_features(bundle, risky_row):
    for cf in counterfactuals(bundle, risky_row, total=3):
        changed = {c["feature"] for c in cf["changes"]}
        assert not changed & {"age", "personal_status"}
