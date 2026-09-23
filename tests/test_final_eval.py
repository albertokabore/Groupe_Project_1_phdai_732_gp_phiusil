"""Tests for the Part 2 metric helpers. No dataset needed; run with `python -m pytest`.

The helpers carry the positive-class convention (phishing = 0), which is the
bug Part 1 had to fix twice, so the tests pin it against sklearn directly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config as C
from src.final_eval import (
    FPR_BUDGET,
    bootstrap_ci,
    choose_thresholds,
    classification_metrics,
    group_error_rates,
    predict_at,
)
from src.tuning import PARAM_GRIDS, _default_params, build_tuning_models

PHISH, LEGIT = C.POS_LABEL, 1 - C.POS_LABEL


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.choice([PHISH, LEGIT], size=2000, p=[0.4, 0.6])
    # Informative but imperfect scores for P(phishing).
    p = np.clip(np.where(y == PHISH, 0.75, 0.25) + rng.normal(0, 0.2, len(y)), 0, 1)
    return y, p


def test_predict_at_marks_high_scores_as_phishing():
    assert list(predict_at(np.array([0.9, 0.5, 0.1]), 0.5)) == [PHISH, PHISH, LEGIT]


def test_metrics_match_sklearn_with_phishing_positive(toy):
    y, p = toy
    pred = predict_at(p, 0.5)
    m = classification_metrics(y, p)
    assert m["precision"] == pytest.approx(precision_score(y, pred, pos_label=PHISH))
    assert m["recall"] == pytest.approx(recall_score(y, pred, pos_label=PHISH))
    assert m["f1"] == pytest.approx(f1_score(y, pred, pos_label=PHISH))
    assert m["phishing_caught"] + m["phishing_missed"] == int((y == PHISH).sum())
    assert m["legit_flagged"] + m["legit_passed"] == int((y == LEGIT).sum())
    assert m["phishing_miss_rate"] == pytest.approx(1 - m["recall"])


def test_probability_errors_are_consistent(toy):
    y, p = toy
    m = classification_metrics(y, p)
    assert m["probability_rmse"] ** 2 == pytest.approx(m["brier"])
    assert 0 <= m["probability_mae"] <= m["probability_rmse"] <= 1


def test_perfect_scores():
    y = np.array([PHISH, PHISH, LEGIT, LEGIT])
    m = classification_metrics(y, np.array([1.0, 0.9, 0.1, 0.0]))
    assert m["f1"] == 1 and m["roc_auc"] == 1 and m["legit_false_alarm_rate"] == 0


def test_bootstrap_interval_brackets_point_estimate(toy):
    y, p = toy
    pred = predict_at(p, 0.5)
    ci = bootstrap_ci(y, pred, n=500)
    point = f1_score(y, pred, pos_label=PHISH)
    assert ci["f1"][0] < point < ci["f1"][1]
    assert ci["f1"][1] - ci["f1"][0] < 0.1


def test_fpr_budget_threshold_respects_budget(toy):
    y, p = toy
    t = choose_thresholds(y, p)
    flagged = np.mean(predict_at(p[y == LEGIT], t["fpr_budget"]) == PHISH)
    assert flagged <= FPR_BUDGET
    assert t["default"] == 0.5
    assert 0 < t["f1_optimal"] < 1


def test_group_rates_suppress_small_groups():
    frame = pd.DataFrame({
        "g": ["a"] * 300 + ["b"] * 10,
        "label": [LEGIT] * 300 + [LEGIT] * 10,
        "pred": [PHISH] * 3 + [LEGIT] * 297 + [PHISH] * 10,
    })
    out = group_error_rates(frame, "g").set_index("g")
    assert out.loc["a", "legit_false_alarm_rate"] == pytest.approx(0.01)
    assert np.isnan(out.loc["b", "legit_false_alarm_rate"])


def test_cv_pr_auc_reads_phishing_column_for_trees(toy):
    """Regression: a dict of make_scorer objects scored trees on P(legitimate)."""
    from sklearn.metrics import average_precision_score
    from sklearn.tree import DecisionTreeClassifier

    from src.tuning import SCORE_NAMES, scoring

    y, p = toy
    X = np.column_stack([p, np.random.default_rng(1).normal(size=len(p))])
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    scores = scoring(tree, X, y)
    assert set(scores) == set(SCORE_NAMES)
    p_phish = tree.predict_proba(X)[:, list(tree.classes_).index(PHISH)]
    assert scores["pr_auc_phishing"] == pytest.approx(
        average_precision_score(y == PHISH, p_phish))
    assert scores["pr_auc_phishing"] > 0.9


def test_every_grid_contains_the_default_configuration():
    roster = build_tuning_models()
    for name, grid in PARAM_GRIDS.items():
        for param, value in _default_params(roster[name], grid).items():
            assert value in grid[param], f"{name}.{param}={value} not searched"
