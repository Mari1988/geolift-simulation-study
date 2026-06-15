"""Unit tests for the core ATT normalization logic."""

import numpy as np
import pandas as pd

from src.python.compute_att import check_coverage, compute_true_att


def _make_panel(
    treated_y: list[float],
    treated_cf: list[float],
    treatment_start: int = 3,
) -> pd.DataFrame:
    """Build a minimal panel with one treated unit, one control."""
    n_total = len(treated_y)
    rows = []
    for t in range(1, n_total + 1):
        rows.append({"geo": "treated", "date": t,
                      "Y": treated_y[t - 1],
                      "Y_counterfactual": treated_cf[t - 1]})
        rows.append({"geo": "control", "date": t,
                      "Y": 100.0, "Y_counterfactual": 100.0})
    return pd.DataFrame(rows)


def test_compute_true_att_with_effect():
    # Pre: days 1-2, Post: days 3-4
    # Treated CF = 100 in post, Treated Y = 107.5 in post -> 7.5% effect
    panel = _make_panel(
        treated_y=[100, 100, 107.5, 107.5],
        treated_cf=[100, 100, 100, 100],
        treatment_start=3,
    )
    level, pct, cf_mean = compute_true_att(panel, ["treated"], treatment_start=3)
    assert abs(level - 7.5) < 1e-10, f"level={level}, expected 7.5"
    assert abs(pct - 0.075) < 1e-10, f"pct={pct}, expected 0.075"
    assert abs(cf_mean - 100.0) < 1e-10, f"cf_mean={cf_mean}, expected 100.0"


def test_compute_true_att_null():
    # No effect: Y == Y_counterfactual in post
    panel = _make_panel(
        treated_y=[100, 100, 100, 100],
        treated_cf=[100, 100, 100, 100],
        treatment_start=3,
    )
    level, pct, cf_mean = compute_true_att(panel, ["treated"], treatment_start=3)
    assert abs(level) < 1e-10, f"level={level}, expected 0.0"
    assert abs(pct) < 1e-10, f"pct={pct}, expected 0.0"


def test_check_coverage_level_contains():
    assert check_coverage(ci_lower=5.0, ci_upper=10.0, true_value=7.5) is True


def test_check_coverage_level_misses():
    assert check_coverage(ci_lower=8.0, ci_upper=10.0, true_value=7.5) is False


def test_check_coverage_nan():
    assert check_coverage(ci_lower=np.nan, ci_upper=10.0, true_value=7.5) is False


if __name__ == "__main__":
    test_compute_true_att_with_effect()
    test_compute_true_att_null()
    test_check_coverage_level_contains()
    test_check_coverage_level_misses()
    test_check_coverage_nan()
    print("All tests passed.")
