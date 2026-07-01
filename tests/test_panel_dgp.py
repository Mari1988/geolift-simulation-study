"""Unit tests for the Python DGP port used by the notebook."""

import numpy as np

from src.python.panel_dgp import generate_scenario_panel


def test_output_columns():
    df, meta = generate_scenario_panel("A1", effect_pct=0.075, seed=1)
    assert set(df.columns) == {"geo", "date", "Y", "Y_counterfactual", "treated"}
    assert set(meta) == {"scenario", "treated_units", "treatment_start", "treatment_end", "effect_pct"}


def test_treated_flag_matches_metadata():
    df, meta = generate_scenario_panel("A1", seed=1)
    treated_geos = set(df.loc[df["treated"], "geo"].unique())
    assert treated_geos == set(meta["treated_units"])
    assert len(treated_geos) == 1


def test_pre_period_equals_counterfactual():
    """No treatment effect should be applied before treatment_start."""
    df, meta = generate_scenario_panel("A1", effect_pct=0.075, seed=1)
    pre = df[(df["date"] < meta["treatment_start"]) & df["treated"]]
    np.testing.assert_allclose(pre["Y"].values, pre["Y_counterfactual"].values)


def test_post_period_lift_matches_effect_pct():
    """Treated geo's post-period Y should equal Y_counterfactual * (1 + effect_pct)."""
    effect_pct = 0.075
    df, meta = generate_scenario_panel("A1", effect_pct=effect_pct, seed=1)
    post = df[(df["date"] >= meta["treatment_start"]) & df["treated"]]
    ratio = post["Y"] / post["Y_counterfactual"]
    np.testing.assert_allclose(ratio.values, 1 + effect_pct, rtol=1e-10)


def test_control_geos_unaffected_by_treatment():
    df, meta = generate_scenario_panel("A1", effect_pct=0.075, seed=1)
    controls = df[~df["treated"]]
    np.testing.assert_allclose(controls["Y"].values, controls["Y_counterfactual"].values)


def test_scenario_panel_sizes():
    df_a1, _ = generate_scenario_panel("A1", seed=1)
    df_a3, _ = generate_scenario_panel("A3", seed=1)
    df_a4, meta_a4 = generate_scenario_panel("A4", seed=1)

    assert df_a1["geo"].nunique() == 21  # 1 treated + 20 control
    assert df_a3["geo"].nunique() == 10  # 1 treated + 9 control
    assert df_a4["date"].max() == 45
    assert meta_a4["treatment_start"] == 31


def test_a2_outlier_inflates_treated_baseline():
    """A2's treated geo should have a much larger average level than A1's."""
    df_a1, meta_a1 = generate_scenario_panel("A1", seed=1)
    df_a2, meta_a2 = generate_scenario_panel("A2", seed=1)

    treated_a1 = meta_a1["treated_units"][0]
    treated_a2 = meta_a2["treated_units"][0]
    mean_a1 = df_a1.loc[df_a1["geo"] == treated_a1, "Y_counterfactual"].mean()
    mean_a2 = df_a2.loc[df_a2["geo"] == treated_a2, "Y_counterfactual"].mean()

    assert mean_a2 > 3 * mean_a1


def test_unknown_scenario_raises():
    try:
        generate_scenario_panel("Z9")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown scenario")
