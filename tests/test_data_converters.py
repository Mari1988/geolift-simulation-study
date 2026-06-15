"""Unit tests for data format converters."""

import pandas as pd

from src.python.data_converters import (
    to_causalpy_format,
    to_geolift_format,
    to_google_mm_format,
)


def _make_panel(treatment_start: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Build a minimal panel: 1 treated + 1 control, 4 days."""
    rows = []
    for t in range(1, 5):
        rows.append({"geo": "treated_city", "date": t, "Y": 100.0 + t,
                      "Y_counterfactual": 100.0})
        rows.append({"geo": "control_city", "date": t, "Y": 200.0 + t,
                      "Y_counterfactual": 200.0})
    return pd.DataFrame(rows), ["treated_city"]


def test_google_mm_period_boundary():
    """Period flags must split exactly at treatment_start."""
    df, treated = _make_panel(treatment_start=3)
    mm = to_google_mm_format(df, treated, treatment_start=3)

    # Days 1,2 -> period 0 (pre), days 3,4 -> period 1 (test)
    pre = mm[mm["period"] == 0]
    test = mm[mm["period"] == 1]
    assert sorted(pre["date"].dt.day.unique()) == [1, 2]
    assert sorted(test["date"].dt.day.unique()) == [3, 4]


def test_google_mm_group_assignment():
    """Treated geos get group=2, controls get group=1."""
    df, treated = _make_panel()
    mm = to_google_mm_format(df, treated, treatment_start=3)

    geo_groups = mm.groupby("geo")["group"].first()
    # geo IDs are integers; treated_city and control_city get mapped
    # alphabetically: control_city=1, treated_city=2
    treated_geo_id = 2
    control_geo_id = 1
    assert geo_groups[treated_geo_id] == 2
    assert geo_groups[control_geo_id] == 1


def test_google_mm_date_conversion():
    """Integer dates should become consecutive datetime64 values."""
    df, treated = _make_panel()
    mm = to_google_mm_format(df, treated, treatment_start=3,
                              start_date="2020-01-01")

    dates = sorted(mm["date"].unique())
    assert dates[0] == pd.Timestamp("2020-01-01")
    assert dates[1] == pd.Timestamp("2020-01-02")
    assert dates[3] == pd.Timestamp("2020-01-04")


def test_geolift_column_names():
    """GeoLift format must have 'location' (not 'geo') and 'Y'."""
    df, _ = _make_panel()
    gl = to_geolift_format(df)
    assert "location" in gl.columns
    assert "geo" not in gl.columns
    assert "Y" in gl.columns


def test_geolift_date_format():
    """Dates should be yyyy-mm-dd strings."""
    df, _ = _make_panel()
    gl = to_geolift_format(df, start_date="2020-01-01")
    assert gl["date"].iloc[0] == "2020-01-01"
    assert gl["date"].iloc[2] == "2020-01-02"


def test_causalpy_shape():
    """Wide format: rows=dates, columns=geos, index=date."""
    df, _ = _make_panel()
    wide = to_causalpy_format(df)
    assert wide.shape == (4, 2)  # 4 days, 2 geos
    assert wide.index.name == "date"
    assert "treated_city" in wide.columns
    assert "control_city" in wide.columns


if __name__ == "__main__":
    test_google_mm_period_boundary()
    test_google_mm_group_assignment()
    test_google_mm_date_conversion()
    test_geolift_column_names()
    test_geolift_date_format()
    test_causalpy_shape()
    print("All tests passed.")
