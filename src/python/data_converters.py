#!/usr/bin/env python3
"""
Data format converters from canonical long format to tool-specific formats.

Canonical format (parquet): geo, date, Y, Y_counterfactual
"""

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def to_causalpy_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to wide format for CausalPy.

    Returns: DataFrame with time as index, cities as columns.
    """
    wide = df.pivot(index="date", columns="geo", values="Y")
    wide.index.name = "date"
    return wide


def to_google_mm_format(
    df: pd.DataFrame,
    treated_units: list[str],
    treatment_start: int = 91,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Convert to Google matched_markets TBR format.

    TBR.fit() expects: date (datetime64), geo, response, period, group, cost.
    group: 1=control, 2=treatment. period: 0=pre, 1=test.
    """
    mm_df = df[["geo", "date", "Y"]].copy()
    mm_df = mm_df.rename(columns={"Y": "response"})

    # Convert integer dates to datetime64 (TBR requires datetime)
    base_date = pd.Timestamp(start_date)
    mm_df["date"] = mm_df["date"].apply(lambda d: base_date + pd.Timedelta(days=int(d) - 1))

    # Period: 0=pre-treatment, 1=test
    mm_df["period"] = (df["date"] >= treatment_start).astype(int)

    # Group: 1=control, 2=treatment (GroupSemantics defaults)
    mm_df["group"] = mm_df["geo"].apply(
        lambda g: 2 if g in treated_units else 1
    )

    # Cost: placeholder (required by TBR)
    mm_df["cost"] = 0.0

    # Encode geo as integers (TBR accepts str or int)
    geo_map = {g: i + 1 for i, g in enumerate(sorted(df["geo"].unique()))}
    mm_df["geo"] = mm_df["geo"].map(geo_map)

    return mm_df


def to_geolift_format(
    df: pd.DataFrame,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Convert to GeoLift format.

    Renames columns to location/date/Y, converts date integers to date strings.
    """
    gl_df = df[["geo", "date", "Y"]].copy()
    gl_df = gl_df.rename(columns={"geo": "location"})

    # Convert integer dates to "yyyy-mm-dd" strings
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    gl_df["date"] = gl_df["date"].apply(
        lambda d: (base_date + timedelta(days=int(d) - 1)).strftime("%Y-%m-%d")
    )

    return gl_df


def verify_data_identity(
    original: pd.DataFrame,
    causalpy_df: pd.DataFrame,
    mm_df: pd.DataFrame,
    geolift_df: pd.DataFrame,
    treated_units: list[str],
) -> bool:
    """Verify all three formats produce identical Y values."""
    # Original Y values sorted
    orig_y = np.sort(original["Y"].values)

    # CausalPy: unpivot back
    cp_y = np.sort(causalpy_df.values.flatten())

    # Google MM: response column
    mm_y = np.sort(mm_df["response"].values)

    # GeoLift: Y column
    gl_y = np.sort(geolift_df["Y"].values)

    h_orig = hashlib.sha256(orig_y.tobytes()).hexdigest()
    h_cp = hashlib.sha256(cp_y.tobytes()).hexdigest()
    h_mm = hashlib.sha256(mm_y.tobytes()).hexdigest()
    h_gl = hashlib.sha256(gl_y.tobytes()).hexdigest()

    all_match = h_orig == h_cp == h_mm == h_gl
    if not all_match:
        print(f"Hash mismatch! orig={h_orig[:12]} cp={h_cp[:12]} "
              f"mm={h_mm[:12]} gl={h_gl[:12]}")
    return all_match
