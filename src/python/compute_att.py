#!/usr/bin/env python3
"""
ATT normalization — fair comparison protocol.

  1. Coverage checks use level-scale CIs (not %-scale).
  2. ATT% for reporting uses the true counterfactual as denominator
     (same for all tools), not each tool's estimated counterfactual.
"""

import numpy as np
import pandas as pd


def compute_true_att(
    panel_df: pd.DataFrame,
    treated_units: list[str],
    treatment_start: int = 91,
) -> tuple[float, float, float]:
    """Compute true ATT at level and percentage scale.

    Returns:
        true_att_level: mean(Y - Y_counterfactual) for treated in post-period
        true_att_pct:   true_att_level / mean(Y_counterfactual)
        true_cf_mean:   mean(Y_counterfactual) for treated in post-period
    """
    post = panel_df[
        (panel_df["date"] >= treatment_start)
        & (panel_df["geo"].isin(treated_units))
    ]

    tau = post["Y"].values - post["Y_counterfactual"].values
    cf_mean = float(post["Y_counterfactual"].mean())

    att_level = float(np.mean(tau))
    att_pct = att_level / cf_mean if cf_mean != 0 else 0.0

    return att_level, att_pct, cf_mean


def check_coverage(
    ci_lower: float,
    ci_upper: float,
    true_value: float,
) -> bool:
    """Does the CI contain the true value? Works at any scale (level or %)."""
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return False
    return ci_lower <= true_value <= ci_upper
