#!/usr/bin/env python3
"""
Google matched_markets (Python) wrapper — TBR regression.

Uses the published google/matched_markets TBR class.
TBR fits OLS on aggregated treatment ~ control series over pre-period,
projects counterfactual into test period. CIs from t-distribution.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from matched_markets.methodology.tbr import TBR


@dataclass
class GoogleMMResult:
    att_level: float
    att_pct: float
    ci_lower: float
    ci_upper: float
    ci_lower_level: float
    ci_upper_level: float
    significant: bool
    r_squared: float
    durbin_watson: float
    runtime_seconds: float


def load_tool_config(base_dir: str = ".") -> dict:
    with open(f"{base_dir}/config/tools.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["google_mm"]


def _durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    return float(np.sum(diff**2) / np.sum(residuals**2))


def run_google_mm(
    mm_df: pd.DataFrame,
    config: dict = None,
    base_dir: str = ".",
) -> GoogleMMResult:
    if config is None:
        config = load_tool_config(base_dir)

    start_time = time.time()

    try:
        tbr_model = TBR(use_cooldown=config.get("use_cooldown", False))
        tbr_model.fit(
            mm_df,
            target="response",
            key_response="response",
            key_cost="cost",
            key_geo="geo",
            key_period="period",
            key_group="group",
            key_date="date",
            group_control=1,
            group_treatment=2,
            period_pre=0,
            period_test=1,
        )

        confidence_level = config["confidence_level"]
        tails = config["tails"]

        summary = tbr_model.summary(
            level=confidence_level,
            threshold=0.0,
            tails=tails,
            report="last",
        )

        # Cumulative estimate and CI bounds
        estimate_cum = float(summary["estimate"].iloc[0])
        ci_lower_cum = float(summary["lower"].iloc[0])
        ci_upper_cum = float(summary["upper"].iloc[0])

        # Diagnostics
        ols_model = tbr_model.pre_period_model
        r_squared = float(ols_model.rsquared)
        resid = ols_model.resid
        pre_residuals = resid.values if hasattr(resid, "values") else np.asarray(resid)
        dw = _durbin_watson(pre_residuals)

        # Convert cumulative ATT to average ATT and ATT%
        # Get treatment group observed post-period totals
        causal_effect = tbr_model.causal_effect(periods=(1,))
        n_post = len(causal_effect)

        # analysis_data: multi-index (group, date), column 'response'
        ad = tbr_model.analysis_data
        treatment_post = ad.loc[2].loc[ad.loc[2]["period"] == 1, "response"]
        observed_treatment_total = treatment_post.sum()

        # Counterfactual = observed - causal effect (cumulative)
        counterfactual_total = observed_treatment_total - estimate_cum
        avg_cf = counterfactual_total / n_post

        # Average ATT per period
        att_level = estimate_cum / n_post

        # Level CIs
        ci_lower_level = ci_lower_cum / n_post
        ci_upper_level = ci_upper_cum / n_post

        # ATT% and CI%
        att_pct = att_level / avg_cf if avg_cf != 0 else np.nan
        ci_lower_pct = ci_lower_level / avg_cf if avg_cf != 0 else np.nan
        ci_upper_pct = ci_upper_level / avg_cf if avg_cf != 0 else np.nan

        # Handle inf from one-sided (shouldn't happen with tails=2)
        if np.isinf(ci_upper_pct):
            ci_upper_pct = np.nan
        if np.isinf(ci_lower_pct):
            ci_lower_pct = np.nan

        significant = bool(
            (not np.isnan(ci_lower_pct) and ci_lower_pct > 0) or
            (not np.isnan(ci_upper_pct) and ci_upper_pct < 0)
        )

        elapsed = time.time() - start_time
        return GoogleMMResult(
            att_level=att_level, att_pct=att_pct,
            ci_lower=ci_lower_pct, ci_upper=ci_upper_pct,
            ci_lower_level=ci_lower_level, ci_upper_level=ci_upper_level,
            significant=significant, r_squared=r_squared,
            durbin_watson=dw, runtime_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  Google MM error: {e}")
        return GoogleMMResult(
            att_level=np.nan, att_pct=np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            ci_lower_level=np.nan, ci_upper_level=np.nan,
            significant=False, r_squared=np.nan,
            durbin_watson=np.nan, runtime_seconds=elapsed,
        )
