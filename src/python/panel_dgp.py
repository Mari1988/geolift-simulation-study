#!/usr/bin/env python3
"""
Python port of the DGP in src/R/generate_panels.R, for interactive/notebook use.

This does not need to bit-match R's RNG — it only needs to reproduce the same
statistical structure (log-normal geo baselines, shared trend, weekly seasonality,
AR(1) noise with sqrt scaling, multiplicative treatment effect) documented there.
Use it for exploration and teaching; the committed results in results/ come from
the R script, not this module.

Canonical output format (long): geo, date, Y, Y_counterfactual, treated
"""

import numpy as np
import pandas as pd

DOW_PROFILE = [-1.0, -0.5, 0.0, 0.2, 0.8, 1.0, 0.5]  # Mon..Sun

DEFAULTS = dict(
    n_treated=1,
    n_control=20,
    baseline_mean=4000,
    baseline_spread=0.6,
    trend_slope=0.001,
    seasonality_amplitude=0.10,
    dow_profile=DOW_PROFILE,
    autocorrelation=0.30,
    noise_level=0.20,
    outlier_multiplier=5.0,
    total_days=105,
    pre_days=90,
)

# Each scenario is a set of overrides on top of DEFAULTS. Mirrors the
# `scenarios` list in src/R/generate_panels.R.
SCENARIOS = {
    "A1": {},
    "A2": {"outlier": True},
    "A3": {"n_control": 9},
    "A4": {"total_days": 45, "pre_days": 30},
}


def draw_baselines(n_geos, baseline_mean, baseline_spread, seed) -> pd.Series:
    """Log-normal geo baselines, sorted ascending, named "City 1".."City N"."""
    rng = np.random.default_rng(seed)
    meanlog = np.log(baseline_mean) - baseline_spread**2 / 2
    baselines = rng.lognormal(mean=meanlog, sigma=baseline_spread, size=n_geos)
    baselines = np.sort(baselines)
    names = [f"City {i + 1}" for i in range(n_geos)]
    return pd.Series(baselines, index=names)


def select_treated(baselines: pd.Series, n_treated: int) -> dict:
    """Pick the geo(s) closest to the median baseline."""
    median_val = baselines.median()
    diffs = (baselines - median_val).abs()
    treated_names = diffs.sort_values(kind="stable").index[:n_treated].tolist()
    treated_idx = [baselines.index.get_loc(name) for name in treated_names]
    return {"treated_names": treated_names, "treated_idx": treated_idx}


def generate_panel(
    baselines: pd.Series,
    treated_idx: list[int],
    effect_pct: float,
    total_days: int,
    pre_days: int,
    trend_slope: float,
    seasonality_amplitude: float,
    dow_profile: list[float],
    autocorrelation: float,
    noise_level: float,
    seed: int,
    noise_baselines: pd.Series = None,
) -> pd.DataFrame:
    """Generate one panel of synthetic geo-level time series.

    DGP for each geo i at time t:
      Y_cf[i,t] = baseline_i * trend_t * season_t * exp(noise_level * scale_i * ar_noise[i,t])
      scale_i = sqrt(noise_baselines_i / mean(noise_baselines))
    Treatment (post-period, treated geos only): Y[i,t] = Y_cf[i,t] * (1 + effect_pct)
    """
    rng = np.random.default_rng(seed)

    if noise_baselines is None:
        noise_baselines = baselines
    mean_baseline = noise_baselines.mean()

    geo_names = baselines.index.tolist()
    n_geos = len(geo_names)
    t_seq = np.arange(1, total_days + 1)

    trend = 1 + trend_slope * t_seq
    dow_idx = (t_seq - 1) % 7
    season = 1 + seasonality_amplitude * np.array(dow_profile)[dow_idx]

    Y_cf = np.empty((total_days, n_geos))
    Y = np.empty((total_days, n_geos))

    for i, geo in enumerate(geo_names):
        innovations = rng.standard_normal(total_days)
        noise = np.empty(total_days)
        noise[0] = innovations[0]
        for t in range(1, total_days):
            noise[t] = autocorrelation * noise[t - 1] + innovations[t]

        scale_i = np.sqrt(noise_baselines.iloc[i] / mean_baseline)
        Y_cf[:, i] = baselines.iloc[i] * trend * season * np.exp(noise_level * scale_i * noise)
        Y[:, i] = Y_cf[:, i]

    post_mask = t_seq > pre_days
    for idx in treated_idx:
        Y[post_mask, idx] = Y_cf[post_mask, idx] * (1 + effect_pct)

    df = pd.DataFrame({
        "geo": np.repeat(geo_names, total_days),
        "date": np.tile(t_seq, n_geos),
        "Y": Y.T.flatten(),
        "Y_counterfactual": Y_cf.T.flatten(),
    })
    treated_names = {geo_names[i] for i in treated_idx}
    df["treated"] = df["geo"].isin(treated_names)
    return df


def generate_scenario_panel(
    scenario_id: str,
    effect_pct: float = 0.075,
    seed: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Generate one panel for a named scenario (A1-A4).

    Returns (df, metadata) where metadata has treated_units, treatment_start,
    treatment_end.
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_id!r}; expected one of {list(SCENARIOS)}")

    overrides = SCENARIOS[scenario_id]
    params = {**DEFAULTS, **overrides}
    n_geos = params["n_treated"] + params["n_control"]
    is_outlier = bool(overrides.get("outlier", False))

    baselines = draw_baselines(n_geos, params["baseline_mean"], params["baseline_spread"], seed=seed)
    sel = select_treated(baselines, params["n_treated"])

    # Keep pre-inflation baselines for noise scaling so A2's outlier inflation
    # doesn't also amplify noise amplitude for the treated geo.
    noise_baselines = baselines.copy()
    if is_outlier:
        for idx in sel["treated_idx"]:
            baselines.iloc[idx] *= params["outlier_multiplier"]

    df = generate_panel(
        baselines=baselines,
        treated_idx=sel["treated_idx"],
        effect_pct=effect_pct,
        total_days=params["total_days"],
        pre_days=params["pre_days"],
        trend_slope=params["trend_slope"],
        seasonality_amplitude=params["seasonality_amplitude"],
        dow_profile=params["dow_profile"],
        autocorrelation=params["autocorrelation"],
        noise_level=params["noise_level"],
        noise_baselines=noise_baselines,
        seed=seed + 100_000,
    )

    metadata = {
        "scenario": scenario_id,
        "treated_units": sel["treated_names"],
        "treatment_start": params["pre_days"] + 1,
        "treatment_end": params["total_days"],
        "effect_pct": effect_pct,
    }
    return df, metadata
