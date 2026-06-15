#!/usr/bin/env python3
"""
CausalPy 0.8.0 wrapper — Bayesian Synthetic Control with Dirichlet prior.

Supports two posterior types (configurable via tools.yaml):
  - mu:    parameter uncertainty only (from CausalPy's post_impact)
  - y_hat: parameter + observation noise (from CausalPy's post_pred)

Data is standardized before fitting so that CausalPy's default
HalfNormal(sigma=1) prior on observation noise is well-scaled, then
ATT estimates are converted back to the original scale.
"""

import time
import warnings
from dataclasses import dataclass

import arviz as az
import causalpy as cp
import numpy as np
import pandas as pd
import yaml


@dataclass
class CausalPyResult:
    """Result from a single CausalPy run, for one posterior type."""
    posterior_type: str
    att_level: float
    att_pct: float
    ci_lower: float
    ci_upper: float
    ci_lower_level: float
    ci_upper_level: float
    significant: bool
    converged: bool
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    n_divergences: int
    runtime_seconds: float


def load_tool_config(base_dir: str = ".") -> dict:
    with open(f"{base_dir}/config/tools.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["causalpy"]


def run_causalpy(
    wide_df: pd.DataFrame,
    treated_units: list[str],
    treatment_time: int = 91,
    config: dict = None,
    base_dir: str = ".",
    iteration: int = None,
) -> list[CausalPyResult]:
    """Run CausalPy SC and extract both mu and y_hat posteriors."""
    if config is None:
        config = load_tool_config(base_dir)

    sampler_cfg = config["sampler"]
    inference_cfg = config["inference"]
    convergence_cfg = config["convergence"]
    retry_cfg = config["retry"]

    control_units = [c for c in wide_df.columns if c not in treated_units]

    # Standardize using pre-treatment statistics so CausalPy's default
    # HalfNormal(sigma=1) prior on observation noise is well-scaled.
    pre_mask = wide_df.index <= treatment_time
    means = wide_df.loc[pre_mask].mean()
    stds = wide_df.loc[pre_mask].std()
    stds = stds.replace(0, 1)
    wide_std = (wide_df - means) / stds

    draws = sampler_cfg["draws"]
    warmup = sampler_cfg["warmup"]
    target_accept = sampler_cfg["target_accept"]
    chains = sampler_cfg["chains"]

    result_obj = None
    converged = False
    n_attempts = 0
    max_attempts = retry_cfg["max_retries"] + 1
    rhat_max = np.nan
    ess_bulk_min = 0.0
    ess_tail_min = 0.0
    n_divergences = 0

    start_time = time.time()

    while n_attempts < max_attempts and not converged:
        n_attempts += 1
        sample_kwargs = {
            "draws": draws, "tune": warmup, "chains": chains,
            "target_accept": target_accept, "progressbar": False,
        }
        if iteration is not None:
            sample_kwargs["random_seed"] = iteration
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_obj = cp.SyntheticControl(
                    data=wide_std,
                    treatment_time=treatment_time,
                    treated_units=treated_units,
                    control_units=control_units,
                    model=cp.pymc_models.WeightedSumFitter(
                        sample_kwargs=sample_kwargs,
                    ),
                )
        except Exception as e:
            print(f"  CausalPy error (attempt {n_attempts}): {e}")
            if n_attempts < max_attempts:
                draws *= retry_cfg["rerun_draws_multiplier"]
                target_accept = retry_cfg["rerun_target_accept"]
                continue
            else:
                elapsed = time.time() - start_time
                return _fail_results(config["posteriors"], elapsed)

        idata = result_obj.idata
        try:
            rhat_vals = az.rhat(idata).to_array().values.flatten()
            rhat_max = float(np.nanmax(rhat_vals))
        except Exception:
            rhat_max = 999.0
        try:
            ess_bulk_min = float(np.nanmin(
                az.ess(idata, method="bulk").to_array().values.flatten()))
        except Exception:
            ess_bulk_min = 0.0
        try:
            ess_tail_min = float(np.nanmin(
                az.ess(idata, method="tail").to_array().values.flatten()))
        except Exception:
            ess_tail_min = 0.0
        try:
            n_divergences = int(idata.sample_stats["diverging"].values.sum())
        except Exception:
            n_divergences = 0

        rhat_ok = rhat_max < convergence_cfg["rhat_pass"]
        ess_ok = (
            ess_bulk_min >= convergence_cfg["ess_bulk_min"]
            and ess_tail_min >= convergence_cfg["ess_tail_min"]
        )
        if rhat_ok and ess_ok:
            converged = True
        elif rhat_max < convergence_cfg["rhat_rerun"]:
            draws *= retry_cfg["rerun_draws_multiplier"]
        else:
            draws *= retry_cfg["rerun_draws_multiplier"]
            target_accept = retry_cfg["rerun_target_accept"]

    elapsed = time.time() - start_time

    if result_obj is None:
        return _fail_results(config["posteriors"], elapsed)

    hdi_prob = inference_cfg["hdi_prob"]

    # Unstandardization scale for treated units
    treated_stds = np.array([float(stds[u]) for u in treated_units])

    # Observed post-treatment data on original scale (for ATT%)
    post_mask = wide_df.index > treatment_time
    obs_post = wide_df.loc[post_mask, treated_units].values  # (n_post, n_treated)
    obs_mean = float(obs_post.mean())

    results = []
    for posterior_type in config["posteriors"]:
        try:
            if posterior_type == "mu":
                # post_impact uses mu: parameter uncertainty only
                # dims: (treated_units, chain, draw, obs_ind)
                impact_std = result_obj.post_impact.values
                n_t = len(treated_units)
                impact_flat = impact_std.reshape(n_t, -1, impact_std.shape[-1]).transpose(1, 0, 2)
            else:
                # y_hat: parameter + observation noise
                # dims: (chain, draw, obs_ind, treated_units)
                y_hat_vals = result_obj.post_pred["posterior_predictive"]["y_hat"].values
                obs_std_vals = result_obj.datapost_treated.values
                impact_raw = obs_std_vals[np.newaxis, np.newaxis, :, :] - y_hat_vals
                n_post = impact_raw.shape[2]
                impact_flat = impact_raw.reshape(-1, n_post, len(treated_units)).transpose(0, 2, 1)

            # impact_flat: (n_samples, n_treated, n_post) on standardized scale
            # Unstandardize: multiply each treated unit's impact by its std
            impact_orig = impact_flat * treated_stds[np.newaxis, :, np.newaxis]

            # ATT per sample: mean over treated units and time
            att_level_samples = impact_orig.mean(axis=(1, 2))

            # Counterfactual mean per sample (original scale)
            cf_mean_samples = obs_mean - att_level_samples
            att_pct_samples = att_level_samples / cf_mean_samples

            att_level = float(np.mean(att_level_samples))
            att_pct = float(np.mean(att_pct_samples))

            hdi = az.hdi(att_pct_samples, hdi_prob=hdi_prob)
            ci_lower = float(hdi[0])
            ci_upper = float(hdi[1])

            hdi_level = az.hdi(att_level_samples, hdi_prob=hdi_prob)
            ci_lower_level = float(hdi_level[0])
            ci_upper_level = float(hdi_level[1])

            significant = (ci_lower > 0) or (ci_upper < 0)

            results.append(CausalPyResult(
                posterior_type=posterior_type,
                att_level=att_level, att_pct=att_pct,
                ci_lower=ci_lower, ci_upper=ci_upper,
                ci_lower_level=ci_lower_level, ci_upper_level=ci_upper_level,
                significant=significant, converged=converged,
                rhat_max=rhat_max, ess_bulk_min=ess_bulk_min,
                ess_tail_min=ess_tail_min, n_divergences=n_divergences,
                runtime_seconds=elapsed,
            ))
        except Exception as e:
            print(f"  CausalPy {posterior_type} extraction error: {e}")
            results.append(CausalPyResult(
                posterior_type=posterior_type,
                att_level=np.nan, att_pct=np.nan,
                ci_lower=np.nan, ci_upper=np.nan,
                ci_lower_level=np.nan, ci_upper_level=np.nan,
                significant=False, converged=converged,
                rhat_max=rhat_max, ess_bulk_min=ess_bulk_min,
                ess_tail_min=ess_tail_min, n_divergences=n_divergences,
                runtime_seconds=elapsed,
            ))

    return results


def _fail_results(posteriors: list[str], elapsed: float) -> list[CausalPyResult]:
    return [
        CausalPyResult(
            posterior_type=pt, att_level=np.nan, att_pct=np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            ci_lower_level=np.nan, ci_upper_level=np.nan,
            significant=False, converged=False,
            rhat_max=np.nan, ess_bulk_min=0, ess_tail_min=0,
            n_divergences=0, runtime_seconds=elapsed,
        )
        for pt in posteriors
    ]
