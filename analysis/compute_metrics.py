#!/usr/bin/env python3
"""
Compute aggregated metrics from raw results.

Per cell (scenario × effect × tool × posterior_type):
    Avg ATT (%)    = mean(att_pct) × 100
    Bias (pct pts) = (mean(att_pct) - mean(true_att_pct)) × 100
    Coverage       = proportion(ci_lower_level <= true_att_level <= ci_upper_level)
    FNR            = 1 - proportion(significant), effect condition only
    FPR            = proportion(significant), null condition only
    Avg CI Width   = mean(ci_upper_level - ci_lower_level)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_results(base_dir: str = ".") -> pd.DataFrame:
    """Load all raw results from JSONL."""
    results_path = Path(base_dir) / "results" / "raw" / "results.jsonl"

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    n_raw = len(df)

    # Deduplicate: keep last record per key (crash-recovery re-runs append
    # duplicates — the last copy is the valid one for CausalPy NaN→valid pairs,
    # and identical for GeoLift/Google MM).
    df["tool_label"] = df.apply(
        lambda r: f"{r['tool']}_{r['posterior_type']}"
        if r["posterior_type"] else r["tool"],
        axis=1,
    )
    dedup_keys = ["scenario", "effect_label", "iteration", "tool_label"]
    df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    n_dedup = n_raw - len(df)
    if n_dedup > 0:
        print(f"Deduplicated: dropped {n_dedup} duplicate records")

    print(f"Loaded {len(df)} raw results ({n_raw} before dedup)")
    return df


def compute_cell_metrics(group: pd.DataFrame, effect_label: str) -> dict:
    """Compute metrics for one cell (scenario × effect × tool × posterior_type)."""
    n = len(group)
    valid = group.dropna(subset=["att_pct"])
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "n_iterations": n,
            "n_valid": 0,
            "drop_rate": 1.0,
        }

    avg_att_pct = valid["att_pct"].mean() * 100
    avg_att_level = valid["att_level"].mean()
    true_att_pct = valid["true_att_pct"].mean() * 100
    bias = avg_att_pct - true_att_pct

    coverage = valid["coverage"].mean() if "coverage" in valid.columns else np.nan

    # CI width in level-scale (absolute units) — removes confound from different
    # counterfactual denominators across tools. A tool can't appear more precise
    # simply because it overestimates the counterfactual.
    ci_width_level = valid["ci_upper_level"] - valid["ci_lower_level"]
    avg_ci_width = ci_width_level.mean()

    sig_rate = valid["significant"].mean()

    metrics = {
        "n_iterations": n,
        "n_valid": n_valid,
        "drop_rate": 1 - n_valid / n if n > 0 else 0,
        "avg_att_pct": round(avg_att_pct, 2),
        "avg_att_level": round(avg_att_level, 2),
        "true_att_pct": round(true_att_pct, 2),
        "bias_pct_pts": round(bias, 2),
        "coverage": round(coverage, 4),
        "avg_ci_width_level": round(avg_ci_width, 2),
    }

    if effect_label == "effect":
        fnr = round(1 - sig_rate, 4)
        metrics["fnr"] = fnr
    elif effect_label == "null":
        metrics["fpr"] = round(sig_rate, 4)
        metrics["avg_null_att_pct"] = round(avg_att_pct, 2)

    # Monte Carlo standard errors for proportions: SE = sqrt(p*(1-p)/n)
    def _mcse(p, n):
        return round(np.sqrt(p * (1 - p) / n), 4) if n > 0 else np.nan

    metrics["mcse_coverage"] = _mcse(coverage, n_valid)
    if effect_label == "effect":
        metrics["mcse_fnr"] = _mcse(1 - sig_rate, n_valid)
    elif effect_label == "null":
        metrics["mcse_fpr"] = _mcse(sig_rate, n_valid)

    return metrics


def compute_all_metrics(base_dir: str = ".") -> pd.DataFrame:
    """Compute metrics for all cells."""
    df = load_raw_results(base_dir)

    # tool_label already created during load/dedup; ensure it exists
    if "tool_label" not in df.columns:
        df["tool_label"] = df.apply(
            lambda r: f"{r['tool']}_{r['posterior_type']}"
            if r["posterior_type"] else r["tool"],
            axis=1,
        )

    group_cols = ["scenario", "effect_label", "effect_pct", "tool_label"]
    all_metrics = []

    for keys, group in df.groupby(group_cols):
        scenario, effect_label, effect_pct, tool_label = keys
        metrics = compute_cell_metrics(group, effect_label)
        metrics.update({
            "scenario": scenario,
            "effect_label": effect_label,
            "effect_pct": effect_pct,
            "tool_label": tool_label,
        })
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Save
    out_dir = Path(base_dir) / "results" / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved aggregated metrics to {out_path}")
    print(f"\n{metrics_df.to_string()}")

    return metrics_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    compute_all_metrics(args.base_dir)
