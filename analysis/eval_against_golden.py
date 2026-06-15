#!/usr/bin/env python3
"""
Eval framework: compare current metrics against a golden reference.

Usage: python analysis/eval_against_golden.py [--base-dir .]

If no golden reference exists at results/golden/metrics.csv, prints current
metrics and exits with code 0 (no comparison to make).
"""

import sys
from pathlib import Path

import pandas as pd


# Per-metric tolerances (from spec)
TOLERANCES = {
    "avg_att_level": {"type": "relative", "value": 0.05},     # +-5% of golden
    "avg_att_pct":   {"type": "absolute", "value": 0.50},     # +-0.50 pct pts
    "bias_pct_pts":  {"type": "absolute", "value": 0.50},     # +-0.50 pct pts
    "coverage":      {"type": "absolute", "value": 0.03},     # +-3pp
    "fnr":           {"type": "absolute", "value": 0.03},     # +-3pp
    "fpr":           {"type": "absolute", "value": 0.02},     # +-2pp
    "avg_ci_width_pct_pts": {"type": "absolute", "value": 1.00},  # +-1.00 pct pts
    "drop_rate":     {"type": "absolute", "value": 0.02},     # +-2pp
}

CELL_KEYS = ["scenario", "effect_label", "tool_label"]


def check_tolerance(metric: str, current_val: float, golden_val: float) -> dict:
    """Check if current value is within tolerance of golden value."""
    if metric not in TOLERANCES:
        return {"metric": metric, "status": "skip", "reason": "no tolerance defined"}

    if pd.isna(current_val) and pd.isna(golden_val):
        return {"metric": metric, "status": "pass", "reason": "both NaN"}

    if pd.isna(current_val) or pd.isna(golden_val):
        return {
            "metric": metric, "status": "fail",
            "reason": f"NaN mismatch (current={current_val}, golden={golden_val})",
        }

    tol = TOLERANCES[metric]
    if tol["type"] == "relative":
        if golden_val == 0:
            threshold = tol["value"]
        else:
            threshold = abs(golden_val) * tol["value"]
    else:
        threshold = tol["value"]

    diff = abs(current_val - golden_val)
    passed = diff <= threshold

    return {
        "metric": metric, "status": "pass" if passed else "fail",
        "current": round(current_val, 4), "golden": round(golden_val, 4),
        "diff": round(diff, 4), "threshold": round(threshold, 4),
    }


def run_eval(base_dir: str = ".") -> bool:
    base = Path(base_dir)
    golden_path = base / "results" / "golden" / "metrics.csv"
    current_path = base / "results" / "aggregated" / "metrics.csv"

    if not current_path.exists():
        print("ERROR: No current metrics found at", current_path)
        print("Run `make metrics` first.")
        return False

    current = pd.read_csv(current_path, keep_default_na=False, na_values=[""])

    if not golden_path.exists():
        print("No golden reference found at", golden_path)
        print("Current metrics (capture with `make eval-capture`):\n")
        print(current.to_string(index=False))
        return True  # not a failure — just nothing to compare

    golden = pd.read_csv(golden_path, keep_default_na=False, na_values=[""])

    print("Eval: comparing current metrics against golden reference")
    print("=" * 70)

    all_passed = True
    n_checks = 0
    n_failures = 0

    # Merge on cell keys
    merged = current.merge(golden, on=CELL_KEYS, suffixes=("_cur", "_gold"), how="outer")

    for _, row in merged.iterrows():
        cell_id = f"{row['scenario']}/{row['effect_label']}/{row['tool_label']}"

        for metric in TOLERANCES:
            cur_col = f"{metric}_cur"
            gold_col = f"{metric}_gold"

            if cur_col not in merged.columns or gold_col not in merged.columns:
                continue

            cur_val = row.get(cur_col)
            gold_val = row.get(gold_col)

            # Skip metrics not present for this effect condition
            if pd.isna(cur_val) and pd.isna(gold_val):
                continue

            result = check_tolerance(metric, cur_val, gold_val)
            n_checks += 1

            if result["status"] == "fail":
                n_failures += 1
                all_passed = False
                print(
                    f"  FAIL  {cell_id} | {metric}: "
                    f"current={result.get('current', 'NaN')} "
                    f"golden={result.get('golden', 'NaN')} "
                    f"diff={result.get('diff', '?')} "
                    f"threshold={result.get('threshold', '?')}"
                )

    print(f"\n{n_checks} checks, {n_failures} failures")
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    passed = run_eval(args.base_dir)
    sys.exit(0 if passed else 1)
