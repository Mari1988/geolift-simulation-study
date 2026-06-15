#!/usr/bin/env python3
"""
Smoke test: validates that all 4 tools produced results and all downstream
artifacts (metrics, tables, figures) were generated correctly.

Usage: python analysis/smoke_test.py [--base-dir .]
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.python.data_converters import (
    to_causalpy_format,
    to_geolift_format,
    to_google_mm_format,
    verify_data_identity,
)

EXPECTED_TOOL_LABELS = {"causalpy_y_hat", "google_mm", "geolift", "causalimpact"}
EXPECTED_SCENARIOS = ["A1", "A2", "A3", "A4"]
EXPECTED_FIGURES = ["att_forest_plot.png", "att_forest_plot_no_causalpy.png", "ci_gallery.png"]


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def derive_tool_label(record: dict) -> str:
    pt = record.get("posterior_type", "")
    return f"{record['tool']}_{pt}" if pt else record["tool"]


def run_smoke_test(base_dir: str = ".") -> bool:
    base = Path(base_dir)
    all_passed = True

    print("Smoke test validation")
    print("=" * 50)

    # 1. results.jsonl exists and is non-empty
    jsonl_path = base / "results" / "raw" / "results.jsonl"
    records = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    all_passed &= check(
        "results.jsonl exists and non-empty",
        len(records) > 0,
        f"{len(records)} records" if records else "file missing or empty",
    )

    if not records:
        print("\nStopping early: no results to validate.")
        return False

    # 2. All 4 tool labels present
    found_labels = {derive_tool_label(r) for r in records}
    missing = EXPECTED_TOOL_LABELS - found_labels
    all_passed &= check(
        "All 4 tool labels present",
        len(missing) == 0,
        f"missing: {sorted(missing)}" if missing else f"found: {sorted(found_labels)}",
    )

    # 3. No tool returned 100% null att_pct
    for label in sorted(EXPECTED_TOOL_LABELS):
        tool_records = [r for r in records if derive_tool_label(r) == label]
        if not tool_records:
            continue
        all_null = all(r.get("att_pct") is None for r in tool_records)
        all_passed &= check(
            f"{label} has non-null att_pct",
            not all_null,
            f"{len(tool_records)} records, all null" if all_null
            else f"{len(tool_records)} records OK",
        )

    # 4. metrics.csv exists with all 4 tool labels
    metrics_path = base / "results" / "aggregated" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics_labels = set(metrics["tool_label"].unique())
        metrics_missing = EXPECTED_TOOL_LABELS - metrics_labels
        all_passed &= check(
            "metrics.csv has all 4 tool labels",
            len(metrics_missing) == 0,
            f"missing: {sorted(metrics_missing)}" if metrics_missing
            else "all present",
        )
    else:
        all_passed &= check("metrics.csv exists", False, "file not found")

    # 5. Figure files exist and are >0 bytes
    for fig_name in EXPECTED_FIGURES:
        fig_path = base / "figures" / fig_name
        exists_ok = fig_path.exists() and fig_path.stat().st_size > 0
        all_passed &= check(
            f"figures/{fig_name}",
            exists_ok,
            f"{fig_path.stat().st_size:,} bytes" if fig_path.exists()
            else "not found",
        )

    # 6. Table files exist for each scenario
    for sc in EXPECTED_SCENARIOS:
        for ext in ["csv", "md"]:
            table_path = base / "results" / "aggregated" / f"table_{sc}.{ext}"
            exists_ok = table_path.exists() and table_path.stat().st_size > 0
            all_passed &= check(
                f"table_{sc}.{ext}",
                exists_ok,
                f"{table_path.stat().st_size:,} bytes" if table_path.exists()
                else "not found",
            )

    # 7. Data converter identity check
    panel_path = base / "panels" / "A1" / "null" / "panel_0001.parquet"
    if panel_path.exists():
        df = pd.read_parquet(panel_path)
        treated_units = df[df["treated"]]["geo"].unique().tolist()
        wide = to_causalpy_format(df)
        mm = to_google_mm_format(df, treated_units)
        gl = to_geolift_format(df)
        ok = verify_data_identity(df, wide, mm, gl, treated_units)
        all_passed &= check("Data converter identity", ok)
    else:
        all_passed &= check("Data converter identity", False, "panel not found")

    # Summary
    print("\n" + "=" * 50)
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

    passed = run_smoke_test(args.base_dir)
    sys.exit(0 if passed else 1)
