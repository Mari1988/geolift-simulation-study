#!/usr/bin/env python3
"""
Main analysis runner — runs CausalPy, GeoLift, and Google MM on all panels.

Usage: python src/python/run_tools.py

Expects:
  panels/{scenario}/metadata.json       (from generate_panels.R)
  panels/{scenario}/{effect}/panel_*.parquet
  config/tools.yaml                     (tool configurations)
  src/R/run_geolift.R                   (GeoLift CLI wrapper)

Output: results/raw/results.jsonl       (append-mode, one JSON object per line)
"""

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.python.compute_att import check_coverage, compute_true_att
from src.python.data_converters import (
    to_causalpy_format,
    to_geolift_format,
    to_google_mm_format,
)
from src.python.run_causalpy import run_causalpy
from src.python.run_google_mm import run_google_mm


# ── Helpers ────────────────────────────────────────────────────


def load_completed_keys(jsonl_path: Path) -> set[tuple]:
    """Scan existing JSONL for crash recovery.

    Returns set of (scenario, effect_label, iteration, tool_label) tuples
    already present in the results file.
    """
    keys = set()
    if not jsonl_path.exists():
        return keys
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                pt = r.get("posterior_type", "")
                tool_label = f"{r['tool']}_{pt}" if pt else r["tool"]
                keys.add((r["scenario"], r["effect_label"], r["iteration"], tool_label))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def discover_scenarios(panels_dir: Path) -> list[dict]:
    """Find all scenario metadata.json files, sorted by scenario ID."""
    scenarios = []
    for meta_path in sorted(panels_dir.glob("*/metadata.json")):
        with open(meta_path) as f:
            scenarios.append(json.load(f))
    return scenarios


def serialize_for_json(obj: dict) -> dict:
    """Convert numpy types to JSON-safe Python natives. NaN → null."""
    cleaned = {}
    for k, v in obj.items():
        if isinstance(v, np.integer):
            cleaned[k] = int(v)
        elif isinstance(v, np.floating):
            cleaned[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            cleaned[k] = bool(v)
        elif isinstance(v, float) and np.isnan(v):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def append_result(result: dict, jsonl_path: Path):
    """Append one result as a JSONL line. Flushes immediately for crash safety."""
    cleaned = serialize_for_json(result)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(cleaned) + "\n")
        f.flush()


def run_geolift_subprocess(
    geolift_df: pd.DataFrame,
    treated_units: list[str],
    treatment_start: int,
    treatment_end: int,
    alpha: float = 0.05,
) -> dict:
    """Run GeoLift as an R subprocess via src/R/run_geolift.R."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_in:
        geolift_df.to_csv(tmp_in.name, index=False)
        input_path = tmp_in.name

    output_path = input_path.replace(".csv", "_result.json")

    try:
        cmd = [
            "Rscript", "src/R/run_geolift.R",
            "--input", input_path,
            "--output", output_path,
            "--locations", ",".join(treated_units),
            "--treatment_start", str(treatment_start),
            "--treatment_end", str(treatment_end),
            "--alpha", str(alpha),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            return {
                "status": "error",
                "att_level": None, "att_pct": None,
                "ci_lower": None, "ci_upper": None,
                "significant": False, "runtime_seconds": 0,
            }

        with open(output_path) as f:
            return json.load(f)
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def run_causalimpact_subprocess(
    ci_df: pd.DataFrame,
    treated_units: list[str],
    treatment_start: int,
    treatment_end: int,
    ci_config: dict,
    iteration: int = None,
) -> dict:
    """Run CausalImpact as an R subprocess via src/R/run_causalimpact.R."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_in:
        ci_df.to_csv(tmp_in.name, index=False)
        input_path = tmp_in.name

    output_path = input_path.replace(".csv", "_result.json")

    try:
        alpha = ci_config["alpha"]
        nseasons = ci_config["model_args"]["nseasons"]
        niter = ci_config["model_args"].get("niter", 1000)

        cmd = [
            "Rscript", "src/R/run_causalimpact.R",
            "--input", input_path,
            "--output", output_path,
            "--locations", ",".join(treated_units),
            "--treatment_start", str(treatment_start),
            "--treatment_end", str(treatment_end),
            "--alpha", str(alpha),
            "--nseasons", str(nseasons),
            "--niter", str(niter),
        ]
        if iteration is not None:
            cmd.extend(["--seed", str(iteration)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            return {
                "status": "error",
                "att_level": None, "att_pct": None,
                "ci_lower": None, "ci_upper": None,
                "significant": False, "runtime_seconds": 0,
            }

        with open(output_path) as f:
            return json.load(f)
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


# ── Per-panel runner ───────────────────────────────────────────


def run_panel(
    panel_path: Path,
    scenario: str,
    effect_label: str,
    effect_pct: float,
    iteration: int,
    treatment_start: int,
    treatment_end: int,
    completed: set[tuple],
    jsonl_path: Path,
    cp_config: dict,
    mm_config: dict,
    gl_alpha: float,
    ci_config: dict,
):
    """Run all tools on one panel. Skips tools already in completed set."""
    df = pd.read_parquet(panel_path)
    treated_units = df[df["treated"]]["geo"].unique().tolist()
    true_att_level, true_att_pct, true_cf_mean = compute_true_att(
        df, treated_units, treatment_start
    )

    # Prepare tool-specific data formats
    wide_df = to_causalpy_format(df)
    mm_df = to_google_mm_format(df, treated_units, treatment_start)
    geolift_df = to_geolift_format(df)
    ci_df = df[["geo", "date", "Y"]].copy()

    # Check which tools still need to run (derive CausalPy keys from config)
    cp_keys = [
        (scenario, effect_label, iteration, f"causalpy_{pt}")
        for pt in cp_config["posteriors"]
    ]
    gl_key = (scenario, effect_label, iteration, "geolift")
    mm_key = (scenario, effect_label, iteration, "google_mm")
    ci_key = (scenario, effect_label, iteration, "causalimpact")

    need_cp = any(k not in completed for k in cp_keys)
    need_gl = gl_key not in completed
    need_mm = mm_key not in completed
    need_ci = ci_key not in completed

    if not (need_cp or need_gl or need_mm or need_ci):
        return 0  # nothing to do

    # Run CausalPy and GeoLift concurrently (different resource pools:
    # CausalPy uses 4 MCMC chains on CPU, GeoLift is an R subprocess)
    cp_results = None
    gl_result = None

    def _run_cp():
        return run_causalpy(
            wide_df=wide_df, treated_units=treated_units,
            treatment_time=treatment_start,
            config=cp_config, iteration=iteration,
        )

    def _run_gl():
        return run_geolift_subprocess(
            geolift_df=geolift_df, treated_units=treated_units,
            treatment_start=treatment_start, treatment_end=treatment_end,
            alpha=gl_alpha,
        )

    def _run_ci():
        return run_causalimpact_subprocess(
            ci_df=ci_df, treated_units=treated_units,
            treatment_start=treatment_start, treatment_end=treatment_end,
            ci_config=ci_config,
            iteration=iteration,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        cp_future = pool.submit(_run_cp) if need_cp else None
        gl_future = pool.submit(_run_gl) if need_gl else None
        ci_future = pool.submit(_run_ci) if need_ci else None

        if cp_future:
            cp_results = cp_future.result()
        if gl_future:
            gl_result = gl_future.result()
        ci_result_raw = None
        if ci_future:
            ci_result_raw = ci_future.result()

    # Google MM is instant — run synchronously
    mm_result = None
    if need_mm:
        mm_result = run_google_mm(mm_df=mm_df, config=mm_config)

    # ── Write results (flush per tool for crash safety) ────────

    if cp_results:
        for cp_res in cp_results:
            tool_label = f"causalpy_{cp_res.posterior_type}"
            coverage = check_coverage(
                cp_res.ci_lower_level, cp_res.ci_upper_level, true_att_level
            )
            att_pct_unified = (
                cp_res.att_level / true_cf_mean
                if true_cf_mean != 0 else np.nan
            )
            append_result({
                "scenario": scenario, "effect_label": effect_label,
                "effect_pct": effect_pct, "iteration": iteration,
                "tool": "causalpy", "posterior_type": cp_res.posterior_type,
                "att_level": cp_res.att_level, "att_pct": att_pct_unified,
                "ci_lower": cp_res.ci_lower, "ci_upper": cp_res.ci_upper,
                "ci_lower_level": cp_res.ci_lower_level,
                "ci_upper_level": cp_res.ci_upper_level,
                "true_att_pct": true_att_pct, "true_att_level": true_att_level,
                "coverage": coverage,
                "significant": cp_res.significant,
                "converged": cp_res.converged,
                "rhat_max": cp_res.rhat_max,
                "ess_bulk_min": cp_res.ess_bulk_min,
                "ess_tail_min": cp_res.ess_tail_min,
                "n_divergences": cp_res.n_divergences,
                "runtime_seconds": cp_res.runtime_seconds,
            }, jsonl_path)
            completed.add((scenario, effect_label, iteration, tool_label))

    if mm_result:
        coverage = check_coverage(
            mm_result.ci_lower_level, mm_result.ci_upper_level, true_att_level
        )
        att_pct_unified = (
            mm_result.att_level / true_cf_mean
            if true_cf_mean != 0 else np.nan
        )
        append_result({
            "scenario": scenario, "effect_label": effect_label,
            "effect_pct": effect_pct, "iteration": iteration,
            "tool": "google_mm", "posterior_type": "",
            "att_level": mm_result.att_level, "att_pct": att_pct_unified,
            "ci_lower": mm_result.ci_lower, "ci_upper": mm_result.ci_upper,
            "ci_lower_level": mm_result.ci_lower_level,
            "ci_upper_level": mm_result.ci_upper_level,
            "true_att_pct": true_att_pct, "true_att_level": true_att_level,
            "coverage": coverage,
            "significant": mm_result.significant,
            "r_squared": mm_result.r_squared,
            "durbin_watson": mm_result.durbin_watson,
            "runtime_seconds": mm_result.runtime_seconds,
        }, jsonl_path)
        completed.add(mm_key)

    if gl_result:
        gl_att_level = gl_result.get("att_level")
        gl_ci_lower_level = gl_result.get("ci_lower_level")
        gl_ci_upper_level = gl_result.get("ci_upper_level")
        coverage = (
            check_coverage(gl_ci_lower_level, gl_ci_upper_level, true_att_level)
            if gl_att_level is not None else False
        )
        att_pct_unified = (
            gl_att_level / true_cf_mean
            if gl_att_level is not None and true_cf_mean != 0 else np.nan
        )
        append_result({
            "scenario": scenario, "effect_label": effect_label,
            "effect_pct": effect_pct, "iteration": iteration,
            "tool": "geolift", "posterior_type": "",
            "att_level": gl_att_level, "att_pct": att_pct_unified,
            "ci_lower": gl_result.get("ci_lower"),
            "ci_upper": gl_result.get("ci_upper"),
            "ci_lower_level": gl_ci_lower_level,
            "ci_upper_level": gl_ci_upper_level,
            "true_att_pct": true_att_pct, "true_att_level": true_att_level,
            "coverage": coverage,
            "significant": gl_result.get("significant", False),
            "p_value": gl_result.get("p_value"),
            "runtime_seconds": gl_result.get("runtime_seconds", 0),
        }, jsonl_path)
        completed.add(gl_key)

    if ci_result_raw:
        ci_att_level = ci_result_raw.get("att_level")
        ci_ci_lower_level = ci_result_raw.get("ci_lower_level")
        ci_ci_upper_level = ci_result_raw.get("ci_upper_level")
        coverage = (
            check_coverage(ci_ci_lower_level, ci_ci_upper_level, true_att_level)
            if ci_att_level is not None else False
        )
        att_pct_unified = (
            ci_att_level / true_cf_mean
            if ci_att_level is not None and true_cf_mean != 0 else np.nan
        )
        append_result({
            "scenario": scenario, "effect_label": effect_label,
            "effect_pct": effect_pct, "iteration": iteration,
            "tool": "causalimpact", "posterior_type": "",
            "att_level": ci_att_level, "att_pct": att_pct_unified,
            "ci_lower": ci_result_raw.get("ci_lower"),
            "ci_upper": ci_result_raw.get("ci_upper"),
            "ci_lower_level": ci_ci_lower_level,
            "ci_upper_level": ci_ci_upper_level,
            "true_att_pct": true_att_pct, "true_att_level": true_att_level,
            "coverage": coverage,
            "significant": ci_result_raw.get("significant", False),
            "runtime_seconds": ci_result_raw.get("runtime_seconds", 0),
        }, jsonl_path)
        completed.add(ci_key)

    return 1  # work was done


# ── Main ───────────────────────────────────────────────────────


def main():
    panels_dir = Path("panels")
    jsonl_path = Path("results/raw/results.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tool configs once (avoids re-reading YAML per iteration)
    with open("config/tools.yaml") as f:
        tools_cfg = yaml.safe_load(f)
    cp_config = tools_cfg["causalpy"]
    mm_config = tools_cfg["google_mm"]
    gl_alpha = tools_cfg["geolift"]["alpha"]
    ci_config = tools_cfg["causalimpact"]

    # Crash recovery: scan existing JSONL for completed keys
    completed = load_completed_keys(jsonl_path)
    if completed:
        print(f"Crash recovery: {len(completed)} results already done, skipping")

    # Discover scenarios from metadata.json files
    scenarios = discover_scenarios(panels_dir)
    if not scenarios:
        print("No scenarios found in panels/. Run `make panels` first.")
        sys.exit(1)

    print(f"Scenarios: {[s['scenario'] for s in scenarios]}")

    total_start = time.time()
    n_processed = 0

    for meta in scenarios:
        sc = meta["scenario"]
        t_start = meta["treatment_start"]
        t_end = meta["treatment_end"]

        print(f"\n=== {sc} (treatment: days {t_start}\u2013{t_end}) ===")

        for effect_label, effect_pct in meta["effect_sizes"].items():
            effect_dir = panels_dir / sc / effect_label
            panel_files = sorted(effect_dir.glob("panel_*.parquet"))

            if not panel_files:
                print(f"  {effect_label}: no panels found")
                continue

            print(f"  {effect_label} ({effect_pct}): {len(panel_files)} panels")

            for pf in panel_files:
                iteration = int(pf.stem.split("_")[1])
                did_work = run_panel(
                    panel_path=pf, scenario=sc,
                    effect_label=effect_label, effect_pct=float(effect_pct),
                    iteration=iteration,
                    treatment_start=t_start, treatment_end=t_end,
                    completed=completed, jsonl_path=jsonl_path,
                    cp_config=cp_config, mm_config=mm_config,
                    gl_alpha=gl_alpha, ci_config=ci_config,
                )
                if did_work:
                    n_processed += 1
                    if n_processed % 10 == 0:
                        elapsed = time.time() - total_start
                        print(f"    {n_processed} panels ({elapsed:.0f}s)")

    elapsed = time.time() - total_start
    print(f"\nDone: {n_processed} panels in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
