#!/usr/bin/env python3
"""
Independent audit of aggregated metrics.

Recomputes every metric from the raw JSONL and diffs against metrics.csv.
Also recomputes coverage and significance from scratch (not trusting pre-computed
fields) to check whether the tool-level computation was correct.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(".")
RAW_PATH = BASE / "results" / "raw" / "results.jsonl"
AGG_PATH = BASE / "results" / "aggregated" / "metrics.csv"

# ── 1. Load raw data ──────────────────────────────────────────────────────────

records = []
with open(RAW_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
n_raw = len(df)
print(f"Raw records loaded: {n_raw}")

# Build tool_label (same logic as compute_metrics.py)
df["tool_label"] = df.apply(
    lambda r: f"{r['tool']}_{r['posterior_type']}"
    if r["posterior_type"] else r["tool"],
    axis=1,
)

# Deduplicate (keep last, same as compute_metrics.py)
dedup_keys = ["scenario", "effect_label", "iteration", "tool_label"]
df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
print(f"After dedup: {len(df)} records (dropped {n_raw - len(df)})")

# ── 2. Recompute metrics from scratch ────────────────────────────────────────

group_cols = ["scenario", "effect_label", "effect_pct", "tool_label"]
audit_rows = []

for keys, g in df.groupby(group_cols):
    scenario, effect_label, effect_pct, tool_label = keys
    valid = g.dropna(subset=["att_pct"])
    n = len(g)
    n_valid = len(valid)

    if n_valid == 0:
        continue

    avg_att_pct = valid["att_pct"].mean() * 100
    avg_att_level = valid["att_level"].mean()
    true_att_pct = valid["true_att_pct"].mean() * 100
    bias_pct_pts = avg_att_pct - true_att_pct

    # Coverage from the pre-computed field (trusting raw data)
    coverage_from_raw = valid["coverage"].mean()

    # Coverage recomputed from scratch
    coverage_recomputed = (
        (valid["ci_lower_level"] <= valid["true_att_level"])
        & (valid["true_att_level"] <= valid["ci_upper_level"])
    ).mean()

    # CI width (level-scale)
    avg_ci_width_level = (valid["ci_upper_level"] - valid["ci_lower_level"]).mean()

    # Significance from the pre-computed field
    sig_from_raw = valid["significant"].mean()

    # Significance recomputed: level-scale for ALL tools
    # significant = CI excludes zero (both bounds same sign)
    sig_recomputed = (
        (valid["ci_lower_level"] > 0) | (valid["ci_upper_level"] < 0)
    ).mean()

    row = {
        "scenario": scenario,
        "effect_label": effect_label,
        "effect_pct": effect_pct,
        "tool_label": tool_label,
        "n_iterations": n,
        "n_valid": n_valid,
        "avg_att_pct": round(avg_att_pct, 2),
        "avg_att_level": round(avg_att_level, 2),
        "true_att_pct": round(true_att_pct, 2),
        "bias_pct_pts": round(bias_pct_pts, 2),
        "coverage_from_raw": round(coverage_from_raw, 4),
        "coverage_recomputed": round(coverage_recomputed, 4),
        "avg_ci_width_level": round(avg_ci_width_level, 2),
        "sig_from_raw": round(sig_from_raw, 4),
        "sig_recomputed": round(sig_recomputed, 4),
    }

    if effect_label == "effect":
        row["fnr_from_raw"] = round(1 - sig_from_raw, 4)
        row["fnr_recomputed"] = round(1 - sig_recomputed, 4)
    elif effect_label == "null":
        row["fpr_from_raw"] = round(sig_from_raw, 4)
        row["fpr_recomputed"] = round(sig_recomputed, 4)
        row["avg_null_att_pct"] = round(avg_att_pct, 2)

    audit_rows.append(row)

audit_df = pd.DataFrame(audit_rows)

# ── 3. Load existing metrics.csv ─────────────────────────────────────────────

existing = pd.read_csv(AGG_PATH, keep_default_na=False, na_values=[""])
# keep_default_na=False prevents pandas from interpreting "null" as NaN,
# but we still want empty cells to be NaN — na_values=[""] handles that.
print(f"\nExisting metrics.csv rows: {len(existing)}")
print(f"Audit recomputed rows: {len(audit_df)}")
print(f"Existing effect_label values: {existing['effect_label'].unique()}")

# ── 4. Merge and diff ────────────────────────────────────────────────────────

merge_keys = ["scenario", "effect_label", "tool_label"]
merged = audit_df.merge(existing, on=merge_keys, suffixes=("_audit", "_csv"))

print("\n" + "=" * 80)
print("PART 1: Recomputed metrics vs metrics.csv")
print("=" * 80)

# Columns to compare (audit_col, csv_col)
compare_cols = [
    ("avg_att_pct_audit", "avg_att_pct_csv", "avg_att_pct"),
    ("avg_att_level_audit", "avg_att_level_csv", "avg_att_level"),
    ("true_att_pct_audit", "true_att_pct_csv", "true_att_pct"),
    ("bias_pct_pts_audit", "bias_pct_pts_csv", "bias_pct_pts"),
    ("coverage_from_raw", "coverage", "coverage (raw field vs csv)"),
    ("avg_ci_width_level_audit", "avg_ci_width_level_csv", "avg_ci_width_level"),
]

any_diff = False
for acol, ccol, label in compare_cols:
    if acol not in merged.columns or ccol not in merged.columns:
        print(f"  SKIP {label}: column not found")
        continue
    diff_mask = merged[acol].round(4) != merged[ccol].round(4)
    n_diff = diff_mask.sum()
    if n_diff > 0:
        any_diff = True
        print(f"\n  MISMATCH in {label}: {n_diff} rows differ")
        cols_show = merge_keys + [acol, ccol]
        print(merged.loc[diff_mask, cols_show].to_string(index=False))
    else:
        print(f"  OK  {label}: all {len(merged)} rows match")

# FNR comparison (effect rows only)
effect_mask = merged["effect_label"] == "effect"
if "fnr_from_raw" in merged.columns and "fnr" in merged.columns:
    eff = merged[effect_mask].copy()
    diff_mask = eff["fnr_from_raw"].round(4) != eff["fnr"].round(4)
    n_diff = diff_mask.sum()
    if n_diff > 0:
        any_diff = True
        print(f"\n  MISMATCH in fnr (raw sig): {n_diff} rows differ")
        print(eff.loc[diff_mask, merge_keys + ["fnr_from_raw", "fnr"]].to_string(index=False))
    else:
        print(f"  OK  fnr (raw sig): all {effect_mask.sum()} effect rows match")

# FPR comparison (null rows only)
null_mask = merged["effect_label"] == "null"
if "fpr_from_raw" in merged.columns and "fpr" in merged.columns:
    nul = merged[null_mask].copy()
    diff_mask = nul["fpr_from_raw"].round(4) != nul["fpr"].round(4)
    n_diff = diff_mask.sum()
    if n_diff > 0:
        any_diff = True
        print(f"\n  MISMATCH in fpr (raw sig): {n_diff} rows differ")
        print(nul.loc[diff_mask, merge_keys + ["fpr_from_raw", "fpr"]].to_string(index=False))
    else:
        print(f"  OK  fpr (raw sig): all {null_mask.sum()} null rows match")

if not any_diff:
    print("\n  >>> ALL recomputed metrics match metrics.csv exactly. <<<")

# ── 5. Coverage: raw field vs recomputed from CI bounds ───────────────────────

print("\n" + "=" * 80)
print("PART 2: Coverage from raw 'coverage' field vs recomputed from CI bounds")
print("=" * 80)

diff_mask = audit_df["coverage_from_raw"].round(4) != audit_df["coverage_recomputed"].round(4)
n_diff = diff_mask.sum()
if n_diff > 0:
    print(f"\n  MISMATCH: {n_diff} rows where coverage differs")
    cols_show = ["scenario", "effect_label", "tool_label",
                 "coverage_from_raw", "coverage_recomputed"]
    print(audit_df.loc[diff_mask, cols_show].to_string(index=False))
    # Show per-tool summary
    print("\n  Per-tool summary of coverage discrepancy:")
    disc = audit_df[diff_mask].copy()
    disc["delta"] = disc["coverage_from_raw"] - disc["coverage_recomputed"]
    for tl in disc["tool_label"].unique():
        sub = disc[disc["tool_label"] == tl]
        print(f"    {tl}: avg delta = {sub['delta'].mean():.4f}, "
              f"max |delta| = {sub['delta'].abs().max():.4f}, "
              f"rows affected = {len(sub)}")
else:
    print("\n  >>> Raw coverage field matches recomputed coverage for all rows. <<<")

# ── 6. Significance: raw field vs recomputed (level-scale for all) ────────────

print("\n" + "=" * 80)
print("PART 3: FNR/FPR from raw 'significant' vs recomputed (level-scale CI for all tools)")
print("=" * 80)

# FNR
effect_rows = audit_df[audit_df["effect_label"] == "effect"].copy()
if "fnr_from_raw" in effect_rows.columns and "fnr_recomputed" in effect_rows.columns:
    diff_mask = effect_rows["fnr_from_raw"].round(4) != effect_rows["fnr_recomputed"].round(4)
    n_diff = diff_mask.sum()
    if n_diff > 0:
        print(f"\n  FNR MISMATCH: {n_diff} rows where fnr differs (raw sig vs level-scale recomputed)")
        cols_show = ["scenario", "effect_label", "tool_label",
                     "fnr_from_raw", "fnr_recomputed"]
        print(effect_rows.loc[diff_mask, cols_show].to_string(index=False))
        print("\n  Per-tool summary:")
        disc = effect_rows[diff_mask].copy()
        disc["delta"] = disc["fnr_from_raw"] - disc["fnr_recomputed"]
        for tl in disc["tool_label"].unique():
            sub = disc[disc["tool_label"] == tl]
            print(f"    {tl}: avg delta = {sub['delta'].mean():.4f}, "
                  f"max |delta| = {sub['delta'].abs().max():.4f}, "
                  f"rows = {len(sub)}")
    else:
        print(f"\n  FNR OK: raw significant field matches level-scale recomputed for all {len(effect_rows)} effect rows")

# FPR
null_rows = audit_df[audit_df["effect_label"] == "null"].copy()
if "fpr_from_raw" in null_rows.columns and "fpr_recomputed" in null_rows.columns:
    diff_mask = null_rows["fpr_from_raw"].round(4) != null_rows["fpr_recomputed"].round(4)
    n_diff = diff_mask.sum()
    if n_diff > 0:
        print(f"\n  FPR MISMATCH: {n_diff} rows where fpr differs (raw sig vs level-scale recomputed)")
        cols_show = ["scenario", "effect_label", "tool_label",
                     "fpr_from_raw", "fpr_recomputed"]
        print(null_rows.loc[diff_mask, cols_show].to_string(index=False))
        print("\n  Per-tool summary:")
        disc = null_rows[diff_mask].copy()
        disc["delta"] = disc["fpr_from_raw"] - disc["fpr_recomputed"]
        for tl in disc["tool_label"].unique():
            sub = disc[disc["tool_label"] == tl]
            print(f"    {tl}: avg delta = {sub['delta'].mean():.4f}, "
                  f"max |delta| = {sub['delta'].abs().max():.4f}, "
                  f"rows = {len(sub)}")
    else:
        print(f"\n  FPR OK: raw significant field matches level-scale recomputed for all {len(null_rows)} null rows")

# ── 7. Detailed record-level diagnostic for any coverage discrepancies ────────

print("\n" + "=" * 80)
print("PART 4: Record-level diagnostic — how many individual records disagree?")
print("=" * 80)

# Coverage per-record check
df["coverage_recomputed"] = (
    (df["ci_lower_level"] <= df["true_att_level"])
    & (df["true_att_level"] <= df["ci_upper_level"])
)
cov_disagree = df["coverage"] != df["coverage_recomputed"]
n_disagree = cov_disagree.sum()
print(f"\n  Coverage field vs recomputed: {n_disagree} / {len(df)} records disagree")
if n_disagree > 0:
    print("  Per-tool breakdown:")
    for tl in sorted(df["tool_label"].unique()):
        sub = df[df["tool_label"] == tl]
        nd = (sub["coverage"] != sub["coverage_recomputed"]).sum()
        if nd > 0:
            print(f"    {tl}: {nd} / {len(sub)} records disagree")

# Significance per-record check
df["sig_recomputed"] = (
    (df["ci_lower_level"] > 0) | (df["ci_upper_level"] < 0)
)
sig_disagree = df["significant"] != df["sig_recomputed"]
n_disagree = sig_disagree.sum()
print(f"\n  Significant field vs level-scale recomputed: {n_disagree} / {len(df)} records disagree")
if n_disagree > 0:
    print("  Per-tool breakdown:")
    for tl in sorted(df["tool_label"].unique()):
        sub = df[df["tool_label"] == tl]
        nd = (sub["significant"] != sub["sig_recomputed"]).sum()
        if nd > 0:
            pct = nd / len(sub) * 100
            # Show direction: how many raw=True but recomputed=False, and vice versa
            raw_true_recomp_false = ((sub["significant"] == True) & (sub["sig_recomputed"] == False)).sum()
            raw_false_recomp_true = ((sub["significant"] == False) & (sub["sig_recomputed"] == True)).sum()
            print(f"    {tl}: {nd} / {len(sub)} ({pct:.1f}%) disagree "
                  f"[raw=sig,recomp=not: {raw_true_recomp_false}; "
                  f"raw=not,recomp=sig: {raw_false_recomp_true}]")

    # For a few sample records where they disagree, show the actual values
    print("\n  Sample disagreeing records (first 10):")
    sample = df[sig_disagree].head(10)[
        ["scenario", "effect_label", "iteration", "tool_label",
         "ci_lower", "ci_upper", "ci_lower_level", "ci_upper_level",
         "significant", "sig_recomputed"]
    ]
    print(sample.to_string(index=False))

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
