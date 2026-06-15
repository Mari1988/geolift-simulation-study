#!/usr/bin/env python3
"""
Generate result tables — one per scenario.

Each table has 4 tool columns: CausalPy (y_hat), Google MM, Meta GeoLift, CausalImpact.
Rows: Avg ATT, Bias, Coverage, FNR, FPR, Avg CI Width.
"""

from pathlib import Path

import pandas as pd


TOOL_ORDER = ["causalpy_y_hat", "google_mm", "geolift", "causalimpact"]
TOOL_LABELS = {
    "causalpy_y_hat": "CausalPy (y_hat)",
    "google_mm": "Google MM",
    "geolift": "Meta GeoLift",
    "causalimpact": "CausalImpact",
}

SCENARIO_NAMES = {
    "A1": "Scenario A1: Textbook",
    "A2": "Scenario A2: Outlier (5x)",
    "A3": "Scenario A3: Small Donor Pool",
    "A4": "Scenario A4: Short Pre-Treatment",
}


def format_metric(value, fmt: str) -> str:
    """Format a metric value for display."""
    if pd.isna(value):
        return "—"
    return fmt.format(value)


def build_scenario_table(
    metrics: pd.DataFrame, scenario: str
) -> pd.DataFrame:
    """Build a result table for one scenario."""
    sc = metrics[metrics["scenario"] == scenario]

    effect = sc[sc["effect_label"] == "effect"]
    null = sc[sc["effect_label"] == "null"]

    rows = []

    for tool in TOOL_ORDER:
        eff_row = effect[effect["tool_label"] == tool]
        null_row = null[null["tool_label"] == tool]

        if len(eff_row) == 0:
            rows.append({
                "Tool": TOOL_LABELS.get(tool, tool),
                "Avg ATT (%)": "—",
                "Bias (pct pts)": "—",
                "Coverage": "—",
                "FNR": "—",
                "FPR": "—",
                "Avg CI Width (daily level)": "—",
            })
            continue

        eff = eff_row.iloc[0]
        null_data = null_row.iloc[0] if len(null_row) > 0 else pd.Series()

        fpr_str = "—"
        if not null_data.empty and "fpr" in null_data and not pd.isna(null_data.get("fpr")):
            fpr_val = null_data["fpr"]
            avg_null = null_data.get("avg_null_att_pct", "—")
            fpr_str = f"{fpr_val:.2%}"
            if avg_null != "—" and not pd.isna(avg_null):
                fpr_str += f" (avg lift: {avg_null:+.2f}%)"

        rows.append({
            "Tool": TOOL_LABELS.get(tool, tool),
            "Avg ATT (%)": format_metric(eff.get("avg_att_pct"), "{:.2f}"),
            "Bias (pct pts)": format_metric(eff.get("bias_pct_pts"), "{:+.2f}"),
            "Coverage": format_metric(eff.get("coverage"), "{:.2%}"),
            "FNR": format_metric(eff.get("fnr"), "{:.2%}"),
            "FPR": fpr_str,
            "Avg CI Width (daily level)": format_metric(
                eff.get("avg_ci_width_level"), "{:.2f}"
            ),
        })

    return pd.DataFrame(rows)


def generate_all_tables(base_dir: str = "."):
    """Generate and save tables for all scenarios."""
    metrics = pd.read_csv(
        Path(base_dir) / "results" / "aggregated" / "metrics.csv",
        keep_default_na=False,
        na_values=[""],
    )

    out_dir = Path(base_dir) / "results" / "aggregated"

    for scenario in ["A1", "A2", "A3", "A4"]:
        table = build_scenario_table(metrics, scenario)
        print(f"\n{SCENARIO_NAMES[scenario]}")
        print("=" * 80)
        print(table.to_string(index=False))

        # Save as CSV
        table.to_csv(out_dir / f"table_{scenario}.csv", index=False)

        # Save as markdown
        md = table.to_markdown(index=False)
        with open(out_dir / f"table_{scenario}.md", "w") as f:
            f.write(f"## {SCENARIO_NAMES[scenario]}\n\n")
            f.write(md)
            f.write("\n")

    print(f"\nTables saved to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    generate_all_tables(args.base_dir)
