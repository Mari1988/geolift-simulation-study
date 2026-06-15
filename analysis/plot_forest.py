#!/usr/bin/env python3
"""
Forest plot of ATT estimates: mean ± 95% simulation interval per tool × scenario.

Two panels (effect / null), rows grouped by scenario, one point+whisker per tool.

Input:  results/raw/results.jsonl
Output: figures/att_forest_plot.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOOL_ORDER = ["geolift", "google_mm", "causalpy_y_hat", "causalimpact"]
TOOL_COLORS = {
    "causalpy_y_hat": "#f4a49e",
    "google_mm": "#2f465b",
    "geolift": "#6c757d",
    "causalimpact": "#5b8c5a",
}
TOOL_LABELS = {
    "causalpy_y_hat": "CausalPy (y_hat)",
    "google_mm": "Google MM",
    "geolift": "Meta GeoLift",
    "causalimpact": "CausalImpact",
}
SCENARIO_ORDER = ["A4", "A3", "A2", "A1"]
SCENARIO_LABELS = {
    "A1": "A1: Textbook",
    "A2": "A2: Outlier (5x)",
    "A3": "A3: Small Donor Pool",
    "A4": "A4: Short Pre-Treatment",
}


def load_results(base_dir: Path) -> pd.DataFrame:
    results_path = base_dir / "results" / "raw" / "results.jsonl"
    records = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["tool_label"] = df.apply(
        lambda r: f"{r['tool']}_{r['posterior_type']}"
        if r["posterior_type"] else r["tool"],
        axis=1,
    )
    df["att_pct_scaled"] = df["att_pct"] * 100
    return df


def _make_forest_figure(
    raw: pd.DataFrame,
    tool_order: list[str],
    title: str,
    out_path: Path,
):
    """Render a forest plot for the given tool subset and save to disk."""
    effect_labels = ["effect", "null"]
    col_titles = ["7.5% Effect", "Null (0%)"]
    true_values = {"effect": 7.5, "null": 0.0}

    n_tools = len(tool_order)
    group_size = n_tools
    group_gap = 1.5

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_tools * 2.2)), sharey=True)

    for col, effect_label in enumerate(effect_labels):
        ax = axes[col]
        true_val = true_values[effect_label]

        y_positions = []
        y_labels = []
        colors = []
        means = []
        ci_lows = []
        ci_highs = []

        for i, scenario in enumerate(SCENARIO_ORDER):
            base_y = i * (group_size + group_gap)
            for j, tool in enumerate(tool_order):
                y = base_y + j
                data = raw[
                    (raw["scenario"] == scenario)
                    & (raw["effect_label"] == effect_label)
                    & (raw["tool_label"] == tool)
                ]["att_pct_scaled"].values

                if len(data) == 0:
                    continue

                mean = np.mean(data)
                lo = np.percentile(data, 2.5)
                hi = np.percentile(data, 97.5)

                y_positions.append(y)
                y_labels.append(TOOL_LABELS[tool])
                colors.append(TOOL_COLORS[tool])
                means.append(mean)
                ci_lows.append(lo)
                ci_highs.append(hi)

        y_positions = np.array(y_positions)
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        # Whiskers (2.5th–97.5th percentile)
        for k in range(len(y_positions)):
            ax.plot(
                [ci_lows[k], ci_highs[k]],
                [y_positions[k], y_positions[k]],
                color=colors[k], linewidth=2, solid_capstyle="round",
            )

        # Mean points
        for k in range(len(y_positions)):
            ax.plot(
                means[k], y_positions[k],
                "o", color=colors[k], markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5,
            )

        # True value line
        ax.axvline(
            x=true_val, color="black", linestyle="--",
            linewidth=1.2, alpha=0.6, zorder=1,
        )
        ax.text(
            true_val, y_positions[-1] + 1.8, f"True: {true_val:.1f}%",
            ha="center", va="bottom", fontsize=9, color="black", alpha=0.7,
        )

        # Scenario separators and labels
        for i, scenario in enumerate(SCENARIO_ORDER):
            base_y = i * (group_size + group_gap)

            if i > 0:
                sep_y = base_y - group_gap / 2
                ax.axhline(y=sep_y, color="#cccccc", linewidth=0.5, linestyle="-")

        # Y-axis labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)

        # Scenario bracket labels on the left
        if col == 0:
            for i, scenario in enumerate(SCENARIO_ORDER):
                base_y = i * (group_size + group_gap)
                mid_y = base_y + (group_size - 1) / 2
                ax.annotate(
                    SCENARIO_LABELS[scenario],
                    xy=(0, mid_y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-110, 0),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    ha="right", va="center",
                )

        ax.set_xlabel("ATT (%)", fontsize=11)
        ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=TOOL_COLORS[t], label=TOOL_LABELS[t],
               markersize=7, markeredgecolor="white", markeredgewidth=0.8, linewidth=2)
        for t in tool_order
    ]
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
               alpha=0.6, label="True ATT")
    )
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=len(tool_order) + 1, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.subplots_adjust(left=0.22, wspace=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_forest(base_dir: str = "."):
    base = Path(base_dir)
    raw = load_results(base)
    out_dir = base / "figures"

    # All tools
    _make_forest_figure(
        raw, TOOL_ORDER,
        title="ATT Estimates: Mean and 95% Simulation Interval",
        out_path=out_dir / "att_forest_plot.png",
    )

    # Excluding CausalPy — tighter x-axis for comparing the other three
    _make_forest_figure(
        raw, [t for t in TOOL_ORDER if t != "causalpy_y_hat"],
        title="ATT Estimates (excl. CausalPy): Mean and 95% Simulation Interval",
        out_path=out_dir / "att_forest_plot_no_causalpy.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    plot_forest(args.base_dir)
