#!/usr/bin/env python3
"""
CI Gallery: 50 confidence intervals per tool, stacked vertically.

Green = interval contains the true ATT%. Red = misses.
Vertical dashed line marks the true effect (7.5%).

Layout: 3 rows (tools) × 4 columns (scenarios).
X-axes shared per scenario column so CI widths are visually comparable.
When one tool's range dwarfs the others (A2 CausalPy), the axis is set to
the majority range and clipped bars get an annotation.

Input:  results/raw/results.jsonl
Output: figures/ci_gallery.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

SCENARIO_ORDER = ["A1", "A2", "A3", "A4"]
SCENARIO_TITLES = {
    "A1": "A1: Textbook",
    "A2": "A2: Outlier (5×)",
    "A3": "A3: Small Donor Pool",
    "A4": "A4: Short Pre-Treatment",
}

TOOL_ORDER = ["causalpy", "google_mm", "geolift", "causalimpact"]
TOOL_LABELS = {
    "causalpy": "CausalPy",
    "google_mm": "Google MM",
    "geolift": "Meta GeoLift",
    "causalimpact": "CausalImpact",
}

N_INTERVALS = 50
TRUE_ATT = 0.075
SEED = 42
OUTLIER_RATIO = 5  # if one tool's range is >5× wider, clip it

# Per-scenario x-axis lower-bound floor (None = use computed). Bars below
# the floor are clipped silently at the axis edge — no annotation.
COLUMN_XLIM_LO_OVERRIDE = {"A1": -100}

COLOR_HIT = "#27ae60"
COLOR_MISS = "#e74c3c"
COLOR_TRUE = "#333333"
COLOR_NULL = "#8e44ad"  # purple — "no effect" (0%) reference line


def load_results(base_dir: Path) -> list[dict]:
    results = []
    with open(base_dir / "results" / "raw" / "results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["effect_label"] == "effect":
                results.append(r)
    return results


def sample_for_cell(all_results, scenario, tool, rng):
    """Sample N_INTERVALS results for one scenario/tool cell."""
    if tool == "causalpy":
        pool = [
            r for r in all_results
            if r["scenario"] == scenario
            and r["tool"] == "causalpy"
            and r.get("posterior_type") == "y_hat"
            and r["ci_lower"] is not None
            and not np.isnan(r["ci_lower"])
        ]
    else:
        pool = [
            r for r in all_results
            if r["scenario"] == scenario
            and r["tool"] == tool
            and r["ci_lower"] is not None
            and not np.isnan(r["ci_lower"])
        ]

    n = min(N_INTERVALS, len(pool))
    if n == 0:
        return []
    return list(rng.choice(pool, size=n, replace=False))


def compute_column_xlims(sampled_data, scenario):
    """Compute shared x-limits for a scenario column.

    If one tool's range is >OUTLIER_RATIO× wider than the others,
    base the shared range on the non-outlier tools and return the
    clipped tool name.
    """
    tool_ranges = {}
    for tool in TOOL_ORDER:
        data = sampled_data[(scenario, tool)]
        if not data:
            continue
        all_lo = [r["ci_lower"] * 100 for r in data]
        all_hi = [r["ci_upper"] * 100 for r in data]
        all_est = [r["att_pct"] * 100 for r in data]
        tool_ranges[tool] = (
            min(min(all_lo), min(all_est)),
            max(max(all_hi), max(all_est)),
        )

    if len(tool_ranges) < 2:
        # Not enough tools to compare — use whatever we have
        all_mins = [v[0] for v in tool_ranges.values()]
        all_maxs = [v[1] for v in tool_ranges.values()]
        lo = min(all_mins)
        hi = max(all_maxs)
        pad = (hi - lo) * 0.08
        return lo - pad, hi + pad, None

    # Check if any tool is an outlier
    spans = {t: (r[1] - r[0]) for t, r in tool_ranges.items()}
    clipped_tool = None

    for tool in TOOL_ORDER:
        if tool not in spans:
            continue
        others = [s for t, s in spans.items() if t != tool]
        if others and spans[tool] > OUTLIER_RATIO * max(others):
            clipped_tool = tool
            break

    if clipped_tool:
        # Base range on non-outlier tools
        non_outlier = {t: r for t, r in tool_ranges.items() if t != clipped_tool}
        lo = min(v[0] for v in non_outlier.values())
        hi = max(v[1] for v in non_outlier.values())
    else:
        lo = min(v[0] for v in tool_ranges.values())
        hi = max(v[1] for v in tool_ranges.values())

    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad, clipped_tool


def plot_ci_gallery(base_dir: str = "."):
    base = Path(base_dir)
    all_results = load_results(base)
    rng = np.random.default_rng(SEED)

    # ── Pass 1: sample data for all cells ──────────────────────
    sampled_data = {}
    for scenario in SCENARIO_ORDER:
        for tool in TOOL_ORDER:
            sampled_data[(scenario, tool)] = sample_for_cell(
                all_results, scenario, tool, rng
            )

    # ── Pass 2: compute shared x-limits per column ─────────────
    col_xlims = {}
    col_clipped = {}
    for scenario in SCENARIO_ORDER:
        xlo, xhi, clipped = compute_column_xlims(sampled_data, scenario)
        floor = COLUMN_XLIM_LO_OVERRIDE.get(scenario)
        if floor is not None:
            xlo = max(xlo, floor)
        col_xlims[scenario] = (xlo, xhi)
        col_clipped[scenario] = clipped

    # ── Pass 3: plot ───────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(20, 13), sharey=True)

    for col, scenario in enumerate(SCENARIO_ORDER):
        xlo, xhi = col_xlims[scenario]
        clipped_tool = col_clipped[scenario]

        for row, tool in enumerate(TOOL_ORDER):
            ax = axes[row, col]
            sampled = sampled_data[(scenario, tool)]
            n = len(sampled)

            if n == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#999")
                ax.set_xlim(xlo, xhi)
                continue

            n_clipped = 0
            max_clipped_val = 0

            for i, r in enumerate(sampled):
                lo = r["ci_lower"] * 100
                hi = r["ci_upper"] * 100
                est = r["att_pct"] * 100
                true = TRUE_ATT * 100

                covers = lo <= true <= hi
                color = COLOR_HIT if covers else COLOR_MISS

                # Track clipped bars
                if hi > xhi or lo < xlo:
                    n_clipped += 1
                    max_clipped_val = max(max_clipped_val, abs(hi), abs(lo))

                ax.plot([lo, hi], [i, i], color=color, linewidth=1.5,
                        solid_capstyle="round", alpha=0.7)

            # True effect line
            ax.axvline(x=TRUE_ATT * 100, color=COLOR_TRUE, linestyle="--",
                       linewidth=1.2, alpha=0.7, zorder=10)

            # No-effect (0%) reference line
            ax.axvline(x=0, color=COLOR_NULL, linestyle="--",
                       linewidth=1.2, alpha=0.7, zorder=10)

            # Shared x-limits
            ax.set_xlim(xlo, xhi)

            # Clipped bars annotation
            if tool == clipped_tool and n_clipped > 0:
                ax.annotate(
                    f"bars extend to {max_clipped_val:.0f}% →",
                    xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom",
                    fontsize=12, fontstyle="italic", color="#888",
                )

            # Coverage annotation
            n_covers = sum(
                1 for r in sampled
                if r["ci_lower"] * 100 <= TRUE_ATT * 100 <= r["ci_upper"] * 100
            )
            cov_pct = n_covers / n * 100
            ax.text(
                0.97, 0.95,
                f"{cov_pct:.0f}%",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=16, fontweight="bold",
                color=COLOR_HIT if cov_pct >= 90 else (
                    "#e67e22" if cov_pct >= 70 else COLOR_MISS
                ),
            )

            # Axis formatting
            ax.set_ylim(-1, n)
            ax.set_yticks([])

            # Add true ATT as a bold tick on the x-axis
            current_ticks = list(ax.get_xticks())
            true_val = TRUE_ATT * 100
            # Remove any tick too close to true_val to avoid overlap
            min_gap = (xhi - xlo) * 0.06
            filtered = [t for t in current_ticks
                        if abs(t - true_val) > min_gap]
            new_ticks = sorted(filtered + [true_val])
            ax.set_xticks(new_ticks)
            labels = []
            for t in new_ticks:
                if t == true_val:
                    labels.append(ax.text(0, 0, ""))  # placeholder
                else:
                    labels.append(ax.text(0, 0, ""))
            # Use a custom formatter
            def make_formatter(true_v, ticks):
                def fmt(x, pos):
                    if abs(x - true_v) < 0.01:
                        return f"{x:.1f}"
                    return f"{x:.0f}"
                return fmt
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(make_formatter(true_val, new_ticks))
            )
            # Bold the true ATT tick label
            ax.set_xticks(new_ticks)
            for label in ax.get_xticklabels():
                if label.get_text() == f"{true_val:.1f}":
                    label.set_fontweight("bold")

            if row == 0:
                ax.set_title(SCENARIO_TITLES[scenario], fontsize=18,
                             fontweight="bold", pad=12)
            ax.set_xlabel("ATT (%)", fontsize=14)
            if col == 0:
                ax.set_ylabel(TOOL_LABELS[tool], fontsize=16,
                              fontweight="bold", rotation=90, labelpad=12)

            ax.grid(axis="x", alpha=0.15)
            ax.tick_params(axis="x", labelsize=13)

    # Legend
    hit_patch = mpatches.Patch(color=COLOR_HIT, alpha=0.7, label="CI contains true effect")
    miss_patch = mpatches.Patch(color=COLOR_MISS, alpha=0.7, label="CI misses true effect")
    true_line = plt.Line2D([0], [0], color=COLOR_TRUE, linestyle="--",
                           linewidth=1.2, label="True ATT (7.5%)")
    null_line = plt.Line2D([0], [0], color=COLOR_NULL, linestyle="--",
                           linewidth=1.2, label="No effect (0%)")
    fig.legend(
        handles=[hit_patch, miss_patch, true_line, null_line],
        loc="lower center", ncol=4, fontsize=18, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Uncertainty intervals coverage and width",
        fontsize=24, fontweight="bold", y=1.055,
    )
    fig.text(
        0.5, 1.005,
        "Nominal coverage = 95%. Empirical coverage from 50 random experiments reported within each plot.",
        ha="center", fontsize=17, fontstyle="italic", color="#555",
    )

    plt.tight_layout()
    out_path = base / "figures" / "ci_gallery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    plot_ci_gallery(args.base_dir)
