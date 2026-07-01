# Geo Model Selection — Notebook Reference

This document tracks the implementation details behind
`notebooks/tool_comparison_walkthrough.ipynb`, a pedagogical notebook that draws
one sample panel per scenario and runs each geo-experiment method on it, so you
can see exactly what each method sees and how it responds — as opposed to the main
study's 1,000-iteration Monte Carlo runs in `results/`.

It is meant to grow as more methods are added (see **Planned methods** below). Keep
this file in sync with the notebook: when a method's wrapper, config, or output
schema changes, update the corresponding section here.

## Scope today

Implemented in the notebook: **GeoLift**, **Google Matched Markets (TBR)**,
**CausalImpact (BSTS)**. CausalPy is excluded from this notebook for now (it's
covered in the full study).

## Data generating process

`src/python/panel_dgp.py` is a Python port of the DGP in `src/R/generate_panels.R`,
built specifically so the notebook doesn't need to shell out to R just to draw a
panel. It does **not** need to bit-match R's RNG — only to reproduce the same
statistical structure — since the notebook is for exploration/teaching, not for
reproducing the committed golden results in `results/`.

DGP for each geo `i` at time `t`:

```
Y_cf[i,t] = baseline_i * trend_t * season_t * exp(noise_level * scale_i * ar_noise[i,t])
scale_i   = sqrt(noise_baselines_i / mean(noise_baselines))
Y[i,t]    = Y_cf[i,t] * (1 + effect_pct)   # treated geo(s), post-period only
```

- `draw_baselines(n_geos, baseline_mean, baseline_spread, seed)` — log-normal geo
  baselines, sorted ascending, named `"City 1"..."City N"`.
- `select_treated(baselines, n_treated)` — picks the geo(s) closest to the median
  baseline.
- `generate_panel(...)` — AR(1) noise per geo, shared trend + weekly seasonality,
  sqrt noise scaling, multiplicative treatment effect. Returns the canonical long
  format: `geo, date, Y, Y_counterfactual, treated`.
- `generate_scenario_panel(scenario_id, effect_pct, seed)` — resolves scenario
  overrides on top of `DEFAULTS`, applies A2's outlier inflation to the treated
  geo's baseline *after* selection (keeping pre-inflation baselines for noise
  scaling — mirrors the subtlety in `generate_panels.R` lines 385–390), and returns
  `(df, metadata)` where `metadata = {treated_units, treatment_start, treatment_end,
  effect_pct}`.

### Scenarios (`SCENARIOS` dict, overrides on top of `DEFAULTS`)

| Scenario | Name | Override | Stress test |
|---|---|---|---|
| A1 | Textbook | none | Clean data, well-behaved donor pool (baseline) |
| A2 | Outlier (5x) | `outlier=True` | Treated geo inflated 5x — convex hull violation |
| A3 | Small donor pool | `n_control=9` | 10 total geos (1 treated + 9 controls) |
| A4 | Short pre-treatment | `total_days=45, pre_days=30` | 30 pre-treatment days instead of 90 |

Tests: `tests/test_panel_dgp.py`.

## Method implementations (current)

Each method is wired into the notebook's `run_all_tools(df, meta)` function, which
calls each method's wrapper and returns a dict keyed by method name. Results are
then normalized by `normalize_result(tool, result, true_att_level, true_cf_mean)`
into a common schema before comparison — see **Common result schema** below.

### GeoLift — Augmented Synthetic Control (Ridge + conformal)

- **Wrapper**: `run_geolift_subprocess()` in `src/python/run_tools.py`, imported
  directly into the notebook (no changes made to this file).
- **Mechanism**: shells out to `Rscript src/R/run_geolift.R` (unchanged production
  script). Panel is converted to GeoLift's `location, date, Y` format via
  `to_geolift_format()` in `src/python/data_converters.py`, written to a temp CSV,
  and the R script's JSON result is read back.
- **Config**: `config/tools.yaml → geolift.alpha` (0.05 → 95% CI).
- **Caveat**: paths inside `run_geolift_subprocess` are relative to the repo root.
  The notebook does `os.chdir(REPO_ROOT)` in its setup cell so this works
  regardless of where Jupyter's working directory started (e.g. if opened from
  `notebooks/`, not the repo root).

### Google Matched Markets — Time-Based Regression (TBR)

- **Wrapper**: `run_google_mm()` in `src/python/run_google_mm.py`, called directly
  (no subprocess — deterministic OLS).
- **Mechanism**: panel converted to TBR format via `to_google_mm_format()` (adds
  `period`/`group`/`cost` columns, encodes geo as int, converts `date` to
  `datetime64`).
- **Config**: `config/tools.yaml → google_mm` (`confidence_level: 0.95`, `tails: 2`,
  `use_cooldown: false`).

### CausalImpact — BSTS (local level + spike-and-slab)

- **Wrapper**: `run_causalimpact_subprocess()` in `src/python/run_tools.py`,
  imported directly into the notebook.
- **Mechanism**: shells out to `Rscript src/R/run_causalimpact.R` (unchanged
  production script). No converter needed — just slices `geo, date, Y` from the
  canonical panel.
- **Config**: `config/tools.yaml → causalimpact` (`alpha: 0.05`,
  `model_args.nseasons: 7`, `model_args.niter: 2000`).
- **Caveat**: same repo-root-relative-path issue as GeoLift; covered by the same
  `os.chdir(REPO_ROOT)` fix.

## Common result schema

`normalize_result()` maps every method's raw output into:

| Field | Meaning |
|---|---|
| `tool` | method name |
| `att_level` | point estimate, level scale |
| `att_pct` | point estimate ÷ **true** counterfactual mean (unified denominator across all tools — not each tool's own estimated counterfactual) |
| `ci_lower_level` / `ci_upper_level` | CI bounds, level scale |
| `ci_width` | `ci_upper_level - ci_lower_level` |
| `covered` | does the level-scale CI contain the true ATT? (`compute_att.check_coverage`) |
| `significant` | tool's own significance flag (CI excludes zero) |
| `runtime_seconds` | wall-clock time for that method |

This mirrors the fair-comparison protocol already enforced in
`src/python/compute_att.py` for the main study: **coverage uses level-scale CIs,
not %-scale**, and **ATT% uses the true DGP counterfactual as denominator for all
tools**.

`compute_true_att(df, treated_units, treatment_start)` returns
`(true_att_level, true_att_pct, true_cf_mean)` and is the ground truth every
method is compared against.

## Notebook structure

1. **Intro** — scenario table + what to watch for per method.
2. **Setup** — imports, `os.chdir(REPO_ROOT)`, load `config/tools.yaml`.
3. **Generate one sample panel per scenario** — `generate_scenario_panel()` for
   A1–A4 at `effect_pct=0.075`, `seed=1`; raw time-series plot per scenario
   (control lines light gray, control average dark navy `#2f465b`, treated geo
   coral `#E87461`, treatment start as a dashed line — same visual language as
   `figures/scenario_timeseries.png`).
4. **Run each method on each scenario** — `run_all_tools()`.
5. **Compare against true ATT** — `normalize_result()` → per-scenario comparison
   `DataFrame`.
6. **Forest plot per scenario** — true ATT as vertical line, each method's point
   estimate ± CI as a horizontal bar.
7. **Cross-scenario summary table** — one combined `DataFrame` (scenario × method).
8. **Discussion prompts** — guided questions per scenario tying observed
   bias/coverage/CI-width back to each method's mechanism.

## Adding a new method

To add a method (e.g. one of the planned ones below), follow this checklist:

1. **Wrapper**: implement (or reuse, if it already exists in `src/python/` or
   `src/R/`) a function that takes the canonical panel (or its converted format)
   plus treatment metadata, and returns a result with at least: `att_level`,
   `ci_lower_level`, `ci_upper_level`, `significant`, `runtime_seconds`.
   - If it's an R method with no Python bindings, follow the existing subprocess
     pattern (`run_geolift_subprocess` / `run_causalimpact_subprocess` in
     `src/python/run_tools.py`) — CSV in, JSON out, paths relative to repo root.
   - If it's a pure Python/statsmodels method, a direct call is simpler (see
     `run_google_mm`).
2. **Data format**: if the method needs a different shape than the canonical
   `geo, date, Y, Y_counterfactual, treated` long format, add a converter to
   `src/python/data_converters.py` (matching the existing `to_geolift_format` /
   `to_google_mm_format` pattern) rather than reshaping inline in the notebook.
3. **Config**: if the method has tunable hyperparameters, add a section to
   `config/tools.yaml` rather than hardcoding values in the notebook, so it stays
   the single source of truth across the full study and this notebook.
4. **Normalize**: extend `normalize_result()` in the notebook with a branch for the
   new method's raw result shape, mapping it into the common schema above.
5. **Wire in**: add the method to `run_all_tools()`, add a color to `TOOL_COLORS`,
   and it will automatically show up in the forest plots and summary table.
6. **Tests**: add unit tests under `tests/` for any new converter or wrapper logic,
   following the style of `tests/test_data_converters.py` / `tests/test_panel_dgp.py`.
7. **Update this file**: add a subsection under **Method implementations**
   describing the method's mechanism, wrapper location, and config keys, and move
   it out of **Planned methods** below.

## Planned methods (not yet implemented)

- **Difference-in-Differences (DiD)** — simplest baseline; two-period (or
  event-study) regression comparing treated vs. control average pre/post. Good
  pedagogical contrast against GeoLift/TBR's donor-weighting approaches — no
  weighting at all, just a raw control average as the counterfactual.
- **Panel data fixed effects** — geo and time fixed-effects regression
  (two-way FE), the natural generalization of DiD to more than 2 periods /
  groups. Useful for showing how FE assumptions (parallel trends) break down
  under the outlier (A2) and short-pre-treatment (A4) scenarios.
- **Synthetic Difference-in-Differences (SDID)** — combines synthetic control
  donor weighting with DiD's time-demeaning; a natural bridge between GeoLift's
  augmented SC and DiD's simplicity. Likely available via the `synthdid` R
  package (no Python implementation) — would follow the subprocess pattern.

When these are added, also consider whether the discussion-prompts section needs
new questions contrasting the added methods' assumptions against the existing
three.
