# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Monte Carlo simulation study comparing four geo-experiment causal inference tools — Meta GeoLift, CausalPy, Google Matched Markets, and Google CausalImpact — across stress-test scenarios. The pipeline runs 1,000 iterations per scenario (32,000 total fits), computes coverage/bias/power metrics, and generates publication figures.

## Commands

```bash
# Setup (creates .venv and restores R packages via renv)
make env

# Individual pipeline stages
make panels              # Generate synthetic panels via R (slow — ~hours)
make panels N_ITERATIONS=5   # Generate fewer panels for testing
make run                 # Run all 4 tools on all panels → results/raw/results.jsonl
make metrics             # Aggregate results.jsonl → results/aggregated/metrics.csv
make tables              # metrics.csv → markdown + CSV tables
make figures             # Generate forest plot and CI gallery PNGs

# Validation
make smoke               # Full end-to-end with N=5 (clean, panels, run, metrics, tables, figures, smoke_test.py)
make eval                # Compare current metrics.csv against golden reference
make eval-capture        # Capture current metrics as new golden reference

# Full pipeline
make all
```

Python venv: `.venv/bin/python`

## Running tests

```bash
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/test_compute_att.py   # single file
```

Tests import from `src/` via the editable install (`pip install -e .`). The `src/` package root requires that install to be in place before tests run.

### Running individual analysis scripts

```bash
.venv/bin/python analysis/compute_metrics.py
.venv/bin/python analysis/smoke_test.py
```

## Architecture

**Pipeline stages** (each is independent — make targets don't auto-chain):

1. `src/R/generate_panels.R` → `panels/<scenario>/<effect>/<iter>.parquet` — DGP with shared trend, AR(1) noise, weekly seasonality, log-normal baselines. Counterfactual `Y_counterfactual` column is written alongside `Y` — this is the ground truth.

2. `src/python/run_tools.py` → `results/raw/results.jsonl` — orchestrates all tools, runs concurrently via joblib, checkpoints to JSONL so crashes resume. Each record includes `tool`, `scenario`, `effect`, `iteration`, ATT estimates, CI bounds, and `posterior_type`.

3. `analysis/compute_metrics.py` → `results/aggregated/metrics.csv` — aggregates JSONL per cell (scenario × effect × tool × posterior_type). Deduplicates on last-written record per key (crash-recovery artifact).

4. `analysis/generate_tables.py`, `analysis/plot_forest.py`, `analysis/plot_ci_gallery.py` → `results/aggregated/` tables and `figures/` PNGs.

**Data flow**:

```
src/R/generate_panels.R
  → panels/{scenario}/{effect}/panel_*.parquet   (canonical long format: geo, date, Y, Y_counterfactual)
  → panels/{scenario}/metadata.json

src/python/run_tools.py                           (pipeline orchestrator)
  ├── data_converters.py  → per-tool format
  ├── run_causalpy.py     → CausalPy (Python, PyMC)
  ├── run_google_mm.py    → Google Matched Markets (Python)
  ├── R subprocess        → src/R/run_geolift.R
  └── R subprocess        → src/R/run_causalimpact.R
  → results/raw/results.jsonl                     (one JSON record per tool × iteration)

analysis/compute_metrics.py  → results/aggregated/metrics.csv
analysis/generate_tables.py  → results/aggregated/table_A*.{md,csv}
analysis/plot_*.py           → figures/*.png
```

**Python tool wrappers** (in `src/python/`):
- `run_causalpy.py` — Bayesian SC via PyMC; has convergence gating (rhat/ESS) and retry logic
- `run_google_mm.py` — TBR via `matched_markets`; deterministic (OLS)
- `compute_att.py` — ATT normalization used by all tools; coverage always checked on level-scale CIs against true ATT from `Y_counterfactual`
- `data_converters.py` — converts canonical long-format panels to each tool's required format

**R tool wrappers** (called as subprocesses from `run_tools.py`):
- `src/R/run_geolift.R` — Augmented SC + block conformal inference
- `src/R/run_causalimpact.R` — BSTS

**Key design decisions**:

**Canonical format**: panels use integer `date` (1-indexed days), string `geo`, observed `Y`, and DGP counterfactual `Y_counterfactual`. Each tool wrapper in `data_converters.py` converts from this to its required format (wide DataFrame for CausalPy, datetime-indexed with group/period columns for Google MM, location/date/Y strings for GeoLift).

**R tools run as subprocesses**: GeoLift and CausalImpact are R packages with no Python bindings. `run_tools.py` launches them via `subprocess` (calling `Rscript src/R/run_geolift.R` / `run_causalimpact.R`), passing panel data through a temp parquet file and reading JSON results back from stdout.

**Crash recovery**: `run_tools.py` appends to `results/raw/results.jsonl` and checks completed `(scenario, effect_label, iteration, tool_label)` tuples on startup, so interrupted runs resume without re-running completed work.

**CausalPy convergence gating**: `run_causalpy.py` checks rhat ≤ 1.01 and ESS ≥ 400/200 (bulk/tail), retrying with doubled draws and higher `target_accept` (up to `max_retries=2`) before marking a result as non-converged.

**Tool config**: `config/tools.yaml` — all tools equalized at 95% confidence; hyperparameters documented there. Single source of truth for tool versions, hyperparameters, and inference settings.

**R environment**: R packages are managed by `renv` — `renv.lock` pins exact versions. Run `Rscript -e "renv::restore()"` (or `make env`) to reproduce the exact R environment. GeoLift and augsynth are installed from specific GitHub commits (see `config/tools.yaml`).

## Data correctness rules

All analysis must use the **equalization protocol** from `config/tools.yaml` and `compute_att.py`:
- Coverage uses **level-scale CIs** vs `true_att_level` (not percent-scale)
- ATT% uses the **true counterfactual** (`Y_counterfactual`) as denominator for all tools, not each tool's estimated counterfactual
- Significance = CI excludes zero (two-sided); GeoLift's conformal p-value is recorded but not used for FPR/FNR

Any change to metric computation must be validated by running `make eval` against the golden reference and verified by an independent check (`analysis/audit_metrics.py`).

## Key data files

- `results/raw/results.jsonl` — one JSON record per (tool, scenario, effect, iteration). Committed.
- `results/aggregated/metrics.csv` — per-cell aggregated stats. Committed.
- `panels/` — 453 MB, not committed; regenerate with `make panels`.
- `results/golden/metrics.csv` — golden reference for regression detection; capture with `make eval-capture`.

## Pedagogical notebook

`notebooks/tool_comparison_walkthrough.ipynb` draws one sample panel per scenario
(via a Python-ported DGP in `src/python/panel_dgp.py`, not the R pipeline) and runs
each geo-experiment method on it, for building intuition about how each method
behaves per scenario — as opposed to the main study's 1,000-iteration runs.

**See `geo_model_selection.MD` for full implementation details** (per-method
wrapper/config references, the common result schema, and a checklist for adding
new methods like difference-in-differences, panel fixed effects, or synthetic
difference-in-differences). Consult and update that file whenever working on this
notebook or adding a new geo-experiment method to it.
