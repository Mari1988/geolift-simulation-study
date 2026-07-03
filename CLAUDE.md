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

## Architecture

**Pipeline stages** (each is independent — make targets don't auto-chain):

1. `src/R/generate_panels.R` → `panels/<scenario>/<effect>/<iter>.parquet` — DGP with shared trend, AR(1) noise, weekly seasonality, log-normal baselines. Counterfactual `Y_counterfactual` column is written alongside `Y` — this is the ground truth.

2. `src/python/run_tools.py` → `results/raw/results.jsonl` — orchestrates all tools, runs concurrently via joblib, checkpoints to JSONL so crashes resume. Each record includes `tool`, `scenario`, `effect`, `iteration`, ATT estimates, CI bounds, and `posterior_type`.

3. `analysis/compute_metrics.py` → `results/aggregated/metrics.csv` — aggregates JSONL per cell (scenario × effect × tool × posterior_type). Deduplicates on last-written record per key (crash-recovery artifact).

4. `analysis/generate_tables.py`, `analysis/plot_forest.py`, `analysis/plot_ci_gallery.py` → `results/aggregated/` tables and `figures/` PNGs.

**Python tool wrappers** (in `src/python/`):
- `run_causalpy.py` — Bayesian SC via PyMC; has convergence gating (rhat/ESS) and retry logic
- `run_google_mm.py` — TBR via `matched_markets`; deterministic (OLS)
- `compute_att.py` — ATT normalization used by all tools; coverage always checked on level-scale CIs against true ATT from `Y_counterfactual`
- `data_converters.py` — converts canonical long-format panels to each tool's required format

**R tool wrappers** (called as subprocesses from `run_tools.py`):
- `src/R/run_geolift.R` — Augmented SC + block conformal inference
- `src/R/run_causalimpact.R` — BSTS

**Tool config**: `config/tools.yaml` — all tools equalized at 95% confidence; hyperparameters documented there.

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
