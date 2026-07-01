# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Monte Carlo simulation study comparing four geo-experiment causal inference tools — Meta GeoLift, CausalPy, Google Matched Markets, and Google CausalImpact — across four stress-test scenarios (32,000 total model fits). Results are pre-computed and committed; panels (453 MB) are not.

## Commands

All pipeline steps run through `make`. The Python venv is at `.venv/`.

```bash
make env          # Create .venv, install Python deps, restore R packages (renv)
make panels       # Generate synthetic panels via R (slow, ~8 hrs for full run)
make run          # Run all four tools on all panels (writes results/raw/results.jsonl)
make metrics      # Aggregate results.jsonl → results/aggregated/metrics.csv
make tables       # Generate markdown + CSV tables from metrics
make figures      # Generate forest plot and CI gallery PNGs
make smoke        # Quick end-to-end check with 5 iterations
make eval         # Regression check vs golden reference in results/golden/
make eval-capture # Capture current metrics as the golden reference
make clean        # Delete panels/, results/, figures/
```

Override iteration count: `make panels N_ITERATIONS=50`

### Running tests

```bash
.venv/bin/pytest tests/
.venv/bin/pytest tests/test_compute_att.py   # single file
```

Tests import from `src/` via the editable install (`pip install -e .`). The `src/` package root requires that install to be in place before tests run.

### Running individual analysis scripts

```bash
.venv/bin/python analysis/compute_metrics.py
.venv/bin/python analysis/smoke_test.py
```

## Architecture

### Data flow

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

### Key design decisions

**Canonical format**: panels use integer `date` (1-indexed days), string `geo`, observed `Y`, and DGP counterfactual `Y_counterfactual`. Each tool wrapper in `data_converters.py` converts from this to its required format (wide DataFrame for CausalPy, datetime-indexed with group/period columns for Google MM, location/date/Y strings for GeoLift).

**R tools run as subprocesses**: GeoLift and CausalImpact are R packages with no Python bindings. `run_tools.py` launches them via `subprocess` (calling `Rscript src/R/run_geolift.R` / `run_causalimpact.R`), passing panel data through a temp parquet file and reading JSON results back from stdout.

**Crash recovery**: `run_tools.py` appends to `results/raw/results.jsonl` and checks completed `(scenario, effect_label, iteration, tool_label)` tuples on startup, so interrupted runs resume without re-running completed work.

**Fair comparison protocol** (enforced in `compute_att.py`):
- Coverage uses level-scale CIs, not %-scale
- ATT% uses the true DGP counterfactual as denominator for all tools
- All tools configured at 95% confidence via `config/tools.yaml`

**CausalPy convergence gating**: `run_causalpy.py` checks rhat ≤ 1.01 and ESS ≥ 400/200 (bulk/tail), retrying with doubled draws and higher `target_accept` (up to `max_retries=2`) before marking a result as non-converged.

### Configuration

`config/tools.yaml` is the single source of truth for tool versions, hyperparameters, and inference settings. All wrappers load from it at runtime.

### R environment

R packages are managed by `renv` — `renv.lock` pins exact versions. Run `Rscript -e "renv::restore()"` (or `make env`) to reproduce the exact R environment. GeoLift and augsynth are installed from specific GitHub commits (see `config/tools.yaml`).

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
