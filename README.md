# Comparing Geo-Experiment Tools: A Monte Carlo Simulation Study

Companion code for [Recast's](https://research.getrecast.com/geolift-sim-study) study comparing four geo-experiment estimation tools — Meta GeoLift, CausalPy, Google Matched Markets, and Google CausalImpact — across realistic stress-test scenarios. The study runs 1,000 Monte Carlo iterations per scenario (32,000 total model fits) using a shared-trend DGP with AR(1) noise, weekly seasonality, and log-normal geo baselines.

The full technical report and executive summary are available on Recast's platforms.

## Repository Structure

```
geolift-study/
├── config/
│   └── tools.yaml                     # Tool configs equalized at 95% confidence
├── src/
│   ├── R/
│   │   ├── generate_panels.R          # Parameterized DGP panel generation
│   │   ├── run_geolift.R             # GeoLift wrapper (augmented SC + conformal)
│   │   └── run_causalimpact.R        # CausalImpact wrapper (BSTS)
│   └── python/
│       ├── run_tools.py               # Pipeline orchestration (all tools)
│       ├── run_causalpy.py            # CausalPy wrapper (Bayesian SC)
│       ├── run_google_mm.py           # Google Matched Markets wrapper (TBR)
│       ├── compute_att.py             # ATT extraction utilities
│       └── data_converters.py         # Panel format converters
├── analysis/                          # Post-simulation analysis scripts
│   ├── compute_metrics.py             # Coverage, bias, CI width, power/FPR
│   ├── generate_tables.py             # Markdown result tables
│   ├── plot_forest.py                 # ATT forest plot (article figure)
│   ├── plot_ci_gallery.py             # CI gallery plot (article figure)
│   ├── smoke_test.py                  # End-to-end validation
│   ├── eval_against_golden.py         # Regression detection vs golden reference
│   └── audit_metrics.py              # Independent metrics audit
├── tests/
│   ├── test_compute_att.py            # ATT computation unit tests
│   └── test_data_converters.py        # Data format converter unit tests
├── results/
│   ├── raw/results.jsonl              # Per-iteration results (32,000 records)
│   └── aggregated/                    # Summary tables (CSV + markdown)
├── figures/                           # Generated plots (PNG)
├── environment/
│   └── install_packages.R             # R dependency bootstrapping
├── VERSIONS.md                        # Tool & dependency version manifest
├── renv.lock                          # Pinned R package versions
├── pyproject.toml                     # Python dependencies
├── requirements.txt                   # Frozen Python dependencies (exact versions)
└── Makefile                           # Pipeline automation (see below)
```

### Script Reference

| Script | Purpose |
|--------|---------|
| `src/R/generate_panels.R` | Parameterized DGP — generates synthetic geo-level panels with shared trend, AR(1) noise, weekly seasonality, and log-normal baselines. Outputs parquet files per scenario/effect/iteration. |
| `src/python/run_tools.py` | Pipeline orchestrator — loops over all panels, runs all 4 tools concurrently, writes results to JSONL with crash recovery. |
| `src/python/run_causalpy.py` | CausalPy wrapper — Bayesian SC with Dirichlet prior, convergence monitoring (rhat/ESS), retry logic. Extracts posterior predictive (y_hat). |
| `src/python/run_google_mm.py` | Google Matched Markets wrapper — TBR regression via `matched_markets` package. Deterministic (OLS). |
| `src/R/run_geolift.R` | GeoLift wrapper — Augmented SC (Ridge) with block conformal inference. Called as R subprocess. |
| `src/R/run_causalimpact.R` | CausalImpact wrapper — BSTS (local level + spike-and-slab). Called as R subprocess. |
| `src/python/compute_att.py` | ATT normalization — computes true ATT from DGP counterfactuals, checks coverage on level-scale CIs. |
| `src/python/data_converters.py` | Format converters — transforms canonical long-format panels to CausalPy (wide), Google MM (TBR), and GeoLift (location/date/Y) formats. |
| `analysis/compute_metrics.py` | Aggregates raw JSONL results into per-cell metrics: ATT, bias, coverage, FNR, FPR, CI width. |
| `analysis/generate_tables.py` | Generates markdown and CSV result tables per scenario from aggregated metrics. |
| `analysis/plot_forest.py` | ATT forest plot — article figure showing point estimates and CIs across tools and scenarios. |
| `analysis/plot_ci_gallery.py` | CI gallery — article figure showing credible/confidence interval bands for sampled iterations. |
| `analysis/smoke_test.py` | End-to-end validation — checks that all tools produced results, all artifacts exist, no 100% null outputs. |
| `analysis/eval_against_golden.py` | Compares current metrics against a golden reference for regression detection. |

### How the pipeline works

The simulation is automated through a `Makefile` — a file that defines commands you can run to reproduce each step of the study without having to know which scripts to call or in what order. You type `make <target>` and it runs the right scripts in the right sequence.

The core of the pipeline is `src/python/run_tools.py`. It's a pure Python script (no AI involved) that loops through every scenario, iteration, and tool: it loads a synthetic panel, feeds it to CausalPy, Google MM, GeoLift, and CausalImpact, collects the results, and writes them to `results/raw/results.jsonl`. It also checkpoints progress so a run can resume if interrupted.

## Reproducing Results

### Prerequisites

- Python 3.12+
- R 4.x with the `renv` package (`install.packages("renv")`)
- GNU Make

### Setup

```bash
git clone https://github.com/getrecast/geolift-study.git
cd geolift-study

# Install dependencies (Python venv + R packages via renv)
make env
```

`make env` creates a Python virtual environment, installs frozen Python dependencies, and restores pinned R packages from `renv.lock`.

### Run the full pipeline

```bash
make all    # panels → run → metrics → tables → figures
```

This generates panels, runs all 32,000 model fits, computes metrics, and produces result tables and figures. The full run takes several hours depending on hardware.

### Pre-computed results

Raw results and figures are committed to this repository. Panels (453 MB) are not — they are regenerated via `make panels`. To skip the simulation and go straight to analysis:

```bash
make metrics   # Compute metrics from existing results
make tables    # Generate summary tables
make figures   # Generate article figures
```

### Available commands

Run `make help` to see all targets:

| Command | What it does |
|:---|:---|
| `make env` | Install Python and R dependencies |
| `make panels` | Generate synthetic panels |
| `make run` | Run GeoLift, CausalPy, Google MM, CausalImpact on all panels |
| `make metrics` | Compute aggregated metrics from results.jsonl |
| `make tables` | Generate result tables (CSV + markdown) |
| `make figures` | Generate article figures |
| `make smoke` | Quick smoke test (5 iterations, validates all outputs) |
| `make eval` | Compare current metrics against golden reference |
| `make eval-capture` | Capture current metrics as golden reference |
| `make all` | Run the full pipeline end to end |
| `make clean` | Remove all generated data and results |

## Scenarios

| Scenario | Name | Stress Test |
|----------|------|-------------|
| A1 | Textbook | Clean data, well-behaved donor pool (baseline) |
| A2 | Outlier (5x) | Treated geo inflated 5x — convex hull violation |
| A3 | Small donor pool | 10 total geos (1 treated + 9 controls) — sparse pool |
| A4 | Short pre-treatment | 30 pre-treatment days instead of 90 — data scarcity |

Each scenario is run under two effect conditions: *null* (0%, for false positive rate calibration) and *effect* (7.5% lift, for power and bias assessment).

## Tools

| Tool | Method | Implementation |
|------|--------|----------------|
| Meta GeoLift | Augmented Synthetic Control (Ridge + conformal inference) | R package via subprocess |
| CausalPy | Bayesian Synthetic Control (Dirichlet-weighted) | Python, PyMC backend |
| Google Matched Markets | Time-Based Regression (TBR) | Python package |
| Google CausalImpact | BSTS (local level + spike-and-slab regression) | R package via subprocess |

All tools are configured at 95% confidence for fair comparison. See `config/tools.yaml` for full configuration details.

## Configuration

Tool configurations are defined in `config/tools.yaml` — tool versions, hyperparameters, and inference settings. Scenario definitions and effect sizes are parameterized in `src/R/generate_panels.R`.

## Equalization Protocol

All tools are configured at 95% confidence for fair comparison. The following choices ensure that result differences reflect estimator methodology, not pipeline artifacts:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Significance test | CI exclusion (all tools) | Two-sided: reject if CI excludes zero. GeoLift's conformal p-value is recorded but not used for FPR/FNR. |
| CI width metric | Level-scale (absolute units) | Avoids confound from different counterfactual denominators across tools. |
| Coverage | Level-scale CIs vs true ATT | Computed identically for all tools in `compute_att.py`. |
| ATT % | Unified denominator (true counterfactual) | All tools use the same denominator for ATT%, not their own estimated counterfactual. |
| Conformal inference | Block permutations (GeoLift) | Deterministic; preserves temporal dependence from AR(1) DGP. |
| MCMC seeding | Per-iteration seed (CausalPy, CausalImpact) | Bitwise reproducibility. Google MM is deterministic (OLS). GeoLift block conformal is deterministic. |
| Convergence monitoring | CausalPy only (rhat/ESS gating) | Each tool is used according to its ecosystem's best practices. CausalImpact uses bsts defaults (niter=2000) without post-hoc convergence filtering. |
