# Environment Setup & Run Plan

Drafted 2026-06-28. Pick up from here when resuming.

## Pre-checks (already verified)

| Requirement | Status |
|---|---|
| Python 3.12.8 | Available (`python3.12 --version`) |
| R 4.6.0 | Available (renv.lock targets 4.5.1 — minor version gap, should work) |
| git | Available |
| `renv/` directory | **MISSING** — see Step 2 for why this matters |

---

## Step 1 — Python virtual environment

```bash
cd /Users/mariappan.subramanian/Documents/repo/geolift-simulation-study

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # 88 exact-pinned packages
.venv/bin/pip install -e .                  # editable install so src/ is importable as a package
```

`requirements.txt` includes `matched_markets` from a specific GitHub commit — requires network + git.

---

## Step 2 — R environment (renv bootstrap)

**Why not `make env`?**  
The Makefile's `env` target runs `Rscript -e "renv::restore()"`, but `.Rprofile` tries to
`source("renv/activate.R")` before that can run. Since `renv/` is gitignored and missing on a
fresh clone, R will error out immediately. We must bypass `.Rprofile` using `--no-init-file`.

```bash
# 1. Install renv into the user R library (bypassing .Rprofile)
Rscript --no-init-file -e "install.packages('renv', repos='https://cloud.r-project.org')"

# 2. Bootstrap renv project structure and restore ~40 CRAN packages from renv.lock
#    This creates renv/ (including activate.R) so future Rscript calls work normally
Rscript --no-init-file -e "renv::restore()"
```

---

## Step 3 — R GitHub-pinned packages

After Step 2, `.Rprofile` works. Now install the three packages that need specific GitHub commits:

```bash
Rscript environment/install_packages.R
```

What this installs (in order — order matters):
1. `ebenmichael/augsynth@65c5a6f` — GeoLift depends on this, must come first
2. `facebookincubator/GeoLift@4d2afd4`
3. `CausalImpact 1.4.1` from CRAN + supporting packages (zoo, arrow, jsonlite, dplyr, ggplot2)

---

## Step 4 — Unit tests

```bash
.venv/bin/pytest tests/ -v
```

Tests in `tests/test_compute_att.py` and `tests/test_data_converters.py`.

---

## Step 5 — Run all code

### Option A: Smoke test (recommended first, ~15–30 min)

`make smoke` runs the complete pipeline at 5 iterations per scenario instead of 1000.
**Note:** it calls `make clean` first, which deletes `panels/`, `results/`, and `figures/`
(including committed pre-computed results). That's fine — they get regenerated.

```bash
make smoke      # clean → panels (5 iter) → run all 4 tools → metrics → tables → figures → smoke_test.py
make eval       # compare output vs golden reference (some diff expected at 5 iter vs 1000-iter baseline)
```

### Option B: Full pipeline (several hours)

```bash
make all        # panels (1000 iter/scenario) → run → metrics → tables → figures
make eval       # should show near-zero drift vs golden reference
```

---

## Key file map (for context)

| File | Role |
|---|---|
| `requirements.txt` | Frozen Python deps (exact versions) |
| `pyproject.toml` | Loose Python dep constraints + editable install config |
| `renv.lock` | Pinned R package versions (R 4.5.1 target) |
| `environment/install_packages.R` | Installs GitHub-pinned R packages (augsynth, GeoLift) |
| `config/tools.yaml` | All tool hyperparameters (confidence levels, MCMC settings, etc.) |
| `src/python/run_tools.py` | Pipeline orchestrator — loops scenarios × tools, writes `results/raw/results.jsonl` |
| `results/raw/results.jsonl` | Pre-computed results (32,000 records, committed to repo) |
| `results/golden/metrics.csv` | Golden reference for regression checks via `make eval` |
