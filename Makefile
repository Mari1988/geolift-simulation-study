.PHONY: all env panels run metrics tables figures smoke eval eval-capture clean help

VENV    := .venv
PYTHON  := $(VENV)/bin/python
RSCRIPT := Rscript
N_ITERATIONS ?= 1000

# ── Full pipeline (sequential — no inter-target deps) ──────
# Each stage is independent so `make run` never accidentally
# re-triggers the 8-hour panel generation.
all:
	$(MAKE) panels
	$(MAKE) run
	$(MAKE) metrics
	$(MAKE) tables
	$(MAKE) figures

# ── Environment ─────────────────────────────────────────────
env:
	python3.12 -m venv $(VENV)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	$(RSCRIPT) -e "renv::restore()"

# ── Stage 1: Generate synthetic panels ──────────────────────
panels:
	$(RSCRIPT) src/R/generate_panels.R --n_iterations $(N_ITERATIONS)

# ── Stage 2: Run all tools on all panels ────────────────────
run:
	$(PYTHON) src/python/run_tools.py

# ── Stage 3: Compute aggregated metrics ─────────────────────
metrics:
	$(PYTHON) analysis/compute_metrics.py

# ── Stage 4: Generate result tables ─────────────────────────
tables:
	$(PYTHON) analysis/generate_tables.py

# ── Stage 5: Generate article figures ───────────────────────
# scenario_timeseries.png is already produced by the panels stage
figures:
	$(PYTHON) analysis/plot_forest.py
	$(PYTHON) analysis/plot_ci_gallery.py

# ── Smoke test (quick end-to-end validation) ───────────
smoke:
	$(MAKE) clean
	$(MAKE) panels N_ITERATIONS=5
	$(MAKE) run
	$(MAKE) metrics
	$(MAKE) tables
	$(MAKE) figures
	$(PYTHON) analysis/smoke_test.py

# ── Eval (compare against golden reference) ──────────
eval:
	$(PYTHON) analysis/eval_against_golden.py

eval-capture:
	mkdir -p results/golden
	cp results/aggregated/metrics.csv results/golden/metrics.csv
	@echo "Golden reference captured."

# ── Utility ─────────────────────────────────────────────────
clean:
	rm -rf panels/ results/ figures/

help:
	@echo "Targets:"
	@echo "  env       Install Python and R dependencies"
	@echo "  panels    Generate $(N_ITERATIONS) synthetic panels per scenario x effect"
	@echo "  run       Run GeoLift, CausalPy, Google MM, CausalImpact on all panels"
	@echo "  metrics   Compute aggregated metrics from results.jsonl"
	@echo "  tables    Generate result tables from metrics.csv"
	@echo "  figures   Generate forest plot and CI gallery"
	@echo "  smoke     Quick smoke test (5 iterations, validates all outputs)"
	@echo "  eval      Compare current metrics against golden reference"
	@echo "  eval-capture  Capture current metrics as golden reference"
	@echo "  all       Full pipeline: panels → run → metrics → tables → figures"
	@echo "  clean     Remove all generated outputs"
	@echo ""
	@echo "Override iterations: make panels N_ITERATIONS=50"
