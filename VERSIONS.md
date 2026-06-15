# Version Manifest

Stack used in the Monte Carlo comparison study.

## Platform

| Component | Version |
|-----------|---------|
| OS | macOS (ARM64, Darwin) |
| Python | 3.12.8 |
| R | (captured at install time — run `R --version`) |

## Main Tools

| Tool | Version | Source |
|------|---------|--------|
| CausalPy | 0.8.0 | PyPI |
| Google matched_markets (TBR) | commit `5e3cd95` | GitHub `google/matched_markets` |
| GeoLift | commit `4d2afd4` | GitHub `facebookincubator/GeoLift` |
| CausalImpact | 1.4.1 | CRAN |

## Key Transitive Dependencies

### Python

See `requirements.txt` for the full frozen dependency list (88 packages).

| Package | Version | Role |
|---------|---------|------|
| PyMC | 5.28.1 | MCMC engine for CausalPy |
| ArviZ | 0.23.4 | Diagnostics + HDI for CausalPy |
| NumPy | 2.3.5 | Numerical computation |
| pandas | 3.0.1 | Data manipulation |
| matplotlib | 3.10.8 | Plotting |

### R

| Package | Role |
|---------|------|
| augsynth (commit `65c5a6f`) | Augmented synthetic control for GeoLift |
| bsts | BSTS engine for CausalImpact |
| BoomSpikeSlab | Spike-and-slab prior for bsts |
| zoo | Time series indexing for CausalImpact |

R package versions are printed during `make env` — check install output for exact versions.
