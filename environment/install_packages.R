#!/usr/bin/env Rscript
# Install R dependencies for geo-simulation-v2
# Pinned to commit SHAs used in the simulation for reproducibility.

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org")
}

# augsynth — pinned to commit used in simulation (must come before GeoLift)
remotes::install_github("ebenmichael/augsynth@65c5a6f", upgrade = "never")

# GeoLift — pinned to commit used in simulation (depends on augsynth)
remotes::install_github("facebookincubator/GeoLift@4d2afd4", upgrade = "never")

# CausalImpact 1.4.1 — pinned for reproducibility
# Note: originally specified as 1.2.4 but that version was archived from CRAN.
# 1.4.1 is the version installed and used for all results in this study.
remotes::install_version("CausalImpact", version = "1.4.1",
                         repos = "https://cloud.r-project.org", upgrade = "never")

# zoo — time series objects required by CausalImpact
if (!requireNamespace("zoo", quietly = TRUE)) {
  install.packages("zoo", repos = "https://cloud.r-project.org")
}

# arrow for parquet I/O (used by generate_panels.R)
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow", repos = "https://cloud.r-project.org")
}

# jsonlite for R<->Python communication
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}

# Additional R packages (plotting and data manipulation)
for (pkg in c("dplyr", "ggplot2", "ggtext", "patchwork", "labeling")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

cat("All R packages installed successfully.\n")
cat("GeoLift version:", as.character(packageVersion("GeoLift")), "\n")
cat("CausalImpact version:", as.character(packageVersion("CausalImpact")), "\n")
cat("augsynth version:", as.character(packageVersion("augsynth")), "\n")
cat("bsts version:", as.character(packageVersion("bsts")), "\n")
cat("BoomSpikeSlab version:", as.character(packageVersion("BoomSpikeSlab")), "\n")
cat("zoo version:", as.character(packageVersion("zoo")), "\n")
