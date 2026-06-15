#!/usr/bin/env Rscript
# CausalImpact wrapper — BSTS with spike-and-slab regression
#
# CLI interface:
#   Rscript run_causalimpact.R --input panel.csv --output result.json \
#     --locations "City 11" --treatment_start 91 --treatment_end 105 \
#     --alpha 0.05 --nseasons 7
#
# Input:  long-format CSV with columns: geo, date, Y (integer dates)
# Output: JSON with att_level, att_pct, ci_lower, ci_upper, p_value, etc.

suppressPackageStartupMessages({
  library(CausalImpact)
  library(zoo)
  library(jsonlite)
})

# ── Parse arguments ──────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx < length(args)) {
    return(args[idx + 1])
  }
  return(default)
}

input_file <- parse_arg("--input")
output_file <- parse_arg("--output")
locations_str <- parse_arg("--locations")
treatment_start <- as.integer(parse_arg("--treatment_start", "91"))
treatment_end <- as.integer(parse_arg("--treatment_end", "105"))
alpha <- as.numeric(parse_arg("--alpha", "0.05"))
nseasons <- as.integer(parse_arg("--nseasons", "7"))
seed <- as.integer(parse_arg("--seed", NA))
niter <- as.integer(parse_arg("--niter", "1000"))

if (is.null(input_file) || is.null(output_file) || is.null(locations_str)) {
  cat("Usage: Rscript run_causalimpact.R --input <csv> --output <json> --locations <comma-separated>\n")
  cat("  [--treatment_start 91] [--treatment_end 105] [--alpha 0.05] [--nseasons 7] [--seed NA] [--niter 1000]\n")
  quit(status = 1)
}

locations <- trimws(strsplit(locations_str, ",")[[1]])

# ── Read data ────────────────────────────────────────────────
panel <- read.csv(input_file, stringsAsFactors = FALSE)

if (!all(c("geo", "date", "Y") %in% colnames(panel))) {
  cat("Error: input CSV must have columns: geo, date, Y\n")
  quit(status = 1)
}

# ── Pivot to wide format ─────────────────────────────────────
# CausalImpact expects: col 1 = response (treated), cols 2-N = covariates (controls)
wide <- reshape(
  panel[, c("geo", "date", "Y")],
  idvar = "date",
  timevar = "geo",
  direction = "wide"
)

# reshape produces columns like Y.City 1, Y.City 2, ...
# Clean column names: strip "Y." prefix
col_names <- colnames(wide)
col_names <- sub("^Y\\.", "", col_names)
colnames(wide) <- col_names

# Sort by date (rows = time points)
wide <- wide[order(wide$date), ]

# Reorder columns: treated geo first, then controls alphabetically
control_names <- sort(setdiff(col_names[col_names != "date"], locations))
wide <- wide[, c(locations, control_names)]

# Sanitize column names: bsts (CausalImpact's engine) parses column names
# as R expressions, so spaces cause parse errors (e.g., "City 11" → error).
colnames(wide) <- make.names(colnames(wide))

# Convert to zoo with integer time index
wide_zoo <- zoo(wide, order.by = seq_len(nrow(wide)))

# ── Run CausalImpact ────────────────────────────────────────
start_time <- proc.time()

result_json <- tryCatch({
  pre.period <- c(1, treatment_start - 1)
  post.period <- c(treatment_start, treatment_end)

  model.args <- list(
    niter = niter,
    nseasons = nseasons,
    season.duration = 1
  )

  # Set seed for MCMC reproducibility (matches CausalPy's per-iteration seeding)
  if (!is.na(seed)) set.seed(seed)

  impact <- CausalImpact(
    data = wide_zoo,
    pre.period = pre.period,
    post.period = post.period,
    model.args = model.args,
    alpha = alpha
  )

  elapsed <- (proc.time() - start_time)["elapsed"]

  # Extract from summary — "Average" row (row 1)
  avg_row <- impact$summary["Average", ]

  att_level <- avg_row$AbsEffect
  ci_lower_level <- avg_row$AbsEffect.lower
  ci_upper_level <- avg_row$AbsEffect.upper
  counterfactual_mean <- avg_row$Pred
  p_value <- avg_row$p

  # ATT% normalization (same formula as GeoLift wrapper)
  if (!is.na(counterfactual_mean) && counterfactual_mean != 0) {
    att_pct <- att_level / counterfactual_mean
    ci_lower_pct <- ci_lower_level / counterfactual_mean
    ci_upper_pct <- ci_upper_level / counterfactual_mean
  } else {
    att_pct <- NA
    ci_lower_pct <- NA
    ci_upper_pct <- NA
  }

  # CI-based significance (two-sided): consistent with CausalPy and Google MM
  # CausalImpact's p_value is one-sided; using CI exclusion avoids threshold mismatch
  significant <- (!is.na(ci_lower_level) && ci_lower_level > 0) ||
                 (!is.na(ci_upper_level) && ci_upper_level < 0)

  list(
    status = "success",
    att_level = att_level,
    att_pct = att_pct,
    ci_lower = ci_lower_pct,
    ci_upper = ci_upper_pct,
    ci_lower_level = ci_lower_level,
    ci_upper_level = ci_upper_level,
    p_value = p_value,
    significant = significant,
    counterfactual_mean = counterfactual_mean,
    runtime_seconds = as.numeric(elapsed)
  )

}, error = function(e) {
  elapsed <- (proc.time() - start_time)["elapsed"]
  list(
    status = "error",
    error_message = conditionMessage(e),
    att_level = NA,
    att_pct = NA,
    ci_lower = NA,
    ci_upper = NA,
    significant = FALSE,
    runtime_seconds = as.numeric(elapsed)
  )
})

# ── Write output ─────────────────────────────────────────────
write(toJSON(result_json, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      output_file)

cat("CausalImpact result written to", output_file, "\n")
