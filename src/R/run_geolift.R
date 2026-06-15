#!/usr/bin/env Rscript
# GeoLift wrapper — Augmented SC with Ridge + conformal inference
#
# CLI interface:
#   Rscript run_geolift.R --input panel.csv --output result.json \
#     --locations "chicago,portland" --treatment_start 91 --treatment_end 105 \
#     --alpha 0.05
#
# Communicates with Python via CSV input -> JSON output

suppressPackageStartupMessages({
  library(GeoLift)
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

if (is.null(input_file) || is.null(output_file) || is.null(locations_str)) {
  cat("Usage: Rscript run_geolift.R --input <csv> --output <json> --locations <comma-separated>\n")
  cat("  [--treatment_start 91] [--treatment_end 105] [--alpha 0.05]\n")
  quit(status = 1)
}

locations <- trimws(strsplit(locations_str, ",")[[1]])

# ── Read data ────────────────────────────────────────────────
panel <- read.csv(input_file, stringsAsFactors = FALSE)

if (!all(c("location", "date", "Y") %in% colnames(panel))) {
  cat("Error: input CSV must have columns: location, date, Y\n")
  quit(status = 1)
}

# ── Run GeoLift ──────────────────────────────────────────────
start_time <- proc.time()

result_json <- tryCatch({
  # GeoDataRead converts the panel into GeoLift's internal format
  geo_data <- GeoDataRead(
    data = panel,
    date_id = "date",
    location_id = "location",
    Y_id = "Y",
    X = c(),
    format = "yyyy-mm-dd"
  )

  # Run GeoLift with conformal CIs enabled
  gl_result <- GeoLift(
    Y_id = "Y",
    data = geo_data,
    locations = locations,
    treatment_start_time = treatment_start,
    treatment_end_time = treatment_end,
    alpha = alpha,
    model = "Ridge",
    ConfidenceIntervals = TRUE,
    method = "conformal",
    conformal_type = "block",
    stat_test = "Total"
  )

  elapsed <- (proc.time() - start_time)["elapsed"]

  # Extract from inference table (1-row dataframe)
  inf <- gl_result$inference
  att_avg <- inf$ATT          # average ATT (level)
  pct_lift <- inf$Perc.Lift   # already in % (e.g., 7.1 means 7.1%)
  p_value <- inf$pvalue

  # Confidence intervals from inference (level)
  ci_lower_level <- inf$Lower.Conf.Int
  ci_upper_level <- inf$Upper.Conf.Int

  # If CIs are still NA, try from summary
  if (is.na(ci_lower_level) || is.na(ci_upper_level)) {
    avg_att_summary <- gl_result$summary$average_att
    ci_lower_level <- avg_att_summary$lower_bound
    ci_upper_level <- avg_att_summary$upper_bound
  }

  # gl_result$lower_bound / $upper_bound are on TOTAL aggregate scale (not per-period).
  # Using them here without dividing by n_post would produce incorrect CIs.
  # If both primary (inference) and secondary (summary) extraction failed,
  # leave CIs as NA rather than silently using the wrong scale.

  # Compute ATT%: att / counterfactual
  # Counterfactual = y_hat (fitted synthetic control for treated unit)
  # y_hat covers all 105 periods; post-treatment is from treatment_start onward
  y_hat_post <- gl_result$y_hat[treatment_start:treatment_end]
  avg_cf <- mean(y_hat_post)

  # ATT% = att_avg / avg_cf
  att_pct <- att_avg / avg_cf

  # CI% from level CIs
  if (!is.na(ci_lower_level) && !is.na(ci_upper_level) && avg_cf != 0) {
    ci_lower_pct <- ci_lower_level / avg_cf
    ci_upper_pct <- ci_upper_level / avg_cf
  } else {
    ci_lower_pct <- NA
    ci_upper_pct <- NA
  }

  # CI-based significance (two-sided): consistent with CausalPy, Google MM, CausalImpact
  # p_value is still recorded in output for supplementary analysis
  significant <- (!is.na(ci_lower_level) && ci_lower_level > 0) ||
                 (!is.na(ci_upper_level) && ci_upper_level < 0)

  # Diagnostics
  diagnostics <- list(
    method = ifelse(is.null(gl_result$results$progfunc), "none", gl_result$results$progfunc),
    l2_imbalance = gl_result$summary$l2_imbalance,
    scaled_l2_imbalance = gl_result$summary$scaled_l2_imbalance,
    pct_lift = pct_lift,
    avg_counterfactual = avg_cf
  )

  list(
    status = "success",
    att_level = att_avg,
    att_pct = att_pct,
    ci_lower = ci_lower_pct,
    ci_upper = ci_upper_pct,
    ci_lower_level = ci_lower_level,
    ci_upper_level = ci_upper_level,
    p_value = p_value,
    significant = significant,
    diagnostics = diagnostics,
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
    diagnostics = list(),
    runtime_seconds = as.numeric(elapsed)
  )
})

# ── Write output ─────────────────────────────────────────────
write(toJSON(result_json, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      output_file)

cat("GeoLift result written to", output_file, "\n")
