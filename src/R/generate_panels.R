#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Parameterized DGP: Synthetic Geo-Level Panel Generator
# ══════════════════════════════════════════════════════════════════════
#
# Generates synthetic geo-level time series for comparing
# geo-experiment tools (GeoLift, CausalPy, Google Matched Markets, CausalImpact).
#
# Everything is parameterized below. No external calibration
# data is used. Adjust any parameter and re-run.
#
# ITERATION COUNT NOTE:
#   n_iterations set to 10 for validation and review. For production
#   Monte Carlo runs, increase to 1000.
#
# CROSS-GEO CORRELATION CAVEAT:
#   The only source of cross-geo variation is baseline level differences:
#   All geos share identical trend & seasonality, but noise is independent
#   across geos. This makes it easy to build counterfactuals because all
#   donors move in lockstep (up to noise). If tools perform too uniformly
#   across scenarios, add geo-specific trend variation (e.g., small random
#   perturbation of trend_slope per geo) as a future extension.
#
# Output: parquet files in panels/{scenario}/{effect}/
# Schema: geo (chr), date (int), Y (dbl), Y_counterfactual (dbl), treated (lgl)
# ══════════════════════════════════════════════════════════════════════

library(arrow)
library(jsonlite)

# ── Parameters ──────────────────────────────────────────────────────
# Adjust these to control every aspect of the generated data.

# Geo structure
n_treated <- 1 # Number of treated geos (always 1 in this study)
n_control <- 20 # Number of control/donor geos

# Treatment effect
# Set to 0.0 for null condition (false positive rate test)
# Set to 0.075 for 7.5% lift (power/accuracy test)
effect_sizes <- c(0.0, 0.075)
effect_labels <- c("null", "effect")

# Panel dimensions
total_days <- 105 # Total simulation length in days
pre_days <- 90 # Days before treatment starts (calibration period)
# post_days = total_days - pre_days

# Geo baseline drawn from a log-normal distribution.
# baseline_mean: desired mean of the resulting distribution.
#   Internally converted to meanlog for R's rlnorm().
# baseline_spread: sdlog parameter of the underlying normal distribution.
#   Controls how spread out the geo sizes are.
#   0.3 = tight cluster, 0.5 = moderate spread (~2-3x range), 0.8 = wide spread
baseline_mean <- 4000
baseline_spread <- 0.6

# Trend = daily multiplicative growth rate shared across all geos.
# 0.001 = 0.1% daily growth (~3% monthly). Set to 0 for no trend.
trend_slope <- 0.001

# Weekly seasonality = each DOW has a typical level relative to the baseline.
# seasonality_amplitude scales the day-of-week profile below.
# 0.05 = subtle, 0.10 = moderate, 0.20 = pronounced
seasonality_amplitude <- 0.10

# Fixed weekly profile (normalized to [-1, +1]):
#   Mon   Tue   Wed   Thu   Fri   Sat   Sun
# -1.0  -0.5   0.0   0.2   0.8   1.0   0.5
# Monday is the weakest day, Saturday the strongest.
# The profile repeats every 7 days. Day 1 of the simulation is Monday.
dow_profile <- c(-1.0, -0.5, 0.0, 0.2, 0.8, 1.0, 0.5)

# Autocorrelation (AR(1) coefficient on the noise term)
# Controls how much today's noise depends on yesterday's.
#   0.0  = each day is independent (white noise)
#   0.15 = low persistence
#   0.3  = moderate persistence (realistic for many business metrics)
#   0.5  = high persistence (strong momentum — still realistic but
#           makes treatment effects harder to detect)
autocorrelation <- 0.30

# Noise level (log-scale)
# Standard deviation of the log-space noise term: Y includes exp(noise_level * scale_i * noise_t).
# Since exp() is always positive, Y cannot go negative regardless of noise draws.
# Noise scales sub-linearly with each geo's baseline via square-root scaling
# (scale_i = sqrt(baseline_i / mean_baseline)): a geo 4x larger has 2x the
# noise amplitude, not 4x. This reflects the portfolio effect in real data.
#   0.05 = low noise (clean signal), 0.10 = moderate, 0.20 = realistic
noise_level <- 0.20

# Outlier
# Multiplier applied to one geo's baseline in scenario A2.
# 5.0 means that geo is 5x the mean baseline.
outlier_multiplier <- 5.0

# Monte Carlo iterations
# Number of panels to generate per scenario x effect combination.
# Default is 10 for quick validation. Set to 1000 for production runs.
n_iterations <- 10

# Master seed for reproducibility
# Changing this produces entirely different data while keeping all
# structural parameters the same.
master_seed <- 42

# ── CLI Overrides ──────────────────────────────────────────────────────
cli_args <- commandArgs(trailingOnly = TRUE)
parse_cli <- function(flag, default) {
  idx <- which(cli_args == flag)
  if (length(idx) > 0 && idx < length(cli_args)) return(cli_args[idx + 1])
  default
}
n_iterations <- as.integer(parse_cli("--n_iterations", n_iterations))
output_base  <- parse_cli("--output_base", "panels")

# ── Helper Functions ────────────────────────────────────────────────

draw_baselines <- function(n_geos, baseline_mean, baseline_spread, seed) {
  # Draw geo baselines from a log-normal distribution.
  #
  # Args:
  #   n_geos: total number of geos (treated + control)
  #   baseline_mean: desired mean of the resulting distribution
  #   baseline_spread: sdlog parameter (dispersion of underlying normal)
  #   seed: RNG seed for reproducibility
  #
  # Returns:
  #   Named numeric vector of baselines, sorted ascending,
  #   labeled "City 1", "City 2", ..., "City N"

  set.seed(seed)

  # Convert desired mean to meanlog parameter.
  # For LogNormal: E[X] = exp(meanlog + sdlog^2 / 2)
  # So: meanlog = log(baseline_mean) - sdlog^2 / 2
  meanlog <- log(baseline_mean) - baseline_spread^2 / 2

  baselines <- rlnorm(n_geos, meanlog = meanlog, sdlog = baseline_spread)
  baselines <- sort(baselines)
  names(baselines) <- paste("City", seq_len(n_geos))

  baselines
}

select_treated <- function(baselines, n_treated) {
  # Select which geo(s) receive treatment.
  #
  # Rule: always pick the geo(s) closest to the median baseline.
  # The same city is treated across ALL scenarios. Scenario-specific
  # modifications (e.g., A2 outlier inflation) are applied to the
  # treated geo's baseline AFTER selection, in the main loop.
  #
  # Args:
  #   baselines: named numeric vector from draw_baselines()
  #   n_treated: number of treated geos (always 1 in this study)
  #
  # Returns:
  #   list with:
  #     treated_names: character vector of treated geo names
  #     treated_idx: integer vector of treated geo indices

  median_val <- median(baselines)
  diffs <- abs(baselines - median_val)
  treated_idx <- order(diffs)[1:n_treated]
  treated_names <- names(baselines)[treated_idx]

  list(
    treated_names = treated_names,
    treated_idx = treated_idx
  )
}

generate_panel <- function(baselines, treated_idx, effect_pct,
                           total_days, pre_days,
                           trend_slope, seasonality_amplitude, dow_profile,
                           autocorrelation, noise_level,
                           noise_baselines = NULL,
                           seed) {
  # Generate one complete panel of synthetic geo-level time series data.
  #
  # DGP for each geo i at time t:
  #   Y_cf_{i,t} = baseline_i * trend_t * season_t *
  #                exp(noise_level * scale_i * ar_noise_{i,t})
  #   where scale_i = sqrt(noise_baselines_i / mean(noise_baselines))
  #   Square-root scaling: a geo 4x larger gets 2x the noise amplitude,
  #   not 4x (portfolio effect). noise_baselines defaults to baselines;
  #   pass pre-inflation baselines in A2 so outlier inflation does not
  #   also amplify noise amplitude.
  #
  # Since exp(x) > 0 for all x, Y is strictly positive by construction.
  #
  # Treatment (post-period, treated geos only):
  #   Y_{i,t} = Y_cf_{i,t} * (1 + effect_pct)
  #
  # Args:
  #   baselines: named numeric vector of geo baselines (controls level)
  #   treated_idx: integer vector of treated geo indices
  #   effect_pct: treatment effect proportion (0.0 or 0.075)
  #   total_days: total number of days
  #   pre_days: number of pre-treatment days
  #   trend_slope: daily multiplicative trend rate
  #   seasonality_amplitude: scales the dow_profile
  #   dow_profile: 7-element numeric vector (Mon-Sun), normalized to [-1, +1]
  #   autocorrelation: AR(1) coefficient on noise
  #   noise_level: log-scale noise magnitude (see parameter block)
  #   noise_baselines: optional named numeric vector used for noise scaling
  #     (scale_i computation). Defaults to baselines when NULL. Pass the
  #     pre-inflation baselines here to prevent outlier inflation from
  #     also amplifying noise amplitude (A2 scenario).
  #   seed: RNG seed for this panel
  #
  # Returns:
  #   data.frame with columns: geo, date, Y, Y_counterfactual

  set.seed(seed)

  n_geos <- length(baselines)
  geo_names <- names(baselines)

  if (is.null(noise_baselines)) noise_baselines <- baselines
  mean_baseline <- mean(noise_baselines)

  # ── Build shared time-series skeleton ──────────────────────────
  t_seq <- seq_len(total_days)

  # Trend: multiplicative daily growth
  trend <- 1 + trend_slope * t_seq

  # Seasonality: day-of-week profile, repeating weekly
  # Day 1 = Monday, so (t-1) %% 7 + 1 maps day t to dow_profile index
  season <- 1 + seasonality_amplitude * dow_profile[((t_seq - 1) %% 7) + 1]

  # ── Generate each geo's series ─────────────────────────────────
  # Pre-allocate matrices: rows = days, cols = geos
  Y_cf <- matrix(NA_real_, nrow = total_days, ncol = n_geos)
  Y <- matrix(NA_real_, nrow = total_days, ncol = n_geos)

  for (i in seq_len(n_geos)) {
    # AR(1) noise process
    noise <- numeric(total_days)
    innovations <- rnorm(total_days)
    noise[1] <- innovations[1]
    for (t in 2:total_days) {
      noise[t] <- autocorrelation * noise[t - 1] + innovations[t]
    }

    # Geo-specific noise scaling: bigger geos get more absolute noise,
    # but sub-linearly (square-root scaling). A geo 4x larger has 2x the
    # noise amplitude, not 4x — consistent with the portfolio effect seen
    # in real revenue data.
    scale_i <- sqrt(noise_baselines[i] / mean_baseline)

    # Counterfactual: baseline * trend * seasonality * exp(scaled noise)
    Y_cf[, i] <- baselines[i] * trend * season *
      exp(noise_level * scale_i * noise)

    # Start with counterfactual
    Y[, i] <- Y_cf[, i]
  }

  # ── Apply treatment effect ─────────────────────────────────────
  post_days_idx <- (pre_days + 1):total_days
  for (idx in treated_idx) {
    Y[post_days_idx, idx] <- Y_cf[post_days_idx, idx] * (1 + effect_pct)
  }

  # ── Assemble long-format data.frame ────────────────────────────
  df <- data.frame(
    geo = rep(geo_names, each = total_days),
    date = rep(t_seq, times = n_geos),
    Y = as.vector(Y),
    Y_counterfactual = as.vector(Y_cf),
    stringsAsFactors = FALSE
  )

  df$treated <- df$geo %in% names(baselines)[treated_idx]

  df
}

save_panel <- function(df, scenario_id, effect_label, iteration,
                       output_base = "panels") {
  # Save a panel as a parquet file.
  #
  # Output path: {output_base}/{scenario_id}/{effect_label}/panel_{iteration}.parquet
  # Iteration is 1-indexed and zero-padded to 4 digits.
  #
  # Args:
  #   df: data.frame from generate_panel()
  #   scenario_id: e.g. "A1", "A2", etc.
  #   effect_label: "null" or "effect"
  #   iteration: integer iteration number (1-indexed)
  #   output_base: base directory for output

  out_dir <- file.path(output_base, scenario_id, effect_label)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  fname <- sprintf("panel_%04d.parquet", iteration)
  fpath <- file.path(out_dir, fname)

  write_parquet(df, fpath)
}

# ── Scenario Definitions ────────────────────────────────────────────
# Each scenario is a named list of parameter overrides.
# Only parameters that differ from the defaults above need to be listed.
#
# | Scenario | What It Tests                                        |
# |----------|------------------------------------------------------|
# | A1       | Textbook — clean baseline, can tools agree?          |
# | A2       | Outlier — treated geo inflated 5x above its baseline |
# | A3       | Small pool — only 10 geos instead of 21              |
# | A4       | Short pre-treatment — 30 days of history, not 90     |

scenarios <- list(
  A1 = list(),
  A2 = list(outlier = TRUE),
  A3 = list(n_control = 9),
  A4 = list(total_days = 45, pre_days = 30)
)

# ── Main Loop ───────────────────────────────────────────────────────
cat("══════════════════════════════════════════════════════\n")
cat("Parameterized DGP: Generating panels\n")
cat("══════════════════════════════════════════════════════\n\n")

total_panels <- 0

for (sc_id in names(scenarios)) {
  sc <- scenarios[[sc_id]]

  # Resolve scenario-specific overrides
  sc_n_control <- if (!is.null(sc$n_control)) sc$n_control else n_control
  sc_total_days <- if (!is.null(sc$total_days)) sc$total_days else total_days
  sc_pre_days <- if (!is.null(sc$pre_days)) sc$pre_days else pre_days
  sc_is_outlier <- isTRUE(sc$outlier)

  n_geos <- n_treated + sc_n_control

  cat(sprintf(
    "── Scenario %s: %d geos, %d days (%d pre + %d post) ──\n",
    sc_id, n_geos, sc_total_days, sc_pre_days,
    sc_total_days - sc_pre_days
  ))

  # ── Emit metadata.json (one per scenario) ─────────────────
  metadata <- list(
    scenario = sc_id,
    n_geos = n_geos,
    n_treated = n_treated,
    total_days = sc_total_days,
    pre_days = sc_pre_days,
    treatment_start = sc_pre_days + 1,
    treatment_end = sc_total_days,
    effect_sizes = setNames(as.list(effect_sizes), effect_labels),
    outlier = sc_is_outlier,
    outlier_multiplier = if (sc_is_outlier) outlier_multiplier else NULL,
    n_iterations = n_iterations,
    master_seed = master_seed
  )
  meta_path <- file.path(output_base, sc_id, "metadata.json")
  dir.create(dirname(meta_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(toJSON(metadata, auto_unbox = TRUE, pretty = TRUE), meta_path)

  for (ei in seq_along(effect_sizes)) {
    eff <- effect_sizes[ei]
    eff_label <- effect_labels[ei]

    cat(sprintf("  Effect: %s (%.1f%%)\n", eff_label, eff * 100))

    for (it in seq_len(n_iterations)) {
      # Seed: unique per scenario + iteration, shared across effect sizes
      # so null and effect panels have identical pre-treatment data
      panel_seed <- master_seed * 1000 + match(sc_id, names(scenarios)) * 10000 + it

      # Step 1: Draw baselines (same seed for null and effect)
      baselines <- draw_baselines(n_geos, baseline_mean, baseline_spread,
        seed = panel_seed
      )

      # Step 2: Select treated geo (always the median-sized — same city across scenarios)
      sel <- select_treated(baselines, n_treated)

      # Step 3: Apply outlier inflation to the TREATED geo (A2 only).
      # Save baselines before any outlier inflation — used for noise scaling
      noise_baselines <- baselines
      if (sc_is_outlier) {
        baselines[sel$treated_idx] <- outlier_multiplier * baselines[sel$treated_idx]
      }

      # Step 4: Generate panel (use offset seed to avoid correlation with baselines)
      df <- generate_panel(
        baselines = baselines,
        treated_idx = sel$treated_idx,
        effect_pct = eff,
        total_days = sc_total_days,
        pre_days = sc_pre_days,
        trend_slope = trend_slope,
        seasonality_amplitude = seasonality_amplitude,
        dow_profile = dow_profile,
        autocorrelation = autocorrelation,
        noise_level = noise_level,
        noise_baselines = noise_baselines,
        seed = panel_seed + 100000
      )

      # Step 5: Save
      save_panel(df, sc_id, eff_label, it, output_base)
      total_panels <- total_panels + 1
    }

    cat(sprintf(
      "    Saved %d panels to %s/%s/%s/\n",
      n_iterations, output_base, sc_id, eff_label
    ))
  }
  cat("\n")
}

cat(sprintf("Done. Total panels generated: %d\n", total_panels))

# ── Diagnostic Plot ─────────────────────────────────────────────────
# 2x2 panel showing one iteration per scenario (null condition),
# replicating the style of figures/scenario_timeseries.png.
# Gives Michael a quick visual sanity check on the generated data.

cat("\nGenerating diagnostic plot...\n")

scenario_labels <- c(
  A1 = "A1: Textbook",
  A2 = "A2: Outlier (5x)",
  A3 = "A3: Small Pool",
  A4 = "A4: Short Pre-Treatment"
)

# Colors matching the existing figure
col_control_line <- "#c0c0c0" # individual controls: light gray
col_control_avg <- "#2f465b" # control average: dark navy
col_treated <- "#E87461" # treated geo: coral
col_treatment <- "black" # treatment start line

fig_dir <- "figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Caption explaining shared DGP parameters in plain language — split across two lines
# so it fits within the figure width without truncation.
dgp_caption_1 <- paste0(
  "Shared DGP parameters \u2014 ",
  "noise = ", noise_level, " (day-to-day randomness; higher = choppier series)     ",
  "AR(1) = ", autocorrelation, " (day-to-day persistence; higher = smoother runs)"
)
dgp_caption_2 <- paste0(
  "baseline spread = ", baseline_spread, " (inequality in market size across geos)     ",
  "trend = ", trend_slope * 100, "%/day     ",
  "seasonality \u00b1", seasonality_amplitude * 100, "%"
)

png(file.path(fig_dir, "scenario_timeseries.png"),
  width = 16, height = 11, units = "in", res = 150
)

# Extra bottom outer margin to fit the shared-parameter caption
par(mfrow = c(2, 2), oma = c(6, 0, 3, 0), mar = c(4.5, 4.5, 3, 1.5))

for (sc_id in names(scenarios)) {
  sc <- scenarios[[sc_id]]

  # Resolve scenario-specific overrides
  sc_n_control <- if (!is.null(sc$n_control)) sc$n_control else n_control
  sc_total_days <- if (!is.null(sc$total_days)) sc$total_days else total_days
  sc_pre_days <- if (!is.null(sc$pre_days)) sc$pre_days else pre_days
  sc_is_outlier <- isTRUE(sc$outlier)
  n_geos <- n_treated + sc_n_control

  # Re-derive treated geo for iteration 1 using the same seed logic
  panel_seed <- master_seed * 1000 + match(sc_id, names(scenarios)) * 10000 + 1
  baselines <- draw_baselines(n_geos, baseline_mean, baseline_spread,
    seed = panel_seed
  )
  sel <- select_treated(baselines, n_treated)
  treated_name <- sel$treated_names[1]

  # Read null condition, iteration 1
  df <- read_parquet(file.path(output_base, sc_id, "null", "panel_0001.parquet"))

  all_geos <- unique(df$geo)
  control_geos <- setdiff(all_geos, treated_name)
  treatment_start <- sc_pre_days + 0.5 # offset for visual clarity

  # Set up y range across all geos
  y_range <- range(df$Y)

  # Empty plot frame
  plot(NULL,
    xlim = c(1, sc_total_days), ylim = y_range,
    xlab = "Day", ylab = "Y (response)",
    main = scenario_labels[sc_id],
    cex.main = 1.6, cex.lab = 1.4, cex.axis = 1.2
  )

  # Individual control geos (thin gray)
  for (geo in control_geos) {
    gd <- df[df$geo == geo, ]
    gd <- gd[order(gd$date), ]
    lines(gd$date, gd$Y, col = col_control_line, lwd = 0.5)
  }

  # Control average (thick dark navy)
  ctrl_data <- df[df$geo %in% control_geos, ]
  ctrl_avg <- aggregate(Y ~ date, data = ctrl_data, FUN = mean)
  ctrl_avg <- ctrl_avg[order(ctrl_avg$date), ]
  lines(ctrl_avg$date, ctrl_avg$Y, col = col_control_avg, lwd = 2.5)

  # Treated geo (thick coral)
  td <- df[df$geo == treated_name, ]
  td <- td[order(td$date), ]
  lines(td$date, td$Y, col = col_treated, lwd = 2.5)

  # Treatment start (dashed vertical line)
  abline(v = treatment_start, col = col_treatment, lty = 2, lwd = 1.5)

  # Legend with clear labels
  treated_label <- paste0(treated_name, " (treated)")
  legend("topleft",
    legend = c("Control avg", treated_label, "Treatment start"),
    col = c(col_control_avg, col_treated, col_treatment),
    lty = c(1, 1, 2),
    lwd = c(2.5, 2.5, 1.5),
    cex = 1.05, bg = "white", box.lty = 0
  )

  # Per-subplot annotation: scenario-specific parameter values
  # Placed at top-right, away from the legend, showing only what varies
  outlier_note <- if (sc_is_outlier) paste0("\ntreated geo \u00d7", outlier_multiplier) else ""
  param_note <- sprintf("geos = %d | pre = %d days%s", n_geos, sc_pre_days, outlier_note)
  legend("topright",
    legend = param_note,
    bty = "n", cex = 1.00, text.col = "gray30", text.font = 3
  )
}

# Main title
mtext("Simulated panel data by scenario (iteration 1, lift = 0%)",
  outer = TRUE, cex = 1.75, font = 2, line = 1.2
)

# Shared parameter caption at the bottom — two lines
mtext(dgp_caption_1,
  outer = TRUE, side = 1, line = 2.8,
  cex = 1.00, col = "gray20"
)
mtext(dgp_caption_2,
  outer = TRUE, side = 1, line = 4.1,
  cex = 1.00, col = "gray20"
)

dev.off()

cat(sprintf("Saved: %s/scenario_timeseries.png\n", fig_dir))
