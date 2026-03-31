# Databricks notebook source
# MAGIC %md
# MAGIC # InflationDecomposer: Separating Structural from Cyclical Claims Inflation
# MAGIC
# MAGIC UK motor severity has risen ~37% from 2019 to 2023 (FCA, 2025). The standard
# MAGIC approach embeds the full trend as a permanent assumption. But how much is
# MAGIC structural — driven by EV complexity, labour shortages, legal inflation — and
# MAGIC how much is cyclical: credit hire volatility, used car price spikes, post-COVID
# MAGIC backlog effects that will eventually mean-revert?
# MAGIC
# MAGIC This notebook demonstrates `InflationDecomposer` from `insurance-trend`. It
# MAGIC uses the Harvey (1989) structural time series model — a state-space formulation
# MAGIC fitted by maximum likelihood — to separate claims severity into:
# MAGIC
# MAGIC - **Structural trend**: the part to embed in long-run pricing
# MAGIC - **Stochastic cycle**: the part to treat as transient
# MAGIC - **Seasonal**: calendar effects
# MAGIC - **Irregular**: noise
# MAGIC
# MAGIC The implementation uses `statsmodels.tsa.statespace.UnobservedComponents`.
# MAGIC Components are extracted via `res.level.smoothed` and `res.cycle.smoothed`
# MAGIC from the Kalman smoother.

# COMMAND ----------

# MAGIC %pip install insurance-trend>=0.1.5 matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data with known components
# MAGIC
# MAGIC We build a quarterly motor severity index (2014Q1–2024Q4, 44 periods) with:
# MAGIC - 7% pa structural trend (EV cost escalation + labour)
# MAGIC - 5-year sinusoidal cycle with amplitude ±8% (market cycle / credit hire)
# MAGIC - Annual seasonal pattern (Q4 winter claims spike)
# MAGIC - 1% quarterly noise

# COMMAND ----------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from insurance_trend import InflationDecomposer

# --- Generate synthetic motor severity index ---
rng = np.random.default_rng(2024)

n = 44  # 11 years quarterly (2014Q1 to 2024Q4)
t = np.arange(n, dtype=float)

# Known components
structural_pa = 0.07            # 7% pa — persistent cost escalation
cycle_period_years = 5.0        # 5-year market cycle
cycle_amplitude = 0.08          # ±8% cyclical swing
seasonal_amplitude = 0.04       # ±4% seasonal (Q4 higher)
noise_sigma = 0.01

structural = structural_pa / 4 * t
cycle = cycle_amplitude * np.sin(2 * np.pi * t / (cycle_period_years * 4))
seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 4 - np.pi / 2)  # Q4 peak
noise = rng.normal(0, noise_sigma, n)

log_severity = 0.0 + structural + cycle + seasonal + noise  # base at log(1.0)
severity_index = 100.0 * np.exp(log_severity)  # rebased to 100

# Period labels
periods = [f"{y}Q{q}" for y in range(2014, 2025) for q in range(1, 5)][:n]

print(f"Generated {n} quarterly observations: {periods[0]} to {periods[-1]}")
print(f"Index range: {severity_index.min():.1f} to {severity_index.max():.1f}")
print(f"Total index growth: {severity_index[-1]/severity_index[0]-1:.1%}")
print(f"Implied annual rate: {(severity_index[-1]/severity_index[0])**(4/n)-1:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit InflationDecomposer

# COMMAND ----------

decomposer = InflationDecomposer(
    series=severity_index,
    periods=periods,
    cycle=True,
    stochastic_cycle=True,
    seasonal=4,                      # quarterly seasonal
    damped_cycle=True,
    cycle_period_bounds=(2.0, 12.0),  # 2–12 year cycles
    log_transform=True,
    periods_per_year=4,
)

result = decomposer.fit()
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Component quality checks

# COMMAND ----------

# Components should sum to observed
obs = result.observations.to_numpy()
recon = (result.trend.to_numpy()
         + result.cycle.to_numpy()
         + result.seasonal.to_numpy()
         + result.irregular.to_numpy())
max_err = np.max(np.abs(recon - obs))
print(f"Max reconstruction error (log space): {max_err:.6f}")
assert max_err < 0.1, f"Reconstruction error too large: {max_err}"
print("Reconstruction check: PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Decomposition table

# COMMAND ----------

table = result.decomposition_table()
print(table.head(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare structural vs total trend rate

# COMMAND ----------

print(f"\n--- Trend comparison ---")
print(f"True structural rate (pa)   : {structural_pa:.2%}")
print(f"Estimated structural (pa)   : {result.structural_rate:.2%}")
print(f"Total OLS trend (pa)        : {result.total_trend_rate:.2%}")
print(f"\nTrue cycle period           : {cycle_period_years:.1f} years")
print(f"Estimated cycle period (yrs): {result.cycle_period:.1f} years")
print(f"\nCurrent cyclical position   : {result.cyclical_position:+.2%}")
print(f"(positive = above structural trend at latest period)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pricing implication
# MAGIC
# MAGIC The structural rate is the rate to embed in forward pricing assumptions.
# MAGIC The cyclical position tells you whether current experience is elevated or
# MAGIC depressed relative to that structural baseline — meaning next year's
# MAGIC loss cost may be lower or higher than projecting the total trend forward.

# COMMAND ----------

# Forward pricing calculation
n_project_quarters = 4  # 12 months forward

# Approach 1: Project total OLS trend (standard approach — embeds cyclical)
factor_total = (1 + result.total_trend_rate) ** (n_project_quarters / 4)

# Approach 2: Project structural trend only (ignores current cycle position)
factor_structural = (1 + result.structural_rate) ** (n_project_quarters / 4)

# Approach 3: Structural + mean-reversion of current cycle
# Assumes cycle decays by ~50% over 12 months (damping)
cycle_decay = 0.50  # approximate for 5-year cycle with damping
cycle_reversion = result.cyclical_position * (1 - cycle_decay)
factor_structural_adjusted = factor_structural * (1 + cycle_reversion)

current_severity = severity_index[-1]
print(f"Current severity index: {current_severity:.1f}")
print(f"\nProjected severity in 4 quarters:")
print(f"  (1) Total OLS trend         : {current_severity * factor_total:.1f} (factor: {factor_total:.4f})")
print(f"  (2) Structural only          : {current_severity * factor_structural:.1f} (factor: {factor_structural:.4f})")
print(f"  (3) Structural + mean-revert : {current_severity * factor_structural_adjusted:.1f} (factor: {factor_structural_adjusted:.4f})")

diff_pct = (factor_total - factor_structural) / factor_structural * 100
print(f"\nDifference between (1) and (2): {diff_pct:+.1f}%")
print(f"This is the pricing error from conflating cyclical with structural trend.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Plot the decomposition

# COMMAND ----------

fig = result.plot()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Practical usage: motor severity index (without seasonal)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Minimal usage — no seasonal, sensible defaults

# COMMAND ----------

# Simple use case: quarterly motor severity, no seasonal needed
simple_series = 100.0 * np.exp(0.07/4 * t + 0.06*np.sin(2*np.pi*t/20) + rng.normal(0, 0.01, n))

simple_result = InflationDecomposer(
    series=simple_series,
    periods=periods,
    cycle=True,
    cycle_period_bounds=(3.0, 10.0),  # motor market cycle
).fit()

print(simple_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cycle=False for very short series or when cycle unidentifiable

# COMMAND ----------

# Short series (last 20 quarters only)
short_series = severity_index[-20:]
short_periods = periods[-20:]

short_result = InflationDecomposer(
    series=short_series,
    periods=short_periods,
    cycle=False,  # need at least 16 obs for cycle; 20 is borderline, use False for safety
    periods_per_year=4,
).fit()

print(f"Short-series structural rate: {short_result.structural_rate:.2%} pa")
print(f"(cycle=False forces local linear trend only)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Monthly data example

# COMMAND ----------

# Monthly UK motor repair index (simulated)
n_monthly = 60  # 5 years
t_m = np.arange(n_monthly, dtype=float)
monthly_series = 100.0 * np.exp(
    0.07/12 * t_m
    + 0.06 * np.sin(2*np.pi*t_m/(5*12))  # 5-year cycle
    + rng.normal(0, 0.012, n_monthly)
)

monthly_result = InflationDecomposer(
    series=monthly_series,
    periods_per_year=12,
    cycle=True,
    cycle_period_bounds=(2.0, 8.0),
).fit()

print(f"Monthly data structural rate: {monthly_result.structural_rate:.2%} pa")
print(f"Estimated cycle period: {monthly_result.cycle_period:.1f} years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `InflationDecomposer` gives pricing actuaries a principled answer to the question:
# MAGIC *how much of current claims inflation is structural (embed it permanently) vs
# MAGIC cyclical (expect mean reversion)?*
# MAGIC
# MAGIC The key outputs:
# MAGIC - `result.structural_rate`: the annualised rate to project forward
# MAGIC - `result.cyclical_position`: current deviation from structural trend (±%)
# MAGIC - `result.cycle_period`: estimated market cycle length in years
# MAGIC
# MAGIC The Harvey state-space formulation is the correct model for this task. It
# MAGIC respects the stochastic nature of both the trend (level + slope can drift)
# MAGIC and the cycle (amplitude and phase can change), which is more honest than
# MAGIC fitting a deterministic Hodrick-Prescott filter or a fixed-window HP decomposition.
# MAGIC
# MAGIC **When to use `InflationDecomposer` vs `MultiIndexDecomposer`:**
# MAGIC - Use `MultiIndexDecomposer` when you have external economic indices (CPI, HPTH)
# MAGIC   and want to attribute severity trend to named cost drivers.
# MAGIC - Use `InflationDecomposer` when you want to separate the persistent structural
# MAGIC   component from the transient cyclical component in a single series.
# MAGIC - Use both together: first `InflationDecomposer` to understand the cycle, then
# MAGIC   `MultiIndexDecomposer` on the trend component to attribute it to CPI, labour, etc.
