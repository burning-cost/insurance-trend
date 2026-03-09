# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-trend: Loss Cost Trend Analysis Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow for UK personal lines loss cost
# MAGIC trend analysis using `insurance-trend`. We build synthetic motor insurance
# MAGIC data, fit frequency and severity trends separately, detect a structural break
# MAGIC (simulating COVID lockdown), and project forward.
# MAGIC
# MAGIC **Scenario**: UK motor property damage, quarterly data 2018 Q1 – 2023 Q4.
# MAGIC A frequency suppression event (COVID lockdown) occurs at 2020 Q1, lasting
# MAGIC four quarters. Severity shows 8% pa superimposed inflation above ONS HPTH.

# COMMAND ----------

# MAGIC %pip install insurance-trend pandas numpy statsmodels scipy ruptures matplotlib requests polars -q

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Data Generation
# MAGIC
# MAGIC We create 24 quarters of accident period data with:
# MAGIC - Genuine frequency trend: -1.5% pa (fewer claims per vehicle as safety tech improves)
# MAGIC - Genuine severity trend: +9% pa (repair costs, parts inflation)
# MAGIC - COVID lockdown period (2020 Q1–Q4): frequency artificially suppressed by 35%
# MAGIC - Quarterly seasonality in frequency (Q4 is highest — adverse weather)
# MAGIC - A synthetic ONS-style index for motor repair costs (~5% pa)

# COMMAND ----------

rng = np.random.default_rng(42)
n = 24  # 6 years, quarterly

# Period labels
quarters = [f"{yr}Q{q}" for yr in range(2018, 2024) for q in range(1, 5)]

t = np.arange(n)

# Exposure: ~50,000 vehicle-years per quarter, slight growth
exposure = 50_000 + 500 * t + rng.normal(0, 200, n)

# Frequency: -1.5% pa, with Q4 seasonal uplift of 8%, and COVID suppression
freq_pa = -0.015
per_period_freq = (1 + freq_pa) ** (1/4) - 1
base_freq = 0.042  # 4.2% base frequency

seasonal_factor = np.array([1.0, 0.96, 0.98, 1.08] * (n // 4))  # Q4 is highest
covid_mask = np.zeros(n)
covid_mask[8:12] = -0.35  # 2020 Q1-Q4: 35% frequency suppression

true_freq = (
    base_freq
    * (1 + per_period_freq) ** t
    * seasonal_factor
    * (1 + covid_mask)
)
freq_noise = np.exp(rng.normal(0, 0.025, n))
observed_freq = true_freq * freq_noise
claim_counts = np.maximum(1.0, np.round(exposure * observed_freq))

# Severity: +9% pa, with synthetic index inflation of 5% pa (superimposed = ~4%)
sev_pa = 0.09
per_period_sev = (1 + sev_pa) ** (1/4) - 1
base_sev = 3_200.0

true_sev = base_sev * (1 + per_period_sev) ** t
sev_noise = np.exp(rng.normal(0, 0.03, n))
observed_sev = true_sev * sev_noise
total_paid = claim_counts * observed_sev

# Synthetic external index: 5% pa motor repair cost inflation
index_pa = 0.05
per_period_index = (1 + index_pa) ** (1/4) - 1
motor_repair_index = (
    100.0 * (1 + per_period_index) ** t
    * np.exp(rng.normal(0, 0.008, n))
)

print(f"Periods: {quarters[0]} to {quarters[-1]}")
print(f"Exposure range: {exposure.min():.0f} – {exposure.max():.0f} vehicle-years")
print(f"Claim count range: {claim_counts.min():.0f} – {claim_counts.max():.0f}")
print(f"Severity range: £{observed_sev.min():.0f} – £{observed_sev.max():.0f}")
print(f"Loss cost range: £{(total_paid/exposure).min():.0f} – £{(total_paid/exposure).max():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Frequency Trend: Auto-Detecting the COVID Break

# COMMAND ----------

from insurance_trend import FrequencyTrendFitter

freq_fitter = FrequencyTrendFitter(
    periods=quarters,
    claim_counts=claim_counts,
    earned_exposure=exposure,
)
print(freq_fitter.summary())

# COMMAND ----------

# Fit with auto-detection of structural breaks
# The COVID suppression at 2020 Q1 (index 8) should be detected
import warnings
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    freq_result_auto = freq_fitter.fit(
        detect_breaks=True,
        seasonal=True,
        n_bootstrap=500,
    )
    for w in caught_warnings:
        print(f"WARNING: {w.message}")

print(freq_result_auto.summary())

# COMMAND ----------

# Also fit without break detection to show the difference
freq_result_nobreak = freq_fitter.fit(
    detect_breaks=False,
    seasonal=True,
    n_bootstrap=500,
)

print(f"\nWith break detection:    trend = {freq_result_auto.trend_rate:.2%} pa")
print(f"Without break detection: trend = {freq_result_nobreak.trend_rate:.2%} pa")
print(f"True underlying trend:   trend = -1.50% pa")
print(f"\nDetected changepoints: {freq_result_auto.changepoints}")

# COMMAND ----------

fig_freq = freq_result_auto.plot()
display(fig_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Severity Trend with ONS Motor Repair Index Deflation

# COMMAND ----------

from insurance_trend import SeverityTrendFitter, ExternalIndex

# Use our synthetic index (in practice this would be ExternalIndex.from_ons('HPTH'))
idx_series = ExternalIndex.from_series(motor_repair_index, label="HPTH_synthetic")

sev_fitter = SeverityTrendFitter(
    periods=quarters,
    total_paid=total_paid,
    claim_counts=claim_counts,
    external_index=idx_series,
)
print(sev_fitter.summary())
print(f"Deflated severity range: £{sev_fitter.deflated_severity.min():.0f} – £{sev_fitter.deflated_severity.max():.0f}")

# COMMAND ----------

sev_result = sev_fitter.fit(
    detect_breaks=False,   # No COVID effect in severity for this scenario
    seasonal=True,
    n_bootstrap=500,
    projection_periods=8,
)

print(sev_result.summary())
si = sev_fitter.superimposed_inflation()
print(f"\nSuperimposed inflation: {si:.2%} pa")
print(f"(True superimposed: ~4.00% pa = 9% total - 5% index)")

# COMMAND ----------

fig_sev = sev_result.plot()
display(fig_sev)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Combined Loss Cost Trend

# COMMAND ----------

from insurance_trend import LossCostTrendFitter

lc_fitter = LossCostTrendFitter(
    periods=quarters,
    claim_counts=claim_counts,
    earned_exposure=exposure,
    total_paid=total_paid,
    external_index=idx_series,
)
print(lc_fitter.summary())

# COMMAND ----------

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    lc_result = lc_fitter.fit(
        detect_breaks=True,
        seasonal=True,
        n_bootstrap=500,
        projection_periods=8,
    )
    for w in caught_warnings:
        print(f"WARNING: {w.message}")

print(lc_result.summary())

# COMMAND ----------

decomp = lc_result.decompose()
print("\n=== Decomposition ===")
print(f"Frequency trend:          {decomp['freq_trend']:+.2%} pa")
print(f"Severity trend:           {decomp['sev_trend']:+.2%} pa")
print(f"Combined loss cost trend: {decomp['combined_trend']:+.2%} pa")
if decomp['superimposed'] is not None:
    print(f"Superimposed inflation:   {decomp['superimposed']:+.2%} pa")

true_combined = (1 + (-0.015)) * (1 + 0.09) - 1
print(f"\nTrue combined (pre-COVID, post-break): ~{true_combined:.2%} pa")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Forward Projection
# MAGIC
# MAGIC Project 2 years (8 quarters) forward with 95% confidence fan.

# COMMAND ----------

proj = lc_fitter.projected_loss_cost(future_periods=8)
print("Loss cost projection:")
print(proj)

# COMMAND ----------

# Trend factors for standard projection periods
print("=== Trend Factors ===")
for n_quarters in [4, 6, 8, 12]:
    tf = lc_result.trend_factor(n_quarters)
    print(f"  {n_quarters} quarters ({n_quarters/4:.1f} years): {tf:.4f} ({(tf-1)*100:.2f}%)")

# COMMAND ----------

fig_lc = lc_result.plot()
display(fig_lc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary: Using Explicit Changepoints
# MAGIC
# MAGIC In production, an actuary would confirm the break dates rather than relying
# MAGIC solely on auto-detection. Pass `changepoints` explicitly for reproducibility.

# COMMAND ----------

lc_result_explicit = lc_fitter.fit(
    changepoints=[8],      # 2020 Q1 = index 8 in our 24-quarter series
    detect_breaks=False,
    seasonal=True,
    n_bootstrap=500,
    projection_periods=8,
)

print("With explicit changepoint at 2020 Q1 (index 8):")
print(lc_result_explicit.summary())
print("\nDecomposition:")
for k, v in lc_result_explicit.decompose().items():
    if v is not None:
        print(f"  {k}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ExternalIndex: ONS API Usage
# MAGIC
# MAGIC In production, fetch the actual ONS motor repair index rather than using
# MAGIC the synthetic one above.

# COMMAND ----------

# Show the catalogue of available series codes
catalogue = ExternalIndex.list_catalogue()
print("Available ONS series codes:")
print(catalogue)

# COMMAND ----------

# To fetch live data (requires network access):
# motor_repair = ExternalIndex.from_ons('HPTH', start_date='2018-01-01')
#
# For offline use, load from CSV:
# motor_repair = ExternalIndex.from_csv('hpth_data.csv', date_col='date', value_col='index')
#
# Tip: cache the ONS response to avoid repeated API calls:
# motor_repair = ExternalIndex.from_ons('HPTH', cache_path='/tmp/hpth_cache.json')

print("ExternalIndex.CATALOGUE:", ExternalIndex.CATALOGUE)
