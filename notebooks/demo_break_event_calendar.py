# Databricks notebook source
# MAGIC %md
# MAGIC # Break Event Calendar — insurance-trend v0.1.6
# MAGIC
# MAGIC **Purpose**: demonstrate `BreakEventCalendar`, the Phase 3 addition to
# MAGIC `insurance-trend` that maps detected structural breaks to known UK
# MAGIC insurance market events.
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC You run PELT or Bai-Perron on a quarterly motor severity index and get
# MAGIC break points at periods [8, 20, 24]. These are integer indices into your
# MAGIC period array. The algorithm tells you *where* the series changed; it cannot
# MAGIC tell you *why*. Was it the Ogden rate change? COVID lockdowns? The supply
# MAGIC chain shock? You have to answer that manually — or you can use
# MAGIC `BreakEventCalendar`.
# MAGIC
# MAGIC The calendar holds 22 major UK personal lines events (Ogden changes, IPT
# MAGIC rises, Civil Liability Act, GIPP, COVID, supply chain) and matches each
# MAGIC detected break to the nearest event within a configurable tolerance window.
# MAGIC
# MAGIC ## Contents
# MAGIC 1. Install and import
# MAGIC 2. Inspect the default UK event calendar
# MAGIC 3. Synthetic motor severity series with known structural breaks
# MAGIC 4. Detect breaks with `detect_breakpoints`
# MAGIC 5. Attribute breaks using `BreakEventCalendar`
# MAGIC 6. Customise the calendar (add / remove / filter events)
# MAGIC 7. Combine with `InflationDecomposer`

# COMMAND ----------

# MAGIC %pip install insurance-trend>=0.1.6 --quiet

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_trend import (
    BreakEventCalendar,
    CalendarEvent,
    AttributionReport,
    InflationDecomposer,
)
from insurance_trend.breaks import detect_breakpoints

print("insurance-trend imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inspect the default UK event calendar

# COMMAND ----------

cal = BreakEventCalendar()
print(f"Default calendar: {cal.n_events} events, tolerance={cal.tolerance} quarters")

df_events = cal.events_dataframe()
print("\nAll default events (period, category, impact, description):")
print(df_events.select(["period", "category", "impact", "description"]).to_pandas().to_string(index=False))

# COMMAND ----------

# Filter to only legal/regulatory events
legal_cal = cal.filter_events(categories=["legal", "regulation"])
print(f"\nLegal/regulatory events only: {legal_cal.n_events} events")
print(legal_cal.events_dataframe().select(["period", "description"]).to_pandas().to_string(index=False))

# COMMAND ----------

# Filter to upward-pressure events (things that increase claims costs)
upward_cal = cal.filter_events(impact=1)
print(f"\nUpward-pressure events (impact=+1): {upward_cal.n_events} events")
for evt in upward_cal.events:
    print(f"  {evt.period}: {evt.description[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic motor severity series with known structural breaks
# MAGIC
# MAGIC We build a quarterly series from 2012Q1 to 2024Q4 with:
# MAGIC - Structural trend of +6% pa throughout
# MAGIC - Step up at 2017Q1 (Ogden -0.75% shock, index +8)
# MAGIC - Step down at 2020Q1 (COVID lockdown, index -15%)
# MAGIC - Step up at 2020Q3 (post-lockdown bounce and backlog, index +12%)
# MAGIC - Step up at 2022Q1 (supply chain inflation, index +10%)

# COMMAND ----------

rng = np.random.default_rng(42)

# 2012Q1 to 2024Q4 = 52 quarters
quarters = [f"{y}Q{q}" for y in range(2012, 2025) for q in range(1, 5)]
n = len(quarters)
t = np.arange(n, dtype=float)

# Structural trend: 6% pa = 1.5% per quarter
log_trend = 0.015 * t

# Known step changes (in log space)
# Ogden 2017Q1 = index 20 (2012Q1 is 0, ..., 2016Q4 is 19, 2017Q1 is 20)
idx_ogden = quarters.index("2017Q1")
idx_covid_down = quarters.index("2020Q1")
idx_covid_bounce = quarters.index("2020Q3")
idx_supply = quarters.index("2022Q1")

steps = np.zeros(n)
steps[idx_ogden:] += np.log(1.08)        # +8% for Ogden
steps[idx_covid_down:] += np.log(0.85)   # -15% for COVID lockdown
steps[idx_covid_bounce:] += np.log(1.12) # +12% for post-lockdown bounce
steps[idx_supply:] += np.log(1.10)       # +10% for supply chain shock

# Small stochastic cycle + noise
cycle = 0.04 * np.sin(2 * np.pi * t / 24)  # 6-year cycle
noise = rng.normal(0, 0.008, n)

log_severity = 5.5 + log_trend + steps + cycle + noise
severity = np.exp(log_severity)

print(f"Severity series: {n} quarters from {quarters[0]} to {quarters[-1]}")
print(f"True break indices: {idx_ogden} ({quarters[idx_ogden]}), "
      f"{idx_covid_down} ({quarters[idx_covid_down]}), "
      f"{idx_covid_bounce} ({quarters[idx_covid_bounce]}), "
      f"{idx_supply} ({quarters[idx_supply]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Detect breaks with `detect_breakpoints`

# COMMAND ----------

log_sev = np.log(severity)
detected = detect_breakpoints(log_sev, penalty=1.5, max_breaks=6)

print(f"Detected break indices: {detected}")
print(f"Detected break periods: {[quarters[i] for i in detected]}")
print(f"\nTrue break periods: 2017Q1, 2020Q1, 2020Q3, 2022Q1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Attribute breaks to known events
# MAGIC
# MAGIC `attribute_indices` takes the raw integer indices from `detect_breakpoints`
# MAGIC and the `quarters` list to look up the period string for each break.

# COMMAND ----------

calendar = BreakEventCalendar(tolerance=2)

report = calendar.attribute_indices(detected, quarters)
print(report.summary())

# COMMAND ----------

# Structured output as a Polars DataFrame
df_report = report.to_dataframe()
print("\nAttribution as DataFrame:")
print(df_report.to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Attribute period strings directly
# MAGIC
# MAGIC If you know the break periods as strings (e.g., from a prior review),
# MAGIC use `attribute()` directly.

# COMMAND ----------

manual_breaks = ["2017Q1", "2020Q1", "2020Q3", "2022Q1"]
report2 = calendar.attribute(manual_breaks)
print(report2.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Customise the calendar

# COMMAND ----------

# Add a bespoke event for a line-of-business-specific development
custom_cal = BreakEventCalendar(tolerance=2)
custom_cal.add_event(
    period="2023Q4",
    description="FCA premium finance rules effective — affordability monitoring",
    category="regulation",
    impact=0,
    source="FCA CP23/14",
)
print(f"Custom calendar: {custom_cal.n_events} events")

# Check attribution with the new event
r_custom = custom_cal.attribute(["2023Q4"])
print(r_custom.summary())

# COMMAND ----------

# Remove a specific event
n_before = custom_cal.n_events
removed = custom_cal.remove_event("2023Q4", description_contains="premium finance")
print(f"Removed {removed} event(s). Calendar now has {custom_cal.n_events} events (was {n_before}).")

# COMMAND ----------

# Start from scratch with an empty calendar
empty_cal = BreakEventCalendar(include_defaults=False, tolerance=1)
empty_cal.add_event("2017Q1", "Ogden rate change", "legal", 1)
empty_cal.add_event("2020Q1", "COVID lockdown", "covid", -1)
print(f"\nCustom-only calendar: {empty_cal.n_events} events")
r_empty = empty_cal.attribute(["2017Q1", "2020Q2", "2019Q1"])
print(r_empty.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Combine with `InflationDecomposer`
# MAGIC
# MAGIC The typical workflow: fit Harvey decomposition → identify unusual periods
# MAGIC where the irregular or cyclical component spikes → attribute those periods
# MAGIC to known events.

# COMMAND ----------

decomposer = InflationDecomposer(
    series=severity,
    periods=quarters,
    cycle=True,
    cycle_period_bounds=(3, 10),
    log_transform=True,
    periods_per_year=4,
)

result = decomposer.fit()
print(result.summary())

# COMMAND ----------

# Identify periods where |irregular| > 1.5 sigma — potential unexplained breaks
decomp_table = result.decomposition_table()
irreg = decomp_table["irregular"].to_numpy()
sigma_irr = irreg.std()
spike_mask = np.abs(irreg) > 1.5 * sigma_irr
spike_periods = [q for q, m in zip(quarters, spike_mask) if m]

print(f"\nIrregular component sigma: {sigma_irr:.4f}")
print(f"Spike periods (|irregular| > 1.5σ): {spike_periods}")

# Attribute spikes to known events
if spike_periods:
    spike_report = calendar.attribute(spike_periods, tolerance=2)
    print("\n" + spike_report.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `BreakEventCalendar` closes the loop between algorithmic break detection and
# MAGIC actuarial interpretation. The workflow is:
# MAGIC
# MAGIC 1. Fit a trend model or state-space decomposition
# MAGIC 2. Detect structural breaks (PELT, Bai-Perron, CUSUM)
# MAGIC 3. `calendar.attribute_indices(breaks, periods)` — match to known events
# MAGIC 4. Flag unexplained breaks for further investigation or footnoting
# MAGIC
# MAGIC Key design choices:
# MAGIC - **No datetime objects** — all periods are strings like `"2020Q1"`.
# MAGIC   Ordinal arithmetic on (year, sub_period) tuples avoids arbitrary day
# MAGIC   choices that would mislead readers of a pricing review.
# MAGIC - **Tolerance is in periods, not days** — a tolerance of 2 means two
# MAGIC   quarters, which is appropriate given that PELT typically locates breaks
# MAGIC   within one to two periods of the true event date.
# MAGIC - **Mixed-frequency calendars work** — monthly events are silently skipped
# MAGIC   when attributing quarterly breaks, so you can freely add monthly data
# MAGIC   without breaking quarterly workflows.
# MAGIC - **Immutable events** — `CalendarEvent` is a frozen dataclass, so the
# MAGIC   registry cannot be accidentally mutated.
