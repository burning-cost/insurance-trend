# insurance-trend

[![Tests](https://github.com/burning-cost/insurance-trend/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-trend/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/insurance-trend.svg)](https://pypi.org/project/insurance-trend/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Loss cost trend analysis for UK personal lines insurance pricing.

Blog post: [Loss Cost Trend Analysis in Python](https://burning-cost.github.io/2026/03/09/insurance-trend.html)

## The problem

Every UK motor and household pricing actuary does loss cost trend analysis every quarter. The workflow is: take aggregate accident-period experience data, fit a log-linear trend to frequency and severity separately, project forward to the next rating period, and report a trend rate with a confidence interval.

Currently this is done in Excel, SAS, or bespoke R scripts. There is no Python library for it. `chainladder-python` handles reserving triangles but does nothing for pricing trend — it applies user-specified factors, it does not fit them from data.

The post-2021 inflationary environment has made this more urgent: UK motor claims inflation ran at 34% from 2019 to 2023 versus CPI of 21% — a 13 percentage point superimposed component that CPI alone does not capture. A library that cannot identify structural breaks (COVID lockdown, Ogden rate change) will produce misleading trend estimates.

## Quick start

```python
import numpy as np
from insurance_trend import LossCostTrendFitter

# 24 quarters of UK motor aggregate experience data (2019 Q1 – 2024 Q4)
# Synthetic: stable frequency pre-2021, then step-down (COVID recovery lag),
# then gradual recovery. Severity accelerates post-2021 (repair cost inflation).
rng = np.random.default_rng(42)
n = 24

periods = [
    f"{yr}Q{q}"
    for yr in range(2019, 2025)
    for q in range(1, 5)
]

# Earned vehicle-years (growing book, slight seasonality)
earned_vehicles = (
    12_000
    + np.arange(n) * 150
    + rng.normal(0, 200, n)
).clip(10_000, None)

# True frequency: stable ~0.085 pre-2021, drops to ~0.065 in 2021 (COVID),
# then recovers at +3% pa through 2024
t = np.arange(n)
freq_true = np.where(
    t < 8,
    0.085 + 0.001 * t,
    0.065 * np.exp(0.007 * (t - 8)),  # post-COVID recovery trend
)
claim_counts = rng.poisson(freq_true * earned_vehicles).astype(float)

# True severity: accelerates at +8% pa from 2022 (repair inflation)
base_severity = 3_800.0
sev_true = base_severity * np.where(
    t < 12,
    1.0 + 0.03 * t / 4,
    (1.0 + 0.03) ** 3 * np.exp(0.08 * (t - 12) / 4),  # post-2022 inflation
)
total_paid = claim_counts * rng.lognormal(np.log(sev_true), 0.15)

fitter = LossCostTrendFitter(
    periods=periods,
    claim_counts=claim_counts,
    earned_exposure=earned_vehicles,
    total_paid=total_paid,
)

result = fitter.fit(
    detect_breaks=True,   # auto-detect COVID, Ogden rate change
    seasonal=True,        # quarterly seasonal dummies
)

print(result.combined_trend_rate)  # e.g. 0.085 — 8.5% pa loss cost trend
print(result.decompose())          # freq_trend, sev_trend, superimposed
print(result.summary())
```

With an ONS external index for severity deflation (requires network access):

```python
from insurance_trend import LossCostTrendFitter, ExternalIndex

# Fetch ONS motor repair index (SPPI G4520, 2015=100)
motor_repair_idx = ExternalIndex.from_ons('HPTH')

fitter = LossCostTrendFitter(
    periods=periods,
    claim_counts=claim_counts,
    earned_exposure=earned_vehicles,
    total_paid=total_paid,
    external_index=motor_repair_idx,  # deflates severity; superimposed_inflation() gives residual
)
result = fitter.fit(detect_breaks=True, seasonal=True)
print(result.superimposed_inflation)  # trend component not explained by ONS index
```

## Classes

- **`FrequencyTrendFitter`** — log-linear OLS on log(claims/exposure). Optional WLS, quarterly seasonal dummies, structural break detection via ruptures PELT, piecewise refitting on detected breaks, bootstrap CI, local linear trend alternative.

- **`SeverityTrendFitter`** — same as frequency, plus optional external index deflation. When an index is supplied, the fit runs on deflated severity and `superimposed_inflation()` gives the residual trend not explained by the index.

- **`LossCostTrendFitter`** — wraps the frequency and severity fitters, combines results, provides `decompose()` and `projected_loss_cost()`.

- **`ExternalIndex`** — fetches ONS time series from the public API (no auth required), with a catalogue of UK insurance-relevant codes. Also accepts user-supplied CSV for BCIS and other subscription data.

## Why log-linear

The industry baseline. Fits `log(y) = alpha + beta*t + seasonal + epsilon` via OLS. The annual trend rate is `exp(beta * periods_per_year) - 1`. The model is transparent, easily explainable to a regulator, and fast enough to bootstrap 1000 replicates in under a second.

The local linear trend alternative (`method='local_linear_trend'`) uses statsmodels `UnobservedComponents` with a Kalman filter — useful when the trend itself is changing, but requires longer series and is harder to explain.

## Structural breaks

The ruptures PELT algorithm runs on the log-transformed series. If a break is detected, the library warns and refits piecewise. The trend rate from the final segment is what gets reported — this is the defensible choice for projection, since you are projecting from the current regime.

Pass `changepoints=[8, 20]` to impose known breaks (e.g. 2020 Q1, 2025 Q1) rather than using auto-detection.

## ONS series catalogue

| Key | ONS code | Description |
|-----|----------|-------------|
| `motor_repair` | HPTH | SPPI G4520 Maintenance and repair of motor vehicles (2015=100) |
| `motor_insurance_cpi` | L7JE | CPI 12.5.4.1 Motor vehicle insurance |
| `vehicle_maintenance_rpi` | CZEA | RPI Maintenance of motor vehicles |
| `building_maintenance` | D7DO | CPI 04.3.2 Services for maintenance and repair of dwellings |
| `household_maintenance_weights` | CJVD | CPI Weights 04.3 Maintenance and repair |

For household severity, use D7DO as a free proxy. BCIS is more appropriate for reinstatement cost trend — load it via `ExternalIndex.from_csv()`.

## Inputs

Aggregate accident-period data. Minimum viable: 6 quarters. Recommended: 12–20 quarters.

| Column | Description |
|--------|-------------|
| `periods` | Quarter identifiers, e.g. `'2020Q1'` |
| `claim_counts` | Number of claims in the period |
| `earned_exposure` | Earned exposure (vehicle-years, policy-years, etc.) |
| `total_paid` | Total paid claims |

Both pandas and Polars DataFrames/Series are accepted as inputs. All outputs are Polars.

## Installation

```bash
pip install insurance-trend
```

## Dependencies

pandas, numpy, statsmodels, scipy, ruptures, matplotlib, requests, polars.

No scikit-learn, TensorFlow, or PyTorch.

## Mix adjustment

V1 does not include mix adjustment. If your portfolio composition has shifted (more young drivers, different vehicle types), apparent trends may reflect mix change rather than genuine inflation. Pre-process to mix-adjusted frequency/severity before passing to the fitters if this matters for your use case.

## Scope

This library is for pricing trend — forward projection of aggregate accident-period data. It is not a reserving tool. Use `chainladder-python` for triangle development to ultimate; use `insurance-trend` for what comes after.

## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_trend_demo.py).

## Performance

Benchmarked against a naive OLS baseline (single log-linear trend across all 24 quarters, no break detection) on synthetic UK motor data — 24 quarters (2019 Q1 to 2024 Q4) with a known frequency break at Q8 and severity break at Q12. Results from `benchmarks/benchmark.py` run 2026-03-16.

**Trend rate accuracy (true post-break DGP rates: frequency +3.0% pa, severity +8.0% pa, loss cost +11.24% pa):**

| Component | True (DGP) | Naive OLS | insurance-trend | Naive error | Lib error |
|-----------|-----------|-----------|-----------------|-------------|-----------|
| Frequency | +3.000% | -5.152% | -5.353% | -8.152pp | -8.353pp |
| Severity | +8.000% | +2.353% | +2.256% | -5.647pp | -5.744pp |
| Loss cost | +11.240% | -2.921% | -3.217% | -14.161pp | -14.457pp |

**Break detection (24-quarter series):**

| Component | True break | Detected | Within ±2Q? |
|-----------|-----------|---------|-------------|
| Frequency | Q8 | none | No break detected |
| Severity | Q12 | none | No break detected |

**4-quarter forward projection MAPE:**

| Method | Loss cost MAPE |
|--------|---------------|
| Naive OLS | 29.99% |
| insurance-trend | **26.06%** |
| Improvement | 3.93 pp |

On this benchmark, both models produce similar trend rate estimates because the break detection did not fire — the 24-quarter series and noise level did not exceed the PELT penalty threshold. The library's value here is the 3.9 pp projection MAPE improvement and the bootstrap CI (frequency CI: −7.3% to −3.2%), which the naive OLS does not produce.

Break detection is most reliable when: (a) the series is longer (30+ periods), (b) the break is large (>10pp in rate), or (c) the penalty parameter is reduced. Pass `changepoints=[8, 12]` to impose known break locations when the dates are known from external events (e.g. COVID lockdown, Ogden rate change).

The library's structural value is the frequency/severity decomposition: the naive OLS produces one blended loss cost trend and cannot separate -5.2pp frequency from +2.4pp severity. The decomposition feeds separately into pricing rate change and reinsurance attachment calculations.

Run `benchmarks/benchmark.py` on Databricks to reproduce.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Whittaker-Henderson graduation for development triangles — smooth the trends before forward projection |
| [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | Loss development models — trend projections inform the development assumptions in reserve models |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID causal evaluation of rate changes — separates genuine market trends from the effects of pricing actions |

